"""Fixed-input Qwen3 block recovery; FP32 decode, BF16 projections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Optional, Sequence

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from experiments.qwen3_w25.block_weight_graph import block_weight_graph
from experiments.qwen3_w25.local_decoder import _schema, _stage_norm, _storage_bits, _validate_payload_profile
from litebsq.vae_linear import VAELinear


PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
SUPPORTED_SOURCE_PROFILES = ("full_rotation_unprotected", "no_rotation_int8_input")
INPUT_SOURCE = "caller_supplied_fixed_hidden_states"


@dataclass
class BlockBatch:
    """Fixed block inputs/kwargs and teacher output, with valid-token mask.

    Capture kwargs at the real block pre-hook, including position_embeddings and
    the attention implementation's causal mask. All tensors are detached by the
    optimizer. Padding positions contribute neither loss nor token count.
    """

    hidden_states: torch.Tensor
    target_hidden_states: torch.Tensor
    forward_kwargs: dict
    valid_mask: torch.Tensor


def _tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: _tree_to(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree_to(item, device) for item in value)
    return value


def _frozen_digest(block, decoder_names):
    digest = hashlib.sha256()
    for name, tensor in sorted(block.state_dict().items()):
        if name in decoder_names:
            continue
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate(block, train_batches, heldout_batches, steps, lr):
    if not isinstance(block, Qwen3DecoderLayer):
        raise TypeError("A native Qwen3DecoderLayer is required.")
    if type(steps) is not int or steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps must be a positive integer and lr finite and positive.")
    targets = [(name, block.get_submodule(name)) for name in PROJECTIONS]
    if {name for name, child in block.named_modules() if isinstance(child, VAELinear)} != set(PROJECTIONS):
        raise ValueError("The block must contain exactly the seven native VAELinear projections.")
    parameters, decoders = {}, {}
    for name, module in targets:
        decoder = module._modules.get("_parallel_stage_decoder")
        if not isinstance(decoder, nn.Module) or not module.parallel_stage_decode:
            raise ValueError(f"{name}: an already registered packed decoder is required.")
        if module.residual_stages != 2 or module.parallel_parts != 1 or decoder.num_models != 2:
            raise ValueError(f"{name}: exactly two stages and one part are required.")
        if _validate_payload_profile(module) not in SUPPORTED_SOURCE_PROFILES:
            raise ValueError(f"{name}: unsupported local reconstruction source profile.")
        if module.always_use_original or not module.temporary or module.original_weight is not None:
            raise ValueError(f"{name}: only stable compressed projections are supported.")
        if module.trainable_decode or getattr(module, "_sparse_bit_binding", None) is not None:
            raise ValueError(f"{name}: an active training/bit runtime is unsupported.")
        if "forward" in module.__dict__ or getattr(module, "_decoder_compute_dtype", None) not in (None, torch.bfloat16):
            raise ValueError(f"{name}: expected the native BF16 inference policy and forward.")
        local = dict(decoder.named_parameters())
        if not local or len({p.dtype for p in local.values()}) != 1:
            raise ValueError(f"{name}: one floating decoder storage dtype is required.")
        if next(iter(local.values())).dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"{name}: unsupported decoder storage dtype.")
        decoders[name] = local
        parameters.update({name + "._parallel_stage_decoder." + key: p for key, p in local.items()})
    if len({id(p) for p in parameters.values()}) != len(parameters):
        raise ValueError("Shared decoder parameters are unsupported.")
    if any(p.grad is not None for p in block.parameters()):
        raise ValueError("Existing gradients must be cleared before block recovery.")
    if any(isinstance(child, nn.modules.batchnorm._BatchNorm) for child in block.modules()):
        raise ValueError("Mutable BatchNorm statistics are unsupported.")
    device = next(iter(parameters.values())).device
    if any(t.device != device for t in block.state_dict().values()):
        raise ValueError("Place the entire block on one execution device before recovery.")
    if any(t.is_floating_point() and t.dtype != torch.bfloat16
           for name, t in block.state_dict().items() if name not in parameters):
        raise ValueError("Non-decoder floating tensors must retain BF16 storage.")
    for split, batches in (("train", train_batches), ("heldout", heldout_batches)):
        if not batches:
            raise ValueError(f"{split}: at least one captured batch is required.")
        for batch in batches:
            x, y, kwargs, mask = batch.hidden_states, batch.target_hidden_states, batch.forward_kwargs, batch.valid_mask
            if x.ndim != 3 or x.shape != y.shape or x.shape[-1] != block.hidden_size or not x.numel():
                raise ValueError(f"{split}: matching nonempty [batch, tokens, hidden] tensors are required.")
            if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"{split}: fixed inputs and teacher targets must be finite BF16 tensors.")
            if mask.shape != x.shape[:2] or not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.any()):
                raise ValueError(f"{split}: a nonempty binary valid-token mask is required.")
            rope = kwargs.get("position_embeddings")
            if not isinstance(rope, (tuple, list)) or len(rope) != 2 or any(not isinstance(t, torch.Tensor) for t in rope):
                raise ValueError(f"{split}: capture the real Qwen3 position_embeddings tuple.")
            if kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False) or kwargs.get("output_attentions", False):
                raise ValueError("Block recovery requires no KV cache and output_attentions=False.")
            attention_mask = kwargs.get("attention_mask")
            if block.self_attn.config._attn_implementation == "eager" and (attention_mask is None or attention_mask.ndim != 4):
                raise ValueError("Eager Qwen3 requires its captured four-dimensional causal mask.")
    return targets, decoders, parameters, device


def _squared_error(block, batch, device):
    x = batch.hidden_states.detach().to(device)
    kwargs = _tree_to(batch.forward_kwargs, device)
    with torch.autocast(device_type=device.type, enabled=False):
        prediction = block(x, **kwargs)[0]
        if prediction.dtype != torch.bfloat16 or prediction.shape != x.shape:
            raise RuntimeError("The native block must retain its BF16 output and hidden-state shape.")
        truth = batch.target_hidden_states.detach().to(device)
        valid = batch.valid_mask.detach().to(device=device, dtype=torch.bool)
        return (prediction.float() - truth.float())[valid].square().sum()


def optimize_local_block(
    block: Qwen3DecoderLayer,
    train_batches: Sequence[BlockBatch],
    heldout_batches: Sequence[BlockBatch],
    *,
    steps: int,
    lr: float,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Jointly update seven decoders on fixed inputs, without cross-block gradients.

    Each optimizer step accumulates all training batches, weighted by valid token
    count in FP32 weight leaves, then performs one decoder VJP. Held-out batches
    only measure the final iterate. Decoder parameters,
    decode arithmetic and AdamW moments are FP32; the original Qwen3 attention,
    MLP and residual path remain BF16. On success only decoder values change,
    cast to their original storage dtype. On error all decoder values roll back.
    This core does not capture a model, select data, write checkpoints or evaluate
    downstream tasks. CPU exercises the real dense decoder fallback, not CUDA's
    packed kernels or FlashAttention2.
    """
    targets, decoders, parameters, device = _validate(block, train_batches, heldout_batches, steps, lr)
    original = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    parameter_ids = {name: id(p) for name, p in block.named_parameters()}
    grad_flags = [(p, p.requires_grad) for p in block.parameters()]
    training_flags = [(child, child.training) for child in block.modules()]
    runtimes = {name: {key: getattr(module, key) for key in
                     ("trainable_decode", "cache_decoded_weight", "parallel_stage_decode")}
                for name, module in targets}
    policies = {name: (hasattr(module, "_decoder_compute_dtype"), getattr(module, "_decoder_compute_dtype", None))
                for name, module in targets}
    frozen_before, schema_before, bits_before = _frozen_digest(block, parameters), _schema(block), _storage_bits(block)
    train_tokens = sum(int(batch.valid_mask.sum()) for batch in train_batches)
    heldout_tokens = sum(int(batch.valid_mask.sum()) for batch in heldout_batches)
    result, completed = {}, False

    def measure():
        metrics = {}
        with torch.no_grad():
            for split, batches, tokens in (("train", train_batches, train_tokens), ("heldout", heldout_batches, heldout_tokens)):
                value = sum(float(_squared_error(block, batch, device)) for batch in batches) / (tokens * block.hidden_size)
                if not math.isfinite(value):
                    raise FloatingPointError("Non-finite block reconstruction metric.")
                metrics[split + "_mse"] = value
        return metrics

    def native_policy():
        for name, module in targets:
            module.trainable_decode = False
            module.cache_decoded_weight = runtimes[name]["cache_decoded_weight"]
            module._decoder_compute_dtype = torch.bfloat16
            module.clear_decoded_weight_cache()

    def stage_delta():
        return {name: _stage_norm([
            p.detach().float() - original[name + "._parallel_stage_decoder." + key].to(device=device, dtype=torch.float32)
            for key, p in decoders[name].items()], module._parallel_stage_layout)
                for name, module in targets}

    try:
        block.eval().requires_grad_(False)
        native_policy()
        result["before_native"] = measure()
        for p in parameters.values():
            p.data = p.data.float()
        for _, module in targets:
            module.enable_trainable_decode(parallel_stage_decode=True)
            module._decoder_compute_dtype = torch.float32
        with block_weight_graph(targets, requires_grad=False):
            result["before_fp32_decode"] = measure()
        optimizer = torch.optim.AdamW(parameters.values(), lr=lr, weight_decay=0.0)
        losses, gradient_max = [], {name: [0.0, 0.0] for name in PROJECTIONS}
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            with block_weight_graph(targets, requires_grad=True) as graph:
                for batch in train_batches:
                    loss = _squared_error(block, batch, device) / (train_tokens * block.hidden_size)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite block loss at step {step + 1}.")
                    loss.backward()
                    step_loss += float(loss.detach())
                graph.backward_to_decoders()
            if any(p.grad is not None for name, p in block.named_parameters() if name not in parameters):
                raise RuntimeError("A frozen block parameter received a gradient.")
            gradients = {name: _stage_norm([p.grad for p in decoders[name].values()], module._parallel_stage_layout)
                         for name, module in targets}
            for name in PROJECTIONS:
                gradient_max[name] = [max(a, b) for a, b in zip(gradient_max[name], gradients[name])]
            optimizer.step()
            losses.append(step_loss)
            if progress_callback is not None:
                progress_callback({"step": step + 1, "loss": step_loss, "stage_gradient_l2": gradients})
        with block_weight_graph(targets, requires_grad=False):
            result["after_fp32_decode"] = measure()
        fp32_delta = stage_delta()
        if not all(value > 0 for values in list(gradient_max.values()) + list(fp32_delta.values()) for value in values):
            raise RuntimeError("Both stages of all seven decoders must receive gradients and update.")
        moment_dtypes = {str(state[key].dtype) for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq")}
        if moment_dtypes != {"torch.float32"}:
            raise RuntimeError("AdamW moments must remain FP32.")
        del optimizer
        for name, p in parameters.items():
            p.grad = None
            p.data = p.data.to(original[name].dtype)
        native_policy()
        result["after_cast"] = measure()
        saved_delta = stage_delta()
        frozen_after = _frozen_digest(block, parameters)
        if frozen_after != frozen_before or _schema(block) != schema_before or _storage_bits(block) != bits_before:
            raise RuntimeError("Fixed block payload or persistent schema/storage size changed.")
        if parameter_ids != {name: id(p) for name, p in block.named_parameters()}:
            raise RuntimeError("Optimization replaced registered Parameter objects.")
        result.update({
            "settings": {"steps": steps, "lr": lr, "optimizer": "AdamW", "weight_decay": 0.0,
                         "optimizer_dtype": "float32", "decoder_compute_dtype": "torch.float32",
                         "projection_dtype": "torch.bfloat16", "native_decode_dtype": "torch.bfloat16",
                         "gradient_path": "two_phase_fp32_weight_gradient_then_decoder_vjp",
                         "weight_gradient_dtype": "torch.float32", "decoder_graphs_per_step": len(targets),
                         "objective": "block_output_mse", "input_source": INPUT_SOURCE,
                         "train_tokens": train_tokens, "heldout_tokens": heldout_tokens,
                         "storage_dtypes": {name: str(next(iter(local.values())).dtype) for name, local in decoders.items()}},
            "training_loss": losses,
            "evidence": {"stage_gradient_l2_max": gradient_max, "stage_parameter_delta_l2_fp32": fp32_delta,
                         "stage_parameter_delta_l2_saved": saved_delta, "frozen_sha256_before": frozen_before,
                         "frozen_sha256_after": frozen_after, "frozen_unchanged": True,
                         "tensor_schema_unchanged": True, "parameter_ids_unchanged": True,
                         "optimizer_moment_dtypes": sorted(moment_dtypes),
                         "persistent_storage_bits_before": bits_before, "persistent_storage_bits_after": _storage_bits(block)},
        })
        completed = True
        return result
    finally:
        for name, p in parameters.items():
            p.grad = None
            p.data = p.data.to(original[name].dtype) if completed else original[name].to(device=device).clone()
        for p, requires_grad in grad_flags:
            p.requires_grad_(requires_grad)
        for child, training in training_flags:
            child.training = training
        for name, module in targets:
            for key, value in runtimes[name].items():
                setattr(module, key, value)
            existed, policy = policies[name]
            if existed:
                module._decoder_compute_dtype = policy
            elif hasattr(module, "_decoder_compute_dtype"):
                delattr(module, "_decoder_compute_dtype")
            module._build_parallel_stage_decode_plan()
            module.clear_decoded_weight_cache()
