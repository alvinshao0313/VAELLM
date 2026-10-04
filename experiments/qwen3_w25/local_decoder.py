"""Fixed-code, single-Linear output reconstruction with the real packed decoder."""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Optional

import torch
from torch import nn

from experiments.linear_output.objectives import linear_output_mse
from litebsq.protected_channel_quant import PROTECTED_CHANNEL_QUANT_INT8, PROTECTED_CHANNEL_QUANT_NONE
from litebsq.vae_linear import VAELinear


_DECODER_PREFIX = "_parallel_stage_decoder."


def _schema(module):
    return {name: (tuple(t.shape), str(t.dtype)) for name, t in module.state_dict().items()}


def _storage_bits(module):
    storages = {}
    for tensor in module.state_dict().values():
        storage = tensor.untyped_storage()
        storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes() * 8
    return sum(storages.values())


def _frozen_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if name.startswith(_DECODER_PREFIX):
            continue
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _stage_norm(tensors, layout):
    totals = torch.zeros(2, dtype=torch.float64)
    for tensor in tensors:
        if tensor is None:
            raise RuntimeError("A registered packed decoder parameter received no gradient.")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("Non-finite packed decoder gradient/update.")
        if tensor.numel() % 2:
            raise ValueError("Packed decoder parameter cannot be split into two stages.")
        values = tensor.detach().reshape(2, -1).double().square().sum(dim=1).cpu()
        for packed_index, (stage, part) in enumerate(layout):
            if part != 0:
                raise ValueError("Only one part per stage is supported.")
            totals[stage] += values[packed_index]
    return totals.sqrt().tolist()


def _validate_payload_profile(module):
    """Accept the two measured bases without changing their frozen payloads."""
    forbidden = ("low_rank_a", "low_rank_b", "protected_output_indices", "protected_output_weight",
                 "protected_output_qvalues", "protected_output_scales", "protected_residual_indices")
    if any(getattr(module, name, None) is not None for name in forbidden) or module.has_sparse_residual():
        raise ValueError("LoRA, output protection and residual payloads are outside this experiment.")
    indices = module.protected_input_indices
    if indices is None:
        if (module.weight_rotation is None or module.weight_rotation.block_size != 0
                or module.protected_channel_quant_format != PROTECTED_CHANNEL_QUANT_NONE
                or any(getattr(module, name) is not None for name in
                       ("protected_input_weight", "protected_input_qvalues", "protected_input_scales"))):
            raise ValueError("The unprotected profile requires full two-sided weight rotation.")
        return "full_rotation_unprotected"
    if module.weight_rotation is not None or module.protected_channel_quant_format != PROTECTED_CHANNEL_QUANT_INT8:
        raise ValueError("Input protection requires the no-rotation INT8 profile.")
    if (indices.dtype != torch.long or indices.ndim != 1 or not indices.numel()
            or indices.numel() != torch.unique(indices).numel()
            or bool((indices < 0).any()) or bool((indices >= module.in_features).any())
            or any(indices.numel() % dim for dim in module.stage_codebook_dims)):
        raise ValueError("Protected input indices must be unique, in range, and count a codebook_dim multiple.")
    count = indices.numel()
    qvalues, scales = module.protected_input_qvalues, module.protected_input_scales
    if (module.protected_input_weight is not None
            or not isinstance(qvalues, torch.Tensor) or qvalues.dtype != torch.uint8
            or tuple(qvalues.shape) != (count, module.out_features)
            or not isinstance(scales, torch.Tensor) or scales.dtype != torch.bfloat16
            or tuple(scales.shape) != (count,) or not bool(torch.isfinite(scales).all())
            or not bool((scales > 0).all())
            or module.compressed_in_features + count != module.in_features
            or module.compressed_out_features != module.out_features):
        raise ValueError("The INT8 input profile requires its original qvalues/scales and compressed shape.")
    return "no_rotation_int8_input"


def _validate(module, reference_weight, train_inputs, heldout_inputs, steps, lr, chunk_size):
    if not isinstance(module, VAELinear):
        raise TypeError("module must be a real VAELinear.")
    decoder = module._modules.get("_parallel_stage_decoder")
    if not isinstance(decoder, nn.Module) or not module.parallel_stage_decode:
        raise ValueError("An already registered packed stage decoder is required.")
    if module.residual_stages != 2 or module.parallel_parts != 1 or decoder.num_models != 2:
        raise ValueError("Local decoder reconstruction supports exactly two stages and one part.")
    _validate_payload_profile(module)
    if module.always_use_original or not module.temporary or module.original_weight is not None:
        raise ValueError("Only a stable compressed Linear without original_weight is supported.")
    if module.trainable_decode or getattr(module, "_sparse_bit_binding", None) is not None:
        raise ValueError("The module must not have an active training/bit-update runtime.")
    if any(isinstance(child, nn.modules.batchnorm._BatchNorm) for child in module.modules()):
        raise ValueError("BatchNorm would introduce mutable running statistics.")
    if type(steps) is not int or steps < 1 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("steps and chunk_size must be positive integers.")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive.")
    if reference_weight.shape != (module.out_features, module.in_features):
        raise ValueError("reference_weight must have the original Linear [out, in] shape.")
    for name, tensor in (("reference_weight", reference_weight), ("train_inputs", train_inputs),
                         ("heldout_inputs", heldout_inputs)):
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite floating-point values.")
        if name != "reference_weight" and (tensor.ndim != 2 or tensor.shape[1] != module.in_features or not len(tensor)):
            raise ValueError(f"{name} must contain nonempty [tokens, in_features] inputs.")
    if train_inputs.dtype != heldout_inputs.dtype:
        raise ValueError("Training and held-out inputs must use the same activation dtype.")
    parameters = dict(decoder.named_parameters())
    if not parameters or len({p.dtype for p in parameters.values()}) != 1:
        raise ValueError("A decoder with a single floating storage dtype is required.")
    if next(iter(parameters.values())).dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Unsupported decoder storage dtype.")
    if any(p.grad is not None for p in module.parameters()):
        raise ValueError("Existing gradients must be cleared before local reconstruction.")
    device = next(iter(parameters.values())).device
    if any(t.device != device for t in module.state_dict().values()):
        raise ValueError("The caller must place the complete VAELinear on one execution device.")
    return decoder, parameters, device


def optimize_local_decoder(
    module: VAELinear,
    reference_weight: torch.Tensor,
    train_inputs: torch.Tensor,
    heldout_inputs: torch.Tensor,
    *,
    steps: int,
    lr: float,
    chunk_size: int = 256,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Update only the existing packed decoder; return JSON-compatible evidence.

    Every step uses all supplied training tokens. Held-out tokens never contribute
    gradients or select an iterate. The caller owns data selection and module
    placement. Parameters and optimizer moments are FP32 during optimization;
    successful updates are cast back to the original storage dtype. On failure,
    decoder values are rolled back. Runtime policies are restored and caches are
    invalidated in both cases. No model checkpoint is written here.
    """
    decoder, parameters, device = _validate(
        module, reference_weight, train_inputs, heldout_inputs, steps, lr, chunk_size)
    original_parameters = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    original_ids = {name: id(p) for name, p in parameters.items()}
    original_dtype = next(iter(parameters.values())).dtype
    frozen_before, schema_before, bits_before = _frozen_digest(module), _schema(module), _storage_bits(module)
    grad_flags = [(p, p.requires_grad) for p in module.parameters()]
    training_flags = [(child, child.training) for child in module.modules()]
    runtime = {key: getattr(module, key) for key in ("trainable_decode", "cache_decoded_weight", "parallel_stage_decode")}
    had_compute_policy = hasattr(module, "_decoder_compute_dtype")
    compute_policy = getattr(module, "_decoder_compute_dtype", None)
    native_dtype = compute_policy or train_inputs.dtype
    reference = reference_weight.detach().to(device=device, dtype=torch.float32)
    train_x = train_inputs.detach().to(device=device)
    heldout_x = heldout_inputs.detach().to(device=device)
    layout = list(module._parallel_stage_layout)
    completed = False
    result = {}

    def measure(dtype):
        module._decoder_compute_dtype = dtype
        module.clear_decoded_weight_cache()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            weight = module._decode_weight(dtype=dtype).float()
            delta = weight - reference
            metrics = {
                "train_mse": float(linear_output_mse(delta, train_x, chunk_size=chunk_size, recompute=False)),
                "heldout_mse": float(linear_output_mse(delta, heldout_x, chunk_size=chunk_size, recompute=False)),
                "weight_mse": float(delta.square().mean()),
            }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite reconstruction metric.")
        return metrics

    try:
        module.eval()
        for p, _ in grad_flags:
            p.requires_grad_(False)
        result["before_native"] = measure(native_dtype)
        for p in parameters.values():
            p.data = p.data.float()
        module.enable_trainable_decode(parallel_stage_decode=True)
        module._decoder_compute_dtype = torch.float32
        result["before_fp32"] = measure(torch.float32)
        optimizer = torch.optim.AdamW(parameters.values(), lr=lr, weight_decay=0.0)
        losses = []
        stage_gradient_max = [0.0, 0.0]
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=False):
                weight = module._decode_weight(dtype=torch.float32)
                loss = linear_output_mse(weight - reference, train_x, chunk_size=chunk_size, recompute=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}.")
            loss.backward()
            stage_gradients = _stage_norm([p.grad for p in parameters.values()], layout)
            stage_gradient_max = [max(old, new) for old, new in zip(stage_gradient_max, stage_gradients)]
            optimizer.step()
            losses.append(float(loss.detach()))
            if progress_callback is not None:
                progress_callback({"step": step + 1, "loss": losses[-1], "stage_gradient_l2": stage_gradients})
        result["after_fp32"] = measure(torch.float32)
        fp32_delta = _stage_norm([
            p.detach() - original_parameters[name].to(device=device, dtype=torch.float32)
            for name, p in parameters.items()], layout)
        if not all(value > 0 for value in stage_gradient_max) or not all(value > 0 for value in fp32_delta):
            raise RuntimeError("Both packed stages must receive gradients and update FP32 parameters.")
        moment_dtypes = {str(state[key].dtype) for state in optimizer.state.values()
                         for key in ("exp_avg", "exp_avg_sq")}
        if moment_dtypes != {"torch.float32"}:
            raise RuntimeError("AdamW moments must remain FP32.")
        del optimizer
        for name, p in parameters.items():
            p.grad = None
            p.data = p.data.to(dtype=original_parameters[name].dtype)
        module.disable_trainable_decode()
        module._build_parallel_stage_decode_plan()
        result["after_cast"] = measure(native_dtype)
        saved_delta = _stage_norm([
            p.detach().float() - original_parameters[name].to(device=device, dtype=torch.float32)
            for name, p in parameters.items()], layout)
        frozen_after = _frozen_digest(module)
        if frozen_after != frozen_before or _schema(module) != schema_before or _storage_bits(module) != bits_before:
            raise RuntimeError("Fixed payload or persistent dtype/shape/storage size changed.")
        if {name: id(p) for name, p in decoder.named_parameters()} != original_ids:
            raise RuntimeError("Optimization replaced registered decoder Parameter objects.")
        result.update({
            "settings": {"steps": steps, "lr": lr, "chunk_size": chunk_size, "optimizer": "AdamW",
                         "source_profile": _validate_payload_profile(module),
                         "weight_decay": 0.0, "optimizer_dtype": "float32", "storage_dtype": str(original_dtype),
                         "native_decode_dtype": str(native_dtype), "train_tokens": len(train_x), "heldout_tokens": len(heldout_x)},
            "training_loss": losses,
            "evidence": {"stage_gradient_l2_max": stage_gradient_max, "stage_parameter_delta_l2_fp32": fp32_delta,
                         "stage_parameter_delta_l2_saved": saved_delta, "frozen_sha256_before": frozen_before,
                         "frozen_sha256_after": frozen_after, "frozen_unchanged": True,
                         "tensor_schema_unchanged": True, "decoder_parameter_ids_unchanged": True,
                         "optimizer_moment_dtypes": sorted(moment_dtypes),
                         "persistent_storage_bits_before": bits_before, "persistent_storage_bits_after": _storage_bits(module)},
        })
        completed = True
        return result
    finally:
        for name, p in parameters.items():
            p.grad = None
            p.data = (p.data.to(dtype=original_parameters[name].dtype) if completed
                      else original_parameters[name].to(device=device).clone())
        for p, requires_grad in grad_flags:
            p.requires_grad_(requires_grad)
        for child, training in training_flags:
            child.training = training
        for key, value in runtime.items():
            setattr(module, key, value)
        if had_compute_policy:
            module._decoder_compute_dtype = compute_policy
        elif hasattr(module, "_decoder_compute_dtype"):
            delattr(module, "_decoder_compute_dtype")
        module._build_parallel_stage_decode_plan()
        module.clear_decoded_weight_cache()
