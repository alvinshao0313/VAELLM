"""Single-Linear fixed-decoder reconstruction through the production packed-bit graph."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
import time
from typing import Callable, Optional

import torch
from torch import nn

from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.local_decoder import _schema, _storage_bits, _validate, _validate_payload_profile
from litebsq.bit_sensitivity import measure_scale
from litebsq.bitpack import unpack_uint8_tensor_to_bool
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager
from sparse_bit_tuning.triton_kernels import sparse_bit_triton_available


def _code_banks(module):
    state = module.state_dict()
    banks = []
    for stage in range(2):
        tensor = module.get_stage_part_vq_storage(stage, 0)
        spec = module.get_stage_part_vq_spec(stage, 0)
        shape = tuple(spec["logical_shape"])
        if tensor.dtype != torch.uint8 or len(shape) != 3 or shape[1] != 1:
            raise ValueError("Two normal uint8 packed [blocks, 1, bits] code banks are required.")
        names = [name for name, value in state.items()
                 if value.data_ptr() == tensor.data_ptr() and value.shape == tensor.shape
                 and value.dtype == tensor.dtype and value.stride() == tensor.stride()]
        if len(names) != 1:
            raise ValueError("Each packed bank must have one unaliased persistent state entry.")
        banks.append((names[0], shape))
    if banks[0][0] == banks[1][0] or banks[0][1] != banks[1][1]:
        raise ValueError("Independent stage banks with identical logical shapes are required.")
    return banks


def _digest(state, excluded):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name not in excluded:
            digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _changed_bits(before, after, logical_shape):
    xor = torch.bitwise_xor(before, after.detach().cpu()).contiguous()
    tail = logical_shape[-1] % 8
    if tail and bool(torch.bitwise_and(xor[..., -1], 255 ^ ((1 << tail) - 1)).any()):
        raise RuntimeError("Packed padding bits changed.")
    counts = torch.tensor([value.bit_count() for value in range(256)], dtype=torch.int16)
    return int(counts[xor.long()].sum())


def calibrate_local_bits_cpu(module: VAELinear) -> dict:
    """Read-only dense-CPU estimate using the same deterministic scale definition.

    The production manager independently calibrates its packed CUDA path. This
    estimate can set an explicit trial LR, but is not an exact GPU-scale claim.
    Sampling remains measure_scale's at-most-256 evenly spaced rows, all bits.
    """
    if not isinstance(module, VAELinear) or module.residual_stages != 2 or module.parallel_parts != 1:
        raise ValueError("CPU calibration requires a real two-stage, one-part VAELinear.")
    if any(value.device.type != "cpu" for value in module.state_dict().values()):
        raise ValueError("CPU calibration requires a module already resident on CPU.")
    if module.trainable_decode or getattr(module, "_sparse_bit_binding", None) is not None:
        raise ValueError("CPU calibration requires an inactive source module.")
    banks = _code_banks(module)
    stages = []
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for stage, (name, logical_shape) in enumerate(banks):
            decoder = deepcopy(module.get_stage_part_decoder(stage, 0)).float().eval().requires_grad_(False)
            decoder.use_checkpoint = False

            def decode(packed):
                shape = (*packed.shape[:-1], logical_shape[-1])
                return decoder(unpack_uint8_tensor_to_bool(packed, logical_shape=shape).float())

            measured = measure_scale(module.get_stage_part_vq_storage(stage, 0), logical_shape, decode)
            stages.append({"stage": stage, "code_state_key": name, **measured})
    return {"method": "cpu_fp32_dense_decoder_estimate", "sampling": "measure_scale_evenly_spaced_rows",
            "gpu_packed_scale_may_differ_by_rounding": True, "stages": stages}


def optimize_local_bits(
    module: VAELinear,
    reference_weight: torch.Tensor,
    train_inputs: torch.Tensor,
    heldout_inputs: torch.Tensor,
    *,
    steps: int,
    lr: float,
    round_steps: int,
    optimizer: str = "adamw",
    chunk_size: int = 256,
    seed: int = 31,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Optimize both packed stages, preserving every other persistent tensor byte.

    This uses the existing CUDA/Triton decoder_sensitivity coordinates, with
    active_ratio=1 and streaming=False. All training tokens contribute on every step; held-out
    tokens only measure endpoints. Existing SparseBit early round termination is
    retained: after a flip, stable_steps consecutive no-flip steps can end a round
    before round_steps. Each new round resets proxy magnitudes and optimizer state.

    Frozen decoder parameters compute in FP32 and are restored byte-for-byte to
    their original storage dtype. Proxies/gradients are FP32, p=s*(bit-0.5), with
    radius=s/2 and chain-rule gradient 1/s. This changes coordinates and effective
    LR together; it is not a precision-only change to unit coordinates. Successful
    returns require actual committed bit changes.
    Any exception restores the complete original persistent state. Runtime flags
    are restored and derived caches invalidated on both success and failure.
    """
    decoder, parameters, device = _validate(
        module, reference_weight, train_inputs, heldout_inputs, steps, lr, chunk_size)
    if type(round_steps) is not int or round_steps < 1 or type(seed) is not int:
        raise ValueError("round_steps must be a positive integer and seed an integer.")
    if any(isinstance(child, nn.modules.dropout._DropoutNd) for child in module.modules()):
        raise ValueError("Dropout is outside the fixed-decoder experiment.")
    config = SparseBitTuningConfig(
        enabled=True, active_ratio=1.0, optimizer=optimizer, bit_lr=lr,
        weight_decay=0.0, round_steps=round_steps, proxy_coordinates="decoder_sensitivity").normalized()
    banks = _code_banks(module)
    if device.type != "cuda" or not sparse_bit_triton_available():
        raise RuntimeError("Local packed-bit optimization requires CUDA + Triton; no CPU training fallback exists.")

    original = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    original_parameters = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    code_names = {name for name, _ in banks}
    frozen_before = _digest(original, code_names)
    schema_before, bits_before = _schema(module), _storage_bits(module)
    parameter_ids = {name: id(p) for name, p in module.named_parameters()}
    grad_flags = [(p, p.requires_grad) for p in module.parameters()]
    training_flags = [(child, child.training) for child in module.modules()]
    runtime = {key: getattr(module, key) for key in ("trainable_decode", "cache_decoded_weight", "parallel_stage_decode")}
    had_compute_policy = hasattr(module, "_decoder_compute_dtype")
    compute_policy = getattr(module, "_decoder_compute_dtype", None)
    native_dtype = compute_policy or train_inputs.dtype
    reference = reference_weight.detach().to(device=device, dtype=torch.float32)
    train_x, heldout_x = train_inputs.detach().to(device), heldout_inputs.detach().to(device)
    root = nn.Module()
    root.add_module("layer", module)
    manager = None
    completed = False
    result = {}
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def measure(dtype):
        if getattr(module, "_sparse_bit_binding", None) is not None:
            raise RuntimeError("Endpoint measurement must not invalidate a live packed-bit runtime.")
        module._decoder_compute_dtype = dtype
        module.clear_decoded_weight_cache()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            delta = module._decode_weight(dtype=dtype).float() - reference
            values = {
                "train_mse": float(linear_output_mse(delta, train_x, chunk_size=chunk_size, recompute=False)),
                "heldout_mse": float(linear_output_mse(delta, heldout_x, chunk_size=chunk_size, recompute=False)),
                "weight_mse": float(delta.square().mean()),
            }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite packed-bit reconstruction metric.")
        return values

    def detach():
        if manager is not None:
            manager.detach_runtime()
        # Construction can fail after attaching a binding but before returning.
        if hasattr(module, "_sparse_bit_binding"):
            delattr(module, "_sparse_bit_binding")
        if hasattr(root, "sparse_bit_tuning"):
            delattr(root, "sparse_bit_tuning")

    def restore_decoder():
        for name, parameter in parameters.items():
            parameter.grad = None
            parameter.data = original_parameters[name].to(device=device).clone()

    try:
        module.eval()
        for parameter, _ in grad_flags:
            parameter.requires_grad_(False)
        module._build_parallel_stage_decode_plan()
        result["before_native"] = measure(native_dtype)
        for parameter in parameters.values():
            parameter.data = parameter.data.float()
        module.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
        result["before_fp32"] = measure(torch.float32)
        module.train()
        decoder.eval()
        # Build on the execution device before attaching. Rebuilding while the
        # manager is live would discard the non-streaming packed forward buffer.
        module._build_parallel_stage_decode_plan()
        manager = SparseBitTuningManager(
            root_model=root, targets=[("layer", module)], target_devices={"layer": device},
            training_seed=seed, config=config, streaming=False)
        manager.configure_schedule(total_optimizer_steps=steps)
        manager.initialize_scores()
        coordinate_scales = dict(manager.score_module.coordinate_scales)
        stage_coordinates = [{"stage": spec.stage_idx, "scale": coordinate_scales[spec.canonical_key],
                              "radius": manager.score_module.score_radius(spec),
                              "lr_over_radius": lr / manager.score_module.score_radius(spec)}
                             for spec in manager.bank_specs]
        stage_gradient_max = [0.0, 0.0]
        gradient_dtypes, proxy_dtypes, moment_dtypes = set(), set(), set()
        first_step_gradients = []
        losses, telemetry = [], []
        for step in range(steps):
            root.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=False):
                weight = module._decode_weight(dtype=torch.float32)
                loss = linear_output_mse(weight - reference, train_x, chunk_size=chunk_size, recompute=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite packed-bit loss at step {step + 1}.")
            loss.backward()
            if any(parameter.grad is not None for parameter, _ in grad_flags):
                raise RuntimeError("A frozen source parameter received a gradient.")
            stage_gradients = [0.0, 0.0]
            for spec in manager.bank_specs:
                proxy = manager.score_module.score_chunks[spec.chunk_id]
                gradient = proxy.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("Missing or non-finite packed-bit proxy gradient.")
                gradient_dtypes.add(str(gradient.dtype))
                proxy_dtypes.add(str(proxy.dtype))
                bank_gradient = gradient[spec.score_start:spec.score_end]
                value = float(torch.linalg.vector_norm(bank_gradient, dtype=torch.float32))
                stage_gradients[spec.stage_idx] = value
                stage_gradient_max[spec.stage_idx] = max(stage_gradient_max[spec.stage_idx], value)
                if step == 0:
                    absolute = bank_gradient.detach().abs()
                    count = bank_gradient.numel()
                    first_step_gradients.append({
                        "stage": spec.stage_idx, "numel": count,
                        "min": float(bank_gradient.min()), "max": float(bank_gradient.max()),
                        "absolute_max": float(absolute.max()),
                        "nonzero_fraction": int(torch.count_nonzero(bank_gradient)) / count,
                        "absolute_gt_1e_8_fraction": int(torch.count_nonzero(absolute > 1e-8)) / count,
                    })
            step_report = asdict(manager.optimizer_step())
            moment_dtypes.update(str(value.dtype) for value in manager.bit_optimizer.state_tensors())
            losses.append(float(loss.detach()))
            step_report.update({"step": step + 1, "loss": losses[-1], "stage_gradient_l2": stage_gradients})
            telemetry.append(step_report)
            if progress_callback is not None:
                progress_callback(dict(step_report))
        if not all(value > 0 for value in stage_gradient_max):
            raise RuntimeError("Both packed stages must receive nonzero proxy gradients.")
        if manager.score_module.coordinate_scales != coordinate_scales:
            raise RuntimeError("Fixed decoder sensitivity scales changed across optimizer rounds.")
        manager.final_commit()
        detach()
        module.eval()
        module._build_parallel_stage_decode_plan()
        result["after_fp32"] = measure(torch.float32)
        if any(not torch.equal(parameter.detach(), original_parameters[name].to(device=device, dtype=torch.float32))
               for name, parameter in parameters.items()):
            raise RuntimeError("Frozen FP32 decoder parameters changed during bit optimization.")
        restore_decoder()
        module.disable_trainable_decode()
        module.cache_decoded_weight = runtime["cache_decoded_weight"]
        module._build_parallel_stage_decode_plan()
        result["after_native"] = measure(native_dtype)
        stage_changes = [_changed_bits(original[name], module.state_dict()[name], shape) for name, shape in banks]
        if sum(stage_changes) == 0:
            raise RuntimeError("No permanent packed code changed; refusing a no-op result.")
        frozen_after = _digest(module.state_dict(), code_names)
        if (frozen_after != frozen_before or _schema(module) != schema_before
                or _storage_bits(module) != bits_before or _code_banks(module) != banks):
            raise RuntimeError("Non-code state or persistent dtype/shape/storage size changed.")
        if {name: id(p) for name, p in module.named_parameters()} != parameter_ids:
            raise RuntimeError("Optimization replaced registered Parameter objects.")
        torch.cuda.synchronize(device)
        result.update({
            "settings": {"steps": steps, "lr": lr, "round_steps": round_steps, "seed": seed,
                         "source_profile": _validate_payload_profile(module),
                         "chunk_size": chunk_size, "optimizer": config.optimizer, "weight_decay": 0.0,
                         "active_ratio": 1.0, "streaming": False, "proxy_coordinates": "decoder_sensitivity",
                         "stable_steps": manager.stable_steps, "early_stable_round_end": True,
                         "decoder_compute_dtype": "torch.float32", "native_decode_dtype": str(native_dtype),
                         "decoder_storage_dtype": str(next(iter(original_parameters.values())).dtype),
                         "train_tokens": len(train_x), "heldout_tokens": len(heldout_x)},
            "training_loss": losses, "step_telemetry": telemetry,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "evidence": {"stage_gradient_l2_max": stage_gradient_max,
                         "stage_coordinates": stage_coordinates, "coordinate_scales_fixed": True,
                         "first_step_gradients": first_step_gradients,
                         "stage_changed_bits": stage_changes, "final_changed_bits": sum(stage_changes),
                         "cumulative_step_flips": sum(item["step_flip_count"] for item in telemetry),
                         "rounds_ended": sum(item["round_ended"] for item in telemetry),
                         "proxy_parameter_dtypes": sorted(proxy_dtypes), "proxy_gradient_dtypes": sorted(gradient_dtypes),
                         "optimizer_moment_dtypes": sorted(moment_dtypes),
                         "frozen_sha256_before": frozen_before, "frozen_sha256_after": frozen_after,
                         "non_code_bytes_unchanged": True, "decoder_compute_parameters_unchanged": True,
                         "tensor_schema_unchanged": True,
                         "parameter_ids_unchanged": True, "packed_padding_unchanged": True,
                         "code_state_keys": [name for name, _ in banks],
                         "persistent_storage_bits_before": bits_before, "persistent_storage_bits_after": _storage_bits(module)},
        })
        completed = True
        return result
    finally:
        detach()
        restore_decoder()
        if not completed:
            module.load_state_dict(original, strict=True)
        for parameter, requires_grad in grad_flags:
            parameter.grad = None
            parameter.requires_grad_(requires_grad)
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
