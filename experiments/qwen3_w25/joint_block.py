"""One protected Qwen3 block: joint existing-decoder and packed-code recovery.

This isolated core owns no model driver, data selection, quality recipe or save
format. Bit learning rates and round lengths are explicit for every projection.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import math
import time
from typing import Callable, Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from experiments.qwen3_w25.block_weight_graph import block_weight_graph
from experiments.qwen3_w25.local_bits import _changed_bits, _code_banks, _digest
from experiments.qwen3_w25.local_block import BlockBatch, INPUT_SOURCE, PROJECTIONS, _squared_error, _validate
from experiments.qwen3_w25.local_decoder import _schema, _stage_norm, _storage_bits, _validate_payload_profile
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager
from sparse_bit_tuning.triton_kernels import sparse_bit_triton_available


def _bit_configurations(bit_lrs, round_steps, seed):
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    for label, values in (("bit_lrs", bit_lrs), ("round_steps", round_steps)):
        if not isinstance(values, Mapping) or set(values) != set(PROJECTIONS):
            raise ValueError(f"{label} must explicitly specify exactly the seven projection names.")
    configurations = {}
    for name in PROJECTIONS:
        lr, rounds = bit_lrs[name], round_steps[name]
        if isinstance(lr, bool) or not isinstance(lr, (float, int)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"{name}: bit_lrs must be finite positive numbers.")
        if type(rounds) is not int or rounds < 1:
            raise ValueError(f"{name}: round_steps must be positive integers.")
        configurations[name] = SparseBitTuningConfig(
            enabled=True, active_ratio=1.0, optimizer="adamw", bit_lr=float(lr),
            weight_decay=0.0, round_steps=rounds, proxy_coordinates="decoder_sensitivity").normalized()
    return configurations


def optimize_joint_block(
    block: Qwen3DecoderLayer,
    train_batches: Sequence[BlockBatch],
    heldout_batches: Sequence[BlockBatch],
    *,
    steps: int,
    decoder_lr: float,
    bit_lrs: Mapping[str, float],
    round_steps: Mapping[str, int],
    seed: int = 31,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Jointly update seven decoders and fourteen code banks on fixed inputs.

    Every step accumulates all valid training tokens through the original BF16
    Qwen3 block, followed by one FP32 decoded-weight VJP into both variable sets.
    Then native AdamW updates the decoders and each production SparseBit manager
    updates its own two banks. All graphs are released before either update.

    Each manager calibrates fixed FP32 decoder-sensitivity coordinates once and
    retains its independent round/early-stability rule. Held-out data measures
    endpoints only. A bank, or even all banks, may finish with zero changed bits;
    the report distinguishes this from evidence that code learning occurred.
    Success changes only the existing decoder values and logical packed bits.
    Exceptions restore the complete original block, including all packed codes.
    """
    configs = _bit_configurations(bit_lrs, round_steps, seed)
    targets, decoders, parameters, device = _validate(
        block, train_batches, heldout_batches, steps, decoder_lr)
    profiles = {name: _validate_payload_profile(module) for name, module in targets}
    if set(profiles.values()) != {"no_rotation_int8_input"}:
        raise ValueError("Joint block recovery requires the no-rotation INT8 input-protected source.")
    banks = {name: _code_banks(module) for name, module in targets}
    if device.type != "cuda" or not sparse_bit_triton_available():
        raise RuntimeError("Joint block recovery requires real CUDA/Triton SparseBit; no CPU training fallback.")

    original = {name: tensor.detach().cpu().clone() for name, tensor in block.state_dict().items()}
    code_names = {name + "." + key for name, entries in banks.items() for key, _shape in entries}
    allowed = set(parameters) | code_names
    parameter_ids = {name: id(p) for name, p in block.named_parameters()}
    grad_flags = [(p, p.requires_grad) for p in block.parameters()]
    modes = [(child, child.training) for child in block.modules()]
    runtimes = {name: {key: getattr(module, key) for key in
                      ("trainable_decode", "cache_decoded_weight", "parallel_stage_decode")}
                for name, module in targets}
    policies = {name: (hasattr(module, "_decoder_compute_dtype"), getattr(module, "_decoder_compute_dtype", None))
                for name, module in targets}
    frozen_before = _digest(original, allowed)
    schema_before, bits_before = _schema(block), _storage_bits(block)
    train_tokens = sum(int(batch.valid_mask.sum()) for batch in train_batches)
    heldout_tokens = sum(int(batch.valid_mask.sum()) for batch in heldout_batches)
    roots, managers, coordinates = {}, {}, {}
    module_reports = {name: {"step_telemetry": [], "stage_proxy_gradient_l2_max": [0.0, 0.0],
                             "stage_decoder_gradient_l2_max": [0.0, 0.0]}
                      for name in PROJECTIONS}
    proxy_dtypes, gradient_dtypes, bit_moment_dtypes = set(), set(), set()
    result, completed, optimizer = {}, False, None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def measure():
        metrics = {}
        with torch.no_grad():
            for split, batches, tokens in (("train", train_batches, train_tokens),
                                            ("heldout", heldout_batches, heldout_tokens)):
                value = sum(float(_squared_error(block, batch, device)) for batch in batches)
                value /= tokens * block.hidden_size
                if not math.isfinite(value) or value < 0:
                    raise FloatingPointError("Non-finite or negative joint block reconstruction metric.")
                metrics[split + "_mse"] = value
        return metrics

    def native_policy():
        for name, module in targets:
            if getattr(module, "_sparse_bit_binding", None) is not None:
                raise RuntimeError("Detach SparseBit before restoring the native block policy.")
            module.eval()
            module.trainable_decode = False
            module.cache_decoded_weight = runtimes[name]["cache_decoded_weight"]
            module._decoder_compute_dtype = torch.bfloat16
            module._build_parallel_stage_decode_plan()
            module.clear_decoded_weight_cache()

    def detach_managers():
        for manager in managers.values():
            manager.detach_runtime()
        # Construction can attach state before raising and before returning a manager.
        for _name, module in targets:
            if hasattr(module, "_sparse_bit_binding"):
                delattr(module, "_sparse_bit_binding")
        for root in roots.values():
            if hasattr(root, "sparse_bit_tuning"):
                delattr(root, "sparse_bit_tuning")

    def stage_delta(name, module):
        return _stage_norm([
            p.detach().float() - original[name + "._parallel_stage_decoder." + key].to(device, torch.float32)
            for key, p in decoders[name].items()], module._parallel_stage_layout)

    try:
        block.eval().requires_grad_(False)
        native_policy()
        result["before_native"] = measure()
        for p in parameters.values():
            p.data = p.data.float()
        for _name, module in targets:
            module.enable_trainable_sparse_bit_decode_graph(parallel_stage_decode=True)
            module._decoder_compute_dtype = torch.float32
            module.training = True
            module._parallel_stage_decoder.eval()
            module._build_parallel_stage_decode_plan()
        with block_weight_graph(targets, requires_grad=False):
            result["before_fp32_decode"] = measure()

        for name, module in targets:
            # Scores live outside the block state_dict and cannot leak into its save.
            root = roots[name] = nn.Module()
            manager = managers[name] = SparseBitTuningManager(
                root_model=root, targets=[(name, module)], target_devices={name: device},
                training_seed=seed, config=configs[name], streaming=False,
                calibration_dtype=torch.float32)
            manager.configure_schedule(total_optimizer_steps=steps)
            manager.initialize_scores()
            coordinates[name] = dict(manager.score_module.coordinate_scales)
            module_reports[name]["stage_coordinates"] = [
                {"stage": spec.stage_idx, "scale": coordinates[name][spec.canonical_key],
                 "radius": manager.score_module.score_radius(spec), "numel": spec.n_bits,
                 "lr_over_radius": configs[name].resolved_lr() / manager.score_module.score_radius(spec)}
                for spec in sorted(manager.bank_specs, key=lambda spec: spec.stage_idx)]
        optimizer = torch.optim.AdamW(parameters.values(), lr=decoder_lr, weight_decay=0.0)
        losses = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for root in roots.values():
                root.zero_grad(set_to_none=True)
            step_loss = 0.0
            with block_weight_graph(targets, requires_grad=True) as graph:
                for batch in train_batches:
                    loss = _squared_error(block, batch, device) / (train_tokens * block.hidden_size)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite joint block loss at step {step + 1}.")
                    loss.backward()
                    step_loss += float(loss.detach())
                graph.backward_to_decoders()
            if any(p.grad is not None for name, p in block.named_parameters() if name not in parameters):
                raise RuntimeError("A frozen block parameter received a gradient.")
            gradients = {}
            for name, module in targets:
                report = module_reports[name]
                decoder_grad = _stage_norm([p.grad for p in decoders[name].values()], module._parallel_stage_layout)
                proxy_grad, first = [0.0, 0.0], []
                manager = managers[name]
                for spec in sorted(manager.bank_specs, key=lambda spec: spec.stage_idx):
                    score = manager.score_module.score_chunks[spec.chunk_id]
                    gradient = score.grad
                    if (gradient is None or score.dtype != torch.float32 or gradient.dtype != torch.float32
                            or not bool(torch.isfinite(gradient).all())):
                        raise FloatingPointError(f"{name}: missing, non-FP32 or non-finite packed proxy gradient.")
                    proxy_dtypes.add(str(score.dtype))
                    gradient_dtypes.add(str(gradient.dtype))
                    bank_gradient = gradient[spec.score_start:spec.score_end]
                    proxy_grad[spec.stage_idx] = float(torch.linalg.vector_norm(bank_gradient))
                    if step == 0:
                        first.append({"stage": spec.stage_idx, "numel": bank_gradient.numel(),
                                      "nonzero_fraction": int(torch.count_nonzero(bank_gradient)) / bank_gradient.numel(),
                                      "absolute_max": float(bank_gradient.abs().max())})
                report["stage_decoder_gradient_l2_max"] = [max(a, b) for a, b in zip(
                    report["stage_decoder_gradient_l2_max"], decoder_grad)]
                report["stage_proxy_gradient_l2_max"] = [max(a, b) for a, b in zip(
                    report["stage_proxy_gradient_l2_max"], proxy_grad)]
                if step == 0:
                    report["first_step_proxy_gradients"] = first
                gradients[name] = {"stage_decoder_gradient_l2": decoder_grad, "stage_proxy_gradient_l2": proxy_grad}

            # Same ordering as SparseBitCompositeOptimizer; both use pre-update gradients.
            optimizer.step()
            rows = {}
            for name, manager in managers.items():
                row = asdict(manager.optimizer_step())
                row.update(step=step + 1, **gradients[name])
                module_reports[name]["step_telemetry"].append(row)
                rows[name] = row
                bit_moment_dtypes.update(str(t.dtype) for t in manager.bit_optimizer.state_tensors())
            losses.append(step_loss)
            if progress_callback is not None:
                progress_callback(deepcopy({"step": step + 1, "loss": step_loss, "modules": rows}))

        for name, manager in managers.items():
            if coordinates[name] != manager.score_module.coordinate_scales:
                raise RuntimeError(f"{name}: decoder-sensitivity scales changed during joint training.")
            manager.final_commit()
        detach_managers()
        for _name, module in targets:
            module._build_parallel_stage_decode_plan()
            module.clear_decoded_weight_cache()
        with block_weight_graph(targets, requires_grad=False):
            result["after_fp32_decode"] = measure()
        for name, module in targets:
            module_reports[name]["stage_parameter_delta_l2_fp32"] = stage_delta(name, module)
        decoder_moment_dtypes = {str(state[key].dtype) for state in optimizer.state.values()
                                for key in ("exp_avg", "exp_avg_sq")}
        if decoder_moment_dtypes != {"torch.float32"} or bit_moment_dtypes != {"torch.float32"}:
            raise RuntimeError("Both AdamW optimizers must keep FP32 moments.")
        optimizer.zero_grad(set_to_none=True)
        for name, p in parameters.items():
            p.data = p.data.to(original[name].dtype)
        native_policy()
        result["after_native"] = measure()
        final_state = block.state_dict()
        for name, module in targets:
            report = module_reports[name]
            report["stage_parameter_delta_l2_saved"] = stage_delta(name, module)
            for key in ("stage_decoder_gradient_l2_max", "stage_parameter_delta_l2_fp32", "stage_parameter_delta_l2_saved"):
                if not all(value > 0 for value in report[key]):
                    raise RuntimeError(f"{name}: both decoder stages must receive gradients and retain updates ({key}).")
            report["stage_changed_bits"] = [
                _changed_bits(original[name + "." + key], final_state[name + "." + key], shape)
                for key, shape in banks[name]]
            report["final_changed_bits"] = sum(report["stage_changed_bits"])
            report["cumulative_step_flips"] = sum(row["step_flip_count"] for row in report["step_telemetry"])
            report["rounds_ended"] = sum(row["round_ended"] for row in report["step_telemetry"])
            report["stable_steps"] = managers[name].stable_steps
            report["code_state_keys"] = [key for key, _shape in banks[name]]
            if _code_banks(module) != banks[name]:
                raise RuntimeError(f"{name}: persistent code-bank identity/schema changed.")
        frozen_after = _digest(final_state, allowed)
        if frozen_after != frozen_before or _schema(block) != schema_before or _storage_bits(block) != bits_before:
            raise RuntimeError("Frozen block bytes or persistent dtype/shape/storage changed.")
        if {name: id(p) for name, p in block.named_parameters()} != parameter_ids:
            raise RuntimeError("Joint recovery replaced registered Parameter objects.")
        changed_bits = sum(report["final_changed_bits"] for report in module_reports.values())
        result.update(
            settings={"steps": steps, "decoder_lr": decoder_lr, "seed": seed,
                      "bit_lrs": {name: configs[name].resolved_lr() for name in PROJECTIONS},
                      "round_steps": {name: configs[name].round_steps for name in PROJECTIONS},
                      "source_profiles": profiles, "objective": "block_output_mse", "input_source": INPUT_SOURCE,
                      "decoder_optimizer": "AdamW", "bit_optimizer": "adamw", "weight_decay": 0.0,
                      "active_ratio": 1.0, "streaming": False, "proxy_coordinates": "decoder_sensitivity",
                      "early_stable_round_end": True, "independent_projection_rounds": True,
                      "decoder_compute_dtype": "torch.float32", "projection_dtype": "torch.bfloat16",
                      "native_decode_dtype": "torch.bfloat16", "weight_gradient_dtype": "torch.float32",
                      "gradient_path": "fp32_weight_gradient_then_joint_decoder_and_bit_vjp",
                      "optimizer_update_order": "decoder_then_projection_bit_managers",
                      "cross_block_gradients": False, "train_tokens": train_tokens, "heldout_tokens": heldout_tokens,
                      "storage_dtypes": {name: str(original[name + "._parallel_stage_decoder." + next(iter(local))].dtype)
                                         for name, local in decoders.items()}},
            training_loss=losses, modules=module_reports,
            evidence={"allowed_state_keys": sorted(allowed), "code_state_keys": sorted(code_names),
                      "final_changed_bits": changed_bits, "has_committed_code_change": changed_bits > 0,
                      "frozen_sha256_before": frozen_before, "frozen_sha256_after": frozen_after,
                      "frozen_unchanged": True, "tensor_schema_unchanged": True, "parameter_ids_unchanged": True,
                      "packed_padding_unchanged": True, "coordinate_scales_fixed": True,
                      "proxy_parameter_dtypes": sorted(proxy_dtypes), "proxy_gradient_dtypes": sorted(gradient_dtypes),
                      "bit_optimizer_moment_dtypes": sorted(bit_moment_dtypes),
                      "decoder_optimizer_moment_dtypes": sorted(decoder_moment_dtypes),
                      "persistent_storage_bits_before": bits_before, "persistent_storage_bits_after": _storage_bits(block)})
        completed = True
    finally:
        detach_managers()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for name, p in parameters.items():
            p.grad = None
            p.data = p.data.to(original[name].dtype) if completed else original[name].to(device).clone()
        if not completed:
            block.load_state_dict(original, strict=True)
        for p, flag in grad_flags:
            p.grad = None
            p.requires_grad_(flag)
        for child, mode in modes:
            child.training = mode
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
        if completed:
            torch.cuda.synchronize(device)
            result["elapsed_seconds"] = time.perf_counter() - started
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return result
