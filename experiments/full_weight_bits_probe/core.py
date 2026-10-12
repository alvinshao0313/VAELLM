"""Matched full-matrix decoder and native 100%-bit weight reconstruction."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.joint_reconstruction_probe.core import (EPS, LR, decode, error_stats,
    require, schema, stage_norm, storage_bits)
from experiments.rotation_factor_probe.capture import tensor_record
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager


TOTAL_STEPS = 500
BIT_CONFIG = SparseBitTuningConfig(enabled=True, active_ratio=1.0, optimizer="adam",
    bit_lr=2e-5, weight_decay=0.0, round_steps=512, proxy_coordinates="decoder_sensitivity").normalized()


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("The assigned 60-minute total wall-clock budget expired; no automatic retry.")


def popcounts(left, right):
    require(left.shape == right.shape and left.dtype == right.dtype == torch.uint8,
            "Packed XOR inputs must have matching uint8 layouts.")
    lut = torch.tensor([n.bit_count() for n in range(256)], device=left.device, dtype=torch.int16)
    return int(lut[torch.bitwise_xor(left, right.to(left.device)).long()].sum())


def gradient_vector(parameters):
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
            "A registered decoder gradient is missing or nonfinite.")
    return torch.cat([p.grad.detach().reshape(-1) for p in parameters.values()]).cpu()


def vector_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def train(layer, target, *, joint, steps, deadline, observer, control_trace=None):
    """Use the complete native grouped bank on every forward, never remap scores."""
    require(steps in (2, TOTAL_STEPS) and target.dtype == torch.float32 and target.ndim == 2,
            "Only a 2-update mechanism check or the assigned 500 updates is allowed.")
    device = next(layer.parameters()).device
    require(device.type == "cuda" and not layer.transpose and layer.residual_stages == 2,
            "This probe requires the real CUDA two-stage gate geometry.")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    original = {name: value.detach().cpu().clone() for name, value in layer.state_dict().items()}
    before_schema, before_bits = schema(layer), storage_bits(layer)
    banks = [layer.get_stage_part_vq_storage(s, 0).detach().cpu().clone() for s in range(2)]
    initial = {name: value.detach().clone() for name, value in layer._parallel_stage_decoder.named_parameters()}
    before_native = decode(layer, deployment=True)
    report = {"before_deployment": error_stats(before_native, target)}
    denominator = report["before_deployment"]["target_second_moment"]
    del before_native
    for parameter in layer._parallel_stage_decoder.parameters():
        parameter.data = parameter.data.float()
    if joint:
        layer.enable_trainable_sparse_bit_decode_graph(parallel_stage_decode=True)
    else:
        layer.enable_trainable_decode(parallel_stage_decode=True)
    layer._decoder_compute_dtype = torch.bfloat16
    layer.training = True
    layer._parallel_stage_decoder.eval()
    layer._build_parallel_stage_decode_plan()
    parameters = dict(layer._parallel_stage_decoder.named_parameters())
    parameter_ids = {name: id(p) for name, p in parameters.items()}
    require(all(p.requires_grad and p.dtype == torch.float32 for p in parameters.values()),
            "Both modes require registered FP32 decoder masters.")
    target_gpu = target.to(device)
    root = nn.Module()
    manager = None
    telemetry, gradients, events = [], [], [0, 0]
    grad_max, proxy_grad_max = [0.0, 0.0], [0.0, 0.0]
    coordinates = {}
    first_flip = None
    optimizer = torch.optim.AdamW(parameters.values(), lr=LR, betas=(.9, .95), eps=EPS, weight_decay=0)
    try:
        if joint:
            manager = SparseBitTuningManager(root_model=root, targets=[("layer", layer)],
                target_devices={"layer": device}, training_seed=31, config=BIT_CONFIG,
                streaming=False, calibration_dtype=torch.bfloat16)
            manager.configure_schedule(total_optimizer_steps=TOTAL_STEPS)
            manager.initialize_scores()
            require(manager.bit_round_steps == 512 and manager.stable_steps == 10
                    and all(s.n_active == s.n_bits for s in manager.bank_specs), "Native full-code schedule changed.")
            coordinates = dict(manager.score_module.coordinate_scales)
            report["coordinates"] = [{"stage": s.stage_idx, "scale": coordinates[s.canonical_key],
                "radius": manager.score_module.score_radius(s), "n_bits": s.n_bits,
                "n_active": s.n_active, "logical_shape": list(s.logical_shape)} for s in manager.bank_specs]
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            report["before_train_decode"] = error_stats(layer._decode_weight(torch.bfloat16), target)
        for step in range(1, steps + 1):
            check_deadline(deadline)
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            root.zero_grad(set_to_none=True)
            optimizer.param_groups[0]["lr"] = LR * (1.0 - (step - 1) / TOTAL_STEPS)
            with torch.autocast("cuda", enabled=False):
                reconstructed = layer._decode_weight(dtype=torch.bfloat16)
                require(reconstructed.shape == target_gpu.shape and reconstructed.dtype == torch.bfloat16,
                        "Native full-weight output geometry or BF16 boundary changed.")
                raw_mse = F.mse_loss(reconstructed.float(), target_gpu)
                loss = raw_mse / denominator
            require(bool(torch.isfinite(loss)), f"Nonfinite full-weight loss at step {step}.")
            loss.backward()
            grad = gradient_vector(parameters)
            norms = stage_norm([p.grad for p in parameters.values()])
            grad_max = [max(a, b) for a, b in zip(grad_max, norms, strict=True)]
            item = {"step": step, "raw_mse": float(raw_mse), "nmse_loss": float(loss),
                "decoder_lr": optimizer.param_groups[0]["lr"], "decoder_gradient_sha256": vector_hash(grad),
                "stage_decoder_gradient_l2": norms,
                "decoder_gradient_abs_max": float(grad.abs().max()),
                "decoder_gradient_abs_le_eps_fraction": float((grad.abs() <= EPS).float().mean())}
            if control_trace is None:
                gradients.append(grad)
            else:
                reference = control_trace[step - 1]
                delta = grad.double() - reference.double()
                item["decoder_gradient_vs_control"] = {"exact": torch.equal(grad, reference),
                    "relative_l2": float(delta.norm() / reference.double().norm().clamp_min(1e-30)),
                    "max_abs": float(delta.abs().max()), "before_first_bit_flip": first_flip is None}
            grouped_before = None
            if manager is not None:
                proxy_rows = []
                for spec in sorted(manager.bank_specs, key=lambda s: s.stage_idx):
                    score = manager.score_module.score_view(spec)
                    chunk = manager.score_module.score_chunks[spec.chunk_id]
                    require(chunk.grad is not None and chunk.dtype == chunk.grad.dtype == torch.float32,
                            "Native sensitivity score/gradient path is disconnected or not FP32.")
                    bank_grad = chunk.grad[spec.score_start:spec.score_end]
                    require(bool(torch.isfinite(bank_grad).all()), "Nonfinite native score gradient.")
                    norm = float(bank_grad.norm())
                    proxy_grad_max[spec.stage_idx] = max(proxy_grad_max[spec.stage_idx], norm)
                    row = {"stage": spec.stage_idx, "numel": bank_grad.numel(), "gradient_l2": norm}
                    if step <= 2:
                        row.update(nonzero_fraction=float(torch.count_nonzero(bank_grad)) / bank_grad.numel(),
                            abs_max=float(bank_grad.abs().max()), abs_mean=float(bank_grad.abs().mean()),
                            abs_le_eps_fraction=float((bank_grad.abs() <= EPS).float().mean()))
                    proxy_rows.append(row)
                item["proxy_gradients"] = proxy_rows
                # Observe native packed SET events; this does not invent a score mapping.
                grouped_before = layer._get_parallel_stage_grouped_vq_packed(device=device).detach().clone()
            optimizer.step()
            if manager is not None:
                item["bit"] = asdict(manager.optimizer_step())
                grouped_after = layer._get_parallel_stage_grouped_vq_packed(device=device)
                stage_events = [popcounts(grouped_before[:, s, :], grouped_after[:, s, :]) for s in range(2)]
                require(sum(stage_events) == item["bit"]["step_flip_count"], "Native per-stage packed event counters disagree.")
                events = [a + b for a, b in zip(events, stage_events, strict=True)]
                item["stage_step_flip_events"] = stage_events
                if sum(stage_events) and first_flip is None:
                    first_flip = step
                if step <= 2:
                    item["score_interior_fraction"] = [float((manager.score_module.score_view(s).abs()
                        < manager.score_module.score_radius(s)).float().mean()) for s in manager.bank_specs]
            torch.cuda.synchronize(device)
            item["step_seconds"] = time.perf_counter() - step_started
            telemetry.append(item)
            if step <= 2 or step % 50 == 0:
                observer(item)
            del reconstructed, raw_mse, loss, grouped_before, grad
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            live = layer._decode_weight(dtype=torch.bfloat16)
        report["after_live_train_decode"] = error_stats(live, target)
        decoder_moments = sorted({str(state[k].dtype) for state in optimizer.state.values()
                                  for k in ("exp_avg", "exp_avg_sq")})
        require(decoder_moments == ["torch.float32"], "Decoder moments must remain FP32.")
        if manager is not None:
            require(coordinates == manager.score_module.coordinate_scales, "Sensitivity coordinates changed.")
            report["bit_moment_dtypes"] = sorted({str(t.dtype) for t in manager.bit_optimizer.state_tensors()})
            require(report["bit_moment_dtypes"] == ["torch.float32"] and all(v > 0 for v in proxy_grad_max),
                    "Both full banks require real FP32 proxy gradients/moments.")
            manager.final_commit()
            manager.detach_runtime()
        layer._build_parallel_stage_decode_plan()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            committed = layer._decode_weight(dtype=torch.bfloat16)
        require(torch.equal(live, committed), "Final native bit commit changed the live weight function.")
        del live, committed, optimizer, target_gpu
        fp32_delta = stage_norm([p.detach() - initial[name].float() for name, p in parameters.items()])
        for name, p in parameters.items():
            p.grad = None
            p.data = p.data.to(initial[name].dtype)
        layer.eval().disable_trainable_decode()
        layer._build_parallel_stage_decode_plan()
        deployed = decode(layer, deployment=True)
        report["after_deployment"] = error_stats(deployed, target)
        bf16_delta = stage_norm([p.detach().float() - initial[name].float() for name, p in parameters.items()])
        final = layer.state_dict()
        changed = [popcounts(banks[s].to(device), layer.get_stage_part_vq_storage(s, 0)) for s in range(2)]
        require(joint or sum(changed) == 0, "Decoder control changed fixed hard codes.")
        require(schema(layer) == before_schema and storage_bits(layer) == before_bits,
                "Persistent topology, dtype, or bit width changed.")
        allowed = {name for name in original if name.startswith("_parallel_stage_decoder.")}
        if joint:
            allowed.update({"vq_weight", "vq_weight_s1"})
        require(all(torch.equal(value, final[name].cpu()) for name, value in original.items() if name not in allowed),
                "A frozen persistent tensor changed.")
        require(parameter_ids == {name: id(p) for name, p in layer._parallel_stage_decoder.named_parameters()},
                "Registered decoder parameter objects changed.")
        require(all(x > 0 for x in grad_max + fp32_delta + bf16_delta), "Both decoder stages must retain real updates.")
        report.update(status="completed", joint=joint, steps=steps, schedule_steps=TOTAL_STEPS,
            trajectory=telemetry, first_flip_step=first_flip, stage_flip_events=events,
            stage_final_net_xor=changed, stage_decoder_gradient_l2_max=grad_max,
            stage_proxy_gradient_l2_max=proxy_grad_max, stage_decoder_delta_l2_fp32=fp32_delta,
            stage_decoder_delta_l2_bf16=bf16_delta, decoder_moment_dtypes=decoder_moments,
            final_commit_live_decode_exact=True, frozen_state_unchanged=True, persistent_schema_unchanged=True,
            persistent_bits_before=before_bits, persistent_bits_after=storage_bits(layer),
            initial_payload={name: tensor_record(value) for name, value in original.items()},
            final_payload={name: tensor_record(value) for name, value in final.items()},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            elapsed_seconds=time.perf_counter() - start,
            settings={"target": "Complete original rotated FP32 matrix", "loss": "full MSE / fixed original mean(W^2)",
                "normalizer": denominator, "decoder_lr": LR, "decoder_betas": [.9, .95], "decoder_eps": EPS,
                "decoder_weight_decay": 0, "decoder_schedule": "500-step linear, warmup0",
                "decoder_master": "float32", "compute_and_stage_sum": "bfloat16",
                "bit_config": asdict(BIT_CONFIG) if joint else None, "calibration_dtype": "bfloat16",
                "native_early_round_end": "After any flip, ten consecutive no-flip steps; retain original behavior."})
        return deployed.cpu(), report, gradients
    finally:
        if manager is not None:
            manager.detach_runtime()
        if hasattr(root, "sparse_bit_tuning"):
            delattr(root, "sparse_bit_tuning")
