"""Fresh 2000-step fixed-code fitting with the existing native full-matrix loss."""
from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from experiments.full_weight_bits_probe.core import popcounts
from experiments.joint_reconstruction_probe.core import (EPS, LR, decode, error_stats,
    require, schema, stage_norm, storage_bits)
from experiments.rotation_factor_probe.capture import tensor_record


TOTAL_STEPS = 2000


def train(layer, target, *, deadline, observer):
    """Fit both decoder stages from their supplied common BF16 initialization.

    Each update sees the complete rotated matrix. Hard codes never change here;
    replaying the previously selected exact neighbors belongs to the caller.
    The 2000-step LR horizon is a fresh recipe, not a resume of the old 500 steps.
    """
    require(target.dtype == torch.float32 and target.ndim == 2,
            "The original complete rotated target must remain FP32.")
    device = next(layer.parameters()).device
    require(device.type == "cuda" and not layer.transpose and layer.residual_stages == 2,
            "This probe requires the real CUDA two-stage gate geometry.")
    require(getattr(layer, "_sparse_bit_binding", None) is None,
            "Fixed-code fitting must not have an active bit-update runtime.")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    original = {name: value.detach().cpu().clone() for name, value in layer.state_dict().items()}
    before_schema, before_bits = schema(layer), storage_bits(layer)
    banks = [layer.get_stage_part_vq_storage(s, 0).detach().cpu().clone() for s in range(2)]
    initial = {name: value.detach().clone() for name, value in layer._parallel_stage_decoder.named_parameters()}
    require(initial and all(value.dtype == torch.bfloat16 for value in initial.values()),
            "Both arms must start with their original finalized BF16 decoder values.")
    before_native = decode(layer, deployment=True)
    report = {"before_deployment": error_stats(before_native, target)}
    denominator = report["before_deployment"]["target_second_moment"]
    del before_native
    for parameter in layer._parallel_stage_decoder.parameters():
        parameter.data = parameter.data.float()
    layer.enable_trainable_decode(parallel_stage_decode=True)
    layer._decoder_compute_dtype = torch.bfloat16
    layer.training = True
    layer._parallel_stage_decoder.eval()
    layer._build_parallel_stage_decode_plan()
    parameters = dict(layer._parallel_stage_decoder.named_parameters())
    parameter_ids = {name: id(p) for name, p in parameters.items()}
    require(all(p.requires_grad and p.dtype == torch.float32 for p in parameters.values()),
            "Both stages require registered FP32 decoder masters.")
    target_gpu = target.to(device)
    optimizer = torch.optim.AdamW(parameters.values(), lr=LR, betas=(.9, .95), eps=EPS, weight_decay=0)
    telemetry, grad_max = [], [0.0, 0.0]
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        report["before_train_decode"] = error_stats(layer._decode_weight(torch.bfloat16), target)
    for step in range(1, TOTAL_STEPS + 1):
        if time.time() >= deadline:
            raise TimeoutError("The assigned decoder-budget probe deadline expired; no automatic retry.")
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = LR * (1.0 - (step - 1) / TOTAL_STEPS)
        with torch.autocast("cuda", enabled=False):
            reconstructed = layer._decode_weight(dtype=torch.bfloat16)
            require(reconstructed.shape == target_gpu.shape and reconstructed.dtype == torch.bfloat16,
                    "Native full-weight output geometry or BF16 boundary changed.")
            raw_mse = F.mse_loss(reconstructed.float(), target_gpu)
            loss = raw_mse / denominator
        require(bool(torch.isfinite(loss)), f"Nonfinite full-weight loss at step {step}.")
        loss.backward()
        require(all(p.grad is not None and p.grad.dtype == torch.float32
                    and bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
                "A registered FP32 decoder gradient is missing or nonfinite.")
        norms = stage_norm([p.grad for p in parameters.values()])
        grad_max = [max(a, b) for a, b in zip(grad_max, norms, strict=True)]
        optimizer.step()
        torch.cuda.synchronize(device)
        item = {"step": step, "raw_mse": float(raw_mse), "nmse_loss": float(loss),
                "decoder_lr": optimizer.param_groups[0]["lr"],
                "stage_decoder_gradient_l2": norms,
                "step_seconds": time.perf_counter() - step_started}
        telemetry.append(item)
        if step <= 2 or step % 50 == 0:
            observer(item)
        del reconstructed, raw_mse, loss
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        live = layer._decode_weight(dtype=torch.bfloat16)
    report["after_live_train_decode"] = error_stats(live, target)
    moments = sorted({str(state[k].dtype) for state in optimizer.state.values()
                      for k in ("exp_avg", "exp_avg_sq")})
    require(moments == ["torch.float32"], "Decoder moments must remain FP32.")
    layer._build_parallel_stage_decode_plan()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        rebuilt = layer._decode_weight(dtype=torch.bfloat16)
    require(torch.equal(live, rebuilt), "Rebuilding the frozen packed plan changed the live weight function.")
    del live, rebuilt, optimizer, target_gpu
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
    require(changed == [0, 0], "Fixed-code fitting changed a hard bank.")
    require(schema(layer) == before_schema and storage_bits(layer) == before_bits,
            "Persistent topology, dtype, or bit width changed.")
    allowed = {name for name in original if name.startswith("_parallel_stage_decoder.")}
    require(all(torch.equal(value, final[name].cpu()) for name, value in original.items() if name not in allowed),
            "A frozen persistent tensor changed.")
    require(parameter_ids == {name: id(p) for name, p in layer._parallel_stage_decoder.named_parameters()},
            "Registered decoder parameter objects changed.")
    require(all(x > 0 for x in grad_max + fp32_delta + bf16_delta),
            "Both decoder stages must retain real FP32 and deployed BF16 updates.")
    report.update(status="completed", joint=False, steps=TOTAL_STEPS, schedule_steps=TOTAL_STEPS,
        trajectory=telemetry, first_flip_step=None, stage_flip_events=[0, 0], stage_final_net_xor=changed,
        stage_decoder_gradient_l2_max=grad_max, stage_decoder_delta_l2_fp32=fp32_delta,
        stage_decoder_delta_l2_bf16=bf16_delta, decoder_moment_dtypes=moments,
        final_commit_live_decode_exact=True, frozen_state_unchanged=True, persistent_schema_unchanged=True,
        persistent_bits_before=before_bits, persistent_bits_after=storage_bits(layer),
        initial_payload={name: tensor_record(value) for name, value in original.items()},
        final_payload={name: tensor_record(value) for name, value in final.items()},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device), elapsed_seconds=time.perf_counter() - start,
        settings={"target": "Complete original rotated FP32 matrix",
            "loss": "full MSE / fixed original mean(W^2)", "normalizer": denominator,
            "decoder_lr": LR, "decoder_betas": [.9, .95], "decoder_eps": EPS, "decoder_weight_decay": 0,
            "decoder_schedule": "2000-step linear, warmup0", "decoder_master": "float32",
            "compute_and_stage_sum": "bfloat16", "bit_config": None,
            "initialization": "Fresh common native2k BF16 decoder, never an optimizer resume",
            "training_inputs": "Complete matrix only; no calibration documents enter optimization",
            "budget_boundary": "The horizon changes earlier learning rates versus the old 500-step recipe."})
    return deployed.cpu(), report
