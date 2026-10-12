"""One fixed-decoder pass over every genuine single-bit D32 neighbor.

This is an isolated finite search, not a SparseBit optimizer or CAT extension.
The native SparseBit graph is used once, solely to observe initial derivatives.
"""
from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.full_weight_bits_probe.core import BIT_CONFIG, popcounts
from experiments.joint_reconstruction_probe.core import decode, error_stats, require, schema, storage_bits
from experiments.rotation_factor_probe.capture import tensor_record
from sparse_bit_tuning.manager import SparseBitTuningManager


CHUNK = 32768


def score_ordinals(state, logical):
    """Invert the actual native full-coverage affine permutation for observation.

    Mirrors the native kernel's inverse, not an identity-order assumption or a
    new sampling rule. This diagnostic never advances the initial sampler.
    """
    require(state.n_active == state.n_bits and state.cursor == state.coverage_id == 0,
            "Only the native initial full-coverage sampler is supported.")
    return ((logical - state.offset) * state.inverse).remainder(state.n_bits)


def flip_grouped(packed, candidate):
    require(packed.dtype == torch.uint8 and packed.ndim == 3 and packed.shape[1:] == (2, 4)
            and 0 <= candidate < 64, "Expected a two-stage K32 packed chunk and a single candidate.")
    stage, bit = divmod(candidate, 32)
    value = packed.clone()
    value[:, stage, bit // 8].bitwise_xor_(1 << (bit % 8))
    return value


@torch.no_grad()
def packed_decode(layer, packed):
    require(getattr(layer, "_sparse_bit_binding", None) is None,
            "Arbitrary candidate chunks must never inherit a full-bank score binding.")
    stages = layer._decode_packed_u8_with_decoder(layer._parallel_stage_decoder, packed,
        logical_shape=(len(packed), 2, 32), activation_dtype=torch.bfloat16, sparse_bit_grouped=True)
    require(stages is not None and stages.shape == (len(packed), 2, 32)
            and stages.dtype == torch.bfloat16, "Native packed two-stage decode unavailable.")
    return stages.sum(dim=1)  # Preserve the native BF16 sum before FP32 residuals.


def block_sse(value, target):
    require(value.dtype == torch.bfloat16 and target.dtype == torch.float32 and value.shape == target.shape,
            "Candidate scoring requires native BF16 output and original FP32 targets.")
    return (value.float() - target).double().square().sum(dim=1)


def native_predictions(layer, target):
    """One real complete-matrix backward; no parameter or optimizer updates."""
    device = next(layer.parameters()).device
    layer.enable_sparse_bit_decode_graph(parallel_stage_decode=True)
    layer._decoder_compute_dtype = torch.bfloat16
    layer.training = True
    layer._parallel_stage_decoder.eval()
    layer._build_parallel_stage_decode_plan()
    root = nn.Module()
    manager = SparseBitTuningManager(root_model=root, targets=[("layer", layer)],
        target_devices={"layer": device}, training_seed=31, config=BIT_CONFIG,
        streaming=False, calibration_dtype=torch.bfloat16)
    manager.configure_schedule(total_optimizer_steps=500)
    manager.initialize_scores()
    try:
        with torch.autocast("cuda", enabled=False):
            weight = layer._decode_weight(torch.bfloat16)
            denominator = error_stats(weight.detach(), target)["target_second_moment"]
            raw = F.mse_loss(weight.float(), target.to(device))
            loss = raw / denominator
        loss.backward()
        require(all(p.grad is None and not p.requires_grad for p in layer.parameters()),
                "A frozen decoder parameter received a gradient.")
        grouped = layer._get_parallel_stage_grouped_vq_packed(device=device)
        predicted = torch.empty((len(grouped), 2, 32), device=device, dtype=torch.float32)
        banks = []
        for spec in sorted(manager.bank_specs, key=lambda value: value.stage_idx):
            state = manager.sampler_states[spec.canonical_key]
            score = manager.score_module.score_view(spec)
            gradient = manager.score_module.score_chunks[spec.chunk_id].grad
            require(gradient is not None and gradient.dtype == torch.float32,
                    "Real native FP32 score gradients are required.")
            gradient = gradient[spec.score_start:spec.score_end]
            require(bool(torch.isfinite(gradient).all()), "Nonfinite native score gradient.")
            scale = manager.score_module.coordinate_scales[spec.canonical_key]
            for start in range(0, len(grouped), CHUNK):
                end = min(start + CHUNK, len(grouped))
                logical = torch.arange(start * 32, end * 32, device=device, dtype=torch.int64)
                ordinal = score_ordinals(state, logical)
                bit = logical.remainder(32)
                hard = ((grouped[logical // 32, spec.stage_idx, bit // 8].long() >> bit.remainder(8)) & 1)
                require(torch.equal(score[ordinal] >= 0, hard.bool()),
                        "Native inverse mapping disagrees with actual packed score signs.")
                predicted[start:end, spec.stage_idx] = (gradient[ordinal] * scale * (1 - 2 * hard)).view(-1, 32)
            banks.append({"stage": spec.stage_idx, "scale": scale, "sampler": state.to_metadata(),
                "gradient_l2": float(gradient.norm()), "gradient_nonzero_fraction": float(torch.count_nonzero(gradient)) / gradient.numel(),
                "prediction": "native dL/dp * scale * (1-2*original_bit)",
                "all_initial_score_signs_match_packed": True})
        report = {"raw_mse_fp32": float(raw), "nmse_fp32": float(loss),
            "target_second_moment": denominator, "banks": banks,
            "no_optimizer_step": True, "no_decoder_gradients": True}
        original = weight.detach().cpu()
        del weight, loss, raw
        return predicted, original, report
    finally:
        manager.detach_runtime()
        layer._build_parallel_stage_decode_plan()


def probe(layer, target, progress):
    device = next(layer.parameters()).device
    require(device.type == "cuda" and target.dtype == torch.float32 and target.ndim == 2
            and not layer.transpose, "This experiment covers the complete real gate matrices only.")
    started = time.perf_counter()
    original = {name: value.detach().cpu().clone() for name, value in layer.state_dict().items()}
    before_schema, before_bits = schema(layer), storage_bits(layer)
    before_deployment = decode(layer, deployment=True).cpu()
    for parameter in layer._parallel_stage_decoder.parameters():
        parameter.data = parameter.data.float()
    predicted, before_full_packed, derivative = native_predictions(layer, target)
    layer.eval()
    layer._decoder_compute_dtype = torch.bfloat16
    grouped = layer._get_parallel_stage_grouped_vq_packed(device=device)
    targets = target.to(device).view(-1, 32)
    require(grouped.shape == (len(targets), 2, 4), "Block order changed.")
    # The original bank remains immutable until every independent block has been scored.
    selected_packed = grouped.clone()
    selected = torch.zeros(len(targets), device=device, dtype=torch.uint8)
    before_chunk = torch.empty_like(targets, dtype=torch.bfloat16)
    after_chunk = torch.empty_like(before_chunk)
    counts = torch.zeros((2, 3, 3), device=device, dtype=torch.int64)
    choice_hist = torch.zeros(65, device=device, dtype=torch.int64)
    totals = {name: 0 for name in ("beneficial_blocks", "selected_ste_negative", "selected_ste_zero",
        "selected_ste_positive", "same_best_choice", "ste_choice_beneficial", "ste_choice_worsens", "ste_choice_unchanged")}
    chunk_reports = []
    for start in range(0, len(targets), CHUNK):
        end = min(start + CHUNK, len(targets))
        packed, target_chunk = grouped[start:end], targets[start:end]
        baseline = packed_decode(layer, packed)
        before_chunk[start:end] = baseline
        initial_sse = block_sse(baseline, target_chunk)
        best_sse, best_output = initial_sse.clone(), baseline.clone()
        choices = torch.zeros(len(packed), device=device, dtype=torch.long)
        p = predicted[start:end].reshape(len(packed), 64)
        ste_value, ste_index = p.min(dim=1)
        ste_choice = torch.where(ste_value < 0, ste_index + 1, 0)
        ste_actual_delta = torch.zeros_like(initial_sse)
        for candidate in range(64):
            values = packed_decode(layer, flip_grouped(packed, candidate))
            scores = block_sse(values, target_chunk)
            require(bool(torch.isfinite(scores).all()), "Nonfinite true candidate score.")
            delta = scores - initial_sse
            signs = (delta.sign().long() + 1) * 3 + p[:, candidate].sign().long() + 1
            counts[candidate // 32] += torch.bincount(signs, minlength=9).reshape(3, 3)
            ste_actual_delta = torch.where(ste_choice == candidate + 1, delta, ste_actual_delta)
            improved = scores < best_sse  # The current code and earlier candidates win exact ties.
            choices = torch.where(improved, candidate + 1, choices)
            best_sse = torch.where(improved, scores, best_sse)
            best_output = torch.where(improved[:, None], values, best_output)
        beneficial = choices > 0
        chosen_rows = torch.arange(len(packed), device=device)[beneficial]
        bit = choices[beneficial] - 1
        stage = bit // 32
        latent = bit % 32
        updated = selected_packed[start:end]
        updated[chosen_rows, stage, latent // 8] ^= (1 << (latent % 8)).to(torch.uint8)
        selected_decode = packed_decode(layer, updated)
        require(torch.equal(selected_decode, best_output),
                "Mixed chosen codes differ from individually scored candidates at the same chunk shape.")
        require(bool((best_sse <= initial_sse).all()) and bool((best_sse[beneficial] < initial_sse[beneficial]).all()),
                "A committed block did not strictly improve its same-path objective.")
        after_chunk[start:end], selected[start:end] = selected_decode, choices.to(torch.uint8)
        chosen_prediction = p.gather(1, (choices - 1).clamp_min(0)[:, None]).squeeze(1)[beneficial]
        totals["beneficial_blocks"] += int(beneficial.sum())
        for sign, label in ((-1, "negative"), (0, "zero"), (1, "positive")):
            totals[f"selected_ste_{label}"] += int((chosen_prediction.sign() == sign).sum())
        totals["same_best_choice"] += int((choices == ste_choice).sum())
        for sign, label in ((-1, "beneficial"), (0, "unchanged"), (1, "worsens")):
            totals[f"ste_choice_{label}"] += int((ste_actual_delta.sign() == sign).sum())
        choice_hist += torch.bincount(choices, minlength=65)
        item = {"start": start, "end": end, "before_sse": float(initial_sse.sum()),
                "after_sse": float(best_sse.sum()), "beneficial_blocks": int(beneficial.sum())}
        chunk_reports.append(item)
        if start == 0 or end == len(targets) or len(chunk_reports) % 10 == 0:
            progress(item)
    # Persistent writes happen once; rebuild grouped storage rather than only clearing a cache.
    for stage in range(2):
        layer.get_stage_part_vq_storage(stage, 0).copy_(selected_packed[:, stage:stage + 1])
    layer._build_parallel_stage_decode_plan()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        after_full_packed = layer._decode_weight(torch.bfloat16).cpu()
    for parameter in layer._parallel_stage_decoder.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    layer.disable_trainable_decode()
    layer._build_parallel_stage_decode_plan()
    after_deployment = decode(layer, deployment=True).cpu()
    final = layer.state_dict()
    require(schema(layer) == before_schema and storage_bits(layer) == before_bits, "Persistent storage changed.")
    require(all(torch.equal(value, final[name].cpu()) for name, value in original.items()
                if name not in {"vq_weight", "vq_weight_s1"}), "Frozen decoder or non-code state changed.")
    xor = [popcounts(original[name].to(device), final[name]) for name in ("vq_weight", "vq_weight_s1")]
    require(sum(xor) == totals["beneficial_blocks"] == int((selected > 0).sum()),
            "One-bit-per-beneficial-block commit or XOR count changed.")
    before_chunk, after_chunk = before_chunk.view_as(target).cpu(), after_chunk.view_as(target).cpu()
    report = {"status": "completed", "blocks": len(targets), "candidate_count_per_block": 64,
        "chunk_blocks": CHUNK, "passes": 1, "derivative_observation": derivative,
        "selection": {"objective": "FP64 block SSE of native BF16 sum minus original FP32 target (residual computed FP32)",
            "ties": "Keep unchanged if no strict improvement; earlier candidate wins exact ties.",
            "all_committed_blocks_strictly_improved": True, "same_chunk_selected_decode_exact": True,
            "totals": totals, "choice_histogram": choice_hist.cpu().tolist(),
            "stage_xor": xor, "candidate_sign_counts": counts.cpu().tolist(),
            "sign_count_axes": ["stage", "actual_delta_sign(-,0,+)", "STE_prediction_sign(-,0,+)"],
            "chunks": chunk_reports},
        "path_metrics": {"chunk_before": error_stats(before_chunk, target), "chunk_after": error_stats(after_chunk, target),
            "full_packed_before": error_stats(before_full_packed, target), "full_packed_after": error_stats(after_full_packed, target),
            "deployment_before": error_stats(before_deployment, target), "deployment_after": error_stats(after_deployment, target),
            "chunk_vs_full_before": error_stats(before_chunk, before_full_packed),
            "chunk_vs_full_after": error_stats(after_chunk, after_full_packed),
            "full_vs_deploy_before": error_stats(before_full_packed, before_deployment),
            "full_vs_deploy_after": error_stats(after_full_packed, after_deployment)},
        "frozen_state_exact": True, "persistent_bits_before": before_bits, "persistent_bits_after": storage_bits(layer),
        "initial_payload": {name: tensor_record(value) for name, value in original.items()},
        "final_payload": {name: tensor_record(value) for name, value in final.items()},
        "selection_sha256": tensor_record(selected)["sha256"], "elapsed_seconds": time.perf_counter() - started}
    return before_deployment, after_deployment, selected.cpu(), report
