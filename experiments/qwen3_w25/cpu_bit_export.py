"""Read current hard codes from an exact sidecar without restoring training state.

``checkpoint_packed_snapshot`` has already projected the live score signs using
the current (not pending-next) sampler. Source chunk slices are only used to
verify this identity; export never recreates their original device placement.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import torch

from sparse_bit_tuning.config import SparseBitTuningConfig, active_count, resolve_round_steps, resolve_stable_steps
from sparse_bit_tuning.exact_checkpoint import EXACT_FORMAT, EXACT_VERSION, exact_state_path
from sparse_bit_tuning.sampler import AffineSamplerState


_VERIFY_BLOCK = 1 << 20


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, *, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid SparseBit {name}: {value!r}.")
    return value


def _inventory(selected, ratio):
    banks, modules = {}, {}
    for path, module in selected:
        _require(path not in modules, f"Duplicate SparseBit target: {path}.")
        _require(getattr(module, "_sparse_bit_binding", None) is None, f"{path}: export requires detached SparseBit runtime.")
        modules[path] = module
        for stage in range(int(module.residual_stages)):
            for part in range(int(module.parallel_parts)):
                key = f"{path}|stage={stage}|part={part}"
                shape = tuple(module.get_stage_part_vq_spec(stage, part)["logical_shape"])
                target = module.get_stage_part_vq_storage(stage, part)
                _require(len(shape) == 3 and shape[1] == 1 and shape[0] > 0 and shape[2] > 0,
                         f"{key}: invalid logical shape {shape}.")
                _require(target.dtype == torch.uint8 and tuple(target.shape) == (shape[0], 1, (shape[2] + 7) // 8),
                         f"{key}: target must use canonical packed uint8 storage.")
                n_bits = int(shape[0] * shape[2])
                banks[key] = {"module_path": path, "stage_idx": stage, "part_idx": part,
                              "n_bits": n_bits, "n_active": active_count(n_bits, ratio),
                              "logical_in": int(shape[2]), "target": target}
    _require(bool(banks), "SparseBit export requires at least one bank.")
    return banks, modules


def _validate_layout(state, banks, coordinates):
    raw = state.get("banks")
    scores = state.get("score_chunks")
    _require(isinstance(raw, list) and isinstance(scores, (list, tuple)) and bool(scores),
             "SparseBit banks/score_chunks are missing.")
    by_key, intervals, module_chunks = {}, defaultdict(list), defaultdict(set)
    for row in raw:
        _require(isinstance(row, dict), "SparseBit bank entry must be an object.")
        key = row.get("canonical_key")
        _require(key in banks and key not in by_key, f"Unknown or duplicate SparseBit bank: {key}.")
        bank = banks[key]
        for field in ("module_path", "stage_idx", "part_idx", "n_bits", "n_active"):
            _require(row.get(field) == bank[field], f"{key}: bank {field} mismatch.")
        chunk = _integer(row.get("chunk_id"), f"{key} chunk_id")
        start = _integer(row.get("score_start"), f"{key} score_start")
        end = _integer(row.get("score_end"), f"{key} score_end", minimum=1)
        _require(chunk < len(scores) and end - start == bank["n_active"], f"{key}: invalid score slice.")
        by_key[key] = row
        intervals[chunk].append((start, end))
        module_chunks[bank["module_path"]].add(chunk)
    _require(set(by_key) == set(banks), "SparseBit bank set mismatch.")
    _require(set(intervals) == set(range(len(scores))), "SparseBit score chunks contain an unused chunk.")
    _require(all(len(ids) == 1 for ids in module_chunks.values()), "SparseBit module spans score chunks.")
    dtype = torch.float32 if coordinates == "decoder_sensitivity" else torch.float16
    for chunk, score in enumerate(scores):
        _require(torch.is_tensor(score) and score.device.type == "cpu" and score.dtype == dtype
                 and score.ndim == 1 and score.is_contiguous(), f"Invalid SparseBit score chunk {chunk}.")
        cursor = 0
        for start, end in sorted(intervals[chunk]):
            _require(start == cursor, f"SparseBit chunk {chunk} has overlapping or missing score slices.")
            cursor = end
        _require(cursor == score.numel(), f"SparseBit score chunk {chunk} length mismatch.")
    return by_key, scores


def _validate_rounds(state, banks, config, seed, total_steps):
    rounds = resolve_round_steps(config.round_steps, total_optimizer_steps=total_steps, active_ratio=config.active_ratio)
    stable = resolve_stable_steps(rounds)
    _require(state.get("bit_round_steps") == rounds and state.get("stable_steps") == stable,
             "SparseBit saved round schedule disagrees with the training contract.")
    global_round = _integer(state.get("global_bit_round"), "global_bit_round")
    step = _integer(state.get("bit_round_step"), "bit_round_step")
    counter = _integer(state.get("stable_counter"), "stable_counter")
    flips = _integer(state.get("cumulative_flip_count"), "cumulative_flip_count")
    had_flip = state.get("had_flip")
    _require(type(had_flip) is bool and step < rounds and counter < stable,
             "Invalid SparseBit round counters.")
    _require((had_flip and flips > 0) or (not had_flip and flips == counter == 0),
             "SparseBit flip/stability counters disagree.")
    _require(step != 0 or (counter == flips == 0 and not had_flip), "SparseBit new-round counters must reset.")
    raw, pending = state.get("sampler_states"), state.get("pending_next_states")
    _require(isinstance(raw, dict) and set(raw) == set(banks) and isinstance(pending, dict)
             and set(pending).issubset(banks), "SparseBit sampler/pending bank set mismatch.")
    _require(type(state.get("streaming")) is bool, "SparseBit streaming flag must be bool.")
    if pending:
        _require(state["streaming"] and global_round > 0 and step == 0, "Invalid SparseBit pending round transition.")
        pending_modules = {banks[key]["module_path"] for key in pending}
        _require(all(key in pending for key, bank in banks.items() if bank["module_path"] in pending_modules),
                 "SparseBit partial pending module transition.")
    samplers = {}
    for key, bank in banks.items():
        sampler = AffineSamplerState.from_metadata(raw[key])
        ordinal = global_round - int(key in pending)
        rounds_per_coverage = math.ceil(bank["n_bits"] / bank["n_active"])
        expected = AffineSamplerState.create(canonical_key=key, training_seed=seed,
            n_bits=bank["n_bits"], n_active=bank["n_active"],
            coverage_id=ordinal // rounds_per_coverage, cursor=(ordinal % rounds_per_coverage) * bank["n_active"])
        _require(sampler == expected, f"{key}: sampler identity/coverage disagrees with the current round.")
        if key in pending:
            _require(AffineSamplerState.from_metadata(pending[key]) == sampler.advance(),
                     f"{key}: pending sampler is not the next subset.")
        samplers[key] = sampler
    return samplers


def _validate_optimizer_header(state, config, scores):
    # Adam moments are irrelevant to inference. Validate their saved descriptors
    # without faulting their potentially 128-GiB payload into resident memory.
    optimizer = state.get("bit_optimizer")
    _require(isinstance(optimizer, dict) and optimizer.get("format") == "sparse_bit_optimizer_state"
             and optimizer.get("version") == 1, "Invalid SparseBit optimizer state header.")
    for field, value in (("optimizer_name", config.optimizer), ("lr", config.resolved_lr()),
                         ("weight_decay", config.weight_decay)):
        _require(optimizer.get(field) == value, f"SparseBit optimizer {field} disagrees with the contract.")
    chunks = optimizer.get("chunks")
    _require(isinstance(chunks, dict) and set(chunks) == {str(i) for i in range(len(scores))},
             "SparseBit optimizer chunk set mismatch.")
    for i, score in enumerate(scores):
        values = chunks[str(i)]
        _require(isinstance(values, dict) and set(values) == {"exp_avg", "exp_avg_sq"},
                 f"Invalid SparseBit optimizer chunk {i}.")
        mean, variance = values["exp_avg"], values["exp_avg_sq"]
        _require((mean is None) == (variance is None), f"SparseBit chunk {i} contains partial Adam state.")
        if mean is not None:
            _require(config.optimizer != "rms_sgd", "RMS-SGD must not contain Adam moments.")
            for tensor in (mean, variance):
                _require(torch.is_tensor(tensor) and tensor.dtype == torch.float32 and tensor.device.type == "cpu"
                         and tensor.shape == score.shape, f"Invalid SparseBit Adam tensor in chunk {i}.")


def _verify_score_signs(packed, scores, sampler, logical_in, radius):
    """Bound temporary index memory independently of the active ratio/bank size."""
    subset = sampler.subset_meta()
    flat = packed.view(-1)
    radius = torch.tensor(radius, dtype=scores.dtype)
    for start in range(0, sampler.n_active, _VERIFY_BLOCK):
        end = min(start + _VERIFY_BLOCK, sampler.n_active)
        score = scores[start:end]
        _require(bool(torch.isfinite(score).all()) and bool((score.abs() <= radius).all()),
                 f"{sampler.canonical_key}: scores are nonfinite or outside their coordinate radius.")
        q = torch.arange(start, end, dtype=torch.int64)
        positions = q + sampler.cursor
        if subset.tail:
            filler = (subset.secondary_stride * (q - subset.remaining) + subset.secondary_offset) % sampler.cursor
            positions = torch.where(q < subset.remaining, positions, filler)
        logical = (sampler.stride * positions + sampler.offset) % sampler.n_bits
        latent = logical % logical_in
        byte = (logical // logical_in) * ((logical_in + 7) // 8) + latent // 8
        bits = ((flat[byte].to(torch.int64) >> (latent % 8)) & 1).bool()
        _require(torch.equal(bits, score >= 0),
                 f"{sampler.canonical_key}: packed snapshot disagrees with current score signs.")


@torch.no_grad()
def restore_packed_bits_cpu(selected, *, checkpoint, contract):
    """Validate and copy materialized current hard bits, without any optimizer."""
    path = Path(exact_state_path(str(checkpoint)))
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    _require(isinstance(payload, dict) and payload.get("format") == EXACT_FORMAT
             and payload.get("version") == EXACT_VERSION, "Unsupported SparseBit exact sidecar envelope.")
    state = payload.get("state")
    config = SparseBitTuningConfig(enabled=True, **contract["sparse_bit"]).normalized()
    coordinates = config.proxy_coordinates
    version = 2 if coordinates == "decoder_sensitivity" else 1
    _require(isinstance(state, dict) and state.get("format") == "sparse_bit_tuning_exact_state"
             and state.get("version") == version, "Unsupported SparseBit exact state version.")
    _require(state.get("proxy_coordinates", "unit") == coordinates, "SparseBit coordinate mode mismatch.")
    _require(state.get("training_seed") == contract["data"]["seed"]
             and state.get("bit_active_ratio") == config.active_ratio, "SparseBit seed/active ratio mismatch.")
    if contract["runtime"]["offload_mode"] == "none":
        _require(state.get("streaming") is False, "No-offload SparseBit checkpoint must be nonstreaming.")
    banks, modules = _inventory(selected, config.active_ratio)
    _require(state.get("target_modules") == sorted(modules), "SparseBit target module set mismatch.")
    rows, scores = _validate_layout(state, banks, coordinates)
    samplers = _validate_rounds(state, banks, config, contract["data"]["seed"], contract["optimization"]["steps"])
    _validate_optimizer_header(state, config, scores)
    scales = state.get("coordinate_scales", {})
    if coordinates == "decoder_sensitivity":
        _require(isinstance(scales, dict) and set(scales) == set(banks), "SparseBit coordinate scale bank set mismatch.")
        for key, scale in scales.items():
            _require(isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0,
                     f"{key}: invalid coordinate scale.")
            bounds = torch.tensor([scale / 2, 1 / scale], dtype=torch.float32)
            _require(bool((torch.isfinite(bounds) & (bounds > 0)).all()), f"{key}: coordinate scale is not representable.")
    else:
        _require("coordinate_scales" not in state, "Unit coordinates must not contain calibrated scales.")
    packed_banks = state.get("packed_banks")
    _require(isinstance(packed_banks, dict) and set(packed_banks) == set(banks), "SparseBit packed bank set mismatch.")
    for key, bank in banks.items():
        packed = packed_banks[key]
        _require(torch.is_tensor(packed) and packed.device.type == "cpu" and packed.dtype == torch.uint8
                 and packed.shape == bank["target"].shape and packed.is_contiguous(), f"{key}: invalid packed bank payload.")
        row = rows[key]
        score = scores[row["chunk_id"]][row["score_start"]:row["score_end"]]
        _verify_score_signs(packed, score, samplers[key], bank["logical_in"],
                            scales[key] / 2 if coordinates == "decoder_sensitivity" else 1.0)
    # Complete validation before mutating any target. Copies never include score
    # or optimizer tensors, and grouped packed views are refreshed afterwards.
    for key, bank in banks.items():
        bank["target"].copy_(packed_banks[key])
        _require(torch.equal(bank["target"].detach().cpu(), packed_banks[key]), f"{key}: packed copy mismatch.")
    for module in modules.values():
        if bool(getattr(module, "parallel_stage_decode", False)) and getattr(module, "_parallel_stage_decoder", None) is not None:
            module._build_parallel_stage_decode_plan()
        else:
            module.clear_decoded_weight_cache()
    return {"exact_sidecar_hard_codes_restored": True, "exact_training_state_restored": False,
            "internal_bit_optimizer_restored": False,
            "optimizer_steps_executed": 0, "packed_banks_compared": len(banks),
            "global_bit_round": state["global_bit_round"], "bit_round_step": state["bit_round_step"],
            "proxy_coordinates": coordinates, "coordinate_scales_recalibrated": False,
            "coordinate_scale_count": len(scales), "score_dtypes": sorted({str(score.dtype) for score in scores}),
            "sidecar_read_mode": "cpu_mmap", "source_score_chunks": len(scores),
            "active_score_signs_verified": sum(bank["n_active"] for bank in banks.values()),
            "score_validation_block_elements": _VERIFY_BLOCK, "source_chunk_layout_preserved": True}
