"""Decompose real single-bit SSE changes without updating codes or decoders."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import logging
import os
from pathlib import Path
import time

import torch

from experiments.exact_bit_neighbor_probe.core import (
    CHUNK, block_sse, flip_grouped, native_predictions, packed_decode,
)
from experiments.joint_reconstruction_probe.core import rebuild, require
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def terms(before, after, target):
    """Exact finite displacement identity, evaluated in FP64 from native outputs."""
    residual = before.double() - target.double()
    displacement = after.double() - before.double()
    cross = 2 * (residual * displacement).sum(dim=-1)
    energy = displacement.square().sum(dim=-1)
    left = residual.square().sum(dim=-1)
    right = (after.double() - target.double()).square().sum(dim=-1)
    actual = right - left
    bound = 64 * torch.finfo(torch.float64).eps * (left.abs() + right.abs() + cross.abs() + energy)
    require(bool(((actual - cross - energy).abs() <= bound).all()),
            "FP64 finite-displacement identity exceeded its rounding bound.")
    return cross, energy, actual


def prepare(prior_root, output):
    require(not output.exists(), "Never replace an existing probe directory.")
    prior_path, report_path = prior_root / "preparation.json", prior_root / "run/report.json"
    prior = json.loads(prior_path.read_text())
    report = json.loads(report_path.read_text())
    review = json.loads((prior_root / "review.json").read_text())
    require(report["status"] == "completed" and review["status"] == "verified"
            and review["report_sha256"] == sha256(report_path)
            and review["preparation_sha256"] == sha256(prior_path), "Verified exact-neighbor source required.")
    check_hashes(prior["source_sha256"])
    check_hashes(prior["input_sha256"])
    sources = dict(prior["source_sha256"])
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    inputs = {str(prior_root / name): sha256(prior_root / name)
              for name in ("preparation.json", "run/report.json", "review.json")}
    models = {}
    for model, item in prior["models"].items():
        path = Path(item["initial_payload_file"])
        require(sha256(path) == item["initial_payload_sha256"], "Original native payload changed.")
        diagnostic = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        layer = rebuild(diagnostic, "gate_proj")
        state = {k: tensor_record(v) for k, v in layer.state_dict().items()}
        require(state == report["cases"][model]["local"]["initial_payload"], "Original packed state differs.")
        inputs[str(path)] = sha256(path)
        models[model] = {"initial_payload_file": str(path), "initial_state": state,
                         "prior_local": report["cases"][model]["local"]}
    # Tiny deterministic arithmetic check, not a substitute for native CUDA observations.
    before = torch.tensor([[1., 2., -1.]], dtype=torch.bfloat16)
    after = torch.tensor([[1.25, 1.5, -.75]], dtype=torch.bfloat16)
    target = torch.tensor([[1.5, 1.25, -.5]], dtype=torch.float32)
    cross, energy, delta = terms(before, after, target)
    require(torch.equal(cross + energy, delta), "Finite displacement CPU identity failed.")
    ast.parse(Path(__file__).read_text())
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    output.mkdir(parents=True)
    record = {"status": "cpu_ready", "source_sha256": sources, "input_sha256": inputs,
              "models": models, "output_root": str(output), "prior_root": str(prior_root),
              "budget_seconds": 900, "cuda_initialized": False,
              "scope": "All original 64 neighbors; no code/decoder updates or activation data."}
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": "cpu_ready", "sha256": sha256(output / "preparation.json"),
                      "source_count": len(sources), "input_count": len(inputs)}))


@torch.no_grad()
def scan(layer, target, predicted, prior_local, deadline):
    grouped = layer._get_parallel_stage_grouped_vq_packed(device=target.device)
    blocks = target.view(-1, 32)
    predicted = predicted.reshape(len(blocks), 64)
    require(grouped.shape == (len(blocks), 2, 4), "Gate packed geometry changed.")
    # scope(all candidates / most-negative STE per block), stage, actual sign, cross sign.
    negative_counts = torch.zeros((2, 2, 3, 3), dtype=torch.int64, device=target.device)
    legacy_counts = torch.zeros((2, 3, 3), dtype=torch.int64, device=target.device)
    sign_disagreements = 0
    sum_cross = torch.zeros((2, 2), dtype=torch.float64, device=target.device)
    sum_energy, sum_actual = torch.zeros_like(sum_cross), torch.zeros_like(sum_cross)
    max_identity_error = 0.
    for start in range(0, len(blocks), CHUNK):
        require(time.time() < deadline, "Assigned diagnostic budget expired; no retry.")
        end = min(start + CHUNK, len(blocks))
        packed, truth = grouped[start:end], blocks[start:end]
        original = packed_decode(layer, packed)
        sse = block_sse(original, truth)
        p = predicted[start:end]
        best = p.argmin(dim=1)
        for candidate in range(64):
            changed = packed_decode(layer, flip_grouped(packed, candidate))
            cross, energy, delta = terms(original, changed, truth)
            legacy_delta = block_sse(changed, truth) - sse
            stage = candidate // 32
            codes = (legacy_delta.sign().long() + 1) * 3 + p[:, candidate].sign().long() + 1
            legacy_counts[stage] += torch.bincount(codes, minlength=9).reshape(3, 3)
            sign_disagreements += int((legacy_delta.sign() != delta.sign()).sum())
            max_identity_error = max(max_identity_error, float((delta - cross - energy).abs().max()))
            negative = p[:, candidate] < 0
            for scope, mask in enumerate((negative, negative & (best == candidate))):
                codes = (delta[mask].sign().long() + 1) * 3 + cross[mask].sign().long() + 1
                negative_counts[scope, stage] += torch.bincount(codes, minlength=9).reshape(3, 3)
                sum_cross[scope, stage] += cross[mask].sum()
                sum_energy[scope, stage] += energy[mask].sum()
                sum_actual[scope, stage] += delta[mask].sum()
        if start == 0 or end == len(blocks):
            logging.info("finite displacement blocks %s/%s", end, len(blocks))
    require(legacy_counts.cpu().tolist() == prior_local["selection"]["candidate_sign_counts"],
            "All original actual/STE sign counts must reproduce the prior native experiment.")
    return {"negative_ste_counts": negative_counts.cpu().tolist(),
            "axes": ["all_negative_candidates / each_block_most_negative", "stage",
                     "actual_FP64_delta_sign(-,0,+)", "real_cross_term_sign(-,0,+)"],
            "negative_ste_sum_cross": sum_cross.cpu().tolist(),
            "negative_ste_sum_energy": sum_energy.cpu().tolist(),
            "negative_ste_sum_actual": sum_actual.cpu().tolist(),
            "legacy_FP32_residual_sign_counts_exact": True,
            "FP64_vs_legacy_sign_disagreements": sign_disagreements,
            "max_FP64_identity_absolute_error": max_identity_error,
            "all_FP64_identity_bounds_passed": True}


def run(preparation_path):
    preparation = json.loads(preparation_path.read_text())
    require(preparation["status"] == "cpu_ready", "Verified CPU preparation required.")
    check_hashes(preparation["source_sha256"])
    check_hashes(preparation["input_sha256"])
    output = Path(preparation["output_root"]) / "run"
    require(not output.exists(), "Never overwrite a diagnostic run.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One assigned GPU required.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output.mkdir()
    started = time.time()
    report = {"status": "running", "started_unix": started, "pid": os.getpid(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "preparation_sha256": sha256(preparation_path), "cases": {},
              "interpretation_boundary": "Cross uses the actual finite decoded displacement, not a decoder Jacobian. "
                  "A changed cross sign cannot distinguish decoder nonlinearity from native BF16 gradient rounding. "
                  "This is descriptive; no code, decoder or optimizer update and no production integration."}
    write_json(output / "report.json", report)
    try:
        for model, item in preparation["models"].items():
            diagnostic = torch.load(item["initial_payload_file"], weights_only=True, map_location="cpu", mmap=True)
            layer = rebuild(diagnostic, "gate_proj").to(device)
            require({k: tensor_record(v) for k, v in layer.state_dict().items()} == item["initial_state"],
                    "Original native state changed before observation.")
            torch.cuda.reset_peak_memory_stats(device)
            for parameter in layer._parallel_stage_decoder.parameters():
                parameter.data = parameter.data.float()
            prediction, full_weight, derivative = native_predictions(layer, diagnostic["rotated_weight"])
            require(derivative == item["prior_local"]["derivative_observation"],
                    "Native full-matrix derivative observation differs from the original experiment.")
            layer.eval()
            observed = scan(layer, diagnostic["rotated_weight"].to(device), prediction,
                            item["prior_local"], started + preparation["budget_seconds"])
            for parameter in layer._parallel_stage_decoder.parameters():
                parameter.data = parameter.data.to(torch.bfloat16)
            layer.disable_trainable_decode()
            require({k: tensor_record(v) for k, v in layer.state_dict().items()} == item["initial_state"],
                    "Read-only diagnostic changed persistent state.")
            report["cases"][model] = {**observed, "all_persistent_state_unchanged": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            write_json(output / "report.json", report)
            del layer, diagnostic, prediction, full_weight
            gc.collect()
            torch.cuda.empty_cache()
        check_hashes(preparation["source_sha256"])
        check_hashes(preparation["input_sha256"])
        report.update(status="completed", finished_unix=time.time())
        write_json(output / "report.json", report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        write_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-from", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preparation", type=Path)
    args = parser.parse_args()
    if args.prepare_from is not None:
        require(args.output is not None and args.preparation is None, "Preparation requires only --prepare-from/--output.")
        prepare(args.prepare_from.resolve(), args.output.resolve())
    else:
        require(args.preparation is not None and args.output is None, "Run requires only --preparation.")
        run(args.preparation.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
