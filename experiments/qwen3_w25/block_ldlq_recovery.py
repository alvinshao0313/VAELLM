"""Apply the verified fixed-decoder BlockLDLQ pair to a contiguous down shard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.block_ldlq import (
    block_ldl, build_noncentered_hessian, prepare_protected_objective, quantize_reverse_blocks,
)
from experiments.qwen3_w25.block_ldlq_model import (
    CODE_KEYS, SOURCE_ID, NativeStageOracle, apply_codes, native_output_metrics, oracle_weight,
    read_codes, rotate_hessian, validate_layer,
)
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_recovery import BASE_MODEL, capture_shard_inputs
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


ARMS = ("feedback_off", "feedback_on")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def shard_targets(shard):
    if shard not in (0, 1):
        raise ValueError("Two contiguous shards, 0 or 1, are required.")
    return [f"model.layers.{index}.mlp.down_proj" for index in range(shard * 18, (shard + 1) * 18) if index != 27]


def validate_pilot(pilot, source):
    if (pilot["status"] != "verified" or not pilot["feedback_dev_improved"]
            or pilot["source_checkpoint_id"] != SOURCE_ID or Path(pilot["source"]) != source
            or pilot["target"] != "model.layers.27.mlp.down_proj"):
        raise ValueError("The approved source-matched L27 validation and dev improvement are required.")
    for arm in ARMS:
        if (pilot["arms"][arm]["export"]["fresh_tensors_compared"] != 3423
                or not pilot["arms"][arm]["export"]["bit_budget"]["budget_passed"]):
            raise ValueError("L27 persistence or budget validation is incomplete.")


def _native_parity(actual, expected):
    difference = actual.double() - expected.double()
    ratio = float(difference.norm() / expected.double().norm())
    if not bool(torch.isfinite(difference).all()) or ratio > 1e-6:
        raise RuntimeError("Stage oracle differs materially from native joint decoding.")
    return {"max_absolute": float(difference.abs().max()), "relative_rms": ratio}


@torch.no_grad()
def recover_layer(layer, reference, bundle, lengths, output, name):
    """Use exactly the pilot algorithm; return code patches, never fitted decoders."""
    validate_layer(layer)
    device = torch.device("cuda:0")
    before = {key: value.detach().cpu().clone() for key, value in layer.state_dict().items()}
    layer.to(device)
    initial = read_codes(layer)
    oracle = NativeStageOracle(layer)
    baseline_q = oracle_weight(oracle, initial)
    native_q = layer._decode_compressed_weight(dtype=torch.bfloat16)
    initial_parity = _native_parity(baseline_q, native_q)
    baseline = {split: native_output_metrics(layer, reference, bundle[split + "_inputs"], lengths[split])
                for split in lengths}
    hessian = build_noncentered_hessian(bundle["train_inputs"], device=device)
    p = layer.protected_input_indices
    mask = torch.ones(layer.in_features, device=device, dtype=torch.bool)
    mask[p] = False
    c = mask.nonzero().flatten()
    protected = layer._decode_protected_input_weight_rows(dtype=torch.bfloat16, device=device).T.double()
    objective = prepare_protected_objective(reference.to(device=device, dtype=torch.float64), protected, hessian, c, p)
    target = layer.weight_rotation(objective.target)
    rotated_hessian = rotate_hessian(objective.hessian, layer.weight_rotation)
    factors = block_ldl(rotated_hessian, block_size=16)
    damping = objective.damping
    del hessian, objective, protected
    e0 = target - baseline_q.double()
    baseline_proxy = float(((e0 @ rotated_hessian) * e0).sum() / 4096)
    del e0, baseline_q, native_q
    reports = {}
    for arm in ARMS:
        start = time.perf_counter()

        def progress(value):
            if value["visited_blocks"] % 128 == 0 or value["visited_blocks"] == 752:
                print(json.dumps({"target": name, "arm": arm, **value}), flush=True)

        result = quantize_reverse_blocks(target, factors, initial, oracle,
                                         feedback=arm == "feedback_on", progress=progress)
        apply_codes(layer, result.codes)
        native_q = layer._decode_compressed_weight(dtype=torch.bfloat16)
        parity = _native_parity(result.reconstructed_weight, native_q)
        error = target - native_q.double()
        proxy = float(((error @ rotated_hessian) * error).sum() / 4096)
        metrics = {split: native_output_metrics(layer, reference, bundle[split + "_inputs"], lengths[split])
                   for split in lengths}
        layer.to("cpu")
        state = layer.state_dict()
        frozen = {key: value for key, value in state.items() if key not in CODE_KEYS}
        _assert_state_equal(frozen, {key: value for key, value in before.items() if key not in CODE_KEYS},
                            f"{name}/{arm} non-code state")
        patch = {key: state[key].detach().clone() for key in CODE_KEYS}
        file = output / "patches" / f"{arm}.layer{name.split('.')[2]}.pt"
        torch.save(patch, file)
        reports[arm] = {"patch_file": str(file.relative_to(output)), "patch_sha256": _sha(file),
                        "statistics": result.statistics, "metrics": metrics, "baseline": baseline,
                        "proxy": proxy, "baseline_proxy": baseline_proxy, "damping": damping,
                        "initial_oracle_native_parity": initial_parity, "oracle_native_parity": parity,
                        "frozen_tensors_compared": len(frozen),
                        "elapsed_seconds": time.perf_counter() - start,
                        "dev_after_before_ratio": metrics["heldout"]["token_mean_mse"] / baseline["heldout"]["token_mean_mse"]}
        layer.to(device)
        apply_codes(layer, initial)
        del result, native_q, error, patch, frozen, state
    layer.to("cpu")
    _assert_state_equal(layer.state_dict(), before, f"{name} source restored after paired optimization")
    return reports


@torch.no_grad()
def run_worker(source, pilot_dir, output, shard):
    source, pilot_dir, output = [Path(path).resolve() for path in (source, pilot_dir, output)]
    if any(output.is_relative_to(path) for path in (source, pilot_dir)):
        raise ValueError("Worker output must be separate from immutable source and pilot.")
    pilot_path = pilot_dir / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    validate_pilot(pilot, source)
    targets = shard_targets(shard)
    output.mkdir(parents=True, exist_ok=False)
    (output / "patches").mkdir()
    root = Path(__file__).resolve().parents[2]
    pins = dict(pilot["code_sha256"])
    pins["experiments/qwen3_w25/block_ldlq_recovery.py"] = _sha(__file__)
    if any(_sha(root / name) != value for name, value in pins.items()):
        raise ValueError("The validated pilot code changed before expansion.")
    report = {"status": "running", "pid": os.getpid(), "source_checkpoint_id": SOURCE_ID,
              "source": str(source), "source_meta_sha256": _sha(source / "checkpoint_meta.json"),
              "pilot_report_sha256": _sha(pilot_path), "code_sha256": pins,
              "targets": targets, "shard": shard, "configuration": pilot["configuration"], "modules": {}}
    path = output / "worker.json"
    _write(path, report)
    started = time.perf_counter()
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        model, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True, expected_kind="final_model")
        if meta["checkpoint_id"] != SOURCE_ID or report["source_meta_sha256"] != pilot["source_meta_sha256"]:
            raise ValueError("Expansion input differs from the validated source.")
        model.eval().requires_grad_(False)
        for name in targets:
            validate_layer(model.get_submodule(name))
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train, dev, data = build_calibration_batches(tokenizer, seed=31, train_documents=512,
                                                   heldout_documents=64, max_length=256)
        if data != pilot["data"]:
            raise ValueError("Expansion changed the pilot's documents or tokenization.")
        report["data"] = data
        _write(path, report)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device="cuda:0", dtype=torch.bfloat16)
        phase = time.perf_counter()
        print(f"Capturing {len(targets)} down targets in one teacher prefix pass per document", flush=True)
        bundles = capture_shard_inputs(teacher, targets, train, dev)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        report["capture_seconds"] = time.perf_counter() - phase
        lengths = {split: [row["valid_tokens"] for row in data[split]["documents"]] for split in ("train", "heldout")}
        for name in targets:
            bundle = bundles.pop(name)
            report["modules"][name] = recover_layer(model.get_submodule(name), bundle["reference_weight"],
                                                     bundle, lengths, output, name)
            del bundle
            gc.collect()
            torch.cuda.empty_cache()
            _write(path, report)
        report["code_sha256_after"] = {name: _sha(root / name) for name in pins}
        if report["code_sha256_after"] != pins:
            raise RuntimeError("Worker code changed during execution.")
        report.update(status="verified", elapsed_seconds=time.perf_counter() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        _write(path, report)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        _write(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "pilot-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    run_worker(args.source, args.pilot_dir, args.output, args.shard)


if __name__ == "__main__":
    main()
