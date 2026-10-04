"""Independent packed-code-only recovery workers selected by a completed bit pilot."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.bit_pilot import (
    CODE_FILES as PILOT_CODE_FILES, SOURCE_CHECKPOINT_ID, TRAIN_DOCUMENTS,
    _json_digest, _trajectory, pilot_document_plan,
)
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_bits import (
    _changed_bits, _code_banks, _digest, calibrate_local_bits_cpu, optimize_local_bits,
)
from experiments.qwen3_w25.local_recovery import (
    BASE_MODEL, capture_shard_inputs, shard_targets, validate_recovery_source,
)
from experiments.qwen3_w25.pilot_local_decoder import TARGETS as PILOT_TARGETS, _write_report
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


PROTOCOL_VERSION = 1
CODE_FILES = (*PILOT_CODE_FILES, "experiments/qwen3_w25/local_bit_recovery.py",
              "experiments/qwen3_w25/recover_local_bits.py")


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _packed_sha256(tensor):
    if tensor.dtype != torch.uint8:
        raise ValueError("Packed-code digest requires uint8 storage.")
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def bit_recovery_configuration(*, train_documents, num_shards):
    shard_targets(0, num_shards)
    if type(train_documents) is not int or train_documents not in TRAIN_DOCUMENTS:
        raise ValueError("Training documents must be the pilot-selected 32, 128 or 512.")
    return {
        "steps": 128, "round_steps": 128, "seed": 31,
        "maximum_train_documents": 512, "train_documents": train_documents,
        "heldout_documents": 64, "selection_documents": 32, "confirmation_documents": 32,
        "max_length": 256, "chunk_size": 256, "num_shards": num_shards,
        "objective": "full_linear_output_mse", "optimizer": "adamw", "weight_decay": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8, "proxy_coordinates": "decoder_sensitivity",
        "active_ratio": 1.0, "streaming": False, "decoder_compute_dtype": "torch.float32",
        "native_activation_dtype": "torch.bfloat16",
        "lr_rule": "per-target initial CPU max(decoder_sensitivity scale)/64",
    }


def validate_bit_recovery_source(model, meta):
    validate_recovery_source(model, meta)
    if meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID:
        raise ValueError("Code recovery requires the original down3 checkpoint identity.")
    for target in meta["compressed_targets"]:
        layer = model.get_submodule(target)
        if layer.bias is not None:
            raise ValueError(f"The fixed original bias-free projection is required: {target}.")
        _code_banks(layer)


def load_pilot_selection(pilot_json, source_checkpoint_dir, source_meta):
    """Read the completed real pilot, checking provenance and selection without confirmation scores."""
    path, source = Path(pilot_json).resolve(), Path(source_checkpoint_dir).resolve()
    raw = path.read_bytes()
    pilot = json.loads(raw)
    if pilot.get("status") != "complete":
        raise ValueError("Packed-code recovery requires a completed pilot; running/failed reports cannot select a recipe.")
    if source_meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID or pilot.get("source_checkpoint_id") != SOURCE_CHECKPOINT_ID:
        raise ValueError("Pilot and worker must use the fixed original down3 source checkpoint.")
    cfg = pilot["configuration"]
    if Path(cfg["checkpoint_dir"]).resolve() != source or Path(cfg["base_model_path"]).resolve() != Path(BASE_MODEL).resolve():
        raise ValueError("Pilot source/teacher paths differ from the worker source.")
    expected = {
        "targets": list(PILOT_TARGETS), "train_document_arms": list(TRAIN_DOCUMENTS),
        "maximum_train_documents": 512, "heldout_documents": 64,
        "selection_documents": 32, "confirmation_documents": 32,
        "seed": 31, "max_length": 256, "chunk_size": 256, "steps": 128, "round_steps": 128,
        "early_stable_round_end": True, "optimizer": "adamw", "weight_decay": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8, "proxy_coordinates": "decoder_sensitivity",
        "active_ratio": 1.0, "streaming": False, "decoder_frozen": True,
        "decoder_compute_dtype": "torch.float32", "native_activation_dtype": "torch.bfloat16",
        "confirmation_used_for_selection": False, "checkpoint_written": False,
        "lr_rule": "per-target initial CPU max(decoder_sensitivity scale)/64, constant across arms",
        "allow_tf32": False, "deterministic_algorithms": True,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Pilot protocol differs at {key}.")
    repo = Path(__file__).resolve().parents[2]
    for relative in PILOT_CODE_FILES:
        if pilot["code_sha256"].get(relative) != _file_sha256(repo / relative):
            raise ValueError(f"Pilot dependency changed: {relative}.")
    if (set(pilot["targets"]) != set(PILOT_TARGETS)
            or not pilot["source_bit_budget"]["budget_passed"]
            or pilot["document_plan"]["manifest_sha256"] != _json_digest(pilot["data"])):
        raise ValueError("Pilot target coverage, source budget or data manifest identity is invalid.")
    scores = {}
    for count in TRAIN_DOCUMENTS:
        ratios = []
        for target in PILOT_TARGETS:
            arm = pilot["targets"][target]["arms"][str(count)]
            if arm["train_documents"] != count or not arm["source_state_restored_before_arm"]:
                raise ValueError(f"Pilot arm does not describe a fresh {count}-document run: {target}.")
            comparison = arm["native_selection"]
            before, after = comparison["before"]["token_mean_mse"], comparison["after"]["token_mean_mse"]
            if not math.isfinite(before) or not math.isfinite(after) or before <= 0 or after < 0:
                raise ValueError("Pilot selection metrics must be finite with positive baselines.")
            ratio = after / before
            if comparison["after_before_ratio"] != ratio:
                raise ValueError("Pilot selection ratio disagrees with its native MSE values.")
            ratios.append(ratio)
        scores[count] = sum(ratios) / len(PILOT_TARGETS)
        if pilot["selection_scores"][str(count)]["mean_selection_ratio"] != scores[count]:
            raise ValueError("Pilot aggregate selection score is inconsistent.")
    selected = min(TRAIN_DOCUMENTS, key=lambda count: (scores[count], count))
    if type(pilot["selected_train_documents"]) is not int or pilot["selected_train_documents"] != selected:
        raise ValueError("Pilot selected recipe does not match the fixed selection-only rule.")
    return pilot, hashlib.sha256(raw).hexdigest(), selected


def save_code_patch(layer, source_codes, path):
    """Write only authoritative uint8 banks after verifying actual logical bit changes."""
    banks = _code_banks(layer)
    if set(source_codes) != {name for name, _ in banks}:
        raise ValueError("Source code bank inventory differs from the live module.")
    patch, changes = {}, []
    for name, shape in banks:
        original = source_codes[name]
        current = layer.state_dict()[name].detach().cpu().clone()
        if original.dtype != torch.uint8 or current.dtype != torch.uint8 or current.shape != original.shape:
            raise ValueError(f"Packed patch dtype/shape changed: {name}.")
        changes.append(_changed_bits(original, current, shape))
        patch[name] = current
    if sum(changes) == 0:
        raise ValueError("Packed patch must contain an actual logical bit change.")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(patch, path)
    return {
        "stage_changed_bits": changes,
        "stage_code_sha256_before": [_packed_sha256(source_codes[name]) for name, _ in banks],
        "stage_code_sha256_after": [_packed_sha256(patch[name]) for name, _ in banks],
        "code_schema": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in patch.items()},
    }


def run_local_bit_recovery_worker(checkpoint_dir, pilot_json, output_dir, *, shard_index, num_shards):
    targets = shard_targets(shard_index, num_shards)
    source, pilot_path, output = Path(checkpoint_dir).resolve(), Path(pilot_json).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents:
        raise ValueError("Worker output must be outside the immutable source checkpoint.")
    meta, source_audit = _validate_checkpoint(source)
    if not source_audit["budget_passed"]:
        raise ValueError("Worker source exceeds the no-LoRA bit budget.")
    pilot, pilot_sha, selected = load_pilot_selection(pilot_path, source, meta)
    configuration = bit_recovery_configuration(train_documents=selected, num_shards=num_shards)
    configuration_sha = _json_digest(configuration)
    output.mkdir(parents=True, exist_ok=False)
    (output / "patches").mkdir()
    (output / "reports").mkdir()
    report_path, device = output / "worker.json", torch.device("cuda:0")
    repo = Path(__file__).resolve().parents[2]
    started = time.perf_counter()
    report = {
        "protocol_version": PROTOCOL_VERSION, "status": "preparing",
        "source_checkpoint_dir": str(source), "source_checkpoint_id": meta["checkpoint_id"],
        "pilot_json": str(pilot_path), "pilot_sha256": pilot_sha, "selected_train_documents": selected,
        "configuration": configuration, "configuration_sha256": configuration_sha,
        "code_sha256": {relative: _file_sha256(repo / relative) for relative in CODE_FILES},
        "shard_index": shard_index, "targets": list(targets), "modules": {}, "source_bit_budget": source_audit,
    }
    _write_report(report_path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_batch, heldout_batch, data = build_calibration_batches(
            tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
        plan = pilot_document_plan(train_batch, heldout_batch, data)
        if data != pilot["data"] or plan != pilot["document_plan"]:
            raise ValueError("Rebuilt 512+64 documents differ from the completed pilot manifest.")
        report.update(data=data, document_plan=plan, manifest_sha256=plan["manifest_sha256"])
        train_tokens = plan["arms"][str(selected)]["valid_tokens"]
        model, loaded_meta, _ = load_v6_model_checkpoint(
            str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True)
        if loaded_meta["checkpoint_id"] != meta["checkpoint_id"]:
            raise RuntimeError("Source checkpoint changed during worker initialization.")
        model.eval().requires_grad_(False)
        validate_bit_recovery_source(model, loaded_meta)
        _write_report(report_path, report)
        print(json.dumps({"status": "capturing", "shard_index": shard_index, "targets": len(targets),
                          "selected_train_documents": selected, "train_tokens": train_tokens,
                          "heldout_tokens": data["heldout"]["valid_tokens"]}), flush=True)
        phase_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        # Build and verify the full pilot manifest first, then capture only the
        # selected training prefix. Held-out documents still start after all 512.
        capture_train = {key: value[:selected] for key, value in train_batch.items()}
        bundles = capture_shard_inputs(teacher, targets, capture_train, heldout_batch)
        torch.cuda.synchronize(device)
        report["teacher_capture_seconds"] = time.perf_counter() - phase_start
        report["teacher_capture_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["capture_train_documents"] = selected
        report["capture_heldout_documents"] = 64
        report["teacher_cache_cpu_bytes"] = sum(t.numel() * t.element_size() for bundle in bundles.values() for t in bundle.values())
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        report["status"] = "running"
        _write_report(report_path, report)
        specs = {item["name"]: item for item in meta["converted_modules"]}
        for target in targets:
            bundle = bundles.pop(target)
            train_x, heldout_x = bundle["train_inputs"], bundle["heldout_inputs"]
            if (len(train_x) != train_tokens or len(heldout_x) != data["heldout"]["valid_tokens"]
                    or train_x.dtype != torch.bfloat16 or heldout_x.dtype != torch.bfloat16):
                raise ValueError(f"Teacher token inventory/dtype differs from the fixed manifest: {target}.")
            layer = model.get_submodule(target)
            source_module_sha = _digest(layer.state_dict(), set())
            cpu_calibration = calibrate_local_bits_cpu(layer)
            lr = max(stage["scale"] for stage in cpu_calibration["stages"]) / 64.0
            banks = _code_banks(layer)
            source_codes = {name: layer.state_dict()[name].detach().cpu().clone() for name, _ in banks}
            layer.to(device)
            module_started = time.perf_counter()

            def progress(row):
                if row["step"] % 10 == 0 or row["step"] == configuration["steps"]:
                    print(json.dumps({"target": target, "shard_index": shard_index, **row}), flush=True)

            metrics = optimize_local_bits(
                layer, bundle["reference_weight"], train_x[:train_tokens], heldout_x,
                steps=configuration["steps"], lr=lr, round_steps=configuration["round_steps"],
                chunk_size=configuration["chunk_size"], seed=configuration["seed"], progress_callback=progress)
            patch_file, module_file = Path("patches") / (target + ".pt"), Path("reports") / (target + ".json")
            patch_evidence = save_code_patch(layer, source_codes, output / patch_file)
            if patch_evidence["stage_changed_bits"] != metrics["evidence"]["stage_changed_bits"]:
                raise RuntimeError("Serialized patch logical changes differ from the core evidence.")
            metrics.update(
                protocol_version=PROTOCOL_VERSION, target=target, source_checkpoint_dir=str(source),
                source_checkpoint_id=meta["checkpoint_id"], pilot_json=str(pilot_path), pilot_sha256=pilot_sha,
                selected_train_documents=selected, configuration=configuration, configuration_sha256=configuration_sha,
                manifest_sha256=plan["manifest_sha256"], source_module_sha256=source_module_sha,
                source_spec_sha256=_json_digest(specs[target]), cpu_coordinate_calibration=cpu_calibration, lr=lr,
                trajectory=_trajectory(metrics), worker_module_elapsed_seconds=time.perf_counter() - module_started,
                **patch_evidence)
            _write_report(output / module_file, metrics)
            report["modules"][target] = {
                "patch_file": str(patch_file), "report_file": str(module_file),
                "patch_sha256": _file_sha256(output / patch_file), "report_sha256": _file_sha256(output / module_file),
            }
            _write_report(report_path, report)
            layer.cpu()
            layer._build_parallel_stage_decode_plan()
            layer.clear_decoded_weight_cache()
            del bundle, train_x, heldout_x, source_codes
            gc.collect()
            torch.cuda.empty_cache()
        if set(report["modules"]) != set(targets):
            raise RuntimeError("Worker did not complete every assigned projection.")
        report.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        print(json.dumps({"status": "complete", "shard_index": shard_index,
                          "modules": len(targets), "report": str(report_path)}), flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        raise
