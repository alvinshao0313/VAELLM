"""Strict CPU finalization of the 252 fixed-code local decoder updates."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import (
    CATEGORIES, _assert_state_equal, _load_state, _require, _validate_sources,
)
from experiments.qwen3_w25.calibration import DATASET_PATH, DATASET_REVISION
from experiments.qwen3_w25.local_decoder import (
    _DECODER_PREFIX, _frozen_digest, _schema, _stage_norm, _storage_bits,
)
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load, save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


_CODE_FILES = {
    "experiments/qwen3_w25/local_recovery.py", "experiments/qwen3_w25/local_decoder.py",
    "experiments/qwen3_w25/calibration.py", "experiments/linear_output/objectives.py",
}
_FIXED_CONFIG = {
    "seed": 31, "train_documents": 32, "heldout_documents": 32, "max_length": 256,
    "chunk_size": 256, "objective": "full_linear_output_mse", "optimizer": "AdamW",
    "optimizer_dtype": "float32", "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8,
}


def _json(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(result, dict), f"Expected JSON object: {path}.")
    return result


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _inside(directory, relative):
    _require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
             f"Expected a relative shard artifact path: {relative!r}.")
    path = (directory / relative).resolve()
    _require(path.is_relative_to(directory) and path.is_file(), f"Missing/outside shard artifact: {path}.")
    return path


def _validate_data(data):
    _require(data.get("dataset_path") == DATASET_PATH and data.get("revision") == DATASET_REVISION
             and data.get("split") == "train" and data.get("dataset_task") == "lm"
             and data.get("seed") == 31 and data.get("max_length") == 256,
             "Calibration manifest does not match the fixed Alpaca train protocol.")
    seen = {field: set() for field in ("row_index", "group_sha256", "token_sha256")}
    for split in ("train", "heldout"):
        rows = data[split]["documents"]
        _require(isinstance(rows, list) and len(rows) == 32, f"Expected exactly 32 {split} documents.")
        tokens = 0
        for row in rows:
            _require(type(row["valid_tokens"]) is int and 2 <= row["valid_tokens"] <= 256,
                     f"Invalid document token count: {split}.")
            _require(type(row["row_index"]) is int and 0 <= row["row_index"] < data["source_rows"],
                     f"Invalid source document index: {split}.")
            _require(all(_valid_sha(row[field]) for field in ("group_sha256", "token_sha256", "content_sha256")),
                     f"Invalid document fingerprints: {split}.")
            for field in seen:
                _require(row[field] not in seen[field], f"Repeated training/held-out document {field}.")
                seen[field].add(row[field])
            tokens += row["valid_tokens"]
        _require(data[split]["valid_tokens"] == tokens, f"Inconsistent {split} token total.")


def _read_shards(source_dir, source_meta, shard_dirs):
    # Import here so the worker/CLI can import this finalizer without a cycle.
    from experiments.qwen3_w25.local_recovery import PROTOCOL_VERSION, TARGETS, shard_targets

    directories = [Path(path).resolve() for path in shard_dirs]
    _require(directories and len(directories) == len(set(directories)), "Supply every shard exactly once.")
    workers = [_json(path / "worker.json") for path in directories]
    first = workers[0]
    config = first["config"]
    _require(set(config) == set(_FIXED_CONFIG) | {"steps", "lr", "num_shards"}, "Unexpected/missing worker settings.")
    _require(all(config[key] == value for key, value in _FIXED_CONFIG.items()), "Worker settings violate the fixed protocol.")
    _require(type(config["steps"]) is int and config["steps"] > 0
             and type(config["lr"]) in (int, float) and math.isfinite(config["lr"]) and config["lr"] > 0,
             "Invalid recovery steps/lr.")
    count = config["num_shards"]
    _require(type(count) is int and 1 <= count <= 36 and len(workers) == count, "Missing/invalid shard count.")
    _validate_data(first["data"])
    _require(set(first["code_sha256"]) == _CODE_FILES and all(_valid_sha(v) for v in first["code_sha256"].values()),
             "Missing/invalid worker code identities.")
    indices, artifacts, manifests = set(), {}, []
    for directory, worker in zip(directories, workers):
        _require(worker.get("status") == "complete" and worker.get("protocol_version") == PROTOCOL_VERSION,
                 f"Incomplete/wrong-version shard: {directory}.")
        _require(Path(worker["source_checkpoint_dir"]).resolve() == source_dir
                 and worker["source_checkpoint_id"] == source_meta["checkpoint_id"],
                 f"Shard source checkpoint identity mismatch: {directory}.")
        _require(all(worker[key] == first[key] for key in ("config", "data", "code_sha256")),
                 f"Shards differ in config, data or code identity: {directory}.")
        index = worker["shard_index"]
        _require(type(index) is int and 0 <= index < count and index not in indices, "Duplicate/invalid shard index.")
        indices.add(index)
        targets = list(shard_targets(index, count))
        _require(worker["targets"] == targets and set(worker["modules"]) == set(targets),
                 f"Shard target coverage/order mismatch: {directory}.")
        for target in targets:
            _require(target not in artifacts, f"Duplicate target: {target}.")
            entry = worker["modules"][target]
            _require(set(entry) == {"patch_file", "report_file", "patch_sha256"}, f"Unexpected artifact metadata: {target}.")
            patch_path, report_path = [_inside(directory, entry[key]) for key in ("patch_file", "report_file")]
            _require(_valid_sha(entry["patch_sha256"]) and _sha(patch_path) == entry["patch_sha256"],
                     f"Patch SHA256 mismatch: {target}.")
            artifacts[target] = (patch_path, report_path)
        manifests.append({"directory": str(directory), "worker_sha256": _sha(directory / "worker.json"), "shard_index": index})
    _require(indices == set(range(count)) and set(artifacts) == set(TARGETS) and len(artifacts) == 252,
             "Recovery must cover all 252 targets exactly once.")
    return first, artifacts, manifests


def _validate_source(meta):
    for spec in meta["converted_modules"]:
        name = spec["name"]
        _require(spec["residual_stages"] == 2 and spec["parallel_parts"] == 1
                 and spec.get("parallel_stage_decode") is True, f"Expected packed two-stage/one-part source: {name}.")
        rotation = spec.get("weight_rotation") or {}
        _require(rotation.get("type") == "two_sided_hadamard" and rotation.get("block_size") == 0,
                 f"Expected full two-sided rotation: {name}.")
        _require(all(spec.get(field) is None for field in (
            "low_rank_a", "low_rank_b", "protected_input_indices", "protected_output_indices",
            "protected_residual_indices", "sparse_residual_row_indices", "sparse_residual_col_indices",
            "sparse_residual_values", "sparse_residual_active_block_ids", "sparse_residual_qvalues")),
            f"LoRA/protected/sparse payload is forbidden: {name}.")


def _apply_patch(module, patch, report, *, target, source_dir, source_id, config, data):
    _require(report.get("target") == target and report.get("source_checkpoint_id") == source_id
             and report.get("protocol_version") == 1
             and Path(report["source_checkpoint_dir"]).resolve() == source_dir, f"Module report identity mismatch: {target}.")
    state = module.state_dict()
    keys = {key for key in state if key.startswith(_DECODER_PREFIX)}
    _require(isinstance(patch, dict) and keys and set(patch) == keys, f"Patch must contain exactly every packed decoder tensor: {target}.")
    for key, tensor in patch.items():
        _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.layout == torch.strided
                 and tensor.shape == state[key].shape and tensor.dtype == state[key].dtype,
                 f"Patch tensor schema mismatch: {target}.{key}.")
        _require(torch.isfinite(tensor).all().item(), f"Non-finite patch: {target}.{key}.")
    settings, evidence = report["settings"], report["evidence"]
    for key in ("steps", "lr", "chunk_size", "optimizer", "optimizer_dtype", "weight_decay"):
        _require(settings[key] == config[key], f"Module settings mismatch for {key}: {target}.")
    _require(settings["train_tokens"] == data["train"]["valid_tokens"]
             and settings["heldout_tokens"] == data["heldout"]["valid_tokens"], f"Module token counts mismatch: {target}.")
    _require(settings["native_decode_dtype"] == "torch.bfloat16", f"Expected BF16 teacher/native activation protocol: {target}.")
    _require(all(settings["storage_dtype"] == str(t.dtype) for t in patch.values()), f"Storage dtype mismatch: {target}.")
    losses = report["training_loss"]
    _require(len(losses) == config["steps"] and all(math.isfinite(value) and value >= 0 for value in losses),
             f"Incomplete/non-finite training history: {target}.")
    for phase in ("before_native", "before_fp32", "after_fp32", "after_cast"):
        _require(set(report[phase]) == {"train_mse", "heldout_mse", "weight_mse"}
                 and all(math.isfinite(v) and v >= 0 for v in report[phase].values()), f"Invalid reconstruction metrics: {target}/{phase}.")
    _require(all(evidence.get(key) is True for key in (
        "frozen_unchanged", "tensor_schema_unchanged", "decoder_parameter_ids_unchanged")), f"Failed invariant evidence: {target}.")
    frozen, bits, schema = _frozen_digest(module), _storage_bits(module), _schema(module)
    _require(evidence["frozen_sha256_before"] == evidence["frozen_sha256_after"] == frozen,
             f"Frozen payload digest differs from source: {target}.")
    _require(evidence["persistent_storage_bits_before"] == evidence["persistent_storage_bits_after"] == bits,
             f"Persistent storage evidence differs from source: {target}.")
    _require(evidence["optimizer_moment_dtypes"] == ["torch.float32"], f"Optimizer moments are not FP32: {target}.")
    for field in ("stage_gradient_l2_max", "stage_parameter_delta_l2_fp32", "stage_parameter_delta_l2_saved"):
        _require(len(evidence[field]) == 2 and all(math.isfinite(v) and v > 0 for v in evidence[field]),
                 f"Both stages must actually update: {target}/{field}.")
    parameters = dict(module._parallel_stage_decoder.named_parameters())
    deltas = _stage_norm([patch[_DECODER_PREFIX + name].float() - p.detach().float()
                          for name, p in parameters.items()], list(module._parallel_stage_layout))
    _require(all(value > 0 and math.isclose(value, stated, rel_tol=1e-6, abs_tol=1e-10)
                 for value, stated in zip(deltas, evidence["stage_parameter_delta_l2_saved"])),
             f"Saved decoder update differs from evidence/source: {target}.")
    ids = {name: id(p) for name, p in parameters.items()}
    module.load_state_dict({**state, **patch}, strict=True)
    _require(_frozen_digest(module) == frozen and _schema(module) == schema and _storage_bits(module) == bits,
             f"Applying decoder patch changed frozen payload/schema/storage: {target}.")
    _require({name: id(p) for name, p in module._parallel_stage_decoder.named_parameters()} == ids,
             f"Applying patch replaced decoder Parameters: {target}.")
    module.clear_decoded_weight_cache()
    return deltas


def _save_final(model, output_dir, source_meta, expected_state, provenance, tokenizer=None):
    refresh_vae_linear_runtime_after_state_load(model)
    _assert_state_equal(model.state_dict(), expected_state, "Recovered model before save")
    saved = save_v6_full_checkpoint(
        model, str(output_dir), checkpoint_kind="final_model", compressed_targets=source_meta["compressed_targets"],
        pending_dense_targets=(), skip_targets=(), train_mode="none", after_category_mode="none",
        base_model_path=source_meta["base_model_path"], tokenizer=tokenizer,
        extra_meta={"local_decoder_recovery": provenance},
    )
    report = {"status": "invalid", "checkpoint_id": saved["checkpoint_id"],
              "checkpoint_dir": str(output_dir), "provenance": provenance}
    try:
        report["serialized_tensors_compared"] = _assert_state_equal(_load_state(output_dir), expected_state, "Saved recovery")
        loaded, meta, _ = load_v6_model_checkpoint(str(output_dir), map_location="cpu", strict=True, expected_kind="final_model")
        report["reloaded_tensors_compared"] = _assert_state_equal(loaded.state_dict(), expected_state, "Independent recovered v6 load")
        _require(meta["checkpoint_id"] != source_meta["checkpoint_id"] and meta.get("immutable_resume_contract") is None
                 and not (meta.get("extra_meta") or {}).get("cat_runtime_state_file"), "Output inherited source identity/resume state.")
        _require({s["name"]: s for s in meta["converted_modules"]} == {s["name"]: s for s in source_meta["converted_modules"]},
                 "Decoder recovery changed module metadata/topology.")
        _require(set(meta["compressed_targets"]) == set(source_meta["compressed_targets"])
                 and not meta["pending_dense_targets"] and not meta["skip_targets"], "Output projection inventory changed.")
        del loaded
        gc.collect()
        report["bit_budget"] = audit_checkpoint(output_dir, max_bpw=2.5, max_lora_rank=0)
        _require(report["bit_budget"]["budget_passed"], f"Output budget failed: {report['bit_budget']['contract_violations']}")
        report["status"] = "verified"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "finalize.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def finalize_local_recovery(source_dir, shard_dirs, output_dir) -> dict:
    from experiments.qwen3_w25.local_recovery import validate_recovery_source

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    shard_dirs = [Path(path).resolve() for path in shard_dirs]
    _require(not output_dir.exists() and not any(output_dir.is_relative_to(path) for path in [source_dir, *shard_dirs]),
             "Output must be new and outside every source/shard directory.")
    meta, _, _, original = _validate_sources(source_dir, source_dir, list(CATEGORIES))
    _validate_source(meta)
    source_audit = audit_checkpoint(source_dir, max_bpw=2.5, max_lora_rank=0)
    _require(source_audit["budget_passed"], f"Source budget failed: {source_audit['contract_violations']}")
    worker, artifacts, manifests = _read_shards(source_dir, meta, shard_dirs)
    expected = dict(_load_state(source_dir))
    model, _, _ = load_v6_model_checkpoint(str(source_dir), map_location="cpu", strict=True, expected_kind="final_model")
    validate_recovery_source(model, meta)
    module_reports = {}
    for target, (patch_path, report_path) in artifacts.items():
        patch = torch.load(patch_path, map_location="cpu", weights_only=True)
        report = _json(report_path)
        deltas = _apply_patch(model.get_submodule(target), patch, report, target=target, source_dir=source_dir,
                              source_id=meta["checkpoint_id"], config=worker["config"], data=worker["data"])
        expected.update({target + "." + key: tensor for key, tensor in patch.items()})
        module_reports[target] = {"patch_sha256": _sha(patch_path), "report_path": str(report_path),
                                  "report_sha256": _sha(report_path), "stage_parameter_delta_l2_saved": deltas}
    provenance = {"source_checkpoint_dir": str(source_dir), "source_checkpoint_id": meta["checkpoint_id"],
                  "protocol_version": worker["protocol_version"], "config": worker["config"], "data": worker["data"],
                  "code_sha256": worker["code_sha256"], "shards": manifests, "modules": module_reports,
                  "finalizer_sha256": _sha(Path(__file__)), "purpose": "inference_only; no CAT exact-resume contract"}
    tokenizer = AutoTokenizer.from_pretrained(str(original), use_fast=True, local_files_only=True)
    return _save_final(model, output_dir, meta, expected, provenance, tokenizer)
