"""CPU-only assembly and strict verification of fixed-decoder packed-code patches."""

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
from experiments.qwen3_w25.bit_pilot import _json_digest
from experiments.qwen3_w25.calibration import DATASET_PATH, DATASET_REVISION
from experiments.qwen3_w25.local_bits import _changed_bits, _code_banks, _digest
from experiments.qwen3_w25.local_decoder import _schema, _storage_bits
from experiments.qwen3_w25.local_recovery import TARGETS, shard_targets
from experiments.qwen3_w25.local_recovery_finalize import _inside, _json, _sha, _valid_sha, _validate_source
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load, save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def _code_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _validate_data(data, plan):
    _require(data.get("dataset_path") == DATASET_PATH and data.get("revision") == DATASET_REVISION
             and data.get("split") == "train" and data.get("dataset_task") == "lm"
             and data.get("seed") == 31 and data.get("max_length") == 256,
             "Expected the fixed Alpaca train calibration protocol.")
    seen = {key: set() for key in ("row_index", "group_sha256", "token_sha256")}
    lengths = {}
    for split, count in (("train", 512), ("heldout", 64)):
        rows = data[split]["documents"]
        _require(isinstance(rows, list) and len(rows) == count, f"Wrong {split} document count.")
        lengths[split] = []
        for row in rows:
            _require(type(row["row_index"]) is int and 0 <= row["row_index"] < data["source_rows"],
                     "Invalid calibration row index.")
            _require(type(row["valid_tokens"]) is int and 1 <= row["valid_tokens"] <= 256,
                     "Invalid calibration token count.")
            _require(all(_valid_sha(row[key]) for key in ("group_sha256", "token_sha256", "content_sha256")),
                     "Invalid document fingerprint.")
            for key in seen:
                _require(row[key] not in seen[key], f"Repeated calibration identity: {key}.")
                seen[key].add(row[key])
            lengths[split].append(row["valid_tokens"])
        _require(sum(lengths[split]) == data[split]["valid_tokens"], f"Wrong {split} token total.")
    expected = {
        "train_lengths": lengths["train"], "heldout_lengths": lengths["heldout"],
        "arms": {str(count): {"document_indices": list(range(count)),
                              "valid_tokens": sum(lengths["train"][:count]),
                              "documents_sha256": _json_digest(data["train"]["documents"][:count])}
                 for count in (32, 128, 512)},
        "selection": {"heldout_document_indices": list(range(32)),
                      "valid_tokens": sum(lengths["heldout"][:32]),
                      "documents_sha256": _json_digest(data["heldout"]["documents"][:32])},
        "confirmation": {"heldout_document_indices": list(range(32, 64)),
                         "valid_tokens": sum(lengths["heldout"][32:]),
                         "documents_sha256": _json_digest(data["heldout"]["documents"][32:])},
        "manifest_sha256": _json_digest(data),
    }
    _require(plan == expected, "Document plan does not match the calibration manifest.")


def _read_shards(source_dir, source_meta, shard_dirs):
    from experiments.qwen3_w25.local_bit_recovery import (
        CODE_FILES, PROTOCOL_VERSION, bit_recovery_configuration, load_pilot_selection,
    )

    directories = [Path(path).resolve() for path in shard_dirs]
    _require(directories and len(directories) == len(set(directories)), "Supply every shard exactly once.")
    workers = [_json(path / "worker.json") for path in directories]
    first = workers[0]
    config = first["configuration"]
    count = config["num_shards"]
    _require(type(count) is int and 1 <= count <= 36 and len(workers) == count, "Missing/invalid shard count.")
    selected = first["selected_train_documents"]
    _require(config == bit_recovery_configuration(train_documents=selected, num_shards=count),
             "Recovery settings differ from the selected fixed protocol.")
    pilot_path = Path(first["pilot_json"]).resolve()
    pilot, pilot_sha, pilot_selected = load_pilot_selection(pilot_path, source_dir, source_meta)
    _require(selected == pilot_selected and first["pilot_sha256"] == pilot_sha,
             "Worker selection or pilot fingerprint differs from the completed pilot.")
    _require(first["data"] == pilot["data"] and first["document_plan"] == pilot["document_plan"],
             "Worker calibration data/plan differs from the pilot.")
    _validate_data(first["data"], first["document_plan"])
    _require(first["manifest_sha256"] == _json_digest(first["data"]), "Manifest fingerprint mismatch.")
    _require(first["configuration_sha256"] == _json_digest(config), "Configuration fingerprint mismatch.")
    _require(set(first["code_sha256"]) == set(CODE_FILES)
             and all(_valid_sha(value) for value in first["code_sha256"].values()), "Invalid worker code identities.")
    _require(all(first["code_sha256"].get(key) == value for key, value in pilot["code_sha256"].items()),
             "Recovery uses different pilot/core source code.")
    common = ("configuration", "configuration_sha256", "data", "document_plan", "manifest_sha256",
              "code_sha256", "pilot_json", "pilot_sha256", "selected_train_documents")
    indices, artifacts, provenance = set(), {}, []
    for directory, worker in zip(directories, workers):
        _require(worker.get("status") == "complete" and worker.get("protocol_version") == PROTOCOL_VERSION,
                 f"Incomplete/wrong-version shard: {directory}.")
        _require(Path(worker["source_checkpoint_dir"]).resolve() == source_dir
                 and worker["source_checkpoint_id"] == source_meta["checkpoint_id"], "Shard source identity mismatch.")
        _require(all(worker[key] == first[key] for key in common), "Shards differ in configuration/data/pilot/code identity.")
        index = worker["shard_index"]
        _require(type(index) is int and 0 <= index < count and index not in indices, "Duplicate/invalid shard index.")
        indices.add(index)
        targets = list(shard_targets(index, count))
        _require(worker["targets"] == targets and set(worker["modules"]) == set(targets),
                 f"Incomplete/incorrect target coverage: shard {index}.")
        for target in targets:
            _require(target not in artifacts, f"Repeated target: {target}.")
            entry = worker["modules"][target]
            _require(set(entry) == {"patch_file", "report_file", "patch_sha256", "report_sha256"},
                     f"Unexpected artifact metadata: {target}.")
            paths = [_inside(directory, entry[key]) for key in ("patch_file", "report_file")]
            for path, field in zip(paths, ("patch_sha256", "report_sha256")):
                _require(_valid_sha(entry[field]) and _sha(path) == entry[field], f"Artifact SHA256 mismatch: {target}/{field}.")
            artifacts[target] = tuple(paths)
        provenance.append({"directory": str(directory), "shard_index": index,
                           "worker_sha256": _sha(directory / "worker.json")})
    _require(indices == set(range(count)) and set(artifacts) == set(TARGETS) and len(artifacts) == 252,
             "Recovery must cover every one of the 252 targets exactly once.")
    return first, artifacts, provenance


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate_module_report(report, worker, target, source_dir, source_id):
    config = worker["configuration"]
    _require(report.get("target") == target and report.get("protocol_version") == worker["protocol_version"]
             and report.get("source_checkpoint_id") == source_id
             and Path(report["source_checkpoint_dir"]).resolve() == source_dir, f"Module report identity mismatch: {target}.")
    for key in ("pilot_json", "pilot_sha256", "selected_train_documents", "configuration", "configuration_sha256", "manifest_sha256"):
        _require(report[key] == worker[key], f"Module {key} mismatch: {target}.")
    settings, evidence = report["settings"], report["evidence"]
    for key in ("steps", "round_steps", "seed", "chunk_size", "optimizer", "weight_decay",
                "active_ratio", "streaming", "proxy_coordinates", "decoder_compute_dtype"):
        _require(settings[key] == config[key], f"Module optimizer setting mismatch: {target}/{key}.")
    _require(settings["native_decode_dtype"] == "torch.bfloat16"
             and settings["decoder_storage_dtype"] == "torch.bfloat16"
             and settings["stable_steps"] == 10 and settings["early_stable_round_end"] is True,
             f"Unexpected precision/round protocol: {target}.")
    tokens = worker["document_plan"]["arms"][str(worker["selected_train_documents"])]["valid_tokens"]
    _require(settings["train_tokens"] == tokens and settings["heldout_tokens"] == worker["data"]["heldout"]["valid_tokens"],
             f"Module token counts mismatch: {target}.")
    calibration = report["cpu_coordinate_calibration"]
    stages = calibration["stages"]
    _require(calibration["method"] == "cpu_fp32_dense_decoder_estimate" and len(stages) == 2
             and [stage["stage"] for stage in stages] == [0, 1]
             and all(_finite(stage["scale"]) and stage["scale"] > 0 for stage in stages), "Invalid CPU coordinate calibration.")
    lr = max(stage["scale"] for stage in stages) / 64.0
    _require(settings["lr"] == report["lr"] == lr, f"Module learning rate differs from its calibration: {target}.")
    for phase in ("before_native", "before_fp32", "after_fp32", "after_native"):
        _require(set(report[phase]) == {"train_mse", "heldout_mse", "weight_mse"}
                 and all(_finite(value) for value in report[phase].values()), f"Invalid endpoint metrics: {target}/{phase}.")
    losses, telemetry = report["training_loss"], report["step_telemetry"]
    _require(len(losses) == len(telemetry) == config["steps"] and all(_finite(value) for value in losses),
             f"Incomplete/non-finite optimizer history: {target}.")
    round_index = round_step = cumulative = stable = total_flips = ended_count = 0
    had_flip = False
    gradient_max = [0.0, 0.0]
    for step, row in enumerate(telemetry, 1):
        flips = row["step_flip_count"]
        _require(type(flips) is int and flips >= 0, "Invalid actual packed flip count.")
        round_step += 1
        cumulative += flips
        total_flips += flips
        if flips:
            had_flip, stable = True, 0
        elif had_flip:
            stable += 1
        ended = round_step >= config["round_steps"] or (had_flip and stable >= 10)
        expected = {"step": step, "global_bit_round": round_index, "bit_round_step": round_step,
                    "cumulative_flip_count": cumulative, "stable_counter": stable, "stable_steps": 10,
                    "had_flip": had_flip, "round_ended": ended, "loss": losses[step - 1]}
        _require(all(row[key] == value for key, value in expected.items()), f"Inconsistent optimizer trajectory: {target}/{step}.")
        norms = row["stage_gradient_l2"]
        _require(len(norms) == 2 and all(_finite(value) for value in norms), "Invalid stage gradients.")
        gradient_max = [max(old, new) for old, new in zip(gradient_max, norms)]
        if ended:
            round_index += 1
            ended_count += 1
            round_step = cumulative = stable = 0
            had_flip = False
    _require(evidence["stage_gradient_l2_max"] == gradient_max and all(value > 0 for value in gradient_max),
             "Both packed stages must have actual nonzero gradients.")
    _require(evidence["cumulative_step_flips"] == total_flips and evidence["rounds_ended"] == ended_count,
             "Reported flip/round totals differ from the optimizer trajectory.")
    _require(all(evidence.get(key) is True for key in (
        "coordinate_scales_fixed", "non_code_bytes_unchanged", "decoder_compute_parameters_unchanged",
        "tensor_schema_unchanged", "parameter_ids_unchanged", "packed_padding_unchanged")), "Failed invariant evidence.")
    for key in ("proxy_parameter_dtypes", "proxy_gradient_dtypes", "optimizer_moment_dtypes"):
        _require(evidence[key] == ["torch.float32"], f"Expected FP32 optimization: {key}.")
    coordinates = evidence["stage_coordinates"]
    _require(len(coordinates) == 2 and [item["stage"] for item in coordinates] == [0, 1], "Invalid GPU coordinate banks.")
    for item in coordinates:
        _require(_finite(item["scale"]) and item["scale"] > 0 and item["radius"] == item["scale"] / 2
                 and math.isclose(item["lr_over_radius"], lr / item["radius"], rel_tol=1e-12), "Invalid fixed GPU coordinate scale.")


def _apply_code_patch(module, patch, report, *, target):
    """Validate real packed changes against the untouched source, then copy them."""
    state = module.state_dict()
    banks = _code_banks(module)
    keys = [name for name, _ in banks]
    _require(isinstance(patch, dict) and set(patch) == set(keys), f"Patch must contain exactly the two packed code tensors: {target}.")
    _require(_digest(state, set()) == report["source_module_sha256"], f"Source module digest mismatch: {target}.")
    evidence = report["evidence"]
    frozen, bits, schema = _digest(state, set(keys)), _storage_bits(module), _schema(module)
    _require(evidence["code_state_keys"] == keys and evidence["frozen_sha256_before"] == evidence["frozen_sha256_after"] == frozen,
             f"Frozen payload digest/code inventory mismatch: {target}.")
    _require(evidence["persistent_storage_bits_before"] == evidence["persistent_storage_bits_after"] == bits,
             f"Persistent storage evidence differs from source: {target}.")
    changed, before_hashes, after_hashes = [], [], []
    pointers = set()
    for name, logical_shape in banks:
        tensor, original = patch[name], state[name]
        _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.layout == torch.strided
                 and tensor.dtype == original.dtype == torch.uint8 and tensor.shape == original.shape
                 and tensor.is_contiguous() and tensor.storage_offset() == 0
                 and tensor.untyped_storage().nbytes() == original.untyped_storage().nbytes() == tensor.numel(),
                 f"Packed patch schema/storage mismatch: {target}.{name}.")
        pointer = tensor.untyped_storage().data_ptr()
        _require(pointer not in pointers, f"Packed patch banks alias each other: {target}.")
        pointers.add(pointer)
        changed.append(_changed_bits(original.detach().cpu(), tensor, logical_shape))
        before_hashes.append(_code_sha(original))
        after_hashes.append(_code_sha(tensor))
    stated = evidence["stage_changed_bits"]
    _require(len(stated) == 2 and all(type(value) is int and value >= 0 for value in stated)
             and stated == changed == report["stage_changed_bits"] and sum(changed) > 0
             and evidence["final_changed_bits"] == sum(changed),
             f"Actual packed XOR changes disagree with the report or are a no-op: {target}.")
    _require(report["stage_code_sha256_before"] == before_hashes and report["stage_code_sha256_after"] == after_hashes,
             f"Packed bank fingerprint mismatch: {target}.")
    _require(report["code_schema"] == {name: {"shape": list(patch[name].shape), "dtype": str(patch[name].dtype)} for name in keys},
             f"Reported packed code schema mismatch: {target}.")
    ids = {name: id(parameter) for name, parameter in module.named_parameters()}
    module.load_state_dict({**state, **patch}, strict=True)
    _require(_digest(module.state_dict(), set(keys)) == frozen and _schema(module) == schema and _storage_bits(module) == bits,
             f"Applying codes changed non-code bytes/schema/storage: {target}.")
    _require(ids == {name: id(parameter) for name, parameter in module.named_parameters()}, "Applying codes replaced Parameters.")
    module._build_parallel_stage_decode_plan()
    module.clear_decoded_weight_cache()
    return changed


def _save_final(model, output_dir, source_meta, expected_state, source_audit, provenance, tokenizer=None):
    refresh_vae_linear_runtime_after_state_load(model)
    _assert_state_equal(model.state_dict(), expected_state, "Recovered codes before save")
    saved = save_v6_full_checkpoint(
        model, str(output_dir), checkpoint_kind="final_model", compressed_targets=source_meta["compressed_targets"],
        pending_dense_targets=(), skip_targets=(), train_mode="none", after_category_mode="none", lora_config=None,
        base_model_path=source_meta["base_model_path"], tokenizer=tokenizer,
        extra_meta={"local_bit_recovery": provenance},
    )
    report = {"status": "invalid", "checkpoint_id": saved["checkpoint_id"],
              "checkpoint_dir": str(output_dir), "provenance": provenance}
    try:
        report["serialized_tensors_compared"] = _assert_state_equal(_load_state(output_dir), expected_state, "Serialized recovered codes")
        loaded, meta, _ = load_v6_model_checkpoint(str(output_dir), map_location="cpu", strict=True, expected_kind="final_model")
        report["reloaded_tensors_compared"] = _assert_state_equal(loaded.state_dict(), expected_state, "Fresh strict recovered code load")
        _require(meta["checkpoint_id"] != source_meta["checkpoint_id"] and meta.get("immutable_resume_contract") is None
                 and not (meta.get("extra_meta") or {}).get("cat_runtime_state_file"), "Output inherited source identity/resume state.")
        _require({spec["name"]: spec for spec in meta["converted_modules"]}
                 == {spec["name"]: spec for spec in source_meta["converted_modules"]}, "Packed-code recovery changed module specs.")
        _require(set(meta["compressed_targets"]) == set(TARGETS) and not meta["pending_dense_targets"]
                 and not meta["skip_targets"] and meta.get("lora_config") is None, "Output does not retain all 252 rank-zero targets.")
        del loaded
        gc.collect()
        audit = audit_checkpoint(output_dir, max_bpw=2.5, max_lora_rank=0)
        _require(audit["budget_passed"], f"Output budget failed: {audit['contract_violations']}")
        for key in ("original_linear_weights", "user_linear_bits", "packed_linear_bits", "code_padding_bits", "full_model_tensor_bytes"):
            _require(audit[key] == source_audit[key], f"Packed-code recovery changed actual storage: {key}.")
        report.update(status="verified", bit_budget=audit)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "finalize.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def finalize_local_bit_recovery(source_dir, shard_dirs, output_dir) -> dict:
    from experiments.qwen3_w25.local_bit_recovery import validate_bit_recovery_source

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    shard_dirs = [Path(path).resolve() for path in shard_dirs]
    _require(not output_dir.exists() and not any(output_dir.is_relative_to(path) for path in [source_dir, *shard_dirs]),
             "Output must be new and outside every source/shard directory.")
    meta, _, _, original = _validate_sources(source_dir, source_dir, list(CATEGORIES))
    _validate_source(meta)
    audit = audit_checkpoint(source_dir, max_bpw=2.5, max_lora_rank=0)
    _require(audit["budget_passed"], f"Source budget failed: {audit['contract_violations']}")
    worker, artifacts, shards = _read_shards(source_dir, meta, shard_dirs)
    expected = dict(_load_state(source_dir))
    model, _, _ = load_v6_model_checkpoint(str(source_dir), map_location="cpu", strict=True, expected_kind="final_model")
    validate_bit_recovery_source(model, meta)
    specs = {spec["name"]: spec for spec in meta["converted_modules"]}
    modules = {}
    for target, (patch_path, report_path) in artifacts.items():
        patch = torch.load(patch_path, map_location="cpu", weights_only=True)
        report = _json(report_path)
        _validate_module_report(report, worker, target, source_dir, meta["checkpoint_id"])
        _require(report["source_spec_sha256"] == _json_digest(specs[target]), f"Source module metadata fingerprint mismatch: {target}.")
        changed = _apply_code_patch(model.get_submodule(target), patch, report, target=target)
        expected.update({target + "." + key: value for key, value in patch.items()})
        modules[target] = {"patch_sha256": _sha(patch_path), "report_sha256": _sha(report_path),
                           "report_path": str(report_path), "stage_changed_bits": changed}
    provenance = {key: worker[key] for key in (
        "protocol_version", "source_checkpoint_dir", "source_checkpoint_id", "pilot_json", "pilot_sha256",
        "selected_train_documents", "configuration", "configuration_sha256", "data", "document_plan",
        "manifest_sha256", "code_sha256")}
    provenance.update(shards=shards, modules=modules, finalizer_sha256=_sha(Path(__file__)),
                      purpose="inference_only; only existing packed codes changed; no training resume state")
    tokenizer = AutoTokenizer.from_pretrained(str(original), use_fast=True, local_files_only=True)
    return _save_final(model, output_dir, meta, expected, audit, provenance, tokenizer)
