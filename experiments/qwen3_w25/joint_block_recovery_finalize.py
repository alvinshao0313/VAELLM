"""Strict CPU assembly of all 36 independent teacher-input joint block patches."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.bit_pilot import _json_digest
from experiments.qwen3_w25.bit_recipe_pilot import SOURCE_CHECKPOINT_ID, validate_recipe_source
from experiments.qwen3_w25.joint_block_recovery import (
    CODE_FILES, COMMON_FIELDS, PROTOCOL_VERSION, block_state, recovery_configuration,
    load_training_document_selection, selected_plan_from_manifest, shard_layers, validate_joint_trajectory,
)
from experiments.qwen3_w25.joint_block_validation import (
    SOURCE_STATE_SHA256, _release_block, audit_joint_state, bit_configuration,
    collect_state_inventory, require_native_runtime,
)
from experiments.qwen3_w25.local_bit_recovery_finalize import _save_final
from experiments.qwen3_w25.local_decoder import _storage_bits
from experiments.qwen3_w25.local_recovery import BASE_MODEL
from experiments.qwen3_w25.local_recovery_finalize import _inside, _json, _valid_sha
from experiments.qwen3_w25.protected_bit_recovery import (
    _file_sha256, load_recipe_selection, source_state_sha256, validate_pilot_source_model,
)
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def validate_shard_coverage(workers):
    _require(len(workers) == 4, "Exactly four joint shards are required.")
    indices, layers = set(), set()
    for worker in workers:
        index = worker["shard_index"]
        expected = shard_layers(index)
        _require(index not in indices, "Duplicate joint shard index.")
        _require(worker.get("status") == "complete" and worker.get("protocol_version") == PROTOCOL_VERSION,
                 "Incomplete or wrong-version joint shard.")
        _require(worker["layers"] == expected and set(worker["blocks"]) == {str(i) for i in expected},
                 "Joint shard does not cover its nine assigned blocks.")
        indices.add(index)
        layers.update(expected)
    _require(indices == set(range(4)) and layers == set(range(36)), "Joint recovery needs all 36 blocks exactly once.")


def read_shards(source, meta, shard_dirs):
    directories = [Path(p).resolve() for p in shard_dirs]
    _require(len(directories) == len(set(directories)) == 4, "Supply four distinct joint shard directories.")
    workers = [_json(p / "worker.json") for p in directories]
    validate_shard_coverage(workers)
    first = workers[0]
    pilot, pilot_sha, _ = load_recipe_selection(first["pilot_json"], source, meta)
    selection = load_training_document_selection(source, pilot, pilot_sha)
    _require(first["selection_evidence"] == selection, "Joint document-count selection evidence changed.")
    config = recovery_configuration()
    _require(first["configuration"] == config and first["configuration_sha256"] == _json_digest(config),
             "Joint worker configuration differs from the fixed candidate.")
    _require(first["pilot_sha256"] == pilot_sha and first["source_state_sha256"] == SOURCE_STATE_SHA256,
             "Joint pilot/source tensor identity changed.")
    for key in ("data", "document_plan", "source_files_sha256", "source_bit_budget"):
        _require(first[key] == pilot[key], f"Joint worker/pilot {key} mismatch.")
    plan = selected_plan_from_manifest(first["data"], first["document_plan"])
    _require(first["selected_document_plan"] == plan and first["manifest_sha256"] == _json_digest(first["data"]),
             "Joint worker selected different calibration documents.")
    repo = Path(__file__).resolve().parents[2]
    _require(set(first["code_sha256"]) == set(CODE_FILES)
             and all(_valid_sha(value) and value == _file_sha256(repo / key) for key, value in first["code_sha256"].items()),
             "Joint worker dependency inventory or source hashes changed.")
    _require(all(first["code_sha256"].get(key) == value for key, value in pilot["code_sha256"].items()),
             "Joint recovery uses different pilot/core code.")
    artifacts, provenance = {}, []
    for directory, worker in zip(directories, workers, strict=True):
        _require(Path(worker["source_checkpoint_dir"]).resolve() == source
                 and worker["source_checkpoint_id"] == meta["checkpoint_id"], "Joint shard source identity differs.")
        _require(all(worker[key] == first[key] for key in (*COMMON_FIELDS, "data", "document_plan")),
                 "Joint shards differ in source, configuration, data or code identity.")
        _require(worker["persistent_storage_bits_before"] == worker["persistent_storage_bits_after"],
                 "Joint shard storage changed.")
        for index in worker["layers"]:
            entry = worker["blocks"][str(index)]
            _require(set(entry) == {"patch_file", "report_file", "patch_sha256", "report_sha256"},
                     "Unexpected joint artifact inventory.")
            paths = [_inside(directory, entry[key]) for key in ("patch_file", "report_file")]
            for path, field in zip(paths, ("patch_sha256", "report_sha256"), strict=True):
                _require(_valid_sha(entry[field]) and _file_sha256(path) == entry[field], "Joint artifact SHA mismatch.")
            artifacts[index] = tuple(paths)
        provenance.append({"directory": str(directory), "shard_index": worker["shard_index"],
                           "worker_sha256": _file_sha256(directory / "worker.json")})
    return first, artifacts, provenance, pilot


def apply_joint_patch(block, source_state, patch, *, prefix):
    """Apply only the actual seven decoder/two-bank inventories; zero final flips are legal."""
    inventory = collect_state_inventory(block, prefix=prefix)
    _require(isinstance(patch, dict) and set(patch) == set(inventory["allowed_state_keys"]),
             f"Joint patch differs from the actual decoder/code inventory: {prefix}.")
    original = block_state(source_state, prefix)
    actual = {prefix + "." + key: value for key, value in block.state_dict().items()}
    _assert_state_equal(actual, original, f"Unmodified source block before patch {prefix}")
    for key, tensor in patch.items():
        _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                 and tensor.shape == original[key].shape and tensor.dtype == original[key].dtype,
                 f"Joint patch tensor shape/dtype/device mismatch: {key}.")
    merged = {**original, **patch}
    audit = audit_joint_state(original, merged, inventory)
    block.load_state_dict({key.removeprefix(prefix + "."): value for key, value in merged.items()}, strict=True)
    _release_block(block)
    require_native_runtime(block)
    _assert_state_equal({prefix + "." + key: value for key, value in block.state_dict().items()}, merged,
                        f"Applied complete joint block state {prefix}")
    return inventory, audit


def finalize_joint_block_recovery(checkpoint_dir, shard_dirs, output_dir):
    source, output = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    directories = [Path(p).resolve() for p in shard_dirs]
    _require(not output.exists() and not any(output.is_relative_to(p) for p in (source, *directories)),
             "Final joint output must be new and outside source/shard directories.")
    meta, source_audit = _validate_checkpoint(source)
    _require(meta["checkpoint_id"] == SOURCE_CHECKPOINT_ID and source_state_sha256(source) == SOURCE_STATE_SHA256
             and source_audit["budget_passed"], "Joint finalizer requires the unchanged rank-zero BEST within 2.5 bpw.")
    worker, artifacts, shards, pilot = read_shards(source, meta, directories)
    _require(source_audit == worker["source_bit_budget"], "Joint source bit budget changed.")
    source_state = _load_state(source)
    expected = dict(source_state)
    model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL,
        map_location="cpu", strict=True, expected_kind="final_model")
    model.eval().requires_grad_(False)
    validate_recipe_source(model, loaded_meta)
    validate_pilot_source_model(model, pilot)
    _assert_state_equal(model.state_dict(), source_state, "Immutable full joint finalization source")
    storage_before, blocks = _storage_bits(model), {}
    for index in range(36):
        prefix = f"model.layers.{index}"
        patch_path, report_path = artifacts[index]
        report = _json(report_path)
        _require(report.get("status") == "complete" and report["block_index"] == index and report["prefix"] == prefix,
                 "Joint block report target/status mismatch.")
        _require(all(report[key] == worker[key] for key in COMMON_FIELDS), "Joint block provenance differs from its shard.")
        rates, rounds = bit_configuration(report["cpu_coordinate_calibration"])
        _require(rates == report["bit_lrs"] and rounds == report["round_steps"], "Joint block LR/round settings mismatch.")
        patch = torch.load(patch_path, map_location="cpu", weights_only=True)
        inventory, audit = apply_joint_patch(model.get_submodule(prefix), source_state, patch, prefix=prefix)
        _require(report["state_inventory"] == inventory and report["state_audit"] == audit,
                 "Joint saved patch differs from its actual state audit.")
        validate_joint_trajectory(report["local_metrics"], inventory, audit, rates, rounds,
                                  worker["selected_document_plan"], prefix=prefix)
        for split in ("train", "heldout"):
            sizes = report["execution_batch_sizes"][split]
            count = 512 if split == "train" else 64
            _require(sizes and all(type(v) is int and 1 <= v <= 8 for v in sizes) and sum(sizes) == count,
                     "Joint execution batches do not retain the 512-train/64-calibration-dev/cap8 contract.")
        expected.update(patch)
        blocks[str(index)] = {"patch_sha256": _file_sha256(patch_path), "report_sha256": _file_sha256(report_path),
            "report_path": str(report_path), "total_changed_code_bits": audit["total_changed_code_bits"],
            "native_metrics": {phase: report["local_metrics"][phase] for phase in ("before_native", "after_native")}}
    compared = _assert_state_equal(model.state_dict(), expected, "All 36 joint patches and every frozen source tensor")
    _require(_storage_bits(model) == storage_before, "Assembled joint model changed persistent storage.")
    provenance = {key: worker[key] for key in COMMON_FIELDS}
    provenance.update(data=worker["data"], document_plan=worker["document_plan"], shards=shards, blocks=blocks,
        purpose="independent teacher-input joint recovery; no LoRA, no end-to-end loss, inference only",
        complete_model_tensors_compared=compared, full_block_coverage=list(range(36)),
        projection_count=252, code_bank_count=504, finalizer_sha256=_file_sha256(Path(__file__)))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
    return _save_final(model, output, meta, expected, source_audit, provenance, tokenizer)
