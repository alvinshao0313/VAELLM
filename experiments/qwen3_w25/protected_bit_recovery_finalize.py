"""Strict CPU assembly of protected-base code patches without changing INT8 payloads."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import CATEGORIES, _load_state, _require, _validate_sources
from experiments.qwen3_w25.bit_pilot import _json_digest
from experiments.qwen3_w25.bit_recipe_pilot import validate_recipe_source
from experiments.qwen3_w25.local_bit_recovery_finalize import (
    _apply_code_patch, _finite, _save_final, _validate_data,
)
from experiments.qwen3_w25.local_recovery import TARGETS, shard_targets
from experiments.qwen3_w25.local_recovery_finalize import _inside, _json, _sha, _valid_sha
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def validate_data_plan(data, plan):
    # Reuse the existing full document-identity validator, then verify the
    # protected pilot's narrower plan vocabulary without inventing new rows.
    train = [row["valid_tokens"] for row in data["train"]["documents"]]
    heldout = [row["valid_tokens"] for row in data["heldout"]["documents"]]
    canonical = {"train_lengths": train, "heldout_lengths": heldout,
        "arms": {str(count): {"document_indices": list(range(count)), "valid_tokens": sum(train[:count]),
                              "documents_sha256": _json_digest(data["train"]["documents"][:count])}
                 for count in (32, 128, 512)},
        "selection": {"heldout_document_indices": list(range(32)), "valid_tokens": sum(heldout[:32]),
                      "documents_sha256": _json_digest(data["heldout"]["documents"][:32])},
        "confirmation": {"heldout_document_indices": list(range(32, 64)), "valid_tokens": sum(heldout[32:]),
                         "documents_sha256": _json_digest(data["heldout"]["documents"][32:])},
        "manifest_sha256": _json_digest(data)}
    _validate_data(data, canonical)
    expected = {"train_lengths": train, "heldout_lengths": heldout, "train": canonical["arms"]["512"],
                "selection": canonical["selection"], "repeated_diagnostic": canonical["confirmation"],
                "manifest_sha256": canonical["manifest_sha256"]}
    _require(plan == expected, "Protected pilot document plan differs from its complete 512+64 manifest.")


def read_shards(source_dir, meta, shard_dirs):
    from experiments.qwen3_w25.protected_bit_recovery import (
        CODE_FILES, PROTOCOL_VERSION, load_focused_recipe_selection, load_recipe_selection,
        recovery_configuration, source_state_sha256,
    )

    directories = [Path(path).resolve() for path in shard_dirs]
    _require(directories and len(set(directories)) == len(directories), "Supply each shard exactly once.")
    workers = [_json(path / "worker.json") for path in directories]
    first, config = workers[0], workers[0]["configuration"]
    count = config["num_shards"]
    _require(type(count) is int and 1 <= count <= 36 and len(workers) == count, "Missing/invalid shard count.")
    pilot, pilot_sha, divisor = load_recipe_selection(first["pilot_json"], source_dir, meta)
    focused = config["focused_selection"]
    focused = load_focused_recipe_selection(None if focused is None else focused["summary_path"],
                                            pilot_json=first["pilot_json"], source_dir=source_dir)
    _require(config == recovery_configuration(lr_divisor=divisor, num_shards=count, focused_selection=focused)
             and first["selected_lr_divisor"] == divisor and first["pilot_sha256"] == pilot_sha,
             "Worker configuration/selection differs from the completed pilot.")
    _require(first["configuration_sha256"] == _json_digest(config)
             and first["manifest_sha256"] == _json_digest(first["data"]), "Worker fingerprint mismatch.")
    for key in ("data", "document_plan", "source_files_sha256", "source_bit_budget"):
        _require(first[key] == pilot[key], f"Worker/pilot {key} mismatch.")
    _require(first["source_state_sha256"] == source_state_sha256(source_dir), "Source checkpoint tensor file changed.")
    validate_data_plan(first["data"], first["document_plan"])
    repo = Path(__file__).resolve().parents[2]
    _require(set(first["code_sha256"]) == set(CODE_FILES)
             and all(_valid_sha(value) and value == _sha(repo / key) for key, value in first["code_sha256"].items()),
             "Worker dependency inventory/fingerprint changed.")
    _require(all(first["code_sha256"][key] == value for key, value in pilot["code_sha256"].items()),
             "Worker and pilot core code differ.")
    common = ("configuration", "configuration_sha256", "data", "document_plan", "manifest_sha256",
              "code_sha256", "source_files_sha256", "source_state_sha256", "source_bit_budget", "pilot_json", "pilot_sha256", "selected_lr_divisor")
    indices, artifacts, manifests = set(), {}, []
    for directory, worker in zip(directories, workers):
        _require(worker.get("status") == "complete" and worker.get("protocol_version") == PROTOCOL_VERSION,
                 "Incomplete or wrong-version protected shard.")
        _require(Path(worker["source_checkpoint_dir"]).resolve() == source_dir
                 and worker["source_checkpoint_id"] == meta["checkpoint_id"], "Shard source mismatch.")
        _require(all(worker[key] == first[key] for key in common), "Shards differ in source/configuration/data/code/pilot.")
        index = worker["shard_index"]
        _require(type(index) is int and 0 <= index < count and index not in indices, "Duplicate/invalid shard index.")
        indices.add(index)
        targets = list(shard_targets(index, count))
        _require(worker["targets"] == targets and set(worker["modules"]) == set(targets), "Incomplete shard target coverage.")
        for target in targets:
            _require(target not in artifacts, "A projection occurs in multiple shards.")
            entry = worker["modules"][target]
            _require(set(entry) == {"patch_file", "report_file", "patch_sha256", "report_sha256"}, "Invalid artifact inventory.")
            paths = [_inside(directory, entry[key]) for key in ("patch_file", "report_file")]
            for path, key in zip(paths, ("patch_sha256", "report_sha256")):
                _require(_valid_sha(entry[key]) and _sha(path) == entry[key], f"Artifact digest mismatch: {target}/{key}.")
            artifacts[target] = tuple(paths)
        manifests.append({"directory": str(directory), "shard_index": index, "worker_sha256": _sha(directory / "worker.json")})
    _require(indices == set(range(count)) and len(artifacts) == 252 and set(artifacts) == set(TARGETS),
             "Protected recovery requires every one of the 252 projections exactly once.")
    return first, artifacts, manifests


def validate_module_report(report, worker, target):
    from experiments.qwen3_w25.protected_bit_recovery import projection_recipe

    _require(report.get("target") == target, "Wrong target in module report.")
    for key in ("protocol_version", "source_checkpoint_dir", "source_checkpoint_id", "pilot_json", "pilot_sha256",
                "selected_lr_divisor", "configuration", "configuration_sha256", "manifest_sha256",
                "source_files_sha256", "source_state_sha256", "code_sha256"):
        _require(report[key] == worker[key], f"Module {key} mismatch: {target}.")
    config, settings, evidence = worker["configuration"], report["settings"], report["evidence"]
    group, recipe = projection_recipe(config, target)
    _require(report["recipe_group"] == group and report["applied_recipe"] == recipe,
             "Module recipe does not match its projection family.")
    _require(settings["steps"] == recipe["steps"] and settings["round_steps"] == recipe["round_steps"],
             "Module step/round budget differs from the selected projection-family recipe.")
    for key in ("seed", "chunk_size", "optimizer", "weight_decay", "active_ratio",
                "streaming", "proxy_coordinates", "decoder_compute_dtype", "source_profile"):
        _require(settings[key] == config[key], f"Module optimizer setting mismatch: {target}/{key}.")
    _require(settings["decoder_storage_dtype"] == settings["native_decode_dtype"] == "torch.bfloat16"
             and settings["stable_steps"] == 10 and settings["early_stable_round_end"] is True,
             "Unexpected precision/round protocol.")
    _require(settings["train_tokens"] == worker["data"]["train"]["valid_tokens"]
             and settings["heldout_tokens"] == worker["data"]["heldout"]["valid_tokens"], "Wrong token inventory.")
    calibration, stages = report["cpu_coordinate_calibration"], report["cpu_coordinate_calibration"]["stages"]
    _require(calibration["method"] == "cpu_fp32_dense_decoder_estimate" and len(stages) == 2
             and [row["stage"] for row in stages] == [0, 1]
             and all(_finite(row["scale"]) and row["scale"] > 0 for row in stages), "Invalid CPU coordinate calibration.")
    lr = max(row["scale"] for row in stages) / recipe["lr_divisor"]
    _require(settings["lr"] == report["lr"] == lr, "Module LR differs from the selected projection-family recipe.")
    for phase in ("before_native", "before_fp32", "after_fp32", "after_native"):
        _require(set(report[phase]) == {"train_mse", "heldout_mse", "weight_mse"}
                 and all(_finite(value) for value in report[phase].values()), "Invalid native/FP32 metrics.")
    losses, telemetry = report["training_loss"], report["step_telemetry"]
    _require(len(losses) == len(telemetry) == recipe["steps"] and all(_finite(value) for value in losses),
             "Incomplete/non-finite optimizer trajectory.")
    round_index = round_step = cumulative = stable = total_flips = ended_count = 0
    had_flip, gradient_max = False, [0.0, 0.0]
    for step, row in enumerate(telemetry, 1):
        flips = row["step_flip_count"]
        _require(type(flips) is int and flips >= 0, "Invalid actual step flip count.")
        round_step, cumulative, total_flips = round_step + 1, cumulative + flips, total_flips + flips
        if flips:
            had_flip, stable = True, 0
        elif had_flip:
            stable += 1
        ended = round_step >= recipe["round_steps"] or (had_flip and stable >= 10)
        expected = {"step": step, "global_bit_round": round_index, "bit_round_step": round_step,
                    "cumulative_flip_count": cumulative, "stable_counter": stable, "stable_steps": 10,
                    "had_flip": had_flip, "round_ended": ended, "loss": losses[step - 1]}
        _require(all(row[key] == value for key, value in expected.items()), "Inconsistent round/flip trajectory.")
        norms = row["stage_gradient_l2"]
        _require(len(norms) == 2 and all(_finite(value) for value in norms), "Invalid gradient evidence.")
        gradient_max = [max(old, new) for old, new in zip(gradient_max, norms)]
        if ended:
            round_index, ended_count = round_index + 1, ended_count + 1
            round_step = cumulative = stable = 0
            had_flip = False
    _require(evidence["stage_gradient_l2_max"] == gradient_max and all(value > 0 for value in gradient_max)
             and evidence["cumulative_step_flips"] == total_flips and evidence["rounds_ended"] == ended_count,
             "Summary evidence disagrees with actual optimizer trajectory.")
    _require(all(evidence.get(key) is True for key in ("coordinate_scales_fixed", "non_code_bytes_unchanged",
             "decoder_compute_parameters_unchanged", "tensor_schema_unchanged", "parameter_ids_unchanged", "packed_padding_unchanged")),
             "A frozen-payload/schema/padding invariant failed.")
    for key in ("proxy_parameter_dtypes", "proxy_gradient_dtypes", "optimizer_moment_dtypes"):
        _require(evidence[key] == ["torch.float32"], "Expected FP32 code optimization.")
    coordinates = evidence["stage_coordinates"]
    _require(len(coordinates) == 2 and [row["stage"] for row in coordinates] == [0, 1], "Wrong GPU coordinate banks.")
    for row in coordinates:
        _require(_finite(row["scale"]) and row["scale"] > 0 and row["radius"] == row["scale"] / 2
                 and math.isclose(row["lr_over_radius"], lr / row["radius"], rel_tol=1e-12), "Invalid GPU coordinate scale.")


def finalize_protected_bit_recovery(source_dir, shard_dirs, output_dir):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    directories = [Path(path).resolve() for path in shard_dirs]
    _require(not output.exists() and not any(output.is_relative_to(path) for path in [source, *directories]),
             "Output must be new and outside the source/shard directories.")
    meta, _, _, original = _validate_sources(source, source, list(CATEGORIES))
    audit = audit_checkpoint(source, max_bpw=2.5, max_lora_rank=0)
    _require(audit["budget_passed"], "Source actual bit budget failed.")
    worker, artifacts, shards = read_shards(source, meta, directories)
    _require(audit == worker["source_bit_budget"], "Source audit changed after the worker.")
    expected = dict(_load_state(source))
    model, loaded_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True, expected_kind="final_model")
    validate_recipe_source(model, loaded_meta)
    specs = {spec["name"]: spec for spec in meta["converted_modules"]}
    modules = {}
    for target, (patch_path, report_path) in artifacts.items():
        patch, report = torch.load(patch_path, map_location="cpu", weights_only=True), _json(report_path)
        validate_module_report(report, worker, target)
        _require(report["source_spec_sha256"] == _json_digest(specs[target]), "Source module metadata changed.")
        changed = _apply_code_patch(model.get_submodule(target), patch, report, target=target)
        expected.update({target + "." + key: tensor for key, tensor in patch.items()})
        modules[target] = {"patch_sha256": _sha(patch_path), "report_sha256": _sha(report_path),
                           "report_path": str(report_path), "stage_changed_bits": changed}
    provenance = {key: worker[key] for key in ("protocol_version", "source_checkpoint_dir", "source_checkpoint_id",
        "pilot_json", "pilot_sha256", "selected_lr_divisor", "configuration", "configuration_sha256",
        "data", "document_plan", "manifest_sha256", "code_sha256", "source_files_sha256", "source_state_sha256")}
    provenance.update(shards=shards, modules=modules,
        finalizer_sha256={str(path.relative_to(Path(__file__).resolve().parents[2])): _sha(path)
                         for path in (Path(__file__), Path(__file__).with_name("local_bit_recovery_finalize.py"))},
        purpose="inference_only; main packed codes changed; INT8 payload and decoder frozen; no resume state")
    tokenizer = AutoTokenizer.from_pretrained(str(original), use_fast=True, local_files_only=True)
    return _save_final(model, output, meta, expected, audit, provenance, tokenizer)
