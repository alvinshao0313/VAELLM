"""Four independent teacher-input shards of the fixed BEST joint recovery candidate."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.bit_pilot import _json_digest
from experiments.qwen3_w25.bit_recipe_pilot import SOURCE_CHECKPOINT_ID, document_plan, validate_recipe_source
from experiments.qwen3_w25.block_batching import batch_block_documents
from experiments.qwen3_w25.block_calibration import capture_block_batches
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.joint_block_validation import (
    CODE_FILES as VALIDATION_CODE_FILES, SOURCE_STATE_SHA256, _release_block,
    audit_joint_state, bit_configuration, collect_state_inventory, configuration,
    require_native_runtime, validate_core_evidence,
)
from experiments.qwen3_w25.local_bits import calibrate_local_bits_cpu
from experiments.qwen3_w25.local_block import PROJECTIONS
from experiments.qwen3_w25.local_decoder import _storage_bits
from experiments.qwen3_w25.local_recovery import BASE_MODEL
from experiments.qwen3_w25.protected_bit_recovery import (
    _file_sha256, load_recipe_selection, source_state_sha256, validate_pilot_source_model,
)
from experiments.qwen3_w25.protected_bit_recovery_finalize import validate_data_plan
from experiments.qwen3_w25.sequential_recovery import select_calibration_prefix
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.checkpoint_v6 import _atomic_torch_save, _atomic_write_json
from train_utils.v6_model_loader import load_v6_model_checkpoint


PROTOCOL_VERSION = 1
SELECTION_REPORTS = {str(count): f"result/qwen3_8b_w25_20260930/validation_joint_block16_n{count}_s128/run/validation.json"
                     for count in (64, 512)}
CODE_FILES = tuple(dict.fromkeys((*VALIDATION_CODE_FILES,
    "experiments/qwen3_w25/joint_block_recovery.py",
    "experiments/qwen3_w25/joint_block_recovery_finalize.py",
    "experiments/qwen3_w25/recover_joint_blocks.py")))
COMMON_FIELDS = ("protocol_version", "source_checkpoint_dir", "source_checkpoint_id", "source_state_sha256",
    "source_files_sha256", "source_bit_budget", "pilot_json", "pilot_sha256", "configuration",
    "configuration_sha256", "code_sha256", "manifest_sha256", "selected_document_plan", "selection_evidence")


def recovery_configuration():
    result = configuration(train_documents=512)
    del result["block_index"]
    result.update(num_shards=4, layer_assignment="layer_index % 4", block_indices=list(range(36)),
        checkpoint_policy="per-block existing decoder-and-packed-code patches; final inference only",
        recipe_status="512 training documents selected by the L16 common-calibration-dev 64-vs-512 comparison; unexecuted full36 64-document recipe superseded; full-model quality unverified",
        heldout_role="calibration_dev", heldout_used_for_selection=True,
        heldout_selection_usage="one-time L16 training-document-count choice only",
        heldout_used_for_gradients=False, heldout_used_for_early_stopping=False,
        heldout_used_for_per_block_selection=False,
        downstream_evaluation_role="independent eight-task evaluation; not used for this document-count choice",
        selection_evidence=SELECTION_REPORTS.copy(),
        comparison_boundary="independent original teacher inputs; not rolling-student sequential recovery")
    return result


def shard_layers(index):
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError("shard_index must be an integer in [0, 3]; this candidate fixes four shards.")
    return list(range(index, 36, 4))


def selected_plan_from_manifest(data, full_plan):
    validate_data_plan(data, full_plan)
    return {"manifest_sha256": _json_digest(data), "train_document_indices": list(range(512)),
        "train_documents_sha256": _json_digest(data["train"]["documents"]),
        "train_tokens": sum(full_plan["train_lengths"]),
        "heldout_document_indices": list(range(512, 576)),
        "heldout_documents_sha256": _json_digest(data["heldout"]["documents"]),
        "heldout_tokens": sum(full_plan["heldout_lengths"]),
        "train_lengths": full_plan["train_lengths"], "heldout_lengths": full_plan["heldout_lengths"]}


def load_training_document_selection(source, pilot, pilot_sha):
    """Bind the fixed 512 choice to its completed, paired calibration-dev evidence."""
    repo, records, reports = Path(__file__).resolve().parents[2], {}, {}
    plan = selected_plan_from_manifest(pilot["data"], pilot["document_plan"])
    for label, relative in SELECTION_REPORTS.items():
        count, path = int(label), repo / relative
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("status") != "complete" or report["source_checkpoint_id"] != SOURCE_CHECKPOINT_ID
                or report["source_state_sha256"] != SOURCE_STATE_SHA256
                or Path(report["source_checkpoint_dir"]).resolve() != Path(source).resolve()
                or report["pilot_sha256"] != pilot_sha or report["data"] != pilot["data"]
                or report["configuration"] != configuration(train_documents=count)
                or report["independent_reload_output_exact"] is not True
                or report["serialized_state_audit"] != report["in_memory_state_audit"]
                or report["source_bit_budget"] != pilot["source_bit_budget"]
                or report["bit_budget"]["budget_passed"] is not True
                or any(report["bit_budget"][key] != pilot["source_bit_budget"][key] for key in
                       ("original_linear_weights", "user_linear_bits", "packed_linear_bits",
                        "code_padding_bits", "full_model_tensor_bytes"))):
            raise ValueError("Document-count selection report is incomplete or differs in source/data/configuration/storage.")
        expected = {**plan, "train_document_indices": list(range(count)),
            "train_documents_sha256": _json_digest(pilot["data"]["train"]["documents"][:count]),
            "train_tokens": sum(plan["train_lengths"][:count]), "train_lengths": plan["train_lengths"][:count]}
        if report["selected_document_plan"] != expected:
            raise ValueError("Document-count selection changed the shared calibration-dev or training prefix.")
        # The older 64-document entrypoint predated its explicit 512 option;
        # the joint core and every other dependency must still match.
        historical_entrypoints = {"experiments/qwen3_w25/joint_block_validation.py",
                                  "experiments/qwen3_w25/validate_joint_block.py"} if count == 64 else set()
        if (set(report["code_sha256"]) != set(VALIDATION_CODE_FILES)
                or any(value != _file_sha256(repo / name) for name, value in report["code_sha256"].items()
                       if name not in historical_entrypoints)):
            raise ValueError("Document-count selection used different joint core code.")
        validate_joint_trajectory(report["local_metrics"], report["state_inventory"], report["in_memory_state_audit"],
            report["bit_lrs"], report["round_steps"], expected, prefix="model.layers.16")
        records[label] = {"path": str(path), "sha256": _file_sha256(path),
            "before_native": report["local_metrics"]["before_native"],
            "after_native": report["local_metrics"]["after_native"]}
        reports[label] = report
    if (records["64"]["before_native"]["heldout_mse"] != records["512"]["before_native"]["heldout_mse"]
            or reports["64"]["bit_lrs"] != reports["512"]["bit_lrs"]
            or reports["64"]["round_steps"] != reports["512"]["round_steps"]
            or records["512"]["after_native"]["heldout_mse"] >= records["64"]["after_native"]["heldout_mse"]):
        raise ValueError("The paired calibration-dev comparison does not support the fixed 512-document choice.")
    return {"selected_train_documents": 512, "selection_block_index": 16, "reports": records,
        "calibration_dev_documents_sha256": plan["heldout_documents_sha256"],
        "usage": "one-time sample-count selection; no gradients, early stopping or per-block selection"}


def block_state(state, prefix):
    result = {key: value for key, value in state.items() if key.startswith(prefix + ".")}
    if not result:
        raise ValueError(f"Missing source block: {prefix}.")
    return result


def validate_joint_trajectory(result, inventory, audit, rates, rounds, plan, *, prefix):
    validate_core_evidence(result, inventory, audit, rates, rounds, plan, prefix=prefix)
    expected = {"seed": 31, "objective": "block_output_mse", "input_source": "caller_supplied_fixed_hidden_states",
        "decoder_optimizer": "AdamW", "bit_optimizer": "adamw", "weight_decay": 0.0, "active_ratio": 1.0,
        "streaming": False, "proxy_coordinates": "decoder_sensitivity", "early_stable_round_end": True,
        "independent_projection_rounds": True, "decoder_compute_dtype": "torch.float32",
        "projection_dtype": "torch.bfloat16", "native_decode_dtype": "torch.bfloat16",
        "weight_gradient_dtype": "torch.float32", "gradient_path": "fp32_weight_gradient_then_joint_decoder_and_bit_vjp",
        "optimizer_update_order": "decoder_then_projection_bit_managers", "cross_block_gradients": False,
        "storage_dtypes": dict.fromkeys(PROJECTIONS, "torch.bfloat16"),
        "source_profiles": dict.fromkeys(PROJECTIONS, "no_rotation_int8_input")}
    if any(result["settings"].get(key) != value for key, value in expected.items()):
        raise ValueError("Joint core recipe/precision/input contract differs from the fixed candidate.")
    for phase in ("before_native", "before_fp32_decode", "after_fp32_decode", "after_native"):
        metrics = result[phase]
        if set(metrics) != {"train_mse", "heldout_mse"} or any(not math.isfinite(v) or v < 0 for v in metrics.values()):
            raise ValueError(f"Invalid joint endpoint: {phase}.")
    total = 0
    for name in PROJECTIONS:
        metrics = result["modules"][name]
        index = step = cumulative = stable = all_flips = ended_count = 0
        had_flip = False
        maxima = {key: [0.0, 0.0] for key in ("stage_proxy_gradient_l2", "stage_decoder_gradient_l2")}
        for row in metrics["step_telemetry"]:
            flips = row["step_flip_count"]
            if type(flips) is not int or flips < 0:
                raise ValueError("Invalid joint step flip count.")
            step, cumulative, all_flips = step + 1, cumulative + flips, all_flips + flips
            if flips:
                had_flip, stable = True, 0
            elif had_flip:
                stable += 1
            ended = step >= rounds[name] or (had_flip and stable >= 10)
            expected_row = {"global_bit_round": index, "bit_round_step": step, "cumulative_flip_count": cumulative,
                "stable_counter": stable, "stable_steps": 10, "had_flip": had_flip, "round_ended": ended}
            if any(row[key] != value for key, value in expected_row.items()):
                raise ValueError(f"Joint independent-round trajectory mismatch: {name}.")
            for key in maxima:
                values = row[key]
                if len(values) != 2 or any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError("Invalid two-stage joint gradient trajectory.")
                maxima[key] = [max(a, b) for a, b in zip(maxima[key], values)]
            if ended:
                index, ended_count = index + 1, ended_count + 1
                step = cumulative = stable = 0
                had_flip = False
        if (metrics["rounds_ended"] != ended_count or metrics["cumulative_step_flips"] != all_flips
                or metrics["stable_steps"] != 10 or metrics["final_changed_bits"] != sum(metrics["stage_changed_bits"])
                or any(metrics[key + "_max"] != value for key, value in maxima.items())):
            raise ValueError(f"Joint trajectory summary mismatch: {name}.")
        total += metrics["final_changed_bits"]
    if result["evidence"]["final_changed_bits"] != total or total != audit["total_changed_code_bits"]:
        raise ValueError("Joint total code change differs from the actual patch.")


def run_joint_block_recovery(checkpoint_dir, pilot_json, output_dir, *, shard_index):
    from experiments.qwen3_w25.joint_block import optimize_joint_block

    layers = shard_layers(shard_index)
    source, pilot_path, output = (Path(p).resolve() for p in (checkpoint_dir, pilot_json, output_dir))
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Worker output must be new and outside the immutable source.")
    meta, source_audit = _validate_checkpoint(source)
    source_sha = source_state_sha256(source)
    if meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID or source_sha != SOURCE_STATE_SHA256 or not source_audit["budget_passed"]:
        raise ValueError("Joint recovery requires the unchanged rank-zero BEST within 2.5 bpw.")
    pilot, pilot_sha, _ = load_recipe_selection(pilot_path, source, meta)
    config, repo = recovery_configuration(), Path(__file__).resolve().parents[2]
    plan = selected_plan_from_manifest(pilot["data"], pilot["document_plan"])
    selection = load_training_document_selection(source, pilot, pilot_sha)
    common = {"protocol_version": PROTOCOL_VERSION, "source_checkpoint_dir": str(source),
        "source_checkpoint_id": meta["checkpoint_id"], "source_state_sha256": source_sha,
        "source_files_sha256": pilot["source_files_sha256"], "source_bit_budget": source_audit,
        "pilot_json": str(pilot_path), "pilot_sha256": pilot_sha, "configuration": config,
        "configuration_sha256": _json_digest(config), "selected_document_plan": plan, "selection_evidence": selection,
        "manifest_sha256": plan["manifest_sha256"], "code_sha256": {name: _file_sha256(repo / name) for name in CODE_FILES}}
    worker = {**common, "status": "preparing", "shard_index": shard_index, "layers": layers, "blocks": {},
        "data": pilot["data"], "document_plan": pilot["document_plan"],
        "invocation": {"python": sys.executable, "entrypoint": "experiments/qwen3_w25/recover_joint_blocks.py",
            "command": "worker", "checkpoint_dir": str(source), "pilot_json": str(pilot_path),
            "output_dir": str(output), "shard_index": shard_index, "resolved_train_documents": 512,
            "resolved_calibration_dev_documents": 64}}
    output.mkdir(parents=True)
    path, device = output / "worker.json", torch.device("cuda:0")
    _atomic_write_json(worker, str(path))
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        full_train, heldout_batch, data = build_calibration_batches(tokenizer, seed=31,
            train_documents=512, heldout_documents=64, max_length=256)
        if data != pilot["data"] or document_plan(full_train, heldout_batch, data) != pilot["document_plan"]:
            raise ValueError("Joint worker changed the fixed 512+64 document identity.")
        train_batch, heldout_batch, actual_plan = select_calibration_prefix(full_train, heldout_batch, data, 512)
        if actual_plan != plan:
            raise ValueError("Joint worker selected a different training/held-out prefix.")
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL,
            map_location="cpu", strict=True, expected_kind="final_model")
        model.eval().requires_grad_(False)
        validate_recipe_source(model, loaded_meta)
        validate_pilot_source_model(model, pilot)
        source_state = _load_state(source)
        _assert_state_equal(model.state_dict(), source_state, "Unmodified joint shard source")
        expected, storage_before = dict(source_state), _storage_bits(model)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        captured = {}
        for index in layers:
            captured[index] = (capture_block_batches(teacher, index, train_batch),
                               capture_block_batches(teacher, index, heldout_batch))
            print(json.dumps({"phase": "teacher_capture", "block_index": index}), flush=True)
        torch.cuda.synchronize(device)
        worker["teacher_capture"] = {"seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "attention_implementation": teacher.config._attn_implementation, "captured_blocks": layers,
            "train_documents_per_block": 512, "calibration_dev_documents_per_block": 64,
            "cpu_hidden_and_target_bytes": sum(t.numel() * t.element_size()
                for splits in captured.values() for batches in splits for batch in batches
                for t in (batch.hidden_states, batch.target_hidden_states))}
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        worker["status"] = "training"
        _atomic_write_json(worker, str(path))
        for index in layers:
            prefix = f"model.layers.{index}"
            block = model.get_submodule(prefix)
            inventory = collect_state_inventory(block, prefix=prefix)
            calibrations = {name: calibrate_local_bits_cpu(block.get_submodule(name)) for name in PROJECTIONS}
            rates, rounds = bit_configuration(calibrations)
            train, heldout = captured.pop(index)
            train_execution = batch_block_documents(train, microbatch_size=8)
            heldout_execution = batch_block_documents(heldout, microbatch_size=8)
            block.to(device)
            try:
                result = optimize_joint_block(block, train_execution, heldout_execution, steps=128,
                    decoder_lr=1.25e-5, bit_lrs=rates, round_steps=rounds, seed=31,
                    progress_callback=lambda row: print(json.dumps({"block_index": index, **row}), flush=True))
                require_native_runtime(block)
            finally:
                _release_block(block)
                torch.cuda.empty_cache()
            actual = {prefix + "." + key: value for key, value in block.state_dict().items()}
            audit = audit_joint_state(block_state(source_state, prefix), actual, inventory)
            validate_joint_trajectory(result, inventory, audit, rates, rounds, plan, prefix=prefix)
            patch = {key: actual[key].detach().cpu().clone() for key in inventory["allowed_state_keys"]}
            report = {**common, "status": "complete", "block_index": index, "prefix": prefix,
                "state_inventory": inventory, "cpu_coordinate_calibration": calibrations,
                "bit_lrs": rates, "round_steps": rounds, "local_metrics": result, "state_audit": audit,
                "execution_batch_sizes": {"train": [b.hidden_states.shape[0] for b in train_execution],
                    "heldout": [b.hidden_states.shape[0] for b in heldout_execution]}}
            patch_path, report_path = output / f"block_{index:02d}.pt", output / f"block_{index:02d}.json"
            _atomic_torch_save(patch, str(patch_path))
            _atomic_write_json(report, str(report_path))
            worker["blocks"][str(index)] = {"patch_file": patch_path.name, "report_file": report_path.name,
                "patch_sha256": _file_sha256(patch_path), "report_sha256": _file_sha256(report_path)}
            expected.update(patch)
            _atomic_write_json(worker, str(path))
            del train, heldout, train_execution, heldout_execution, patch
            gc.collect()
        compared = _assert_state_equal(model.state_dict(), expected, "Whole joint shard including all frozen source state")
        if _storage_bits(model) != storage_before:
            raise RuntimeError("Joint shard changed persistent tensor storage.")
        worker.update(status="complete", complete_model_tensors_compared=compared,
            persistent_storage_bits_before=storage_before, persistent_storage_bits_after=_storage_bits(model))
        _atomic_write_json(worker, str(path))
        return worker
    except Exception as exc:
        worker.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _atomic_write_json(worker, str(path))
        raise
