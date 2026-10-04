"""Code-only shards on the fixed INT8-protected base, selected by its LR pilot."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.bit_pilot import _json_digest, _trajectory
from experiments.qwen3_w25.bit_recipe_pilot import (
    CODE_FILES as PILOT_CODE_FILES, LR_DIVISORS, SOURCE_CHECKPOINT_ID,
    configuration as pilot_configuration, document_plan, select_recipe, validate_recipe_source,
)
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_bit_recovery import _file_sha256, save_code_patch
from experiments.qwen3_w25.local_bits import _code_banks, _digest, calibrate_local_bits_cpu, optimize_local_bits
from experiments.qwen3_w25.local_recovery import BASE_MODEL, TARGETS, capture_shard_inputs, shard_targets
from experiments.qwen3_w25.local_recovery_finalize import _valid_sha
from experiments.qwen3_w25.pilot_local_decoder import _write_report
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


PROTOCOL_VERSION = 2
CODE_FILES = (*PILOT_CODE_FILES, "experiments/qwen3_w25/local_bit_recovery.py",
              "experiments/qwen3_w25/protected_bit_recovery.py", "experiments/qwen3_w25/recover_protected_bits.py",
              "experiments/qwen3_w25/protected_bit_recovery_finalize.py",
              "experiments/qwen3_w25/local_bit_recovery_finalize.py", "experiments/qwen3_w25/local_recovery_finalize.py",
              "experiments/qwen3_w25/protected_bit_focus.py", "experiments/qwen3_w25/pilot_protected_bit_focus.py",
              "experiments/qwen3_w25/protected_bit_focus_summary.py")

FOCUSED_TARGETS = {"model.layers.9.mlp.gate_proj", "model.layers.27.mlp.down_proj"}
MLP_RECIPES = ((128, 128, 128), (256, 128, 256), (256, 256, 256))


def source_state_sha256(source):
    with (Path(source) / "pytorch_model.bin").open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_pilot_source_model(model, pilot):
    for target, report in pilot["targets"].items():
        if _digest(model.get_submodule(target).state_dict(), set()) != report["source_module_sha256"]:
            raise ValueError(f"Source module bytes differ from the completed pilot: {target}.")


def load_focused_recipe_selection(focused_summary, *, pilot_json, source_dir):
    if focused_summary is None:
        return None
    from experiments.qwen3_w25.protected_bit_focus_summary import load_focused_summary

    path = Path(focused_summary).resolve()
    summary = load_focused_summary(path, pilot_json=pilot_json, source_dir=source_dir)
    return {"summary_path": str(path), "summary_sha256": _file_sha256(path),
            "focused_reports": summary["focused_reports"],
            "selected_attention_recipe": summary["selected_attention_recipe"],
            "selected_mlp_recipe": summary["selected_mlp_recipe"]}


def recovery_configuration(*, lr_divisor, num_shards, focused_selection=None):
    shard_targets(0, num_shards)
    if type(lr_divisor) is not int or lr_divisor not in LR_DIVISORS:
        raise ValueError("LR divisor must be selected from the completed 64/128 pilot.")
    attention = {"lr_divisor": lr_divisor, "steps": 128, "round_steps": 128}
    mlp = dict(attention)
    if focused_selection is not None:
        if (set(focused_selection) != {"summary_path", "summary_sha256", "focused_reports",
                                      "selected_attention_recipe", "selected_mlp_recipe"}
                or not isinstance(focused_selection["summary_path"], str)
                or not Path(focused_selection["summary_path"]).is_absolute()
                or not _valid_sha(focused_selection["summary_sha256"])
                or set(focused_selection["focused_reports"]) != FOCUSED_TARGETS):
            raise ValueError("Focused recipes require their summary and both original report identities.")
        for report in focused_selection["focused_reports"].values():
            if (set(report) != {"path", "sha256"} or not isinstance(report["path"], str)
                    or not Path(report["path"]).is_absolute() or not _valid_sha(report["sha256"])):
                raise ValueError("Invalid focused report path/fingerprint.")
        mlp = focused_selection["selected_mlp_recipe"]
        if (lr_divisor != 128 or focused_selection["selected_attention_recipe"] != attention
                or set(mlp) != set(attention) or any(type(value) is not int for value in mlp.values())
                or tuple(mlp[key] for key in ("lr_divisor", "steps", "round_steps")) not in MLP_RECIPES):
            raise ValueError("The focused experiment fixes attention s128 and one of the three unified MLP recipes.")
    return {
        "source_profile": "no_rotation_int8_input",
        "recipe_mode": "uniform" if focused_selection is None else "attention_mlp",
        "recipes": {"attention": attention, "mlp": dict(mlp)}, "focused_selection": focused_selection,
        "seed": 31, "train_documents": 512, "heldout_documents": 64,
        "selection_documents": 32, "repeated_diagnostic_documents": 32,
        "repeated_diagnostic_used_for_selection": False, "max_length": 256,
        "chunk_size": 256, "num_shards": num_shards,
        "objective": "full_linear_output_mse", "optimizer": "adamw", "weight_decay": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8, "proxy_coordinates": "decoder_sensitivity",
        "active_ratio": 1.0, "streaming": False, "decoder_compute_dtype": "torch.float32",
        "native_activation_dtype": "torch.bfloat16", "teacher_microbatch_size": 1,
        "lr_rule": "per-target initial CPU max(decoder_sensitivity scale)/selected projection-family divisor",
    }


def projection_recipe(configuration, target):
    if target not in TARGETS:
        raise ValueError(f"Expected one of the 252 native projection targets: {target}.")
    group = "attention" if target.split(".")[3] == "self_attn" else "mlp"
    return group, dict(configuration["recipes"][group])


def load_recipe_selection(pilot_json, source_dir, source_meta):
    path, source = Path(pilot_json).resolve(), Path(source_dir).resolve()
    raw = path.read_bytes()
    pilot = json.loads(raw)
    if pilot.get("status") != "complete":
        raise ValueError("Protected recovery requires a completed combined pilot.")
    if (source_meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID
            or pilot.get("source_checkpoint_id") != SOURCE_CHECKPOINT_ID
            or Path(pilot["source_checkpoint_dir"]).resolve() != source):
        raise ValueError("Pilot and worker require the fixed protected source identity/path.")
    if pilot["configuration"] != pilot_configuration() or not pilot["source_bit_budget"]["budget_passed"]:
        raise ValueError("Pilot configuration or actual source budget is invalid.")
    if set(pilot["source_files_sha256"]) != {"checkpoint_meta.json", "config.json"}:
        raise ValueError("Pilot source metadata inventory is invalid.")
    for name, digest in pilot["source_files_sha256"].items():
        if _file_sha256(source / name) != digest:
            raise ValueError(f"Pilot source metadata changed: {name}.")
    repo = Path(__file__).resolve().parents[2]
    if set(pilot["code_sha256"]) != set(PILOT_CODE_FILES):
        raise ValueError("Pilot dependency inventory is incomplete.")
    for name, digest in pilot["code_sha256"].items():
        if _file_sha256(repo / name) != digest:
            raise ValueError(f"Pilot dependency changed: {name}.")
    from experiments.qwen3_w25.protected_bit_recovery_finalize import validate_data_plan
    validate_data_plan(pilot["data"], pilot["document_plan"])
    selected = select_recipe(pilot["targets"], pilot["document_plan"]["heldout_lengths"])
    if any(pilot.get(key) != value for key, value in selected.items()):
        raise ValueError("Pilot recipe does not match the fixed selection-only rule.")
    for target in pilot["targets"].values():
        initial = target["source_module_sha256"]
        if target["final_restored_sha256"] != initial:
            raise ValueError("Pilot source was not restored after its arms.")
        for divisor, arm in target["arms"].items():
            metrics = arm["local_metrics"]
            settings, evidence = metrics["settings"], metrics["evidence"]
            lr = max(row["scale"] for row in target["cpu_coordinate_calibration"]["stages"]) / int(divisor)
            if (arm["lr_divisor"] != int(divisor) or arm["lr"] != settings["lr"] or arm["lr"] != lr
                    or arm["steps"] != 128 or settings["steps"] != 128 or settings["round_steps"] != 128
                    or settings["source_profile"] != "no_rotation_int8_input"
                    or settings["train_tokens"] != pilot["data"]["train"]["valid_tokens"]
                    or settings["heldout_tokens"] != pilot["data"]["heldout"]["valid_tokens"]
                    or arm["restored_sha256_before_arm"] != initial or arm["restored_sha256_after_arm"] != initial
                    or evidence["frozen_sha256_before"] != evidence["frozen_sha256_after"]
                    or not evidence["non_code_bytes_unchanged"] or not evidence["packed_padding_unchanged"]
                    or evidence["persistent_storage_bits_before"] != evidence["persistent_storage_bits_after"]
                    or evidence["final_changed_bits"] <= 0 or len(metrics["training_loss"]) != 128
                    or [row["step"] for row in metrics["step_telemetry"]] != list(range(1, 129))):
                raise ValueError("Pilot has incomplete optimization/frozen-payload evidence.")
    return pilot, hashlib.sha256(raw).hexdigest(), selected["selected_lr_divisor"]


def run_protected_bit_recovery(checkpoint_dir, pilot_json, output_dir, *, shard_index, num_shards,
                               focused_summary=None):
    targets = shard_targets(shard_index, num_shards)
    source, pilot_path, output = map(lambda p: Path(p).resolve(), (checkpoint_dir, pilot_json, output_dir))
    if source == output or source in output.parents:
        raise ValueError("Worker output must be outside its immutable source checkpoint.")
    meta, audit = _validate_checkpoint(source)
    if not audit["budget_passed"]:
        raise ValueError("Source exceeds the rank-zero actual bit budget.")
    pilot, pilot_sha, divisor = load_recipe_selection(pilot_path, source, meta)
    focused = load_focused_recipe_selection(focused_summary, pilot_json=pilot_path, source_dir=source)
    config = recovery_configuration(lr_divisor=divisor, num_shards=num_shards, focused_selection=focused)
    output.mkdir(parents=True, exist_ok=False)
    for child in ("patches", "reports"):
        (output / child).mkdir()
    path, device, started = output / "worker.json", torch.device("cuda:0"), time.perf_counter()
    repo = Path(__file__).resolve().parents[2]
    common = {"protocol_version": PROTOCOL_VERSION, "source_checkpoint_dir": str(source),
              "source_checkpoint_id": meta["checkpoint_id"], "pilot_json": str(pilot_path),
              "pilot_sha256": pilot_sha, "selected_lr_divisor": divisor, "configuration": config,
              "configuration_sha256": _json_digest(config), "source_files_sha256": pilot["source_files_sha256"],
              "source_state_sha256": source_state_sha256(source),
              "code_sha256": {name: _file_sha256(repo / name) for name in CODE_FILES}}
    report = {**common, "status": "preparing", "shard_index": shard_index,
              "targets": list(targets), "modules": {}, "source_bit_budget": audit}
    _write_report(path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train, heldout, data = build_calibration_batches(tokenizer, seed=31, train_documents=512,
                                                       heldout_documents=64, max_length=256)
        plan = document_plan(train, heldout, data)
        if data != pilot["data"] or plan != pilot["document_plan"]:
            raise ValueError("Worker calibration documents differ from the completed pilot.")
        common["manifest_sha256"] = plan["manifest_sha256"]
        report.update(data=data, document_plan=plan, manifest_sha256=plan["manifest_sha256"])
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL,
                                                        map_location="cpu", strict=True)
        model.eval().requires_grad_(False)
        validate_recipe_source(model, loaded_meta)
        validate_pilot_source_model(model, pilot)
        phase = time.perf_counter()
        print(json.dumps({"status": "capturing", "targets": len(targets), "recipe_mode": config["recipe_mode"],
                          "recipes": config["recipes"]}), flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundles = capture_shard_inputs(teacher, targets, train, heldout)
        torch.cuda.synchronize(device)
        report.update(teacher_capture_seconds=time.perf_counter() - phase,
                      teacher_capture_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      teacher_cache_cpu_bytes=sum(t.numel() * t.element_size() for b in bundles.values() for t in b.values()))
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        report["status"] = "running"
        _write_report(path, report)
        specs = {item["name"]: item for item in loaded_meta["converted_modules"]}
        for target in targets:
            bundle, layer = bundles.pop(target), model.get_submodule(target)
            for key, split in (("train_inputs", "train"), ("heldout_inputs", "heldout")):
                if len(bundle[key]) != data[split]["valid_tokens"] or bundle[key].dtype != torch.bfloat16:
                    raise ValueError(f"Captured token inventory/dtype differs: {target}/{key}.")
            source_sha = _digest(layer.state_dict(), set())
            calibration = calibrate_local_bits_cpu(layer)
            group, recipe = projection_recipe(config, target)
            lr = max(item["scale"] for item in calibration["stages"]) / recipe["lr_divisor"]
            source_codes = {key: layer.state_dict()[key].clone() for key, _ in _code_banks(layer)}
            layer.to(device)
            phase = time.perf_counter()

            def progress(row):
                if row["step"] % 10 == 0 or row["step"] == recipe["steps"]:
                    print(json.dumps({"target": target, "shard_index": shard_index, **row}), flush=True)

            metrics = optimize_local_bits(layer, bundle["reference_weight"], bundle["train_inputs"],
                bundle["heldout_inputs"], steps=recipe["steps"], lr=lr, round_steps=recipe["round_steps"], chunk_size=256,
                seed=31, progress_callback=progress)
            patch_file, report_file = Path("patches") / (target + ".pt"), Path("reports") / (target + ".json")
            patch = save_code_patch(layer, source_codes, output / patch_file)
            if patch["stage_changed_bits"] != metrics["evidence"]["stage_changed_bits"]:
                raise RuntimeError("Packed patch XOR differs from optimizer evidence.")
            metrics.update(**common, target=target, source_module_sha256=source_sha,
                source_spec_sha256=_json_digest(specs[target]), cpu_coordinate_calibration=calibration,
                recipe_group=group, applied_recipe=recipe,
                lr=lr, trajectory=_trajectory(metrics), worker_module_elapsed_seconds=time.perf_counter() - phase, **patch)
            _write_report(output / report_file, metrics)
            report["modules"][target] = {"patch_file": str(patch_file), "report_file": str(report_file),
                "patch_sha256": _file_sha256(output / patch_file), "report_sha256": _file_sha256(output / report_file)}
            _write_report(path, report)
            layer.cpu()
            layer._build_parallel_stage_decode_plan()
            layer.clear_decoded_weight_cache()
            del bundle, source_codes
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        _write_report(path, report)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(path, report)
        raise
