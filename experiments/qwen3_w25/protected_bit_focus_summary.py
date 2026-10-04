"""Select one MLP recipe by the predeclared equal-gate/down selection rule."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments.qwen3_w25.bit_recipe_pilot import SOURCE_CHECKPOINT_ID, _ratio, configuration
from experiments.qwen3_w25.protected_bit_focus import CODE_FILES as FOCUS_CODE_FILES, CONTROL, TARGETS, focus_configuration, prefix_comparison
from experiments.qwen3_w25.pilot_local_decoder import _write_report


RECIPES = {
    CONTROL: {"lr_divisor": 128, "steps": 128, "round_steps": 128},
    "s256_steps128_round256": {"lr_divisor": 256, "steps": 128, "round_steps": 256},
    "s256_steps256_round256": {"lr_divisor": 256, "steps": 256, "round_steps": 256},
}


def select_unified_mlp_recipe(target_reports, heldout_lengths):
    if set(target_reports) != set(TARGETS.values()) or len(heldout_lengths) != 64:
        raise ValueError("The unified MLP selection requires exactly gate/down and their 64 held-out lengths.")
    if any(set(report["arms"]) != set(RECIPES) for report in target_reports.values()):
        raise ValueError("Each MLP target must contain all three complete recipes.")
    scores = {}
    for name in RECIPES:
        ratios = {target: _ratio(report["arms"][name]["native_selection"], heldout_lengths[:32])
                  for target, report in target_reports.items()}
        scores[name] = {"mean_selection_ratio": sum(ratios.values()) / 2, "per_target_ratios": ratios}
    selected = min(RECIPES, key=lambda name: (scores[name]["mean_selection_ratio"],
                                             RECIPES[name]["steps"], -RECIPES[name]["lr_divisor"]))
    diagnostic = {target: _ratio(report["arms"][selected]["native_repeated_diagnostic"], heldout_lengths[32:])
                  for target, report in target_reports.items()}
    return {"selected_mlp_arm": selected, "selected_mlp_recipe": RECIPES[selected],
            "mlp_selection_scores": scores,
            "selected_mlp_repeated_diagnostic": {"mean_ratio": sum(diagnostic.values()) / 2,
                "per_target_ratios": diagnostic, "used_for_selection": False, "new_blind_confirmation": False}}


def build_protected_bit_focus_summary(pilot_json, focused_dirs):
    pilot_path = Path(pilot_json).resolve()
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot_sha = hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    if (pilot["status"] != "complete" or pilot["source_checkpoint_id"] != SOURCE_CHECKPOINT_ID
            or pilot["configuration"] != configuration() or not pilot["source_bit_budget"]["budget_passed"]):
        raise ValueError("Expected the completed original protected-source LR pilot.")
    if len(focused_dirs) != 2:
        raise ValueError("Exactly two completed focused target directories are required.")
    reports, references = {}, {}
    repo = Path(__file__).resolve().parents[2]
    for directory in focused_dirs:
        path = Path(directory).resolve() / "pilot.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        target = report["configuration"]["target"]
        if target in reports or target not in TARGETS.values():
            raise ValueError("Duplicate or unexpected focused MLP target.")
        original = pilot["targets"][target]
        if (report["status"] != "complete" or report["configuration"] != focus_configuration(target)
                or report["source_checkpoint_id"] != SOURCE_CHECKPOINT_ID
                or report["source_checkpoint_dir"] != pilot["source_checkpoint_dir"]
                or report["source_bit_budget"] != pilot["source_bit_budget"]
                or report["pilot_sha256"] != pilot_sha or Path(report["pilot_json"]).resolve() != pilot_path
                or report["data"] != pilot["data"] or report["document_plan"] != pilot["document_plan"]
                or report["source_module_sha256"] != original["source_module_sha256"]
                or report["final_restored_sha256"] != original["source_module_sha256"]
                or report["cpu_coordinate_calibration"] != original["cpu_coordinate_calibration"]
                or set(report["code_sha256"]) != set(FOCUS_CODE_FILES)
                or any(report["code_sha256"].get(name) != value for name, value in pilot["code_sha256"].items())
                or any(hashlib.sha256((repo / name).read_bytes()).hexdigest() != value
                       for name, value in report["code_sha256"].items())):
            raise ValueError("Incomplete focused report or changed source, original control, calibration or full-state restoration.")
        if {k: v for k, v in report["arms"][CONTROL].items() if k != "reused_from_completed_pilot"} != original["arms"]["128"]:
            raise ValueError("The reused control differs from the immutable completed pilot.")
        for arm, recipe in RECIPES.items():
            arm_report = report["arms"][arm]
            metrics = arm_report["local_metrics"]
            settings, evidence = metrics["settings"], metrics["evidence"]
            expected_lr = max(row["scale"] for row in report["cpu_coordinate_calibration"]["stages"]) / recipe["lr_divisor"]
            if (settings["steps"] != recipe["steps"] or settings["round_steps"] != recipe["round_steps"]
                    or settings["lr"] != expected_lr or settings["source_profile"] != "no_rotation_int8_input"
                    or arm_report["lr"] != expected_lr or arm_report["steps"] != recipe["steps"]
                    or (arm != CONTROL and arm_report["round_steps"] != recipe["round_steps"])
                    or settings["stable_steps"] != 10 or not settings["early_stable_round_end"]
                    or settings["train_tokens"] != pilot["data"]["train"]["valid_tokens"]
                    or settings["heldout_tokens"] != pilot["data"]["heldout"]["valid_tokens"]
                    or len(metrics["training_loss"]) != recipe["steps"]
                    or any(not math.isfinite(value) or value < 0 for value in metrics["training_loss"])
                    or [row["step"] for row in metrics["step_telemetry"]] != list(range(1, recipe["steps"] + 1))
                    or evidence["frozen_sha256_before"] != evidence["frozen_sha256_after"]
                    or not evidence["non_code_bytes_unchanged"] or not evidence["packed_padding_unchanged"]
                    or not all(evidence[key] for key in ("tensor_schema_unchanged", "parameter_ids_unchanged",
                                                        "decoder_compute_parameters_unchanged", "coordinate_scales_fixed"))
                    or evidence["persistent_storage_bits_before"] != evidence["persistent_storage_bits_after"]
                    or evidence["final_changed_bits"] <= 0
                    or report["arms"][arm]["restored_sha256_before_arm"] != original["source_module_sha256"]
                    or report["arms"][arm]["restored_sha256_after_arm"] != original["source_module_sha256"]):
                raise ValueError("Focused arm lacks the required recipe, token coverage or frozen-payload evidence.")
        expected_prefix = prefix_comparison(report["arms"]["s256_steps128_round256"]["local_metrics"],
                                            report["arms"]["s256_steps256_round256"]["local_metrics"])
        if report["prefix_comparison"] != expected_prefix:
            raise ValueError("Stored 128-step prefix comparison disagrees with actual trajectories.")
        reports[target] = report
        references[target] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if len({json.dumps(report["code_sha256"], sort_keys=True) for report in reports.values()}) != 1:
        raise ValueError("Focused targets used different implementation versions.")
    result = {
        "status": "complete", "purpose": "two_training_recipe_selection_not_downstream_accuracy",
        "selection_rule": "one MLP recipe minimizing equal-gate/down selection ratios; tie fewer steps then lower LR; no per-target optimum stitching",
        "source_checkpoint_dir": pilot["source_checkpoint_dir"], "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "source_bit_budget": pilot["source_bit_budget"], "pilot_json": str(pilot_path), "pilot_sha256": pilot_sha,
        "focused_reports": references, "source_code_sha256": next(iter(reports.values()))["code_sha256"],
        "summary_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data": pilot["data"], "document_plan": pilot["document_plan"],
        "selected_attention_recipe": {"lr_divisor": 128, "steps": 128, "round_steps": 128},
        "attention_selection_ratios": {target: pilot["targets"][target]["arms"]["128"]["native_selection"]["after_before_ratio"]
                                        for target in ("model.layers.3.self_attn.v_proj", "model.layers.18.self_attn.o_proj")},
        "prefix_comparisons": {target: report["prefix_comparison"] for target, report in reports.items()},
        **select_unified_mlp_recipe(reports, pilot["document_plan"]["heldout_lengths"]),
    }
    return result


def summarize_protected_bit_focus(pilot_json, focused_dirs, output_path):
    result = build_protected_bit_focus_summary(pilot_json, focused_dirs)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    _write_report(output, result)
    return result


def load_focused_summary(summary_path, *, pilot_json, source_dir):
    """Read-only reconstruction from both original reports, including every SHA."""
    path = Path(summary_path).resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (summary["status"] != "complete" or summary["source_checkpoint_id"] != SOURCE_CHECKPOINT_ID
            or Path(summary["source_checkpoint_dir"]).resolve() != Path(source_dir).resolve()
            or Path(summary["pilot_json"]).resolve() != Path(pilot_json).resolve()
            or set(summary["focused_reports"]) != set(TARGETS.values())):
        raise ValueError("Focused summary has the wrong source, pilot or target coverage.")
    directories = []
    for target in TARGETS.values():
        reference = summary["focused_reports"][target]
        report_path = Path(reference["path"])
        if (report_path.name != "pilot.json"
                or hashlib.sha256(report_path.read_bytes()).hexdigest() != reference["sha256"]):
            raise ValueError("A focused source report digest changed.")
        directories.append(report_path.parent)
    actual = build_protected_bit_focus_summary(pilot_json, directories)
    if summary != actual:
        raise ValueError("Focused summary disagrees with its validated source reports or implementation version.")
    return actual
