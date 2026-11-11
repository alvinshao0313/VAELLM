"""Explicit mathematical rule fixtures and actual zero-KD native CPU export."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
from experiments.cat_auto_protocol_cmcc_20261104.validate_recovery_cpu import tiny_native_model, CATEGORIES
from experiments.cat_auto_protocol_cmcc_20261104.vae_muon import tensor_digest
from train_utils import checkpoint_v6 as v6
from train_utils.cat_runtime_state_v6 import build_cat_runtime_state
from train_utils.cat_train_runtime import load_cat_resume_distill_progress
from .boundary import _pins, complete_boundary_reference, export_zero_kd_boundary
from .selection import FullEightScore, NodeScore, RecoveryQuota, RecoverySettings, SelectedCategory, TrajectoryResult


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError("Expected rejection: " + function.__name__)


def score(value, source):
    return FullEightScore({"fixture_task_" + str(index): value for index in range(8)},
        "explicit_math_fixture_" + source, dict(checkpoint_id="fixture_deployment_" + source,
            source_checkpoint_id=source, protocol_id="explicit_math_fixture_only",
            evaluation_digest=digest(dict(source=source, value=value)), leaf_count=8, sample_count=8,
            expected_leaf_count=8, expected_sample_count=8, phase="cpu_fixture"))


def trajectory(category="a", *, raw="raw_a", anchor=True, values=(.6, .6, .6, .6),
        raw_value=.5, settings=None):
    settings = settings or RecoverySettings()
    identity = f"{category}_{raw}_{settings.scope}_{settings.decoder_lr}_{settings.schedule_max}"
    return TrajectoryResult(category, "live_parent", raw, "recipe_" + raw, "candidate_" + raw, identity,
        anchor, 100, 120, settings, score(raw_value, raw), tuple(NodeScore(step,
            score(value, f"{identity}_{step}"), f"{identity}_{step}")
            for step, value in zip(settings.node_steps, values, strict=True)))


def rule_checks():
    quota = RecoveryQuota(["a", "b", "c", "d", "e", "f"])
    selected = []
    for category in quota.categories:
        values = (.51, .52, .53, .54) if category in ("a", "b", "e") else (.7, .66, .65, .64)
        raw_value = .8 if category in ("d", "f") else .5
        if category == "d":
            values = (.6,) * 4
        rows = [trajectory(category, raw=category + "_anchor", values=values, raw_value=raw_value)]
        if category not in ("a", "f"):
            rows.append(trajectory(category, raw=category + "_different", anchor=False,
                values=(.4,) * 4, raw_value=.9 if category == "c" else .3))
        for row in rows:
            quota.register_start(category_id=category, raw_id=row.raw_id, settings=row.settings, kind="baseline")
        prefix = ["current"] if category == "f" else ["old", "current"]
        arm = quota.decide(rows, prefix_targets=prefix, current_targets=["current"], future_dense_targets=["future"])
        if category in ("a", "b"):
            require(arm.settings.schedule_max == 1000 and not arm.interaction_eligible, "Long slots replace the third arm without interaction allocation.")
        elif category == "c":
            require(arm.settings.decoder_lr == .5e-5 and arm.raw_id == "c_anchor" and arm.interaction_eligible,
                "The trigger reads its own raw0, never the other raw's higher score.")
        elif category == "d":
            require(arm.settings.scope == "remaining_lora_current_decoder" and arm.interaction_eligible,
                "The second actually distinct 500-step variant gets the second interaction slot.")
        elif category == "e":
            require(arm.settings.decoder_lr == 2e-5 and not arm.interaction_eligible
                and rows[0].fixed_train_readout_id is None and all(node.fixed_train_kd is None for node in rows[0].nodes),
                "Double-LR nomination cannot require fixed-train KD values.")
        else:
            require(arm is None, "Equivalent first-class scope leaves the third slot empty without switching to LR.")
        if arm is not None:
            rejects(quota.select_completed_category, rows)
            by_raw = {row.raw_id: row for row in rows}
            extras = [(arm.raw_id, "third")]
            if arm.interaction_eligible:
                extras.append((quota.interaction_raw(arm, rows), "interaction"))
            for raw, kind in extras:
                quota.register_start(category_id=category, raw_id=raw, settings=arm.settings, kind=kind)
                original = by_raw[raw]
                rows.append(trajectory(category, raw=raw, anchor=original.is_anchor,
                    values=(.3,) * len(arm.settings.node_steps), raw_value=float(original.raw_score.mean), settings=arm.settings))
            if category == "c":
                rejects(quota.select_completed_category, rows[:-1])
                wrong = replace(rows[-1], raw_score=score(.8, rows[-1].raw_id))
                rejects(quota.select_completed_category, [*rows[:-1], wrong])
        winner, node, certificate = quota.select_completed_category(rows)
        selected.append((category, node.step, certificate["selected_kind"]))
        if category == "c":
            require(node.step == 0 and winner.raw_id == "c_different" and certificate["raw_zero_strictly_above_all_recovery"],
                "Only global raw0 strictly above all completed recovery nodes may win.")
    require(quota.long_categories == ["a", "b"] and quota.interaction_categories == ["c", "d"],
        "Long and interaction slots have separate counters and use the first qualifying categories.")
    require(quota.counts()["actual_started"] <= quota.counts()["max_trajectories"], "Keep the global 3C+2 limit.")
    tie = trajectory(raw_value=.6)
    tie_quota = RecoveryQuota(["a"])
    require(tie_quota.decide([tie], prefix_targets=["current"], current_targets=["current"], future_dense_targets=[]) is None,
        "Raw/node ties enter scope nomination, including equivalent empty scope.")
    _winner, node, _certificate = tie_quota.select_completed_category([tie])
    require(node.step == 500, "Exact raw0 ties retain the later recovery node.")
    rejects(replace, tie, nodes=tie.nodes[:-1])
    rejects(FullEightScore, {"partial": .5}, "partial", {})
    foreign = replace(tie.raw_score, evidence={**tie.raw_score.evidence, "source_checkpoint_id": "other_raw"})
    rejects(replace, tie, raw_score=foreign)
    rejects(SelectedCategory, "a", "r", "c", RecoverySettings(), 0, "t", "training_node", "raw", {})
    zero = SelectedCategory("a", "r", "c", RecoverySettings(schedule_max=1000), 0, "t", "raw", "raw", {})
    require(zero.training_plan()["skip_kd"] and zero.training_plan()["schedule_max"] == 1000,
        "Raw0 replay skips training while retaining the frozen original recipe.")
    return dict(phase="cpu_fixture", pure_math_fixtures=True, kd_loss_not_a_nomination_gate=True,
        global_raw_strict_and_recovery_tie=True, trigger_own_raw=True, incomplete_cells_rejected=True,
        first_two_qualifying_500_categories=True, long_and_interaction_counters_separate=True,
        quota=quota.counts(), selected=selected)


def actual_zero_boundary(directory):
    model, prefix, current, future = tiny_native_model()
    original_state_digest = tensor_digest(model.state_dict())
    tokens = torch.tensor([[2, 5, 6, 7, 3]])
    with torch.inference_mode():
        original_logits = model(input_ids=tokens).logits
    runtime = build_cat_runtime_state(activation_runtime=None, global_adaptive_plan=None,
        runtime_identity=dict(phase="cpu_fixture", raw_runtime="actual_tiny_raw", cat_shared=dict(channel_refresh_after_category=False)))
    raw_dir = directory / "round_base"
    raw = v6.save_v6_full_checkpoint(model, str(raw_dir), checkpoint_kind="round_base",
        compressed_targets=prefix, pending_dense_targets=future, completed_categories=["q_proj"],
        compression_categories=CATEGORIES, target_layers=[0, 1], target_modules=CATEGORIES,
        cat_runtime_state=runtime, extra_meta=dict(cat_auto_phase="cpu_fixture", distill_stage_history=[dict(category="q_proj", did_train=False)]))
    actual_raw, _raw_fresh = fresh_native(raw_dir)
    actual_raw.eval().requires_grad_(False)
    with torch.inference_mode():
        native_raw_logits = actual_raw(input_ids=tokens).logits
    original_state_digest = tensor_digest(actual_raw.state_dict())
    selected = SelectedCategory("k_proj", "explicit_CPU_recipe", "explicit_CPU_candidate", RecoverySettings(),
        0, "actual_raw_zero_choice", raw["checkpoint_id"], raw["checkpoint_id"], {}, phase="cpu_fixture")
    report = export_zero_kd_boundary(round_base_dir=raw_dir, destination=directory / "zero_boundary",
        selected=selected, current_targets=current, phase="cpu_fixture")
    boundary_meta = v6.load_v6_meta(report["path"])
    restored_runtime = v6.load_v6_cat_runtime_state(report["path"], required=True, map_location="cpu")
    fresh, evidence = fresh_native(report["path"])
    fresh.eval().requires_grad_(False)
    with torch.inference_mode():
        logits = fresh(input_ids=tokens).logits
    require(torch.equal(logits, native_raw_logits) and torch.isfinite(logits).all().item()
        and tensor_digest(fresh.state_dict()) == original_state_digest, "Real zero-KD fresh state and forward must preserve the compressed raw.")
    progress = load_cat_resume_distill_progress(report["path"])
    require(list(progress.completed_categories) == ["q_proj", "k_proj"] and progress.lora_round_idx == 0
        and not progress.distill_stage_history[-1]["did_train"], "Zero-KD must advance actual category progress without adding a training round.")
    require(boundary_meta["checkpoint_id"] != raw["checkpoint_id"]
        and restored_runtime["runtime_identity"] != runtime["runtime_identity"]
        and restored_runtime["runtime_identity"]["completed_prefix"] == ["q_proj", "k_proj"]
        and not list(Path(report["path"]).glob("optimizer*")) and not list(Path(report["path"]).glob("trainer_state*")),
        "Actual zero-KD export needs new UUID/runtime and no optimizer/Trainer artifacts.")
    rejects(export_zero_kd_boundary, round_base_dir=raw_dir, destination=directory / "wrong_category",
        selected=replace(selected, category_id="v_proj"), current_targets=current, phase="cpu_fixture")
    rejects(export_zero_kd_boundary, round_base_dir=raw_dir, destination=directory / "fake_formal",
        selected=replace(selected, phase="formal"), current_targets=current, phase="formal", model_factory=lambda path: architecture(path))
    model_identity = dict(schema="cpu_fixture_dense_content", model_type="qwen3", content_digest=original_state_digest)
    projections = []
    # The fee reader uses the original actual dense projection sizes, including the future complement.
    for layer_id in range(2):
        for category in CATEGORIES:
            name = f"model.layers.{layer_id}.{'self_attn' if category in CATEGORIES[:4] else 'mlp'}.{category}"
            layer = model.get_submodule(name)
            shape = [layer.out_features, layer.in_features]
            projections.append(dict(actual_name=name, category_id=category, layer_id=layer_id,
                shape=shape, dtype="bfloat16", numel=shape[0] * shape[1]))
    inventory = dict(model_identity=model_identity, source_evidence=dict(path=str(raw_dir)),
        adapter_id="actual_tiny_CPU_zero", adapter_version=1, category_ids=list(CATEGORIES),
        projections=projections, total_target_numel=sum(row["numel"] for row in projections))
    accepted = complete_boundary_reference(export_report=report, inventory=inventory)
    require(not accepted["formal_eligible"] and not accepted["mutable_manifest"], "Actual CPU raw0 boundary remains explicitly ineligible for formal quality claims.")
    changed = {**report, "uuid": "unrelated_artifact"}
    rejects(complete_boundary_reference, export_report=changed, inventory=inventory)
    return dict(phase="cpu_fixture", actual_architecture="Qwen3 2 layers hidden16 native VAELinear",
        actual_updates=0, full_raw_state_exact=True, actual_cpu_forward_exact=True,
        source_consumed_through_same_native_loader=True,
        pre_serialization_factory_forward_max_abs=float((native_raw_logits.float() - original_logits.float()).abs().max()),
        completed_prefix=list(progress.completed_categories), actual_runtime_rebuilt=True,
        new_uuid=True, lora_round_idx=progress.lora_round_idx, optimizer_and_trainer_absent=True,
        own_ledger_measured=True, wrong_category_artifact_phase_rejected=True, formal_eligible=False,
        limitation="Actual tiny CPU finalization/export/load/forward mechanism only; no formal model, task evaluation, or GPU claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "Hide CUDA for this CPU-only check.")
    output = Path(args.output).resolve()
    require(not output.exists(), "Do not replace existing CPU evidence.")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    began = time.monotonic()
    rules = rule_checks()
    with tempfile.TemporaryDirectory(prefix="sc_cat_zero_CPU_", dir=output.parent) as directory:
        zero = actual_zero_boundary(Path(directory))
    from experiments.cat_auto_protocol_cmcc_20261104.live import _sha
    report = dict(status="selection_and_actual_zero_boundary_CPU_verified", rules=rules, actual_zero=zero,
        source_pins=dict(_pins(), **{str(Path(__file__).resolve()): _sha(Path(__file__).resolve())}),
        elapsed_seconds=time.monotonic() - began, CUDA_initialized=torch.cuda.is_initialized(), temporary_models_removed=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
