"""Freeze the selected seven-class recipe and replay real native schedule prefixes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import functools
import json
import os
from pathlib import Path
import signal
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture
from experiments.cat_auto_protocol_cmcc_20261104.native_builder import RecoveryExecutionPolicy, make_native_recovery_trainer
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds
from experiments.sc_cat_live_cmcc_20261106 import live_io
from experiments.sc_cat_live_cmcc_20261106.boundary import export_selected_node, export_zero_kd_boundary
from experiments.sc_cat_live_cmcc_20261106.recovery_io import binding, package_pins, read_bound_json, read_json, write_json
from experiments.sc_cat_live_cmcc_20261106.selection import SelectedCategory
from train_utils import checkpoint_v6 as v6


def resolve_categories(recipe_documents, selected_settings, inventory_receipt, data_plan_path):
    """Resolve actual source recipes, including a legitimately incomplete frozen prefix."""
    settings_source = read_bound_json(selected_settings)
    inventory = read_bound_json(inventory_receipt)["inventory"]
    data = torch.load(live_io.bound_path(data_plan_path), map_location="cpu", weights_only=False)
    require(settings_source["status"] == "completed" and digest(data["manifest"]) == data["data_identity"]
        and data["manifest"]["model_identity"] == inventory["model_identity"]
        and settings_source["run_identity"]["data_identity"] == data["data_identity"],
        "Freeze the actual completed SC-CAT source settings and unchanged original data/model.")
    order = settings_source["frozen_order"]["ordered_categories"]
    candidates = {}
    for category in settings_source["categories"]:
        for reference in (category["anchor"], category["best_different"]["result"]):
            original = read_bound_json(reference)
            candidates[original["candidate_id"]] = (original, reference)
    choices = {}
    for reference in recipe_documents:
        document = read_bound_json(reference)
        require(document["phase"] == "formal" and document["frozen_order"] == order,
            "Only the actual frozen main-chain order and selected recipes may define replay.")
        for selected in document["categories"]:
            category = selected["category_id"]
            if category in choices:
                require(choices[category]["selected"] == selected, "Overlapping recipe documents disagree about a selected prefix.")
                continue
            recipe, original_ref = candidates[selected["source_candidate_id"]]
            recovery = RecoverySettings(**selected["settings"])
            step = selected["selected_step"]
            require(recipe["recipe"]["category_id"] == category and recipe["recipe"]["S"] == 2
                and digest(recipe["recipe"]) == selected["recipe_id"]
                and type(step) is int and step in (0, *recovery.node_steps)
                and (step != 0 or recovery == RecoverySettings())
                and selected["seed_identity"] == recovery_seeds(inventory["model_identity"], category, data["manifest"]["search_seed"]),
                "Keep the exact selected S2 recipe, original schedule, measured stop and independent semantic seeds.")
            choices[category] = dict(category_id=category, selected=selected,
                resolved_recipe=recipe["recipe"], original_result=original_ref,
                source_recipe_document=reference, expected_recovery_updates=step)
    actual = [category for category in order if category in choices]
    require(actual == order[:len(actual)], "A frozen recipe can only be one contiguous actual prefix.")
    return dict(frozen_order=order, categories=[choices[category] for category in actual],
        model_identity=inventory["model_identity"], original_HF_source=inventory["source_evidence"],
        data_identity=data["data_identity"], search_seed=data["manifest"]["search_seed"],
        fit_code_identity=settings_source["run_identity"]["code_identity"],
        consumer_identity=settings_source["run_identity"]["consumer_identity"])


def freeze_manifest(recipe_documents, selected_settings, inventory_receipt, data_plan_path, final_boundary):
    resolved = resolve_categories(recipe_documents, selected_settings, inventory_receipt, data_plan_path)
    boundary = read_bound_json(final_boundary)
    require(len(resolved["categories"]) == len(resolved["frozen_order"])
        and boundary["completed_prefix"] == resolved["frozen_order"] and boundary["formal_eligible"]
        and boundary["phase"] == "formal"
        and boundary["export_evidence"]["source_node_id"] == resolved["categories"][-1]["selected"]["source_node_id"],
        "Freeze only the finally selected complete seven-category product, after S9 selection.")
    result = dict(schema="sc_cat_fixed_recipe_BF16_replay_v1", phase="formal", **resolved,
        recipe_documents=recipe_documents, selected_settings=selected_settings,
        inventory_receipt=inventory_receipt, data_plan_path=data_plan_path, selected_product=final_boundary,
        schedule_policy="original_schedule_max_stop_at_frozen_selected_step", root_policy="original_verified_BF16_then_own_fresh_live_fits",
        replay_fits=len(resolved["categories"]), replay_exports=len(resolved["categories"]),
        replay_KD_updates=sum(row["expected_recovery_updates"] for row in resolved["categories"]),
        repeat_search=False, repeat_node_selection=False, repeat_terminal_pair=False,
        implementation=binding(__file__), source_pins=package_pins())
    result["manifest_id"] = digest(result)
    return result


def read_manifest(reference):
    value = read_bound_json(reference)
    body = dict(value)
    identity = body.pop("manifest_id")
    require(value["schema"] == "sc_cat_fixed_recipe_BF16_replay_v1" and digest(body) == identity
        and [row["category_id"] for row in value["categories"]] == value["frozen_order"]
        and not value["repeat_search"] and not value["repeat_node_selection"] and not value["repeat_terminal_pair"],
        "The complete chosen replay recipe changed after freezing.")
    return value


def stage_plan(manifest, category_id):
    matches = [row for row in manifest["categories"] if row["category_id"] == category_id]
    require(len(matches) == 1, "Replay exactly one frozen category, never a new candidate.")
    row = matches[0]
    selected = row["selected"]
    settings = RecoverySettings(**selected["settings"])
    return dict(category_id=category_id, resolved_recipe=row["resolved_recipe"], settings=asdict(settings),
        selected_step=selected["selected_step"], schedule_max=settings.schedule_max,
        seed_identity=selected["seed_identity"], skip_KD=selected["selected_step"] == 0,
        maximum_KD_updates=selected["selected_step"], source_candidate_id=selected["source_candidate_id"],
        recipe_id=selected["recipe_id"])


def _inputs(request, kind, *, with_sources):
    from .bindings import seconds, validate_worker_request
    from .workers import _raw_provenance, _source_fields
    require(request["phase"] == "formal" and request["stage"] == "replay",
        "A technical/core request cannot be relabeled as formal recipe replay.")
    manifest = read_manifest(request["replay_manifest"])
    plan = stage_plan(manifest, request["category_id"])
    actual = dict(request, phase="formal", stage="replay", settings=plan["settings"], selected_step=plan["selected_step"],
        inventory_receipt=manifest["inventory_receipt"], data_plan_path=manifest["data_plan_path"])
    authorization = validate_worker_request(actual, kind=kind)
    _source_fields(actual, authorization)
    deadline_kind = "replay_train" if kind == "replay_train" else "replay_export"
    limit = seconds(authorization, deadline_kind, category=plan["category_id"],
        settings=RecoverySettings(**plan["settings"]), selected_step=plan["selected_step"])
    require(0 < actual["deadline_epoch"] - time.time() <= limit,
        "The replay stage exceeds its selected-prefix/category-specific authorized window.")
    require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == str(actual["gpu"]),
        "Use caller-activated bitvae and exactly the assigned replay GPU.")
    raw = read_bound_json(actual["live_raw"])
    _raw_provenance(actual, authorization, raw)
    parent = live_io.actual_parent(actual)
    producer_parent = live_io.actual_parent(read_bound_json(raw["request"]))
    require(raw["source_parent_uuid"] == parent["uuid"] == producer_parent["uuid"]
        and raw["run_identity"]["parent_weights_digest"] == parent["digest"] == producer_parent["digest"],
        "The fresh replay fit must read this exact actual replay parent, not a core/donor prefix.")
    inventory, prepared, sources, tokenizer = live_io.load_inputs(actual, with_sources=with_sources)
    meta = v6.load_v6_meta(raw["round_base_dir"])
    index = manifest["frozen_order"].index(request["category_id"])
    require(raw["phase"] == "formal" and raw["formal_eligible"]
        and raw["current_category"] == request["category_id"] and raw["recipe"] == plan["resolved_recipe"]
        and meta["checkpoint_id"] == raw["round_base_uuid"] and meta["completed_categories"] == manifest["frozen_order"][:index]
        and (meta.get("extra_meta") or {}).get("cat_auto_phase") == "formal"
        and prepared["data_identity"] == manifest["data_identity"],
        "The replay raw must be freshly fitted on its own actual replay prefix with the frozen exact recipe.")
    actual["trajectory_id"] = digest(dict(manifest_id=manifest["manifest_id"], category_id=request["category_id"], raw_id=raw["round_base_uuid"]))
    actual.update(recipe_id=plan["recipe_id"], source_candidate_id=plan["source_candidate_id"], execution_policy=request["execution_policy"])
    return actual, manifest, plan, raw, inventory, prepared, sources, tokenizer


def _retained_nodes(run, raw, actual, plan):
    steps = tuple(range(125, plan["selected_step"] + 1, 125))
    require(tuple(row["step"] for row in run.trainer.node_save_audits) == steps, "Retain the actually executed original-schedule prefix, never a fabricated complete grid.")
    rows = []
    for step in steps:
        directory = Path(run.trainer.args.output_dir) / f"checkpoint-{step}"
        meta = v6.load_v6_training_step_meta(str(directory))
        state = read_json(directory / "trainer_state.json")
        source, source_meta = v6.resolve_training_step_round_base_ref(str(directory), meta)
        require(state["global_step"] == step and state["max_steps"] == plan["schedule_max"]
            and source_meta["checkpoint_id"] == raw["round_base_uuid"], "The actual replay node changed its original scheduler/raw identity.")
        files = ("checkpoint_meta.json", "training_model_state.pt", "optimizer.pt", "scheduler.pt", "trainer_state.json")
        rng = sorted(directory.glob("rng_state*.pth"))
        require(all((directory / name).is_file() for name in files) and rng, "The replay node lacks complete mutable/optimizer/scheduler/RNG state.")
        rows.append(dict(step=step, schedule_max=plan["schedule_max"], node_directory=str(directory.resolve()),
            source_node_id=meta["checkpoint_id"], source_round_base_id=source_meta["checkpoint_id"],
            source_round_base=str(Path(source).resolve()), trajectory_id=actual["trajectory_id"],
            recipe_id=plan["recipe_id"], source_candidate_id=plan["source_candidate_id"], category_id=plan["category_id"],
            settings=plan["settings"], semantic_seeds=plan["seed_identity"], checkpoint_meta=binding(directory / "checkpoint_meta.json"),
            native_files={name: dict(path=str((directory / name).resolve()), bytes=(directory / name).stat().st_size) for name in files},
            rng_files=[dict(path=str(path.resolve()), bytes=path.stat().st_size) for path in rng],
            mutable_manifest=meta[v6.MUTABLE_STATE_MANIFEST_KEY], exact_resume_verified=False))
    return rows


def train_replay(request):
    began = time.monotonic()
    actual, manifest, plan, raw, inventory, prepared, sources, tokenizer = _inputs(request, "replay_train", with_sources=True)
    require(not plan["skip_KD"] and not Path(actual["output"]).exists(), "Zero-KD replay bypasses the Trainer; positive replay starts from its own new raw.")
    settings = RecoverySettings(**plan["settings"])
    run = make_native_recovery_trainer(round_base_dir=raw["round_base_dir"], inventory=inventory,
        category_id=plan["category_id"], settings=settings, prepared_data=prepared, sources=sources, tokenizer=tokenizer,
        output_dir=actual["output"], execution_policy=RecoveryExecutionPolicy(**actual["execution_policy"]),
        phase="formal", replay_selected_step=plan["selected_step"])
    try:
        result = run.train()
        require(result["actual_steps"] == plan["selected_step"] and result["schedule_max"] == plan["schedule_max"]
            and result["provenance"]["semantic_seeds"] == plan["seed_identity"], "The genuine formal replay stopped at the wrong original-schedule prefix.")
        nodes = _retained_nodes(run, raw, actual, plan)
        report = dict(status="completed_fixed_recipe_replay_stage", exit_code=0, phase="formal", stage="replay",
            manifest=request["replay_manifest"], manifest_id=manifest["manifest_id"], plan=plan,
            actual_run=result, nodes=nodes, live_raw_reference=request["live_raw"], live_raw=raw,
            original_schedule_max=plan["schedule_max"], selected_stop=plan["selected_step"],
            technical_stop_used=False, exact_resume_verified=False, source_pins=package_pins(),
            implementation=binding(__file__), elapsed_seconds=time.monotonic() - began)
        write_json(Path(actual["output"]) / "replay_result.json", report)
        return report
    finally:
        run.close()


def export_replay(request):
    from .workers import finish_export
    began = time.monotonic()
    actual, manifest, plan, raw, inventory, _prepared, _sources, tokenizer = _inputs(request, "replay_export", with_sources=False)
    current = [row["actual_name"] for row in inventory["projections"] if row["category_id"] == plan["category_id"]]
    settings = RecoverySettings(**plan["settings"])
    factory = functools.partial(architecture, attn_implementation=actual["execution_policy"]["attention_implementation"])
    if plan["skip_KD"]:
        selected = SelectedCategory(plan["category_id"], plan["recipe_id"], plan["source_candidate_id"], settings, 0,
            actual["trajectory_id"], raw["round_base_uuid"], raw["round_base_uuid"], plan["seed_identity"], phase="formal")
        report = export_zero_kd_boundary(round_base_dir=raw["round_base_dir"], destination=request["destination"],
            selected=selected, current_targets=current, phase="formal", model_factory=factory)
        actual["operation"] = "export_zero"
    else:
        result = read_bound_json(request["replay_result"])
        require(result["status"] == "completed_fixed_recipe_replay_stage" and result["exit_code"] == 0
            and result["manifest_id"] == manifest["manifest_id"] and result["plan"] == plan
            and result["live_raw"]["round_base_uuid"] == raw["round_base_uuid"], "Export only the exact frozen replay's actual selected prefix.")
        matches = [row for row in result["nodes"] if row["step"] == plan["selected_step"]]
        require(len(matches) == 1, "The actual selected replay node was not saved.")
        receipt = matches[0]
        require(binding(Path(receipt["node_directory"]) / "checkpoint_meta.json") == receipt["checkpoint_meta"], "The saved replay node changed.")
        selected = SelectedCategory(plan["category_id"], plan["recipe_id"], plan["source_candidate_id"], settings,
            plan["selected_step"], actual["trajectory_id"], receipt["source_node_id"], raw["round_base_uuid"], plan["seed_identity"], phase="formal")
        report = export_selected_node(node_directory=receipt["node_directory"], destination=request["destination"],
            selected=selected, current_targets=current, phase="formal", model_factory=factory)
        report["source_node"] = receipt
        report["replay_result_reference"] = request["replay_result"]
        actual["operation"] = "export"
    report.update(source_raw_reference=request["live_raw"], replay_manifest=request["replay_manifest"], replay_plan=plan,
        selected_step_fixed_before_training=True, no_node_reselection=True, source_pins=package_pins(), implementation=binding(__file__))
    return finish_export(actual, report, tokenizer, began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    if request["operation"] == "freeze":
        report = freeze_manifest(**{key: request[key] for key in
            ("recipe_documents", "selected_settings", "inventory_receipt", "data_plan_path", "final_boundary")})
        write_json(Path(request["output"]) / "replay_manifest.json", report)
    else:
        def deadline_exhausted(_signum, _frame):
            raise TimeoutError("The assigned fixed-recipe replay stage deadline expired.")
        signal.signal(signal.SIGALRM, deadline_exhausted)
        signal.setitimer(signal.ITIMER_REAL, max(.01, request["deadline_epoch"] - time.time()))
        torch.set_num_threads(4)
        report = {"train_replay": train_replay, "export_replay": export_replay}[request["operation"]](request)
    print(json.dumps(dict(status=report.get("status", "fixed_recipe_replay_export_complete"), output=request["output"])), flush=True)


if __name__ == "__main__":
    main()
