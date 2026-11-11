"""Build a reviewable real-input cost proposal; never create user approval."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import bindings as io


def prepare(cost_path, output, *, output_root=None):
    cost_path = Path(cost_path).resolve()
    cost = io.read(cost_path)
    io.require(cost["status"] == "read_only_cost_audit_budget_proposal_not_approval", "Read the actual cost audit, not an approval substitute.")
    source = Path(cost["source_run"]).resolve()
    previous = source / "live_k_o_20261106" / "formal"
    manifest_path = source / "manifest.json"
    manifest = io.read(manifest_path)
    proposal = cost["proposal"]
    search, replay = proposal["remaining_search"], proposal["fixed_recipe_BF16_replay"]
    prefix = {key: io.binding(previous / name) for key, name in dict(after_o="after_o.json", frozen_recipe="frozen_recipe.json",
        quotas="quotas.json", controller_state="controller_state.json", controller_exit="controller.exit.json").items()}
    actual_prefix = io.load_prefix(prefix)
    component_view = dict(proposal["controller_component_view"])
    total = component_view.pop("total")
    per_update = {category: row["per_update_seconds_proposed"] for category, row in replay["fixed_classes"].items()}
    per_update.update(replay["remaining_per_update_seconds_proposed"])
    fits = {category: search["raw_fit_single_cap_seconds"] for category in io.ORDER}
    limits = dict(fit_by_category=fits, train500_by_category=search["per_class500_train_cap_seconds"],
        train1000_by_category=search["per_class1000_train_cap_seconds"], raw_export=search["raw_export_single_cap_seconds"],
        node_export=search["node_export_single_cap_seconds"], full8=search["full8_single_cap_seconds"], replay_export=900,
        replay_train_startup_seconds=replay["startup_seconds_per_actual_replay_category"],
        replay_seconds_per_update_by_category=per_update,
        longbench=proposal["terminal_two_LB"]["each_artifact_hard_GPU_hours"] * 3600,
        longbench_shard=proposal["LongBench_root_selected_partition_policy"]["partitions"][0]["hard_seconds"])
    reference = cost["fixed_CAT_reference"]
    evidence = [reference[name] for name in ("evaluation", "acceptance", "metadata", "pipeline_report", "source_common")]
    for item in evidence:
        io.bound_path(item)
    repo = Path(__file__).resolve().parents[2]
    longbench_archive = io.binding(repo / "result/prepare_eval_20261030/longbench/manifest.json")
    baseline_scores = io.binding("/data1/cmcc/VAELLM/artifacts/cmcc_baselines_20261030/qwen3_8b/longbench/scores.json")
    migration = io.binding("/data1/cmcc/VAELLM/artifacts/cmcc_baselines_20261030/qwen3_8b/migration_binding.json")
    document = dict(schema="sc_cat_continuation_authorization_v1", status="plan_draft_not_authorization",
        phase="planned_formal", confirmed=False, user_approved=False, not_authorization=True,
        authorization_reference="", requires_actual_new_user_approval=True,
        categories=list(io.CATEGORIES), frozen_order=list(io.ORDER), stages=["core", "terminal_longbench", "replay"],
        gpus=list(range(6)), train_gpus=[0, 1], readout_gpus=[2, 3, 4, 5], protected_gpus=[6, 7],
        output_root=str(Path(output_root).resolve() if output_root else source / "live_gate_up_20261107"),
        prefix=prefix, source_manifest=io.binding(manifest_path), selected_settings=io.binding(source / "selected_settings.json"),
        protocol=io.binding(repo / "docs/sc_cat_protocol.md"), execution_protocol=io.binding(repo / "docs/sc_cat_execution.md"),
        baseline=manifest["baseline"], longbench_archive=longbench_archive, baseline_longbench_scores=baseline_scores,
        baseline_longbench_migration=migration, cost_audit=io.binding(cost_path),
        total_GPU_hours=total, budget_by_component=component_view, expected_GPU_hours=proposal["combined_expected_new_GPU_hours"],
        task_seconds=limits, reference_mode="reuse_only", reference_evidence=evidence,
        reference_comparison_boundary=reference["differences"], reference_original_unrounded_score9=reference["score9"],
        prior_actual_GPU_hours_separate=actual_prefix["prior_actual_GPU_hours"], old_unused_budget_transferred=False,
        counting="Actual nonoverlapping per-device ownership windows including failures/export/fresh; no automatic retry/extension; CPU windows separate.",
        first_formal_execution="Start the two actual gate fits and verify original main-loop progress; no repeated GPU startup short test.",
        replay_first_new_stop_path="Actual fixed k250/T500, then its single required export/fresh/full8; no extra125 training.",
        LongBench_partition_policy=proposal["LongBench_root_selected_partition_policy"],
        missing_approval_is_runtime_blocking=True)
    document.update({"maximum_" + key: value for key, value in {**io.CORE_COUNTS, **io.REPLAY_COUNTS}.items()})
    io.validate_plan(document)
    io.write(output, document)
    print(dict(status=document["status"], output=io.binding(output), total_proposed_GPU_hours=total,
        confirmed=False, user_approved=False, GPU_started=False), flush=True)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    prepare(args.cost_audit, args.output, output_root=args.output_root)


if __name__ == "__main__":
    main()
