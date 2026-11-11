"""Prepare and verify the complete replay task graph; this does not authorize GPU work."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time

from experiments.sc_cat_continue_cmcc_20261107 import bindings as io
from experiments.sc_cat_continue_cmcc_20261107 import replay, run
from experiments.sc_cat_finish_cmcc_20261108 import frozen
from experiments.sc_cat_finish_cmcc_20261108.worker import wrapped_command


ROOT = Path("/home/cmcc/VAELLM/artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31")
CONTINUATION = ROOT / "live_gate_up_20261107/gpu0_3"
PREVIOUS = CONTINUATION / "repair_20261108/after_fresh_fix"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "Use bitvae; preparation is CPU only.")
    output = Path(args.output).resolve()
    io.require(not output.exists(), "Create one new draft, do not overwrite evidence.")
    snapshot_path = PREVIOUS / "formal_snapshot.json"
    snapshot = frozen.install(snapshot_path)
    authorization = io.read_authorization(snapshot["authorization"])
    old_manifest = io.read(PREVIOUS / "formal/replay/manifest.json")
    manifest = replay.freeze_manifest(old_manifest["recipe_documents"], old_manifest["selected_settings"],
        old_manifest["inventory_receipt"], old_manifest["data_plan_path"], old_manifest["selected_product"])
    io.require(manifest == old_manifest, "Resolve exactly the original complete frozen recipe, including semantic seeds.")
    plans = [replay.stage_plan(manifest, category) for category in io.ORDER]
    io.require([row["selected_step"] for row in plans] == [250, 125, 500, 375, 1000, 875, 375]
        and [row["schedule_max"] for row in plans] == [500, 500, 500, 500, 1000, 1000, 500],
        "Keep all seven historical stops and original schedules.")
    prefix = io.read(io.bound_path(authorization["prefix"]["frozen_recipe"]))
    bf16 = io.read(io.bound_path(prefix["shared_BF16_root"]))
    meta = io.read(Path(bf16["path"]) / "checkpoint_meta.json")
    io.require(io.binding(Path(bf16["path"]) / meta["state_dict_file"])["sha256"] == bf16["digest"],
        "Verify the actual shared original BF16 payload, not just the old receipt.")
    past = io.read(CONTINUATION / "repair_20261108/replay_restart/formal/budget.json")
    measured = sum(task.get("actual_seconds", 0) for task in past["tasks"].values()
        if task["component"] == "replay") / 3600
    own_pins = {str(path.resolve()): io.binding(path)["sha256"] for path in Path(__file__).parent.glob("*.py")}
    pins = {**snapshot["original_source_pins"], **snapshot["repair_source_pins"], **own_pins}
    io.base.verify_sources(pins)
    limits = deepcopy(authorization["task_seconds"])
    limits.update(fit_by_category=dict.fromkeys(io.ORDER, 1500), replay_export=450,
        replay_train_startup_seconds=300,
        replay_seconds_per_update_by_category=dict(zip(io.ORDER, [4, 6, 9, 11, 11, 15, 20])),
        longbench_shard=12000, longbench=24000)
    chains = {}
    reserved = 0
    for index, name in enumerate(("a", "b")):
        auth = deepcopy(authorization)
        auth.update(status="plan_draft_not_authorization", confirmed=False, user_approved=False,
            not_authorization=True, authorization_reference="pending_this_exact_new_batch",
            approval_user_message=None, approval_recorded_at=None,
            output_root=str(output / name), total_GPU_hours=25., expected_GPU_hours=measured,
            budget_by_component=dict(technical=0., core=0., terminal_longbench=0., replay=25., reference=0.),
            task_seconds=limits, prior_actual_GPU_hours_separate=past["total_actual_GPU_hours"],
            old_unused_budget_transferred=False,
            actual_scope="Only one fixed seven-category BF16 replay; legacy core counters are inactive, core budget is zero.")
        io.validate_plan(auth)
        graph = run.replay_task_graph(auth, plans)
        cost = sum(task["seconds"] for task in graph.values()) / 3600
        io.require(len(graph) == 30 and cost <= 25 and all(row["component"] == "replay" for row in graph.values()),
            "Reserve all seven fits/trains/exports/evaluations and both complete LongBench partitions within25GPUh.")
        reserved += cost
        chains[name] = dict(root=str(output / name), train_gpu=index, readout_gpu=index + 2,
            authorization_template=auth, task_graph=graph, reserved_GPU_hours=cost)
    training_request = io.read(io.bound_path(snapshot["training_request"]))
    output.mkdir(parents=True)
    io.require(shutil.disk_usage(output).free >= 700 * 1024**3, "Retain at least700GiB for paired diagnostics.")
    plan = dict(schema="sc_cat_deterministic_paired_replay_v1", status="prepared_awaiting_budget_approval",
        output_root=str(output), snapshot=io.binding(snapshot_path), frozen_recipe=manifest["recipe_documents"][-1],
        selected_product=manifest["selected_product"], manifest=io.binding(PREVIOUS / "formal/replay/manifest.json"),
        source_pins=pins, chains=chains, execution_policy=training_request["execution_policy"],
        expected_GPU_hours=2 * measured, total_GPU_hours=50., reserved_GPU_hours=reserved,
        storage_reserve_bytes=700 * 1024**3, gpus=[0, 1, 2, 3],
        counts=dict(fits=14, recovery_updates=7000, exports=14, full8=14, longbench21=2),
        source_cost_budget=io.binding(CONTINUATION / "repair_20261108/replay_restart/formal/budget.json"),
        prior_cost_transferred=False, new_parameter_search=False, seed_search=False,
        execution_change="Only enable deterministic FA2 backward before importing Transformers; preserve other runtime policy.",
        paired_boundary_policy="Compare actual fit reports, exported tensor digests, and complete8 scores after each category; preserve differences for diagnosis, no reselection/retry.",
        stop_conditions=["A worker failure stops that chain's dependent work without retry.",
            "A per-task deadline or budget exhaustion stops that chain.",
            "After both complete, report paired9 results and assess any residual differences before another batch."])
    io.write(output / "plan.json", plan)
    validation = dict(status="passed", phase="cpu_only", checked_epoch=time.time(),
        plan=io.binding(output / "plan.json"), actual_BF16_digest=bf16["digest"],
        checks=dict(original_full_manifest_equal=True, original_stops_and_schedules=True,
            original_source_pins_unchanged=True, full_graph_within_limits=True, disjoint_chain_devices=True,
            old_budget_not_transferred=True, output_on_data2=output.is_relative_to(Path('/data2'))),
        expected_GPU_hours=2 * measured, reserved_GPU_hours=reserved, GPU_work_started=False,
        reused_numerical_proof=io.binding(ROOT / "reproducibility_20261109/prefix_verification.json"))
    for module in ("fit_worker", "replay", "evaluate", "longbench"):
        argv = wrapped_command(run.command(module, output / "request.json"), snapshot_path)
        argv[argv.index("-m") + 1] = "experiments.sc_cat_reproducibility_cmcc_20261109.worker"
        io.require("--snapshot" in argv and "--target" in argv, "Preserve the frozen worker binding.")
    import torch
    io.require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    io.write(output / "cpu_validation.json", validation)
    print(dict(plan=str(output / "plan.json"), validation=validation), flush=True)


if __name__ == "__main__":
    main()
