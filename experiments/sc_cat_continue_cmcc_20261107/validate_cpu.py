"""Real-prefix and complete finite-graph checks; an unapproved plan remains blocked."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import tempfile
import time

from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.longbench_v1.protocol import DATASETS
from experiments.sc_cat_live_cmcc_20261106 import bindings as old_io
from experiments.sc_cat_live_cmcc_20261106.validate_selection_cpu import trajectory
from . import bindings as io
from .run import Controller, Processes, _category_method, _finite_run, evaluation_request, longbench_tasks, replay_task_graph
from .selection import terminal_endpoints


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected invalid continuation rejection.")


def worst_core_graph(plan):
    longest = set(sorted(io.CATEGORIES, key=lambda category: io.seconds(plan, "train", category=category,
        settings=RecoverySettings(schedule_max=1000)) - io.seconds(plan, "train", category=category,
        settings=RecoverySettings()), reverse=True)[:2])
    tasks = {}
    for category in io.CATEGORIES:
        for role in ("anchor", "different"):
            prefix = category + "/" + role
            tasks.update(io.raw_tasks(prefix, plan, category))
            tasks.update(io.trajectory_tasks(prefix + "/baseline", plan, category, RecoverySettings()))
        tasks.update(io.trajectory_tasks(category + "/third", plan, category,
            RecoverySettings(schedule_max=1000 if category in longest else 500)))
    return tasks, longest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-draft", required=True)
    parser.add_argument("--workers-validation", required=True)
    parser.add_argument("--terminal-validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use caller bitvae and explicitly hidden CUDA.")
    started = time.time()
    plan = io.validate_plan(io.read(args.plan_draft))
    io.require(plan["status"] == "plan_draft_not_authorization" and not plan["confirmed"] and not plan["user_approved"]
        and plan["not_authorization"] and not plan["authorization_reference"], "This CPU exercise cannot create or infer actual new approval.")
    rejects(lambda: io.read_authorization(args.plan_draft))
    prefix = io.load_prefix(plan["prefix"])
    manifest = io.read(io.bound_path(plan["source_manifest"]))
    worker_proof, terminal_proof = io.read(args.workers_validation), io.read(args.terminal_validation)
    io.require(worker_proof["exit_code"] == terminal_proof["exit_code"] == 0, "Reuse the actual successful worker and terminal/replay CPU evidence.")
    core, long_categories = worst_core_graph(plan)
    io.require(len([name for name in core if name.endswith("/fit")]) == 10
        and len([name for name in core if name.endswith("/train")]) == 15
        and len([name for name in core if name.endswith("/raw_full8")]) == 10
        and len([name for name in core if name.endswith("/full8")]) == 68,
        "The complete worst permitted five-class graph must retain all10raw/15trajectories/78evaluations.")
    core_seconds = sum(row["seconds"] for row in core.values())
    cost = io.read(io.bound_path(plan["cost_audit"]))
    io.require(math.isclose(core_seconds / 3600, cost["proposal"]["remaining_search"]["core_excluding_technical_GPU_hours"] - 2,
        rel_tol=0, abs_tol=1e-10), "The actual task graph must close against the independently audited core ceiling and its unallocated boundary reserve.")
    plans = []
    for category in io.ORDER:
        if category in io.ORDER[:2]:
            chosen = next(row for row in prefix["frozen"]["categories"] if row["category_id"] == category)
            plans.append(dict(category_id=category, settings=chosen["settings"], selected_step=chosen["selected_step"]))
        else:
            settings = RecoverySettings(schedule_max=1000 if category in long_categories else 500)
            plans.append(dict(category_id=category, settings=asdict(settings), selected_step=settings.schedule_max))
    replay = replay_task_graph(plan, plans)
    io.require(sum(row["selected_step"] for row in plans) == 3875
        and all(replay["replay/" + category + "/export"]["seconds"] == 900 for category in io.ORDER),
        "Replay uses actual fixed k250/o125, at most3500 later selected updates, and its independent900second export/fresh window.")
    replay_seconds = sum(row["seconds"] for row in replay.values())
    io.require(math.isclose(replay_seconds / 3600, cost["proposal"]["fixed_recipe_BF16_replay"]["task_windows_hard_sum_GPU_hours"],
        rel_tol=0, abs_tol=1e-10), "Actual selected-prefix replay task windows must close against the independent audited ceiling.")
    zero = [dict(row, selected_step=0) if row["category_id"] == "up_proj" else row for row in plans]
    zero_tasks = replay_task_graph(plan, zero)
    io.require("replay/up_proj/train" not in zero_tasks and "replay/up_proj/export" in zero_tasks,
        "Zero-KD replay has no training reservation while retaining the real compressed export/full8.")
    terminal = {}
    for endpoint in ("fixture_endpoint_a", "fixture_endpoint_b"):
        terminal.update(longbench_tasks(plan, "terminal/" + endpoint, "terminal_longbench"))
    io.require(sum(row["seconds"] for row in terminal.values()) == 24 * 3600
        and len(io.LONG_PARTITION_ONE) == 9 and set(io.LONG_PARTITION_ONE) <= set(DATASETS)
        and len(set(DATASETS) - set(io.LONG_PARTITION_ONE)) == 12, "Exactly two complete-task shards per actual endpoint preserve21 tasks and the24GPUh terminal cap.")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="continuation_CPU_", dir=output.parent) as directory:
        directory = Path(directory)
        budget = io.Budget(directory / "budget.json", plan, prior_GPU_hours=prefix["prior_actual_GPU_hours"])
        io.require(budget.value["total_actual_GPU_hours"] == 0
            and budget.value["prior_batch_GPU_hours"] == prefix["prior_actual_GPU_hours"]
            and not budget.value["prior_cost_transferred"], "The prior12.401GPUh and unused old cap cannot become new budget.")
        budget.reserve(core, counts={key: value for key, value in io.CORE_COUNTS.items() if key != "terminal_longbench"})
        budget.reserve(terminal, counts=dict(terminal_longbench=2))
        budget.reserve(replay, counts=dict(io.REPLAY_COUNTS))
        rejects(lambda: budget.reserve({"extra/fit": dict(component="core", seconds=30)}, counts=dict(raws=1)))
        rejects(lambda: budget.reserve({"extra/LB": dict(component="terminal_longbench", seconds=30)}, counts=dict(terminal_longbench=1)))
        request_path = directory / "CPU_request.json"
        io.write(request_path, dict(phase="cpu_fixture", purpose="Budget contract only; no real GPU execution claimed."))
        first = "gate_proj/anchor/fit"
        now = time.time()
        budget.start(first, gpu=0, start=now - 20, deadline=now + 100,
            command=["explicit_CPU_contract_only"], log=directory / "none.log", request=request_path)
        rejects(lambda: budget.start("gate_proj/different/fit", gpu=0, start=now, deadline=now + 100,
            command=["explicit_CPU_contract_only"], log=directory / "none.log", request=request_path))
        budget.finish(first, end=now, exit_code=17, error="CPU fixture failure charged")
        io.require(budget.value["actual_seconds_by_component"]["core"] == 20
            and budget.value["tasks"][first]["status"] == "failed", "A failed task retains its actual own window once.")
        processes = Processes(directory, budget, {}, (2, 3))
        io.require(tuple(processes.gpus["readout"].get_nowait() for _ in range(2)) == (2, 3)
            and processes.gpus["readout"].empty(), "An explicitly selected authorized subset cannot use4–7.")
    baselines = [trajectory("gate_proj", raw="CPU_gate_anchor", values=(.7, .68, .66, .64)),
        trajectory("gate_proj", raw="CPU_gate_other", anchor=False, values=(.6,) * 4)]
    arm = prefix["quota"].decide(baselines, prefix_targets=["k", "o", "gate"], current_targets=["gate"], future_dense_targets=["future"])
    io.require(arm is not None and not arm.interaction_eligible and prefix["quota"].interaction_categories == ["k_proj", "o_proj"],
        "The two consumed full-chain interaction slots cannot reset in this batch.")
    no_zero = [trajectory("up_proj", raw="CPU_up_a", values=(.6,) * 4, raw_value=.8),
        trajectory("up_proj", raw="CPU_up_b", anchor=False, values=(.9,) * 4)]
    endpoints, eligibility = terminal_endpoints(no_zero)
    io.require(not eligibility["global_raw_zero_strictly_qualified"] and all(node.step == 500 for _row, node in endpoints),
        "Beating one's own recovery cannot qualify raw0 against another recipe's stronger recovery.")
    qualified = [trajectory("up_proj", raw="CPU_up_a", values=(.6,) * 4, raw_value=.95), no_zero[1]]
    endpoints, eligibility = terminal_endpoints(qualified)
    io.require(eligibility["global_raw_zero_strictly_qualified"] and endpoints[0][1].step == 0 and endpoints[1][1].step == 500,
        "Only the globally strictly superior actual raw may replace its own one endpoint.")
    common = dict(source_manifest=plan["source_manifest"], inventory_receipt=manifest["inventory"],
        data_plan_path=manifest["data_plan"], authorization=io.binding(args.plan_draft), phase="formal", stage="core")
    request = evaluation_request(common, manifest["baseline"], plan["prefix"]["after_o"], "/CPU_fixture", "replay")
    io.require(all(request[key] == common[key] for key in ("source_manifest", "inventory_receipt", "data_plan_path", "authorization"))
        and request["stage"] == "replay", "The real full8 request builder must retain every source field and explicitly override the phase component.")
    io.require(old_io.Processes.run.__globals__["command"] is old_io.command
        and _finite_run.__globals__["command"] is not old_io.command
        and _category_method(plan).__globals__["io"].raw_tasks("gate_proj/anchor")["gate_proj/anchor/fit"]["seconds"] == 4500,
        "Private actual-source adapters cannot mutate frozen modules or reuse the old fixed budget.")
    import torch
    io.require(not torch.cuda.is_initialized(), "CPU review cannot initialize CUDA or execute a new model path.")
    io.write(output, dict(status="completed", exit_code=0, phase="cpu_fixture", plan_draft=io.binding(args.plan_draft),
        prefix=plan["prefix"], workers_validation=io.binding(args.workers_validation), terminal_validation=io.binding(args.terminal_validation),
        source_pins=io.runtime_pins(manifest), worst_case_task_GPU_hours=dict(core=core_seconds / 3600,
            terminal_longbench=sum(row["seconds"] for row in terminal.values()) / 3600, replay=replay_seconds / 3600),
        prior_actual_GPU_hours_separate=prefix["prior_actual_GPU_hours"],
        checks=dict(actual_after_o_and_strict_chain_verified=True, actual_prefix_quota_not_reset=True,
            unapproved_draft_rejected_by_runtime=True, fixed_core_count_and_category_windows=True,
            full_dependency_budget_closed=True, replay3875_bound_and_original_T_policy=True,
            replay_export900_and_zero_no_train=True, all21_tasks_two_unique_partition_owners=True,
            failed_window_charged_once=True, GPU_overlap_rejected=True, explicit_resource_subset=True,
            global_raw_zero_eligibility_preserved=True, full8_source_fields_preserved=True,
            immutable_actual_consumer_adapters=True, no_GPU_started_or_CUDA_initialized=True),
        elapsed_seconds=time.time() - started,
        boundary="CPU assembly and actual existing evidence only. No new approval, model training, final native LongBench or BF16 replay GPU completion is claimed."))


if __name__ == "__main__":
    main()
