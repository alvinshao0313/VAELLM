"""Continue o export and the fixed BF16 replay, preserving all previous costs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import signal
import threading
import time
import traceback

from experiments.sc_cat_continue_cmcc_20261107 import bindings as io
from experiments.sc_cat_continue_cmcc_20261107 import replay
from experiments.sc_cat_continue_cmcc_20261107 import run as original
from experiments.sc_cat_finish_cmcc_20261108 import frozen
from experiments.sc_cat_finish_cmcc_20261108.run import Processes, RestoredBudget
from experiments.sc_cat_live_cmcc_20261106.boundary import complete_boundary_reference


PREFIX = "replay/restart_o/"


def remaining_tasks(authorization, manifest, budget):
    graph = original.replay_task_graph(authorization,
        [replay.stage_plan(manifest, row["category_id"]) for row in manifest["categories"]])
    pending = {}
    for key, spec in graph.items():
        old = budget["tasks"][key]
        if old["status"] == "completed":
            io.require(old["exit_code"] == 0, "A completed predecessor must have exited successfully.")
            continue
        io.require(old["status"] in ("failed", "reserved", "cancelled_before_execution"),
            "Do not duplicate a running task or silently reclassify its state.")
        spec = dict(spec, seconds=spec["seconds"] - old.get("actual_seconds", 0.))
        io.require(spec["seconds"] > 15, "The remaining original task window is exhausted.")
        pending[key.replace("replay/", PREFIX, 1)] = spec
    io.require(PREFIX + "o_proj/export" in pending and PREFIX + "o_proj/train" not in pending
        and not any(key.startswith(PREFIX + "k_proj/") for key in pending),
        "Reuse the completed k boundary and o fit/train; start at o export.")
    return pending


class Controller(original.Controller):
    def __init__(self, restart_path):
        io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Use bitvae with a CPU-only controller.")
        self.restart_reference = io.binding(restart_path)
        restart = io.read(restart_path)
        io.require(restart["status"] == "validated" and all(restart["checks"].values()),
            "The explicit restart must have completed its CPU and input checks.")
        self.snapshot_path = io.bound_path(restart["original_snapshot"])
        self.snapshot = frozen.install(self.snapshot_path)
        io.base.verify_sources(restart["implementation"])
        self.authorization_reference = self.snapshot["authorization"]
        self.authorization = io.read_authorization(self.authorization_reference)
        previous = io.read(io.bound_path(restart["previous_state"]))
        budget = io.read(io.bound_path(restart["previous_budget"]))
        terminal = io.read(io.bound_path(restart["previous_exit"]))
        io.require(previous["status"] == terminal["status"] == "failed" and terminal["exit_code"] == 1
            and terminal["state"] == restart["previous_state"] and len(previous["replay_stages"]) == 1,
            "Continue only the verified stopped replay, never an active or completed run.")
        self.manifest_reference = previous["replay_manifest"]
        self.manifest = io.read(io.bound_path(self.manifest_reference))
        self.o_request = io.read(io.bound_path(restart["o_export_request"]))
        for key in ("parent", "live_raw", "replay_result"):
            io.bound_path(self.o_request[key])
        self.root = Path(restart["output_root"]).resolve()
        io.require(not self.root.exists() and self.root.is_relative_to(Path(self.authorization["output_root"]).resolve()),
            "Use a new output within the original authorized batch.")
        io.require(shutil.disk_usage(self.root.parent).free >= restart["estimated_additional_storage_bytes"],
            "The remaining replay must fit before any new GPU work starts.")
        resources = io.base.check_resources((0, 1, 2, 3))
        self.root.mkdir()
        source_manifest = io.read(io.bound_path(self.authorization["source_manifest"]))
        self.batch = dict(manifest=source_manifest,
            inventory=io.read(io.bound_path(source_manifest["inventory"]))["inventory"])
        self.common = {key: self.o_request[key] for key in ("authorization", "source_manifest", "inventory_receipt",
            "data_plan_path", "phase", "stage", "execution_policy")}
        self.pins = {**self.snapshot["original_source_pins"], **self.snapshot["repair_source_pins"], **restart["implementation"]}
        self.budget = RestoredBudget(self.root / "budget.json", self.authorization, budget)
        self.budget.reserve(remaining_tasks(self.authorization, self.manifest, budget), counts={})
        io.require(self.budget.value["counter_reserved"] == budget["counter_reserved"],
            "A restart cannot reset or allocate new candidate/update counters.")
        self.processes = Processes(self.root, self.budget, self.pins, self.snapshot_path, self.snapshot)
        self.state_lock, self.deployments = threading.RLock(), {}
        self.state = deepcopy(previous)
        for key in ("error", "traceback", "exit_code", "ended_epoch"):
            self.state.pop(key, None)
        self.state.update(status="initialized", controller_pid=os.getpid(), started_epoch=time.time(),
            replay_restart=self.restart_reference, previous_failed_state=restart["previous_state"],
            previous_failed_exit=restart["previous_exit"], replay_restart_sources=restart["implementation"],
            initial_resources=resources, storage_projection_bytes=restart["estimated_additional_storage_bytes"],
            available_bytes_at_start=shutil.disk_usage(self.root).free)
        self._persist("reuse_complete_k_and_o_training_start_at_failed_export")

    def _persist(self, event):
        with self.state_lock:
            self.state.update(last_event=event, updated_epoch=time.time())
            path = self.root / "controller_state.json"
            io.write(path, self.state, replace=path.exists())
        print(dict(event=event, status=self.state["status"], replay_stages=len(self.state["replay_stages"])), flush=True)

    def run(self):
        self.state["status"] = "running"
        parent_reference = self.o_request["parent"]
        last_deployment = None
        for row in self.manifest["categories"][1:]:
            category = row["category_id"]
            plan = replay.stage_plan(self.manifest, category)
            prefix, output = PREFIX + category, self.root / "replay" / category
            if category == "o_proj":
                raw_reference = self.o_request["live_raw"]
                result_reference = self.o_request["replay_result"]
            else:
                raw_reference = self.processes.run(prefix + "/fit", "fit_worker",
                    dict(self.common, category_id=category, candidate_id=plan["source_candidate_id"],
                        recipe=plan["resolved_recipe"], parent=parent_reference, replay_manifest=self.manifest_reference,
                        output=str(output / "fit"), execution_policy=dict(attention_implementation="sdpa")),
                    output_file=output / "fit" / "live_raw.json")
                request = dict(self.common, replay_manifest=self.manifest_reference, category_id=category,
                    live_raw=raw_reference, parent=parent_reference, output=str(output / "train"), operation="train_replay")
                result_reference = self.processes.run(prefix + "/train", "replay", request,
                    output_file=output / "train" / "replay_result.json") if plan["selected_step"] else None
            export_request = dict(self.common, replay_manifest=self.manifest_reference, category_id=category,
                live_raw=raw_reference, parent=parent_reference, operation="export_replay",
                output=str(output / "export"), destination=str(output / "deployment"))
            if result_reference is not None:
                export_request["replay_result"] = result_reference
            report_reference = self.processes.run(prefix + "/export", "replay", export_request,
                output_file=output / "export" / "deployment_report.json")
            score = self._score(prefix + "/full8", report_reference, output / "full8")
            report = io.read(io.bound_path(report_reference))
            with self.budget.cpu_window(prefix + "/accept_actual_replayed_boundary"):
                boundary = complete_boundary_reference(export_report=report, inventory=self.batch["inventory"], full8_score=score)
                io.require(boundary["completed_prefix"] == list(io.ORDER[:io.ORDER.index(category) + 1])
                    and report["export_evidence"]["actual_updates"] == plan["selected_step"]
                    and report["export_evidence"]["schedule_max"] == plan["schedule_max"],
                    "Keep the frozen selected step on its original schedule and actual replay parent.")
            io.write(output / "boundary.json", boundary)
            parent_reference = io.binding(output / "boundary.json")
            last_deployment = self.deployments[score.evidence["source_checkpoint_id"]]
            self.state["replay_stages"].append(dict(category_id=category, selected_step=plan["selected_step"],
                schedule_max=plan["schedule_max"], raw=raw_reference, actual_training=result_reference,
                deployment=report_reference, boundary=parent_reference))
            self._persist("fixed_recipe_replay_stage_actual_prefix_completed")
        nine = self._longbench(last_deployment, PREFIX + "final_longbench", "replay")
        io.write(self.root / "replay" / "final_boundary.json", boundary)
        for reference in self.authorization["reference_evidence"]:
            io.bound_path(reference)
        self.state.update(status="completed", exit_code=0, ended_epoch=time.time(), complete_seven_category_chain=True,
            replay_final_boundary=parent_reference, replay_nine_task=nine, BF16_recipe_replay_completed=True)
        self._persist("original_fixed_recipe_replay_and_complete_nine_finished")
        io.write(self.root / "controller.exit.json", dict(status="completed", exit_code=0,
            ended_epoch=self.state["ended_epoch"], state=io.binding(self.root / "controller_state.json"),
            actual_GPU_hours=self.budget.value["total_actual_GPU_hours"],
            core_nine_task=self.state["core_nine_task"], replay_nine_task=nine))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", required=True)
    args = parser.parse_args()
    controller = None
    try:
        controller = Controller(args.restart)
        def interrupted(signum, _frame):
            controller.processes.cancel()
            raise InterruptedError("Replay restart received signal " + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        controller.run()
    except BaseException as exc:
        if controller is not None:
            controller.fail(exc)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
