"""Overlap unchanged native training with each committed node's own export/full8."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import traceback
import types

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest
from experiments.cat_auto_protocol_cmcc_20261104.nodes import FullEightScore, NodeScore, RecoverySettings, TrajectoryResult, trajectory_identity
from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds
from experiments.cmcc_runtime_20261030.sequence import proc_start
from experiments.sc_cat_handoff_cmcc_20261106.run import wait_original
from experiments.sc_cat_live_cmcc_20261106 import bindings as io
from experiments.sc_cat_live_cmcc_20261106.boundary import StrictMainChain
from experiments.sc_cat_live_cmcc_20261106.evaluate import read_score
from experiments.sc_cat_live_cmcc_20261106.run import Controller, restore_quota, trajectory_record
from .handoff import COUNTS, TRAIN_IDS, expected_tasks, validate_state


def own_sources():
    return {str(path.resolve()): io.binding(path)["sha256"] for path in Path(__file__).parent.glob("*.py")}


def exact_node_closure(early, final):
    io.require(early == final, "A published/early-evaluated native node differs from its final real trajectory receipt.")


def remember_node(receipts, step, receipt):
    io.require(step not in receipts, "A committed node may be consumed exactly once.")
    receipts[step] = receipt


class PipelineBudget(io.Budget):
    def __init__(self, path, authorization, value):
        self.path, self.authorization, self.value = Path(path), authorization, deepcopy(value)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.reused_initial_reservation = False

    def reserve(self, tasks, *, counts):
        if not self.reused_initial_reservation:
            io.require(tasks == expected_tasks() and counts == COUNTS, "Reuse exactly the one original k baseline dependency/count reservation.")
            self.reused_initial_reservation = True
            return
        return super().reserve(tasks, counts=counts)

    def start(self, *args, **kwargs):
        super().start(*args, **kwargs)
        with self.changed:
            self.changed.notify_all()

    def settle_adopted(self, task_id, *, observed, receipt, successful, error=None, receipt_exit_code=None):
        with self.changed:
            row = self.value["tasks"][task_id]
            io.require(task_id in TRAIN_IDS and row["status"] == "running" and not row.get("pipeline_settlement"), "Settle each original trainer exactly once.")
            elapsed = observed["observed_epoch"] - row["started_epoch"]
            io.require(elapsed > 0, "Preserve the actual original trainer ownership start.")
            if elapsed > row["seconds"]:
                successful, error = False, "Original training ownership maximum exceeded: " + str(error)
            row.update(status="completed" if successful else "failed", ended_epoch=observed["observed_epoch"],
                actual_seconds=elapsed, exit_code=None, os_exit_code=None, worker_receipt_exit_code=receipt_exit_code,
                error=error, completion_receipt=receipt, observed_process_terminal=observed, pipeline_settlement=True,
                terminal_basis="Actual trainer receipt plus pidfd or conservative adoption-time exit observation; non-child OS code unavailable.")
            self._save()
            self.changed.notify_all()
            io.require(elapsed <= row["seconds"], "The original trainer exceeded its hard maximum; all observed cost was retained.")


def _command(module, request_path):
    if module == "node_export":
        return [sys.executable, "-B", "-u", "-m", "experiments.sc_cat_pipeline_cmcc_20261106.node_export", "--request", str(Path(request_path).resolve())]
    return io.command(module, request_path)


# A private binding extends only this driver's concrete worker command. The
# original runner bytecode, globals, files, timeout and accounting stay intact.
_runner_globals = dict(io.Processes.run.__globals__, command=_command)
_finite_run = types.FunctionType(io.Processes.run.__code__, _runner_globals, "pipeline_finite_run",
    io.Processes.run.__defaults__, io.Processes.run.__closure__)


class PipelineProcesses(io.Processes):
    def __init__(self, root, budget, original_sources, workers, pipeline_sources):
        super().__init__(root, budget, original_sources, (2, 3))
        self.workers, self.pipeline_sources = workers, pipeline_sources
        self.gpus["train"] = queue.Queue()
        self.adopted_claimed = set()

    def run(self, task_id, module, request, *, output_file):
        io.verify_sources(self.pipeline_sources)
        if task_id not in self.workers:
            return _finite_run(self, task_id, module, request, output_file=output_file)
        with self.lock:
            io.require(task_id not in self.adopted_claimed, "Never adopt one trainer twice.")
            self.adopted_claimed.add(task_id)
        worker = self.workers[task_id]
        original = io.read(io.bound_path(worker["request"]))
        io.require(module == "recovery_worker" and request == {key: value for key, value in original.items() if key not in ("gpu", "deadline_epoch")}
            and Path(output_file).resolve() == Path(original["output"]).resolve() / "recovery_result.json",
            "Adopt the exact original full500 trainer request and output without rerunning training.")
        observed = None
        result_path = Path(output_file)
        try:
            observed = wait_original(worker, self.cancelled)
            io.require(not observed.get("original_deadline_termination"), "The adopted trainer exhausted its original deadline.")
            result = io.read(result_path)
            io.require(result["status"] == "completed_native_recovery" and result["exit_code"] == 0
                and result["phase"] == "formal" and result["trajectory_id"] == original["trajectory_id"]
                and result["settings"] == original["settings"] and result["actual_run"]["actual_steps"] == original["settings"]["schedule_max"]
                and result["live_raw_reference"] == original["live_raw"], "The adopted trainer must complete its actual unchanged full native trajectory.")
            io.verify_sources(self.sources)
            io.verify_sources(self.pipeline_sources)
        except BaseException as exc:
            if observed is None:
                observed = dict(pid=worker["pid"], start_ticks=worker["start_ticks"], state="adopted_wait_failed",
                    observed_epoch=time.time(), os_exit_code=None, time_basis="Conservative observation-time upper bound; actual OS terminal status unknown.")
            failure = Path(original["output"]) / "recovery_failed.json"
            receipt_path = result_path if result_path.is_file() else failure if failure.is_file() else None
            self.budget.settle_adopted(task_id, observed=observed, receipt=None if receipt_path is None else io.binding(receipt_path),
                successful=False, error=type(exc).__name__ + ": " + str(exc),
                receipt_exit_code=None if receipt_path is None else io.read(receipt_path).get("exit_code"))
            raise
        self.budget.settle_adopted(task_id, observed=observed, receipt=io.binding(result_path), successful=True, receipt_exit_code=0)
        self.gpus["train"].put(worker["gpu"])
        return io.binding(result_path)


class PipelineController(Controller):
    def _fit_raw(self, category, entry, parent, prefix):
        if category != "k_proj":
            return super()._fit_raw(category, entry, parent, prefix)
        cached = self.cached_raws[entry["candidate_id"]]
        context = self.cached_contexts[entry["candidate_id"]]
        io.require(context["parent_id"] == parent["uuid"] and context["entry"] == entry, "Reuse exactly the current actual parent/raw/recipe and its completed raw score.")
        self.state["categories"][category]["raws"][entry["candidate_id"]] = deepcopy(cached)
        self._persist("reuse_actual_completed_raw_fit_export_full8")
        return context

    def _trajectory(self, context, settings, kind, prefix):
        from .node_export import read_committed_node

        raw, entry = context["raw"], context["entry"]
        category, task_id = raw["current_category"], prefix + "/train"
        identity = trajectory_identity(category_id=category, parent_id=context["parent_id"], raw_id=raw["round_base_uuid"],
            recipe_id=context["recipe_id"], settings=settings, seed_identity=context["seeds"])
        output = self.root / "artifacts" / prefix
        with self.state_lock:
            if task_id in self.initial_train_ids:
                io.require(task_id not in self.reused_registration and kind == "baseline" and settings == RecoverySettings()
                    and any(row["category_id"] == category and row["raw_id"] == raw["round_base_uuid"]
                        and row["settings"] == settings and row["kind"] == kind for row in self.quota.started),
                    "Reuse each exact already-registered baseline once.")
                self.reused_registration.add(task_id)
            else:
                self.quota.register_start(category_id=category, raw_id=raw["round_base_uuid"], settings=settings, kind=kind)
            self._persist("pipeline_trajectory_registered_or_exactly_reused")
        request = dict(self.common, operation="train", live_raw=context["raw_reference"], settings=asdict(settings),
            output=str(output / "train"), trajectory_id=identity, recipe_id=context["recipe_id"],
            source_candidate_id=entry["candidate_id"], technical_stop_after=None)
        early = {}
        early_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=1) as trainer_executor:
            training = trainer_executor.submit(self.processes.run, task_id, "recovery_worker", request,
                output_file=output / "train" / "recovery_result.json")
            def training_terminal(future):
                try:
                    future.result()
                except BaseException:
                    self.processes.cancel()
            training.add_done_callback(training_terminal)

            def node(step):
                while True:
                    io.require(not self.processes.cancelled.is_set(), "An announced dependency failed; no further node work may start.")
                    if training.done():
                        training.result()
                    with self.budget.changed:
                        train_task = self.budget.value["tasks"][task_id]
                        reference = train_task.get("request")
                        if reference is None:
                            if training.done():
                                training.result()
                                raise RuntimeError("Trainer exited without publishing its actual request.")
                            self.budget.changed.wait(timeout=1)
                            continue
                    receipt = read_committed_node(reference, step)
                    if receipt is not None:
                        break
                    if training.done():
                        training.result()
                        raise RuntimeError("A successful full trajectory lacks its announced committed node: " + str(step))
                    self.processes.cancelled.wait(1)
                with early_lock:
                    remember_node(early, step, receipt)
                node_prefix, node_output = prefix + f"/node_{step}", output / f"node_{step}"
                report_reference = self.processes.run(node_prefix + "/export", "node_export",
                    dict(training_request=reference, node_step=step, output=str(node_output / "export"),
                        destination=str(node_output / "deployment"), authorization=self.common["authorization"]),
                    output_file=node_output / "export" / "deployment_report.json")
                exported = io.read(io.bound_path(report_reference))
                exact_node_closure(receipt, exported["source_node"])
                score = self._score(node_prefix + "/full8", report_reference, node_output / "full8")
                return NodeScore(step, score, receipt["source_node_id"])

            try:
                nodes = self._parallel([lambda step=step: node(step) for step in settings.node_steps], workers=len(settings.node_steps))
                result_reference = training.result()
            except BaseException:
                self.processes.cancel()
                try:
                    training.result()
                except BaseException:
                    pass
                raise
        result = io.read(io.bound_path(result_reference))
        io.require(result["trajectory_id"] == identity and result["settings"] == asdict(settings)
            and result["actual_run"]["actual_steps"] == settings.schedule_max
            and tuple(row["step"] for row in result["nodes"]) == settings.node_steps,
            "Selection still requires the actual whole unchanged schedule and complete native node grid.")
        for final in result["nodes"]:
            exact_node_closure(early[final["step"]], final)
        fee = raw["actual_ledger"]["per_category"][category]
        row = TrajectoryResult(category_id=category, parent_id=context["parent_id"], raw_id=raw["round_base_uuid"],
            recipe_id=context["recipe_id"], candidate_id=entry["candidate_id"], trajectory_id=identity,
            is_anchor=entry["is_anchor"], logical_bits=fee["logical_bits"], packed_bits=fee["packed_bits"], settings=settings,
            raw_score=context["raw_score"], nodes=tuple(sorted(nodes, key=lambda value: value.step)))
        io.write(output / "complete_trajectory.json", trajectory_record(row))
        with self.state_lock:
            self.state["categories"][category]["trajectories"][identity] = dict(kind=kind,
                result=io.binding(output / "complete_trajectory.json"), recovery_result=result_reference, full8_complete=True,
                published_nodes_matched_final_receipts=True, streamed=True)
            self._persist("pipeline_complete_grid_and_actual_final_receipts_closed")
        return context, row


def restore_controller(snapshot_path, validation_path):
    snapshot_path, validation_path = Path(snapshot_path).resolve(), Path(validation_path).resolve()
    snapshot, validation = io.read(snapshot_path), io.read(validation_path)
    io.require(snapshot["schema"] == "sc_cat_two_baseline_pipeline_handoff_v1" and snapshot["readout_gpus"] == [2, 3], "Use the exact approved baseline pipeline snapshot.")
    io.require(validation["status"] == "completed" and validation["exit_code"] == 0 and validation["phase"] == "cpu_fixture"
        and validation["snapshot"] == io.binding(snapshot_path) and all(validation["checks"].values()), "Bind successful CPU validation of this exact handoff.")
    io.verify_sources(validation["pipeline_source_pins"])
    io.require(validation["pipeline_source_pins"] == own_sources(), "Freeze all actual pipeline implementation files after CPU validation.")
    io.require(proc_start(snapshot["old_controller"]["pid"]) is None, "Root must stop only the old controller before this driver takes ownership.")
    for reference in snapshot["old_references"].values():
        io.bound_path(reference)
    validate_state(snapshot["state"], snapshot["budget"], snapshot["quota"])
    io.verify_sources(snapshot["old_runtime_source_pins"])
    io.check_resources((2, 3))
    batch = io.load_batch(io.bound_path(snapshot["authorization"]), io.bound_path(snapshot["root_boundary"]))
    controller = PipelineController.__new__(PipelineController)
    controller.batch, controller.root = batch, batch["root"] / "formal"
    controller.budget = PipelineBudget(controller.root / "budget.json", batch["authorization"], snapshot["budget"])
    controller.processes = PipelineProcesses(controller.root, controller.budget, snapshot["old_runtime_source_pins"],
        snapshot["workers"], validation["pipeline_source_pins"])
    controller.quota = restore_quota(snapshot["old_references"]["quota"])
    controller.chain = StrictMainChain(bf16_root_id=batch["root_boundary"]["uuid"], categories=io.ORDER)
    controller.state_lock, controller.deployments = threading.RLock(), {}
    controller.state = deepcopy(snapshot["state"])
    controller.state.update(controller_pid=os.getpid(), readout_gpus=[2, 3],
        pipeline_handoff=dict(snapshot=io.binding(snapshot_path), old_controller=snapshot["old_controller"],
            validation=io.binding(validation_path), implementation_sources=validation["pipeline_source_pins"],
            reason=snapshot["reason"], training_restarted=False, prior_raw_stages_repeated=False))
    controller.initial_train_ids, controller.reused_registration = set(TRAIN_IDS), set()
    controller.cached_raws = deepcopy(snapshot["state"]["categories"]["k_proj"]["raws"])
    controller.cached_contexts = {}
    for entry in batch["recipes"]["k_proj"]:
        cached = controller.cached_raws[entry["candidate_id"]]
        raw = io.read(io.bound_path(cached["live_raw"]))
        score_path = Path(cached["raw_score"]["evaluation_id"]).parent / "score.json"
        score = read_score(score_path)
        io.require(dict(score.task_scores) == cached["raw_score"]["task_scores"] and dict(score.evidence) == cached["raw_score"]["evidence"]
            and raw["source_parent_uuid"] == cached["parent_id"] == batch["root_boundary"]["uuid"], "Reuse this actual bound own-parent raw/full8 evidence only.")
        seeds = recovery_seeds(batch["inventory"]["model_identity"], "k_proj", batch["manifest"]["run_identity"]["search_seed"])
        controller.cached_contexts[entry["candidate_id"]] = dict(entry=entry, parent_id=cached["parent_id"],
            raw_reference=cached["live_raw"], raw=raw, raw_score=score, seeds=seeds, recipe_id=cached["recipe_id"])
        controller.deployments[raw["round_base_uuid"]] = dict(deployment_report=cached["raw_deployment"], score=io.binding(score_path))
    original = io.read(io.bound_path(snapshot["workers"][TRAIN_IDS[0]]["request"]))
    controller.common = {key: original[key] for key in ("authorization", "source_manifest", "inventory_receipt", "data_plan_path", "phase", "execution_policy")}
    controller._persist("pipeline_handoff_keep_existing_trainers_and_raw_scores")
    return controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use caller bitvae and hidden CUDA for the finite driver.")
    controller = None
    try:
        controller = restore_controller(args.snapshot, args.validation)
        def interrupted(signum, _frame):
            controller.processes.cancel()
            raise InterruptedError("Pipeline driver received signal " + str(signum))
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
