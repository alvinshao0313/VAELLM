"""Adopt exactly the existing two k fits, retaining frozen workers and budgets."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import queue
import select
import signal
import threading
import time
import traceback

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest
from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.cmcc_runtime_20261030.sequence import pidfd_open, proc_start
from experiments.sc_cat_live_cmcc_20261106 import bindings as io
from experiments.sc_cat_live_cmcc_20261106.boundary import StrictMainChain
from experiments.sc_cat_live_cmcc_20261106.run import Controller, restore_quota


FIT_IDS = ("k_proj/anchor/fit", "k_proj/different/fit")
COUNTS = dict(raws=2, trajectories=2, KD_updates=1000, raw_full8=2, node_full8=8)


def own_sources():
    return {str(path.resolve()): io.binding(path)["sha256"] for path in Path(__file__).parent.glob("*.py")}


def validate_limited_state(state, budget, quota):
    expected = {}
    for role in ("anchor", "different"):
        prefix = "k_proj/" + role
        expected.update(io.raw_tasks(prefix))
        expected.update(io.trajectory_tasks(prefix + "/baseline", RecoverySettings()))
    io.require(state["status"] == "running" and state["current_category"] == "k_proj"
        and state["chain"] == [] and set(state["categories"]) == {"k_proj"}
        and state["categories"]["k_proj"]["raws"] == {} and state["categories"]["k_proj"]["trajectories"] == {}
        and quota["selected_categories"] == [] and quota["started"] == [] and quota["decisions"] == {},
        "This handoff is restricted to the original two fits before any raw export or KD starts.")
    io.require(budget["counter_reserved"] == COUNTS and set(budget["tasks"]) == set(expected),
        "Reuse only the exact already reserved k baseline dependency graph and counts.")
    for name, expected_row in expected.items():
        row = budget["tasks"][name]
        io.require(all(row[key] == value for key, value in expected_row.items())
            and row["status"] == ("running" if name in FIT_IDS else "reserved"),
            "A task changed or another operation already started: " + name)
    io.require({budget["tasks"][name]["gpu"] for name in FIT_IDS} == {0, 1}, "Adopt only one actual fit on each original training GPU.")
    io.require(state["quota"] == quota, "State and durable quota disagree.")
    return expected


def prepare(*, authorization, root_boundary, output, parent_request=None):
    batch = io.load_batch(authorization, root_boundary)
    root = batch["root"] / "formal"
    paths = {key: root / name for key, name in dict(state="controller_state.json", budget="budget.json",
        quota="quotas.json", source_pins="source_pins.json").items()}
    references = {key: io.binding(path) for key, path in paths.items()}
    values = {key: io.read(path) for key, path in paths.items()}
    validate_limited_state(values["state"], values["budget"], values["quota"])
    io.verify_sources(values["source_pins"])
    controller = values["state"]["controller_pid"]
    controller_ticks = proc_start(controller)
    request = None if parent_request is None else io.read(parent_request)
    if request is None:
        io.require(controller_ticks is not None, "Snapshot the original live controller or provide root's explicit parent-only stop record.")
    else:
        io.require(request["status"] == "parent_only_SIGKILL_sent"
            and request["before_state"] == values["state"] and request["before_budget"] == values["budget"]
            and request["old_controller"]["pid"] == controller and request["new_readout_gpus"] == [2, 3]
            and proc_start(controller) is None, "Root's parent-only stop record must match this exact unchanged state.")
        controller_ticks = request["old_controller"]["start_ticks"]
    workers = {}
    for name in FIT_IDS:
        task = values["budget"]["tasks"][name]
        ticks = proc_start(task["pid"])
        if request is not None:
            captured = request["active_fit_processes"][name]
            io.require(captured["pid"] == task["pid"] and ticks in (None, captured["start_ticks"]), "The captured original fit process identity changed.")
            ticks = captured["start_ticks"]
        request_path = io.bound_path(task["request"])
        if proc_start(task["pid"]) is not None:
            io.require(ticks is not None and os.getpgid(task["pid"]) == task["pid"], "Snapshot each actual independently grouped in-flight fit.")
            arguments = Path(f"/proc/{task['pid']}/cmdline").read_bytes().split(b"\0")
            io.require(os.fsencode(str(request_path)) in arguments, "The running fit command changed before handoff.")
        else:
            io.require(request is not None, "A vanished fit requires root's captured process identity.")
        workers[name] = dict(pid=task["pid"], start_ticks=ticks, gpu=task["gpu"],
            original_started_epoch=task["started_epoch"], original_deadline_epoch=task["deadline_epoch"], request=task["request"])
    io.require(all(io.binding(path) == references[key] for key, path in paths.items()),
        "The old controller advanced while preparing the limited snapshot; do not stop it using this snapshot.")
    value = dict(schema="sc_cat_two_live_fit_handoff_v1", phase="formal", created_epoch=time.time(),
        reason="GPU5 acquired by another session after initial resource check; use only readout2/3 without stopping or refitting existing own fits.",
        authorization=batch["authorization_reference"], root_boundary=batch["root_boundary_reference"],
        old_controller=dict(pid=controller, start_ticks=controller_ticks), workers=workers,
        old_references=references, state=values["state"], budget=values["budget"], quota=values["quota"],
        parent_handoff_request=None if parent_request is None else io.binding(parent_request),
        old_source_pins=values["source_pins"], handoff_source_pins=own_sources(),
        readout_gpus=[2, 3], train_gpus=[0, 1], never_use_gpus=[4, 5, 6, 7],
        process_stop_performed=False, execution_not_started=True)
    output = Path(output).resolve()
    io.require(output.is_relative_to(batch["root"]) and not output.is_relative_to(root / "artifacts"), "Keep the handoff snapshot inside its authorized batch.")
    io.write(output, value)
    print(dict(status="handoff_snapshot_ready_no_process_stopped", snapshot=io.binding(output), old_controller=value["old_controller"], workers=workers), flush=True)


class AdoptedBudget(io.Budget):
    def __init__(self, path, authorization, snapshot):
        self.path, self.authorization, self.lock = Path(path), authorization, threading.RLock()
        self.value = deepcopy(snapshot["budget"])
        self.reused_initial_reservation = False

    def reserve(self, tasks, *, counts):
        if not self.reused_initial_reservation:
            expected = {name: dict(component=row["component"], seconds=row["seconds"]) for name, row in self.value["tasks"].items()}
            io.require(tasks == expected and counts == COUNTS, "The one adopted k reservation must match every original task and count exactly.")
            self.reused_initial_reservation = True
            return
        return super().reserve(tasks, counts=counts)

    def finish_adopted(self, task_id, *, end, receipt, observed, successful=True, error=None, receipt_exit_code=0):
        with self.lock:
            task = self.value["tasks"][task_id]
            io.require(task_id in FIT_IDS and task["status"] == "running" and not task.get("handoff_settlement"),
                "Settle each original fit once without another reservation or ownership start.")
            elapsed = end - task["started_epoch"]
            io.require(elapsed > 0, "The original adopted fit ownership window must have positive elapsed time.")
            if elapsed > task["seconds"]:
                successful, error = False, "The observed adopted ownership window exceeded its original hard maximum: " + str(error)
            task.update(status="completed" if successful else "failed", ended_epoch=end, actual_seconds=elapsed,
                exit_code=None, os_exit_code=None, worker_receipt_exit_code=receipt_exit_code, error=error,
                terminal_basis="Independent worker completed.json/live_raw/fresh proof plus actual PID exit observation; orphan OS exit code unavailable.",
                completion_receipt=receipt, handoff_settlement=True, observed_process_terminal=observed)
            self._save()
            io.require(elapsed <= task["seconds"], "The original adopted fit ownership deadline was exceeded; actual cost was retained.")


class AdoptedProcesses(io.Processes):
    def __init__(self, root, budget, old_sources, workers, own_pins):
        super().__init__(root, budget, old_sources, (2, 3))
        self.workers, self.own_pins = workers, own_pins
        self.gpus["train"] = queue.Queue()
        self.adopted_claimed = set()

    def run(self, task_id, module, request, *, output_file):
        io.verify_sources(self.own_pins)
        if task_id not in self.workers:
            return super().run(task_id, module, request, output_file=output_file)
        io.require(module == "fit_worker", "Adopt only the two originally running fitter operations.")
        with self.lock:
            io.require(task_id not in self.adopted_claimed, "No repeated adoption of an existing fit.")
            self.adopted_claimed.add(task_id)
        worker = self.workers[task_id]
        task = self.budget.value["tasks"][task_id]
        original = io.read(io.bound_path(task["request"]))
        expected = {key: value for key, value in original.items() if key not in ("gpu", "deadline_epoch")}
        io.require(request == expected, "The adopted operation must preserve its actual original full fitter request.")
        io.require(Path(output_file).resolve() == Path(original["output"]).resolve() / "live_raw.json", "Adopt this actual original fit output only.")
        try:
            observation = wait_original(worker, self.cancelled)
        except BaseException as exc:
            observed = dict(pid=worker["pid"], start_ticks=worker["start_ticks"],
                state="original_wait_failed", observed_epoch=time.time(), os_exit_code=None,
                time_basis="Conservative observation-time upper bound after wait failure; no OS exit code inferred.")
            self.budget.finish_adopted(task_id, end=observed["observed_epoch"], receipt=None, observed=observed,
                successful=False, error=type(exc).__name__ + ": " + str(exc), receipt_exit_code=None)
            raise
        end = observation["observed_epoch"]
        completed_path = Path(original["output"]) / "completed.json"
        try:
            io.require(not observation.get("original_deadline_termination"), "The adopted fit exceeded its original finite deadline.")
            completed = io.read(completed_path)
            raw = io.read(io.bound_path(completed["live_raw"]))
            io.require(completed["status"] == raw["status"] == "completed" and completed["exit_code"] == raw["exit_code"] == 0
                and completed["phase"] == raw["phase"] == "formal" and raw["formal_eligible"]
                and completed["request_identity"] == raw["request_identity"] == digest(original)
                and raw["request"] == task["request"] and raw["source_parent_uuid"] == request["parent"]["uuid"]
                and raw["fresh_evidence"]["gpu_validation"] and raw["fresh_evidence"]["same_input_same_BF16_path_logits_exact"]
                and raw["fit_report"]["technical_steps"] is None
                and len(raw["fit_report"]["stages"]) == 2
                and all(stage["actual_steps"] == 10000 for stage in raw["fit_report"]["stages"]),
                "The original fit must finish its own full formal2×10K, actual save and independent GPU fresh proof.")
            io.verify_sources(self.sources)
            io.verify_sources(self.own_pins)
        except BaseException as exc:
            failure_path = Path(original["output"]) / "failed.json"
            receipt_path = completed_path if completed_path.is_file() else failure_path if failure_path.is_file() else None
            self.budget.finish_adopted(task_id, end=end,
                receipt=None if receipt_path is None else io.binding(receipt_path), observed=observation,
                successful=False, error=type(exc).__name__ + ": " + str(exc),
                receipt_exit_code=None if receipt_path is None else io.read(receipt_path).get("exit_code"))
            raise
        self.budget.finish_adopted(task_id, end=end, receipt=io.binding(completed_path), observed=observation)
        self.gpus["train"].put(worker["gpu"])
        return io.binding(output_file)


def wait_original(worker, cancelled):
    """Wait on exact original process identity without claiming a non-child waitpid code."""
    pid, ticks = worker["pid"], worker["start_ticks"]
    actual = proc_start(pid)
    io.require(actual in (None, ticks), "The original fitter PID was reused; do not adopt another process.")
    if actual is None:
        return dict(pid=pid, start_ticks=ticks, state="already_absent_at_adoption", observed_epoch=time.time(), os_exit_code=None,
            time_basis="Conservative adoption-time upper bound; actual process exit time unknown, including any adoption gap cost.")
    try:
        descriptor = pidfd_open(pid)
    except ProcessLookupError:
        return dict(pid=pid, start_ticks=ticks, state="exited_while_opening_pidfd", observed_epoch=time.time(), os_exit_code=None,
            time_basis="Conservative pidfd-acquisition upper bound; actual process exit time unknown.")
    try:
        io.require(proc_start(pid) in (None, ticks), "The fit identity changed during pidfd acquisition.")
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        while True:
            remaining = worker["original_deadline_epoch"] - time.time()
            if remaining <= 0:
                io.require(proc_start(pid) in (None, ticks), "Do not terminate a reused process at the original deadline.")
                if proc_start(pid) is not None:
                    io.require(os.getpgid(pid) == pid, "The adopted fit's actual original process group changed.")
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if not poller.poll(5000):
                    io.require(proc_start(pid) in (None, ticks), "Do not kill a recycled PID at the original deadline.")
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    io.require(poller.poll(5000), "The original bounded fit did not terminate after its own deadline.")
                return dict(pid=pid, start_ticks=ticks, state="original_deadline_terminated_owned_fit",
                    observed_epoch=time.time(), original_deadline_termination=True, os_exit_code=None,
                    time_basis="Original ownership start through bounded termination and actual pidfd terminal observation.")
            if poller.poll(min(1000, max(1, int(remaining * 1000)))):
                return dict(pid=pid, start_ticks=ticks, state="pidfd_exit_observed", observed_epoch=time.time(), os_exit_code=None,
                    time_basis="Original start through actual pidfd terminal observation; OS exit status unavailable for a non-child.")
    finally:
        os.close(descriptor)


def restore_controller(snapshot_path):
    snapshot_path = Path(snapshot_path).resolve()
    snapshot = io.read(snapshot_path)
    io.require(snapshot["schema"] == "sc_cat_two_live_fit_handoff_v1" and snapshot["readout_gpus"] == [2, 3], "Use the explicitly approved exact two-fit2/3 handoff.")
    old = snapshot["old_controller"]
    io.require(proc_start(old["pid"]) is None, "Root must stop only the old controller PID before the new driver may own its state.")
    for reference in snapshot["old_references"].values():
        io.bound_path(reference)
    validate_limited_state(snapshot["state"], snapshot["budget"], snapshot["quota"])
    io.verify_sources(snapshot["old_source_pins"])
    io.verify_sources(snapshot["handoff_source_pins"])
    io.check_resources((2, 3))
    batch = io.load_batch(io.bound_path(snapshot["authorization"]), io.bound_path(snapshot["root_boundary"]))
    controller = Controller.__new__(Controller)
    controller.batch, controller.root = batch, batch["root"] / "formal"
    controller.budget = AdoptedBudget(controller.root / "budget.json", batch["authorization"], snapshot)
    controller.processes = AdoptedProcesses(controller.root, controller.budget, snapshot["old_source_pins"],
        snapshot["workers"], snapshot["handoff_source_pins"])
    controller.quota = restore_quota(snapshot["old_references"]["quota"])
    controller.chain = StrictMainChain(bf16_root_id=batch["root_boundary"]["uuid"], categories=io.ORDER)
    controller.state_lock, controller.deployments = threading.RLock(), {}
    controller.state = deepcopy(snapshot["state"])
    controller.state.update(controller_pid=os.getpid(), readout_gpus=[2, 3],
        controller_handoff=dict(snapshot=io.binding(snapshot_path), old_controller=old,
            actual_new_controller_pid=os.getpid(), old_OS_exit_code=None,
            reason=snapshot["reason"], handoff_source_pins=snapshot["handoff_source_pins"]))
    original = io.read(io.bound_path(snapshot["workers"][FIT_IDS[0]]["request"]))
    controller.common = {key: original[key] for key in ("authorization", "source_manifest", "inventory_receipt", "data_plan_path", "phase")}
    controller.common["execution_policy"] = dict(train_device="cuda:0", attention_implementation="flash_attention_2",
        teacher_model_offload="none", teacher_output_offload="cpu", teacher_output_pin_memory=True,
        teacher_output_chunk_tokens=8, deterministic=False)
    controller._persist("explicit_handoff_adopt_original_fits_readout2_3")
    return controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="component", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--authorization", required=True)
    prepare_parser.add_argument("--root-boundary", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--parent-request")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Keep the handoff driver in caller bitvae with CUDA hidden.")
    if args.component == "prepare":
        prepare(authorization=args.authorization, root_boundary=args.root_boundary, output=args.output, parent_request=args.parent_request)
        return
    controller = None
    try:
        controller = restore_controller(args.snapshot)
        def interrupted(signum, _frame):
            controller.processes.cancel()
            raise InterruptedError("Handoff driver received signal " + str(signum))
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
