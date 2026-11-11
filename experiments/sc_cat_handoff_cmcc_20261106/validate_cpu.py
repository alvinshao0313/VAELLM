"""Read actual handoff state; test restored reservations/accounting without GPU work."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import time

from experiments.sc_cat_live_cmcc_20261106 import bindings as io
from experiments.sc_cat_live_cmcc_20261106.run import restore_quota
from .run import AdoptedBudget, AdoptedProcesses, COUNTS, FIT_IDS, own_sources, validate_limited_state, wait_original
from experiments.cmcc_runtime_20261030.sequence import proc_start


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected invalid handoff rejection.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-request", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only confirmed environment.")
    request = io.read(args.parent_request)
    state, original = request["before_state"], request["before_budget"]
    expected = validate_limited_state(state, original, state["quota"])
    pins = io.read(io.bound_path(state["source_pins"]))
    io.verify_sources(pins)
    authorization = io.read(args.authorization)
    output = Path(args.output)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        directory = Path(directory)
        snapshot = dict(budget=original)
        budget = AdoptedBudget(directory / "budget.json", authorization, snapshot)
        original_counts, original_reserved = deepcopy(original["counter_reserved"]), deepcopy(original["reserved_seconds_by_component"])
        rejects(lambda: budget.reserve(expected, counts=dict(COUNTS, KD_updates=1500)))
        budget.reserve(expected, counts=COUNTS)
        io.require(budget.value["counter_reserved"] == original_counts
            and budget.value["reserved_seconds_by_component"] == original_reserved, "The existing dependency group must never be counted twice.")
        rejects(lambda: budget.reserve(expected, counts=COUNTS))
        processes = AdoptedProcesses(directory, budget, pins, {}, own_sources())
        io.require(processes.gpus["train"].empty()
            and tuple(processes.gpus["readout"].get_nowait() for _ in range(2)) == (2, 3)
            and processes.gpus["readout"].empty(), "Neither in-flight training card may be allocated; readout is exactly2/3.")
        task = original["tasks"][FIT_IDS[0]]
        budget.finish_adopted(FIT_IDS[0], end=task["started_epoch"] + 300,
            receipt=dict(phase="cpu_fixture", purpose="Accounting rule only; no formal worker completion asserted."),
            observed=dict(phase="cpu_fixture", os_exit_code=None))
        finished = budget.value["tasks"][FIT_IDS[0]]
        io.require(finished["status"] == "completed" and finished["os_exit_code"] is None
            and finished["exit_code"] is None and finished["worker_receipt_exit_code"] == 0
            and finished["actual_seconds"] == 300 and finished["started_epoch"] == task["started_epoch"]
            and budget.value["actual_seconds_by_component"]["fit"] == 300
            and budget.value["reserved_seconds_by_component"]["fit"] == 6300
            and budget.value["counter_reserved"] == original_counts,
            "Settle one original ownership window once, preserving unknown OS status and unchanged counts.")
        rejects(lambda: budget.finish_adopted(FIT_IDS[0], end=task["started_epoch"] + 300, receipt=None, observed={}))
        second = original["tasks"][FIT_IDS[1]]
        budget.finish_adopted(FIT_IDS[1], end=second["started_epoch"] + 320, receipt=None,
            observed=dict(phase="cpu_fixture", os_exit_code=None), successful=False, error="CPU fixture failure accounting", receipt_exit_code=None)
        io.require(budget.value["actual_seconds_by_component"]["fit"] == 620
            and budget.value["tasks"][FIT_IDS[1]]["status"] == "failed", "A failed adopted fit retains its original cost too.")
        quota_path = directory / "quota.json"
        io.write(quota_path, state["quota"])
        quota = restore_quota(io.binding(quota_path))
        io.require(quota.selected_categories == [] and quota.started == [] and quota.decisions == {}, "Do not fabricate past recovery decisions.")
        changed = deepcopy(original)
        changed["tasks"]["k_proj/anchor/raw_export"]["status"] = "running"
        rejects(lambda: validate_limited_state(state, changed, state["quota"]))
    for duration, deadline, forced in ((.2, 10, False), (30, .2, True)):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(' + str(duration) + ')'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        observed = wait_original(dict(pid=child.pid, start_ticks=proc_start(child.pid),
            original_deadline_epoch=time.time() + deadline), threading.Event())
        code = child.wait(timeout=1)
        io.require(observed['os_exit_code'] is None and bool(observed.get('original_deadline_termination')) == forced
            and (code < 0 if forced else code == 0), "CPU subprocess verifies actual pidfd wait and own-process deadline termination without guessing OS status.")
    import torch
    io.require(not torch.cuda.is_initialized(), "No actual GPU validation is part of handoff wiring.")
    io.write(output, dict(status="completed", exit_code=0, phase="cpu_fixture", parent_request=io.binding(args.parent_request),
        checks=dict(actual_exact_limited_state=True, original_pins_unchanged=True, original_reservation_reused_once=True,
            all_original_counts_unchanged=True, original_fit_window_charged_once=True, failed_fit_charged=True,
            unknown_OS_exit_preserved=True, train_pool_empty_until_fit_verified=True, readout_exactly2_3=True,
            quota_restored=True, advanced_state_rejected=True, no_CUDA_initialized=True), handoff_source_pins=own_sources(),
        boundary="CPU adoption/accounting contracts and actual unchanged handoff records only; original fits were never stopped, rerun or treated as completed."))


if __name__ == "__main__":
    main()
