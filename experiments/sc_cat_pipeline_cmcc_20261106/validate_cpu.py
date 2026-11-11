"""Validate the fixed live-baseline handoff and streamed-node selection contracts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile

from experiments.sc_cat_live_cmcc_20261106 import bindings as io
from experiments.sc_cat_live_cmcc_20261106.run import restore_quota
from .handoff import COUNTS, TRAIN_IDS, expected_tasks, validate_state
from .run import PipelineBudget, PipelineProcesses, _command, _finite_run, exact_node_closure, own_sources, remember_node


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected pipeline-state rejection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--node-reader-validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only caller bitvae.")
    snapshot, reader_proof = io.read(args.snapshot), io.read(args.node_reader_validation)
    validate_state(snapshot["state"], snapshot["budget"], snapshot["quota"])
    io.verify_sources(snapshot["old_runtime_source_pins"])
    for reference in snapshot["old_references"].values():
        io.bound_path(reference)
    io.require(reader_proof["exit_code"] == 0, "Reuse the successful actual technical-node reader verification.")
    authorization = io.read(io.bound_path(snapshot["authorization"]))
    output = Path(args.output).resolve()
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        directory = Path(directory)
        budget = PipelineBudget(directory / "budget.json", authorization, snapshot["budget"])
        counts, reserved = deepcopy(budget.value["counter_reserved"]), deepcopy(budget.value["reserved_seconds_by_component"])
        rejects(lambda: budget.reserve(expected_tasks(), counts=dict(COUNTS, trajectories=3)))
        budget.reserve(expected_tasks(), counts=COUNTS)
        io.require(budget.value["counter_reserved"] == counts and budget.value["reserved_seconds_by_component"] == reserved,
            "Reuse all six completed raw stages and the exact full dependency reservation without another count/charge.")
        rejects(lambda: budget.reserve(expected_tasks(), counts=COUNTS))
        processes = PipelineProcesses(directory, budget, snapshot["old_runtime_source_pins"], snapshot["workers"], own_sources())
        io.require(processes.gpus["train"].empty()
            and tuple(processes.gpus["readout"].get_nowait() for _ in range(2)) == (2, 3)
            and processes.gpus["readout"].empty(), "Neither original running train GPU is available; use only readout2/3.")
        before_actual = budget.value["actual_seconds_by_component"]["recovery"]
        first = snapshot["budget"]["tasks"][TRAIN_IDS[0]]
        observed = dict(phase="cpu_fixture", observed_epoch=first["started_epoch"] + 400, os_exit_code=None)
        budget.settle_adopted(TRAIN_IDS[0], observed=observed,
            receipt=dict(phase="cpu_fixture", purpose="Accounting contract only, no live trainer terminal claim."),
            successful=True, receipt_exit_code=0)
        io.require(budget.value["tasks"][TRAIN_IDS[0]]["os_exit_code"] is None
            and budget.value["tasks"][TRAIN_IDS[0]]["exit_code"] is None
            and budget.value["actual_seconds_by_component"]["recovery"] == before_actual + 400
            and budget.value["counter_reserved"] == counts, "One original full training window is charged once with unknown OS status preserved.")
        rejects(lambda: budget.settle_adopted(TRAIN_IDS[0], observed=observed, receipt=None, successful=True))
        second = snapshot["budget"]["tasks"][TRAIN_IDS[1]]
        budget.settle_adopted(TRAIN_IDS[1], observed=dict(phase="cpu_fixture", observed_epoch=second["started_epoch"] + 430),
            receipt=None, successful=False, error="CPU fixture failure accounting")
        io.require(budget.value["actual_seconds_by_component"]["recovery"] == before_actual + 830
            and budget.value["tasks"][TRAIN_IDS[1]]["status"] == "failed", "A failed adopted trajectory cannot vanish from its original cost.")
        quota_path = directory / "quotas.json"
        io.write(quota_path, snapshot["quota"])
        quota = restore_quota(io.binding(quota_path))
        io.require(len(quota.started) == 2 and quota.counts() == snapshot["quota"]["counts"], "Keep both already-registered baselines and unchanged full-model quotas.")
        rejects(lambda: quota.register_start(category_id=quota.started[0]["category_id"], raw_id=quota.started[0]["raw_id"],
            settings=quota.started[0]["settings"], kind="baseline"))
    actual_technical = io.read(io.bound_path(io.read(io.bound_path(snapshot["budget"]["technical_receipts"][-1]))["results"]["k_recovery"]))
    node = actual_technical["nodes"][0]
    receipts = {}
    remember_node(receipts, node["step"], node)
    rejects(lambda: remember_node(receipts, node["step"], node))
    exact_node_closure(receipts[node["step"]], deepcopy(node))
    changed = deepcopy(node)
    changed["source_node_id"] = "wrong_final_native_UUID"
    rejects(lambda: exact_node_closure(receipts[node["step"]], changed))
    changed = deepcopy(node)
    changed["checkpoint_meta"]["sha256"] = "wrong_final_native_meta_SHA"
    rejects(lambda: exact_node_closure(receipts[node["step"]], changed))
    invalid = deepcopy(snapshot["budget"])
    invalid["tasks"]["k_proj/anchor/baseline/node_125/export"]["status"] = "running"
    rejects(lambda: validate_state(snapshot["state"], invalid, snapshot["quota"]))
    io.require(io.Processes.run.__globals__["command"] is io.command
        and _finite_run.__globals__["command"] is _command
        and "experiments.sc_cat_pipeline_cmcc_20261106.node_export" in _command("node_export", "/tmp/cpu_request.json"),
        "The private runner extends this driver's command only, without mutating frozen original module globals.")
    import torch
    io.require(not torch.cuda.is_initialized(), "No model/GPU short test is part of pipeline wiring verification.")
    io.write(output, dict(status="completed", exit_code=0, phase="cpu_fixture", snapshot=io.binding(args.snapshot),
        node_reader_validation=io.binding(args.node_reader_validation), pipeline_source_pins=own_sources(),
        checks=dict(exact_initial_state=True, original_sources_unchanged=True, six_completed_raw_stages_reused=True,
            reservation_and_counts_not_doubled=True, baseline_registration_preserved=True, failure_cost_retained=True,
            OS_status_not_invented=True, train_cards_unavailable_until_terminal=True, readout_exactly2_3=True,
            committed_node_duplicate_rejected=True, final_UUID_and_meta_mismatch_rejected=True,
            immutable_original_runner_globals=True, advanced_state_rejected=True, CUDA_not_initialized=True),
        boundary="CPU state/accounting/duplicate/final-receipt contracts plus already verified actual technical-node reader evidence; no GPU or formal quality claim."))


if __name__ == "__main__":
    main()
