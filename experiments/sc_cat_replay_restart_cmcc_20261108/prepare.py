"""Verify the real stopped artifacts and bound remaining windows on CPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

import torch

from experiments.sc_cat_finish_cmcc_20261108 import frozen
from experiments.sc_cat_finish_cmcc_20261108.run import RestoredBudget
from .run import io, remaining_tasks, PREFIX


def prepare(previous_root, output_root):
    previous_root, output_root = Path(previous_root).resolve(), Path(output_root).resolve()
    snapshot_path = previous_root.parent / "formal_snapshot.json"
    snapshot = frozen.install(snapshot_path)
    state_reference = io.binding(previous_root / "controller_state.json")
    state = io.read(io.bound_path(state_reference))
    budget_reference = io.binding(previous_root / "budget.json")
    budget = io.read(io.bound_path(budget_reference))
    terminal_reference = io.binding(previous_root / "controller.exit.json")
    terminal = io.read(io.bound_path(terminal_reference))
    auth = io.read_authorization(snapshot["authorization"])
    manifest = io.read(io.bound_path(state["replay_manifest"]))
    request_reference = io.binding(previous_root / "execution/replay/o_proj/export/request.json")
    request = io.read(io.bound_path(request_reference))
    result = io.read(io.bound_path(request["replay_result"]))
    raw = io.read(io.bound_path(request["live_raw"]))
    boundary = io.read(io.bound_path(request["parent"]))
    node, = [x for x in result["nodes"] if x["step"] == 125]
    io.bound_path(node["checkpoint_meta"])
    for entry in node["native_files"].values():
        io.require(Path(entry["path"]).is_file() and Path(entry["path"]).stat().st_size == entry["bytes"],
            "The saved o125 payload is missing or truncated.")
    raw_model = Path(raw["round_base_dir"]) / "pytorch_model.bin"
    io.require(raw_model.is_file() and raw_model.stat().st_size == 15173525333,
        "Retain the actual completed o raw; do not substitute or refit it.")
    pending = remaining_tasks(auth, manifest, budget)
    checks = dict(cpu_only=not torch.cuda.is_initialized(),
        failed_terminal=terminal["status"] == state["status"] == "failed" and terminal["state"] == state_reference,
        seven_classes_and_terminal_nine=len(state["chain"]) == 7 and io.read(io.bound_path(state["core_nine_task"]))["exit_code"] == 0,
        k_complete=len(state["replay_stages"]) == 1 and state["replay_stages"][0]["category_id"] == "k_proj"
            and request["parent"] == state["replay_stages"][0]["boundary"] and boundary["completed_prefix"] == ["k_proj"],
        o_training_complete=result["exit_code"] == 0 and result["selected_stop"] == 125 and result["original_schedule_max"] == 500
            and result["live_raw"]["round_base_uuid"] == raw["round_base_uuid"],
        same_manifest=request["replay_manifest"] == state["replay_manifest"] and result["manifest_id"] == manifest["manifest_id"],
        no_new_selection=not manifest["repeat_search"] and not manifest["repeat_node_selection"] and manifest["replay_KD_updates"] == 3500,
        no_running_predecessor=not any(x["status"] == "running" for x in budget["tasks"].values()),
        failed_export_budget_subtracted=pending[PREFIX + "o_proj/export"]["seconds"] ==
            auth["task_seconds"]["replay_export"] - budget["tasks"]["replay/o_proj/export"]["actual_seconds"],
        authorized_output=output_root.is_relative_to(Path(auth["output_root"]).resolve()) and not (output_root / "formal").exists(),
        original_gpu_scope=auth["gpus"] == [0, 1, 2, 3])
    with tempfile.TemporaryDirectory(prefix="sc_cat_replay_budget_") as temp:
        resumed = RestoredBudget(Path(temp) / "budget.json", auth, budget)
        resumed.reserve(pending, counts={})
        checks["cost_and_counters_preserved"] = resumed.value["counter_reserved"] == budget["counter_reserved"] and \
            resumed.value["total_actual_GPU_hours"] == budget["total_actual_GPU_hours"]
        bad = deepcopy(budget)
        bad["tasks"]["replay/o_proj/export"]["status"] = "running"
        try:
            remaining_tasks(auth, manifest, bad)
        except ValueError:
            checks["reject_running_task"] = True
        else:
            checks["reject_running_task"] = False
    # Actual analogous category payloads total under110GiB, all old trainer states
    # under12GiB; allow a dense export temporary and further evidence files.
    storage_estimate = 160 * 2**30
    checks["storage_available"] = shutil.disk_usage(output_root).free >= storage_estimate
    io.require(all(checks.values()), "Replay restart CPU/input checks failed: " + json.dumps(checks))
    record = dict(schema="sc_cat_explicit_replay_restart_v1", status="validated", exit_code=0, checks=checks,
        user_instruction="重新启动吧", original_snapshot=io.binding(snapshot_path), previous_state=state_reference,
        previous_budget=budget_reference, previous_exit=terminal_reference, o_export_request=request_reference,
        output_root=str(output_root / "formal"), estimated_additional_storage_bytes=storage_estimate,
        available_bytes=shutil.disk_usage(output_root).free,
        implementation={str(p.resolve()): io.binding(p)["sha256"] for p in Path(__file__).parent.glob("*.py")},
        next_task=PREFIX + "o_proj/export", remaining_task_count=len(pending),
        additional_candidates=0, additional_replay_updates=3500 - 250 - 125,
        prior_actual_GPU_hours=budget["total_actual_GPU_hours"],
        cleanup=io.binding(output_root / "cleanup.json"))
    io.write(output_root / "restart.json", record)
    print(json.dumps(dict(status=record["status"], checks=checks, remaining_tasks=len(pending),
        remaining_updates=record["additional_replay_updates"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    prepare(args.previous_root, args.output_root)


if __name__ == "__main__":
    main()
