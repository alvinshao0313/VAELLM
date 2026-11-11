"""Snapshot only the current two baseline trainers before any node dispatch."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.cmcc_runtime_20261030.sequence import proc_start
from experiments.sc_cat_live_cmcc_20261106 import bindings as io


TRAIN_IDS = ("k_proj/anchor/baseline/train", "k_proj/different/baseline/train")
COUNTS = dict(raws=2, trajectories=2, KD_updates=1000, raw_full8=2, node_full8=8)


def expected_tasks():
    tasks = {}
    for role in ("anchor", "different"):
        prefix = "k_proj/" + role
        tasks.update(io.raw_tasks(prefix))
        tasks.update(io.trajectory_tasks(prefix + "/baseline", RecoverySettings()))
    return tasks


def validate_state(state, budget, quota):
    io.require(state["status"] == "running" and state["current_category"] == "k_proj" and state["chain"] == []
        and set(state["categories"]) == {"k_proj"} and state["categories"]["k_proj"]["status"] == "running"
        and len(state["categories"]["k_proj"]["raws"]) == 2 and state["categories"]["k_proj"]["trajectories"] == {}
        and state["readout_gpus"] == [2, 3] and state["train_gpus"] == [0, 1],
        "Prepare only the current two k baselines after both complete raw evaluations and before node dispatch.")
    io.require(state["quota"] == quota and quota["selected_categories"] == [] and quota["decisions"] == {}
        and quota["baselines"] == {} and quota["long_categories"] == [] and quota["interaction_categories"] == []
        and len(quota["started"]) == 2 and all(row["kind"] == "baseline" and row["category_id"] == "k_proj"
            and row["settings"] == RecoverySettings().__dict__ for row in quota["started"]),
        "Keep the original two registered baselines and unspent full-model optional quotas.")
    io.require(budget["counter_reserved"] == COUNTS and set(budget["tasks"]) == set(expected_tasks()),
        "The fixed original k dependency graph/counts must remain unchanged.")
    for name, specification in expected_tasks().items():
        row = budget["tasks"][name]
        status = "running" if name in TRAIN_IDS else "reserved" if "/baseline/node_" in name else "completed"
        io.require(row["status"] == status and all(row[key] == value for key, value in specification.items()),
            "A node operation already started or the original reservation changed: " + name)
    io.require({budget["tasks"][name]["gpu"] for name in TRAIN_IDS} == {0, 1}, "Keep each actual original trainer on its own physical GPU.")


def runtime_sources(state):
    sources = io.read(io.bound_path(state["source_pins"]))
    previous = state.get("controller_handoff")
    if previous is not None:
        sources.update(previous["handoff_source_pins"])
    io.verify_sources(sources)
    return sources


def prepare(*, authorization, root_boundary, output):
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use caller bitvae with CUDA hidden.")
    batch = io.load_batch(authorization, root_boundary)
    root = batch["root"] / "formal"
    paths = {key: root / name for key, name in dict(state="controller_state.json", budget="budget.json", quota="quotas.json").items()}
    references = {key: io.binding(path) for key, path in paths.items()}
    values = {key: io.read(path) for key, path in paths.items()}
    validate_state(values["state"], values["budget"], values["quota"])
    sources = runtime_sources(values["state"])
    old_pid = values["state"]["controller_pid"]
    old_ticks = proc_start(old_pid)
    io.require(old_ticks is not None, "Capture the current live controller before root stops only its single PID.")
    workers = {}
    for name in TRAIN_IDS:
        task = values["budget"]["tasks"][name]
        pid, ticks = task["pid"], proc_start(task["pid"])
        request_path = io.bound_path(task["request"])
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        io.require(ticks is not None and os.getpgid(pid) == pid and os.fsencode(str(request_path)) in arguments,
            "Capture only the actual independently grouped original trainer process.")
        workers[name] = dict(pid=pid, start_ticks=ticks, gpu=task["gpu"], request=task["request"],
            original_started_epoch=task["started_epoch"], original_deadline_epoch=task["deadline_epoch"])
    io.require(all(io.binding(path) == references[key] for key, path in paths.items()),
        "The controller advanced during snapshot preparation; do not stop it using this snapshot.")
    output = Path(output).resolve()
    io.require(output.is_relative_to(batch["root"]) and not output.exists(), "Keep a new immutable snapshot inside this authorized batch.")
    io.write(output, dict(schema="sc_cat_two_baseline_pipeline_handoff_v1", phase="formal", created_epoch=time.time(),
        authorization=batch["authorization_reference"], root_boundary=batch["root_boundary_reference"],
        reason="Overlap published immutable node exports/full8 on2/3 with unchanged baseline and later approved training on0/1.",
        old_controller=dict(pid=old_pid, start_ticks=old_ticks), old_references=references,
        state=values["state"], budget=values["budget"], quota=values["quota"], workers=workers,
        old_runtime_source_pins=sources, readout_gpus=[2, 3], train_gpus=[0, 1], never_use_gpus=[4, 5, 6, 7],
        preparation_source=io.binding(__file__), process_stop_performed=False, GPU_task_started=False))
    print(dict(status="pipeline_snapshot_ready_no_process_stopped", snapshot=io.binding(output), old_controller_pid=old_pid,
        workers=workers, original_counts=values["budget"]["counter_reserved"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--root-boundary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(authorization=args.authorization, root_boundary=args.root_boundary, output=args.output)


if __name__ == "__main__":
    main()
