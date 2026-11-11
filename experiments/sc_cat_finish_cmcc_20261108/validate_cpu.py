"""Actual-document/node/budget closure; no model or CUDA fixture."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

import torch

from experiments.sc_cat_continue_cmcc_20261107 import run as original
from experiments.sc_cat_pipeline_cmcc_20261106.node_export import read_committed_node
from . import frozen
from .resume import equal_tree, write_proof
from .worker import wrapped_command, is_resume_train


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshot = frozen.install(args.snapshot)
    io = frozen.io
    checks = {}
    auth = io.read_authorization(snapshot["authorization"])
    checks["unchanged_actual_approval_and_four_cards"] = auth["gpus"] == [0, 1, 2, 3] and auth["total_GPU_hours"] < 208.5
    for name, pair in snapshot["document_versions"].items():
        checks[name + "_exact_old_bytes_relocated"] = io.binding(io.bound_path(pair["original"]))["sha256"] == pair["original"]["sha256"]
    try:
        io.bound_path(dict(snapshot["document_versions"]["protocol"]["original"], sha256="0" * 64))
    except ValueError:
        checks["unknown_binding_still_rejected"] = True
    else:
        checks["unknown_binding_still_rejected"] = False
    state = io.read(io.bound_path(snapshot["state"]))
    checks["six_real_categories_complete"] = len(state["chain"]) == 6 and all(state["categories"][x["selected"]["category_id"]]["status"] == "completed" for x in state["chain"])
    from experiments.sc_cat_live_cmcc_20261106.run import restore_quota
    quota = restore_quota(snapshot["quota"])
    checks["global_quotas_preserved"] = quota.long_categories == ["v_proj", "down_proj"] and quota.interaction_categories == ["k_proj", "o_proj"]
    checks["both_up_baselines_complete"] = len(quota.baselines["up_proj"]) == 2 and all(len(row.nodes) == 4 for row in quota.baselines["up_proj"])
    nodes = [read_committed_node(snapshot["training_request"], step) for step in (125, 250)]
    checks["both_prefix_native_commits_complete"] = all(nodes) and all(
        nodes[-1]["checkpoint_meta"][key] == snapshot["checkpoint250"][key] for key in ("path", "sha256"))
    cp = Path(nodes[-1]["node_directory"])
    optimizer = torch.load(cp / "optimizer.pt", map_location="cpu", weights_only=False)
    scheduler = torch.load(cp / "scheduler.pt", map_location="cpu", weights_only=False)
    rng = torch.load(cp / "rng_state.pth", map_location="cpu", weights_only=False)
    states = list(optimizer["state"].values())
    checks["actual_saved_Adam_and_schedule_at250"] = bool(states) and all(int(v["step"]) == 250 for v in states) and scheduler["last_epoch"] == 250
    checks["actual_saved_CPU_CUDA_RNG_present"] = isinstance(rng["cpu"], torch.Tensor) and "cuda" in rng
    checks["tensor_exact_comparator_actual_state"] = equal_tree(optimizer, deepcopy(optimizer))
    changed = deepcopy(optimizer)
    changed["state"][next(iter(changed["state"]))]["step"] += 1
    checks["changed_optimizer_step_rejected"] = not equal_tree(optimizer, changed)
    writer_path = Path(args.output).with_suffix(".writer.json")
    write_proof(writer_path, dict(status="state_loaded"))
    write_proof(writer_path, dict(status="RNG_verified"))
    checks["resume_progress_writer_can_update"] = io.read(writer_path)["status"] == "RNG_verified"
    writer_path.unlink()
    old = snapshot["budget_value"]
    checks["failed_cost_kept_not_reset"] = old["total_actual_GPU_hours"] >= 39.1954 and old["tasks"]["up_proj/third/train"]["actual_seconds"] > 0
    remaining = {key: value["seconds"] - value.get("actual_seconds", 0.) for key, value in old["tasks"].items()
        if key.startswith("up_proj/third/") and value["status"] != "completed"}
    checks["same_original_remaining_windows"] = all(x > 15 for x in remaining.values()) and old["counter_reserved"]["KD_updates"] == 8500
    from .run import RestoredQuota
    restored = RestoredQuota(quota, None)
    before = len(restored.started)
    arm = restored.decisions["up_proj"]
    restored.register_start(category_id="up_proj", raw_id=arm.raw_id, settings=arm.settings, kind="third")
    checks["third_arm_registration_reused_without_new_slot"] = restored.resumed_registration and len(restored.started) == before
    argv = original.command("recovery_worker", "/actual/request.json")
    wrapped = wrapped_command(argv, args.snapshot)
    checks["nested_fresh_uses_bound_document_bootstrap"] = ("experiments.sc_cat_finish_cmcc_20261108.worker" in wrapped
        and "--snapshot" in wrapped and "experiments.sc_cat_continue_cmcc_20261107.workers" in wrapped
        and wrapped[-1] == "/actual/request.json")
    repair = Path(snapshot["document_versions"]["protocol"]["snapshot"]["path"]).parent
    fresh = repair / "formal/artifacts/up_proj/third_resume/node_250/export/fresh_request.json"
    if fresh.exists():
        request = io.read(fresh)
        checks["actual_fresh_with_inherited_resume_flag_is_not_train"] = bool(request.get("resume_checkpoint")) and not is_resume_train(request)
    checks["actual_original_train_only_resumes_when_explicit"] = is_resume_train(dict(io.read(io.bound_path(snapshot["training_request"])), resume_checkpoint=snapshot["checkpoint250"]))
    checks["CPU_only"] = not torch.cuda.is_initialized()
    io.require(all(checks.values()), "The actual repair CPU contract failed: " + str(checks))
    io.write(args.output, dict(status="completed", exit_code=0, phase="cpu_fixture", snapshot=io.binding(args.snapshot),
        checks=checks, elapsed_seconds=time.monotonic() - start, repair_source_pins=snapshot["repair_source_pins"],
        CUDA_initialized=torch.cuda.is_initialized(), remaining_original_task_seconds=remaining))
    print(dict(status="completed", checks=len(checks), output=args.output), flush=True)


if __name__ == "__main__":
    main()
