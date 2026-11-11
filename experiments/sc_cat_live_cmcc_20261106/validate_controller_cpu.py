"""CPU-only checks of real frozen inputs and finite orchestration contracts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from . import bindings as io
from .run import checked_fit_receipts, quota_record, restore_quota
from .selection import RecoveryQuota


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("An invalid dependency/count/budget request was accepted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--root-boundary", required=True)
    parser.add_argument("--technical-fit", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use confirmed bitvae with CUDA hidden.")
    started = time.time()
    batch = io.load_batch(args.authorization, args.root_boundary)
    paths, categories = checked_fit_receipts(args.technical_fit, require_both=False)
    io.require(categories and len(batch["recipes"]) == 2 and sum(map(len, batch["recipes"].values())) == 4,
        "Check the four actual frozen live recipes and actual completed fit evidence.")
    windows, references = io.technical_windows(paths)
    baseline = RecoverySettings()
    longer = RecoverySettings(schedule_max=1000)
    tasks = {}
    for category in io.ORDER[:2]:
        for role in ("anchor", "different"):
            prefix = category + "/" + role
            tasks.update(io.raw_tasks(prefix))
            tasks.update(io.trajectory_tasks(prefix + "/baseline", baseline))
        tasks.update(io.trajectory_tasks(category + "/third", longer))
    sums = {component: sum(row["seconds"] for row in tasks.values() if row["component"] == component)
        for component in io.COMPONENT_HOURS}
    io.require(sums == dict(technical=0, fit=21600, recovery=57600, evaluation=32400, boundary=0),
        "The entire worst-case4raw/4000KD/36full8 dependency graph must close exactly within its component caps.")
    mirror = io.trajectory_tasks("mirror", baseline)
    io.require(sum(row["seconds"] for row in mirror.values() if row["component"] == "recovery") == 7200
        and len([key for key in mirror if key.endswith("/full8")]) == 4, "A mirrored500 arm retains all four nodes and its complete2h recovery reservation.")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="controller_cpu_", dir=output.parent) as directory:
        directory = Path(directory)
        budget = io.Budget(directory / "budget.json", batch["authorization"], windows, references)
        budget.reserve(tasks, counts=dict(raws=4, trajectories=6, KD_updates=4000, raw_full8=4, node_full8=32))
        rejects(lambda: budget.reserve(io.raw_tasks("extra"), counts=dict(raws=1, raw_full8=1)))
        rejects(lambda: budget.reserve(io.trajectory_tasks("extra", baseline), counts=dict(trajectories=1, KD_updates=500, node_full8=4)))
        rejects(lambda: budget.reserve({"overcap": dict(component="fit", seconds=20)}, counts={}))
        request = directory / "request.json"
        io.write(request, dict(phase="cpu_fixture", purpose="Budget contract timestamps only; no actual GPU execution or quality evidence."))
        task = "k_proj/anchor/fit"
        before = budget.value["total_actual_GPU_hours"]
        now = time.time()
        budget.start(task, gpu=0, start=now - 12, deadline=now + 30,
            command=["cpu_fixture_contract_only"], log=directory / "none.log", request=request)
        rejects(lambda: budget.start("k_proj/different/fit", gpu=0, start=now, deadline=now + 30,
            command=["cpu_fixture_contract_only"], log=directory / "none.log", request=request))
        budget.finish(task, end=now, exit_code=17, error="CPU fixture verifies failed ownership is charged")
        io.require(budget.value["tasks"][task]["status"] == "failed"
            and abs((budget.value["total_actual_GPU_hours"] - before) * 3600 - 12) < 1e-7,
            "Failure must retain its actual cost and free only the unused reservation.")
        io.require(io.read(budget.path)["tasks"][task]["exit_code"] == 17, "Actual budget/exit state must be durable.")
        cpu_source = directory / "source.py"
        cpu_source.write_text("value = 1\n", encoding="utf-8")
        pins = {str(cpu_source): io.binding(cpu_source)["sha256"]}
        io.verify_sources(pins)
        cpu_source.write_text("value = 2\n", encoding="utf-8")
        rejects(lambda: io.verify_sources(pins))
        rejects(lambda: io.command("unapproved_worker", request))
        from .validate_selection_cpu import trajectory
        quota = RecoveryQuota(io.ORDER)
        baselines = [trajectory("k_proj", raw="fixture_anchor", values=(.6, .65, .7, .8)),
            trajectory("k_proj", raw="fixture_other", anchor=False, values=(.6,) * 4)]
        for row in baselines:
            quota.register_start(category_id="k_proj", raw_id=row.raw_id, settings=row.settings, kind="baseline")
        arm = quota.decide(baselines, prefix_targets=["k"], current_targets=["k"], future_dense_targets=["o"])
        quota.register_start(category_id="k_proj", raw_id=arm.raw_id, settings=arm.settings, kind="third")
        extra = trajectory("k_proj", raw=arm.raw_id, settings=arm.settings, values=(.4,) * 8)
        quota.select_completed_category([*baselines, extra])
        quota_path = directory / "quota.json"
        io.write(quota_path, quota_record(quota))
        restored = restore_quota(io.binding(quota_path))
        io.require(quota_record(restored) == quota_record(quota) and restored.long_categories == ["k_proj"]
            and restored.selected_categories == ["k_proj"], "A later batch must restore the consumed long slot and selected prefix.")
    quota = quota_record(RecoveryQuota(io.ORDER))
    io.require(quota["categories"] == list(io.ORDER) and quota["long_categories"] == []
        and quota["interaction_categories"] == [] and quota["selected_categories"] == [],
        "The initial durable quota belongs to the full seven-category chain.")
    import torch
    io.require(not torch.cuda.is_initialized(), "CPU controller validation must not initialize CUDA.")
    io.write(output, dict(status="completed_controller_CPU_contract_validation", exit_code=0, phase="cpu_fixture",
        CUDA_initialized=False, actual_four_recipes=[dict(category_id=category, candidate_id=row["candidate_id"],
            is_anchor=row["is_anchor"], source=row["source"]) for category, rows in batch["recipes"].items() for row in rows],
        actual_fit_technical_receipts=references, actual_fit_categories=sorted(categories),
        worst_case_reserved_seconds_by_component=sums,
        checks=dict(complete_dependency_budget=True, batch_count_caps=True, failed_window_cost_persisted=True,
            same_gpu_overlap_rejected=True, new_code_pin_change_rejected=True, global_quota_serialized=True,
            consumed_quota_roundtrip_preserved=True,
            no_GPU_launched=True), elapsed_seconds=time.time() - started,
        implementation_sources={name: io.binding(Path(__file__).with_name(name)) for name in
            ("bindings.py", "run.py", "validate_controller_cpu.py")},
        boundary="CPU orchestration/input/accounting checks only. No formal fit, KD, deployment, full8 quality or complete k/o result is claimed."))


if __name__ == "__main__":
    main()
