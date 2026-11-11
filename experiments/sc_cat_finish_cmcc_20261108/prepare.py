"""Bind the actual failed state and exact historical document bytes once."""
from __future__ import annotations

import argparse
from pathlib import Path
from copy import deepcopy

from experiments.sc_cat_continue_cmcc_20261107 import bindings as io
from .frozen import own_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--technical", action="store_true")
    parser.add_argument("--previous-budget")
    args = parser.parse_args()
    root = Path(args.original_root).resolve()
    output = Path(args.output).resolve()
    repair = root / "repair_20261108"
    state = io.read(root / "formal/controller_state.json")
    budget_path = root / "formal/budget.json" if args.previous_budget is None else Path(args.previous_budget).resolve()
    io.require(budget_path.is_relative_to(root), "Only inherit an actually owned budget of this same approved batch.")
    budget = io.read(budget_path)
    auth = io.read(root / "authorization.json")
    io.require(state["status"] == "failed" and len(state["chain"]) == 6 and state["current_category"] == "up_proj"
        and not Path("/proc", str(state["controller_pid"])).exists(), "Only prepare this actually terminated up250 state.")
    original = budget["tasks"]["up_proj/third/train"]
    request = io.read(io.bound_path(original["request"]))
    node125 = root / "formal/artifacts/up_proj/third/node_125"
    value = dict(schema="sc_cat_up250_finish_v1", authorization=io.binding(root / "authorization.json"),
        state=io.binding(root / "formal/controller_state.json"), budget=io.binding(budget_path),
        quota=io.binding(root / "formal/quotas.json"), exit=io.binding(root / "formal/controller.exit.json"),
        training_request=original["request"], checkpoint250=io.binding(Path(request["output"]) / "trainer_state/checkpoint-250/checkpoint_meta.json"),
        node125_export=io.binding(node125 / "export/deployment_report.json"), node125_score=io.binding(node125 / "full8/score.json"),
        budget_value=deepcopy(budget), original_source_pins=state["source_pins"], repair_source_pins=own_sources(),
        document_versions={name: dict(original=auth[name], snapshot=io.binding(repair / (name + ".md")))
            for name in ("protocol", "execution_protocol")}, output_root=str(output.parent),
        resume_prefix="up_proj/" + output.parent.name, recovered_node250_export=None,
        unchanged_approval=True, new_candidate_count=0, additional_formal_updates=250,
        new_grid_or_protocol_adoption=False, technical_proof=None)
    if args.technical:
        technical = io.read(repair / "technical_resume_v2.json")
        io.require(technical["status"] == "completed" and technical["exit_code"] == 0 and technical["new_optimizer_updates"] == 2,
            "The actual native resume diagnostic must pass before formal continuation.")
        value["technical_proof"] = io.binding(repair / "technical_resume_v2.json")
        value["budget_value"]["tasks"]["repair/native_resume/technical"] = dict(component="technical", kind="native_resume_validation",
            seconds=1800, status="completed", gpu=0, actual_seconds=technical["actual_seconds"], exit_code=0,
            started_epoch=technical["started_epoch"], ended_epoch=technical["ended_epoch"], receipt=value["technical_proof"])
        first = io.read(repair / "technical_execution.json")
        value["budget_value"]["tasks"]["repair/native_resume/initial_writer_failure"] = dict(component="technical", kind="native_resume_validation",
            seconds=1800, status="failed", gpu=0, actual_seconds=first["actual_seconds"], exit_code=first["exit_code"],
            started_epoch=first["started_epoch"], ended_epoch=first["ended_epoch"], receipt=io.binding(repair / "technical_execution.json"),
            reason="Saved-state and RNG checks passed; the initial diagnostic used an immutable writer for a changing progress receipt, before any new update.")
        recovered = repair / "fresh_validation/recovered_node250_export.json"
        if recovered.exists():
            execution = io.read(repair / "fresh_validation/execution.json")
            io.require(execution["status"] == "completed" and execution["exit_code"] == 0,
                "Only reuse the actually completed native independent-fresh repair.")
            value["recovered_node250_export"] = io.binding(recovered)
            value["budget_value"]["tasks"]["repair/native_fresh/technical"] = dict(component="technical", kind="native_fresh_validation",
                seconds=450, status="completed", gpu=2, actual_seconds=execution["actual_seconds"], exit_code=0,
                started_epoch=execution["started_epoch"], ended_epoch=execution["ended_epoch"],
                receipt=io.binding(repair / "fresh_validation/execution.json"))
    io.require(output.is_relative_to(repair) and not output.exists(), "Keep a new snapshot within this same approved batch.")
    io.write(output, value)
    print(dict(status="prepared_bound_repair", snapshot=io.binding(output), technical_verified=bool(value["technical_proof"])), flush=True)


if __name__ == "__main__":
    main()
