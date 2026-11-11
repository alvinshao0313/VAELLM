"""Adopt six completed categories and up baselines, finish only the missing graph."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import signal
import threading
import time
import traceback
import types

from experiments.cat_auto_protocol_cmcc_20261104.nodes import FullEightScore, NodeScore, RecoverySettings, TrajectoryResult
from experiments.sc_cat_continue_cmcc_20261107 import run as original
from experiments.sc_cat_continue_cmcc_20261107 import bindings as io
from experiments.sc_cat_continue_cmcc_20261107.selection import ContinuationQuota
from experiments.sc_cat_live_cmcc_20261106.boundary import StrictMainChain, complete_boundary_reference
from experiments.sc_cat_live_cmcc_20261106.evaluate import read_score
from experiments.sc_cat_live_cmcc_20261106.run import restore_quota
from experiments.sc_cat_live_cmcc_20261106.selection import selected_category
from . import frozen
from .worker import wrapped_command


def trajectory(reference):
    value = io.read(io.bound_path(reference))
    def score(row):
        return FullEightScore(**row)
    return TrajectoryResult(**dict(value, settings=RecoverySettings(**value["settings"]), raw_score=score(value["raw_score"]),
        nodes=tuple(NodeScore(row["step"], score(row["score"]), row["checkpoint_id"]) for row in value["nodes"])))


class RestoredBudget(io.Budget):
    def __init__(self, path, authorization, value):
        self.path, self.authorization, self.value = Path(path), authorization, deepcopy(value)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        for task in self.value["tasks"].values():
            if task["status"] == "reserved":
                task.update(status="cancelled_before_execution", actual_seconds=0.,
                    reason="The failed owner closed this unexecuted reservation; an explicitly bounded remaining task replaces it.")
        self._save()


class RestoredQuota(ContinuationQuota):
    def __init__(self, previous, terminal_chooser):
        super().__init__(previous, terminal_chooser)
        self.resumed_registration = False

    def register_start(self, *, category_id, raw_id, settings, kind):
        arm = self.decisions["up_proj"]
        io.require(not self.resumed_registration and category_id == "up_proj" and kind == "third"
            and raw_id == arm.raw_id and settings == arm.settings
            and sum(row["category_id"] == category_id and row["raw_id"] == raw_id
                and row["settings"] == settings and row["kind"] == kind for row in self.started) == 1,
            "Resume exactly the existing third-arm slot once; never allocate another quota or candidate.")
        self.resumed_registration = True


class Processes(original.Processes):
    def __init__(self, root, budget, sources, snapshot_path, snapshot):
        super().__init__(root, budget, sources, (2, 3))
        self.snapshot_path, self.snapshot = snapshot_path, snapshot
        self.reused = set()

    def run(self, task_id, module, request, *, output_file):
        io.base.verify_sources(self.snapshot["repair_source_pins"])
        old = self.snapshot["budget_value"]["tasks"]
        prefix = self.snapshot["resume_prefix"]
        if task_id in (prefix + "/node_125/export", prefix + "/node_125/full8"):
            io.require(task_id not in self.reused, "Reuse each completed node125 readout exactly once.")
            self.reused.add(task_id)
            source = task_id.replace(prefix, "up_proj/third")
            io.require(old[source]["status"] == "completed" and old[source]["exit_code"] == 0,
                "Only an actually completed original node125 readout can be reused.")
            return self.snapshot["node125_export" if task_id.endswith("export") else "node125_score"]
        if task_id == prefix + "/node_250/export" and self.snapshot["recovered_node250_export"] is not None:
            io.require(task_id not in self.reused, "Adopt the recovered native node250 deployment once.")
            self.reused.add(task_id)
            return self.snapshot["recovered_node250_export"]
        if task_id == prefix + "/train":
            request = dict(request, resume_checkpoint=self.snapshot["checkpoint250"])
        command = lambda mod, path: wrapped_command(original.command(mod, path), self.snapshot_path)
        finite = types.FunctionType(original._finite_run.__code__,
            dict(original._finite_run.__globals__, command=command), "finished_finite_run",
            original._finite_run.__defaults__, original._finite_run.__closure__)
        return finite(self, task_id, module, request, output_file=output_file)


class Controller(original.Controller):
    def __init__(self, snapshot_path, validation_path):
        io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Keep the activated bitvae caller and CPU-only controller.")
        self.snapshot_path = Path(snapshot_path).resolve()
        self.snapshot = snapshot = frozen.install(self.snapshot_path)
        validation = io.read(validation_path)
        io.require(validation["status"] == "completed" and validation["exit_code"] == 0
            and all(validation["checks"].values()) and validation["snapshot"] == io.binding(snapshot_path)
            and snapshot["technical_proof"] is not None,
            "Bind the actual CPU closure and successful two-update native GPU resume proof.")
        technical = io.read(io.bound_path(snapshot["technical_proof"]))
        io.require(technical["status"] == "completed" and technical["exit_code"] == 0,
            "A failed native resume diagnostic cannot launch the remaining formal graph.")
        self.authorization_reference = snapshot["authorization"]
        self.authorization = io.read_authorization(self.authorization_reference)
        old_state = io.read(io.bound_path(snapshot["state"]))
        manifest = io.read(io.bound_path(self.authorization["source_manifest"]))
        selected = io.read(io.bound_path(self.authorization["selected_settings"]))
        self.prefix = io.load_prefix(self.authorization["prefix"])
        self.root = Path(snapshot["output_root"]).resolve() / "formal"
        io.require(not self.root.exists() and old_state["status"] == "failed" and old_state["current_category"] == "up_proj"
            and len(old_state["chain"]) == 6, "Finish this exact stopped six-category prefix in a new isolated output.")
        self.root.mkdir(parents=True)
        resources = io.base.check_resources((0, 1, 2, 3))
        inventory = io.read(io.bound_path(manifest["inventory"]))["inventory"]
        recipes = {}
        for row in selected["categories"]:
            originals = [io.read(io.bound_path(ref)) for ref in (row["anchor"], row["best_different"]["result"])]
            recipes[row["category_id"]] = [dict(candidate_id=value["candidate_id"], recipe=value["recipe"], is_anchor=index == 0)
                for index, value in enumerate(originals)]
        self.batch = dict(authorization=self.authorization, authorization_reference=self.authorization_reference,
            manifest=manifest, inventory=inventory, selected=selected, recipes=recipes)
        self.pins = snapshot["original_source_pins"]
        self.budget = RestoredBudget(self.root / "budget.json", self.authorization, snapshot["budget_value"])
        self.processes = Processes(self.root, self.budget, self.pins, self.snapshot_path, snapshot)
        self.quota = RestoredQuota(restore_quota(snapshot["quota"]), self._choose_terminal)
        self.chain = StrictMainChain(bf16_root_id=self.prefix["root"]["uuid"], categories=io.ORDER)
        for entry in old_state["chain"]:
            from experiments.sc_cat_live_cmcc_20261106.selection import SelectedCategory
            raw = entry["selected"]
            chosen = SelectedCategory(**dict(raw, settings=RecoverySettings(**raw["settings"])))
            certificate = io.read(io.bound_path(old_state["categories"][chosen.category_id]["selection_certificate"]))
            actual = self.chain.commit(chosen, live_parent_id=self.chain.live_parent_id,
                boundary_reference=entry["boundary_reference"], selection_certificate=certificate,
                actual_round_base_id=chosen.source_round_base_id, restored_node_id=chosen.source_node_id,
                runtime_identity=entry["runtime_identity"])
            io.require(actual == entry, "Never reselect or relabel a completed predecessor.")
        self.state_lock, self.deployments = threading.RLock(), {}
        self.initial_train_ids, self.reused_registration = set(), set()
        self.common = dict(authorization=self.authorization_reference, source_manifest=self.authorization["source_manifest"],
            inventory_receipt=manifest["inventory"], data_plan_path=manifest["data_plan"], phase="formal", stage="core",
            execution_policy=io.read(io.bound_path(snapshot["training_request"]))["execution_policy"])
        self.state = deepcopy(old_state)
        for key in ("ended_epoch", "exit_code", "error", "traceback"):
            self.state.pop(key, None)
        self.state.update(status="initialized", controller_pid=os.getpid(), started_epoch=time.time(),
            repaired_from=io.binding(self.snapshot_path), repair_source_pins=snapshot["repair_source_pins"],
            original_failed_state=snapshot["state"], original_failed_exit=snapshot["exit"], initial_resources=resources,
            readout_gpus=[2, 3], train_gpus=[0, 1])
        self._persist("six_categories_and_up_baselines_reused_no_new_search")

    def finish_up(self):
        category = "up_proj"
        baselines = list(self.quota.baselines[category])
        contexts = {}
        raw_category = self.state["categories"][category]
        for row in baselines:
            ref = raw_category["raws"][row.candidate_id]["live_raw"]
            raw = io.read(io.bound_path(ref))
            contexts[row.raw_id] = dict(category=category, parent_id=row.parent_id,
                entry=next(e for e in self.batch["recipes"][category] if e["candidate_id"] == row.candidate_id),
                raw=raw, raw_reference=ref, recipe_id=row.recipe_id, raw_score=row.raw_score)
            from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds
            contexts[row.raw_id]["seeds"] = recovery_seeds(self.batch["inventory"]["model_identity"], category,
                self.batch["manifest"]["run_identity"]["search_seed"])
            for source, score in [(row.raw_id, row.raw_score), *((node.checkpoint_id, node.score) for node in row.nodes)]:
                score_file = Path(score.evaluation_id).parent / "score.json"
                report = raw_category["raws"][row.candidate_id]["raw_deployment"] if source == row.raw_id else io.binding(Path(score.evaluation_id).parent.parent / "export/deployment_report.json")
                self.deployments[source] = dict(deployment_report=report,
                    score=io.binding(score_file))
        arm = self.quota.decisions[category]
        io.require(arm is not None and not arm.interaction_eligible and arm.settings.schedule_max == 500,
            "Resume only the already declared up500 third arm; do not allocate a new choice.")
        prefix = self.snapshot["resume_prefix"]
        tasks = io.trajectory_tasks(prefix, self.authorization, category, arm.settings)
        old = self.snapshot["budget_value"]["tasks"]
        for key in list(tasks):
            source = key.replace(prefix, "up_proj/third")
            if old[source]["status"] == "completed":
                del tasks[key]
            elif key.endswith("/node_250/export") and self.snapshot["recovered_node250_export"] is not None:
                del tasks[key]
            else:
                suffix = source.split("/", 2)[2]
                tasks[key]["seconds"] -= sum(value.get("actual_seconds", 0.) for name, value in old.items()
                    if name.startswith("up_proj/") and name.split("/", 2)[2] == suffix)
        self.budget.reserve(tasks, counts={})
        context, additional = self._trajectory(contexts[arm.raw_id], arm.settings, "third", prefix)
        results = [*baselines, additional]
        winner, node, certificate = self.quota.select_completed_category(results)
        chosen = selected_category(winner, node, seed_identity=contexts[winner.raw_id]["seeds"])
        deployment = self.deployments[node.checkpoint_id]
        report = io.read(io.bound_path(deployment["deployment_report"]))
        score = read_score(deployment["score"])
        boundary = complete_boundary_reference(export_report=report, inventory=self.batch["inventory"], full8_score=score)
        committed = self.chain.commit(chosen, live_parent_id=self.chain.live_parent_id, boundary_reference=boundary,
            selection_certificate=certificate, actual_round_base_id=winner.raw_id, restored_node_id=node.checkpoint_id,
            runtime_identity=boundary["runtime_identity"])
        output = self.root / "categories" / category
        io.write(output / "selection_certificate.json", certificate)
        io.write(output / "selected_recipe.json", dict(selected=asdict(chosen), replay=chosen.training_plan(),
            actual_schedule_max=winner.settings.schedule_max, selected_step=node.step,
            actual_completed_trajectories=[row.trajectory_id for row in results], evaluated_artifact=deployment))
        io.write(output / "boundary.json", boundary)
        self.state["categories"][category].update(status="completed", selected=asdict(chosen),
            selection_certificate=io.binding(output / "selection_certificate.json"), boundary=io.binding(output / "boundary.json"),
            source_deployment=deployment, chain_commit=committed)
        self._persist("all_original_up_comparisons_and_two_same_artifact_S9_choices_completed")
        return boundary

    def run(self):
        self.state["status"] = "running"
        parent = self.finish_up()
        io.write(self.root / "after_up.json", parent)
        frozen_recipe = dict(schema="sc_cat_complete_frozen_recipe_v1", phase="formal", frozen_order=list(io.ORDER),
            completed_categories=list(io.ORDER), shared_BF16_root=self.prefix["frozen"]["shared_BF16_root"],
            selected_settings=self.authorization["selected_settings"], categories=[self.state["categories"][c]["selected"] for c in io.ORDER],
            final_boundary=io.binding(self.root / "after_up.json"), quotas=io.binding(self.root / "quotas.json"),
            core_nine_task=self.state["core_nine_task"], complete_seven_category_chain=True, BF16_recipe_replay_completed=False)
        io.write(self.root / "frozen_recipe.json", frozen_recipe)
        self._replay(io.binding(self.root / "frozen_recipe.json"), io.binding(self.root / "after_up.json"))
        for reference in self.authorization["reference_evidence"]:
            io.bound_path(reference)
        self.state.update(status="completed", exit_code=0, ended_epoch=time.time(), complete_seven_category_chain=True)
        self._persist("original_finite_batch_completed_after_bound_document_relocation_and_native_resume")
        io.write(self.root / "controller.exit.json", dict(status="completed", exit_code=0, ended_epoch=self.state["ended_epoch"],
            state=io.binding(self.root / "controller_state.json"), actual_GPU_hours=self.budget.value["total_actual_GPU_hours"],
            core_nine_task=self.state["core_nine_task"], replay_nine_task=self.state["replay_nine_task"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    controller = None
    try:
        controller = Controller(args.snapshot, args.validation)
        def interrupted(signum, frame):
            controller.processes.cancel()
            raise InterruptedError("Finish controller received signal " + str(signum))
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
