"""Finish the fixed five-category main chain, real endpoint S9 choice and BF16 replay."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import types

from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
from experiments.longbench_v1.protocol import DATASETS
from experiments.sc_cat_live_cmcc_20261106 import bindings as old_io
from experiments.sc_cat_live_cmcc_20261106 import live_io
from experiments.sc_cat_live_cmcc_20261106.boundary import complete_boundary_reference
from experiments.sc_cat_live_cmcc_20261106.evaluate import read_score
from experiments.sc_cat_live_cmcc_20261106.run import Controller as OriginalController
from experiments.sc_cat_pipeline_cmcc_20261106.run import PipelineController
from . import bindings as io
from .selection import ContinuationQuota


def command(module, request_path):
    if module in ("fit_worker", "recovery_worker", "node_export", "evaluate"):
        return [sys.executable, "-B", "-u", "-m", "experiments.sc_cat_continue_cmcc_20261107.workers",
            "--kind", module, "--request", str(Path(request_path).resolve())]
    if module in ("longbench", "replay"):
        return [sys.executable, "-B", "-u", "-m", "experiments.sc_cat_continue_cmcc_20261107." + module,
            "--request", str(Path(request_path).resolve())]
    raise ValueError("Unknown concrete continuation worker: " + module)


_finite_run = types.FunctionType(old_io.Processes.run.__code__, dict(old_io.Processes.run.__globals__, command=command),
    "continuation_finite_run", old_io.Processes.run.__defaults__, old_io.Processes.run.__closure__)


class Processes(old_io.Processes):
    def run(self, *args, **kwargs):
        return _finite_run(self, *args, **kwargs)


def _category_method(authorization):
    proxy = types.SimpleNamespace(**dict(vars(old_io),
        raw_tasks=lambda prefix: io.raw_tasks(prefix, authorization, prefix.split("/")[0]),
        trajectory_tasks=lambda prefix, settings: io.trajectory_tasks(prefix, authorization, prefix.split("/")[0], settings)))
    function = OriginalController._category
    return types.FunctionType(function.__code__, dict(function.__globals__, io=proxy),
        "continuation_category", function.__defaults__, function.__closure__)


def evaluation_request(common, baseline, deployment_report, output, stage):
    return dict(common, deployment_report=deployment_report, baseline=baseline, output=str(output), phase="formal", stage=stage)


def longbench_tasks(authorization, prefix, stage):
    return {prefix + "/part_" + str(index): dict(component=stage, kind="longbench_shard",
        seconds=io.seconds(authorization, "longbench_shard")) for index in range(2)}


def replay_task_graph(authorization, plans):
    tasks = {}
    for plan in plans:
        category, step = plan["category_id"], plan["selected_step"]
        prefix = "replay/" + category
        tasks[prefix + "/fit"] = dict(component="replay", kind="fit", seconds=io.seconds(authorization, "fit", category=category))
        if step:
            tasks[prefix + "/train"] = dict(component="replay", kind="replay_train", seconds=io.seconds(authorization,
                "replay_train", category=category, settings=RecoverySettings(**plan["settings"]), selected_step=step))
        tasks[prefix + "/export"] = dict(component="replay", kind="replay_export", seconds=io.seconds(authorization, "replay_export"))
        tasks[prefix + "/full8"] = dict(component="replay", kind="full8", seconds=io.seconds(authorization, "full8"))
    tasks.update(longbench_tasks(authorization, "replay/final_longbench", "replay"))
    return tasks


class Controller(OriginalController):
    _trajectory = PipelineController._trajectory

    def __init__(self, authorization_path, validation_path, readout_gpus=None):
        require = io.require
        require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use the confirmed caller environment with controller CUDA hidden.")
        self.authorization_reference = io.binding(authorization_path)
        self.authorization = io.read_authorization(self.authorization_reference)
        self.prefix = io.load_prefix(self.authorization["prefix"])
        manifest = io.read(io.bound_path(self.authorization["source_manifest"]))
        selected = io.read(io.bound_path(self.authorization["selected_settings"]))
        require(selected["status"] == "completed" and selected["run_identity"] == manifest["run_identity"]
            and selected["protocol_identity"] == manifest["protocol_identity"]
            and self.prefix["frozen"]["selected_settings"] == self.authorization["selected_settings"], "Preserve the exact frozen offline recipe/model/data/consumer identities.")
        validation = io.read(validation_path)
        self.pins = io.runtime_pins(manifest)
        require(validation["status"] == "completed" and validation["exit_code"] == 0
            and validation["phase"] == "cpu_fixture" and all(validation["checks"].values())
            and validation["source_pins"] == self.pins and validation["prefix"] == self.authorization["prefix"],
            "Bind completed CPU verification and every actual unchanged consumer to this exact prior prefix.")
        old_io.verify_sources(self.pins)
        readout_gpus = old_io.readout_devices(self.authorization["readout_gpus"] if readout_gpus is None else readout_gpus)
        require(set(readout_gpus) <= set(self.authorization["readout_gpus"]), "Use only an actual selected subset of the approved readout GPUs.")
        resources = old_io.check_resources((0, 1, *readout_gpus))
        self.root = Path(self.authorization["output_root"]).resolve() / "formal"
        require(not self.root.exists(), "Use a new isolated continuation output; no overwrite or automatic retry.")
        self.root.mkdir(parents=True)
        inventory = io.read(io.bound_path(manifest["inventory"]))["inventory"]
        recipes = {}
        for row in selected["categories"]:
            originals = [io.read(io.bound_path(reference)) for reference in (row["anchor"], row["best_different"]["result"])]
            require(originals[0]["recipe"] != originals[1]["recipe"], "Use the actual two different frozen live recipes.")
            recipes[row["category_id"]] = [dict(candidate_id=value["candidate_id"], recipe=value["recipe"], is_anchor=index == 0,
                source=(row["anchor"], row["best_different"]["result"])[index]) for index, value in enumerate(originals)]
        self.batch = dict(authorization=self.authorization, authorization_reference=self.authorization_reference,
            manifest=manifest, inventory=inventory, selected=selected, recipes=recipes)
        self.budget = io.Budget(self.root / "budget.json", self.authorization, prior_GPU_hours=self.prefix["prior_actual_GPU_hours"])
        self.processes = Processes(self.root, self.budget, self.pins, readout_gpus)
        self.quota = ContinuationQuota(self.prefix["quota"], self._choose_terminal)
        self.chain = self.prefix["chain"]
        self.state_lock, self.deployments = threading.RLock(), {}
        self.initial_train_ids, self.reused_registration = set(), set()
        self.common = dict(authorization=self.authorization_reference, source_manifest=self.authorization["source_manifest"],
            inventory_receipt=manifest["inventory"], data_plan_path=manifest["data_plan"], phase="formal", stage="core",
            execution_policy=dict(train_device="cuda:0", attention_implementation="flash_attention_2", teacher_model_offload="none",
                teacher_output_offload="cpu", teacher_output_pin_memory=True, teacher_output_chunk_tokens=8, deterministic=False))
        self.state = dict(schema="sc_cat_five_category_continuation_v1", status="initialized", controller_pid=os.getpid(),
            started_epoch=time.time(), authorization=self.authorization_reference, validation=io.binding(validation_path),
            prefix=self.authorization["prefix"], prior_actual_GPU_hours=self.prefix["prior_actual_GPU_hours"],
            source_manifest=self.authorization["source_manifest"], selected_settings=self.authorization["selected_settings"],
            source_pins=self.pins, train_gpus=self.authorization["train_gpus"], readout_gpus=list(readout_gpus),
            authorized_readout_gpus=self.authorization["readout_gpus"],
            initial_resources=resources, frozen_order=list(io.ORDER), approved_categories=list(io.CATEGORIES),
            current_category=None, categories=deepcopy(self.prefix["state"]["categories"]), quota=None,
            chain=deepcopy(self.chain.boundaries), reference_mode="reuse_only", reference_training_started=False)
        io.write(self.root / "source_pins.json", self.pins)
        self._category = types.MethodType(_category_method(self.authorization), self)
        self._persist("new_authorized_five_category_batch_preserve_after_o_and_global_quotas")

    def _score(self, prefix, report_reference, output):
        stage = "replay" if prefix.startswith("replay/") else "core"
        reference = self.processes.run(prefix, "evaluate", evaluation_request(self.common,
            self.batch["manifest"]["baseline"], report_reference, output, stage), output_file=output / "score.json")
        score = read_score(reference)
        report = io.read(io.bound_path(report_reference))
        io.require(score.evidence["checkpoint_id"] == report["uuid"]
            and score.evidence["source_checkpoint_id"] == report["export_evidence"]["source_node_id"], "Bind the actual deployed UUID and its exact own source node/raw.")
        with self.state_lock:
            source = score.evidence["source_checkpoint_id"]
            io.require(source not in self.deployments, "Never duplicate a source node's completed full-eight evaluation.")
            self.deployments[source] = dict(deployment_report=report_reference, score=reference)
        return score

    def _longbench_tasks(self, prefix, stage):
        return longbench_tasks(self.authorization, prefix, stage)

    def _longbench(self, deployment, prefix, stage):
        from . import longbench
        from .final_score import summarize
        output = self.root / prefix
        shared = dict(deployment_report=deployment["deployment_report"], inventory_receipt=self.batch["manifest"]["inventory"],
            archive=self.authorization["longbench_archive"])
        with self.budget.cpu_window(prefix + "/prepare_same_artifact"):
            longbench.prepare(dict(shared, output=str(output / "prepared")))
        config = io.binding(output / "prepared" / "config.json")
        first = list(io.LONG_PARTITION_ONE)
        second = [name for name in DATASETS if name not in first]
        def partition(index, tasks):
            destination = output / ("part_" + str(index))
            return self.processes.run(prefix + "/part_" + str(index), "longbench",
                dict(operation="predict", phase="formal", stage=stage, config=config, datasets=tasks,
                    authorization=self.authorization_reference, output=str(destination)),
                output_file=destination / "partition_result.json")
        parts = self._parallel([lambda index=index, tasks=tasks: partition(index, tasks)
            for index, tasks in enumerate((first, second))], workers=2)
        with self.budget.cpu_window(prefix + "/merge_and_score_official21"):
            merged = longbench.merge(dict(operation="merge", config=config, partitions=parts, output=str(output / "complete")))
            request = dict(shared, full8_score=deployment["score"], baseline=self.batch["manifest"]["baseline"],
                baseline_longbench_scores=self.authorization["baseline_longbench_scores"],
                compressed_longbench_scores=merged["scores"], output=str(output / "nine"))
            nine = summarize(request)
            io.write(output / "nine" / "nine_task.json", nine)
        return io.binding(output / "nine" / "nine_task.json")

    def _choose_terminal(self, endpoints):
        from .final_score import select_reported_endpoint
        tasks = {}
        for _winner, node in endpoints:
            tasks.update(self._longbench_tasks("terminal/" + node.checkpoint_id, "terminal_longbench"))
        self.budget.reserve(tasks, counts=dict(terminal_longbench=len(endpoints)))
        def score_endpoint(pair):
            winner, node = pair
            nine = self._longbench(self.deployments[node.checkpoint_id], "terminal/" + node.checkpoint_id, "terminal_longbench")
            return dict(nine_task=nine, checkpoint_id=node.score.evidence["checkpoint_id"], source_node_id=node.checkpoint_id,
                is_anchor=winner.is_anchor, logical_bits=winner.logical_bits, packed_bits=winner.packed_bits,
                candidate_id=winner.candidate_id, trajectory_id=winner.trajectory_id, raw_id=winner.raw_id,
                recipe_id=winner.recipe_id, selected_step=node.step, deployment=self.deployments[node.checkpoint_id])
        reported = self._parallel([lambda pair=pair: score_endpoint(pair) for pair in endpoints], workers=2)
        decision = select_reported_endpoint(reported)
        selected = decision["selected"]
        winner, node = next(pair for pair in endpoints if pair[1].checkpoint_id == selected["source_node_id"])
        io.write(self.root / "terminal" / "selection.json", decision)
        self.state["terminal_selection"] = io.binding(self.root / "terminal" / "selection.json")
        self.state["core_nine_task"] = selected["nine_task"]
        self._persist("terminal_two_same_artifact_full_nine_complete_choose_existing_winner")
        return winner, node, decision

    def _replay(self, frozen_reference, final_boundary):
        from . import replay
        manifest = replay.freeze_manifest([self.prefix["references"]["frozen_recipe"], frozen_reference],
            self.authorization["selected_settings"], self.batch["manifest"]["inventory"], self.batch["manifest"]["data_plan"], final_boundary)
        io.require(manifest["replay_KD_updates"] <= self.authorization["maximum_replay_KD_updates"], "The actually chosen recipe exceeds the approved exact replay-update maximum.")
        io.write(self.root / "replay" / "manifest.json", manifest)
        manifest_reference = io.binding(self.root / "replay" / "manifest.json")
        tasks = replay_task_graph(self.authorization, [replay.stage_plan(manifest, row["category_id"]) for row in manifest["categories"]])
        self.budget.reserve(tasks, counts=dict(replay_fits=7, replay_exports=7, replay_full8=7,
            replay_longbench=1, replay_KD_updates=manifest["replay_KD_updates"]))
        self.state["replay_manifest"] = manifest_reference
        self.state["replay_stages"] = []
        self._persist("reserve_actual_chosen_BF16_replay_without_search_or_node_reselection")
        parent = self.prefix["root"]
        parent_reference = self.prefix["frozen"]["shared_BF16_root"]
        common = dict(self.common, stage="replay")
        last_deployment = None
        for row in manifest["categories"]:
            category = row["category_id"]
            plan = replay.stage_plan(manifest, category)
            prefix, output = "replay/" + category, self.root / "replay" / category
            raw_reference = self.processes.run(prefix + "/fit", "fit_worker",
                dict(common, category_id=category, candidate_id=plan["source_candidate_id"], recipe=plan["resolved_recipe"],
                    parent=parent_reference, replay_manifest=manifest_reference,
                    output=str(output / "fit"), execution_policy=dict(attention_implementation="sdpa")),
                output_file=output / "fit" / "live_raw.json")
            request = dict(common, replay_manifest=manifest_reference, category_id=category, live_raw=raw_reference,
                parent=parent_reference, output=str(output / "train"), operation="train_replay")
            result_reference = None
            if plan["selected_step"]:
                result_reference = self.processes.run(prefix + "/train", "replay", request,
                    output_file=output / "train" / "replay_result.json")
            export_request = dict(request, operation="export_replay", output=str(output / "export"),
                destination=str(output / "deployment"))
            if result_reference is not None:
                export_request["replay_result"] = result_reference
            report_reference = self.processes.run(prefix + "/export", "replay", export_request,
                output_file=output / "export" / "deployment_report.json")
            score = self._score(prefix + "/full8", report_reference, output / "full8")
            report = io.read(io.bound_path(report_reference))
            with self.budget.cpu_window(prefix + "/accept_actual_replayed_boundary"):
                boundary = complete_boundary_reference(export_report=report, inventory=self.batch["inventory"], full8_score=score)
                io.require(boundary["completed_prefix"] == list(io.ORDER[:io.ORDER.index(category) + 1])
                    and report["export_evidence"]["actual_updates"] == plan["selected_step"]
                    and report["export_evidence"]["schedule_max"] == plan["schedule_max"], "Replay keeps the original schedule and stops at the actually frozen selected step.")
            io.write(output / "boundary.json", boundary)
            parent = boundary
            parent_reference = io.binding(output / "boundary.json")
            last_deployment = self.deployments[score.evidence["source_checkpoint_id"]]
            self.state["replay_stages"].append(dict(category_id=category, selected_step=plan["selected_step"],
                schedule_max=plan["schedule_max"], raw=raw_reference, actual_training=result_reference,
                deployment=report_reference, boundary=io.binding(output / "boundary.json")))
            self._persist("fixed_recipe_replay_stage_actual_prefix_completed")
        nine = self._longbench(last_deployment, "replay/final_longbench", "replay")
        io.write(self.root / "replay" / "final_boundary.json", parent)
        self.state.update(replay_final_boundary=io.binding(self.root / "replay" / "final_boundary.json"), replay_nine_task=nine,
            BF16_recipe_replay_completed=True)
        self._persist("original_BF16_fixed_recipe_replay_and_its_same_artifact_full_nine_completed")

    def run(self):
        self.state["status"] = "running"
        self._persist("start_new_authorized_continuation_no_old_budget_transfer")
        parent = self.prefix["parent"]
        for category in io.CATEGORIES:
            parent = self._category(category, parent)
        io.require(self.quota.selected_categories == list(io.ORDER) and len(self.chain.boundaries) == 7,
            "Complete exactly one seven-category chain with inherited global quotas.")
        io.write(self.root / "after_up.json", parent)
        frozen = dict(schema="sc_cat_complete_frozen_recipe_v1", phase="formal", frozen_order=list(io.ORDER),
            completed_categories=list(io.ORDER), shared_BF16_root=self.prefix["frozen"]["shared_BF16_root"],
            selected_settings=self.authorization["selected_settings"], categories=[self.state["categories"][category]["selected"] for category in io.ORDER],
            final_boundary=io.binding(self.root / "after_up.json"), quotas=io.binding(self.root / "quotas.json"),
            core_nine_task=self.state["core_nine_task"], complete_seven_category_chain=True, BF16_recipe_replay_completed=False)
        io.write(self.root / "frozen_recipe.json", frozen)
        self._replay(io.binding(self.root / "frozen_recipe.json"), io.binding(self.root / "after_up.json"))
        if self.authorization.get("reference_evidence"):
            for reference in self.authorization["reference_evidence"]:
                io.bound_path(reference)
            self.state["fixed_reference"] = dict(mode="verified_existing_evidence_only", evidence=self.authorization["reference_evidence"], new_GPU_tasks=0)
        self.state.update(status="completed", exit_code=0, ended_epoch=time.time(), complete_seven_category_chain=True)
        self._persist("finite_continuation_terminal_selection_and_BF16_replay_completed")
        io.write(self.root / "controller.exit.json", dict(status="completed", exit_code=0, ended_epoch=self.state["ended_epoch"],
            state=io.binding(self.root / "controller_state.json"), actual_GPU_hours=self.budget.value["total_actual_GPU_hours"],
            prior_batch_GPU_hours=self.prefix["prior_actual_GPU_hours"], core_nine_task=self.state["core_nine_task"], replay_nine_task=self.state["replay_nine_task"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--readout-gpus", nargs="+", type=int)
    args = parser.parse_args()
    controller = None
    try:
        controller = Controller(args.authorization, args.validation, args.readout_gpus)
        def interrupted(signum, _frame):
            controller.processes.cancel()
            raise InterruptedError("Continuation controller received signal " + str(signum))
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
