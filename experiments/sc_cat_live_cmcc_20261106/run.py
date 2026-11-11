"""Execute only the approved finite actual k→o main chain, then exit."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import os
from pathlib import Path
import signal
import shutil
import threading
import time
import traceback

from . import bindings as io


def score_record(score):
    return dict(task_scores=dict(score.task_scores), evaluation_id=score.evaluation_id, evidence=dict(score.evidence))


def trajectory_record(row):
    return dict(category_id=row.category_id, parent_id=row.parent_id, raw_id=row.raw_id,
        recipe_id=row.recipe_id, candidate_id=row.candidate_id, trajectory_id=row.trajectory_id,
        is_anchor=row.is_anchor, logical_bits=row.logical_bits, packed_bits=row.packed_bits,
        settings=asdict(row.settings), raw_score=score_record(row.raw_score),
        nodes=[dict(step=node.step, checkpoint_id=node.checkpoint_id, score=score_record(node.score)) for node in row.nodes])


def quota_record(quota):
    """The complete seven-category quota state is a later batch's immutable input."""
    return dict(schema="sc_cat_global_recovery_quota_v1", categories=list(quota.categories),
        decisions={key: None if value is None else asdict(value) for key, value in quota.decisions.items()},
        long_categories=list(quota.long_categories), interaction_categories=list(quota.interaction_categories),
        started=[dict(row, settings=asdict(row["settings"])) for row in quota.started],
        selected_categories=list(quota.selected_categories),
        baselines={key: [trajectory_record(row) for row in values] for key, values in quota.baselines.items()}, counts=quota.counts())


def restore_quota(reference):
    """Rehydrate a bound earlier batch's full-chain quotas without reallocating past slots."""
    from experiments.cat_auto_protocol_cmcc_20261104.nodes import FullEightScore, NodeScore, RecoverySettings, ThirdArm, TrajectoryResult
    from .selection import RecoveryQuota

    record = io.read(io.bound_path(reference))
    io.require(record["schema"] == "sc_cat_global_recovery_quota_v1" and record["categories"] == list(io.ORDER), "Restore the same frozen global category order.")
    quota = RecoveryQuota(record["categories"])
    quota.selected_categories = record["selected_categories"]
    io.require(quota.selected_categories == list(io.ORDER[:len(quota.selected_categories)]), "Persist only one selected category prefix.")
    quota.long_categories, quota.interaction_categories = record["long_categories"], record["interaction_categories"]
    io.require(len(set(quota.long_categories)) == len(quota.long_categories) <= 2
        and len(set(quota.interaction_categories)) == len(quota.interaction_categories) <= 2,
        "Do not reset or enlarge the full-model long/interaction quotas.")
    quota.decisions = {key: None if value is None else ThirdArm(**dict(value, settings=RecoverySettings(**value["settings"])))
        for key, value in record["decisions"].items()}
    quota.started = [dict(row, settings=RecoverySettings(**row["settings"])) for row in record["started"]]
    def score(value):
        return FullEightScore(value["task_scores"], value["evaluation_id"], value["evidence"])
    for category, values in record["baselines"].items():
        quota.baselines[category] = tuple(TrajectoryResult(**dict(row, settings=RecoverySettings(**row["settings"]),
            raw_score=score(row["raw_score"]), nodes=tuple(NodeScore(node["step"], score(node["score"]), node["checkpoint_id"])
                for node in row["nodes"]))) for row in values)
    io.require(quota.counts() == record["counts"], "The persisted global quota counts changed during reconstruction.")
    return quota


def checked_fit_receipts(fit_paths, *, require_both=True):
    """Read actual fit outputs; a one-category check is useful while validating the controller on CPU."""
    paths = [Path(path).resolve() for path in fit_paths]
    seen = set()
    for path in paths:
        row = io.read(path)
        io.require(row["status"] != "running" and row.get("ended_epoch", row.get("finished_epoch")),
            "A required technical fit is still active.")
        request = io.read(io.bound_path(row["request"]))
        io.require(request["phase"] == "technical_validation" and request["category_id"] in io.ORDER[:2],
            "Bind the actual k/o technical fit request.")
        if row["status"] == "completed" and row["exit_code"] == 0:
            output = Path(request["output"])
            completed = io.read(output / "completed.json")
            raw = io.read(io.bound_path(completed["live_raw"]))
            io.require(completed["exit_code"] == 0 and completed["phase"] == raw["phase"] == "technical_validation"
                and raw["current_category"] == request["category_id"] and not raw["formal_eligible"]
                and raw["fit_report"]["technical_steps"] == 2
                and len(raw["fit_report"]["stages"]) == 2
                and all(stage["actual_steps"] == 2 for stage in raw["fit_report"]["stages"])
                and raw["fresh_evidence"]["gpu_validation"], "The actual technical fit/raw/fresh evidence is incomplete.")
            seen.add(request["category_id"])
    io.require(not require_both or seen == set(io.ORDER[:2]), "Both actual k and o technical fits must finish successfully before formal launch.")
    return paths, seen


def checked_technical(fit_paths, recovery_path):
    """Bind explicit successful k/o fits and the whole recovery validation owner window."""
    paths, _seen = checked_fit_receipts(fit_paths)
    recovery_path = Path(recovery_path).resolve()
    recovery = io.read(recovery_path)
    io.require(recovery["status"] == "completed" and recovery["exit_code"] == 0
        and recovery.get("ended_epoch", recovery.get("finished_epoch")),
        "The whole k/o technical recovery sequence has not completed successfully.")
    checks = ("k_native_KD_save_return", "k_formal_shape_1024_backward", "k_step1_deploy_fresh_CUDA",
        "k_raw0_deploy_fresh_CUDA", "o_prefix_native_KD_save_return", "o_prefix_node_deploy_fresh_CUDA",
        "o_current_native_KD_save_return", "o_current_node_deploy_fresh_CUDA",
        "o_prefix_nonzero_mutable_and_merge", "o_current_nonzero_mutable_and_merge")
    io.require(set(recovery["checks"]) == set(checks) and all(recovery["checks"][key] is True for key in checks)
        and recovery["stages"] and all(stage["exit_code"] == 0 and stage["status"] == "completed" for stage in recovery["stages"]),
        "Require every actual k/o save-return, full-shape backward and independent deployment/fresh check.")
    results = recovery["results"]
    for name in ("k_recovery", "o_prefix_recovery", "o_current_recovery"):
        result = io.read(io.bound_path(results[name]))
        io.require(result["status"] == "completed_native_recovery" and result["exit_code"] == 0
            and result["phase"] == "technical_validation" and result["actual_run"]["actual_steps"] == 2
            and tuple(node["step"] for node in result["nodes"]) == (1, 2),
            "The technical recovery output must contain both actual native saved updates.")
        expected_category = "k_proj" if name == "k_recovery" else "o_proj"
        expected_scope = "remaining_lora_current_decoder" if name == "o_current_recovery" else "remaining_lora_prefix_decoder"
        io.require(result["live_raw"]["current_category"] == expected_category
            and result["settings"]["scope"] == expected_scope
            and result["settings"]["schedule_max"] == 500,
            "The actual technical result must validate its declared category and decoder scope.")
    for name in ("k_step1_deployment", "k_raw0_deployment", "o_prefix_node_deployment", "o_current_node_deployment"):
        report = io.read(io.bound_path(results[name]))
        proof = report["fresh_forward_evidence"]
        io.require(report["phase"] == "technical_validation" and proof["checkpoint_id"] == report["uuid"]
            and proof["weight_digest"] == report["digest"] and proof["exit_code"] == 0 and proof["state_exact"]
            and proof["forward_executed"] and proof["gpu_validation"], "A required actual technical deployment/fresh proof is incomplete.")
        expected = 0 if name == "k_raw0_deployment" else 1 if name == "k_step1_deployment" else 2
        io.require(report["export_evidence"]["actual_updates"] == expected,
            "Validate the actual saved source update, including both nonzero o step2 exports.")
    for scope in ("prefix", "current"):
        proof = io.read(io.bound_path(results[f"o_{scope}_nonzero_mutable_and_merge"]))
        deployment = io.read(io.bound_path(results[f"o_{scope}_node_deployment"]))
        result_reference = results[f"o_{scope}_recovery"]
        io.require(proof["status"] == "actual_nonzero_mutable_and_BF16_merge_verified"
            and proof["exit_code"] == 0 and proof["phase"] == "technical_validation" and not proof["formal_eligible"]
            and proof["selected_step"] == 2
            and proof["source_node_id"] == deployment["export_evidence"]["source_node_id"]
            and proof["raw_uuid"] == deployment["export_evidence"]["round_base_id"]
            and proof["deployment_uuid"] == deployment["uuid"]
            and proof["step1_to2_changed_mutable_components"]["lora"] > 0
            and proof["step1_to2_changed_mutable_components"]["decoder"] > 0
            and proof["step2_nonzero_LoRA_B_leaves"] == proof["LoRA_B_leaves"] > 0
            and proof["actual_changed_future_dense_owners"] > 0
            and proof["actual_changed_future_dense_weight_elements"] > 0
            and proof["max_abs_BF16_dense_delta"] > 0 and proof["actual_changed_persistent_decoder_fields"] > 0
            and io.bound_path(proof["result"]) == io.bound_path(result_reference)
            and io.bound_path(proof["exported"]) == io.bound_path(results[f"o_{scope}_node_deployment"]),
            "The actual saved nonzero decoder/LoRA updates and BF16 dense merge must belong to this exact o deployment.")
    io.bound_path(results["k_successor_boundary"])
    paths.append(recovery_path)
    return io.technical_windows(paths)


class Controller:
    def __init__(self, *, authorization, root_boundary, technical_fit, technical_recovery, readout_gpus=(2, 3, 4, 5)):
        from experiments.cat_auto_protocol_cmcc_20261104.contracts import validate_boundary
        from .boundary import StrictMainChain
        from .selection import RecoveryQuota

        io.require(os.environ.get("CONDA_DEFAULT_ENV") == "bitvae", "Activate bitvae in the caller shell.")
        readout_gpus = io.readout_devices(readout_gpus)
        self.batch = io.load_batch(authorization, root_boundary)
        validate_boundary(self.batch["root_boundary"])
        self.root = self.batch["root"] / "formal"
        io.require(not self.root.exists(), "Do not replace or retry an existing formal batch.")
        windows, receipts = checked_technical(technical_fit, technical_recovery)
        sources = io.frozen_sources(self.batch["manifest"])
        resources = io.check_resources((0, 1, *readout_gpus))
        self.root.mkdir()
        self.budget = io.Budget(self.root / "budget.json", self.batch["authorization"], windows, receipts)
        io.write(self.root / "source_pins.json", sources)
        self.processes = io.Processes(self.root, self.budget, sources, readout_gpus)
        self.quota = RecoveryQuota(io.ORDER)
        self.chain = StrictMainChain(bf16_root_id=self.batch["root_boundary"]["uuid"], categories=io.ORDER)
        self.state_lock = threading.RLock()
        self.state = dict(schema="sc_cat_live_k_o_controller_v1", status="initialized", controller_pid=os.getpid(),
            started_epoch=time.time(), authorization=self.batch["authorization_reference"],
            shared_BF16_root=self.batch["root_boundary_reference"], source_manifest=self.batch["authorization"]["source_manifest"],
            selected_settings=self.batch["authorization"]["selected_settings"], technical_receipts=receipts,
            source_pins=io.binding(self.root / "source_pins.json"), initial_resources=resources,
            train_gpus=[0, 1], readout_gpus=list(readout_gpus),
            frozen_order=list(io.ORDER), approved_categories=list(io.ORDER[:2]), current_category=None,
            categories={}, quota=None, chain=[], last_event=None)
        root_meta = io.read(Path(self.batch["root_boundary"]["path"]) / "checkpoint_meta.json")
        dense_bytes = (Path(self.batch["root_boundary"]["path"]) / root_meta["state_dict_file"]).stat().st_size
        self.state["storage_projection"] = dict(shared_BF16_payload_bytes=dense_bytes,
            maximum_current_category_deployments=18, retained_raws_over_batch=4, retained_previous_winner=1,
            conservative_dense_payload_bytes=23 * dense_bytes, available_bytes_at_start=shutil.disk_usage(self.root).free,
            method="Per-category pruning: at most18 current deployments, four preserved raw sources and one preceding selected deployment; each upper-estimated by the dense BF16 payload. Native mutable/optimizer/RNG and logs are additional.",
            estimated_dense_bytes_are_not_measured_peak=True)
        self.deployments = {}
        self.common = dict(authorization=self.batch["authorization_reference"], source_manifest=self.batch["authorization"]["source_manifest"],
            inventory_receipt=self.batch["manifest"]["inventory"], data_plan_path=self.batch["manifest"]["data_plan"], phase="formal",
            execution_policy=dict(train_device="cuda:0", attention_implementation="flash_attention_2",
                teacher_model_offload="none", teacher_output_offload="cpu", teacher_output_pin_memory=True,
                teacher_output_chunk_tokens=8, deterministic=False))
        self._persist("initialized")

    def _persist(self, event):
        with self.state_lock:
            self.state.update(last_event=event, updated_epoch=time.time(), quota=quota_record(self.quota), chain=list(self.chain.boundaries))
            io.write(self.root / "controller_state.json", self.state, replace=(self.root / "controller_state.json").exists())
            io.write(self.root / "quotas.json", self.state["quota"], replace=(self.root / "quotas.json").exists())
        print(dict(event=event, category=self.state["current_category"], status=self.state["status"]), flush=True)

    def _parallel(self, functions, *, workers):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = [executor.submit(function) for function in functions]
            try:
                return [future.result() for future in as_completed(pending)]
            except BaseException:
                self.processes.cancel()
                for future in pending:
                    future.cancel()
                raise

    def _score(self, prefix, report_reference, output):
        from .evaluate import read_score

        reference = self.processes.run(prefix, "evaluate", dict(deployment_report=report_reference,
            baseline=self.batch["manifest"]["baseline"], output=str(output), authorization=self.common["authorization"], phase="formal"),
            output_file=output / "score.json")
        score = read_score(reference)
        report = io.read(io.bound_path(report_reference))
        io.require(score.evidence["checkpoint_id"] == report["uuid"]
            and score.evidence["source_checkpoint_id"] == report["export_evidence"]["source_node_id"],
            "Every full8 must bind both the actual deployment and its own raw/node source.")
        with self.state_lock:
            source = score.evidence["source_checkpoint_id"]
            io.require(source not in self.deployments, "Do not evaluate or substitute a second deployment for a selected source.")
            self.deployments[source] = dict(deployment_report=report_reference, score=reference)
        return score

    def _fit_raw(self, category, entry, parent, prefix):
        from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest
        from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings, trajectory_identity
        from experiments.cat_auto_protocol_cmcc_20261104.recovery import recovery_seeds

        output = self.root / "artifacts" / prefix
        fit = output / "fit"
        request = dict(self.common, category_id=category, candidate_id=entry["candidate_id"], recipe=entry["recipe"],
            parent=parent, output=str(fit), execution_policy=dict(attention_implementation="sdpa"))
        raw_ref = self.processes.run(prefix + "/fit", "fit_worker", request, output_file=fit / "live_raw.json")
        raw = io.read(io.bound_path(raw_ref))
        io.require(raw["formal_eligible"] and raw["phase"] == "formal"
            and raw["source_parent_uuid"] == parent["uuid"], "The actual live raw must be fully fitted on this sole current parent.")
        seeds = recovery_seeds(self.batch["inventory"]["model_identity"], category, self.batch["manifest"]["run_identity"]["search_seed"])
        recipe_id = digest(entry["recipe"])
        baseline_id = trajectory_identity(category_id=category, parent_id=parent["uuid"],
            raw_id=raw["round_base_uuid"], recipe_id=recipe_id, settings=RecoverySettings(), seed_identity=seeds)
        raw_export = output / "raw_export"
        report = self.processes.run(prefix + "/raw_export", "recovery_worker", dict(self.common,
            operation="export_zero", live_raw=raw_ref, settings=asdict(RecoverySettings()), output=str(raw_export),
            destination=str(output / "raw_deployment"), trajectory_id=baseline_id, recipe_id=recipe_id,
            source_candidate_id=entry["candidate_id"]), output_file=raw_export / "deployment_report.json")
        score = self._score(prefix + "/raw_full8", report, output / "raw_full8")
        context = dict(entry=entry, parent_id=parent["uuid"], raw_reference=raw_ref, raw=raw,
            raw_score=score, seeds=seeds, recipe_id=recipe_id)
        with self.state_lock:
            self.state["categories"][category]["raws"][entry["candidate_id"]] = dict(live_raw=raw_ref,
                raw_deployment=report, raw_score=score_record(score), recipe_id=recipe_id, baseline_trajectory_id=baseline_id,
                parent_id=parent["uuid"], source_candidate_id=entry["candidate_id"],
                actual_live_candidate_id=raw["candidate_id"], offline_source_candidate_id=entry["candidate_id"],
                is_anchor=entry["is_anchor"])
            self._persist("actual_raw_full8_complete")
        return context

    def _trajectory(self, context, settings, kind, prefix):
        from experiments.cat_auto_protocol_cmcc_20261104.nodes import NodeScore, TrajectoryResult, trajectory_identity

        raw, entry = context["raw"], context["entry"]
        category = raw["current_category"]
        trajectory_id = trajectory_identity(category_id=category, parent_id=context["parent_id"], raw_id=raw["round_base_uuid"],
            recipe_id=context["recipe_id"], settings=settings, seed_identity=context["seeds"])
        output = self.root / "artifacts" / prefix
        with self.state_lock:
            self.quota.register_start(category_id=category, raw_id=raw["round_base_uuid"], settings=settings, kind=kind)
            self._persist("trajectory_start_reserved")
        request = dict(self.common, operation="train", live_raw=context["raw_reference"], settings=asdict(settings),
            output=str(output / "train"), trajectory_id=trajectory_id, recipe_id=context["recipe_id"],
            source_candidate_id=entry["candidate_id"], technical_stop_after=None)
        result_ref = self.processes.run(prefix + "/train", "recovery_worker", request,
            output_file=output / "train" / "recovery_result.json")
        result = io.read(io.bound_path(result_ref))
        io.require(result["settings"] == asdict(settings) and result["trajectory_id"] == trajectory_id
            and result["actual_run"]["actual_steps"] == settings.schedule_max
            and tuple(row["step"] for row in result["nodes"]) == settings.node_steps,
            "Training must complete its own unchanged full schedule and all declared native nodes.")

        def node(step):
            node_prefix = prefix + f"/node_{step}"
            node_output = output / f"node_{step}"
            export_request = dict(self.common, operation="export", recovery_result=result_ref,
                settings=asdict(settings), node_step=step, output=str(node_output / "export"),
                destination=str(node_output / "deployment"))
            report = self.processes.run(node_prefix + "/export", "recovery_worker", export_request,
                output_file=node_output / "export" / "deployment_report.json")
            score = self._score(node_prefix + "/full8", report, node_output / "full8")
            source = next(row["source_node_id"] for row in result["nodes"] if row["step"] == step)
            return NodeScore(step, score, source)

        nodes = self._parallel([lambda step=step: node(step) for step in settings.node_steps], workers=4)
        fee = raw["actual_ledger"]["per_category"][category]
        row = TrajectoryResult(category_id=category, parent_id=context["parent_id"], raw_id=raw["round_base_uuid"],
            recipe_id=context["recipe_id"], candidate_id=entry["candidate_id"], trajectory_id=trajectory_id,
            is_anchor=entry["is_anchor"], logical_bits=fee["logical_bits"], packed_bits=fee["packed_bits"],
            settings=settings, raw_score=context["raw_score"], nodes=tuple(sorted(nodes, key=lambda node: node.step)))
        io.write(output / "complete_trajectory.json", trajectory_record(row))
        with self.state_lock:
            self.state["categories"][category]["trajectories"][trajectory_id] = dict(kind=kind, result=io.binding(output / "complete_trajectory.json"),
                recovery_result=result_ref, full8_complete=True)
            self._persist("complete_trajectory_full8")
        return context, row

    def _category(self, category, parent):
        from experiments.cat_auto_protocol_cmcc_20261104.nodes import RecoverySettings
        from .boundary import complete_boundary_reference
        from .evaluate import read_score
        from .selection import selected_category

        io.require(parent["uuid"] == self.chain.live_parent_id, "Do not precompute a child before its unique selected parent exists.")
        entries = self.batch["recipes"][category]
        prefixes = [category + "/" + ("anchor" if entry["is_anchor"] else "different") for entry in entries]
        tasks = {}
        for prefix in prefixes:
            tasks.update(io.raw_tasks(prefix))
            tasks.update(io.trajectory_tasks(prefix + "/baseline", RecoverySettings()))
        self.budget.reserve(tasks, counts=dict(raws=2, trajectories=2, KD_updates=1000, raw_full8=2, node_full8=8))
        with self.state_lock:
            self.state.update(status="running", current_category=category)
            self.state["categories"][category] = dict(parent_id=parent["uuid"], raws={}, trajectories={}, status="running")
            self._persist("category_complete_baseline_dependencies_reserved")

        def baseline(entry, prefix):
            context = self._fit_raw(category, entry, parent, prefix)
            return self._trajectory(context, RecoverySettings(), "baseline", prefix + "/baseline")

        completed = self._parallel([lambda entry=entry, prefix=prefix: baseline(entry, prefix)
            for entry, prefix in zip(entries, prefixes, strict=True)], workers=2)
        contexts = {context["raw"]["round_base_uuid"]: context for context, _row in completed}
        results = sorted([row for _context, row in completed], key=lambda row: not row.is_anchor)
        index = io.ORDER.index(category)
        projections = self.batch["inventory"]["projections"]
        owners = lambda categories: [row["actual_name"] for row in projections if row["category_id"] in categories]
        arm = self.quota.decide(results, prefix_targets=owners(io.ORDER[:index + 1]),
            current_targets=owners((category,)), future_dense_targets=owners(io.ORDER[index + 1:]))
        self._persist("optional_arm_quota_decided_before_results")
        if arm is not None:
            plans = [("third", arm.raw_id, category + "/third")]
            if arm.interaction_eligible:
                plans.append(("interaction", self.quota.interaction_raw(arm, results), category + "/interaction"))
            optional_tasks = {}
            for _kind, _raw, prefix in plans:
                optional_tasks.update(io.trajectory_tasks(prefix, arm.settings))
            self.budget.reserve(optional_tasks, counts=dict(trajectories=len(plans),
                KD_updates=len(plans) * arm.settings.schedule_max,
                node_full8=len(plans) * len(arm.settings.node_steps)))
            self._persist("all_announced_optional_dependencies_reserved")
            additional = self._parallel([lambda kind=kind, raw=raw, prefix=prefix:
                self._trajectory(contexts[raw], arm.settings, kind, prefix) for kind, raw, prefix in plans], workers=2)
            results.extend(row for _context, row in additional)
        winner, node, certificate = self.quota.select_completed_category(results)
        selected = selected_category(winner, node, seed_identity=contexts[winner.raw_id]["seeds"])
        deployment = self.deployments[node.checkpoint_id]
        report = io.read(io.bound_path(deployment["deployment_report"]))
        score = read_score(deployment["score"])
        with self.budget.cpu_window(category + "/accept_existing_evaluated_boundary"):
            boundary = complete_boundary_reference(export_report=report, inventory=self.batch["inventory"], full8_score=score)
            committed = self.chain.commit(selected, live_parent_id=parent["uuid"], boundary_reference=boundary,
                selection_certificate=certificate, actual_round_base_id=winner.raw_id, restored_node_id=node.checkpoint_id,
                runtime_identity=boundary["runtime_identity"])
        category_output = self.root / "categories" / category
        io.write(category_output / "selection_certificate.json", certificate)
        io.write(category_output / "selected_recipe.json", dict(selected=asdict(selected), replay=selected.training_plan(),
            actual_schedule_max=winner.settings.schedule_max, selected_step=node.step,
            actual_completed_trajectories=[row.trajectory_id for row in results], evaluated_artifact=deployment))
        io.write(category_output / "boundary.json", boundary)
        self.state["categories"][category].update(status="completed", selected=asdict(selected),
            selection_certificate=io.binding(category_output / "selection_certificate.json"),
            boundary=io.binding(category_output / "boundary.json"), source_deployment=deployment, chain_commit=committed)
        self._persist("category_selected_and_strict_chain_committed")
        self._prune_unselected_deployments(category, selected.source_node_id, results)
        return boundary

    def _prune_unselected_deployments(self, category, selected_source, results):
        """After complete selection, discard only unreferenced generated dense payloads."""
        sources = {row.raw_id for row in results} | {node.checkpoint_id for row in results for node in row.nodes}
        paths = []
        for source in sorted(sources - {selected_source}):
            deployment = self.deployments[source]
            report = io.read(io.bound_path(deployment["deployment_report"]))
            directory = Path(report["path"]).resolve()
            io.require(directory.is_relative_to(self.root / "artifacts" / category)
                and report["phase"] == "formal" and report["export_evidence"]["source_node_id"] == source,
                "Cleanup is restricted to this completed category's own nonselected deployments.")
            io.bound_path(deployment["score"])
            meta = io.read(directory / "checkpoint_meta.json")
            weight = directory / meta["state_dict_file"]
            io.require(not weight.is_symlink() and weight.resolve().is_relative_to(directory)
                and weight.is_file() and meta["checkpoint_id"] == report["uuid"],
                "Only the exact actual generated nonselected weight file may be removed.")
            paths.append(dict(path=str(weight.resolve()), bytes=weight.stat().st_size, source_checkpoint_id=source,
                deployment_id=report["uuid"], reason="All announced full8 comparisons finished; the sole next parent is another selected deployment. Keep config, native metadata, reports, raw source, mutable nodes and original evaluation shards."))
        cleanup = self.root / "categories" / category / "deployment_cleanup.json"
        io.write(cleanup, dict(status="planned", exact_files=paths, released_bytes=0))
        released = 0
        for row in paths:
            path = Path(row["path"])
            io.require(path.stat().st_size == row["bytes"], "Nonselected deployment changed before cleanup.")
            path.unlink()
            io.require(not path.exists(), "The exact nonselected weight was not removed.")
            released += row["bytes"]
        io.write(cleanup, dict(status="completed", exact_files=paths, released_bytes=released,
            selected_source_id=selected_source, preserved="All actual raw sources/native training nodes/selected deployment/full8 original evidence"), replace=True)
        self.state["categories"][category]["cleanup"] = io.binding(cleanup)
        self._persist("completed_category_nonselected_dense_payloads_pruned")

    def run(self):
        self.state["status"] = "running"
        self._persist("formal_batch_started")
        parent = self.batch["root_boundary"]
        for category in io.ORDER[:2]:
            parent = self._category(category, parent)
        io.require(self.quota.selected_categories == list(io.ORDER[:2]) and len(self.chain.boundaries) == 2,
            "The finite batch stops exactly after the unique o boundary.")
        io.write(self.root / "after_o.json", parent)
        io.write(self.root / "frozen_recipe.json", dict(schema="sc_cat_partial_frozen_recipe_v1", phase="formal",
            frozen_order=list(io.ORDER), completed_categories=list(io.ORDER[:2]),
            shared_BF16_root=self.batch["root_boundary_reference"],
            selected_settings=self.batch["authorization"]["selected_settings"],
            categories=[self.state["categories"][category]["selected"] for category in io.ORDER[:2]],
            final_boundary=io.binding(self.root / "after_o.json"), quotas=io.binding(self.root / "quotas.json"),
            subsequent_categories_started=False, BF16_recipe_replay_completed=False,
            LongBench_completed=False, complete_seven_category_chain=False))
        self.state.update(status="completed", exit_code=0, ended_epoch=time.time(), after_o=io.binding(self.root / "after_o.json"))
        self._persist("finite_k_o_batch_completed_no_further_tasks")
        io.write(self.root / "controller.exit.json", dict(status="completed", exit_code=0,
            ended_epoch=self.state["ended_epoch"], state=io.binding(self.root / "controller_state.json"),
            actual_GPU_hours=self.budget.value["total_actual_GPU_hours"]))

    def fail(self, exc):
        self.processes.cancel()
        self.state.update(status="failed", exit_code=1, ended_epoch=time.time(),
            error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        self._persist("dependency_failed_stop_no_retry")
        io.write(self.root / "controller.exit.json", dict(status="failed", exit_code=1,
            ended_epoch=self.state["ended_epoch"], state=io.binding(self.root / "controller_state.json"),
            actual_GPU_hours=self.budget.value["total_actual_GPU_hours"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--root-boundary", required=True)
    parser.add_argument("--technical-fit", nargs="+", required=True)
    parser.add_argument("--technical-recovery", required=True)
    parser.add_argument("--readout-gpus", nargs="+", type=int, default=[2, 3, 4, 5])
    args = parser.parse_args()
    controller = None
    try:
        controller = Controller(authorization=args.authorization, root_boundary=args.root_boundary,
            technical_fit=args.technical_fit, technical_recovery=args.technical_recovery, readout_gpus=args.readout_gpus)
        def interrupted(signum, _frame):
            controller.processes.cancel()
            raise InterruptedError("Finite controller received signal " + str(signum))
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
