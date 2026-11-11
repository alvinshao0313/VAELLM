"""Use the native v6 loader, original scheduler and original data skip at up250."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from transformers import TrainerCallback

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.data import batch_digest
from experiments.cat_auto_protocol_cmcc_20261104.live import build_live_recovery
from experiments.sc_cat_live_cmcc_20261106 import recovery_io
from experiments.sc_cat_pipeline_cmcc_20261106.node_export import read_committed_node
from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state
from experiments.sc_cat_continue_cmcc_20261107.bindings import write


def write_proof(path, proof):
    write(path, proof, replace=Path(path).exists())


def equal_tree(actual, expected):
    if isinstance(expected, np.ndarray):
        return isinstance(actual, np.ndarray) and actual.dtype == expected.dtype and np.array_equal(actual, expected)
    if isinstance(expected, torch.Tensor):
        return isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype and torch.equal(actual.detach().cpu(), expected.cpu())
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(equal_tree(actual[key], val) for key, val in expected.items())
    if isinstance(expected, (tuple, list)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(equal_tree(a, b) for a, b in zip(actual, expected))
    return actual == expected


class ResumeObserver:
    def __init__(self, manifest, checkpoint, start):
        self.manifest, self.checkpoint, self.start, self.observed = manifest, checkpoint, start, []

    def __call__(self, inputs, update_index):
        index = int(update_index)
        require(index == self.start + len(self.observed), "The resumed actual update cursor changed.")
        expected = self.manifest["batches"][index]
        actual = batch_digest(inputs)
        require(actual == expected["input_sha256"], "Resumed tokens/labels/padding differ from the original update index.")
        self.observed.append(dict(update_index=index, input_sha256=actual,
            record_ids=[row["record_id"] for row in expected["records"]]))
        return self.observed[-1]

    def report(self):
        return dict(actual_updates=self.start + len(self.observed), actual_record_presentations=4 * (self.start + len(self.observed)),
            completed_prefix_updates=self.start, completed_prefix_basis=self.checkpoint,
            resumed_updates=len(self.observed), resumed_record_presentations=4 * len(self.observed),
            resumed_token_order_digest=digest(self.observed), rows=self.observed, group_by_length=False,
            rows_scope="Only actually observed resumed updates; the saved prefix is bound separately, never fabricated.")


class RestoreProof(TrainerCallback):
    def __init__(self, run, checkpoint, destination, technical):
        self.run, self.checkpoint, self.destination, self.technical = run, checkpoint, Path(destination), technical
        self.checked = False

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.run.trainer
        require(state.global_step == 250 and args.max_steps == 500 and trainer._v6_exact_resume_loaded,
            "The native loader must restore step250 under the original500 schedule.")
        expected = torch.load(Path(self.checkpoint) / "training_model_state.pt", map_location="cpu", weights_only=True)
        model = trainer.accelerator.unwrap_model(trainer.model)
        actual, _components, _manifest = collect_model_level_mutable_state(model,
            selection=trainer.model_level_trainable_selection, selected_vae_modules=trainer._v6_selected_vae_modules)
        require(equal_tree(actual, expected), "Every saved mutable master must be restored exactly before training.")
        optimizer = torch.load(Path(self.checkpoint) / "optimizer.pt", map_location="cpu", weights_only=False)
        scheduler = torch.load(Path(self.checkpoint) / "scheduler.pt", map_location="cpu", weights_only=False)
        require(equal_tree(trainer.optimizer.state_dict(), optimizer), "The actual Adam state/groups differ after native resume.")
        require(equal_tree(trainer.lr_scheduler.state_dict(), scheduler), "The original500 scheduler was not restored exactly.")
        self.proof = dict(status="saved_masters_Adam_scheduler_restored", phase="technical_validation" if self.technical else "formal",
            formal_eligible=False, global_step=250, schedule_max=500, mutable_exact=True, optimizer_exact=True,
            scheduler_exact=True, native_immutable_contract_checked=True, checkpoint=str(self.checkpoint),
            paired_uninterrupted_trajectory_verified=False)
        write_proof(self.destination, self.proof)

    def on_step_begin(self, args, state, control, **kwargs):
        if self.checked:
            return
        require(state.global_step == 250, "Verify restored RNG before the first resumed update.")
        rng = torch.load(Path(self.checkpoint) / "rng_state.pth", map_location="cpu", weights_only=False)
        require(torch.equal(torch.get_rng_state(), rng["cpu"]), "CPU RNG differs before the resumed forward.")
        expected = rng["cuda"]
        actual = torch.cuda.random.get_rng_state_all() if isinstance(expected, list) else torch.cuda.random.get_rng_state()
        require(equal_tree(actual, expected), "CUDA RNG differs before the resumed forward.")
        require(equal_tree(random.getstate(), rng["python"]) and equal_tree(np.random.get_state(), rng["numpy"]),
            "Python/NumPy RNG differs before the resumed forward.")
        self.checked = True
        self.proof.update(status="native_saved_state_and_RNG_verified", CPU_RNG_exact=True, CUDA_RNG_exact=True,
            Python_RNG_exact=True, Numpy_RNG_exact=True)
        write_proof(self.destination, self.proof)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 252:
            require(self.checked and len(self.run.observer.observed) == 2, "Both first actual resumed batches must be observed.")
            self.proof.update(status="native_resume_first_two_actual_updates_verified", resumed_actual_steps=2,
                first_actual_update_indices=[row["update_index"] for row in self.run.observer.observed],
                resumed_input_hashes_verified=True, actual_global_step=252)
            write_proof(self.destination, self.proof)
            print(dict(event="native_resume_verified", global_step=252, original_schedule=500,
                actual_batch_indices=self.proof["first_actual_update_indices"]), flush=True)
            if self.technical:
                control.should_training_stop = True
        return control


class ResumedRun:
    def __init__(self, run, checkpoint, prefix, technical, actual_request_reference):
        self.__dict__.update(run.__dict__)
        self.original, self.checkpoint, self.prefix, self.technical = run, Path(checkpoint), prefix, technical
        self.actual_request_reference = actual_request_reference
        self.observer = ResumeObserver(run.observer.manifest, prefix[-1], 250)
        self.trainer.input_observer = self.observer
        self.proof = RestoreProof(self, self.checkpoint, Path(self.stage.output_dir) / "resume_verification.json", technical)
        self.trainer.add_callback(self.proof)

    def train(self):
        require(self.trainer.state.global_step == 0, "Construct from the actual original raw, then use the native saved-state loader.")
        result = self.trainer.train(resume_from_checkpoint=str(self.checkpoint))
        stop = 252 if self.technical else 500
        require(self.trainer.state.global_step == stop and len(self.observer.observed) == stop - 250
            and self.proof.checked, "The resumed actual segment did not complete its declared original schedule.")
        provenance = dict(self.provenance, native_resume_source=self.prefix[-1],
            saved_state_restore_verified=True, paired_uninterrupted_trajectory_verified=False)
        return dict(status="completed_native_KD_trajectory", actual_steps=stop, schedule_max=500,
            provenance=provenance, actual_data=self.observer.report(), save_audits=list(self.trainer.node_save_audits),
            inherited_node_commits=self.prefix, trainer_metrics=result.metrics, formal_eligible=False,
            boundary="The saved prefix and actual resumed segment still require complete native node exports/full8.")

    def close(self):
        self.original.close()


def build(snapshot, request, *, technical=False, actual_request_reference=None, **kwargs):
    original = recovery_io.read_bound_json(snapshot["training_request"])
    require(request["live_raw"] == original["live_raw"] and request["settings"] == original["settings"]
        and request["trajectory_id"] == original["trajectory_id"] and request["phase"] == original["phase"] == "formal"
        and request["settings"]["schedule_max"] == 500 and request["settings"]["scope"] == original["settings"]["scope"],
        "Resume exactly the interrupted actual up trajectory, not another raw or configuration.")
    prefix = [read_committed_node(snapshot["training_request"], step) for step in (125, 250)]
    require(all(prefix), "Both original prefix nodes must be complete native commits.")
    run = build_live_recovery(**kwargs)
    directory = Path(run.trainer.args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for row in prefix:
        target = directory / f"checkpoint-{row['step']}"
        require(not target.exists(), "Never replace a resume output or existing saved node.")
        target.symlink_to(row["node_directory"], target_is_directory=True)
    require(technical or actual_request_reference is not None, "Bind the real formal execution request before native resume.")
    return ResumedRun(run, prefix[-1]["node_directory"], prefix, technical, actual_request_reference)


def node_receipts(run, **kwargs):
    require([row["step"] for row in run.trainer.node_save_audits] == [375, 500],
        "Only the two new saved nodes have new save-return audits; retain the two actual old commits separately.")
    rows = [read_committed_node(run.actual_request_reference, step) for step in (125, 250, 375, 500)]
    require(all(rows) and rows[:2] == run.prefix, "The resumed trajectory changed either original prefix node.")
    require(all(row["trajectory_id"] == kwargs["trajectory_id"] and row["recipe_id"] == kwargs["recipe_id"]
        and row["source_candidate_id"] == kwargs["source_candidate_id"] for row in rows),
        "Every actual inherited/resumed node retains its original semantic trajectory and recipe.")
    return rows
