"""Reuse actual native KD/save/load, retaining a500 schedule across250 resume."""
from pathlib import Path

import torch
from transformers import TrainerCallback

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require
from experiments.cat_auto_protocol_cmcc_20261104.recovery import NativeNodeTrainer as OriginalTrainer, recovery_seeds
from experiments.sc_cat_finish_cmcc_20261108.resume import ResumeObserver, equal_tree
from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state


class StopAtNode(TrainerCallback):
    def __init__(self, stop):
        self.stop = stop

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.stop:
            control.should_save = True
            control.should_training_stop = True
        return control


class NativeNodeTrainer(OriginalTrainer):
    def __init__(self, *args, stop_at=None, **kwargs):
        super().__init__(*args, **kwargs)
        require(stop_at is None or stop_at in self.recovery_settings.node_steps, 'Only original-grid pause points are supported.')
        if stop_at is not None:
            self.add_callback(StopAtNode(stop_at))

    def create_optimizer(self):
        require(self.optimizer is not None, 'Construct the actual owned D/P/L optimizer before native training.')
        return self.optimizer

    def _save_checkpoint(self, model, trial):
        result = super()._save_checkpoint(model, trial)
        from .io import write
        audit = self.node_save_audits[-1]
        write(Path(audit['directory'])/'node_ready.json', audit)
        return result


class VerifyRestore(TrainerCallback):
    def __init__(self, run, checkpoint):
        self.run, self.checkpoint, self.checked = run, Path(checkpoint), False
        self.start = run.resume_step

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.run.trainer
        require(state.global_step == self.start and args.max_steps == 500 and trainer._v6_exact_resume_loaded,
            'Resume the original saved node under the unchanged500 scheduler.')
        actual, _, _ = collect_model_level_mutable_state(trainer.accelerator.unwrap_model(trainer.model),
            selection=trainer.model_level_trainable_selection, selected_vae_modules=trainer._v6_selected_vae_modules)
        require(equal_tree(actual, torch.load(self.checkpoint/'training_model_state.pt', map_location='cpu', weights_only=True)),
            'Complete D/P/L masters changed on resume.')
        for filename, actual in (('optimizer.pt', trainer.optimizer.state_dict()), ('scheduler.pt', trainer.lr_scheduler.state_dict())):
            require(equal_tree(actual, torch.load(self.checkpoint/filename, map_location='cpu', weights_only=False)),
                'Saved optimizer/kappa/scheduler changed: '+filename)

    def on_step_begin(self, args, state, control, **kwargs):
        if self.checked:
            return
        import random
        import numpy as np
        saved = torch.load(self.checkpoint/'rng_state.pth', map_location='cpu', weights_only=False)
        require(state.global_step == self.start and torch.equal(torch.get_rng_state(), saved['cpu'])
            and equal_tree(random.getstate(), saved['python']) and equal_tree(np.random.get_state(), saved['numpy']),
            'Host RNG differs before the first actual resumed forward.')
        if torch.cuda.is_initialized():
            actual = torch.cuda.get_rng_state_all() if isinstance(saved['cuda'], list) else torch.cuda.get_rng_state()
            require(equal_tree(actual, saved['cuda']), 'CUDA RNG differs on resume.')
        self.checked = True


class ResumedRun:
    def __init__(self, run, checkpoint):
        from train_utils import checkpoint_v6 as v6
        self.__dict__.update(run.__dict__)
        self.original, self.checkpoint = run, Path(checkpoint).resolve()
        node = v6.load_v6_training_step_meta(str(self.checkpoint))
        from .io import read
        self.resume_step = read(self.checkpoint/'trainer_state.json')['global_step']
        require(self.stage.config.opt.steps == 500 and self.resume_step in (250,375)
            and self.checkpoint.name == f'checkpoint-{self.resume_step}'
            and node['immutable_resume_contract']['sc_cat_recovery_settings'] == self.provenance['settings'],
            'Resume only the same complete saved state and settings.')
        self.observer = ResumeObserver(run.observer.manifest, str(self.checkpoint), self.resume_step)
        self.trainer.input_observer = self.observer
        self.proof = VerifyRestore(self, self.checkpoint)
        self.trainer.add_callback(self.proof)

    def train(self):
        result = self.trainer.train(resume_from_checkpoint=str(self.checkpoint))
        require(self.trainer.state.global_step == self.expected_stop and self.proof.checked
            and len(self.observer.observed) == self.expected_stop-self.resume_step, 'Incomplete resumed original trajectory.')
        return dict(status='completed_native_KD_trajectory', actual_steps=self.expected_stop,
            schedule_max=500, resumed_from=str(self.checkpoint), provenance=self.provenance,
            starting_step=self.resume_step, actual_update_count=self.expected_stop-self.resume_step,
            actual_data=self.observer.report(), save_audits=list(self.trainer.node_save_audits),
            trainer_metrics=result.metrics, saved_state_restore_verified=True,
            paired_uninterrupted_trajectory_verified=False)

    def close(self):
        self.original.close()
