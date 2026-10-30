"""Reuse mature Q/K inventory checks and record native saves without evaluation."""
from pathlib import Path

import common as c
from experiments.llama_fullqk_qkonly_lm.callback import QKStepCallback
from train_utils.checkpoint_v6 import load_v6_training_step_meta

SAVE_STEPS = [400, 800, 1200, 1600, 2000]


class AlignedQKCallback(QKStepCallback):
    def __init__(self, *, record, source_dir, run_root_dir):
        self.record = record
        self.source_dir = Path(source_dir).resolve()
        self.run_root_dir = Path(run_root_dir).resolve()
        self.completed_steps = []

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        c.require(step == SAVE_STEPS[len(self.completed_steps)], 'Unexpected native QK save boundary.')
        checkpoint = Path(args.output_dir).resolve() / f'checkpoint-{step}'
        meta = load_v6_training_step_meta(str(checkpoint))
        saved = c.read(checkpoint / 'trainer_state.json')
        c.require(meta['round_base_checkpoint_id'] == c.SOURCE_ID and meta['train_mode'] == 'decoder'
                  and meta['target_layers'] == list(range(36))
                  and set(meta['target_modules']) == {'q_proj', 'k_proj'}
                  and saved['global_step'] == step and saved['max_steps'] == 2000,
                  'Saved native checkpoint changed source, Q/K scope or original schedule.')
        self.completed_steps.append(step)
        c.write_json(c.OUTPUT / 'native_saves.json', dict(completed_steps=self.completed_steps,
                     latest_checkpoint=str(checkpoint), source_checkpoint_id=c.SOURCE_ID,
                     external_export_or_evaluation=False))
        return control
