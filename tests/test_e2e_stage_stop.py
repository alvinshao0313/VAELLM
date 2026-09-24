from __future__ import annotations

import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import IterableDataset
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerState

from compressed_e2e_fintuning import mid_eval
from compressed_e2e_fintuning.runtime_v6_pipeline import _validate_stop_after_step
from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from compressed_e2e_fintuning.v6_runtime_state import build_e2e_immutable_resume_contract
from train_utils import checkpoint_v6 as v6
from train_utils.config.cli import parse_e2e_cli


def _cfg(*extra):
    return parse_e2e_cli([
        '--student_checkpoint_dir', '/tmp/base', '--dataset_mix', 'openorca',
        '--steps', '4000', '--eval_after_save', 'true', '--eval_tasks', 'boolq,rte',
        *extra,
    ])


def test_stop_budget_is_optional_and_does_not_change_optimization():
    default = _cfg()
    paused = _cfg('--stop_after_step', '200')
    assert default.stop_after_step is None
    assert paused.stop_after_step == 200
    assert default.opt == paused.opt
    contract_args = dict(
        training_args=SimpleNamespace(save_strategy='steps', save_steps=200, world_size=2),
        tokenizer=SimpleNamespace(name_or_path='same-tokenizer'), input_checkpoint_id='same-base',
        resolved_target_layers=[0], resolved_target_modules=['q_proj'], teacher_identity=None,
    )
    assert build_e2e_immutable_resume_contract(cfg=default, **contract_args) == build_e2e_immutable_resume_contract(
        cfg=paused, **contract_args)
    for value in ('0', '-1', '4000', '4001'):
        with pytest.raises(SystemExit):
            _cfg('--stop_after_step', value)


def test_stage_stop_requires_saved_evaluated_future_boundary(tmp_path):
    cfg = _cfg('--stop_after_step', '200')
    args = SimpleNamespace(max_steps=4000, save_strategy='steps', save_steps=200)
    assert _validate_stop_after_step(cfg, args) == 200
    args.save_steps = 300
    with pytest.raises(ValueError, match='boundary'):
        _validate_stop_after_step(cfg, args)
    args.save_steps = 0.1
    with pytest.raises(ValueError, match='integer'):
        _validate_stop_after_step(cfg, args)
    args.save_steps = 200
    cfg.runtime.evaluation.eval_after_save = False
    with pytest.raises(ValueError, match='eval_after_save'):
        _validate_stop_after_step(cfg, args)
    cfg.runtime.evaluation.eval_after_save = True
    cfg.resume_from_checkpoint = str(tmp_path)
    TrainerState(global_step=200).save_to_json(str(tmp_path / 'trainer_state.json'))
    with pytest.raises(ValueError, match='exceed'):
        _validate_stop_after_step(cfg, args)
    cfg.stop_after_step = 400
    assert _validate_stop_after_step(cfg, args) == 400


class _Rows(IterableDataset):
    def __iter__(self):
        for i in range(32):
            yield {'input_ids': torch.arange(4, dtype=torch.float32) + i / 10}


class _DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, input_ids):
        return self.proj(self.dropout(input_ids))


class _MSETrainer(VAEDecoderE2ETrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        output = model(inputs['input_ids'])
        loss = output.square().mean()
        return (loss, {'logits': output}) if return_outputs else loss


def _context(round_base, checkpoint_id):
    return {
        'round_base_dir': str(round_base), 'round_base_checkpoint_id': checkpoint_id,
        'train_mode': 'none', 'compressed_targets': (), 'pending_dense_targets': (),
        'skip_targets': (), 'legacy_original_only_sources': (),
        'norm_train_mode': 'none', 'lm_head_train_mode': 'none', 'lora_config': None,
        'resolved_learning_rates': {'learning_rate': 1e-3}, 'compression_categories': (),
        'target_layers': (), 'target_modules': (),
        'immutable_resume_contract': {'max_steps': 4},
        'base_model_path': 'stage-stop-cpu-test', 'runtime_audit': {}, 'hf_artifact_refs': {},
    }


def _trainer(root, round_base, checkpoint_id, initial, stop):
    model = _DropoutModel()
    model.load_state_dict(initial)
    args = TrainingArguments(
        output_dir=str(root), use_cpu=True, max_steps=4, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=1e-3, lr_scheduler_type='cosine',
        warmup_steps=1, save_strategy='steps', save_steps=2, save_safetensors=False,
        report_to=[], disable_tqdm=True, seed=55, data_seed=55,
    )
    callback = mid_eval.EvalAfterSaveCallback(
        e2e_args=SimpleNamespace(eval_after_save=True, eval_device='cpu'),
        tokenizer=None, base_model_path='stage-stop-cpu-test', run_output_dir=str(root),
        log=logging.getLogger(__name__), parallel_mode='dp', stop_after_step=stop,
    )
    trainer = _MSETrainer(model=model, args=args, train_dataset=_Rows(), callbacks=[callback])
    trainer.configure_v6_step_checkpoint(
        context=_context(round_base, checkpoint_id), selected_vae_modules=(),
    )
    callback.bind_trainer(trainer)
    return trainer, callback


def _assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_tree_equal(x, y)
    else:
        assert a == b


def test_v6_resume_after_evaluation_preserves_next_updates_and_rng(tmp_path, monkeypatch):
    # Isolate lm-eval's documented seed-reset side effect; actual model updates,
    # v6 checkpoint IO, Trainer data skip, optimizer and scheduler remain real.
    observed = []

    def evaluation(**kwargs):
        random.seed(0)
        np.random.seed(1234)
        torch.manual_seed(1234)
        random.random()
        np.random.random()
        torch.rand(3)
        observed.append(torch.get_rng_state().clone())

    monkeypatch.setattr(mid_eval, 'run_e2e_lm_eval', evaluation)
    torch.manual_seed(7)
    model = _DropoutModel()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    base = tmp_path / 'round_base'
    meta = v6.save_v6_full_checkpoint(
        model, str(base), checkpoint_kind='round_base', compressed_targets=(),
        train_mode='none', base_model_path='stage-stop-cpu-test', save_config=False,
    )
    continuous, continuous_cb = _trainer(tmp_path / 'continuous', base, meta['checkpoint_id'], initial, None)
    continuous.train()
    assert continuous.state.global_step == 4
    assert continuous_cb.stopped_checkpoint_dir is None

    interrupted, paused_cb = _trainer(tmp_path / 'paused', base, meta['checkpoint_id'], initial, 2)
    interrupted.train()
    assert interrupted.state.global_step == 2
    step = Path(paused_cb.stopped_checkpoint_dir)
    assert step.name == 'checkpoint-2'
    rng = torch.load(step / 'rng_state.pth', weights_only=False)
    assert torch.equal(rng['cpu'], observed[-1])
    assert rng['python'] == random.getstate()
    assert np.array_equal(rng['numpy'][1], np.random.get_state()[1])
    assert (step / 'optimizer.pt').is_file()
    assert (step / 'scheduler.pt').is_file()
    assert (step / v6.TRAINING_MODEL_STATE_FILENAME).is_file()
    assert not (step / 'pytorch_model.bin').exists()

    resumed, _ = _trainer(tmp_path / 'paused', base, meta['checkpoint_id'], initial, None)
    resumed.train(resume_from_checkpoint=str(step))
    assert resumed._v6_exact_resume_loaded
    assert resumed.state.global_step == 4
    _assert_tree_equal(continuous.model.state_dict(), resumed.model.state_dict())
    _assert_tree_equal(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    _assert_tree_equal(continuous.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
