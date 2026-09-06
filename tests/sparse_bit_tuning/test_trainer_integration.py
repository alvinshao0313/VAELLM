import tempfile
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.module import BankSpec, SparseBitTuningModule
from sparse_bit_tuning.optimizer import BitOptimizerManager, SparseBitCompositeOptimizer


class _TinyModel(nn.Module):
    def __init__(self, *, train_main: bool):
        super().__init__()
        self.main = nn.Parameter(torch.tensor([2.0], dtype=torch.float32), requires_grad=train_main)
        spec = BankSpec(
            canonical_key="layer|stage=0|part=0",
            module_path="layer",
            stage_idx=0,
            part_idx=0,
            logical_shape=(2, 1, 8),
            n_bits=16,
            n_active=2,
            device=torch.device("cpu"),
        )
        self.add_module("sparse_bit_tuning", SparseBitTuningModule([spec]))

    def forward(self, input_ids=None, labels=None, **kwargs):
        del input_ids, labels, kwargs
        score = self.sparse_bit_tuning.score_chunks[0]
        loss = self.main.float().sum() + score.float().sum() * 0.0
        return {"loss": loss, "logits": self.main.reshape(1, 1, 1)}


class _FakeManager:
    def __init__(self, model):
        self.score_module = model.sparse_bit_tuning
        self.bit_optimizer = BitOptimizerManager(
            self.score_module,
            SparseBitTuningConfig(enabled=True, optimizer="rms_sgd", bit_lr=0.05),
        )
        self.configured_steps = None
        self.initialized = False

    def configure_schedule(self, *, total_optimizer_steps):
        self.configured_steps = int(total_optimizer_steps)

    def initialize_scores(self):
        self.initialized = True

    def optimizer_step(self):
        return SimpleNamespace(
            global_bit_round=0,
            bit_round_step=1,
            step_flip_count=0,
            cumulative_flip_count=0,
            stable_counter=0,
            stable_steps=3,
            had_flip=False,
            round_ended=False,
        )


def _trainer(
    model,
    manager,
    output_dir,
    *,
    max_grad_norm=1.0,
    decoder_param_ids=None,
    decoder_lr=None,
):
    args = TrainingArguments(
        output_dir=output_dir,
        use_cpu=True,
        learning_rate=1e-3,
        max_grad_norm=max_grad_norm,
        report_to=[],
    )
    return VAEDecoderE2ETrainer(
        model=model,
        args=args,
        loss_type="sft",
        decoder_param_ids=decoder_param_ids,
        decoder_lr=decoder_lr,
        sparse_bit_manager=manager,
    )


def test_composite_optimizer_excludes_bit_scores_from_main_optimizer():
    with tempfile.TemporaryDirectory() as tmp:
        model = _TinyModel(train_main=True)
        manager = _FakeManager(model)
        trainer = _trainer(model, manager, tmp)
        optimizer = trainer.create_optimizer()
        assert isinstance(optimizer, SparseBitCompositeOptimizer)
        assert optimizer.main_optimizer is not None
        bit_ids = model.sparse_bit_tuning.bit_parameter_ids()
        main_ids = {
            id(param)
            for group in optimizer.main_optimizer.param_groups
            for param in group["params"]
        }
        assert id(model.main) in main_ids
        assert main_ids.isdisjoint(bit_ids)
        scheduler = trainer.create_scheduler(17)
        assert manager.configured_steps == 17
        assert manager.initialized
        assert getattr(scheduler, "optimizer", None) is optimizer.main_optimizer


def test_composite_optimizer_preserves_decoder_specific_lr():
    from train_utils.model_level_optimizer import (
        ModelLevelOptimizerLRConfig,
        attach_model_level_optimizer_contract,
        selection_from_component_parameters,
    )

    with tempfile.TemporaryDirectory() as tmp:
        model = _TinyModel(train_main=True)
        manager = _FakeManager(model)
        trainer = _trainer(
            model,
            manager,
            tmp,
            decoder_param_ids=(id(model.main),),
            decoder_lr=2e-4,
        )
        attach_model_level_optimizer_contract(
            trainer,
            selection=selection_from_component_parameters(
                decoder_parameters={"decoder::main": model.main},
            ),
            lr_config=ModelLevelOptimizerLRConfig(
                learning_rate=1e-3,
                weight_decay=0.0,
                decoder_lr=2e-4,
            ),
        )
        optimizer = trainer.create_optimizer()
        assert isinstance(optimizer, SparseBitCompositeOptimizer)
        assert optimizer.main_optimizer is not None
        main_group = next(
            group
            for group in optimizer.main_optimizer.param_groups
            if any(param is model.main for param in group["params"])
        )
        assert main_group["lr"] == pytest.approx(2e-4)


def test_pure_bit_has_no_main_optimizer_and_uses_lifecycle_scheduler():
    with tempfile.TemporaryDirectory() as tmp:
        model = _TinyModel(train_main=False)
        manager = _FakeManager(model)
        trainer = _trainer(model, manager, tmp)
        optimizer = trainer.create_optimizer()
        assert isinstance(optimizer, SparseBitCompositeOptimizer)
        assert optimizer.main_optimizer is None
        scheduler = trainer.create_scheduler(9)
        assert getattr(scheduler, "_is_sparse_bit_noop_scheduler", False)
        assert not hasattr(scheduler, "optimizer")
        scheduler.step()
        assert scheduler.get_last_lr() == [0.0]


def test_hf_clip_entry_clips_main_only_and_leaves_bit_grad_unclipped():
    with tempfile.TemporaryDirectory() as tmp:
        model = _TinyModel(train_main=True)
        manager = _FakeManager(model)
        trainer = _trainer(model, manager, tmp, max_grad_norm=1.0)
        trainer.create_optimizer()
        bit = model.sparse_bit_tuning.score_chunks[0]
        model.main.grad = torch.tensor([10.0], dtype=torch.float32)
        bit.grad = torch.full_like(bit, 10.0)
        norm = trainer.accelerator.clip_grad_norm_(model.parameters(), 1.0)
        assert float(norm) == pytest.approx(10.0, rel=1e-5)
        assert abs(float(model.main.grad.item())) == pytest.approx(1.0, rel=1e-5)
        assert torch.equal(bit.grad, torch.full_like(bit, 10.0))


def test_bit_false_trainer_still_uses_stock_optimizer_path():
    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Linear(2, 2)
        args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to=[])
        trainer = VAEDecoderE2ETrainer(model=model, args=args, loss_type="sft")
        optimizer = trainer.create_optimizer()
        assert not isinstance(optimizer, SparseBitCompositeOptimizer)
        assert not hasattr(trainer.accelerator, "_sparse_bit_original_clip_grad_norm")
