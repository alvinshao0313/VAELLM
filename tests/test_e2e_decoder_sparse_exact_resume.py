from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.manager import SparseBitTuningManager
from train_utils import checkpoint_v6 as v6
from train_utils.config.configs import AuxTrainableConfig
from train_utils.decoder_execution import enable_vae_linear_by_execution_plan
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract
from train_utils.model_level_trainables import build_model_level_trainable_selection


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Sparse Bit packed autograd")


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(991)
        self.x = torch.randn(12, 4, generator=generator)
        self.target = torch.randn(12, 4, generator=generator)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return {
            "x": self.x[int(index)].clone(),
            "target": self.target[int(index)].clone(),
        }


def _decoder() -> Decoder:
    return Decoder(
        in_dim=9,
        out_dim=4,
        hidden_dim=8,
        num_res_blocks=0,
        norm_type="layer",
        decoder_type="linear",
        use_checkpoint=False,
        num_models=1,
    )


def _vae_linear() -> VAELinear:
    bits = torch.tensor(
        [
            [[1, 0, 1, 0, 1, 0, 1, 0, 1]],
            [[0, 1, 0, 1, 0, 1, 0, 1, 0]],
            [[1, 1, 0, 0, 1, 1, 0, 0, 1]],
            [[0, 0, 1, 1, 0, 0, 1, 1, 0]],
        ],
        dtype=torch.bool,
    )
    return VAELinear(
        in_features=4,
        out_features=4,
        bias=None,
        original_weight=None,
        vq_weight=bits,
        decoder=_decoder(),
        codebook_dim=4,
        transpose=False,
    )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _vae_linear()


class _SparseMSETrainer(VAEDecoderE2ETrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        parameter = next(model.layer.get_stage_part_decoder(0, 0).parameters())
        x = inputs["x"].to(device=parameter.device, dtype=parameter.dtype)
        target = inputs["target"].to(device=parameter.device, dtype=parameter.dtype)
        output = model.layer(x)
        loss = (output - target).float().square().mean()
        return (loss, {"output": output}) if return_outputs else loss


class _StopAtStepTwo(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        del args, kwargs
        if int(state.global_step) == 2:
            control.should_training_stop = True
        return control


def _context(round_base: Path, checkpoint_id: str) -> dict:
    return {
        "round_base_dir": str(round_base),
        "round_base_checkpoint_id": str(checkpoint_id),
        "train_mode": "decoder_sparse_bit",
        "compressed_targets": ("layer",),
        "pending_dense_targets": (),
        "skip_targets": (),
        "legacy_original_only_sources": (),
        "norm_train_mode": "none",
        "lm_head_train_mode": "none",
        "lora_config": None,
        "resolved_learning_rates": {
            "learning_rate": 1e-3,
            "decoder_lr": 1e-3,
            "weight_decay": 0.0,
        },
        "compression_categories": (),
        "target_layers": (),
        "target_modules": ("layer",),
        "immutable_resume_contract": {"exact_resume_test": 1, "max_steps": 4},
        "base_model_path": "tiny-decoder-sparse",
        "runtime_audit": {"test": "decoder_sparse_bit_exact_resume"},
        "hf_artifact_refs": {},
    }


def _load_round_base(round_base: Path) -> _TinyModel:
    model = _TinyModel()
    model, _meta, _result = v6.load_v6_full_checkpoint_into_model(
        model,
        str(round_base),
        expected_kind="round_base",
        strict=True,
    )
    return model


def _build_trainer(
    *,
    round_base: Path,
    round_base_checkpoint_id: str,
    output_dir: Path,
    stop_at_two: bool,
):
    device = torch.device("cuda:0")
    model = _load_round_base(round_base).to(device=device, dtype=torch.bfloat16)
    model.layer.packed_vq_decoder_linear = True
    selected = [("layer", model.layer)]
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(),
        compressed_modules=selected,
        dense_target_modules=(),
        rank=12,
        alpha=24.0,
        dropout=0.03,
        train_decoder=True,
        train_lora=False,
        decoder_execution_mode="decoder_sparse_bit",
        freeze=True,
    )
    model = selection.peft_model or model
    manager = SparseBitTuningManager(
        root_model=model,
        targets=selected,
        target_devices={"layer": device},
        training_seed=123,
        config=SparseBitTuningConfig(
            enabled=True,
            active_ratio=0.5,
            optimizer="adam",
            bit_lr=0.02,
            round_steps=5,
        ),
        streaming=False,
    )
    callbacks = [_StopAtStepTwo()] if stop_at_two else []
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        max_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        save_strategy="steps",
        save_steps=2,
        save_safetensors=False,
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        seed=123,
        data_seed=123,
        dataloader_num_workers=0,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        bf16=True,
    )
    # This is intentionally a single-process/single-GPU exact-resume fixture.
    # The production guard against torch.nn.DataParallel must remain enabled.
    args._n_gpu = 1
    trainer = _SparseMSETrainer(
        model=model,
        args=args,
        train_dataset=_TinyDataset(),
        loss_type="sft",
        sparse_bit_manager=manager,
        callbacks=callbacks,
    )
    attach_model_level_optimizer_contract(
        trainer,
        selection=selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            decoder_lr=1e-3,
        ),
    )
    trainer.configure_v6_step_checkpoint(
        context=_context(round_base, round_base_checkpoint_id),
        selected_vae_modules=selected,
    )
    return trainer, model, manager


def _tensor_tree_equal(lhs, rhs) -> None:
    if torch.is_tensor(lhs) or torch.is_tensor(rhs):
        assert torch.is_tensor(lhs) and torch.is_tensor(rhs)
        assert torch.equal(lhs.detach().cpu(), rhs.detach().cpu())
        return
    if isinstance(lhs, dict) or isinstance(rhs, dict):
        assert isinstance(lhs, dict) and isinstance(rhs, dict)
        assert set(lhs) == set(rhs)
        for key in lhs:
            _tensor_tree_equal(lhs[key], rhs[key])
        return
    if isinstance(lhs, (list, tuple)) or isinstance(rhs, (list, tuple)):
        assert isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple))
        assert len(lhs) == len(rhs)
        for left_item, right_item in zip(lhs, rhs):
            _tensor_tree_equal(left_item, right_item)
        return
    assert lhs == rhs


def _decoder_state(model: _TinyModel) -> dict:
    decoder = model.layer.get_stage_part_decoder(0, 0)
    return {name: tensor.detach().cpu().clone() for name, tensor in decoder.state_dict().items()}


def test_decoder_sparse_bit_interrupted_resume_matches_uninterrupted_exactly(tmp_path: Path):
    torch.manual_seed(77)
    round_base_model = _TinyModel()
    round_base = tmp_path / "round_base"
    saved = v6.save_v6_full_checkpoint(
        round_base_model,
        str(round_base),
        checkpoint_kind="round_base",
        compressed_targets=("layer",),
        train_mode="none",
        base_model_path="tiny-decoder-sparse",
        save_config=False,
    )
    round_base_id = str(saved["checkpoint_id"])

    continuous, continuous_model, continuous_manager = _build_trainer(
        round_base=round_base,
        round_base_checkpoint_id=round_base_id,
        output_dir=tmp_path / "continuous",
        stop_at_two=False,
    )
    continuous.train()
    assert int(continuous.state.global_step) == 4

    interrupted, _interrupted_model, _interrupted_manager = _build_trainer(
        round_base=round_base,
        round_base_checkpoint_id=round_base_id,
        output_dir=tmp_path / "interrupted",
        stop_at_two=True,
    )
    interrupted.train()
    assert int(interrupted.state.global_step) == 2
    step_dir = tmp_path / "interrupted" / "checkpoint-2"
    assert (step_dir / v6.META_FILENAME).is_file()
    assert (step_dir / v6.TRAINING_MODEL_STATE_FILENAME).is_file()

    resumed, resumed_model, resumed_manager = _build_trainer(
        round_base=round_base,
        round_base_checkpoint_id=round_base_id,
        output_dir=tmp_path / "interrupted",
        stop_at_two=False,
    )
    resumed.train(resume_from_checkpoint=str(step_dir))
    assert resumed._v6_exact_resume_loaded is True
    assert int(resumed.state.global_step) == 4

    _tensor_tree_equal(_decoder_state(continuous_model), _decoder_state(resumed_model))
    _tensor_tree_equal(continuous_manager.exact_state_dict(), resumed_manager.exact_state_dict())
    _tensor_tree_equal(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    _tensor_tree_equal(continuous.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())

    exact = resumed_manager.exact_state_dict()
    assert exact["packed_banks"]
    assert exact["score_chunks"]
    assert exact["sampler_states"]
    assert exact["global_bit_round"] == continuous_manager.global_bit_round
    assert exact["bit_round_step"] == continuous_manager.bit_round_step
    assert exact["stable_counter"] == continuous_manager.stable_counter
    assert exact["cumulative_flip_count"] == continuous_manager.cumulative_flip_count
    assert exact["had_flip"] == continuous_manager.had_flip
    assert any(
        chunk["exp_avg"] is not None and chunk["exp_avg_sq"] is not None
        for chunk in exact["bit_optimizer"]["chunks"].values()
    )


def test_sparse_bit_only_one_step_uses_no_continuous_optimizer(tmp_path: Path):
    torch.manual_seed(91)
    round_base = tmp_path / "sparse_only_round_base"
    v6.save_v6_full_checkpoint(
        _TinyModel(),
        str(round_base),
        checkpoint_kind="round_base",
        compressed_targets=("layer",),
        train_mode="none",
        base_model_path="tiny-sparse-only",
        save_config=False,
    )

    device = torch.device("cuda:0")
    model = _load_round_base(round_base).to(device=device, dtype=torch.bfloat16)
    model.layer.packed_vq_decoder_linear = True
    selected = [("layer", model.layer)]
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(),
        compressed_modules=selected,
        dense_target_modules=(),
        rank=12,
        alpha=24.0,
        dropout=0.03,
        train_decoder=False,
        train_lora=False,
        freeze=True,
    )
    assert not selection.decoder_parameters
    assert not selection.lora_parameters
    enable_vae_linear_by_execution_plan(model.layer, mode="sparse_bit")
    decoder = model.layer.get_stage_part_decoder(0, 0)
    assert all(not parameter.requires_grad for parameter in decoder.parameters())

    manager = SparseBitTuningManager(
        root_model=model,
        targets=selected,
        target_devices={"layer": device},
        training_seed=321,
        config=SparseBitTuningConfig(
            enabled=True,
            active_ratio=0.5,
            optimizer="adam",
            bit_lr=0.02,
            round_steps=5,
        ),
        streaming=False,
    )
    args = TrainingArguments(
        output_dir=str(tmp_path / "sparse_only"),
        per_device_train_batch_size=1,
        max_steps=1,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        seed=321,
        data_seed=321,
        bf16=True,
    )
    args._n_gpu = 1
    trainer = _SparseMSETrainer(
        model=model,
        args=args,
        train_dataset=_TinyDataset(),
        loss_type="sft",
        sparse_bit_manager=manager,
    )
    trainer.train()

    assert trainer._sparse_bit_main_optimizer is None
    assert int(trainer.state.global_step) == 1
    assert int(manager.bit_round_step) == 1
    exact = manager.exact_state_dict()
    assert exact["score_chunks"]
    assert any(
        chunk["exp_avg"] is not None and chunk["exp_avg_sq"] is not None
        for chunk in exact["bit_optimizer"]["chunks"].values()
    )
