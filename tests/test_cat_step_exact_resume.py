from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, TrainerCallback, TrainingArguments

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as v6
from train_utils.cat_after_category_common import CanonicalCatSFTTrainer
from train_utils.config.configs import AuxTrainableConfig
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract
from train_utils.model_level_trainables import build_model_level_trainable_selection


def _tiny_tokenizer(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = output_dir / "vocab.txt"
    vocab_file.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\ntiny\ntrain\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)
    tokenizer.eos_token = tokenizer.sep_token
    return tokenizer


def _stack_collator(features):
    return {
        "x": torch.stack([item["x"] for item in features], dim=0),
        "target": torch.stack([item["target"] for item in features], dim=0),
    }


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(901)
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
    ).to(dtype=torch.float32)


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
        self.q_proj = _vae_linear()
        self.config = SimpleNamespace(
            _name_or_path="tiny-cat-step-resume",
            use_cache=False,
            model_type="tiny",
            architectures=["TinyModel"],
        )

    def forward(self, x=None, **kwargs):
        del kwargs
        return self.q_proj(x)


class _MSECatTrainer(CanonicalCatSFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        parameter = next(param for param in model.parameters() if param.requires_grad)
        x = inputs["x"].to(device=parameter.device, dtype=parameter.dtype)
        target = inputs["target"].to(device=parameter.device, dtype=parameter.dtype)
        output = model(x=x)
        loss = (output - target).float().square().mean()
        return (loss, {"output": output}) if return_outputs else loss


class _StopAtStepTwo(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        del args, kwargs
        if int(state.global_step) == 2:
            control.should_training_stop = True
        return control


def _save_round_base(path: Path, *, mode: str = "current_decoder") -> str:
    torch.manual_seed(37)
    model = _TinyModel()
    result = v6.save_v6_full_checkpoint(
        model,
        str(path),
        checkpoint_kind="round_base",
        compressed_targets=("q_proj",),
        pending_dense_targets=(),
        skip_targets=(),
        train_mode="none",
        after_category_mode=str(mode),
        completed_categories=(),
        compression_categories=("q_proj",),
        target_layers=None,
        target_modules=("q_proj",),
        base_model_path="tiny-cat-step-resume",
        save_config=False,
        extra_meta={"active_category": "q_proj"},
    )
    return str(result["checkpoint_id"])


def _load_round_base(path: Path) -> _TinyModel:
    model = _TinyModel()
    model, _meta, _result = v6.load_v6_full_checkpoint_into_model(
        model,
        str(path),
        expected_kind="round_base",
        strict=True,
    )
    return model


def _context(round_base: Path, checkpoint_id: str, *, mode: str) -> dict:
    return {
        "round_base_dir": str(round_base),
        "round_base_checkpoint_id": str(checkpoint_id),
        "active_category": "q_proj",
        "after_category_mode": str(mode),
        "compressed_targets": ("q_proj",),
        "pending_dense_targets": (),
        "skip_targets": (),
        "completed_categories": (),
        "compression_categories": ("q_proj",),
        "target_layers": None,
        "target_modules": ("q_proj",),
        "norm_train_mode": "none",
        "lm_head_train_mode": "none",
        "lora_config": (
            None
            if str(mode) == "current_decoder"
            else {
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "rank_pattern": {},
                "target_modules": ["q_proj"],
            }
        ),
        "resolved_learning_rates": {
            "learning_rate": 1e-3,
            "decoder_lr": (1e-3 if str(mode) in {"current_decoder", "current_lora_decoder"} else None),
            "norm_lr": None,
            "lm_head_lr": None,
        },
        "immutable_resume_contract": {"fixture": str(mode), "max_steps": 4},
        "base_model_path": "tiny-cat-step-resume",
        "runtime_audit": {"test": "cat_step_exact_resume"},
        "distill_stage_history": [],
        "round_idx": 0,
    }


def _build_trainer(
    *,
    round_base: Path,
    checkpoint_id: str,
    output_dir: Path,
    stop_at_two: bool,
    save_strategy: str,
    mode: str = "current_decoder",
):
    model = _load_round_base(round_base)
    torch.manual_seed(12345)
    selected = (("q_proj", model.q_proj),)
    train_decoder = str(mode) in {"current_decoder", "current_lora_decoder"}
    train_lora = str(mode) in {"current_lora", "current_lora_decoder"}
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(),
        compressed_modules=selected,
        dense_target_modules=(),
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=train_decoder,
        train_lora=train_lora,
        freeze=True,
    )
    model = selection.peft_model or model
    callbacks = [_StopAtStepTwo()] if stop_at_two else []
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        max_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        save_strategy=save_strategy,
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
    )
    args._n_gpu = 1
    trainer = _MSECatTrainer(
        model=model,
        args=args,
        train_dataset=_TinyDataset(),
        processing_class=_tiny_tokenizer(output_dir / "tokenizer"),
        data_collator=_stack_collator,
        loss_type="sft",
        teacher_output_offload="none",
        teacher_output_pin_memory=False,
        teacher_output_chunk_tokens=8,
        callbacks=callbacks,
    )
    attach_model_level_optimizer_contract(
        trainer,
        selection=selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            decoder_lr=(1e-3 if train_decoder else None),
        ),
    )
    trainer.configure_v6_step_checkpoint(
        context=_context(round_base, checkpoint_id, mode=mode),
        selected_vae_modules=(selected if train_decoder else ()),
    )
    return trainer, model


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


def _selection_state(trainer) -> dict:
    selection = trainer.model_level_trainable_selection
    out = {}
    for inventory_name in (
        "decoder_parameters",
        "lora_parameters",
        "norm_parameters",
        "lm_head_parameters",
    ):
        inventory = getattr(selection, inventory_name)
        out[inventory_name] = {
            str(name): tensor.detach().cpu().clone()
            for name, tensor in inventory.items()
        }
    return out


@pytest.mark.parametrize("mode", ["current_decoder", "current_lora", "current_lora_decoder"])
def test_cat_current_family_interrupted_resume_matches_uninterrupted_exactly(tmp_path: Path, mode: str):
    round_base = tmp_path / "round_base"
    checkpoint_id = _save_round_base(round_base, mode=mode)

    continuous_trainer, _continuous_model = _build_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "continuous",
        stop_at_two=False,
        save_strategy="no",
        mode=mode,
    )
    continuous_trainer.train()
    continuous_state = _selection_state(continuous_trainer)
    continuous_optimizer = continuous_trainer.optimizer.state_dict()
    continuous_scheduler = continuous_trainer.lr_scheduler.state_dict()

    interrupted_trainer, _interrupted_model = _build_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "interrupted",
        stop_at_two=True,
        save_strategy="steps",
        mode=mode,
    )
    interrupted_trainer.train()
    checkpoint_two = tmp_path / "interrupted" / "checkpoint-2"
    assert checkpoint_two.is_dir()
    assert (checkpoint_two / v6.TRAINING_MODEL_STATE_FILENAME).is_file()
    assert (checkpoint_two / v6.META_FILENAME).is_file()
    assert not (checkpoint_two / v6.STATE_DICT_FILENAME).exists()
    assert (checkpoint_two / "optimizer.pt").is_file()
    assert (checkpoint_two / "scheduler.pt").is_file()
    assert (checkpoint_two / "trainer_state.json").is_file()

    resumed_trainer, _resumed_model = _build_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "resumed",
        stop_at_two=False,
        save_strategy="no",
        mode=mode,
    )
    resumed_trainer.train(resume_from_checkpoint=str(checkpoint_two))

    _tensor_tree_equal(continuous_state, _selection_state(resumed_trainer))
    _tensor_tree_equal(continuous_optimizer, resumed_trainer.optimizer.state_dict())
    _tensor_tree_equal(continuous_scheduler, resumed_trainer.lr_scheduler.state_dict())
    assert int(resumed_trainer.state.global_step) == 4


class _RemainingTinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = _vae_linear()
        self.k_proj = _vae_linear()
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.config = SimpleNamespace(
            _name_or_path="tiny-cat-step-resume-remaining",
            use_cache=False,
            model_type="tiny",
            architectures=["RemainingTinyModel"],
        )

    def forward(self, x=None, **kwargs):
        del kwargs
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


def _save_remaining_round_base(path: Path, *, mode: str) -> str:
    torch.manual_seed(51)
    model = _RemainingTinyModel()
    result = v6.save_v6_full_checkpoint(
        model,
        str(path),
        checkpoint_kind="round_base",
        compressed_targets=("q_proj", "k_proj"),
        pending_dense_targets=("v_proj",),
        skip_targets=(),
        train_mode="none",
        after_category_mode=str(mode),
        completed_categories=("q_proj",),
        compression_categories=("q_proj", "k_proj", "v_proj"),
        target_layers=None,
        target_modules=("q_proj", "k_proj", "v_proj"),
        base_model_path="tiny-cat-step-resume-remaining",
        save_config=False,
        extra_meta={"active_category": "k_proj"},
    )
    return str(result["checkpoint_id"])


def _load_remaining_round_base(path: Path) -> _RemainingTinyModel:
    model = _RemainingTinyModel()
    model, _meta, _result = v6.load_v6_full_checkpoint_into_model(
        model,
        str(path),
        expected_kind="round_base",
        strict=True,
    )
    return model


def _remaining_context(round_base: Path, checkpoint_id: str, *, mode: str) -> dict:
    return {
        "round_base_dir": str(round_base),
        "round_base_checkpoint_id": str(checkpoint_id),
        "active_category": "k_proj",
        "after_category_mode": str(mode),
        "compressed_targets": ("q_proj", "k_proj"),
        "pending_dense_targets": ("v_proj",),
        "skip_targets": (),
        "completed_categories": ("q_proj",),
        "compression_categories": ("q_proj", "k_proj", "v_proj"),
        "target_layers": None,
        "target_modules": ("q_proj", "k_proj", "v_proj"),
        "norm_train_mode": "none",
        "lm_head_train_mode": "none",
        "lora_config": {
            "rank": 2,
            "alpha": 4.0,
            "dropout": 0.0,
            "rank_pattern": {},
            "target_modules": ["v_proj"],
        },
        "resolved_learning_rates": {
            "learning_rate": 1e-3,
            "decoder_lr": (None if mode == "remaining_lora" else 1e-3),
            "norm_lr": None,
            "lm_head_lr": None,
        },
        "immutable_resume_contract": {"fixture": str(mode), "max_steps": 4},
        "base_model_path": "tiny-cat-step-resume-remaining",
        "runtime_audit": {"test": "cat_step_exact_resume_remaining"},
        "distill_stage_history": [],
        "round_idx": 1,
    }


def _build_remaining_trainer(
    *,
    round_base: Path,
    checkpoint_id: str,
    output_dir: Path,
    stop_at_two: bool,
    save_strategy: str,
    mode: str,
):
    model = _load_remaining_round_base(round_base)
    torch.manual_seed(54321)
    decoder_targets = ()
    if mode == "remaining_lora_current_decoder":
        decoder_targets = (("k_proj", model.k_proj),)
    elif mode == "remaining_lora_prefix_decoder":
        decoder_targets = (("q_proj", model.q_proj), ("k_proj", model.k_proj))
    elif mode != "remaining_lora":
        raise ValueError(mode)
    selection = build_model_level_trainable_selection(
        model,
        aux=AuxTrainableConfig(),
        compressed_modules=(),
        dense_target_modules=("v_proj",),
        decoder_modules=decoder_targets,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        train_decoder=bool(decoder_targets),
        train_lora=True,
        freeze=True,
    )
    model = selection.peft_model or model
    callbacks = [_StopAtStepTwo()] if stop_at_two else []
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        max_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        save_strategy=save_strategy,
        save_steps=2,
        save_safetensors=False,
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        seed=321,
        data_seed=321,
        dataloader_num_workers=0,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
    )
    args._n_gpu = 1
    trainer = _MSECatTrainer(
        model=model,
        args=args,
        train_dataset=_TinyDataset(),
        processing_class=_tiny_tokenizer(output_dir / "tokenizer"),
        data_collator=_stack_collator,
        loss_type="sft",
        teacher_output_offload="none",
        teacher_output_pin_memory=False,
        teacher_output_chunk_tokens=8,
        callbacks=callbacks,
    )
    attach_model_level_optimizer_contract(
        trainer,
        selection=selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            decoder_lr=(1e-3 if decoder_targets else None),
        ),
    )
    trainer.configure_v6_step_checkpoint(
        context=_remaining_context(round_base, checkpoint_id, mode=mode),
        selected_vae_modules=decoder_targets,
    )
    return trainer


@pytest.mark.parametrize(
    "mode",
    [
        "remaining_lora",
        "remaining_lora_current_decoder",
        "remaining_lora_prefix_decoder",
    ],
)
def test_cat_remaining_family_interrupted_resume_matches_uninterrupted_exactly(tmp_path: Path, mode: str):
    round_base = tmp_path / "round_base"
    checkpoint_id = _save_remaining_round_base(round_base, mode=mode)
    continuous = _build_remaining_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "continuous",
        stop_at_two=False,
        save_strategy="no",
        mode=mode,
    )
    continuous.train()
    continuous_state = _selection_state(continuous)
    continuous_optimizer = continuous.optimizer.state_dict()
    continuous_scheduler = continuous.lr_scheduler.state_dict()

    interrupted = _build_remaining_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "interrupted",
        stop_at_two=True,
        save_strategy="steps",
        mode=mode,
    )
    interrupted.train()
    checkpoint_two = tmp_path / "interrupted" / "checkpoint-2"
    assert checkpoint_two.is_dir()
    assert (checkpoint_two / v6.TRAINING_MODEL_STATE_FILENAME).is_file()
    assert not (checkpoint_two / v6.STATE_DICT_FILENAME).exists()

    resumed = _build_remaining_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "resumed",
        stop_at_two=False,
        save_strategy="no",
        mode=mode,
    )
    resumed.train(resume_from_checkpoint=str(checkpoint_two))

    _tensor_tree_equal(continuous_state, _selection_state(resumed))
    _tensor_tree_equal(continuous_optimizer, resumed.optimizer.state_dict())
    _tensor_tree_equal(continuous_scheduler, resumed.lr_scheduler.state_dict())
    assert int(resumed.state.global_step) == 4


def test_cat_step_resume_rejects_immutable_contract_change(tmp_path: Path):
    round_base = tmp_path / "round_base"
    checkpoint_id = _save_round_base(round_base, mode="current_lora")
    interrupted, _model = _build_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "interrupted",
        stop_at_two=True,
        save_strategy="steps",
        mode="current_lora",
    )
    interrupted.train()
    checkpoint_two = tmp_path / "interrupted" / "checkpoint-2"
    resumed, _model = _build_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "resumed",
        stop_at_two=False,
        save_strategy="no",
        mode="current_lora",
    )
    resumed._v6_step_checkpoint_context["immutable_resume_contract"] = {
        "fixture": "changed",
        "max_steps": 4,
    }
    with pytest.raises(ValueError, match="immutable contract mismatch"):
        resumed.train(resume_from_checkpoint=str(checkpoint_two))


def test_cat_step_resume_rejects_after_category_topology_change(tmp_path: Path):
    round_base = tmp_path / "round_base"
    checkpoint_id = _save_round_base(round_base, mode="current_lora")
    interrupted, _model = _build_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "interrupted",
        stop_at_two=True,
        save_strategy="steps",
        mode="current_lora",
    )
    interrupted.train()
    checkpoint_two = tmp_path / "interrupted" / "checkpoint-2"
    resumed, _model = _build_trainer(
        round_base=round_base,
        checkpoint_id=checkpoint_id,
        output_dir=tmp_path / "resumed",
        stop_at_two=False,
        save_strategy="no",
        mode="current_lora_decoder",
    )
    with pytest.raises(ValueError, match="topology mismatch for after_category_mode"):
        resumed.train(resume_from_checkpoint=str(checkpoint_two))
