"""Task 7 optimizer invariant tests — inventory -> LR/WD mapping."""

from __future__ import annotations

import tempfile

import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from train_utils.model_level_optimizer import (
    GROUP_DECODER,
    GROUP_LM_HEAD,
    GROUP_LORA,
    GROUP_NORM,
    ModelLevelOptimizerLRConfig,
    attach_model_level_optimizer_contract,
    build_model_level_param_groups,
    create_model_level_optimizer,
    selection_from_component_parameters,
)
from train_utils.model_level_trainables import ModelLevelTrainableSelection


class _TinyTrainable(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_lora = nn.Linear(4, 4, bias=False)
        self.decoder = nn.Linear(4, 4, bias=True)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 4, bias=False)
        self.frozen = nn.Linear(4, 4, bias=False)
        self.frozen.weight.requires_grad_(False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        del labels, kwargs
        x = torch.zeros((1, 4), dtype=self.decoder.weight.dtype)
        if input_ids is not None and torch.is_tensor(input_ids):
            x = x.to(device=input_ids.device, dtype=self.decoder.weight.dtype)
        y = self.lm_head(self.norm(self.decoder(self.backbone_lora(x))))
        return {"loss": y.float().sum() * 0.0, "logits": y.unsqueeze(0)}


def _selection_for(model: _TinyTrainable) -> ModelLevelTrainableSelection:
    return selection_from_component_parameters(
        lora_parameters={"lora::backbone": model.backbone_lora.weight},
        decoder_parameters={
            "decoder::w": model.decoder.weight,
            "decoder::b": model.decoder.bias,
        },
        norm_parameters={"norm::w": model.norm.weight, "norm::b": model.norm.bias},
        lm_head_parameters={"lm_head::w": model.lm_head.weight},
    )


def _group_map(groups):
    return {str(g["group_name"]): g for g in groups}


def test_four_inventory_lr_wd_mapping_exact():
    model = _TinyTrainable()
    selection = _selection_for(model)
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.05,
            decoder_lr=2e-4,
            norm_lr=3e-4,
            lm_head_lr=4e-4,
        ),
        model=model,
    )
    by_name = _group_map(groups)
    assert set(by_name) == {GROUP_LORA, GROUP_DECODER, GROUP_NORM, GROUP_LM_HEAD}
    assert by_name[GROUP_LORA]["lr"] == pytest.approx(1e-3)
    assert by_name[GROUP_LORA]["weight_decay"] == pytest.approx(0.05)
    assert by_name[GROUP_DECODER]["lr"] == pytest.approx(2e-4)
    assert by_name[GROUP_DECODER]["weight_decay"] == pytest.approx(0.0)
    assert by_name[GROUP_NORM]["lr"] == pytest.approx(3e-4)
    assert by_name[GROUP_NORM]["weight_decay"] == pytest.approx(0.0)
    assert by_name[GROUP_LM_HEAD]["lr"] == pytest.approx(4e-4)
    assert by_name[GROUP_LM_HEAD]["weight_decay"] == pytest.approx(0.0)


def test_decoder_lr_none_falls_back_to_main_lr():
    model = _TinyTrainable()
    selection = _selection_for(model)
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1.5e-3,
            weight_decay=0.0,
            decoder_lr=None,
        ),
        model=model,
    )
    assert _group_map(groups)[GROUP_DECODER]["lr"] == pytest.approx(1.5e-3)


def test_norm_and_lm_head_lr_none_fall_back_to_main_lr():
    model = _TinyTrainable()
    selection = _selection_for(model)
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=9e-4,
            weight_decay=0.01,
            norm_lr=None,
            lm_head_lr=None,
        ),
        model=model,
    )
    by_name = _group_map(groups)
    assert by_name[GROUP_NORM]["lr"] == pytest.approx(9e-4)
    assert by_name[GROUP_LM_HEAD]["lr"] == pytest.approx(9e-4)


def test_same_adapter_backbone_lora_and_lm_head_lora_different_lr():
    model = _TinyTrainable()
    selection = selection_from_component_parameters(
        lora_parameters={"lora::backbone": model.backbone_lora.weight},
        lm_head_parameters={"lm_head::lora": model.lm_head.weight},
        decoder_parameters={},
        norm_parameters={},
    )
    # Freeze unused trainables so coverage stays exact.
    model.decoder.weight.requires_grad_(False)
    model.decoder.bias.requires_grad_(False)
    model.norm.weight.requires_grad_(False)
    model.norm.bias.requires_grad_(False)
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(
            learning_rate=1e-3,
            weight_decay=0.1,
            lm_head_lr=5e-5,
        ),
        model=model,
    )
    by_name = _group_map(groups)
    assert by_name[GROUP_LORA]["lr"] == pytest.approx(1e-3)
    assert by_name[GROUP_LM_HEAD]["lr"] == pytest.approx(5e-5)
    assert by_name[GROUP_LORA]["weight_decay"] == pytest.approx(0.1)
    assert by_name[GROUP_LM_HEAD]["weight_decay"] == pytest.approx(0.0)


def test_duplicate_parameter_id_across_inventories_hard_error():
    model = _TinyTrainable()
    shared = model.backbone_lora.weight
    with pytest.raises(RuntimeError, match="conflict across component inventories"):
        selection_from_component_parameters(
            lora_parameters={"lora": shared},
            lm_head_parameters={"lm_head": shared},
        )


def test_train_mode_none_with_aux_only_builds_optimizer():
    model = _TinyTrainable()
    for p in model.parameters():
        p.requires_grad_(False)
    model.norm.weight.requires_grad_(True)
    model.norm.bias.requires_grad_(True)
    model.lm_head.weight.requires_grad_(True)
    selection = selection_from_component_parameters(
        norm_parameters={"norm::w": model.norm.weight, "norm::b": model.norm.bias},
        lm_head_parameters={"lm_head::w": model.lm_head.weight},
    )
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-3, weight_decay=0.2),
        model=model,
    )
    by_name = _group_map(groups)
    assert set(by_name) == {GROUP_NORM, GROUP_LM_HEAD}
    assert by_name[GROUP_NORM]["weight_decay"] == pytest.approx(0.0)
    assert by_name[GROUP_LM_HEAD]["weight_decay"] == pytest.approx(0.0)


def test_main_plus_aux_combination_no_missing_or_duplicate():
    model = _TinyTrainable()
    selection = _selection_for(model)
    groups = build_model_level_param_groups(
        selection,
        lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-3, weight_decay=0.01, decoder_lr=1e-4),
        model=model,
    )
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    expected = {
        id(model.backbone_lora.weight),
        id(model.decoder.weight),
        id(model.decoder.bias),
        id(model.norm.weight),
        id(model.norm.bias),
        id(model.lm_head.weight),
    }
    assert set(ids) == expected
    assert id(model.frozen.weight) not in set(ids)


def test_optimizer_params_all_requires_grad_and_frozen_excluded():
    model = _TinyTrainable()
    selection = _selection_for(model)
    # Inject frozen into inventory must hard-error.
    bad = selection_from_component_parameters(
        lora_parameters={"frozen": model.frozen.weight},
    )
    with pytest.raises(RuntimeError, match="Frozen parameter"):
        build_model_level_param_groups(
            bad,
            lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-3, weight_decay=0.0),
        )


def test_missing_trainable_hard_error():
    model = _TinyTrainable()
    selection = selection_from_component_parameters(
        lora_parameters={"lora": model.backbone_lora.weight},
    )
    with pytest.raises(RuntimeError, match="missing from model-level optimizer"):
        build_model_level_param_groups(
            selection,
            lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-3, weight_decay=0.0),
            model=model,
        )


def test_trainer_create_model_level_optimizer_uses_inventories():
    from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer

    with tempfile.TemporaryDirectory() as tmp:
        model = _TinyTrainable()
        args = TrainingArguments(
            output_dir=tmp,
            use_cpu=True,
            learning_rate=1e-3,
            weight_decay=0.05,
            report_to=[],
        )
        trainer = VAEDecoderE2ETrainer(model=model, args=args, loss_type="sft")
        selection = _selection_for(model)
        attach_model_level_optimizer_contract(
            trainer,
            selection=selection,
            lr_config=ModelLevelOptimizerLRConfig(
                learning_rate=1e-3,
                weight_decay=0.05,
                decoder_lr=2e-4,
                norm_lr=3e-4,
                lm_head_lr=4e-4,
            ),
        )
        optimizer = create_model_level_optimizer(trainer)
        by_name = _group_map(optimizer.param_groups)
        assert by_name[GROUP_DECODER]["lr"] == pytest.approx(2e-4)
        assert by_name[GROUP_DECODER]["weight_decay"] == pytest.approx(0.0)
        assert by_name[GROUP_LORA]["weight_decay"] == pytest.approx(0.05)
        assert by_name[GROUP_LM_HEAD]["lr"] == pytest.approx(4e-4)
