"""CPU contract checks only; these synthetic tensors are not training evidence."""
from copy import deepcopy

import pytest
import torch

from experiments.long_context_recovery.contract import validate_config
from experiments.long_context_recovery.observations import batch_record
from experiments.long_context_recovery.prepare import training_arguments
from train_utils.config.cli import parse_e2e_cli


def configuration():
    args = training_arguments("/tmp/diagnostic_manifest.json", "/tmp/diagnostic_output", "longalpaca")
    return parse_e2e_cli(args[2:])


def test_existing_native_cli_preserves_schedule_and_scientific_contract():
    cfg = configuration()
    validate_config(cfg)
    assert cfg.opt.steps == 2000 and cfg.opt.warmup_ratio == .01
    assert cfg.loss.loss_type == "kl_top_mass" and cfg.data.dataset_task == "sft"


@pytest.mark.parametrize("section,key,value", [
    ("opt", "steps", 4), ("opt", "batch_size", 4), ("data", "dataset_task", "lm"),
    ("data", "model_max_length", 1024), ("runtime", "teacher_model_offload", "cpu"),
    ("loss", "loss_type", "kd_top_mass"), ("aux", "lm_head_train_mode", "full"),
])
def test_resource_shortcut_or_scientific_change_is_rejected(section, key, value):
    cfg = deepcopy(configuration())
    setattr(getattr(cfg, section), key, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_actual_effective_length_and_shifted_supervision_exclude_padding_and_specials():
    ids = torch.tensor([[10, 20, 30, 99, 0]])
    labels = torch.tensor([[-100, -100, 30, 99, -100]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    before = (ids.clone(), labels.clone(), attention.clone())
    record = batch_record(dict(input_ids=ids, labels=labels, attention_mask=attention), special_ids=[99, 0])
    assert record["effective_tokens"] == 4
    assert record["response_tokens_shifted"] == 2 and record["prompt_tokens_shifted"] == 1
    assert record["response_non_special_tokens_shifted"] == 1
    assert all(torch.equal(a, b) for a, b in zip(before, (ids, labels, attention)))
