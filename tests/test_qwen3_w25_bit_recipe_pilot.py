from copy import deepcopy

import pytest
import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.bit_recipe_pilot import (
    BASE_MODEL, TARGETS, configuration, document_plan, select_recipe, shard_targets,
)
from experiments.qwen3_w25.calibration import build_calibration_batches


def _comparison(ratio, lengths):
    def metrics(value):
        return {"per_document_mse": [value] * len(lengths), "token_mean_mse": value,
                "document_mean_mse": value, "valid_tokens": sum(lengths)}
    return {"before": metrics(1.0), "after": metrics(ratio), "after_before_ratio": ratio}


def test_fixed_shards_cover_all_targets_once():
    assert shard_targets(0) == (TARGETS[0], TARGETS[2])
    assert shard_targets(1) == (TARGETS[1], TARGETS[3])
    assert set(shard_targets(0)) | set(shard_targets(1)) == set(TARGETS)
    for index in (False, -1, 2, 0.0):
        with pytest.raises(ValueError, match="shard indices"):
            shard_targets(index)
    config = configuration()
    assert config["lr_divisors"] == [64, 128]
    assert config["train_documents"] == 512 and config["steps"] == config["round_steps"] == 128
    assert config["repeated_diagnostic_used_for_selection"] is False


def test_lr_selection_uses_four_equal_targets_and_ignores_repeated_diagnostics():
    lengths = [2] * 32 + [3] * 32
    results = {name: {"arms": {str(divisor): {
        "native_selection": _comparison(0.7 if divisor == 128 else 0.8, lengths[:32]),
        "native_repeated_diagnostic": _comparison(1.7 if divisor == 128 else 0.2, lengths[32:]),
    } for divisor in (64, 128)}} for name in TARGETS}
    selected = select_recipe(results, lengths)
    assert selected["selected_lr_divisor"] == 128
    assert selected["selected_repeated_diagnostic"]["mean_ratio"] == pytest.approx(1.7)
    assert selected["selected_repeated_diagnostic"]["new_blind_confirmation"] is False
    tied = deepcopy(results)
    for target in tied.values():
        target["arms"]["64"]["native_selection"] = deepcopy(target["arms"]["128"]["native_selection"])
    assert select_recipe(tied, lengths)["selected_lr_divisor"] == 128
    with pytest.raises(ValueError, match="all four"):
        select_recipe(dict(list(results.items())[:-1]), lengths)
    inconsistent = deepcopy(results)
    inconsistent[TARGETS[0]]["arms"]["128"]["native_selection"]["after"]["token_mean_mse"] = 0.3
    with pytest.raises(ValueError, match="token-weighted"):
        select_recipe(inconsistent, lengths)


def test_real_512_document_data_keeps_original_64_heldout_partition():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train, heldout, manifest = build_calibration_batches(
        tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
    plan = document_plan(train, heldout, manifest)
    assert len(plan["train"]["document_indices"]) == 512
    assert plan["train"]["valid_tokens"] == int(train["attention_mask"].sum())
    assert plan["selection"]["heldout_document_indices"] == list(range(32))
    assert plan["repeated_diagnostic"]["heldout_document_indices"] == list(range(32, 64))
    cut = plan["selection"]["valid_tokens"]
    flattened = heldout["input_ids"][heldout["attention_mask"].bool()]
    assert torch.equal(flattened[:cut], heldout["input_ids"][:32][heldout["attention_mask"][:32].bool()])
    assert torch.equal(flattened[cut:], heldout["input_ids"][32:][heldout["attention_mask"][32:].bool()])
