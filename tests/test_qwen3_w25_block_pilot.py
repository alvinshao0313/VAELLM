from copy import deepcopy

import pytest
import torch

from experiments.qwen3_w25.block_pilot import (
    BLOCKS, STEPS, _comparison, _document_plan, _frozen_digest, _native_documents, _restore,
    select_block_steps, shard_blocks,
)
from test_qwen3_w25_local_block import _fixture


def _metrics(value, length):
    return {"per_document_mse": [value] * 32, "document_tokens": [length] * 32,
            "valid_tokens": length * 32, "token_mean_mse": value, "document_mean_mse": value}


def _blocks(ratios):
    results = {}
    for i, index in enumerate(BLOCKS):
        # Different baseline scales and token counts must not weight the four
        # blocks unequally when comparing their relative changes.
        before = _metrics(10.0 ** i, i + 1)
        results[str(index)] = {"arms": {
            str(steps): {"native_selection": _comparison(before, _metrics(before["token_mean_mse"] * ratios[steps][i], i + 1)),
                         "native_confirmation": _comparison(before, _metrics(before["token_mean_mse"] * (1.5 if steps == 64 else 0.5), i + 1))}
            for steps in STEPS}}
    return results


def test_two_fixed_shards_cover_each_block_once():
    assert shard_blocks(0) == (3, 18)
    assert shard_blocks(1) == (9, 27)
    assert set(shard_blocks(0) + shard_blocks(1)) == set(BLOCKS)
    for value in (-1, 2, True, 0.0):
        with pytest.raises(ValueError, match="shard_index"):
            shard_blocks(value)


def test_equal_block_ratio_selection_and_independent_confirmation():
    blocks = _blocks({64: [0.4, 0.4, 0.4, 1.4], 192: [0.7] * 4})
    result = select_block_steps(blocks)
    assert result["selected_steps"] == 64
    assert result["selection_scores"]["64"]["mean_selection_ratio"] == pytest.approx(0.65)
    assert result["selected_confirmation"]["mean_ratio"] == pytest.approx(1.5)
    assert not result["selected_confirmation"]["used_for_selection"]
    for block in blocks.values():
        block["arms"]["64"]["native_confirmation"] = _comparison(_metrics(1.0, 1), _metrics(50.0, 1))
    assert select_block_steps(blocks)["selected_steps"] == 64
    assert select_block_steps(_blocks({64: [0.5] * 4, 192: [0.5] * 4}))["selected_steps"] == 64


def test_missing_arm_or_block_and_bad_ratio_are_rejected():
    blocks = _blocks({64: [0.8] * 4, 192: [0.7] * 4})
    missing = deepcopy(blocks)
    missing.pop("27")
    with pytest.raises(ValueError, match="complete coverage"):
        select_block_steps(missing)
    missing = deepcopy(blocks)
    missing["3"]["arms"].pop("192")
    with pytest.raises(ValueError, match="both"):
        select_block_steps(missing)
    blocks["3"]["arms"]["64"]["native_selection"]["after_before_ratio"] = 0.01
    with pytest.raises(ValueError, match="ratio"):
        select_block_steps(blocks)


def test_real_qwen_native_document_metrics_and_complete_block_restore():
    block, train, _ = _fixture()
    block.eval().requires_grad_(False)
    # The pilot captures one document per batch; use the one-document fixture.
    batches = [train[1]]
    initial = deepcopy(block.state_dict())
    digest = _frozen_digest(block, set())
    baseline = _native_documents(block, batches, torch.device("cpu"))
    assert baseline["document_tokens"] == [3]
    assert baseline["token_mean_mse"] == baseline["per_document_mse"][0]
    with torch.no_grad():
        block.mlp.down_proj._parallel_stage_decoder.linear_out.conv.bias.add_(0.5)
        block.input_layernorm.weight.add_(0.125)
    assert _frozen_digest(block, set()) != digest
    assert _restore(block, initial, digest) == digest
    assert _native_documents(block, batches, torch.device("cpu")) == baseline
    assert all(torch.equal(value, initial[name]) for name, value in block.state_dict().items())


def test_document_plan_reserves_disjoint_selection_and_confirmation():
    batch = {"input_ids": torch.ones((64, 4), dtype=torch.long), "attention_mask": torch.ones((64, 4), dtype=torch.long)}
    data = {"seed": 31, "max_length": 256}
    for split, start in (("train", 0), ("heldout", 64)):
        data[split] = {"valid_tokens": 256, "documents": [
            {"row_index": index, "group_sha256": f"group{index}", "token_sha256": f"token{index}", "valid_tokens": 4}
            for index in range(start, start + 64)]}
    plan = _document_plan(batch, batch, data)
    assert plan["selection_indices"] == list(range(32))
    assert plan["confirmation_indices"] == list(range(32, 64))
    data["heldout"]["documents"][0]["group_sha256"] = "group0"
    with pytest.raises(ValueError, match="disjoint"):
        _document_plan(batch, batch, data)
