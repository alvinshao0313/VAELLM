"""CPU properties for the selected/frozen boundary and deployment projection."""

import copy

import pytest
import torch

from experiments.qwen3_w25.e2e_subset_audit import audit_subset_state
from experiments.qwen3_w25.audit_e2e_decoder import _validate_export_storage


def states():
    original, inventory, banks = {}, {}, {}
    for index in range(3):
        name = f"model.layers.{index}.mlp.down_proj"
        inventory[name] = [f"{name}._parallel_stage_decoder.leaf{n}" for n in range(6)]
        for key in inventory[name]:
            original[key] = torch.ones(3, dtype=torch.bfloat16)
        banks[name] = [(f"{name}.vq_weight{suffix}", (2, 1, 10)) for suffix in ("", "_s1")]
        for key, _ in banks[name]:
            original[key] = torch.tensor([[[3, 1]], [[129, 2]]], dtype=torch.uint8)
    original["model.norm.weight"] = torch.ones(3, dtype=torch.bfloat16)
    original["lm_head.weight"] = torch.ones((2, 3), dtype=torch.bfloat16)
    selected = ("model.layers.0.mlp.down_proj",)
    mutable = {key: original[key].float().clone() for key in inventory[selected[0]]}
    mutable[inventory[selected[0]][0]][0] += 1e-4
    packed = {key: original[key].clone() for key, _ in banks[selected[0]]}
    return original, copy.deepcopy(original), inventory, banks, selected, mutable, packed


def audit(values):
    original, actual, inventory, banks, selected, mutable, packed = values
    return audit_subset_state(original, actual, inventory, banks, selected_names=selected,
                              training_decoder_state=mutable, packed_snapshot=packed)


def test_real_fp32_update_can_round_away_with_complete_storage_accounting():
    values = states()
    report = audit(values)
    assert report["selected_decoder_fp32_changed_elements"] == 1
    assert report["changed_decoder_elements"] == 0
    assert not report["selected_decoder_update_survived_bf16"]
    assert report["selected_decoder_matches_fp32_cast_exactly"]
    assert report["frozen_tensors_compared"] == len(values[0]) - 8
    assert report["decoder_tensor_count"] == 18
    assert report["selected_decoder_tensor_count"] == 6
    assert report["code_bank_count"] == 6 and report["selected_code_bank_count"] == 2
    # The storage contract consumes all decoders, not just the trainable subset.
    storage = report["decoder_storage_source"]
    assert storage["budget_bits"] == 18 * 3 * 16
    budget = dict(budget_passed=True, original_linear_weights=123, code_padding_bits=0,
                  outside_linear_bits=96, components={"decoder": storage, "codes": {"bits": 192}},
                  user_linear_bits=1234, packed_linear_bits=1234, full_model_tensor_bytes=321)
    _validate_export_storage(budget, copy.deepcopy(budget), report)


def test_exact_projection_and_single_hard_bit_change_are_reported():
    values = states()
    original, actual, inventory, banks, selected, mutable, packed = values
    key = inventory[selected[0]][0]
    mutable[key][1] += 0.02
    actual[key] = mutable[key].bfloat16()
    code = banks[selected[0]][0][0]
    packed[code][0, 0, 0] ^= 4
    actual[code] = packed[code].clone()
    report = audit(values)
    assert report["selected_decoder_fp32_changed_elements"] == 2
    assert report["changed_decoder_elements"] == 1
    assert report["changed_code_bits"] == 1
    assert report["stage_changed_bits"][selected[0]] == [1, 0]
    assert report["packed_padding_unchanged"] and report["frozen_bytes_unchanged"]


@pytest.mark.parametrize("violation", ["frozen_decoder", "frozen_code", "head", "projection", "snapshot", "padding", "no_update", "extra_mutable"])
def test_subset_audit_rejects_contract_violations(violation):
    values = states()
    original, actual, inventory, banks, selected, mutable, packed = values
    first = inventory[selected[0]][0]
    other = "model.layers.1.mlp.down_proj"
    if violation == "frozen_decoder":
        actual[inventory[other][0]][0] += 1
    elif violation == "frozen_code":
        actual[banks[other][0][0]][0, 0, 0] ^= 1
    elif violation == "head":
        actual["lm_head.weight"][0, 0] += 1
    elif violation == "projection":
        actual[first][0] += 1
    elif violation == "snapshot":
        packed[banks[selected[0]][0][0]][0, 0, 0] ^= 1
    elif violation == "padding":
        key = banks[selected[0]][0][0]
        packed[key][0, 0, -1] ^= 128
        actual[key] = packed[key].clone()
    elif violation == "no_update":
        mutable[first] = original[first].float()
    elif violation == "extra_mutable":
        mutable[inventory[other][0]] = original[inventory[other][0]].float()
    with pytest.raises((ValueError, RuntimeError)):
        audit(values)
