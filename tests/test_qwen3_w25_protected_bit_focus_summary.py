from copy import deepcopy

import pytest

from experiments.qwen3_w25.protected_bit_focus_summary import RECIPES, TARGETS, select_unified_mlp_recipe


def _comparison(value):
    before = {"per_document_mse": [1.0] * 32, "token_mean_mse": 1.0, "document_mean_mse": 1.0, "valid_tokens": 32}
    after = {"per_document_mse": [value] * 32, "token_mean_mse": value, "document_mean_mse": value, "valid_tokens": 32}
    return {"before": before, "after": after, "after_before_ratio": value}


def test_one_mlp_recipe_uses_equal_targets_without_stitching_local_winners():
    values = ((0.4, 1.4), (0.8, 0.8), (0.9, 0.5))
    reports = {name: {"arms": {}} for name in TARGETS.values()}
    for arm, pair in zip(RECIPES, values):
        for target, ratio in zip(TARGETS.values(), pair):
            reports[target]["arms"][arm] = {"native_selection": _comparison(ratio),
                                           "native_repeated_diagnostic": _comparison(3.0 if pair == values[-1] else 0.1)}
    selected = select_unified_mlp_recipe(reports, [1] * 64)
    assert selected["selected_mlp_arm"] == "s256_steps256_round256"
    assert selected["mlp_selection_scores"]["s256_steps256_round256"]["mean_selection_ratio"] == pytest.approx(0.7)
    assert selected["selected_mlp_repeated_diagnostic"]["mean_ratio"] == pytest.approx(3.0)
    assert selected["selected_mlp_repeated_diagnostic"]["used_for_selection"] is False
    missing = deepcopy(reports)
    missing.pop(TARGETS["down"])
    with pytest.raises(ValueError, match="exactly gate/down"):
        select_unified_mlp_recipe(missing, [1] * 64)


def test_unified_tie_prefers_fewer_steps_then_lower_lr():
    reports = {name: {"arms": {arm: {"native_selection": _comparison(0.8),
                                     "native_repeated_diagnostic": _comparison(0.9)} for arm in RECIPES}}
               for name in TARGETS.values()}
    assert select_unified_mlp_recipe(reports, [1] * 64)["selected_mlp_arm"] == "s256_steps128_round256"
