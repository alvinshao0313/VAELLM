from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.qwen3_w25.protected_bit_focus import (
    CONTROL, TARGETS, focus_configuration, prefix_comparison, select_focus_recipe,
)


def test_configuration_fixes_round256_for_both_step_budgets():
    for target in TARGETS.values():
        config = focus_configuration(target)
        assert config["lr_divisor"] == config["round_steps"] == 256
        assert config["step_arms"] == [128, 256]
        assert config["early_stable_round_end"] and config["stable_steps"] == 10
        assert config["repeated_diagnostic_used_for_selection"] is False
    with pytest.raises(ValueError, match="only L9/gate"):
        focus_configuration("model.layers.18.self_attn.o_proj")


def test_prefix_comparison_reports_differences_without_claiming_exactness():
    short = {"training_loss": [0.5] * 128,
             "step_telemetry": [{"step": i + 1, "step_flip_count": i, "round_ended": False} for i in range(128)]}
    long = deepcopy(short)
    long["training_loss"] += [0.25] * 128
    long["step_telemetry"] += [{"step": i + 1, "step_flip_count": i, "round_ended": i == 255} for i in range(128, 256)]
    result = prefix_comparison(short, long)
    assert result["loss_exact"] and result["all_step_telemetry_exact"]
    assert not result["long_prefix_round_end_steps"]
    long["training_loss"][37] += 0.125
    result = prefix_comparison(short, long)
    assert not result["loss_exact"]
    assert result["max_absolute_loss_difference"] == 0.125
    assert result["max_relative_loss_difference"] == 0.25
    with pytest.raises(ValueError, match="complete 128/256"):
        prefix_comparison(short, short)


def test_completed_real_control_retains_exact_selection_and_diagnostic():
    pilot = json.loads(Path("result/qwen3_8b_w25_20260930/local_bits_recipe_protected/pilot.json").read_text())
    control = pilot["targets"][TARGETS["down"]]["arms"]["128"]
    arms = {CONTROL: deepcopy(control), "s256_steps128_round256": deepcopy(control),
            "s256_steps256_round256": deepcopy(control)}
    arms["s256_steps128_round256"]["local_metrics"]["settings"]["lr"] *= 0.5
    arms["s256_steps256_round256"]["local_metrics"]["settings"].update(steps=256, lr=control["lr"] * 0.5)
    result = select_focus_recipe(arms, pilot["document_plan"]["heldout_lengths"])
    assert result["selected_arm"] == "s256_steps128_round256"
    assert result["selection_scores"][CONTROL] == control["native_selection"]["after_before_ratio"]
    assert result["selected_repeated_diagnostic"]["new_blind_confirmation"] is False
    with pytest.raises(ValueError, match="both completed"):
        select_focus_recipe({CONTROL: control}, pilot["document_plan"]["heldout_lengths"])
