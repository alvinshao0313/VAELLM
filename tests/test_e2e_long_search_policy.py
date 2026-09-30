from __future__ import annotations

import pytest

from experiments.e2e_0920_search.search_policy import (
    best_evaluation,
    coordinate_winner,
    coordinates,
    estimate_base_additional_steps,
    select_promotions,
)


def _candidate(name, scores):
    return {"name": name, "evaluations": [
        {"step": step, "mean_percent": score} for step, score in scores
    ]}


def test_scope_does_not_enable_loss_search_without_authorization():
    optimization = coordinates("optimization", 7)
    assert len(optimization) == 6
    assert {coordinate.key for coordinate in optimization}.isdisjoint(
        {"hidden_loss_weight", "pre_mlp_hidden_loss_weight", "alpha"})
    assert [coordinate.key for coordinate in coordinates("loss_weights", 3)] == [
        "lm_head_lr", "norm_lr", "hidden_loss_weight"]
    assert len(coordinates("loss_weights", 14)) == 9
    assert estimate_base_additional_steps("optimization", 7) == 57000
    with pytest.raises(ValueError, match="scope"):
        coordinates("all", 7)


def test_common_prefix_excludes_later_scores_and_preserves_incumbent_ties():
    incumbent = _candidate("incumbent", [(1500, 67), (2500, 67), (5000, 69)])
    challenger = _candidate("challenger", [(1500, 67), (2500, 66.9), (5000, 70)])
    assert coordinate_winner([incumbent, challenger], through_step=2500) == "incumbent"
    assert coordinate_winner([incumbent, challenger]) == "challenger"
    assert best_evaluation(incumbent["evaluations"], through_step=2500)["step"] == 1500
    incomplete = _candidate("incomplete", [(1500, 80)])
    with pytest.raises(ValueError, match="common step 2500"):
        select_promotions([incumbent, incomplete])


def test_rotated_high_score_does_not_win_but_completed_endpoint_is_still_required():
    rows = [{"step": 500, "mean_percent": 72, "available": False},
            {"step": 1000, "mean_percent": 70, "available": True},
            {"step": 5000, "mean_percent": 69, "available": False}]
    assert best_evaluation(rows, through_step=5000)["step"] == 1000
    assert coordinate_winner([
        {"name": "rotated", "evaluations": rows},
        _candidate("retained", [(1000, 71), (5000, 70)]),
    ]) == "retained"
    with pytest.raises(ValueError, match="common step"):
        best_evaluation(rows[:-1], through_step=5000)
    rows[1]["available"] = False
    with pytest.raises(ValueError, match="recoverable"):
        best_evaluation(rows, through_step=5000)


def test_mandatory_promotion_and_bounded_extras_do_not_require_target_score():
    candidates = [
        _candidate("best", [(1500, 67), (2500, 67)]),
        _candidate("near", [(1500, 66.6), (2500, 66.75)]),
        _candidate("rising", [(1500, 66), (2500, 66.25)]),
        _candidate("flat", [(1500, 66.3), (2500, 66.3)]),
        _candidate("too_far", [(1500, 65), (2500, 66)]),
    ]
    assert select_promotions(candidates) == ["best"]
    assert select_promotions(candidates, extra_slots=1) == ["best", "near"]
    assert select_promotions(candidates, extra_slots=3) == ["best", "near", "rising"]
    with pytest.raises(ValueError, match="extra_slots"):
        select_promotions(candidates, extra_slots=4)


def test_rising_rule_uses_last_thousand_endpoint_instead_of_earlier_peak():
    best = _candidate("best", [(1500, 67), (2500, 67)])
    fading = _candidate("fading", [(500, 66.7), (1500, 66.6), (2500, 66.3)])
    assert select_promotions([best, fading], extra_slots=1) == ["best"]
    missing_trend = _candidate("missing", [(2500, 66.5)])
    with pytest.raises(ValueError, match="trend evaluation"):
        select_promotions([best, missing_trend], extra_slots=1)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 101])
def test_invalid_metrics_cannot_win_or_silently_drop_out(score):
    with pytest.raises(ValueError, match="mean_percent"):
        select_promotions([_candidate("invalid", [(2500, score)])])
