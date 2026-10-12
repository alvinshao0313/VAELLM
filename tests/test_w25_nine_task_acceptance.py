"""Synthetic score math/acceptance tests; these fixtures are not benchmark results."""
from copy import deepcopy
import math

import pytest

from experiments.longbench_v1.protocol import DATASETS
from experiments.qwen3_w25.summarize_results import TASK_METRICS
from experiments.w25_eval.nine_task_acceptance import combine_scores


def inputs():
    tasks = {key: 0.7 for key in TASK_METRICS}
    rows = {key: {"dense": float(i + 10), "compressed": float(i + 10) - 7.2}
            for i, key in enumerate(DATASETS)}
    lb = {"status": "complete", "sample_count_per_model": 4750, "tasks": rows,
          "task_macro_mean": {"dense": 20.0, "compressed": 12.8},
          "category_macro_mean": {"dense": 95.0, "compressed": 1.0}}
    budget = {"user_linear_bits": 249, "original_linear_weights": 100,
              "budget_passed": True, "contract_violations": []}
    return tasks, deepcopy(tasks), lb, budget


def test_longbench_has_one_ninth_weight_and_uses_task_macro():
    result = combine_scores("qwen3", *inputs())
    assert result["baseline_nine_task_score"] == pytest.approx((8 * 70 + 20) / 9)
    assert result["mean_drop_points"] == pytest.approx(0.8)
    assert result["passed"]
    assert result["compressed_scores"]["longbench_v1"] == pytest.approx(12.8)


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_mmlu_has_no_independent_floor(model_type):
    baseline, candidate, lb, budget = inputs()
    for row in lb["tasks"].values():
        row["compressed"] = row["dense"]
    lb["task_macro_mean"]["compressed"] = 20.0
    candidate["mmlu"] = 0.64
    result = combine_scores(model_type, baseline, candidate, lb, budget)
    assert result["mean_passed"] and result["mmlu_passed"] and result["passed"]
    assert result["schema"] == "w25_nine_task_acceptance_v2"
    assert result["compressed_scores"]["mmlu"] == 64.0
    assert result["protocol"]["minimum_mmlu_percent"] is None
    assert result["protocol"]["mmlu_hard_gate_applicable"] is False
    assert result["protocol"]["mmlu_preference"]
    assert not result["failure_reasons"]


def test_mmlu_still_contributes_to_the_nine_term_drop():
    baseline, candidate, lb, budget = inputs()
    for row in lb["tasks"].values():
        row["compressed"] = row["dense"]
    lb["task_macro_mean"]["compressed"] = 20.0
    candidate["mmlu"] = 0.62
    result = combine_scores("qwen3", baseline, candidate, lb, budget)
    assert result["mean_drop_points"] == pytest.approx(8 / 9)
    assert result["mmlu_passed"] and result["budget_passed"]
    assert not result["mean_passed"] and not result["passed"]
    assert len(result["failure_reasons"]) == 1
    assert result["failure_reasons"][0].startswith("Nine-term score drop")


def test_actual_width_bound_is_strict_for_llama():
    values = inputs()
    values[-1]["user_linear_bits"] = 250
    assert combine_scores("qwen3", *values)["budget_passed"]
    assert not combine_scores("llama", *values)["budget_passed"]
    values[-1]["user_linear_bits"] = 251
    assert not combine_scores("qwen3", *values)["budget_passed"]


@pytest.mark.parametrize("change", ["partial", "task", "macro", "nonfinite", "downstream"])
def test_missing_or_inconsistent_results_cannot_pass(change):
    values = inputs()
    if change == "partial":
        values[2]["sample_count_per_model"] = 4749
    elif change == "task":
        values[2]["tasks"].pop("lcc")
    elif change == "macro":
        values[2]["task_macro_mean"]["compressed"] = 13
    elif change == "nonfinite":
        values[1]["mmlu"] = math.nan
    else:
        values[1].pop("rte")
    with pytest.raises(ValueError):
        combine_scores("qwen3", *values)


def test_small_real_excess_is_not_rounded_into_a_pass():
    values = inputs()
    values[1]["rte"] -= 0.000001
    assert not combine_scores("qwen3", *values)["mean_passed"]
