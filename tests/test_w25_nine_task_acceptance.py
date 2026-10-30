"""Synthetic score math/acceptance tests; these fixtures are not benchmark results."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from experiments.longbench_v1.protocol import DATASETS
from experiments.qwen3_w25.summarize_results import MMLU_SAMPLE_COUNTS, SAMPLE_COUNTS, TASK_METRICS
from experiments.w25_eval.nine_task_acceptance import combine_scores
from experiments.w25_eval.transfer_nine_task_acceptance import (
    TRANSFER_GOAL, _sample_artifacts, combine_scores as combine_transfer_scores,
)


def inputs():
    tasks = {key: 0.7 for key in TASK_METRICS}
    rows = {key: {"dense": float(i + 10), "compressed": float(i + 10) - 7.2}
            for i, key in enumerate(DATASETS)}
    lb = {"status": "complete", "sample_count_per_model": 4750, "tasks": rows,
          "task_macro_mean": {"dense": 20.0, "compressed": 12.8},
          "category_macro_mean": {"dense": 95.0, "compressed": 1.0}}
    budget = {"user_linear_bits": 249, "original_linear_weights": 100,
              "packed_linear_bpw": 2.56,
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


@pytest.mark.parametrize("goal", [None, TRANSFER_GOAL])
@pytest.mark.parametrize("change", ["partial", "task", "macro", "nonfinite", "downstream"])
def test_missing_or_inconsistent_results_cannot_pass(change, goal):
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
        (combine_transfer_scores if goal else combine_scores)("qwen3", *values)


def test_small_real_excess_is_not_rounded_into_a_pass():
    values = inputs()
    values[1]["rte"] -= 0.000001
    assert not combine_scores("qwen3", *values)["mean_passed"]


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
@pytest.mark.parametrize("compressed_lb,passed", [
    (math.nextafter(11.9, math.inf), True),
    (11.9, False),
    (math.nextafter(11.9, -math.inf), False),
])
def test_transfer_goal_requires_strict_drop_without_a_tolerance_band(model_type, compressed_lb, passed):
    baseline, candidate, lb, budget = inputs()
    lb["tasks"] = {key: {"dense": 20.0, "compressed": compressed_lb} for key in DATASETS}
    lb["task_macro_mean"] = {"dense": 20.0, "compressed": compressed_lb}
    result = combine_transfer_scores(model_type, baseline, candidate, lb, budget)
    assert result["mean_passed"] is passed and result["passed"] is passed
    assert result["mean_drop_points"] == pytest.approx(0.9)
    assert result["protocol"]["strict_score_drop_bound"] is True
    assert result["protocol"]["score_boundary_roundoff_tolerance"] == 0
    assert result["protocol"]["minimum_compressed_nine_task_score_is_exclusive"] is True


def test_transfer_goal_compares_decimal_downstream_scores_before_float_percent_conversion():
    baseline, candidate, lb, budget = inputs()
    candidate = {key: 0.691 for key in TASK_METRICS}
    lb["tasks"] = {key: {"dense": 20.0, "compressed": 19.1} for key in DATASETS}
    lb["task_macro_mean"] = {"dense": 20.0, "compressed": 19.1}
    result = combine_transfer_scores("qwen3", baseline, candidate, lb, budget)
    assert result["mean_drop_points"] == 0.9
    assert not result["mean_passed"]


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_transfer_goal_has_inclusive_logical_2_6_bound_and_keeps_rank16(model_type):
    values = inputs()
    values[-1].update(user_linear_bits=260, packed_linear_bpw=2.7)
    result = combine_transfer_scores(model_type, *values)
    assert result["passed"] and result["budget_passed"]
    assert result["schema"] == "w25_transfer_nine_task_acceptance_v1"
    assert result["goal"] == TRANSFER_GOAL
    assert result["protocol"]["max_linear_bpw"] == 2.6
    assert result["protocol"]["strict_bpw_bound"] is False
    assert result["protocol"]["max_lora_rank"] == 16
    assert result["packed_bpw_reported_separately"] == 2.7
    assert result["transferability_evaluated"] is False
    values[-1]["user_linear_bits"] = 261
    assert not combine_transfer_scores(model_type, *values)["budget_passed"]
    values[-1].update(user_linear_bits=260, budget_passed=False, contract_violations=["Actual rank violation."])
    result = combine_transfer_scores(model_type, *values)
    assert not result["passed"] and not result["budget_passed"]
    assert "Actual rank violation." in result["failure_reasons"]


@pytest.mark.parametrize("change", [None, "sha", "count", "task", "total", "missing"])
def test_transfer_goal_preserves_all_downstream_raw_sample_bindings(tmp_path, change):
    paths = []
    for index, counts in enumerate((MMLU_SAMPLE_COUNTS, SAMPLE_COUNTS)):
        rows = {}
        for task, count in counts.items():
            artifact = tmp_path / (task + ".jsonl")
            artifact.write_text("Synthetic sample-binding unit fixture.\n", encoding="utf-8")
            rows[task] = {"path": str(artifact), "count": count,
                          "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        samples = {"total_count": sum(counts.values()), "tasks": rows}
        if index == 0:
            first = next(iter(rows))
            if change == "sha":
                rows[first]["sha256"] = "0" * 64
            elif change == "count":
                rows[first]["count"] -= 1
            elif change == "task":
                rows.pop(first)
            elif change == "total":
                samples["total_count"] -= 1
            elif change == "missing":
                (tmp_path / (first + ".jsonl")).unlink()
        payload = {"n_samples": {task: {"effective": count} for task, count in counts.items()},
                   "sample_artifacts": samples}
        path = tmp_path / f"shard_{index}.json"
        path.write_text(json.dumps({"evals": {"lm_eval": payload}}), encoding="utf-8")
        paths.append(path)
    if change is None:
        result = _sample_artifacts(paths)
        assert result["leaf_count"] == len(result["sha256"]) == 64
        assert result["sample_count"] == 24742
    else:
        with pytest.raises((ValueError, OSError)):
            _sample_artifacts(paths)
