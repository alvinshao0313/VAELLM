from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from experiments.down_layer_sensitivity import mmlu_eval


def test_build_tokenizer_calls_auto_tokenizer_from_checkpoint():
    with patch.object(mmlu_eval, "AutoTokenizer") as mock_auto_tokenizer:
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        mmlu_eval.build_tokenizer("/ckpt")
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "/ckpt",
            use_fast=False,
            trust_remote_code=True,
            token=None,
        )


def test_subject_metric_priority_prefers_acc_norm_none():
    lm_result = {
        "raw_results": {
            "mmlu_test_subject": {
                "acc_norm,none": 0.5,
                "acc,none": 0.3,
                "samples": 10,
            }
        }
    }
    rows = mmlu_eval.extract_subject_metrics(lm_result)
    assert len(rows) == 1
    assert rows[0]["metric_key"] == "acc_norm,none"
    assert rows[0]["accuracy"] == 0.5


def test_subject_rows_are_sorted_by_subject_name():
    lm_result = {
        "raw_results": {
            "mmlu_zebra": {"acc,none": 0.1, "samples": 1},
            "mmlu_alpha": {"acc,none": 0.2, "samples": 2},
        }
    }
    rows = mmlu_eval.extract_subject_metrics(lm_result)
    assert [row["subject_name"] for row in rows] == ["mmlu_alpha", "mmlu_zebra"]


def test_n_samples_total_equals_sum_of_subject_samples():
    mock_result = {
        "task_metrics": {"mmlu": 0.15},
        "task_metric_keys": {"mmlu": "acc,none"},
        "raw_results": {
            "mmlu_a": {"acc,none": 0.1, "samples": 3},
            "mmlu_b": {"acc,none": 0.2, "samples": 7},
        },
    }
    with patch.object(mmlu_eval, "run_lm_eval", return_value=mock_result):
        result = mmlu_eval.evaluate_mmlu(MagicMock(), MagicMock(), "/ckpt", lm_limit=2)
    assert result["n_samples_total"] == 10


@pytest.mark.parametrize(
    "mock_result",
    [
        {"task_metrics": {}, "task_metric_keys": {"mmlu": "acc,none"}, "raw_results": {}},
        {
            "task_metrics": {"mmlu": float("nan")},
            "task_metric_keys": {"mmlu": "acc,none"},
            "raw_results": {},
        },
        {
            "task_metrics": {"mmlu": 1.5},
            "task_metric_keys": {"mmlu": "acc,none"},
            "raw_results": {},
        },
        {
            "task_metrics": {"mmlu": 0.5},
            "task_metric_keys": {},
            "raw_results": {"mmlu_a": {"acc,none": 0.5, "samples": 1}},
        },
    ],
)
def test_non_finite_or_missing_aggregate_mmlu_accuracy_raises(mock_result):
    with patch.object(mmlu_eval, "run_lm_eval", return_value=mock_result):
        with pytest.raises(ValueError):
            mmlu_eval.evaluate_mmlu(MagicMock(), MagicMock(), "/ckpt")


def test_evaluate_mmlu_passes_fixed_lm_eval_args():
    mock_result = {
        "task_metrics": {"mmlu": 0.5},
        "task_metric_keys": {"mmlu": "acc,none"},
        "raw_results": {"mmlu_x": {"acc,none": 0.5, "samples": 1}},
    }
    with patch.object(mmlu_eval, "run_lm_eval", return_value=mock_result) as mock_run_lm_eval:
        mmlu_eval.evaluate_mmlu(MagicMock(), MagicMock(), "/ckpt/path", lm_limit=2)

    mock_run_lm_eval.assert_called_once()
    lm_args = mock_run_lm_eval.call_args[0][2]
    assert lm_args.tasks == "mmlu"
    assert lm_args.num_fewshot == 0
    assert lm_args.batch_size == "auto"
    assert lm_args.lm_limit == 2
    assert lm_args.model_path == "/ckpt/path"
    assert lm_args.eval_log_dir is None
    assert lm_args.eval_run_ts is None
    assert lm_args.mmlu_debug_samples == 0
    assert lm_args.mmlu_debug_log_dir is None
    assert lm_args.mmlu_debug_run_ts is None
