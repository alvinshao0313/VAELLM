from __future__ import annotations

import argparse
import math
from typing import Any

import torch
from transformers import AutoTokenizer

from train_utils.eval_utils import run_lm_eval

_SUBJECT_METRIC_KEYS = ("acc_norm,none", "acc,none", "acc_norm", "acc")


def build_tokenizer(checkpoint_dir: str, access_token=None):
    return AutoTokenizer.from_pretrained(
        checkpoint_dir,
        use_fast=False,
        trust_remote_code=True,
        token=access_token,
    )


def _pick_subject_metric(task_result: dict[str, Any]) -> tuple[str, float] | None:
    for key in _SUBJECT_METRIC_KEYS:
        value = task_result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return key, float(value)
    return None


def extract_subject_metrics(lm_result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_results = lm_result.get("raw_results")
    if not isinstance(raw_results, dict):
        raise ValueError("lm_result['raw_results'] must be a dict.")

    subject_metrics: list[dict[str, Any]] = []
    for subject_name in sorted(raw_results.keys()):
        if not subject_name.startswith("mmlu_"):
            continue
        task_result = raw_results[subject_name]
        if not isinstance(task_result, dict):
            raise ValueError(f"raw_results[{subject_name!r}] must be a dict.")
        picked = _pick_subject_metric(task_result)
        if picked is None:
            raise ValueError(
                f"No finite subject metric found for {subject_name!r} in {_SUBJECT_METRIC_KEYS}."
            )
        metric_key, accuracy = picked
        subject_metrics.append(
            {
                "subject_name": subject_name,
                "metric_key": metric_key,
                "accuracy": accuracy,
                "samples": int(task_result.get("samples", 0) or 0),
            }
        )
    return subject_metrics


def _validate_mmlu_aggregate(lm_result: dict[str, Any]) -> tuple[float, str]:
    task_metrics = lm_result.get("task_metrics")
    if not isinstance(task_metrics, dict):
        raise ValueError("result['task_metrics'] must be a dict.")
    if "mmlu" not in task_metrics:
        raise ValueError("result['task_metrics']['mmlu'] must exist.")

    accuracy = task_metrics["mmlu"]
    if not isinstance(accuracy, (int, float)) or not math.isfinite(float(accuracy)):
        raise ValueError("result['task_metrics']['mmlu'] must be a finite float in [0, 1].")
    accuracy = float(accuracy)
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("result['task_metrics']['mmlu'] must be a finite float in [0, 1].")

    task_metric_keys = lm_result.get("task_metric_keys")
    if not isinstance(task_metric_keys, dict):
        raise ValueError("result['task_metric_keys'] must be a dict.")
    if "mmlu" not in task_metric_keys:
        raise ValueError("result['task_metric_keys']['mmlu'] must exist.")

    return accuracy, str(task_metric_keys["mmlu"])


def evaluate_mmlu(model, tokenizer, checkpoint_dir: str, *, lm_limit=None) -> dict[str, Any]:
    lm_args = argparse.Namespace(
        tasks="mmlu",
        num_fewshot=0,
        batch_size="auto",
        lm_limit=lm_limit,
        model_path=checkpoint_dir,
        eval_log_dir=None,
        eval_run_ts=None,
        mmlu_debug_samples=0,
        mmlu_debug_log_dir=None,
        mmlu_debug_run_ts=None,
    )

    with torch.no_grad():
        result = run_lm_eval(model, tokenizer, lm_args)

    raw_results = result.get("raw_results")
    if not isinstance(raw_results, dict):
        raise ValueError("raw_results must be dict")

    accuracy, metric_key = _validate_mmlu_aggregate(result)
    subject_metrics = extract_subject_metrics(result)
    n_samples_total = sum(int(row["samples"]) for row in subject_metrics)

    return {
        "accuracy": accuracy,
        "metric_key": metric_key,
        "raw_results": raw_results,
        "n_samples_total": n_samples_total,
    }
