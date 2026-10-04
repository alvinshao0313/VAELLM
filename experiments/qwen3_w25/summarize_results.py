"""Validate two full cat_eval shards for the Qwen3-8B W<=2.5, no-LoRA goal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import validate_v6_meta


TASK_METRICS = {
    "boolq": "acc,none", "rte": "acc,none", "winogrande": "acc,none",
    "arc_easy": "acc_norm,none", "arc_challenge": "acc_norm,none",
    "openbookqa": "acc_norm,none", "piqa": "acc_norm,none", "mmlu": "acc,none",
}
SAMPLE_COUNTS = {
    "boolq": 3270, "rte": 277, "winogrande": 1267, "arc_easy": 2376,
    "arc_challenge": 1172, "openbookqa": 500, "piqa": 1838,
}
# Full test sizes verified against the local hails/mmlu_no_train dataset metadata.
MMLU_SAMPLE_COUNTS = {"mmlu_" + name: count for name, count in {
    "abstract_algebra": 100, "anatomy": 135, "astronomy": 152, "business_ethics": 100,
    "clinical_knowledge": 265, "college_biology": 144, "college_chemistry": 100,
    "college_computer_science": 100, "college_mathematics": 100, "college_medicine": 173,
    "college_physics": 102, "computer_security": 100, "conceptual_physics": 235,
    "econometrics": 114, "electrical_engineering": 145, "elementary_mathematics": 378,
    "formal_logic": 126, "global_facts": 100, "high_school_biology": 310,
    "high_school_chemistry": 203, "high_school_computer_science": 100,
    "high_school_european_history": 165, "high_school_geography": 198,
    "high_school_government_and_politics": 193, "high_school_macroeconomics": 390,
    "high_school_mathematics": 270, "high_school_microeconomics": 238,
    "high_school_physics": 151, "high_school_psychology": 545, "high_school_statistics": 216,
    "high_school_us_history": 204, "high_school_world_history": 237, "human_aging": 223,
    "human_sexuality": 131, "international_law": 121, "jurisprudence": 108,
    "logical_fallacies": 163, "machine_learning": 112, "management": 103, "marketing": 234,
    "medical_genetics": 100, "miscellaneous": 783, "moral_disputes": 346, "moral_scenarios": 895,
    "nutrition": 306, "philosophy": 311, "prehistory": 324, "professional_accounting": 282,
    "professional_law": 1534, "professional_medicine": 272, "professional_psychology": 612,
    "public_relations": 110, "security_studies": 245, "sociology": 201,
    "us_foreign_policy": 100, "virology": 166, "world_religions": 171,
}.items()}
MMLU_TASKS = set(MMLU_SAMPLE_COUNTS)
DATASET_CONFIGS = {
    "boolq": ("super_glue", "boolq", "validation"), "rte": ("glue", "rte", "validation"),
    "winogrande": ("winogrande", "winogrande_xl", "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "openbookqa": ("openbookqa", "main", "test"), "piqa": ("piqa", None, "validation"),
}
PROJECTION_SHAPES = {
    "self_attn.q_proj": (4096, 4096), "self_attn.k_proj": (1024, 4096),
    "self_attn.v_proj": (1024, 4096), "self_attn.o_proj": (4096, 4096),
    "mlp.gate_proj": (12288, 4096), "mlp.up_proj": (12288, 4096),
    "mlp.down_proj": (4096, 12288),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected JSON object: {path}")
    return payload


def _score(value, label):
    _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
             f"Missing, non-finite or out-of-range accuracy: {label}={value!r}")
    return float(value)


def _validate_checkpoint(checkpoint_dir):
    meta = validate_v6_meta(_read_json(checkpoint_dir / "checkpoint_meta.json"),
                            expected_kind="final_model")
    expected = {f"model.layers.{layer}.{suffix}": shape
                for layer in range(36) for suffix, shape in PROJECTION_SHAPES.items()}
    _require(set(meta["compressed_targets"]) == set(expected), "Expected exactly all 252 Qwen3-8B projections.")
    _require(not meta["pending_dense_targets"] and not meta["skip_targets"], "Pending/skip projections are forbidden.")
    _require((meta.get("extra_meta") or {}).get("residual_lora") is None, "Residual LoRA is forbidden.")
    for spec in meta["converted_modules"]:
        name = spec["name"]
        _require((spec["out_features"], spec["in_features"]) == expected[name], f"Wrong Qwen3-8B shape: {name}")
        _require(spec.get("low_rank_a") is None and spec.get("low_rank_b") is None, f"LoRA is forbidden: {name}")
        _require(not spec.get("always_use_original") and not spec.get("has_original_weight"),
                 f"Original dense weight is present/enabled: {name}")
        dims = spec.get("stage_codebook_dims") or [spec["codebook_dim"]]
        _require(all(type(dim) is int and dim > 0 for dim in dims), f"Invalid codebook dimensions: {name}")
        for axis in ("input", "output"):
            indices = spec.get(f"protected_{axis}_indices")
            if indices is not None:
                count = math.prod(indices["shape"])
                _require(all(count % dim == 0 for dim in dims), f"Protected channels are not codebook aligned: {name}")
    audit = audit_checkpoint(checkpoint_dir, max_bpw=2.5, max_lora_rank=0)
    _require(audit["compressed_target_count"] == 252 and audit["original_linear_weights"] == 6945767424,
             "Checkpoint tensor inventory does not cover all Qwen3-8B projections.")
    _require(all(audit["components"].get(key, {}).get("storage_bits", 0) == 0
                 for key in ("lora", "residual_lora")), "Checkpoint tensors contain LoRA.")
    return meta, audit


def summarize(checkpoint_dir, result_paths):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    meta, audit = _validate_checkpoint(checkpoint_dir)
    scores, sample_counts, subject_scores, sources = {}, {}, {}, []
    base_model_path = None
    for path in result_paths:
        path = Path(path).resolve()
        payload = _read_json(path)
        _require(Path(payload["checkpoint_dir"]).resolve() == checkpoint_dir
                 and payload.get("checkpoint_id") == meta["checkpoint_id"], f"Checkpoint identity mismatch: {path}")
        _require(payload.get("checkpoint_loader") == "v6" and payload.get("strict_load") is True
                 and payload.get("adapter_dir") is None, f"Expected independent strict v6 load without adapter: {path}")
        _require(payload.get("eval_hif4_act") is False, f"Expected weight-only evaluation: {path}")
        _require(payload.get("tokenizer_class") == "Qwen2TokenizerFast" and payload.get("tokenizer_is_fast") is True,
                 f"Tokenizer must match the fast Qwen3 teacher reference: {path}")
        current_base = payload.get("base_model_path")
        _require(isinstance(current_base, str) and bool(current_base), f"Missing base_model_path: {path}")
        if base_model_path is None:
            base_model_path = current_base
        _require(current_base == base_model_path, "Result shards use different base model/tokenizer paths.")
        result = payload["evals"]["lm_eval"]
        tasks = result["tasks"]
        _require(isinstance(tasks, list) and tasks and len(tasks) == len(set(tasks)), f"Empty/duplicate tasks: {path}")
        _require(set(tasks) <= set(TASK_METRICS) and not set(tasks) & set(scores), f"Unexpected/overlapping task shards: {path}")
        _require(result.get("num_fewshot") == 0 and "limit" in result and result["limit"] is None,
                 f"Expected full zero-shot evaluation: {path}")
        _require(set(result["task_metrics"]) == set(tasks) and set(result["task_metric_keys"]) == set(tasks),
                 f"Task summaries do not exactly cover the declared shard: {path}")
        leaves = (set(tasks) - {"mmlu"}) | (MMLU_TASKS if "mmlu" in tasks else set())
        _require(set(result["n_samples"]) == leaves, f"Missing/extra leaf tasks (including all 57 MMLU subjects): {path}")
        for task in sorted(leaves):
            counts = result["n_samples"][task]
            count = counts["original"]
            _require(type(count) is int and count > 0 and counts["effective"] == count,
                     f"Incomplete sample coverage: {task}")
            expected_count = MMLU_SAMPLE_COUNTS[task] if task in MMLU_TASKS else SAMPLE_COUNTS[task]
            _require(count == expected_count, f"Unexpected standard sample count: {task}={count}, expected {expected_count}")
            _require(task in result["n_shot"] and result["n_shot"][task] == 0, f"Nonzero/missing few-shot count: {task}")
            config = result["configs"][task]
            _require(config.get("output_type") == "multiple_choice", f"Wrong task output type: {task}")
            expected_config = (("hails/mmlu_no_train", task.removeprefix("mmlu_"), "test")
                               if task in MMLU_TASKS else DATASET_CONFIGS[task])
            actual_config = (config.get("dataset_path"), config.get("dataset_name"),
                             config.get("test_split") or config.get("validation_split"))
            _require(actual_config == expected_config, f"Wrong evaluation dataset/config/split: {task}={actual_config}")
            metric_key = "acc,none" if task in MMLU_TASKS else TASK_METRICS[task]
            value = _score(result["raw_results"][task].get(metric_key), f"{task}/{metric_key}")
            sample_counts[task] = count
            if task in MMLU_TASKS:
                subject_scores[task] = value
        for task in tasks:
            key = TASK_METRICS[task]
            _require(result["task_metric_keys"][task] == key, f"Wrong metric key for {task}: expected {key}")
            value = _score(result["task_metrics"].get(task), f"summary/{task}")
            if task == "mmlu":
                groups = [group[task] for group in (result.get("group_results", {}), result["raw_results"]) if task in group]
                _require(groups, "MMLU group accuracy is missing; refusing a fallback over available subjects.")
                total = sum(sample_counts[name] for name in MMLU_TASKS)
                _require(total == 14042, f"Unexpected MMLU test sample count: {total}")
                raw_value = math.fsum(subject_scores[name] * sample_counts[name] for name in MMLU_TASKS) / total
                for group in groups:
                    _require(math.isclose(_score(group.get(key), "mmlu/group"), raw_value, rel_tol=0, abs_tol=1e-12),
                             "MMLU group differs from the sample-weighted mean of all 57 subjects.")
            else:
                raw_value = _score(result["raw_results"][task].get(key), f"raw/{task}")
            _require(math.isclose(value, raw_value, rel_tol=0, abs_tol=1e-12), f"Summary/raw metric mismatch: {task}")
            scores[task] = raw_value
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tasks": tasks})
    _require(set(scores) == set(TASK_METRICS), f"Missing required tasks: {sorted(set(TASK_METRICS) - set(scores))}")
    mean_percent = math.fsum(scores[task] for task in TASK_METRICS) / 8 * 100
    accuracy_passed = mean_percent >= 69.0
    passed = accuracy_passed and audit["budget_passed"]
    return {
        "status": "passed" if passed else "not_met", "passed": passed,
        "checkpoint_dir": str(checkpoint_dir), "checkpoint_id": meta["checkpoint_id"],
        "base_model_path": base_model_path, "tokenizer_class": "Qwen2TokenizerFast",
        "protocol": {"num_fewshot": 0, "limit": None, "lora_rank": 0,
                     "max_user_linear_bpw": 2.5, "minimum_mean_percent": 69.0,
                     "mmlu_aggregation": "57 subjects weighted by test sample count"},
        "task_metrics": {task: scores[task] for task in TASK_METRICS},
        "task_metric_keys": TASK_METRICS, "mean_percent": mean_percent,
        "accuracy_passed": accuracy_passed, "budget_passed": audit["budget_passed"],
        "failure_reasons": ([] if accuracy_passed else [f"Eight-task mean {mean_percent:.9f}% is below 69%."])
                           + audit["contract_violations"],
        "n_samples": sample_counts, "mmlu_subject_metrics": subject_scores,
        "bit_budget": audit, "input_results": sources,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Exit codes: 0 = passed; 1 = valid results below the accuracy/bit target; 2 = invalid input.",
    )
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs=2, required=True, metavar=("SHARD1", "SHARD2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.checkpoint_dir, args.results)
        exit_code = 0 if report["passed"] else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {"status": "invalid", "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "mean_percent", "error") if key in report}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
