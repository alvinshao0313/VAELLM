"""Paired eight-task plus complete LongBench acceptance for Qwen3/Llama3.1."""
from __future__ import annotations

import math
from pathlib import Path

from experiments.compare_longbench_v1 import comparison
from experiments.longbench_v1.common import check_hashes, check_stat
from experiments.longbench_v1.protocol import DATASETS
from experiments.qwen3_w25.summarize_results import TASK_METRICS, _require
from experiments.w25_eval.acceptance import collect_results, read_json
from experiments.w25_eval.model_inventory import model_inventory, sha256
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import validate_v6_meta


def combine_scores(model_type, baseline_tasks, compressed_tasks, longbench, budget):
    """Use LongBench's 21-task macro as exactly one of nine equally weighted terms."""
    _require(model_type in ("qwen3", "llama"), "Unsupported acceptance architecture.")
    _require(set(baseline_tasks) == set(compressed_tasks) == set(TASK_METRICS),
             "All eight downstream tasks are required.")
    _require(longbench.get("status") == "complete" and longbench.get("sample_count_per_model") == 4750
             and set(longbench["tasks"]) == set(DATASETS), "Complete paired LongBench results are required.")
    for scores in (baseline_tasks, compressed_tasks):
        _require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                     for v in scores.values()), "Downstream metrics must be finite fractions.")
    lb_scores = {}
    for side in ("dense", "compressed"):
        values = [row[side] for row in longbench["tasks"].values()]
        _require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 100 for v in values),
                 "LongBench metrics must be finite 0–100 scores.")
        lb_scores[side] = math.fsum(values) / 21
        _require(math.isclose(lb_scores[side], longbench["task_macro_mean"][side],
                             rel_tol=0, abs_tol=1e-10), "LongBench task macro does not match its 21 tasks.")
    dense = {key: baseline_tasks[key] * 100 for key in TASK_METRICS}
    compressed = {key: compressed_tasks[key] * 100 for key in TASK_METRICS}
    dense["longbench_v1"] = lb_scores["dense"]
    compressed["longbench_v1"] = lb_scores["compressed"]
    dense_mean, compressed_mean = (math.fsum(values.values()) / 9 for values in (dense, compressed))
    drop = math.fsum(dense[key] - compressed[key] for key in dense) / 9
    # Absorb only ordinary double-precision aggregation rounding at the boundary.
    mean_passed = drop <= 0.8 or math.isclose(drop, 0.8, rel_tol=0, abs_tol=1e-12)
    # Keep the existing report/CLI field; MMLU has no independent hard threshold.
    mmlu_passed = True
    bits, weights = budget["user_linear_bits"], budget["original_linear_weights"]
    _require(type(bits) is int and type(weights) is int and weights > 0, "Invalid actual storage accounting.")
    width_passed = bits * 2 <= weights * 5 if model_type == "qwen3" else bits * 2 < weights * 5
    budget_passed = bool(budget["budget_passed"]) and width_passed
    passed = mean_passed and budget_passed
    failures = []
    if not mean_passed:
        failures.append(f"Nine-term score drop {drop:.12g} exceeds 0.8 points.")
    if not budget_passed:
        failures.append("Actual projection storage/rank/protection budget does not pass.")
    failures.extend(budget.get("contract_violations", []))
    return {
        "schema": "w25_nine_task_acceptance_v2", "status": "passed" if passed else "not_met",
        "passed": passed, "quality_target_evaluated": True, "model_type": model_type,
        "protocol": {"aggregation": "(eight downstream scores * 100 + LongBench21 task macro) / 9",
                     "longbench_tasks": 21, "longbench_samples_per_model": 4750,
                     "longbench_macro": "unweighted task macro; not six-category macro",
                     "maximum_absolute_drop_points": 0.8,
                     "minimum_mmlu_percent": None, "mmlu_hard_gate_applicable": False,
                     "mmlu_preference": "One equally weighted downstream task; no independent threshold or priority.",
                     "max_linear_bpw": 2.5, "strict_bpw_bound": model_type == "llama",
                     "max_lora_rank": 16, "num_fewshot": 0, "limit": None,
                     "score_boundary_roundoff_tolerance": 1e-12},
        "baseline_scores": dense, "compressed_scores": compressed,
        "baseline_nine_task_score": dense_mean, "compressed_nine_task_score": compressed_mean,
        "minimum_compressed_nine_task_score": dense_mean - 0.8,
        "mean_drop_points": drop,
        "score_deltas": {key: compressed[key] - dense[key] for key in dense},
        "mean_passed": mean_passed, "mmlu_passed": mmlu_passed, "budget_passed": budget_passed,
        "bit_budget": budget, "failure_reasons": failures,
        "metric_note": "LongBench mixes official metrics; the nine-term result is a composite score, not uniform accuracy.",
    }


def _candidate(directory, inventory):
    directory = Path(directory).resolve()
    meta = validate_v6_meta(read_json(directory / "checkpoint_meta.json"), expected_kind="final_model")
    shapes = inventory["projection_shapes"]
    _require(Path(meta["base_model_path"]).resolve() == Path(inventory["base_model_path"]), "Wrong candidate base model.")
    _require(set(meta["compressed_targets"]) == set(shapes) and not meta["pending_dense_targets"]
             and not meta["skip_targets"], "All actual model projections must be compressed.")
    specs = {row["name"]: row for row in meta["converted_modules"]}
    _require(set(specs) == set(shapes) and len(specs) == len(meta["converted_modules"]), "Projection inventory differs.")
    _require(not meta.get("post_norm_head_linear") and not (meta.get("extra_meta") or {}).get("residual_lora"),
             "Auxiliary mappings must be finalized; residual skip branches are outside this acceptance.")
    for name, spec in specs.items():
        _require([spec["out_features"], spec["in_features"]] == shapes[name]
                 and not spec.get("has_original_weight") and not spec.get("always_use_original"),
                 f"Invalid or dense-bypassed projection: {name}")
        dims = spec.get("stage_codebook_dims") or [spec["codebook_dim"]]
        _require(all(type(dim) is int and dim > 0 for dim in dims), "Invalid codebook dimensions.")
        for axis in ("input", "output"):
            indices = spec.get(f"protected_{axis}_indices")
            if indices is not None:
                _require(all(math.prod(indices["shape"]) % dim == 0 for dim in dims),
                         f"Protection is not a codebook-dimension multiple: {name}")
    budget = audit_checkpoint(directory, max_bpw=2.5, max_lora_rank=16)
    _require(budget["original_linear_weights"] == inventory["original_linear_weights"]
             and budget["compressed_target_count"] == len(shapes), "Actual weight inventory differs.")
    return directory, meta, budget


def _task_configs(paths, compressed):
    configs = {}
    for path in paths:
        raw = read_json(path)
        configs.update((raw["evals"]["lm_eval"] if compressed else raw)["configs"])
    return configs


def summarize(base_model_path, checkpoint_dir, baseline_results, compressed_results,
              longbench_manifest, baseline_longbench_scores, compressed_longbench_scores):
    inventory = model_inventory(base_model_path)
    final, meta, budget = _candidate(checkpoint_dir, inventory)
    baseline = collect_results(baseline_results, inventory)
    compressed = collect_results(compressed_results, inventory, checkpoint=(final, meta))
    _require(baseline["versions"] == compressed["versions"]
             and _task_configs(baseline_results, False) == _task_configs(compressed_results, True),
             "Paired downstream task versions/configurations differ.")
    lb = comparison(baseline_longbench_scores, compressed_longbench_scores, longbench_manifest)
    dense_model, low_model = lb["models"]["dense"], lb["models"]["compressed"]
    for model in (dense_model, low_model):
        _require(Path(model["hf_path"]).resolve() == Path(inventory["base_model_path"]),
                 "LongBench and downstream base/tokenizer identities differ.")
        check_hashes(model["hf_metadata_sha256"])
        for stat in model["hf_weight_stats"]:
            check_stat(stat)
    _require(Path(low_model["checkpoint"]).resolve() == final and low_model["checkpoint_id"] == meta["checkpoint_id"]
             and low_model["metadata_sha256"] == sha256(final / "checkpoint_meta.json"),
             "LongBench and downstream scores belong to different compressed models.")
    check_stat(low_model["weight_stat"])
    result = combine_scores(inventory["model_type"], baseline["task_metrics"], compressed["task_metrics"], lb, budget)
    result.update(checkpoint_dir=str(final), checkpoint_id=meta["checkpoint_id"],
                  base_model_path=inventory["base_model_path"],
                  checkpoint_meta_sha256=sha256(final / "checkpoint_meta.json"),
                  architecture=inventory, downstream_baseline=baseline, downstream_compressed=compressed,
                  longbench_inputs=lb["inputs"], longbench_model_identities=lb["models"],
                  implementation_sha256=sha256(Path(__file__)))
    return result
