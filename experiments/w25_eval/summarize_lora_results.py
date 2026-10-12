"""Legacy eight-task screening for selected LoRA; not the nine-score goal acceptance."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.export_e2e_selected import _read_json, _sha256
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.lora_audit import load_state, validate_deployment
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.lora_state import saved_lora_selection
from train_utils.checkpoint_v6 import _atomic_write_json


def summarize(checkpoint_dir, result_paths, audit_path, *, minimum_mean_percent, minimum_mmlu_percent):
    _require(all(math.isfinite(value) and 0 <= value <= 100
                 for value in (minimum_mean_percent, minimum_mmlu_percent)), "Invalid explicit score thresholds.")
    final = Path(checkpoint_dir).resolve()
    meta = _read_json(final / "checkpoint_meta.json")
    audit = _read_json(audit_path)
    _require(audit["status"] == "verified" and audit["checkpoint_id"] == meta["checkpoint_id"]
             and audit["checkpoint_meta_sha256"] == _sha256(final / "checkpoint_meta.json"), "Missing matching LoRA audit.")
    architecture = model_inventory(meta["base_model_path"])
    shapes, lora = saved_lora_selection(meta, architecture)
    budget = validate_deployment(meta, load_state(final))
    _require(budget == audit["bit_budget"], "Deployment budget changed since audit.")
    scores = collect_results(result_paths, architecture, checkpoint=(final, meta))
    mean, mmlu = scores["mean_percent"], scores["task_metrics"]["mmlu"] * 100
    passed = mean >= minimum_mean_percent and mmlu >= minimum_mmlu_percent and budget["budget_passed"]
    return {"status": "passed" if passed else "not_met", "passed": passed, **scores,
            "checkpoint_id": meta["checkpoint_id"], "checkpoint_dir": str(final),
            "mmlu_percent": mmlu, "bit_budget": budget, "audit_sha256": _sha256(audit_path),
            "goal_acceptance": False, "scope": "legacy eight-task thresholds only; full LongBench is absent",
            "protocol": {"minimum_mean_percent": minimum_mean_percent, "minimum_mmlu_percent": minimum_mmlu_percent,
                "max_linear_bpw": 2.5, "lora_rank": lora["rank"], "lora_alpha": lora["alpha"],
                "selected_target_count": len(shapes), "num_fewshot": 0, "limit": None,
                "mmlu_batch_size": 64, "seven_batch_size": 32}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs=2, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum_mean_percent", type=float, required=True)
    parser.add_argument("--minimum_mmlu_percent", type=float, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "Refusing to overwrite acceptance.")
    report = summarize(args.checkpoint_dir, args.results, args.audit,
        minimum_mean_percent=args.minimum_mean_percent, minimum_mmlu_percent=args.minimum_mmlu_percent)
    _atomic_write_json(report, str(args.output))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
