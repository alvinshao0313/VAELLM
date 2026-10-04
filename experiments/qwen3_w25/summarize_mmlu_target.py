"""Validate full Qwen3-8B results for mean>=69.5%, MMLU>=65%, W<=2.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.summarize_results import summarize


MINIMUM_MEAN_PERCENT = 69.5
MINIMUM_MMLU_PERCENT = 65.0


def summarize_mmlu_target(checkpoint_dir, result_paths):
    # Keep the existing raw-result, checkpoint and tensor-storage audits intact.
    report = summarize(checkpoint_dir, result_paths)
    mean_percent = report["mean_percent"]
    mmlu_percent = report["task_metrics"]["mmlu"] * 100
    mean_passed = mean_percent >= MINIMUM_MEAN_PERCENT
    mmlu_passed = mmlu_percent >= MINIMUM_MMLU_PERCENT
    report["protocol"].update({
        "minimum_mean_percent": MINIMUM_MEAN_PERCENT,
        "minimum_mmlu_percent": MINIMUM_MMLU_PERCENT,
    })
    report.update({
        "mmlu_percent": mmlu_percent,
        "mean_passed": mean_passed,
        "mmlu_passed": mmlu_passed,
        "mean_shortfall_pp": max(0.0, MINIMUM_MEAN_PERCENT - mean_percent),
        "mmlu_shortfall_pp": max(0.0, MINIMUM_MMLU_PERCENT - mmlu_percent),
        "accuracy_passed": mean_passed and mmlu_passed,
    })
    report["passed"] = report["accuracy_passed"] and report["budget_passed"]
    report["status"] = "passed" if report["passed"] else "not_met"
    report["failure_reasons"] = (
        ([] if mean_passed else [
            f"Eight-task mean {mean_percent:.9f}% is below {MINIMUM_MEAN_PERCENT:g}%."
        ])
        + ([] if mmlu_passed else [
            f"MMLU accuracy {mmlu_percent:.9f}% is below {MINIMUM_MMLU_PERCENT:g}%."
        ])
        + report["bit_budget"]["contract_violations"]
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Exit codes: 0 = passed; 1 = valid results below a target; 2 = invalid input/output.",
    )
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs=2, required=True, metavar=("SHARD1", "SHARD2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        print(f"Refusing to overwrite an existing report: {args.output}", file=sys.stderr)
        return 2
    try:
        report = summarize_mmlu_target(args.checkpoint_dir, args.results)
        exit_code = 0 if report["passed"] else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {"status": "invalid", "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        exit_code = 2
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects an output created during the audit.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except OSError as exc:
        print(f"Cannot create report: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in
                      ("status", "mean_percent", "mmlu_percent", "budget_passed", "error")
                      if key in report}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
