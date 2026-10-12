"""Accept Qwen/Llama using eight tasks plus complete LongBench as nine equal terms."""
import argparse
import json
from pathlib import Path

from experiments.w25_eval.nine_task_acceptance import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base_model_path", "checkpoint_dir", "longbench_manifest",
                 "baseline_longbench_scores", "compressed_longbench_scores"):
        parser.add_argument("--" + name, required=True)
    for name in ("baseline_results", "compressed_results"):
        parser.add_argument("--" + name, required=True, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"Refusing to overwrite an existing output: {args.output}")
    try:
        report = summarize(**{key: value for key, value in vars(args).items() if key != "output"})
        code = 0 if report["passed"] else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {"status": "invalid", "passed": False, "quality_target_evaluated": False,
                  "error": f"{type(exc).__name__}: {exc}"}
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "baseline_nine_task_score",
        "compressed_nine_task_score", "mean_drop_points", "mmlu_passed", "budget_passed", "error") if key in report}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
