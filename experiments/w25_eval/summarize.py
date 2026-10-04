"""Summarize Qwen/Llama BF16 baseline; paired <=1pp low-bit acceptance is Llama-only."""
import argparse
import json
from pathlib import Path

from experiments.w25_eval.acceptance import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model-path', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--baseline-results', nargs=2, required=True)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--compressed-results', nargs=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f'Refusing to overwrite an existing output: {args.output}')
    try:
        report = summarize(args.base_model_path, args.source_manifest, args.baseline_results,
                           checkpoint_dir=args.checkpoint_dir, compressed_results=args.compressed_results)
        code = 0 if report['status'] in ('passed', 'baseline_verified') else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {'status': 'invalid', 'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'mean_drop_percentage_points', 'error') if k in report}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
