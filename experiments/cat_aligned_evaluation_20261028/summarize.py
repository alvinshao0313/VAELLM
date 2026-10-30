"""Thin CLI for the existing complete native2.6/rank0 nine-term formula."""
import argparse
import json
from pathlib import Path
import bindings as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base_model_path', 'checkpoint_dir', 'longbench_manifest', 'baseline_longbench_scores', 'compressed_longbench_scores'):
        parser.add_argument('--' + name, required=True)
    for name in ('baseline_results', 'compressed_results'):
        parser.add_argument('--' + name, required=True, nargs=2)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b.cpu_environment()
    b.require(not args.output.exists(), 'Never overwrite nine-term evidence.')
    try:
        result = b.formula().summarize(**{key: value for key, value in vars(args).items() if key != 'output'})
        b.require(abs(result['baseline_nine_task_score'] - 68.04714491790848) < 1e-12
                  and abs(result['minimum_compressed_nine_task_score'] - 67.24714491790849) < 1e-12,
                  'The original Qwen BF16 complete-nine reference changed.')
        code = 0 if result['passed'] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = dict(status='invalid', passed=False, quality_target_evaluated=False, error=repr(error))
        code = 2
    b.write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('status', 'compressed_nine_task_score', 'mean_drop_points', 'error') if key in result}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
