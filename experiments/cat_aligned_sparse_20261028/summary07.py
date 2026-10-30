"""Original nine-term formula and mature Rice collector, with actual07 identity."""
import argparse
import json
import os
from pathlib import Path
import sys
import sparse07_bindings as b


def candidate(directory, inventory):
    checkpoint, audit, meta = b.check_export(quality=True)
    b.runtime.check_gpu(checkpoint)
    b.require(Path(directory).resolve() == checkpoint and inventory['projection_count'] == 252
        and Path(meta['base_model_path']).resolve() == Path(inventory['base_model_path'])
        and audit['bit_budget']['original_linear_weights'] == inventory['original_linear_weights'], 'Actual Rice07 architecture differs.')
    return checkpoint, meta, audit['bit_budget']


def formula():
    directory = b.R / 'qwen_b32_mlp_s3_20261023'
    sys.path.insert(0, str(directory))
    value = b.c.module(directory / 'summarize_nine.py', '_aligned07_original_nine_formula')
    value._candidate = candidate
    value.collect_results = b.evaluator.collect_results
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base_model_path', 'checkpoint_dir', 'longbench_manifest', 'baseline_longbench_scores', 'compressed_longbench_scores'):
        parser.add_argument('--'+name, required=True)
    for name in ('baseline_results', 'compressed_results'):
        parser.add_argument('--'+name, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    b.require(not os.path.lexists(args.output), 'Never overwrite actual nine-task evidence.')
    try:
        result = formula().summarize(**{key: value for key, value in vars(args).items() if key != 'output'})
        b.require(abs(result['baseline_nine_task_score']-68.04714491790848) < 1e-12
            and abs(result['minimum_compressed_nine_task_score']-67.24714491790849) < 1e-12,
            'Original BF16/nine-term target changed.')
        result.update(schema='CG_aligned_abs005_Rice7_nine_task_acceptance_v1', checkpoint_loader=b.FORMAT)
        code = 0 if result['passed'] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        result, code = dict(status='invalid', passed=False, quality_target_evaluated=False, error=repr(error)), 2
    b.write(args.output, result)
    print(json.dumps({key: result[key] for key in ('status', 'compressed_nine_task_score', 'error') if key in result}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
