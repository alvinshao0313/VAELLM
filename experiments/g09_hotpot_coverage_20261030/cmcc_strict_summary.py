"""Current strict nine-task formula with an actual native rank0 storage audit."""
import argparse
from pathlib import Path

from experiments.g09_hotpot_coverage_20261030 import cmcc_export as c


def rank0_report(report, checkpoint_dir):
    from train_utils.checkpoint_bpw import audit_checkpoint
    meta = c.read(Path(checkpoint_dir) / 'checkpoint_meta.json')
    budget = audit_checkpoint(checkpoint_dir, max_bpw=2.6, max_lora_rank=0)
    c.require(report['quality_target_evaluated'] and report['checkpoint_id'] == meta['checkpoint_id']
              and report['protocol']['maximum_absolute_drop_points'] == .9
              and report['protocol']['strict_score_drop_bound']
              and budget['budget_passed'] and not budget['contract_violations']
              and meta['lora_config'] is None and not (meta.get('extra_meta') or {}).get('residual_lora')
              and all(budget['components'].get(key, {}).get('storage_bits', 0) == 0
                      for key in ('lora', 'residual_lora'))
              and all(budget[key] == report['bit_budget'][key] for key in c.FIELDS),
              'Require the same complete current-profile report and actual native rank0 storage.')
    report = {**report, 'protocol': {**report['protocol'], 'max_lora_rank': 0}, 'bit_budget': budget,
              'rank0_checkpoint_meta_binding': c.bind(Path(checkpoint_dir) / 'checkpoint_meta.json'),
              'rank0_audit_source': c.bind(Path(__file__))}
    c.require(report['budget_passed'] and report['passed'] == bool(report['mean_passed']),
              'The stricter rank0 audit must preserve the already-validated quality decision.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base_model_path', 'checkpoint_dir', 'longbench_manifest', 'baseline_longbench_scores', 'compressed_longbench_scores'):
        parser.add_argument('--' + name, required=True)
    for name in ('baseline_results', 'compressed_results'):
        parser.add_argument('--' + name, required=True, nargs=2)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    c.require(not args.output.exists(), 'Never overwrite strict nine-task evidence.')
    try:
        from experiments.w25_eval.transfer_nine_task_acceptance import summarize
        values = {key: value for key, value in vars(args).items() if key != 'output'}
        report = rank0_report(summarize(**values), args.checkpoint_dir)
        code = 0 if report['passed'] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = dict(status='invalid', passed=False, quality_target_evaluated=False, error=repr(error))
        code = 2
    c.write(args.output, report)
    print({key: report[key] for key in ('status', 'compressed_nine_task_score', 'passed', 'error') if key in report}, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
