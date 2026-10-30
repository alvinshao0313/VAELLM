"""Complete-nine evaluation of the verified G02 -> D+scale32 native endpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / 'experiments/cat_aligned_evaluation_20261028'
sys.path.insert(0, str(EVALUATION))
import evaluate as original

b = original.b
SOURCE = Path('/data/star/VAELLM/cat_aligned_recovery_20261028/G/03_protection/scale32')
CHECKPOINT_ID = '6a446110-a963-45a5-9847-8458f5d88c40'
AUDIT_SHA256 = '29b7c6381aec24a045fb8e8ee643dc5a8a2251739cb773b7ade232121bc2df44'
OUTPUT = ROOT / 'result/qwen3_8b_mmlu65_20261004/g09_recovery_20261029/evaluation'


def source(profile, stage, checkpoint=None, native_audit=None):
    b.require(profile == 'G' and stage == 'scale32' and checkpoint is native_audit is None,
              'This entry binds only the actual completed G02 -> D+scale32 endpoint.')
    audit_path = SOURCE / 'native_audit.json'
    b.require(b.sha256(audit_path) == AUDIT_SHA256, 'The assigned scale32 native audit changed.')
    binding, meta = b.source('G', '03a', checkpoint=SOURCE / 'native_final/final_model', native_audit=audit_path)
    b.require(binding['checkpoint_id'] == meta['checkpoint_id'] == CHECKPOINT_ID,
              'The original03a resolver must return the assigned real scale32 native UUID.')
    return {**binding, 'stage': 'scale32', 'original_aligned_stage': '03a'}, meta


def source_pins(binding):
    paths = (Path(__file__), ROOT / 'experiments/w25_eval/summarize_transfer_nine_task.py',
             ROOT / 'experiments/w25_eval/transfer_nine_task_acceptance.py')
    return {**b.source_pins(binding), **{str(path): b.sha256(path) for path in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=original.gpu_list, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    b.cpu_environment()
    facade = SimpleNamespace(**vars(b))
    facade.source, facade.source_pins, facade.OUTPUT = source, source_pins, OUTPUT
    original.b = facade
    record = original.evaluate(SimpleNamespace(profile='G', stage='scale32', checkpoint=None,
        native_audit=None, check_only=args.check_only, gpus=args.gpus))
    if args.check_only:
        return
    plan = b.read(record['longbench_plan'])
    output = OUTPUT / 'G/scale32/transfer_acceptance.json'
    argv = [str(b.PYTHON), '-B', '-m', 'experiments.w25_eval.summarize_transfer_nine_task',
        *plan['nine_task_argv'][2:]]
    argv[argv.index('--output') + 1] = str(output)
    b.check_hashes(record['source_sha256'])
    environment = {**os.environ, 'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': ''}
    with output.with_suffix('.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
    b.require(result.returncode in (0, 1), 'The current-goal complete-nine report failed validation.')
    report = b.read(output)
    b.require(report['quality_target_evaluated'] and report['checkpoint_id'] == record['checkpoint_id'] == CHECKPOINT_ID
        and report['protocol']['maximum_absolute_drop_points'] == .9
        and report['protocol']['strict_score_drop_bound'] and report['protocol']['max_linear_bpw'] == 2.6,
        'Require this same complete scale32 endpoint and the current strict-drop/logical-storage profile.')
    outcome = dict(status='complete_full8_full21_current_goal', checkpoint_id=record['checkpoint_id'],
        original_evaluation=record, acceptance_argv=argv, acceptance_exit_code=result.returncode,
        acceptance_report=str(output), score=report['compressed_nine_task_score'], passed=report['passed'])
    b.write_json(output.with_name('current_goal_result.json'), outcome)
    print(json.dumps({key: outcome[key] for key in ('status', 'checkpoint_id', 'score', 'passed')}), flush=True)


if __name__ == '__main__':
    main()
