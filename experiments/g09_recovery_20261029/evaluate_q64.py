"""Complete-nine evaluation of the actual G02 -> D+q64 formal64 native endpoint."""
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
SOURCE = Path('/data/star/VAELLM/cat_aligned_recovery_20261028/G/03_protection/q64')
PARENT_ID = '38a7461e-a2f1-4586-80e2-b99883ef2628'
OUTPUT = ROOT / 'result/qwen3_8b_mmlu65_20261004/g09_recovery_20261029/evaluation'


def source(profile, stage, checkpoint=None, native_audit=None):
    b.require(profile == 'G' and stage == 'q64' and checkpoint is native_audit is None,
              'This entry binds only the actual G02 -> D+q64 formal64 endpoint.')
    audit_path = SOURCE / 'native_audit.json'
    directory = SOURCE / 'native_final/final_model'
    b.require(audit_path.is_file() and (directory / 'checkpoint_meta.json').is_file(),
              'The q64 native terminal/audit is not ready: ' + str(SOURCE))
    audit = b.read(audit_path)
    b.require(audit['source_checkpoint_id'] == audit['sources']['source_checkpoint_id'] == PARENT_ID
        and audit['sources']['parent_checkpoint_ids'] == [PARENT_ID],
        'Require the actual G02 parent, never scale32 or another predecessor.')
    binding, meta = b.source('G', '03b', checkpoint=directory, native_audit=audit_path)
    return {**binding, 'stage': 'q64', 'original_aligned_stage': '03b',
            'source_checkpoint_id': PARENT_ID}, meta


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
    b.require(not os.path.lexists(OUTPUT / 'G/q64'), 'Never overwrite a q64 endpoint evaluation.')
    facade = SimpleNamespace(**vars(b))
    facade.source, facade.source_pins, facade.OUTPUT = source, source_pins, OUTPUT
    original.b = facade
    record = original.evaluate(SimpleNamespace(profile='G', stage='q64', checkpoint=None,
        native_audit=None, check_only=args.check_only, gpus=args.gpus))
    if args.check_only:
        return
    plan = b.read(record['longbench_plan'])
    output = OUTPUT / 'G/q64/transfer_acceptance.json'
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
    b.require(report['quality_target_evaluated'] and report['checkpoint_id'] == record['checkpoint_id']
        and report['protocol']['maximum_absolute_drop_points'] == .9
        and report['protocol']['strict_score_drop_bound'] and report['protocol']['max_linear_bpw'] == 2.6,
        'Require this same complete q64 endpoint and the current strict-drop/logical-storage profile.')
    outcome = dict(status='complete_full8_full21_current_goal', checkpoint_id=record['checkpoint_id'],
        source_checkpoint_id=PARENT_ID, original_evaluation=record,
        acceptance_argv=argv, acceptance_exit_code=result.returncode,
        acceptance_report=str(output), score=report['compressed_nine_task_score'], passed=report['passed'])
    b.write_json(output.with_name('current_goal_result.json'), outcome)
    print(json.dumps({key: outcome[key] for key in ('status', 'checkpoint_id', 'score', 'passed')}), flush=True)


if __name__ == '__main__':
    main()
