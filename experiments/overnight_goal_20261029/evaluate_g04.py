"""Fixed G04 C1 only: scale32 MLP plus q64 Q/K/V/O, complete full8/full21."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/cat_aligned_evaluation_20261028'))
import evaluate as original

b = original.b
PARENT = Path('/data/star/VAELLM/cat_aligned_recovery_20261028/G')
SOURCE = PARENT / '04_C1'
PARENT_ID = '38a7461e-a2f1-4586-80e2-b99883ef2628'
OUTPUT = ROOT / 'result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/evaluation'


def source(profile, stage, checkpoint=None, native_audit=None):
    b.require(profile == 'G04' and stage == '' and checkpoint is native_audit is None,
              'Only the fixed scale32-MLP/q64-attention G04 C1 endpoint is accepted.')
    directory = SOURCE / 'final_model'
    audit_path = SOURCE / 'assembly_report.json'
    b.require(audit_path.is_file() and (directory / 'checkpoint_meta.json').is_file(),
              'Actual G04 native assembly/fresh audit is not ready: ' + str(SOURCE))
    audit = b.read(audit_path)
    provenance = audit['sources']
    parents = {}
    for arm, steps in (('scale32', 32), ('q64', 64)):
        path = PARENT / '03_protection' / arm / 'native_audit.json'
        parent = b.read(path)
        b.require(parent['schema'] == 'CG_aligned_protection_native_audit_v1'
            and parent['status'] == 'verified' and parent['profile'] == 'G' and parent['arm'] == arm
            and parent['phase'] == 'formal' and parent['source_checkpoint_id'] == PARENT_ID
            and parent['actual_steps'] == parent['planned_steps'] == steps,
            'Require both actual same-G02 formal32/64 native parent audits.')
        field = 'base_audit' if arm == 'scale32' else 'donor_audit'
        b.require(provenance[field] == dict(path=str(path), bytes=path.stat().st_size, sha256=b.sha256(path)),
                  'A fixed G04 parent audit changed after assembly.')
        parents[arm] = parent
    b.require(provenance['aligned02_parent_checkpoint_id'] == PARENT_ID and provenance['profile'] == 'G'
        and provenance['base_arm'] == 'scale32' and provenance['base_actual_steps'] == 32
        and provenance['donor_arm'] == 'q64' and provenance['donor_actual_steps'] == 64
        and provenance['categories'] == ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        and provenance['base_checkpoint_id'] == parents['scale32']['checkpoint_id']
        and provenance['donor_checkpoint_id'] == parents['q64']['checkpoint_id']
        and provenance['parent_checkpoint_ids'] == [parents['scale32']['checkpoint_id'], parents['q64']['checkpoint_id']],
        'Require this one fixed whole-family combination, never reverse/family/row selection.')
    binding, meta = b.source('G', '04', checkpoint=directory, native_audit=audit_path)
    return {**binding, 'stage': 'G04', 'original_aligned_stage': '04', 'source_checkpoint_id': PARENT_ID}, meta


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
    b.require(not os.path.lexists(OUTPUT / 'G04'), 'Never overwrite a G04 endpoint evaluation.')
    facade = SimpleNamespace(**vars(b))
    facade.source, facade.source_pins, facade.OUTPUT = source, source_pins, OUTPUT
    original.b = facade
    record = original.evaluate(SimpleNamespace(profile='G04', stage='', checkpoint=None,
        native_audit=None, check_only=args.check_only, gpus=args.gpus))
    if args.check_only:
        return
    plan = b.read(record['longbench_plan'])
    output = OUTPUT / 'G04/transfer_acceptance.json'
    argv = [str(b.PYTHON), '-B', '-m', 'experiments.w25_eval.summarize_transfer_nine_task', *plan['nine_task_argv'][2:]]
    argv[argv.index('--output') + 1] = str(output)
    b.check_hashes(record['source_sha256'])
    with output.with_suffix('.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': ''},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    b.require(result.returncode in (0, 1), 'The current-goal complete-nine report failed validation.')
    report = b.read(output)
    b.require(report['quality_target_evaluated'] and report['checkpoint_id'] == record['checkpoint_id']
        and report['protocol']['maximum_absolute_drop_points'] == .9 and report['protocol']['strict_score_drop_bound']
        and report['protocol']['max_linear_bpw'] == 2.6, 'Require this same complete G04/current strict profile.')
    outcome = dict(status='complete_full8_full21_current_goal', checkpoint_id=record['checkpoint_id'],
        source_checkpoint_id=PARENT_ID, original_evaluation=record, acceptance_argv=argv,
        acceptance_exit_code=result.returncode, acceptance_report=str(output),
        score=report['compressed_nine_task_score'], passed=report['passed'])
    b.write_json(output.with_name('current_goal_result.json'), outcome)
    print(json.dumps({key: outcome[key] for key in ('status', 'checkpoint_id', 'score', 'passed')}), flush=True)


if __name__ == '__main__':
    main()
