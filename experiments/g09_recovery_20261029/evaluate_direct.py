"""Complete-nine evaluation of the actual G09 direct OPD32 native endpoint."""
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
SOURCE = Path('/data/star/VAELLM/g09_recovery_20261029/direct_opd32')
PARENT_ID = 'e8e5efd4-655e-462c-b6e0-fedf4e2e0346'
OUTPUT = ROOT / 'result/qwen3_8b_mmlu65_20261004/g09_recovery_20261029/evaluation'


def source(profile, stage, checkpoint=None, native_audit=None):
    b.require(profile == 'G' and stage == 'direct_opd32' and checkpoint is native_audit is None,
              'This entry binds only the actual G09 direct OPD32 endpoint.')
    audit_path = SOURCE / 'native_audit.json'
    audit = b.read(audit_path)
    execution, prepared, report = (b.read(SOURCE / name) for name in
        ('execution.json', 'prepared.json', 'formal32/report.json'))
    directory = SOURCE / 'native_final/final_model'
    meta = b.read(directory / 'checkpoint_meta.json')
    b.require(audit['schema'] == 'G09_direct_online_OPD_native_audit_v1'
        and audit['status'] == 'verified' and audit['profile'] == profile
        and execution['status'] == 'completed_native_audit_verified'
        and execution['training_exit_code'] == execution['native_export_exit_code'] == 0
        and report['status'] == 'G09_direct_nativegreedy_OPD32_complete'
        and report['steps'] == 32 and report['phase'] == 'formal32'
        and (SOURCE / 'formal32/run.exit').read_text().strip() == '0'
        and report['source_restored_exact'] and report['endpoint']['save_reload_install_exact']
        and [row['group_index'] for row in report['training']['trajectory']] == list(range(32))
        and report['actual_target_contract'] == prepared['actual_target_contract']
        and report['prepared_sha256'] == b.sha256(SOURCE / 'prepared.json')
        and report['source_checkpoint_id'] == prepared['source_checkpoint_id']
            == audit['source_checkpoint_id'] == execution['source_checkpoint_id'] == PARENT_ID
        and audit['checkpoint_id'] == execution['checkpoint_id'] == meta['checkpoint_id'] != PARENT_ID
        and Path(audit['checkpoint_dir']).resolve() == directory.resolve()
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['frozen_tensors_compared'] == 1911 and audit['code_bank_count'] == 504
        and audit['fresh_strict_reload_state_exact'] and audit['source_geometry_unchanged'],
        'Require the completed direct32 training and its independently verified native endpoint.')
    pins = dict(audit['sources']['input_evidence_sha256'])
    pins.update({item['path']: item['sha256'] for item in prepared['bindings'].values()})
    b.check_hashes(pins)
    b.check_stat(audit['checkpoint_weight_stat'])
    b.require(audit['checkpoint_weight_stat'] == b.file_stat(directory / meta['state_dict_file'])
        and audit['checkpoint_meta_sha256'] == b.sha256(directory / 'checkpoint_meta.json'),
        'Direct32 deployment metadata or weight identity changed.')
    actual_meta, budget = b.recovery().checkpoint(directory)
    b.require(actual_meta == meta and all(budget[key] == audit['bit_budget'][key]
        == audit['source_bit_budget'][key] for key in b.BUDGET_FIELDS),
        'Actual native logical/packed storage differs from the verified direct32 audit.')
    for path in (audit_path, SOURCE / 'execution.json', SOURCE / 'formal32/run.exit', directory / 'checkpoint_meta.json'):
        pins[str(path)] = b.sha256(path)
    return dict(profile=profile, stage=stage, checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'],
        native_audit=str(audit_path), native_audit_sha256=b.sha256(audit_path),
        checkpoint_weight_stat=audit['checkpoint_weight_stat'], bit_budget=budget, evidence_sha256=pins), meta


def source_pins(binding):
    return {**b.source_pins(binding), str(Path(__file__)): b.sha256(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=original.gpu_list, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    b.cpu_environment()
    facade = SimpleNamespace(**vars(b))
    facade.source, facade.source_pins, facade.OUTPUT = source, source_pins, OUTPUT
    original.b = facade
    record = original.evaluate(SimpleNamespace(profile='G', stage='direct_opd32', checkpoint=None,
        native_audit=None, check_only=args.check_only, gpus=args.gpus))
    if args.check_only:
        return
    plan = b.read(record['longbench_plan'])
    output = OUTPUT / 'G/direct_opd32/transfer_acceptance.json'
    argv = [str(b.PYTHON), '-B', '-m', 'experiments.w25_eval.summarize_transfer_nine_task',
        *plan['nine_task_argv'][2:]]
    argv[argv.index('--output') + 1] = str(output)
    environment = {**os.environ, 'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': ''}
    with output.with_suffix('.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
    b.require(result.returncode in (0, 1), 'The current-goal complete-nine report failed validation.')
    report = b.read(output)
    b.require(report['quality_target_evaluated'] and report['checkpoint_id'] == record['checkpoint_id'],
              'The new report must describe this same complete direct32 endpoint.')
    outcome = dict(status='complete_full8_full21_current_goal', checkpoint_id=record['checkpoint_id'],
        original_evaluation=record, acceptance_argv=argv, acceptance_exit_code=result.returncode,
        acceptance_report=str(output), score=report['compressed_nine_task_score'], passed=report['passed'])
    b.write_json(output.with_name('current_goal_result.json'), outcome)
    print(json.dumps({key: outcome[key] for key in ('status', 'checkpoint_id', 'score', 'passed')}), flush=True)


if __name__ == '__main__':
    main()
