"""Complete-nine evaluation of the actual finished G09 fresh Muon32 endpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / 'experiments/cat_aligned_evaluation_20261028'
sys.path.insert(0, str(EVALUATION))
import evaluate as original

b = original.b
SOURCE = Path('/data/star/VAELLM/g09_recovery_20261029/muon_opd32')
PARENT_ID = 'e8e5efd4-655e-462c-b6e0-fedf4e2e0346'
OUTPUT = ROOT / 'result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/evaluation'
DESTINATION = OUTPUT / 'Muon'


def file_binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=b.sha256(path))


def source(profile, stage, checkpoint=None, native_audit=None):
    # The facade owns one endpoint; its empty path component keeps output at Muon/.
    b.require(profile == 'Muon' and stage == '' and checkpoint is native_audit is None,
        'This entry binds only the actual completed G09 fresh Muon formal32 endpoint.')
    report_path = SOURCE / 'formal32/report.json'
    audit_path = SOURCE / 'native_audit.json'
    prepared_path = SOURCE / 'prepared.json'
    directory = SOURCE / 'native_final/final_model'
    meta_path = directory / 'checkpoint_meta.json'
    required = (report_path, audit_path, prepared_path, meta_path,
        SOURCE / 'formal32/run.exit', SOURCE / 'formal32.exit')
    b.require(all(path.is_file() for path in required),
        'Muon formal32 real terminal/report/exits/native audit are not ready: ' + str(SOURCE))
    report, audit, prepared, meta = (b.read(path) for path in (report_path, audit_path, prepared_path, meta_path))
    b.require((SOURCE / 'formal32/run.exit').read_text().strip() == '0'
        and (SOURCE / 'formal32.exit').read_text().strip() == '0',
        'The actual Muon training and complete export pipeline must both have exited0.')
    training = report['training']
    record = prepared['record']
    b.require(prepared['schema'] == 'G09_Muon_OPD_prepared_v1'
        and prepared['source_checkpoint_id'] == report['source_checkpoint_id'] == audit['source_checkpoint_id'] == PARENT_ID
        and report['schema'] == 'G09_Muon_OPD_execution_v1'
        and report['status'] == 'G09_Muon_nativegreedy_OPD_formal32_complete'
        and report['phase'] == audit['phase'] == 'formal32'
        and report['steps'] == training['steps'] == 32
        and training['group_indices'] == list(range(32))
        and [row['step'] for row in training['trajectory']] == list(range(1,33))
        and [row['group_index'] for row in training['trajectory']] == list(range(32))
        and training['trainable_tensors'] == len(record['decoder_parameters']) == 1512
        and training['trainable_parameters'] == record['trainable_elements'] == 4762368
        and training['frozen_state_before'] == training['frozen_state_after'] == record['validated_frozen_state']
        and training['frozen_state_after']['tensors'] == 1911
        and report['source_restored_exact'] and report['saved_decoder_matches_final_FP32_cast_exact']
        and report['endpoint']['save_reload_install_exact']
        and report['prepared_sha256'] == audit['prepared_sha256'] == b.sha256(prepared_path)
        and report['contract'] == prepared['contract'],
        'Require the real same-G09/all1512 fresh Muon32 trajectory, frozen complement, and endpoint proof.')
    kappa = training['Muon_fixed_kappa']
    state = training['optimizer_final_state']
    b.require(type(kappa) in (int,float) and math.isfinite(kappa) and kappa > 0
        and audit['Muon_fixed_kappa'] == kappa and state['kappa'] == kappa
        and state['Muon_leaves'] == 504 and state['Adam_leaves'] == 1008
        and state['steps'] == [32] and state['state_FP32_finite'], 'Require actual continuous final Muon/aux Adam state32.')
    b.require(audit['schema'] == 'G09_Muon_online_OPD_native_audit_v1'
        and audit['status'] == 'verified' and audit['profile'] == 'G'
        and Path(audit['checkpoint_dir']).resolve() == directory
        and audit['checkpoint_id'] == meta['checkpoint_id'] != PARENT_ID
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['frozen_tensors_compared'] == 1911 and audit['code_bank_count'] == 504
        and audit['fresh_strict_reload_state_exact'] and audit['source_geometry_unchanged']
        and audit['endpoint_report'] == file_binding(report_path)
        and audit['checkpoint_meta_sha256'] == b.sha256(meta_path),
        'Require this actual formal32 new-UUID native3423/504-bank/1911-frozen strict export.')
    uuid.UUID(meta['checkpoint_id'])
    short_audit = SOURCE / 'short2/native_audit.json'
    if short_audit.is_file():
        b.require(meta['checkpoint_id'] != b.read(short_audit)['checkpoint_id'], 'A short2 export is not the formal32 endpoint.')
    b.check_stat(audit['checkpoint_weight_stat'])
    actual_meta, budget = b.recovery().checkpoint(directory)
    b.require(actual_meta == meta and budget['budget_passed']
        and audit['checkpoint_weight_stat'] == b.file_stat(directory / meta['state_dict_file'])
        and all(budget[key] == audit['bit_budget'][key] == audit['source_bit_budget'][key]
            for key in b.BUDGET_FIELDS), 'Actual Muon rank0/logical2.6 native storage differs from its verified source/audit.')
    pins = {item['path']:item['sha256'] for item in prepared['bindings'].values()}
    pins.update(audit['sources']['input_evidence_sha256'])
    pins.update({str(path):b.sha256(path) for path in required})
    b.check_hashes(pins)
    return dict(profile='G', stage='Muon', original_phase='formal32', checkpoint_dir=str(directory),
        checkpoint_id=meta['checkpoint_id'], source_checkpoint_id=PARENT_ID,
        native_audit=str(audit_path), native_audit_sha256=b.sha256(audit_path),
        checkpoint_weight_stat=audit['checkpoint_weight_stat'], bit_budget=budget, evidence_sha256=pins), meta


def source_pins(binding):
    paths = (Path(__file__).resolve(), ROOT / 'experiments/w25_eval/summarize_transfer_nine_task.py',
        ROOT / 'experiments/w25_eval/transfer_nine_task_acceptance.py')
    return {**b.source_pins(binding), **{str(path):b.sha256(path) for path in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=original.gpu_list, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    b.cpu_environment()
    facade = SimpleNamespace(**vars(b))
    facade.source, facade.source_pins, facade.OUTPUT = source, source_pins, OUTPUT
    original.b = facade
    record = original.evaluate(SimpleNamespace(profile='Muon', stage='', checkpoint=None,
        native_audit=None, check_only=args.check_only, gpus=args.gpus))
    if args.check_only: return
    plan = b.read(record['longbench_plan'])
    output = DESTINATION / 'transfer_acceptance.json'
    argv = [str(b.PYTHON), '-B', '-m', 'experiments.w25_eval.summarize_transfer_nine_task',
        *plan['nine_task_argv'][2:]]
    argv[argv.index('--output')+1] = str(output)
    b.check_hashes(record['source_sha256'])
    environment = {**os.environ, 'PYTHONPATH':str(ROOT), 'CUDA_VISIBLE_DEVICES':''}
    with output.with_suffix('.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
    b.require(result.returncode in (0,1), 'The current-goal full-nine report failed validation.')
    report = b.read(output)
    b.require(report['quality_target_evaluated'] and report['checkpoint_id'] == record['checkpoint_id']
        and report['protocol']['maximum_absolute_drop_points'] == .9
        and report['protocol']['strict_score_drop_bound'] and report['protocol']['max_linear_bpw'] == 2.6,
        'Require the same actual complete Muon endpoint and current strict0.9/logical2.6 profile.')
    outcome = dict(status='complete_full8_full21_current_goal', checkpoint_id=record['checkpoint_id'],
        source_checkpoint_id=PARENT_ID, original_evaluation=record, acceptance_argv=argv,
        acceptance_exit_code=result.returncode, acceptance_report=str(output),
        score=report['compressed_nine_task_score'], passed=report['passed'])
    b.write_json(DESTINATION / 'current_goal_result.json', outcome)
    print(json.dumps({key:outcome[key] for key in ('status','checkpoint_id','score','passed')}), flush=True)


if __name__ == '__main__':
    main()
