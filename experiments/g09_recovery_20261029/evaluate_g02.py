"""Bind the verified G02 endpoint to the unchanged complete-nine evaluator."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT/'experiments/cat_aligned_evaluation_20261028'
sys.path.insert(0, str(EVALUATION))
import evaluate as original

b = original.b
OUTPUT = ROOT/'result/qwen3_8b_mmlu65_20261004/g09_recovery_20261029/evaluation'


def source(profile, stage, checkpoint=None, native_audit=None):
    b.require(profile == 'G' and stage == '02' and checkpoint is native_audit is None,
              'This entry binds only the actual completed G02 endpoint.')
    protection = b.module(ROOT/'experiments/cat_aligned_protection_20261028/bindings.py', '_G02_actual_source')
    directory, meta, audit = protection.source()
    actual_meta, budget = b.recovery().checkpoint(directory)
    b.require(actual_meta == meta and all(budget[key] == audit['bit_budget'][key] for key in b.BUDGET_FIELDS),
              'Actual G02 native storage differs from its completed source audit.')
    pins = dict(audit['sources']['input_evidence_sha256'])
    pins.update({str(path): b.sha256(path) for path in
        (protection.SOURCE_AUDIT, directory/'checkpoint_meta.json', Path(protection.__file__))})
    return dict(profile='G', stage='02', checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'],
        native_audit=str(protection.SOURCE_AUDIT), native_audit_sha256=b.sha256(protection.SOURCE_AUDIT),
        checkpoint_weight_stat=audit['checkpoint_weight_stat'], bit_budget=budget, evidence_sha256=pins), meta


def source_pins(binding):
    return {**b.source_pins(binding), str(Path(__file__)): b.sha256(Path(__file__))}


def main():
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Keep the evaluation coordinator on CPU.')
    # Only the endpoint resolver and output namespace differ. The original
    # inference, partitioning, raw sample checks, and old report stay intact.
    bindings = SimpleNamespace(**vars(b))
    bindings.source, bindings.source_pins, bindings.OUTPUT = source, source_pins, OUTPUT
    original.b = bindings
    args = SimpleNamespace(profile='G', stage='02', checkpoint=None, native_audit=None,
                           check_only=False, gpus=['5', '6'])
    record = original.evaluate(args)
    plan = b.read(record['longbench_plan'])
    output = OUTPUT/'G/02/transfer_acceptance.json'
    argv = [str(b.PYTHON), '-B', '-m', 'experiments.w25_eval.summarize_transfer_nine_task',
            *plan['nine_task_argv'][2:]]
    argv[argv.index('--output')+1] = str(output)
    environment = {**os.environ, 'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': ''}
    with (OUTPUT/'G/02/transfer_acceptance.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    b.require(result.returncode in (0, 1), 'New-goal acceptance failed validation; keep the original complete outputs.')
    report = b.read(output)
    b.require(report['quality_target_evaluated'] and report['checkpoint_id'] == record['checkpoint_id'],
              'The new report must describe this same complete actual endpoint.')
    b.write_json(OUTPUT/'G/02/current_goal_result.json', dict(status='complete_full8_full21_current_goal',
        source_checkpoint_id=record['checkpoint_id'], original_evaluation=record,
        acceptance_argv=argv, acceptance_exit_code=result.returncode,
        acceptance_report=str(output), score=report['compressed_nine_task_score'], passed=report['passed']))
    print(json.dumps(dict(status='complete_full8_full21_current_goal', score=report['compressed_nine_task_score'],
                          passed=report['passed'])), flush=True)


if __name__ == '__main__':
    main()
