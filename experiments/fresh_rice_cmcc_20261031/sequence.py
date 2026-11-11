"""One fixed run -> actual audit -> complete-nine sequence; no wait/retry/resume."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from . import sources as s
from experiments.cmcc_runtime_20261030 import sequence as existing
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, TOPIC

ROOT = ARTIFACTS / 'fresh_rice_cmcc_20261031'
EVIDENCE = TOPIC / 'fresh_rice_cmcc_20261031'
SUPPORT = ARTIFACTS / 'adaptive_class_cmcc_20261030/evaluation_support'


def prepare(output):
    existing.cpu_environment()
    output = output.resolve()
    s.c.require(output.is_relative_to(EVIDENCE) and not os.path.lexists(output), 'Prepare a fresh isolated C1 plan directory.')
    binding = s.bind(s.COMMON)
    from .evaluate import setup
    setup(SUPPORT)
    support = s.c.read(SUPPORT / 'prepared.json')
    pins = {**s.source_pins(), **support['source_sha256'],
        str(SUPPORT / 'prepared.json'): s.c.sha(SUPPORT / 'prepared.json'), **binding['common_files_sha256']}
    existing.check_pins(pins)
    target, evaluation = ROOT / 'formal', EVIDENCE / 'evaluation'
    s.c.require(not target.exists() and not evaluation.exists(), 'Never overwrite an existing C1 model/evaluation.')
    formal = dict(name='C1_Rice_formal', argv=[str(PYTHON), '-B', '-u', '-m',
        'experiments.fresh_rice_cmcc_20261031.run_pipeline', '--common', str(s.COMMON), '--output', str(target)],
        environment=existing.environment('0'), gpus=['0'], output=str(target), report=str(target / 'report.json'),
        log=str(output / 'formal.log'), exit_file=str(output / 'formal.exit'))
    formal['environment'].update(PYTHONHASHSEED='31', CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_MODE='offline')
    audit = dict(name='C1_Rice_actual_audit', argv=[str(PYTHON), '-B', '-u', '-m',
        'experiments.fresh_rice_cmcc_20261031.evaluate', '--support', str(SUPPORT), '--component', 'audit',
        '--checkpoint', str(target / 'final_model'), '--pipeline-report', str(target / 'report.json'),
        '--output', str(output / 'final_audit.json')], environment=existing.environment(''),
        log=str(output / 'audit.log'), exit_file=str(output / 'audit.exit'))
    quality = dict(name='C1_Rice_complete_nine', argv=[str(PYTHON), '-B', '-u', '-m',
        'experiments.fresh_rice_cmcc_20261031.evaluate', '--support', str(SUPPORT), '--checkpoint', str(target / 'final_model'),
        '--pipeline-report', str(target / 'report.json'), '--output', str(evaluation), '--gpus', '0,1,2'],
        environment=existing.environment(''), gpus=['0', '1', '2'], output=str(evaluation),
        log=str(output / 'evaluation.log'), exit_file=str(output / 'evaluation.exit'))
    plan = dict(schema='fresh_C1_Rice_single_sequence_v1', created_unix=time.time(), output=str(output),
        source_binding=binding, source_sha256=pins, formal=formal, audit=audit, evaluation=quality,
        lock_file=str(output / 'sequence.lock'), automatic_retry=False, automatic_resume=False,
        order=['C1_Rice_formal', 'C1_Rice_actual_audit', 'C1_Rice_complete_nine'],
        budget=dict(extra_gate_fit_updates=0, third_stage_fit_updates=0, newly_built_Rice_gates=36,
            recovery_updates=1500, own_live_suffix_VAE_updates=40000, full8_evaluations=1, LongBench21_evaluations=1,
            additional_Muon_updates=0),
        boundary='Fresh Rice only. No old patch, third-stage fit, Muon32, waiting, polling, reservation or retry.')
    output.mkdir(parents=True)
    s.c.write(output / 'plan.json', plan)
    print(dict(status='prepared_fresh_C1_Rice_single_sequence', plan=str(output / 'plan.json')), flush=True)


def run(plan_path):
    existing.cpu_environment()
    plan_path = plan_path.resolve(strict=True)
    plan = s.c.read(plan_path)
    s.c.require(plan['schema'] == 'fresh_C1_Rice_single_sequence_v1', 'Use this reviewed fixed sequence.')
    existing.check_pins(plan['source_sha256'])
    output = Path(plan['output'])
    record_path = output / 'run.json'
    s.c.require(not record_path.exists(), 'This one-shot C1 sequence already started.')
    with Path(plan['lock_file']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s.c.require(not record_path.exists(), 'This one-shot C1 sequence already ran before lock acquisition.')
        record = dict(status='validating_source', pid=os.getpid(), start_ticks=existing.proc_start(os.getpid()),
            command=[sys.executable, *sys.argv], started_unix=time.time(), plan_sha256=s.c.sha(plan_path), stages={})
        s.c.write(record_path, record)
        try:
            s.c.require(s.bind(s.COMMON) == plan['source_binding'], 'The reviewed actual common source changed.')

            def changed(stage):
                record['stages'][stage['name']] = stage
                record['status'] = stage['name'] + '_' + stage['status']
                s.c.write(record_path, record)

            record['formal_resource_check'] = existing.check_resources(plan['formal']['gpus'])
            existing.launch_child(plan['formal'], changed)
            existing.check_pins(plan['source_sha256'])
            existing.launch_child(plan['audit'], changed)
            audit = s.c.read(output / 'final_audit.json')
            s.c.require(audit['status'] == 'verified_actual_fresh_C1_Rice_final' and audit['exit_code'] == 0,
                'The actual final audit failed; do not evaluate.')
            record['final_checkpoint_id'] = audit['checkpoint_id']
            record['evaluation_resource_check'] = existing.check_resources(plan['evaluation']['gpus'])
            existing.launch_child(plan['evaluation'], changed)
            record['complete_nine'] = existing.validate_evaluation(plan['evaluation']['output'], audit['checkpoint_id'])
            existing.check_pins(plan['source_sha256'])
            record.update(status='completed_C1_Rice_and_complete_nine', exit_code=0)
        except BaseException:
            record.update(status='failed', exit_code=1, error=traceback.format_exc())
            raise
        finally:
            record['finished_unix'] = time.time()
            s.c.write(record_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path)
    action.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        run(args.run)


if __name__ == '__main__':
    main()
