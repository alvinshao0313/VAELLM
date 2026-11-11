"""Consume a verified completed common source or wait once, then C0/full-nine."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import sources as s

from experiments.cmcc_runtime_20261030 import sequence as existing
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC

ROOT = ARTIFACTS / 'fresh_fixed2_cmcc_20261030'
COMMON = ARTIFACTS / 'fresh_common_cmcc_20261030/formal'
COMMON_EXIT = COMMON.with_suffix('.exit')
SUPPORT = ARTIFACTS / 'adaptive_class_cmcc_20261030/evaluation_support'


def prepare(output, common_shell_pid, completed_common=False):
    existing.cpu_environment()
    output = output.resolve()
    s.c.require(output.is_relative_to(ROOT) and not os.path.lexists(output), 'Use a new isolated C0 sequence directory.')
    run = s.c.read(COMMON / 'run.json')
    request = s.c.read(COMMON / 'request.json')
    parent = dict(directory=str(COMMON), driver_pid=run['pid'], driver_start_ticks=run['start_ticks'],
        exit_file=str(COMMON_EXIT), request_sha256=s.c.sha(COMMON / 'request.json'),
        completed_before_plan=completed_common)
    if completed_common:
        s.c.require(common_shell_pid is None and COMMON_EXIT.read_text().strip() == '0',
            'The completed common prerequisite requires its actual exit0, without a live-shell attachment.')
        s.bind(COMMON)
    else:
        driver_start = existing.proc_start(run['pid'])
        shell_start = existing.proc_start(common_shell_pid)
        s.c.require(run['status'] == 'running' and run['validation_only'] is False
            and request['formal_parent_eligible'] is True and driver_start == run['start_ticks']
            and shell_start is not None and not COMMON_EXIT.exists(), 'Attach only to the actual running formal common task.')
        fields = Path(f'/proc/{run["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
        shell_command = Path(f'/proc/{common_shell_pid}/cmdline').read_bytes().split(b'\0')
        s.c.require(int(fields[1]) == common_shell_pid and any(str(COMMON_EXIT).encode() in item for item in shell_command),
            'The bound shell must actually own the common driver and write its terminal exit file.')
        parent.update(shell_pid=common_shell_pid, shell_start_ticks=shell_start)
    s.common.s.check_pins(request['source_sha256'])
    from .evaluate import setup
    from experiments.adaptive_class_evaluation_20261030 import bindings as b
    setup(SUPPORT)
    support = s.c.read(SUPPORT / 'prepared.json')
    pins = {**s.source_pins(), **support['source_sha256'], str(SUPPORT / 'prepared.json'): s.c.sha(SUPPORT / 'prepared.json'),
        str(COMMON / 'request.json'): s.c.sha(COMMON / 'request.json')}
    existing.check_pins(pins)
    target = ROOT / (output.name + '_C0')
    evaluation = TOPIC / 'fresh_fixed2_cmcc_20261030' / output.name
    s.c.require(not os.path.lexists(target) and not os.path.lexists(evaluation), 'Never overwrite a C0 model or quality evaluation.')
    formal = dict(name='C0_fixed2', argv=[str(PYTHON), '-B', '-u', '-m',
        'experiments.fresh_fixed2_cmcc_20261030.run_pipeline', '--common', str(COMMON), '--output', str(target)],
        environment=existing.environment('5'), gpus=['5'], output=str(target),
        report=str(target / 'report.json'), log=str(output / 'C0_formal.log'), exit_file=str(output / 'C0_formal.exit'))
    formal['environment'].update(PYTHONHASHSEED='31', CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_MODE='offline')
    quality = dict(name='C0_complete_nine', argv=[str(PYTHON), '-B', '-u', '-m',
        'experiments.fresh_fixed2_cmcc_20261030.evaluate', '--support', str(SUPPORT), '--checkpoint', str(target / 'final_model'),
        '--pipeline-report', str(target / 'report.json'), '--output', str(evaluation), '--gpus', '6,7'],
        environment=existing.environment(''), gpus=['6', '7'], output=str(evaluation),
        log=str(output / 'C0_evaluation.log'), exit_file=str(output / 'C0_evaluation.exit'))
    plan = dict(schema='fresh_C0_fixed2_single_sequence_v1', created_unix=time.time(), output=str(output),
        common=parent,
        source_sha256=pins, formal=formal, evaluation=quality, automatic_retry=False, automatic_resume=False,
        lock_file=str(ROOT / 'sequence.lock'), budget=dict(extra_gate_fit_updates=0, third_stage_fit_updates=0,
            recovery_updates=1500, live_suffix_VAE_updates=40000, full8_evaluations=1, LongBench21_evaluations=1,
            additional_recovery_R_updates=0),
        boundary='Only one already-budgeted C0 suffix and complete-nine. Common prerequisite stops at its unchanged raw2 boundary. No C1, Muon, search, reservation or retry.')
    output.mkdir(parents=True)
    s.c.write(output / 'plan.json', plan)
    print(json.dumps(dict(status='prepared_fresh_C0_single_sequence', plan=str(output / 'plan.json'),
        common_driver=run['pid'], common_shell=common_shell_pid, formal_argv=formal['argv'], evaluation_argv=quality['argv'])), flush=True)


def run(plan_path):
    existing.cpu_environment()
    plan_path = plan_path.resolve(strict=True)
    plan = s.c.read(plan_path)
    s.c.require(plan['schema'] == 'fresh_C0_fixed2_single_sequence_v1', 'Use the reviewed single C0 sequence.')
    existing.check_pins(plan['source_sha256'])
    output = Path(plan['output'])
    record_path = output / 'run.json'
    s.c.require(not record_path.exists(), 'This one-shot sequence has already been started.')
    with Path(plan['lock_file']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s.c.require(not record_path.exists(), 'This one-shot sequence already ran before the lock was acquired.')
        record = dict(status='validating_completed_common' if plan['common'].get('completed_before_plan')
            else 'waiting_actual_common_shell', pid=os.getpid(), start_ticks=existing.proc_start(os.getpid()),
            started_unix=time.time(), command=[sys.executable, *sys.argv], plan_sha256=s.c.sha(plan_path), stages={})
        s.c.write(record_path, record)
        try:
            parent = plan['common']
            record['common_wait'] = [] if parent.get('completed_before_plan') else [
                existing.wait_identity(parent['driver_pid'], parent['driver_start_ticks']),
                existing.wait_identity(parent['shell_pid'], parent['shell_start_ticks'])]
            s.c.require(Path(parent['exit_file']).read_text().strip() == '0'
                and s.c.sha(Path(parent['directory']) / 'request.json') == parent['request_sha256'],
                'The actual common shell did not finish exit0 or its immutable request changed.')
            binding = s.bind(Path(parent['directory']))
            s.c.require(binding['source_common_pid'] == parent['driver_pid']
                and binding['source_common_start_ticks'] == parent['driver_start_ticks'], 'The common terminal report belongs to another process.')
            record.update(status='common_complete_C0_ready', source_binding=binding)
            s.c.write(record_path, record)
            existing.check_pins(plan['source_sha256'])

            def changed(stage):
                record['stages'][stage['name']] = stage
                record['status'] = stage['name'] + '_' + stage['status']
                s.c.write(record_path, record)

            record['formal_resource_check'] = existing.check_resources(plan['formal']['gpus'])
            s.c.write(record_path, record)
            existing.launch_child(plan['formal'], changed)
            from .evaluate import setup
            from experiments.adaptive_class_evaluation_20261030 import bindings as b
            setup(SUPPORT)
            model, _meta = b.source(Path(plan['formal']['output']) / 'final_model', Path(plan['formal']['report']))
            record['C0_final_binding'] = model
            record['evaluation_resource_check'] = existing.check_resources(plan['evaluation']['gpus'])
            s.c.write(record_path, record)
            existing.launch_child(plan['evaluation'], changed)
            record['complete_nine'] = existing.validate_evaluation(plan['evaluation']['output'], model['checkpoint_id'])
            existing.check_pins(plan['source_sha256'])
            record.update(status='completed_C0_and_complete_nine', exit_code=0)
        except BaseException:
            record.update(status='failed', exit_code=1, error=traceback.format_exc())
            raise
        finally:
            record['finished_unix'] = time.time()
            s.c.write(record_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--common-shell-pid', type=int)
    source.add_argument('--completed-common', action='store_true',
        help='Consume the real completed formal common source and its exit0 without waiting on old PIDs.')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.prepare:
        s.c.require(args.run is None and (args.common_shell_pid is not None or args.completed_common),
            'Preparation needs the live common shell PID or explicit completed-common source.')
        prepare(args.prepare, args.common_shell_pid, args.completed_common)
    else:
        s.c.require(args.run is not None and args.common_shell_pid is None and not args.completed_common,
            'Run the reviewed one-shot plan.')
        run(args.run)


if __name__ == '__main__':
    main()
