"""Four fixed overnight evaluations, with bounded process-exit dependencies."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.night_sequence_20261026 import queue_wait as q

HERE = Path(__file__).resolve().parent
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
TOPIC = ROOT / 'result/qwen3_8b_mmlu65_20261004'
OUTPUT = TOPIC / 'overnight_goal_20261029'
MANIFEST = OUTPUT / 'night_manifest.json'
DEADLINE = datetime(2026, 10, 30, 10, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()


def definitions():
    cat = Path('/data/star/VAELLM/cat_structure_full_20261029')
    nodes = Path('/data/star/VAELLM/cat_category_selection_20261029')
    muon = Path('/data/star/VAELLM/g09_recovery_20261029/muon_opd32')
    owners = {
        'A': (425645, cat / 'A/run.exit', 'run_A.sh'),
        'B': (425650, cat / 'B/run.exit', 'run_B.sh'),
        'q64': (560500, TOPIC / 'g09_recovery_20261029/q64_full9.exit', 'evaluate_q64.py'),
        'nodes': (584652, nodes / 'pair_full8.exit', 'evaluate.py'),
        'Muon': (557588, muon / 'formal32.exit', 'muon/run.py'),
        'assembly': (604543, OUTPUT / 'G04_assembly.exit', 'assemble.py'),
    }
    jobs = {
        'A': dict(gpus=[3, 6], data='A', resources=['nodes'],
                  command=[str(PYTHON), '-B', '-u', str(HERE / 'evaluate_cat.py'), '--arm', 'A', '--gpus', '3,6']),
        'B': dict(gpus=[4, 0], data='B', resources=['q64'],
                  command=[str(PYTHON), '-B', '-u', str(HERE / 'evaluate_cat.py'), '--arm', 'B', '--gpus', '4,0']),
        'Muon': dict(gpus=[7, 1, 2], data='Muon', resources=['q64'],
                     command=[str(PYTHON), '-B', '-u', str(HERE / 'evaluate_muon.py'), '--gpus', '7,1,2']),
        'G04': dict(gpus=[5], data='assembly', resources=[],
                    command=[str(PYTHON), '-B', '-u', str(HERE / 'evaluate_g04.py'), '--gpus', '5']),
    }
    return owners, jobs


def pins():
    paths = [Path(__file__), HERE / 'evaluate_cat.py', HERE / 'evaluate_muon.py',
             HERE / 'evaluate_g04.py', Path(q.__file__)]
    return {str(path): q.sha256(path) for path in paths}


def wait_bound(owner, deadline):
    if owner.get('completed_before_prepare'):
        return
    actual = q.process_start(owner['pid'])
    if actual is None:
        return
    q.require(actual == owner['start_ticks'], 'Predecessor PID was reused: ' + str(owner['pid']))
    remaining = deadline - time.time()
    q.require(remaining > 0, 'Morning deadline reached before predecessor exit; no new stage.')
    try:
        descriptor = q.exit_descriptor(owner['pid'])
    except ProcessLookupError:
        return
    try:
        q.require(q.process_start(owner['pid']) in (None, owner['start_ticks']), 'Predecessor identity changed.')
        ready, _, _ = select.select([descriptor], [], [], remaining)
        q.require(bool(ready), 'Morning deadline reached while waiting; predecessor is left untouched.')
    finally:
        os.close(descriptor)


def validate_wait():
    # Real child processes exercise exit events, identity rejection and timeout.
    child = subprocess.Popen([str(PYTHON), '-c', 'import time; time.sleep(.3)'])
    owner = dict(pid=child.pid, start_ticks=q.process_start(child.pid))
    try:
        try:
            wait_bound({**owner, 'start_ticks': owner['start_ticks'] + 1}, time.time() + 2)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('PID identity mismatch was not rejected.')
        try:
            wait_bound(owner, time.time() + .01)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Bounded wait did not time out.')
        q.require(child.poll() is None, 'A wait timeout must not stop its predecessor.')
        wait_bound(owner, time.time() + 2)
        q.require(child.wait() == 0, 'Real exit-event child failed.')
        wait_bound(owner, time.time() + 1)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait()
    return dict(real_exit_event=True, reused_pid_rejected=True,
                timeout_preserves_predecessor=True, already_exited_accepted=True)


def prepare():
    q.require(Path(sys.executable).resolve() == PYTHON.resolve(), 'Activate bitvae in the caller shell.')
    q.require(not MANIFEST.exists(), 'Never replace the fixed night manifest.')
    owners, jobs = definitions()
    assigned = [gpu for job in jobs.values() for gpu in job['gpus']]
    q.require(sorted(assigned) == list(range(8)), 'Night jobs must have disjoint explicit GPU assignments.')
    bindings = {}
    for name, (pid, terminal, token) in owners.items():
        start = q.process_start(pid)
        state = None if start is None else Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        if start is None or state == 'Z':
            q.require(terminal.is_file(), 'Missing predecessor and terminal record: ' + name)
            code = int(terminal.read_text().strip())
            bindings[name] = dict(pid=pid, start_ticks=start, terminal=str(terminal), observed_exit_code=code,
                                  completed_before_prepare=True)
        else:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
            q.require(token in command, 'The actual predecessor command differs: ' + name)
            bindings[name] = dict(pid=pid, start_ticks=start, terminal=str(terminal), observed_exit_code=None)
    record = dict(schema='fixed_four_full9_overnight_v1', prepared_unix=time.time(), deadline_unix=DEADLINE,
        deadline_local='2026-10-30 10:00 Asia/Shanghai', owners=bindings, jobs=jobs, source_sha256=pins(),
        wait_validation=validate_wait(), automatic_retry=False, polling=False, occupancy=False,
        evaluation_timeout_seconds=90000,
        morning_policy='Stop waiting for unready predecessors at10:00; an evaluation already running keeps its25h execution limit, not a10:00 kill.',
        data_dependency='Own real training/export must exit0; evaluator verifies the actual endpoint again.',
        resource_dependency='Wait for the bound owner to exit, then require the assigned GPUs free; unrelated quality or execution failure is not a data dependency.',
        quality_not_met_is_execution_failure=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    q.write_json(MANIFEST, record)
    print(dict(status='prepared_fixed_four_evaluations', manifest=str(MANIFEST), jobs=list(jobs)), flush=True)


def run_job(name):
    manifest = q.read_json(MANIFEST)
    q.require(manifest['schema'] == 'fixed_four_full9_overnight_v1' and pins() == manifest['source_sha256'],
              'Prepared night implementation changed.')
    _, expected = definitions()
    q.require(manifest['jobs'] == expected, 'The fixed night task list changed.')
    job = manifest['jobs'][name]
    directory = OUTPUT / 'jobs' / name
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'report.json'
    report = dict(status='waiting_predecessors', pid=os.getpid(), start_ticks=q.process_start(os.getpid()),
        started_unix=time.time(), job=name, gpus=job['gpus'], manifest_sha256=q.sha256(MANIFEST),
        automatic_retry=False, polling=False, occupancy=False, dependencies={})
    q.write_json(path, report)
    try:
        for key in [job['data'], *job['resources']]:
            owner = manifest['owners'][key]
            report['waiting_for'] = key
            q.write_json(path, report)
            wait_bound(owner, manifest['deadline_unix'])
            terminal = Path(owner['terminal'])
            code = int(terminal.read_text().strip()) if terminal.is_file() else None
            report['dependencies'][key] = dict(role='data' if key == job['data'] else 'resource',
                pid=owner['pid'], start_ticks=owner['start_ticks'], terminal=str(terminal), exit_code=code)
            if key == job['data']:
                q.require(code == 0, 'Required training/export failed or has no actual terminal0: ' + key)
        q.require(time.time() < manifest['deadline_unix'], 'Morning deadline reached before launching evaluation.')
        q.require(pins() == manifest['source_sha256'], 'Bound night sources changed while waiting.')
        report['resource_checks'] = [q.free_gpu(gpu) for gpu in job['gpus']]
        report.update(status='running_evaluation', waiting_for=None)
        seconds = manifest['evaluation_timeout_seconds']
        command = ['timeout', '--signal=TERM', '--kill-after=60s', str(seconds) + 's', *job['command']]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        q.run_stage(command, cwd=ROOT, environment=environment, log=directory / 'evaluation.log',
            report=report, report_path=path, name='complete_full9')
        current_path = OUTPUT / 'evaluation' / name / 'current_goal_result.json'
        current = q.read_json(current_path)
        acceptance_path = OUTPUT / 'evaluation' / name / 'transfer_acceptance.json'
        acceptance = q.read_json(acceptance_path)
        q.require(acceptance['quality_target_evaluated'] and acceptance['protocol']['strict_score_drop_bound']
            and acceptance['protocol']['maximum_absolute_drop_points'] == .9
            and acceptance['protocol']['max_linear_bpw'] == 2.6, 'Current complete-nine acceptance is absent.')
        report.update(status='complete', exit_code=0, current_result=str(current_path),
            current_result_sha256=q.sha256(current_path), acceptance=str(acceptance_path),
            acceptance_sha256=q.sha256(acceptance_path), score=acceptance['compressed_nine_task_score'],
            passed=acceptance['passed'], checkpoint_id=acceptance['checkpoint_id'])
    except BaseException as error:
        report.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_unix'] = time.time()
        q.write_json(path, report)
        (directory / 'exit').write_text(str(report.get('exit_code', 1)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--job', choices=('A', 'B', 'Muon', 'G04'))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run_job(args.job)
