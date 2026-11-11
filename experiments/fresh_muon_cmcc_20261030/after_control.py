"""Attach once to the bound C0 v3 formal exit, then run its budgeted Muon32."""
import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import struct
import sys
import time
import traceback

from experiments.cmcc_runtime_20261030 import sequence as executor
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO

ROOT = ARTIFACTS/'fresh_muon_cmcc_20261030'
DIRECTORY = ROOT/'after_control_C0_v3'
CONTROL_PLAN = ARTIFACTS/'fresh_fixed2_cmcc_20261030/single_20261030_v3/plan.json'
TARGET = ROOT/'C0_H96_Muon32'
CONTROLLER = (288378, 10671306)
WORKER = (288496, 10671964)
IN_CLOSE_WRITE, IN_MOVED_TO = 0x8, 0x80
IN_DELETE_SELF, IN_MOVE_SELF, IN_Q_OVERFLOW, IN_IGNORED = 0x400, 0x800, 0x4000, 0x8000
require, read, write, sha = executor.require, executor.read, executor.write, executor.sha
BUDGET = dict(C0_recovery_updates=32, complete_nine_evaluations=1, extra_C0_training_updates=0,
    seed=137, schedule_steps=500, group_indices=list(range(32)), training_gpus=[5], evaluation_gpus=[5, 6, 7])


def pipeline_request():
    return dict(name='C0_H96_Muon32', argv=[str(PYTHON), '-B', '-u', '-m',
        'experiments.fresh_muon_cmcc_20261030.pipeline', '--control-plan', str(CONTROL_PLAN),
        '--output', str(TARGET), '--gpu', '5'], environment=executor.environment(''),
        log=str(DIRECTORY/'pipeline.log'), exit_file=str(DIRECTORY/'pipeline.exit'))


def process_identity(pid):
    """Read one actual process and reject a PID change across the /proc reads."""
    first = executor.proc_start(pid)
    require(first is not None, f'Process {pid} is absent; no live identity can be prepared.')
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    argv = [os.fsdecode(value) for value in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if value]
    require(executor.proc_start(pid) == first and int(fields[19]) == first and argv,
            f'Process {pid} exited or changed while binding its command.')
    return dict(pid=pid, start_ticks=first, parent_pid=int(fields[1]), argv=argv)


def open_identity(identity):
    """Open the existing pidfd API without attaching to a recycled numeric PID."""
    actual = executor.proc_start(identity['pid'])
    if actual is None:
        return None
    require(actual == identity['start_ticks'], 'A bound C0 process PID was reused.')
    try:
        descriptor = executor.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return None
    try:
        require(executor.proc_start(identity['pid']) in (None, identity['start_ticks']),
                'A bound C0 process changed while opening its pidfd.')
        if not select.select([descriptor], [], [], 0)[0]:
            try:
                actual_identity = process_identity(identity['pid'])
            except (FileNotFoundError, ValueError):
                require(select.select([descriptor], [], [], 0)[0],
                        'A live process identity changed while attaching its pidfd.')
            else:
                require(actual_identity == identity, 'A live bound process command or parent identity changed.')
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


class PublicationEvents:
    """One directory watch, with nonblocking drains and blocking kernel waits."""
    def __init__(self, directory):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes, libc.inotify_init1.restype = [ctypes.c_int], ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_add_watch.restype = ctypes.c_int
        self.fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        if self.fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        self.wd = libc.inotify_add_watch(self.fd, os.fsencode(directory),
            IN_CLOSE_WRITE | IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF)
        if self.wd < 0:
            error = ctypes.get_errno()
            self.close()
            raise OSError(error, os.strerror(error), str(directory))

    def close(self):
        os.close(self.fd)

    def drain(self):
        events = []
        while True:
            try:
                payload = os.read(self.fd, 65536)
            except BlockingIOError:
                return events
            require(payload, 'The C0 publication watch unexpectedly reached EOF.')
            offset = 0
            while offset < len(payload):
                wd, mask, _cookie, length = struct.unpack_from('iIII', payload, offset)
                offset += struct.calcsize('iIII')
                name = os.fsdecode(payload[offset:offset+length].split(b'\0', 1)[0])
                offset += length
                require(not mask & (IN_DELETE_SELF | IN_MOVE_SELF | IN_Q_OVERFLOW | IN_IGNORED),
                        'The bound C0 publication directory changed or its event queue overflowed.')
                require(wd == self.wd, 'Unexpected C0 publication watch descriptor.')
                events.append((mask, name))


def formal_record(parent):
    record = read(parent['record_path'])
    require((record['pid'], record['start_ticks']) ==
            (parent['controller']['pid'], parent['controller']['start_ticks'])
        and record['plan_sha256'] == parent['control_plan_sha256'],
        'The C0 controller record belongs to another process or plan.')
    stage = record['stages'][parent['stage_name']]
    require((stage['pid'], stage['start_ticks']) == (parent['worker']['pid'], parent['worker']['start_ticks'])
        and stage['argv'] == parent['worker']['argv']
        and stage['exit_file'] == parent['exit_file']
        and stage['status'] in ('running', 'exited'),
        'The actual C0 formal stage identity, command, exit path or state differs.')
    return stage


def wait_formal(parent, observe):
    """Require worker exit and a closed successful publication, not file presence."""
    exit_path, record_path = Path(parent['exit_file']), Path(parent['record_path'])
    require(exit_path.parent == record_path.parent and exit_path.parent.is_dir(),
            'Watch the existing bound C0 control directory only.')
    watcher = PublicationEvents(exit_path.parent)
    descriptors = {}
    try:
        # Watch installation precedes both the terminal read and pidfd checks.
        observe('publication_watch_installed', dict(directory=str(exit_path.parent)))
        for name in ('worker', 'controller'):
            descriptor = open_identity(parent[name])
            if descriptor is not None:
                descriptors[name] = descriptor
        done = {name: name not in descriptors for name in ('worker', 'controller')}
        poller = select.poll()
        poller.register(watcher.fd, select.POLLIN)
        for descriptor in descriptors.values():
            poller.register(descriptor, select.POLLIN)
        closed_event = False
        while True:
            for mask, name in watcher.drain():
                if name == exit_path.name and mask & IN_CLOSE_WRITE:
                    closed_event = True
                    observe('formal_exit_close_observed', dict(exit_file=str(exit_path)))
            for name, descriptor in list(descriptors.items()):
                if not done[name] and select.select([descriptor], [], [], 0)[0]:
                    done[name] = True
                    poller.unregister(descriptor)
                    observe(name+'_pidfd_exit_observed', parent[name])
            stage = formal_record(parent)
            terminal = stage['status'] == 'exited'
            if done['worker'] and (terminal or closed_event):
                code = int(exit_path.read_text(encoding='utf-8').strip())
                require(not terminal or type(stage['exit_code']) is int and stage['exit_code'] == code,
                        'The closed C0 formal exit file and same-worker terminal record disagree.')
                require(code == 0, f'The bound C0 formal stage exited {code}; Muon was not started.')
                result = dict(worker=parent['worker'], controller=parent['controller'], exit_file=str(exit_path),
                    exit_code=code, worker_exit_observed=True,
                    closed_publication='same_worker_terminal_record' if terminal else 'inotify_close_write',
                    controller_still_active=not done['controller'], formal_record_status=stage['status'])
                observe('actual_C0_formal_exit0_published', result)
                return result
            require(not done['controller'],
                    'The C0 controller exited without a closed successful same-worker formal exit; stop the dependency.')
            # No sleep, timeout, log polling or retry. Ready pidfds are removed.
            poller.poll()
    finally:
        watcher.close()
        for descriptor in descriptors.values():
            os.close(descriptor)


@contextmanager
def claim_once(directory, record):
    path = Path(directory)/'run.json'
    require(not path.exists(), 'This fixed one-shot continuation already started; no retry or automatic resume.')
    with (Path(directory)/'after_control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        yield path


def prepare():
    executor.cpu_environment()
    require(not os.path.lexists(DIRECTORY) and not os.path.lexists(TARGET),
            'The fixed continuation plan or its recovery target already exists; no duplicate preparation.')
    control = read(CONTROL_PLAN)
    require(control['schema'] == 'fresh_C0_fixed2_single_sequence_v1'
        and Path(control['output']).resolve(strict=True) == CONTROL_PLAN.parent
        and control['formal']['name'] == 'C0_fixed2' and control['formal']['gpus'] == ['5']
        and control['evaluation']['gpus'] == ['6', '7'], 'Require the bound actual C0 v3 plan.')
    record_path = CONTROL_PLAN.parent/'run.json'
    run = read(record_path)
    stage = run['stages']['C0_fixed2']
    require((run['pid'], run['start_ticks']) == CONTROLLER
        and (stage['pid'], stage['start_ticks']) == WORKER
        and run['status'] == 'C0_fixed2_running' and stage['status'] == 'running'
        and run['plan_sha256'] == sha(CONTROL_PLAN)
        and not Path(control['formal']['exit_file']).exists(),
        'Bind only the actual active C0 v3 controller/formal pair, before its successful exit publication.')
    controller, worker = process_identity(run['pid']), process_identity(stage['pid'])
    require(controller['start_ticks'] == CONTROLLER[1] and worker['start_ticks'] == WORKER[1]
        and worker['parent_pid'] == controller['pid'] and worker['argv'] == control['formal']['argv'] == stage['argv']
        and controller['argv'] == [str(PYTHON), '-B', '-u', '-m',
            'experiments.fresh_fixed2_cmcc_20261030.sequence', '--run', str(CONTROL_PLAN)],
        'The actual C0 commands or worker/controller parent relationship differ from this one fixed dependency.')
    parent = dict(controller=controller, worker=worker, record_path=str(record_path),
        control_plan_sha256=sha(CONTROL_PLAN), stage_name='C0_fixed2', exit_file=control['formal']['exit_file'])
    formal_record(parent)
    executor.check_pins(control['source_sha256'])
    from . import bindings as b
    recovery_pins = {name: item['sha256'] for name, item in b.code_bindings().items()}
    pins = dict(control['source_sha256'])
    for name, digest in recovery_pins.items():
        require(name not in pins or pins[name] == digest, 'Control and recovery source pins disagree: '+name)
        pins[name] = digest
    pins[str(Path(executor.__file__).resolve())] = sha(executor.__file__)
    executor.check_pins(pins)
    pipeline = pipeline_request()
    plan = dict(schema='fresh_C0_v3_after_formal_Muon32_v1', created_unix=time.time(),
        control_plan=str(CONTROL_PLAN), parent=parent, source_sha256=pins, target=str(TARGET),
        pipeline=pipeline, budget=BUDGET, automatic_retry=False, automatic_resume=False,
        boundary='Wait only for this formal worker and closed exit0 publication. Existing pipeline validates real C0, '
            'trains on GPU5 independently of C0 evaluation, then waits on its controller pidfd before full9. '
            'No source/model fabrication, search, retry, reservation or C0 modification.')
    DIRECTORY.mkdir(parents=True)
    write(DIRECTORY/'plan.json', plan)
    print(json.dumps(dict(status='prepared_single_C0_v3_Muon32_continuation', plan=str(DIRECTORY/'plan.json'),
        controller=controller, worker=worker, target=str(TARGET), pipeline_argv=pipeline['argv'])), flush=True)


def run():
    executor.cpu_environment()
    plan_path = DIRECTORY/'plan.json'
    plan = read(plan_path)
    require(plan['schema'] == 'fresh_C0_v3_after_formal_Muon32_v1'
        and plan['target'] == str(TARGET) and plan['control_plan'] == str(CONTROL_PLAN)
        and (plan['parent']['controller']['pid'], plan['parent']['controller']['start_ticks']) == CONTROLLER
        and (plan['parent']['worker']['pid'], plan['parent']['worker']['start_ticks']) == WORKER
        and plan['pipeline'] == pipeline_request() and plan['budget'] == BUDGET
        and plan['automatic_retry'] is plan['automatic_resume'] is False
        and sha(CONTROL_PLAN) == plan['parent']['control_plan_sha256'], 'The fixed prepared dependency identity changed.')
    record = dict(schema=plan['schema'], status='waiting_bound_C0_formal_publication', pid=os.getpid(),
        start_ticks=executor.proc_start(os.getpid()), started_unix=time.time(), argv=[sys.executable, *sys.argv],
        plan_sha256=sha(plan_path), target=str(TARGET), events=[], stages={})
    with claim_once(DIRECTORY, record) as record_path:
        def observe(event, evidence):
            row = dict(event=event, unix=time.time(), evidence=evidence)
            record['events'].append(row)
            write(record_path, record)
            print(json.dumps(row), flush=True)

        code = 1
        try:
            require(not os.path.lexists(TARGET), 'The recovery target already exists; never overwrite, resume or retry.')
            executor.check_pins(plan['source_sha256'])
            record['formal_exit'] = wait_formal(plan['parent'], observe)
            require(not os.path.lexists(TARGET), 'The recovery target appeared while waiting; stop without launching.')
            require(sha(CONTROL_PLAN) == plan['parent']['control_plan_sha256'], 'The immutable C0 control plan changed.')
            executor.check_pins(plan['source_sha256'])
            record['status'] = 'starting_existing_C0_Muon32_pipeline'
            write(record_path, record)

            def changed(stage):
                record['stages'][stage['name']] = stage
                record['status'] = stage['name']+'_'+stage['status']
                write(record_path, record)

            executor.launch_child(plan['pipeline'], changed)
            code = 0
            record.update(status='completed_existing_C0_Muon32_pipeline', exit_code=0)
        except BaseException:
            record.update(status='failed', exit_code=1, error=traceback.format_exc())
            raise
        finally:
            record['finished_unix'] = time.time()
            write(record_path, record)
            (DIRECTORY/'run.exit').write_text(str(code)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true', help='Bind the active actual C0 v3 pair into the fixed plan directory.')
    mode.add_argument('--run', action='store_true', help='Run the reviewed fixed single continuation once.')
    args = parser.parse_args()
    prepare() if args.prepare else run()


if __name__ == '__main__':
    main()
