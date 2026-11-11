"""Real CPU process/file-event tests; never construct a C0 model or run Muon."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from . import after_control as a


PRODUCER = r'''
import json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
def started(pid):return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19])
worker=subprocess.Popen([sys.executable,"-B","-u","-c","import sys; sys.stdin.buffer.read(1)"],stdin=subprocess.PIPE)
record=dict(pid=os.getpid(),start_ticks=started(os.getpid()),plan_sha256=sys.argv[2],stages={"CPU_formal":dict(name="CPU_formal",pid=worker.pid,start_ticks=started(worker.pid),argv=[sys.executable,"-B","-u","-c","import sys; sys.stdin.buffer.read(1)"],exit_file=str(root/"formal.exit"),status="running")})
def save():
    path=root/"run.json.writing"; path.write_text(json.dumps(record));path.replace(root/"run.json")
def tell(event):print(json.dumps(dict(event=event)),flush=True)
save();tell("ready")
try:
    command=sys.stdin.readline().strip()
    worker.stdin.close();worker.wait();tell("worker_exited")
    if command=="die":raise SystemExit(0)
    while True:
        command=sys.stdin.readline().strip()
        if not command or command=="stop":break
        if command.startswith("open "):
            output=(root/"formal.exit").open("w");output.write(command.split()[1]+"\n");output.flush();tell("exit_open")
        elif command=="close":
            output.close();tell("exit_closed")
        elif command=="record":
            record["stages"]["CPU_formal"].update(status="exited",exit_code=int((root/"formal.exit").read_text().strip()))
            save();tell("record_published")
        else:raise ValueError(command)
finally:
    if worker.poll() is None:worker.stdin.close();worker.wait()
'''


class Producer:
    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='after_control_CPU_')
        self.root = Path(self.temporary.name)
        (self.root/'plan.json').write_text('{"scope":"real CPU event protocol only"}\n', encoding='utf-8')
        self.process = subprocess.Popen([sys.executable, '-B', '-u', '-c', PRODUCER,
            str(self.root), a.sha(self.root/'plan.json')], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.event('ready')
        record = a.read(self.root/'run.json')
        self.parent = dict(controller=a.process_identity(self.process.pid),
            worker=a.process_identity(record['stages']['CPU_formal']['pid']),
            record_path=str(self.root/'run.json'), control_plan_sha256=a.sha(self.root/'plan.json'),
            stage_name='CPU_formal', exit_file=str(self.root/'formal.exit'))
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        return self

    def event(self, expected):
        if not select.select([self.process.stdout], [], [], 5)[0]:
            raise AssertionError('CPU producer did not publish '+expected)
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError('CPU producer exited: '+self.process.stderr.read())
        if json.loads(line)['event'] != expected:
            raise AssertionError('Unexpected CPU producer event: '+line)

    def command(self, value, expected):
        self.process.stdin.write(value+'\n')
        self.process.stdin.flush()
        self.event(expected)

    def waiting(self, parent=None, *, consume=False):
        installed = threading.Event()
        self.events = []

        def observed(event, evidence):
            self.events.append((event, evidence))
            if event == 'publication_watch_installed':
                installed.set()

        def wait_and_consume():
            result = a.wait_formal(parent or self.parent, observed)
            if consume:
                subprocess.run([sys.executable, '-B', '-c',
                    'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("real CPU consumer\\n")',
                    str(self.root/'consumer_started')], check=True, stdin=subprocess.DEVNULL)
            return result

        self.future = self.pool.submit(wait_and_consume)
        if not installed.wait(5):
            raise AssertionError('The actual inotify watch was not installed.')
        return self.future

    def __exit__(self, *_args):
        if self.process.poll() is None:
            self.process.stdin.close()
        self.process.wait(timeout=5)
        self.pool.shutdown(wait=True)
        if self.process.returncode != 0:
            error = self.process.stderr.read()
            raise AssertionError('CPU controller failed: '+error)
        self.process.stdout.close()
        self.process.stderr.close()
        self.temporary.cleanup()


class AfterControlTests(unittest.TestCase):
    def test_delayed_closed_exit_returns_before_controller_finishes(self):
        with Producer() as p:
            future = p.waiting(consume=True)
            p.command('worker', 'worker_exited')
            self.assertFalse(future.done())
            p.command('open 0', 'exit_open')
            self.assertTrue(Path(p.parent['exit_file']).is_file())
            self.assertFalse(future.done())
            p.command('close', 'exit_closed')
            result = future.result(timeout=5)
            self.assertEqual(result['closed_publication'], 'inotify_close_write')
            self.assertEqual(result['formal_record_status'], 'running')
            self.assertTrue(result['controller_still_active'])
            self.assertIsNone(p.process.poll())
            self.assertEqual((p.root/'consumer_started').read_text(), 'real CPU consumer\n')

    def test_controller_exit_without_formal_publication_is_rejected(self):
        with Producer() as p:
            future = p.waiting()
            p.command('die', 'worker_exited')
            with self.assertRaisesRegex(ValueError, 'without a closed successful'):
                future.result(timeout=5)
            self.assertFalse(Path(p.parent['exit_file']).exists())

    def test_nonzero_closed_exit_never_reaches_a_CPU_consumer(self):
        with Producer() as p:
            future = p.waiting(consume=True)
            p.command('worker', 'worker_exited')
            p.command('open 7', 'exit_open')
            p.command('close', 'exit_closed')
            with self.assertRaisesRegex(ValueError, 'exited 7; Muon was not started'):
                future.result(timeout=5)
            self.assertFalse((p.root/'consumer_started').exists())

    def test_completed_same_worker_terminal_record_is_accepted(self):
        with Producer() as p:
            p.command('worker', 'worker_exited')
            p.command('open 0', 'exit_open')
            p.command('close', 'exit_closed')
            p.command('record', 'record_published')
            p.process.stdin.close()
            p.process.wait(timeout=5)
            result = p.waiting().result(timeout=5)
            self.assertEqual(result['closed_publication'], 'same_worker_terminal_record')
            self.assertFalse(result['controller_still_active'])

    def test_atomic_record_publication_covers_close_before_watch(self):
        with Producer() as p:
            p.command('worker', 'worker_exited')
            p.command('open 0', 'exit_open')
            p.command('close', 'exit_closed')
            future = p.waiting()
            self.assertFalse(future.done())
            p.command('record', 'record_published')
            result = future.result(timeout=5)
            self.assertEqual(result['closed_publication'], 'same_worker_terminal_record')
            self.assertTrue(result['controller_still_active'])

    def test_terminal_record_with_foreign_worker_identity_is_rejected(self):
        with Producer() as p:
            p.command('worker', 'worker_exited')
            p.command('open 0', 'exit_open')
            p.command('close', 'exit_closed')
            p.command('record', 'record_published')
            foreign = deepcopy(p.parent)
            foreign['worker']['start_ticks'] += 1
            with self.assertRaisesRegex(ValueError, 'formal stage identity'):
                p.waiting(foreign).result(timeout=5)

    def test_live_PID_start_and_command_mismatch_are_rejected(self):
        with Producer() as p:
            changed = deepcopy(p.parent['worker'])
            changed['start_ticks'] += 1
            with self.assertRaisesRegex(ValueError, 'PID was reused'):
                a.open_identity(changed)
            changed = deepcopy(p.parent['worker'])
            changed['argv'][-1] = 'unrelated CPU command'
            with self.assertRaisesRegex(ValueError, 'command or parent identity changed'):
                a.open_identity(changed)

    def test_completed_record_and_exit_code_disagreement_is_rejected(self):
        with Producer() as p:
            p.command('worker', 'worker_exited')
            p.command('open 0', 'exit_open')
            p.command('close', 'exit_closed')
            p.command('record', 'record_published')
            Path(p.parent['exit_file']).write_text('9\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'disagree'):
                p.waiting().result(timeout=5)

    def test_flock_and_existing_run_each_prevent_a_second_start(self):
        with tempfile.TemporaryDirectory(prefix='after_control_claim_CPU_') as temporary:
            root = Path(temporary)
            with (root/'after_control.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    with a.claim_once(root, dict(scope='CPU claim only')):
                        self.fail('A concurrent claim entered.')
            self.assertFalse((root/'run.json').exists())
            with a.claim_once(root, dict(scope='CPU claim only', status='started')):
                self.assertEqual(a.read(root/'run.json')['status'], 'started')
            with self.assertRaisesRegex(ValueError, 'already started'):
                with a.claim_once(root, dict(scope='second CPU claim')):
                    self.fail('A repeated claim entered.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a.executor.cpu_environment()
    output = args.output.resolve()
    a.require(not output.exists(), 'Choose a new CPU protocol report directory.')
    output.mkdir(parents=True)
    unchanged = [a.REPO/'experiments/fresh_muon_cmcc_20261030'/name for name in
        ('pipeline.py', 'bindings.py', 'prepare.py', 'run.py', 'fresh.py', 'evaluate.py')]
    unchanged += [a.REPO/'experiments/muon_compact_cmcc_20261030'/name for name in
        ('bindings.py', 'fit.py', 'inputs.py', 'lifecycle.py', 'optimizer.py', 'rollout.py', 'motion.py')]
    unchanged += [a.REPO/'experiments/fresh_fixed2_cmcc_20261030'/name
                  for name in ('sequence.py', 'run_pipeline.py', 'sources.py')]
    unchanged += [Path(a.executor.__file__)]
    before = {str(path): a.sha(path) for path in unchanged}
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AfterControlTests))
    a.require(before == {str(path): a.sha(path) for path in unchanged}, 'An original C0/Muon helper changed during CPU tests.')
    report = dict(schema='after_control_CPU_protocol_validation_v1', status='passed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun, failures=[str(test) for test, _error in result.failures],
        errors=[str(test) for test, _error in result.errors], elapsed_seconds=time.monotonic()-started,
        python=sys.executable, argv=[sys.executable, *sys.argv], CUDA_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch_imported='torch' in sys.modules, original_helpers_unchanged=before,
        implementation_sha256={str(path): a.sha(path) for path in (Path(a.__file__), Path(__file__))},
        real_CPU_children_and_kernel_pidfd_inotify=True, temporary_fixtures_removed=True,
        actual_CPU_consumer_gated_by_successful_closed_publication=True,
        fixed_artifact_plan_prepared=False, GPU_started=False, actual_pipeline_called=False,
        scope='Process identity, closed exit publication, controller liveness, atomic records and one-shot locks only; no C0/model or GPU substitute.')
    a.write(output/'validation.json', report)
    print(json.dumps(dict(status=report['status'], tests_run=result.testsRun, report=str(output/'validation.json'))), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
