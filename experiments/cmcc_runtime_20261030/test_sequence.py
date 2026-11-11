"""CPU regressions using real acceptance evidence and actual Linux child processes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from . import sequence as s

REAL_ACCEPTANCE = None
REAL_ACCEPTANCE_EXIT = None


class SequenceTests(unittest.TestCase):
    def test_actual_complete_not_met_is_a_valid_result(self):
        report = s.read(REAL_ACCEPTANCE)
        self.assertEqual(report['status'], 'not_met')
        actual = s.validate_acceptance(REAL_ACCEPTANCE, REAL_ACCEPTANCE_EXIT, report['checkpoint_id'])
        self.assertFalse(actual['passed'])
        self.assertEqual(actual['quality_status'], 'not_met')
        self.assertAlmostEqual(actual['nine_task_score'], report['compressed_nine_task_score'], places=12)
        self.assertEqual(set(actual['task_scores']), set(s.TASKS))
        with self.assertRaises(ValueError):
            s.validate_acceptance(REAL_ACCEPTANCE, REAL_ACCEPTANCE_EXIT, 'a-foreign-UUID')

    def test_existing_incomplete_gpu_report_is_rejected(self):
        # Real failed H192 report: full8 succeeded, full21 never ran.
        original = s.H192 / 'evaluation/Muon'
        with self.assertRaises(ValueError):
            s.validate_evaluation(original, '74329fcf-c762-4498-9f41-218df9645b3f')

    def test_pidfd_observes_actual_exit_and_rejects_reused_identity(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.3)'], stdin=subprocess.DEVNULL)
        started = s.proc_start(child.pid)
        self.assertIsNotNone(started)
        try:
            with self.assertRaises(ValueError):
                s.wait_identity(child.pid, started + 1)
            before = time.monotonic()
            observed = s.wait_identity(child.pid, started)
            self.assertEqual(observed['state'], 'pidfd_exit_observed')
            self.assertGreater(time.monotonic() - before, .1)
        finally:
            child.wait()
        self.assertEqual(s.wait_identity(child.pid, started)['state'], 'already_absent')

    def test_real_child_exit_and_no_second_launch(self):
        with tempfile.TemporaryDirectory(prefix='sequence_child_') as directory:
            root = Path(directory)
            base = dict(name='cpu_child', argv=[sys.executable, '-c', "print('actual CPU child')"],
                environment={}, log=str(root / 'child.log'), exit_file=str(root / 'child.exit'))
            records = []
            result = s.launch_child(base, records.append)
            self.assertEqual(result['exit_code'], 0)
            self.assertEqual((root / 'child.exit').read_text().strip(), '0')
            self.assertEqual((root / 'child.log').read_text().strip(), 'actual CPU child')
            self.assertEqual([row['status'] for row in records], ['running', 'exited'])
            with self.assertRaises(ValueError):
                s.launch_child(base, records.append)
            failing = dict(base, name='cpu_failing_child', argv=[sys.executable, '-c', 'raise SystemExit(7)'],
                log=str(root / 'failed.log'), exit_file=str(root / 'failed.exit'))
            with self.assertRaises(ValueError):
                s.launch_child(failing, records.append)
            self.assertEqual((root / 'failed.exit').read_text().strip(), '7')

    def test_global_flock_prevents_second_driver_before_any_launch(self):
        with tempfile.TemporaryDirectory(prefix='sequence_lock_') as directory:
            root = Path(directory)
            lock = root / 'sequence.lock'
            plan = dict(schema='cmcc_fixed_sequence_20261030_v1', output=str(root), lock_file=str(lock))
            s.write(root / 'plan.json', plan)
            with lock.open('w') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    s.run(root / 'plan.json')
            self.assertFalse((root / 'status.json').exists())


def main():
    global REAL_ACCEPTANCE, REAL_ACCEPTANCE_EXIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance-report', type=Path, required=True)
    parser.add_argument('--acceptance-exit', type=Path, required=True)
    args = parser.parse_args()
    REAL_ACCEPTANCE, REAL_ACCEPTANCE_EXIT = args.acceptance_report, args.acceptance_exit
    s.cpu_environment()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SequenceTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
