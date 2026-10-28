"""Minimal real-child checks for the fixed sequence's process handoff and failure propagation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_20261026.queue_wait import (
    process_start, read_json, run_stage, wait_for_process,
)


class ProcessHandoffTest(unittest.TestCase):
    def test_bound_process_exit_event(self):
        process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(.2)"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started = process_start(process.pid)
        before = time.monotonic()
        wait_for_process(process.pid, started)
        elapsed = time.monotonic() - before
        self.assertEqual(process.wait(), 0)
        self.assertGreater(elapsed, .1)
        self.assertIsNone(process_start(process.pid))

    def test_pid_identity_mismatch_rejected(self):
        process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(.2)"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started = process_start(process.pid)
        with self.assertRaisesRegex(RuntimeError, "reused"):
            wait_for_process(process.pid, started + 1)
        self.assertEqual(process.wait(), 0)

    def run_pair(self, directory: Path, first_code: int) -> dict:
        report = {"stages": {}}
        report_path = directory / "report.json"
        first = [sys.executable, "-B", "-c", f"import sys; print('actual child'); sys.exit({first_code})"]
        marker = directory / "dependent_started"
        second = [sys.executable, "-B", "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('started')",
                  str(marker)]
        run_stage(first, cwd=ROOT, environment=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                  log=directory / "first.log", report=report, report_path=report_path, name="first")
        run_stage(second, cwd=ROOT, environment=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                  log=directory / "dependent.log", report=report, report_path=report_path, name="dependent")
        return read_json(report_path)

    def test_success_runs_exactly_one_dependent(self):
        with tempfile.TemporaryDirectory(prefix="night_queue_success_", dir=ROOT / "result") as temporary:
            directory = Path(temporary)
            report = self.run_pair(directory, 0)
            self.assertEqual(set(report["stages"]), {"first", "dependent"})
            self.assertTrue(all(row["exit_code"] == 0 for row in report["stages"].values()))
            self.assertEqual((directory / "dependent_started").read_text(), "started")
            print(f"cleanup {directory}: real-child test completed, "
                  f"{sum(path.stat().st_size for path in directory.iterdir())} bytes", flush=True)

    def test_failure_preserves_exit_and_does_not_launch_dependent(self):
        with tempfile.TemporaryDirectory(prefix="night_queue_failure_", dir=ROOT / "result") as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "exit 7"):
                self.run_pair(directory, 7)
            report = read_json(directory / "report.json")
            self.assertEqual(set(report["stages"]), {"first"})
            self.assertEqual(report["stages"]["first"]["exit_code"], 7)
            self.assertFalse((directory / "dependent_started").exists())
            self.assertFalse((directory / "dependent.log").exists())
            print(f"cleanup {directory}: exit7 propagation verified, "
                  f"{sum(path.stat().st_size for path in directory.iterdir())} bytes", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
