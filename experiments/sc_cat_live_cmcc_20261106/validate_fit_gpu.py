"""Run one bounded real-model technical fit and record its actual GPU window."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import live_io as io


def save(path, value):
    temporary = Path(str(path) + '.partial')
    io.write(temporary, value)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--log', required=True)
    args = parser.parse_args()
    request = io.read(args.request)
    io.require(request['phase'] == 'technical_validation', 'This entry only executes the allocated original-schedule two-step validation.')
    io.require(not Path(args.execution).exists() and not Path(args.log).exists(), 'Use a new validation log and execution record.')
    began = time.time()
    io.require(0 < request['deadline_epoch'] - began <= 1800, 'This validation has at most the assigned1800-second window.')
    command = [sys.executable, '-B', '-u', '-m', 'experiments.sc_cat_live_cmcc_20261106.fit_worker', '--request', str(Path(args.request).resolve())]
    with Path(args.log).open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report = dict(status='running', gpu=request['gpu'], controller_pid=os.getpid(), worker_pid=child.pid,
            command=command, request=io.binding(args.request), started_epoch=began,
            deadline_epoch=request['deadline_epoch'], allocated_max_GPU_hours=.5, log=str(Path(args.log).resolve()))
        save(args.execution, report)
        error = None
        try:
            code = child.wait(timeout=max(.01, request['deadline_epoch'] - time.time()))
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = child.returncode
        ended = time.time()
    completed = Path(request['output']) / 'completed.json'
    ok = code == 0 and completed.is_file() and io.read(completed).get('exit_code') == 0
    report.update(status='completed' if ok else 'failed', exit_code=code, ended_epoch=ended,
        actual_seconds=ended-began, actual_GPU_hours=(ended-began)/3600, error=error,
        completion=io.binding(completed) if completed.is_file() else None)
    save(args.execution, report)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
