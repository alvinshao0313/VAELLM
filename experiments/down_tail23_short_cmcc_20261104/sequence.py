"""One once-only GPU2 pair with a single absolute60min process-group deadline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.cmcc_runtime_20261030.paths import PYTHON, REPO
from experiments.night_sequence_20261026.queue_wait import free_gpu, process_start, require, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '2' and not output.exists(),
        'Use caller bitvae, assigned GPU2 and one new output; no retry or occupied GPU.')
    resource = free_gpu(2)
    output.mkdir(parents=True)
    started = time.time()
    deadline = started + 3600
    command = [sys.executable, '-B', '-u', '-m', 'experiments.down_tail23_short_cmcc_20261104.run',
        '--output', str(output), '--deadline-unix', str(deadline)]
    report = dict(status='running', pid=os.getpid(), start_ticks=process_start(os.getpid()),
        command=[sys.executable, *sys.argv], started_unix=started, deadline_unix=deadline,
        common_pair_wall_limit_seconds=3600, assigned_gpu=resource, worker_command=command,
        worker_log=str(output / 'worker.log'), output=str(output), retries=0,
        source_sha256={str(path): sha256(path) for path in (Path(__file__).resolve(),
            REPO / 'experiments/cmcc_runtime_20261030/paths.py',
            REPO / 'experiments/night_sequence_20261026/queue_wait.py')})
    path = output / 'sequence.json'
    write_json(path, report)
    process = None
    try:
        with (output / 'worker.log').open('x') as log:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            report.update(worker_pid=process.pid, worker_start_ticks=process_start(process.pid))
            write_json(path, report)
            try:
                code = process.wait(timeout=max(.001, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
                report['deadline_enforced_on_entire_worker_process_group'] = True
        report.update(exit_code=code, worker_returncode=process.returncode,
            status='completed_pair' if code == 0 else 'failed_stop_no_followup')
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        report.update(status='failed_stop_no_followup', exit_code=1)
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started)
        write_json(path, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    raise SystemExit(report['exit_code'])


if __name__ == '__main__':
    main()
