"""One four-hour absolute deadline per explicitly selected LR arm."""
import argparse
import os
import signal
import subprocess
import sys
import time
import traceback

from . import bindings as b


def terminate_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(b.ARMS), required=True)
    arm = parser.parse_args().arm
    b.cpu_environment()
    prepared = b.read(b.RESULT / 'prepared.json')
    b.require(prepared['status'] == 'cpu_verified', 'Require shared source/config verification.')
    b.check_pins(prepared['source_sha256'])
    output = b.RESULT / arm
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'sequence.json'
    b.require(not path.exists(), 'Do not duplicate a launched LR arm.')
    start, gpu = time.time(), str(b.ARMS[arm]['gpu'])
    report = dict(status='running', arm=arm, pid=os.getpid(), started_unix=start,
        deadline_unix=start + 14400, authorized_gpu_hours=4, GPU=gpu,
        command=[sys.executable, *sys.argv], stages=[], automatic_retry=False,
        automatic_recovery_suffix=False, prepared_sha256=b.sha(b.RESULT / 'prepared.json'))
    b.write(path, report)
    try:
        for name, module, args, visible in (
            ('fit_fresh', 'run', ['--arm', arm, '--phase', 'formal'], gpu),
            ('raw_full8', 'evaluate', ['--component', 'raw_full8', '--arm', arm], ''),
            ('train_readout', 'evaluate', ['--component', 'readout', '--arm', arm], gpu),
        ):
            command = [sys.executable, '-B', '-u', '-m', 'experiments.down_d16_lr_cmcc_20261104.' + module, *args]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=visible, PYTHONHASHSEED='31',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false')
            allowance = report['deadline_unix'] - time.time()
            b.require(allowance > 0, 'The arm absolute deadline expired before this dependency.')
            with (output / (name + '.log')).open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=b.REPO, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, pid=process.pid, command=command, status='running', started_unix=time.time())
                report['stages'].append(stage)
                b.write(path, report)
                try:
                    code = process.wait(timeout=allowance)
                except BaseException:
                    terminate_group(process.pid)
                    process.wait()
                    raise
                finally:
                    terminate_group(process.pid)
            stage.update(status='exited', exit_code=code, finished_unix=time.time())
            b.write(path, report)
            b.require(code == 0, 'A required LR arm component failed; stop its dependents without retry.')
        report.update(status='completed_fit_fresh_full8_readout', exit_code=0)
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - start)
        b.write(path, report)


if __name__ == '__main__':
    main()
