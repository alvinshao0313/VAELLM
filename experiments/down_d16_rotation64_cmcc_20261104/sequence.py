"""One bounded, explicitly approved raw-stage chain; no retries or suffixes."""
import os
from pathlib import Path
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
    b.cpu_environment()
    prepared = b.read(b.RESULT / 'prepared.json')
    b.require(prepared['status'] == 'cpu_verified', 'Complete CPU source/resolved-consumer verification first.')
    b.check_pins(prepared['source_sha256'])
    path = b.RESULT / 'sequence.json'
    b.require(not path.exists(), 'Do not repeat a previously launched chain.')
    start = time.time()
    record = dict(status='running', pid=os.getpid(), started_unix=start,
        deadline_unix=start + 4 * 3600, authorized_gpu_hours=4,
        command=[sys.executable, *sys.argv], GPU=4, stages=[], automatic_retry=False,
        automatic_recovery_suffix=False, prepared_sha256=b.sha(b.RESULT / 'prepared.json'))
    b.write(path, record)
    try:
        for name, module, args, gpu in (
            ('fit_fresh', 'run', ['--arm', b.ARM, '--phase', 'formal'], '4'),
            ('raw_full8', 'evaluate', ['--component', 'raw_full8', '--arm', b.ARM], ''),
            ('train_readout', 'evaluate', ['--component', 'readout', '--arm', b.ARM], '4'),
        ):
            command = [sys.executable, '-B', '-u', '-m', 'experiments.down_d16_rotation64_cmcc_20261104.' + module, *args]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONHASHSEED='31',
                HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            remaining = record['deadline_unix'] - time.time()
            b.require(remaining > 0, 'The absolute four-hour chain deadline has expired.')
            with (b.RESULT / (name + '.log')).open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=b.REPO, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, pid=process.pid, command=command, started_unix=time.time(), status='running')
                record['stages'].append(stage)
                b.write(path, record)
                try:
                    code = process.wait(timeout=remaining)
                except BaseException:
                    terminate_group(process.pid)
                    process.wait()
                    raise
                finally:
                    # The nested consumer may outlive a failed leader; clean this session's group unconditionally.
                    terminate_group(process.pid)
            stage.update(exit_code=code, finished_unix=time.time(), status='exited')
            b.write(path, record)
            b.require(code == 0, 'A required raw-stage component failed; stop its dependent steps.')
        record.update(status='completed_fit_fresh_full8_readout', exit_code=0)
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - start)
        b.write(path, record)


if __name__ == '__main__':
    main()
