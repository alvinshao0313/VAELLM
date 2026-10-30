"""Fresh BF16 -> new after_o -> selected A/B -> its live up/down, in two processes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import fresh_support as s


def gpu_available(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = {int(parts[0]): (parts[1].strip(), int(parts[2]))
               for row in rows.splitlines() if (parts := row.split(','))}
    uuid, memory = devices[gpu]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True)
    s.require(memory <= 256 and not any(line.split(',')[0].strip() == uuid for line in apps.splitlines()),
              'Selected GPU is occupied; no reservation, wait or retry.')
    return dict(gpu=gpu, uuid=uuid, used_MiB=memory, checked_unix=time.time())


def worker_command(request_path, phase):
    return [str(s.PYTHON), '-B', '-u', str(s.HERE / 'worker.py'), '--request', str(request_path), '--phase', phase]


def execute(base_model_path, output_dir, arm, gpu):
    s.require(Path(sys.executable).resolve() == s.PYTHON.resolve(), 'Activate bitvae in the caller shell.')
    base = Path(base_model_path).resolve(strict=True)
    output = Path(output_dir).resolve()
    s.require(not output.exists(), 'Use a new output directory; this entry never resumes an existing run.')
    identity = s.hf_identity(base)
    resource = gpu_available(gpu)
    environment = s.worker_environment(gpu)
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / 'request.json'
    request = dict(schema='cat_structure_fresh_request_v1', base_model_path=str(base), output_dir=str(output),
        arm=arm, gpu=gpu, hf_identity=identity, source_sha256=s.source_pins(),
        effective_contract=s.policy(arm), initial_checkpoint=None,
        process_protocol='prefix exit0 after actual save, then a separate seed31 suffix process')
    s.write(request_path, request)
    report = dict(status='running', pid=os.getpid(), started_unix=time.time(), command=[sys.executable, *sys.argv],
        resources=resource, request_sha256=s.sha(request_path), phases=[],
        evaluation='None; final full9 and actual logical/packed/full-model storage audit are separate.')
    report_path = output / 'run.json'
    s.write(report_path, report)
    try:
        for phase in ('prefix', 'suffix'):
            if phase == 'suffix':
                s.own_after_o(request)
            command = worker_command(request_path, phase)
            with (output / (phase + '.log')).open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=s.SNAPSHOT, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                event = dict(phase=phase, pid=process.pid, command=command, started_unix=time.time())
                report['phases'].append(event)
                s.write(report_path, report)
                code = process.wait()
            event.update(exit_code=code, elapsed_seconds=time.time() - event['started_unix'])
            (output / (phase + '.exit')).write_text(str(code) + '\n', encoding='utf-8')
            s.write(report_path, report)
            s.require(code == 0, f'{phase} failed with exit {code}; see {phase}.log. No retry or next phase.')
            worker = s.read(output / (phase + '.json'))
            s.require(worker['status'] == ('prefix_completed' if phase == 'prefix' else 'completed')
                      and worker['pid'] == process.pid and worker['seed_calls'] == [31], 'Worker did not complete its native phase.')
        s.require(report['phases'][0]['pid'] != report['phases'][1]['pid'], 'Prefix and suffix must be distinct processes.')
        report.update(status='completed', exit_code=0,
                      final_endpoint=s.read(output / 'suffix.json')['saved_boundaries'][-1])
    except BaseException as error:
        report.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        s.write(report_path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_model_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    execute(args.base_model_path, args.output_dir, args.arm, args.gpu)
