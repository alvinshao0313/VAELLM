"""Run five explicitly bound native evaluation shards once, without a queue."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.llama_prefix_stage_full8.prepare import check, read, require
from experiments.longbench_v1.common import sha256, write_json


def run(record_path):
    record_path = Path(record_path).resolve(); record = read(record_path)
    output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and Path.cwd().resolve() == Path(record['cwd']) and not record['queued'] and not record['automatic_retry'],
            'Use the explicitly bound GPU, repository and interpreter, without queue or retry.')
    require(all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
            'Use the prepared offline/allocator/numerical environment.')
    require(record['prepared_sha256'] == sha256(output / 'launch.prepared.json'), 'Prepared configuration changed.')
    require(not (output / 'run.exit').exists(), 'Do not overwrite an existing run.')
    for row in record['stage_commands']:
        require(not Path(row['argv'][-1]).exists(), 'Do not overwrite a native result.')
    check(record)
    occupied = subprocess.check_output(['nvidia-smi', '--id=' + str(record['gpu']), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    require(not occupied, 'The explicitly assigned GPU is occupied: ' + occupied)
    record.update(status='running', worker_pid=os.getpid(),
        worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19]),
        started_unix=time.time(), stages={})
    write_json(record_path, record)
    commands = [*record['stage_commands'], {'name': 'comparison', 'argv': [sys.executable, '-m',
        'experiments.llama_prefix_stage_full8.compare', '--launch-record', str(record_path)]}]
    try:
        for row in commands:
            check(record); name = row['name']; env = os.environ.copy()
            if name == 'comparison':
                env['CUDA_VISIBLE_DEVICES'] = ''
            with (output / (name + '.log')).open('x') as log:
                child = subprocess.Popen(row['argv'], cwd=record['cwd'], env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                record['stages'][name] = {'argv': row['argv'], 'pid': child.pid, 'started_unix': time.time()}
                record['current_stage'] = name; write_json(record_path, record)
                result = child.wait()
            (output / (name + '.exit')).write_text(str(result) + '\n')
            record['stages'][name].update(exit_code=result, finished_unix=time.time()); write_json(record_path, record)
            require(result == 0, name + ' failed; no retry or subsequent shard.')
        record.update(status='completed', comparison_sha256=sha256(output / 'comparison.json'))
        code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(record_path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().launch_record))
