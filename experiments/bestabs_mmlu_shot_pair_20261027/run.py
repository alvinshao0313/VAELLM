"""One prepared same-B16 native MMLU arm; no retry or protocol changes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from .prepare import CHECKPOINT, read, require, sha
from experiments.night_sequence_20261026.queue_wait import free_gpu, process_start, run_stage, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--shot', choices=('0', '5'), required=True)
    args = parser.parse_args()
    manifest = args.run_dir / 'prepared.json'
    prepared = read(manifest)
    require(prepared['status'] == 'prepared_no_GPU_started' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'The arm driver is CPU-only and consumes the completed preparation.')
    for pins in (prepared['source_sha256'], prepared['reusable_baseline_pins'], prepared['hf_source']['metadata_sha256']):
        for path, expected in pins.items():
            require(sha(path) == expected, 'A bound source or reused control changed: ' + path)
    for item in (*prepared['hf_source']['weight_files'], *prepared['dataset_cache_stats']):
        actual = Path(item['path']).stat()
        require(item == dict(path=item['path'], size=actual.st_size, mtime_ns=actual.st_mtime_ns,
            inode=actual.st_ino), 'The HF or complete MMLU cache identity changed: ' + item['path'])
    for baseline in prepared['reused_BF16'].values():
        require(sha(baseline['path']) == baseline['sha256'], 'The reused complete BF16 result changed.')
    stat = (CHECKPOINT / 'pytorch_model.bin').stat()
    require(prepared['source_weight_stat'] == dict(path=str(CHECKPOINT / 'pytorch_model.bin'),
        bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino), 'The actual bestabs source weights changed.')
    command = prepared['commands'][args.shot]
    directory = args.run_dir / f'bestabs_shot{args.shot}'
    require(not directory.exists(), 'Do not overwrite or resume a started MMLU arm.')
    resource = free_gpu(command['gpu'])
    directory.mkdir()
    report_path = directory / 'run.json'
    environment = {**os.environ, **prepared['environment'], 'CUDA_VISIBLE_DEVICES': str(command['gpu'])}
    report = dict(status='running', shot=int(args.shot), gpu=command['gpu'],
        driver_pid=os.getpid(), driver_start_ticks=process_start(os.getpid()), started_unix=time.time(),
        prepared_path=str(manifest), prepared_sha256=sha(manifest),
        source_sha256={str(Path(__file__)): sha(__file__),
            str(Path(__file__).parents[1] / 'night_sequence_20261026/queue_wait.py'):
                sha(Path(__file__).parents[1] / 'night_sequence_20261026/queue_wait.py')},
        resource=resource, environment={key: environment[key] for key in prepared['environment']} | {'CUDA_VISIBLE_DEVICES': str(command['gpu'])},
        outer_deadline_seconds=command['outer_timeout_seconds'], termination_grace_seconds=20, stages={})
    write_json(report_path, report)
    try:
        run_stage(command['argv'], cwd=Path(command['cwd']), environment=environment,
            log=directory / 'mmlu.log', report=report, report_path=report_path, name='mmlu')
        require((directory / 'mmlu.json').is_file(), 'Native evaluator returned without the original full result.')
        report['status'] = 'completed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['finished_unix'] = time.time()
        write_json(report_path, report)


if __name__ == '__main__':
    main()
