"""Measure one native CAT invocation without replacing its training algorithm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from experiments.w25_eval.model_inventory import model_inventory, sha256, verify_source_manifest


def read_json(path):
    return json.loads(Path(path).read_text())


def check_inputs(record):
    for field in ('source_sha256', 'input_evidence_sha256', 'model_metadata_sha256'):
        for name, expected in record[field].items():
            if sha256(name) != expected:
                raise ValueError(f'Input changed: {name}')
    for row in record['input_data_stat'] + record.get('source_weight_stat', []):
        stat = Path(row['path']).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (row['size'], row['mtime_ns'], row['inode']):
            raise ValueError(f'Input stat changed: {row["path"]}')
    source = record['source']
    if (not source['all_official_weights_verified'] or not source['source_manifest_sha256']
            or sha256(source['source_manifest']) != source['source_manifest_sha256']):
        raise ValueError('Bind the completed official source verification before running.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    args = parser.parse_args()
    record = read_json(args.launch)
    output = Path(record['training_resource_report'])
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    check_inputs(record)
    architecture = model_inventory(record['source']['model_path'])
    identity = verify_source_manifest(record['source']['source_manifest'], architecture)
    if record['argv'][:2] != ['python', 'tools/cat_train.py']:
        raise ValueError('This diagnostic invokes the unchanged native CAT entry only.')
    if any(Path(record['cat_output_root']).iterdir()):
        raise ValueError('The diagnostic CAT output root must be fresh.')
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'pid': os.getpid(), 'start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
              'cuda_initialized_before_resource_measurement': torch.cuda.is_initialized(),
              'source_identity': identity, 'argv': record['argv'],
              'scope': 'One native CAT process, including calibration, compression, KD, fusion and saves. No per-stage reset or sampling monitor.'}
    from tools.cat_train import main as cat_main
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    report['cuda_initialized_before_native_main'] = torch.cuda.is_initialized()
    report['allocated_before_native_main_bytes'] = torch.cuda.memory_allocated()
    started = time.monotonic()
    try:
        cat_main(record['argv'][2:])
        check_inputs(record)
        report['status'] = 'completed'
        report['exit_code'] = 0
    except BaseException as exc:
        report.update(status='failed', exit_code=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      finished_at=datetime.now(timezone.utc).isoformat())
        with output.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')


if __name__ == '__main__':
    main()
