"""Append one fixed original LongBench prefix and score only the complete run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from common import (JOBS, OUTPUT, ROOT, check_gpu, check_hashes, check_imports, check_prefix_bytes,
                    longbench, read, recovery_eval, require, sha256, stat, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, choices=list(JOBS))
    args = parser.parse_args()
    prepared_path = OUTPUT / 'prepared.json'
    prepared = read(prepared_path)
    job = prepared['jobs'][args.job]
    require(prepared['status'] == 'cpu_verified', 'Require completed CPU prefix verification.')
    check_hashes(prepared['source_sha256'])
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == job['gpu'], 'Use exactly the assigned original GPU.')
    check_gpu(job['gpu'])
    output, checkpoint = Path(job['output']), Path(job['checkpoint'])
    terminal_dir = OUTPUT / args.job
    terminal_dir.mkdir(exist_ok=True)
    terminal = terminal_dir / 'recovery.json'
    require(not terminal.exists(), 'This independent resume was already started; no automatic retries.')
    record = dict(schema='qwen_longbench_exact_prefix_resume_20261028_v1', status='running',
        job=args.job, pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        started_unix=time.time(), command=[sys.executable, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(ROOT), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        environment={key: os.environ.get(key) for key in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES',
            'PYTORCH_CUDA_ALLOC_CONF', 'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG',
            'TOKENIZERS_PARALLELISM', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE',
            'TRANSFORMERS_OFFLINE', 'WANDB_MODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'PYTHONDONTWRITEBYTECODE')},
        prepared_path=str(prepared_path), prepared_sha256=sha256(prepared_path),
        checkpoint_id=job['checkpoint_id'], checkpoint=str(checkpoint), output=str(output),
        original_report=job['original_report'], original_status='running',
        original_interruption=job['original_interruption'], original_reports_modified=False,
        driver_change_note=prepared['driver_change_note'], prediction_resume=True, full8_rerun=False,
        automatic_retry=False, max_runtime_seconds=86400,
        preserved_sha256=job['preserved_sha256'], source_sha256=job['source_sha256'],
        prediction_prefixes=job['prediction_prefixes'], checkpoint_weight_stat=job['checkpoint_weight_stat'])
    record['command_shell'] = shlex.join(record['command'])
    write_json(terminal, record)
    try:
        record['imports'] = check_imports()
        record['gpu_driver'] = subprocess.check_output(['nvidia-smi', '--id=' + job['gpu'],
            '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
        check_hashes(job['preserved_sha256'])
        check_hashes(job['source_sha256'])
        check_prefix_bytes(job['prediction_prefixes'], exact_length=True)
        require(stat(job['checkpoint_weight_stat']['path']) == job['checkpoint_weight_stat'], 'Bound checkpoint changed.')
        require(not (output / 'longbench/scores.json').exists()
                and not (output / 'nine_task_acceptance.json').exists(), 'Never overwrite an existing complete score.')
        config_path = Path(job['config_path'])
        require(sha256(config_path) == job['config_sha256'], 'Original config changed.')
        write_json(terminal, record)
        longbench.runner.load_model = longbench.load_model
        result = longbench.runner.run(config_path, output / 'longbench', resume=True, max_runtime_seconds=86400)
        record['longbench_run'] = result
        check_prefix_bytes(job['prediction_prefixes'])
        check_hashes(job['preserved_sha256'])
        check_hashes(job['source_sha256'])
        check_hashes(prepared['source_sha256'])
        require(stat(job['checkpoint_weight_stat']['path']) == job['checkpoint_weight_stat'], 'Bound native weights changed.')
        require(result['already_present'] == job['already_present'] and result['exit_code'] == 0,
                'Native runner did not consume the exact original prefix.')
        record['original_prefix_unchanged'] = True
        if result['status'] == 'partial_runtime_budget':
            record.update(status='partial_runtime_budget', exit_code=124)
        else:
            require(result['status'] == 'partition_complete' and result['new_predictions'] == job['remaining'],
                    'All4750 canonical predictions must complete before scoring.')
            original_write = recovery_eval.write
            def preserve_or_write(path, value):
                path = Path(path)
                if path.exists():
                    require(read(path) == value, 'Existing full8 summary differs: ' + str(path))
                else:
                    original_write(path, value)
            recovery_eval.write = preserve_or_write
            full8, baseline_paths, template = recovery_eval.full8_summary(checkpoint, output)
            nine = recovery_eval.summarize_nine(checkpoint, output, config_path, baseline_paths, template)
            check_hashes(job['preserved_sha256'])
            check_hashes(job['source_sha256'])
            check_prefix_bytes(job['prediction_prefixes'])
            record.update(status='complete_existing_full8_plus_full21_nine', exit_code=0,
                full8_mean_percent=full8['mean_percent'], nine_task_score=nine['compressed_nine_task_score'],
                baseline_nine_task_score=nine['baseline_nine_task_score'], passed=nine['passed'],
                scores=str(output / 'longbench/scores.json'), nine=str(output / 'nine_task_acceptance.json'))
    except BaseException as exc:
        record.update(status='failed', exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        write_json(terminal, record)
    print(json.dumps({key: record[key] for key in ('status', 'exit_code', 'original_prefix_unchanged')}), flush=True)
    return record['exit_code']


if __name__ == '__main__':
    sys.exit(main())
