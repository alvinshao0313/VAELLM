"""Verify all seven original configurations/prefixes without loading model weights."""
from __future__ import annotations

import os
from pathlib import Path
import time

from common import (HERE, JOBS, OUTPUT, SNAPSHOT, check_hashes, check_imports, longbench,
                    prediction_prefixes, read, require, sha256, stat, write_json)


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation is CPU-only.')
    require(not OUTPUT.exists(), 'Never overwrite an existing resume preparation.')
    imports = check_imports()
    sources = {str(path): sha256(path) for path in sorted(HERE.glob('*.py'))}
    records = {}
    for key, job in JOBS.items():
        output = job['output']
        config_path = output / 'longbench_config.json'
        config = read(config_path)
        require(config_path.read_bytes() == (output / 'longbench/config.json').read_bytes(),
                'Original config bytes differ: ' + key)
        require(config['mode'] == 'full' and config['model']['checkpoint_id'] == job['checkpoint_id'],
                'Wrong original model/config identity: ' + key)
        checkpoint = Path(config['model']['checkpoint'])
        meta = read(checkpoint / 'checkpoint_meta.json')
        require(meta['checkpoint_id'] == job['checkpoint_id'], 'Wrong checkpoint metadata: ' + key)
        require(config['model']['recompression']['max_linear_bpw'] == 2.6
                and config['model']['recompression']['max_lora_rank'] == 0, 'Changed native budget: ' + key)
        longbench.runner.check_config(config)
        pins = dict(config['source_sha256'])
        pins.update(config['model']['recompression']['source_sha256'])
        check_hashes(pins)
        old_runs = sorted((output / 'longbench/runs').glob('*.json'))
        require(len(old_runs) == 1, 'Expected one interrupted original LongBench run: ' + key)
        old = read(old_runs[0])
        require(old['status'] == 'running' and not Path(f"/proc/{old['pid']}").exists(),
                'Original run must remain interrupted with no live process: ' + key)
        for name in ('mmlu', 'seven'):
            require((output / (name + '.exit')).read_text().strip() == '0', 'Full8 stage incomplete: ' + key)
        require(read(output / 'full8_summary.json')['sample_count'] == 24742, 'Full8 is not complete: ' + key)
        require(not (output / 'longbench/scores.json').exists()
                and not (output / 'nine_task_acceptance.json').exists(), 'Never overwrite completed scores: ' + key)
        protected = [output / name for name in ('evaluation.json', 'full8_summary.json', 'mmlu.json',
            'seven.json', 'mmlu.exit', 'seven.exit', 'longbench_config.json', 'longbench/config.json')]
        protected += old_runs + sorted(output.glob('*_process.json'))
        protected += sorted(output.glob('longbench_supplement*/report.json'))
        prefixes = prediction_prefixes(output, config)
        require(sum(item['rows'] for item in prefixes.values()) == job['rows'], 'Interrupted prefix count changed: ' + key)
        records[key] = dict(gpu=job['gpu'], checkpoint_id=job['checkpoint_id'], output=str(output),
            checkpoint=str(checkpoint), config_path=str(config_path), config_sha256=sha256(config_path),
            checkpoint_weight_stat=stat(checkpoint / meta['state_dict_file']),
            preserved_sha256={str(path): sha256(path) for path in protected},
            source_sha256=pins, prediction_prefixes=prefixes,
            already_present=job['rows'], remaining=4750 - job['rows'],
            original_report=str(old_runs[0]), original_status='running',
            original_interruption='server reboot on2026-10-27 around22:02 Asia/Shanghai; no exit0 claimed')
        print(dict(job=key, verified_prefix=job['rows'], remaining=4750 - job['rows']), flush=True)
    OUTPUT.mkdir()
    write_json(OUTPUT / 'prepared.json', dict(schema='qwen_seven_longbench_resume_20261028_v1',
        status='cpu_verified', prepared_unix=time.time(), source_sha256=sources, imports=imports,
        snapshot=str(SNAPSHOT), jobs=records, max_runtime_seconds=86400,
        driver_change_note='Original driver570.86.10; resumed driver590.48.01; same torch2.6.0+cu124. Normal floating-point differences are allowed; existing prediction bytes are immutable.',
        full8_rerun=False, automatic_retry=False))


if __name__ == '__main__':
    main()
