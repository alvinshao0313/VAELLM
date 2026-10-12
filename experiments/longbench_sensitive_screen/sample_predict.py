"""Explicit H50 inference from frozen S, reusing the original single-sample generator."""
import argparse
import fcntl
import gc
import json
import logging
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from sample_core import check_rows, check_score_config, load_manifest, selected_rows
from experiments.longbench_v1 import common, data, models, runner

FROZEN_ROOT = Path('/home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01')


def load_inputs(config_path, manifest_path):
    if Path.cwd() != FROZEN_ROOT or os.environ.get('PYTHONPATH') != str(FROZEN_ROOT):
        raise ValueError('Call the absolute sample entry from frozen S with PYTHONPATH=S.')
    for module in (common, data, models, runner):
        if not Path(module.__file__).resolve().is_relative_to(FROZEN_ROOT):
            raise ValueError('Prediction dependencies must resolve to frozen S: ' + module.__name__)
    config_path, manifest_path = Path(config_path).resolve(), Path(manifest_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    sample, archive, rows = load_manifest(manifest_path)
    check_score_config(config, sample, archive)
    models.check_config(config)
    return config, sample, archive, selected_rows(sample, rows)


def run(config_path, manifest_path, output, gpu, *, resume=False, max_runtime_seconds=None):
    if type(gpu) is not int or gpu < 0 or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Explicit --gpu must match exactly one assigned CUDA_VISIBLE_DEVICES.')
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError('Runtime budget must be positive; original generation caps remain unchanged.')
    config_path, manifest_path, output = map(lambda p: Path(p).resolve(), (config_path, manifest_path, output))
    config, sample, archive, targets = load_inputs(config_path, manifest_path)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError('Explicit --resume needs the existing H50 output.')
        for name, source in (('config.json', config_path), ('sample_manifest.json', manifest_path)):
            if (output / name).read_bytes() != source.read_bytes():
                raise ValueError('Resume requires unchanged original config and H50 manifest.')
    else:
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(config_path, output / 'config.json')
        shutil.copyfile(manifest_path, output / 'sample_manifest.json')
    config_sha, sample_sha = common.sha256(config_path), common.sha256(manifest_path)
    prediction_dir, reports = output / 'predictions', output / 'runs'
    prediction_dir.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    report_path = reports / f'{time.time_ns()}_{os.getpid()}.json'
    record = {'schema': 'longbench_v1_h50_execution_v1', 'mode': 'sample_partial', 'kind': 'sample_partial',
        'is_full_benchmark': False, 'status': 'running', 'pid': os.getpid(),
        'start_tick': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'started_unix': time.time(),
        'config_sha256': config_sha, 'sample_manifest_sha256': sample_sha,
        'indices_sha256': sample['indices_sha256'], 'archive_sha256': sample['archive']['sha256'],
        'selected_tasks': sample['high_tasks'], 'target_sample_count': 600, 'new_predictions': 0,
        'already_present': 0, 'max_runtime_seconds': max_runtime_seconds,
        'runtime_budget_note': 'Only checked between complete official samples; never shorten input or generation.',
        'source_sha256': {str(p): common.sha256(p) for p in
            (Path(__file__).resolve(), Path(__file__).with_name('sample_core.py').resolve(),
             Path(__file__).with_name('core.py').resolve(), Path(runner.__file__).resolve(), Path(models.__file__).resolve())},
        'acceptance_note': sample['acceptance_note']}
    common.write_json(report_path, record)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('longbench_h50')
    handles, model, tokens = {}, None, None
    try:
        remaining = []
        for task in sample['high_tasks']:
            path = prediction_dir / (task + '.jsonl')
            stream = path.open('a+', encoding='utf-8')
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles[task] = stream; stream.seek(0)
            existing = [json.loads(line) for line in stream]
            references = [row for row in targets if row['dataset'] == task]
            if existing and not resume:
                raise FileExistsError('Existing predictions require explicit --resume: ' + task)
            check_rows(existing, references, config_sha, sample_sha256=sample_sha, complete=False)
            record['already_present'] += len(existing); remaining.extend(references[len(existing):])
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        if remaining:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(archive['tokenizer']['path'], use_fast=True, local_files_only=True)
            model, record['load_audit'] = models.load_model(config, logger)
            record['gpu'] = {'name': torch.cuda.get_device_name(0),
                'total_memory': torch.cuda.get_device_properties(0).total_memory}
            common.write_json(report_path, record)
            tokens = np.memmap(archive['inputs']['path'], dtype='<u4', mode='r')
        for row in remaining:
            if max_runtime_seconds is not None and time.time() - record['started_unix'] >= max_runtime_seconds:
                record['status'] = 'sample_partial_runtime_budget'
                break
            result = runner.generate_one(model, tokenizer, runner.read_input(tokens, row), row,
                diagnostic=False, protocol=config['protocol'])
            prediction = {key: row[key] for key in
                ('dataset', 'row_index', '_id', 'answers', 'all_classes', 'length', 'input_ids_sha256')}
            prediction.update(result, input_tokens=row['input_tokens'], removed_tokens=row['removed_tokens'],
                config_sha256=config_sha, diagnostic=False, evaluation_kind='sample_partial',
                sample_manifest_sha256=sample_sha)
            stream = handles[row['dataset']]
            stream.write(json.dumps(prediction, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
            record['new_predictions'] += 1
            logger.info('dataset=%s row=%d input=%d generated=%d elapsed=%.3fs', row['dataset'], row['row_index'],
                row['input_tokens'], result['generated_tokens'], result['elapsed_seconds'])
        else:
            record['status'] = 'sample_partial_complete'
        models.check_config(config)
        if common.sha256(manifest_path) != sample_sha:
            raise ValueError('H50 manifest changed during inference.')
        common.check_hashes(record['source_sha256'])
        record['input_pins_unchanged_after'] = True; record['exit_code'] = 0
    except BaseException as error:
        record.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc())
        raise
    finally:
        for stream in handles.values(): stream.close()
        del tokens, model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        record.update(finished_unix=time.time(), duration_seconds=time.time() - record['started_unix'])
        common.write_json(report_path, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--max-runtime-seconds', type=float)
    args = parser.parse_args()
    report = run(args.config, args.manifest, args.output, args.gpu,
        resume=args.resume, max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps({key: report[key] for key in ('status', 'mode', 'is_full_benchmark', 'new_predictions', 'already_present')}))
