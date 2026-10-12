"""Run one fixed sixteen-case retrieval diagnostic with the snapshot loader."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from experiments.longbench_attribution.restore import frozen_inventory
from experiments.longbench_v1.common import sha256, write_json
from experiments.longbench_v1.models import load_model
from experiments.longbench_v1.runner import generate_one, read_input
from experiments.longbench_v1.scoring import score_example
from experiments.rht_retrieval_probe.prepare import check_preparation


def run(preparation):
    from transformers import AutoTokenizer

    preparation = Path(preparation).resolve()
    record, manifest = check_preparation(preparation)
    source_root = Path(record['source_root']).resolve()
    if (record['status'] != 'bound_ready' or record['queued'] or record['automatic_retry']
            or type(record['gpu']) is not int
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(record['gpu'])
            or Path(sys.executable).resolve() != Path(record['python_executable']).resolve()
            or Path.cwd() != source_root or Path(__file__).resolve().parents[2] != source_root
            or any(os.environ.get(k) != v for k, v in record['environment'].items())
            or any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))):
        raise ValueError('Use the bound GPU, snapshot source and explicit prepared environment.')
    imports = {}
    for name in ('train_utils.v6_model_loader', 'litebsq.vae_linear', 'rotation.hadamard_utils',
                 'rotation.hadamard_extended', 'rotation.weight_preprocess', 'tools.cat_eval',
                 'experiments.longbench_v1.models', 'experiments.longbench_v1.runner',
                 'experiments.longbench_v1.protocol', 'experiments.longbench_v1.scoring',
                 'experiments.longbench_attribution.restore', 'experiments.rht_retrieval_probe.prepare'):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError('Runtime import escaped the full-RHT snapshot: ' + name)
        imports[name] = str(path)
    output = Path(record['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = record['config']
    report = {'schema': 'rht_retrieval_probe_result_v1', 'status': 'running', 'arm': record['arm'],
        'checkpoint_id': config['model']['checkpoint_id'], 'checkpoint': config['model']['checkpoint'],
        'preparation_sha256': sha256(preparation), 'pid': os.getpid(),
        'start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'runtime_import_paths': imports,
        'started_unix': time.time(), 'scope': record['scope'], 'sample_count': 0, 'cases': [],
        'model_mutations': False, 'model_exported': False,
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    report_path = output / 'report.json'
    write_json(report_path, report)
    model = tokens = None
    code = 1
    try:
        seed = config['protocol']['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer']['path'], use_fast=True, local_files_only=True)
        model, report['load_audit'] = load_model(config, logging.getLogger(__name__))
        if (not report['load_audit']['rope_buffers']
                or any(value['dtype'] != 'torch.float32' for value in report['load_audit']['rope_buffers'].values())):
            raise ValueError('Original RoPE frequency buffers must stay FP32.')
        before = frozen_inventory(model, ())
        tokens = np.memmap(manifest['inputs']['path'], dtype='<u4', mode='r')
        with (output / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
            for i, row in enumerate(record['cases']):
                generated = generate_one(model, tokenizer, read_input(tokens, row), row,
                                         diagnostic=True, protocol=config['protocol'])
                result = {key: row[key] for key in ('dataset', 'row_index', '_id', 'answers', 'all_classes',
                                                   'length', 'input_ids_sha256', 'input_tokens')}
                result.update(generated, diagnostic=True, checkpoint_id=report['checkpoint_id'],
                              preparation_sha256=report['preparation_sha256'])
                result['official_score'] = score_example(row['dataset'], generated['pred'], row['answers'], row['all_classes'])
                result['control_scores'] = {kind: values[i]['official_score'] for kind, values in record['controls'].items()}
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush(); os.fsync(handle.fileno())
                report['cases'].append({key: result[key] for key in ('dataset', 'row_index', 'input_ids_sha256',
                    'official_score', 'control_scores', 'elapsed_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')})
                report['sample_count'] += 1
                write_json(report_path, report)
                logging.info('arm=%s task=%s row=%d completed=%d/16', record['arm'], row['dataset'],
                             row['row_index'], report['sample_count'])
        after = frozen_inventory(model, ())
        if before != after or report['sample_count'] != 16:
            raise ValueError('Generation changed model state/identity or did not finish all sixteen cases.')
        check_preparation(preparation)
        report.update(status='diagnostic_complete', exit_code=0, input_pins_unchanged_after=True,
            frozen_state_and_identity_unchanged=True, frozen_state_tensors=len(before['state']),
            frozen_buffers=len(before['buffers']),
            frozen_state_sha256=hashlib.sha256(json.dumps(before['state'], sort_keys=True).encode()).hexdigest(),
            predictions_sha256=sha256(output / 'predictions.jsonl'))
        report['selected_task_scores'] = {task: {
            'candidate': 100 * sum(c['official_score'] for c in report['cases'] if c['dataset'] == task) / 8,
            **{kind: 100 * sum(c['control_scores'][kind] for c in report['cases'] if c['dataset'] == task) / 8
               for kind in record['controls']}} for task in ('passage_retrieval_en', 'passage_retrieval_zh')}
        code = 0
    except BaseException as error:
        report.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        write_json(report_path, report)
        (output / 'run.exit').write_text(str(code) + '\n')
        del model, tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(args.preparation)
