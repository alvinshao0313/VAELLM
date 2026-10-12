"""Run one explicitly bound Llama conditional restoration using native generation."""
from __future__ import annotations

import argparse
import gc
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
from transformers import AutoTokenizer

from experiments.longbench_attribution.restore import frozen_inventory, restore_targets
from experiments.longbench_v1.common import sha256, write_json
from experiments.longbench_v1.models import load_model
from experiments.longbench_v1.runner import generate_one, read_input
from experiments.longbench_v1.scoring import score_example
from experiments.longbench_llama_attribution.prepare import PROFILES, TASKS, ROOT, check_preparation, require
from experiments.w25_eval.model_inventory import model_inventory


def run(preparation, arm):
    preparation = Path(preparation).resolve()
    record, manifest = check_preparation(preparation)
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == ROOT and record['source_root'] == str(ROOT)
            and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and all(os.environ.get(key) == value for key, value in record['environment'].items())
            and not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))
            and not record['queued'] and not record['automatic_retry'] and arm in record['arms'],
            'Use the explicitly bound GPU, public source and prepared environment, without queue or retry.')
    config = record['config']; architecture = model_inventory(config['model']['hf_path'])
    targets = tuple(name for name in architecture['projection_shapes'] if name.rsplit('.', 1)[-1] in record['arms'][arm])
    require(architecture == record['architecture'] and list(targets) == record['selected_targets'][arm]
            and len(targets) == 32 * len(record['arms'][arm]), 'Actual restoration scope differs.')
    output = Path(record['output_root']) / arm; output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'llama_retrieval_restoration_result_v1', 'status': 'running', 'profile': record['profile'], 'arm': arm,
        'checkpoint_id': config['model']['checkpoint_id'], 'preparation_sha256': sha256(preparation),
        'pid': os.getpid(), 'start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'started_unix': time.time(),
        'scope': record['scope'], 'sample_count': 0, 'cases': [], 'model_exported': False, 'goal_acceptance': False}
    path = output / 'report.json'; write_json(path, report)
    model = tokens = None; code = 1
    try:
        seed = config['protocol']['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer']['path'], use_fast=True, local_files_only=True)
        model, report['source_load_audit'] = load_model(config, logging.getLogger(__name__))
        require(report['source_load_audit']['rope_buffers'] and all(
            value['dtype'] == 'torch.float32' for value in report['source_load_audit']['rope_buffers'].values()),
            'Keep original Llama FP32 RoPE buffers.')
        torch.cuda.reset_peak_memory_stats()
        report['restoration'] = restore_targets(model, config['model']['hf_path'], targets)
        report['restoration'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     peak_reserved_bytes=torch.cuda.max_memory_reserved())
        torch.cuda.empty_cache()
        tokens = np.memmap(manifest['inputs']['path'], dtype='<u4', mode='r')
        with (output / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
            for i, row in enumerate(record['cases']):
                generated = generate_one(model, tokenizer, read_input(tokens, row), row,
                                         diagnostic=True, protocol=config['protocol'])
                value = {key: row[key] for key in ('dataset', 'row_index', '_id', 'answers', 'all_classes',
                                                   'length', 'input_ids_sha256', 'input_tokens')}
                value.update(generated, diagnostic=True, arm=arm, checkpoint_id=report['checkpoint_id'],
                             preparation_sha256=report['preparation_sha256'])
                value['official_score'] = score_example(row['dataset'], generated['pred'], row['answers'], row['all_classes'])
                value['control_scores'] = {kind: records[i]['official_score'] for kind, records in record['controls'].items()}
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush(); os.fsync(handle.fileno())
                report['cases'].append({key: value[key] for key in ('dataset', 'row_index', 'input_ids_sha256',
                    'official_score', 'control_scores', 'elapsed_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')})
                report['sample_count'] += 1; write_json(path, report)
                logging.info('arm=%s task=%s row=%d completed=%d/16', arm, row['dataset'], row['row_index'], report['sample_count'])
        require(report['sample_count'] == 16 and frozen_inventory(model, targets) == report['restoration']['frozen_inventory'],
                'Inference changed non-target state/identity or omitted a case.')
        check_preparation(preparation)
        report.update(status='diagnostic_complete', exit_code=0, non_target_state_unchanged_after=True,
            input_pins_unchanged_after=True, predictions_sha256=sha256(output / 'predictions.jsonl'),
            selected_task_scores={task: {'restored': 100 * sum(c['official_score'] for c in report['cases'] if c['dataset'] == task) / 8,
                **{kind: 100 * sum(c['control_scores'][kind] for c in report['cases'] if c['dataset'] == task) / 8
                   for kind in record['controls']}} for task in TASKS})
        code = 0
    except BaseException as error:
        report.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        write_json(path, report); (output / 'run.exit').write_text(str(code) + '\n')
        del model, tokens; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(arm for arms in PROFILES.values() for arm in arms), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(args.preparation, args.arm)
