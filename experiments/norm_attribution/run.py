"""One fresh norm candidate arm; unchanged native retrieval and MMLU inference."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.longbench_attribution.restore import frozen_inventory, tensor_digest
from experiments.longbench_v1.common import sha256, write_json
from experiments.longbench_v1.models import load_model
from experiments.longbench_v1.runner import generate_one, read_input
from experiments.longbench_v1.scoring import score_example
from experiments.mmlu_attribution.run import observe_pair_encoding, verify_samples
from experiments.mmlu_attribution.protocol import require
from .protocol import ARMS, check_preparation, require_mmlu_pairing


def restore_norms(model, record, arm):
    keys = record['selected_norms'][arm]
    excluded_modules = [key.rsplit('.', 1)[0] for key in keys]
    before = frozen_inventory(model, excluded_modules)
    source = torch.load(Path(record['restoration_source']) / 'pytorch_model.bin',
                        mmap=True, map_location='cpu', weights_only=True) if keys else {}
    restored = {}
    with torch.no_grad():
        for key in keys:
            target, value = model.get_parameter(key), source[key]
            require(list(target.shape) == list(value.shape) == record['norm_shapes'][key]
                    and target.dtype == value.dtype == torch.bfloat16 and not target.requires_grad,
                    'Norm copy representation differs: ' + key)
            identity = [id(target), target.data_ptr()]
            target.copy_(value.to(target.device))
            require(torch.equal(target.detach().cpu(), value) and identity == [id(target), target.data_ptr()],
                    'Norm copy changed identity or values.')
            restored[key] = {'source_tensor_sha256': tensor_digest(value), 'shape': list(value.shape),
                             'dtype': str(value.dtype), 'parameter_identity': [*identity, target._version]}
    del source
    return before, excluded_modules, restored


def retrieval(model, tokenizer, norm, reference, manifest, output, arm, preparation_sha):
    config = norm['config']
    seed = config['protocol']['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokens = np.memmap(manifest['inputs']['path'], dtype='<u4', mode='r')
    scores = []
    with (output / 'retrieval.jsonl').open('x', encoding='utf-8') as handle:
        for i, row in enumerate(norm['cases']):
            generated = generate_one(model, tokenizer, read_input(tokens, row), row,
                                     diagnostic=True, protocol=config['protocol'])
            prediction = {key: row[key] for key in ('dataset', 'row_index', '_id', 'answers', 'all_classes',
                'length', 'input_ids_sha256', 'input_tokens')}
            prediction.update(generated, diagnostic=True, arm=arm, preparation_sha256=preparation_sha)
            prediction['official_score'] = score_example(row['dataset'], generated['pred'], row['answers'], row['all_classes'])
            prediction['historical_control_scores'] = {kind: values[i]['official_score'] for kind, values in reference['controls'].items()}
            handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush(); os.fsync(handle.fileno())
            scores.append(prediction['official_score'])
            logging.info('arm=%s retrieval completed=%d/16', arm, i + 1)
    del tokens
    return {'sample_count': 16, 'correct': sum(scores), 'predictions_sha256': sha256(output / 'retrieval.jsonl')}


def mmlu_eval(model, tokenizer, mmlu, cases, output, control_directory):
    from train_utils.eval_utils import run_lm_eval
    model.config.use_cache = False
    observation = {'pairs': {}, 'batches': [], 'request_order': []}
    args = SimpleNamespace(tasks='mmlu', num_fewshot=0, batch_size=64, lm_limit=32,
        model_path=mmlu['hf_path'], eval_log_dir=str(output), eval_run_ts='paired32',
        mmlu_debug_samples=1824, mmlu_debug_log_dir=str(output), mmlu_debug_run_ts='paired32')
    with observe_pair_encoding(observation):
        metrics = run_lm_eval(model, tokenizer, args)
    require(metrics['num_fewshot'] == 0 and metrics['limit'] == 32 and metrics['batch_size'] == 64
            and metrics['task_metric_keys']['mmlu'] == 'acc,none'
            and set(metrics['n_samples']) == set(mmlu['subjects']), 'Native MMLU protocol changed.')
    for name, subject in mmlu['subjects'].items():
        require(metrics['n_samples'][name] == {'original': subject['original_samples'], 'effective': 32}
                and metrics['n_shot'][name] == 0, 'MMLU coverage/fewshot changed.')
    rows = verify_samples(metrics['mmlu_debug']['path'], cases, observation['pairs'])
    require_mmlu_pairing(observation, rows, control_directory)
    correct = sum(row['correct'] for row in rows)
    require(abs(correct / 1824 - metrics['task_metrics']['mmlu']) < 1e-12, 'MMLU aggregate disagrees with paired choices.')
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'paired_choices.json', {'cases': rows, 'actual_native_pair_observation': True})
    write_json(output / 'native_inputs.json', observation)
    return {'sample_count': 1824, 'correct': correct, 'accuracy_percent': correct * 100 / 1824,
        'actual_native_policy_inputs_match_control': True,
        'files_sha256': {name: sha256(output / name) for name in ('metrics.json', 'paired_choices.json', 'native_inputs.json')}}


def run(preparation, arm, output):
    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    record, norm, reference, manifest, mmlu, cases = check_preparation(preparation)
    require(arm in ARMS and not output.exists() and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, 'Require one assigned GPU and a new arm directory.')
    binding = json.loads((preparation.parent / 'binding.json').read_text())
    require(binding['status'] == 'bound' and type(binding['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(binding['gpu'])
            and binding['preparation_sha256'] == sha256(preparation), 'Bind the reviewed preparation before running.')
    output.mkdir(parents=True)
    report = {'status': 'running', 'arm': arm, 'preparation_sha256': sha256(preparation),
        'candidate_id': record['candidate_id'], 'restoration_source_id': record['restoration_source_id'],
        'pid': os.getpid(), 'start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'started_unix': time.time(), 'gpu': binding['gpu'], 'scope': record['scope'],
        'goal_acceptance': False, 'model_exported': False}
    write_json(output / 'report.json', report)
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer']['path'], use_fast=True, local_files_only=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        model, report['load_audit'] = load_model(norm['config'], logging.getLogger(__name__))
        require(all(value['dtype'] == 'torch.float32' for value in report['load_audit']['rope_buffers'].values()),
                'Original FP32 RoPE buffers changed.')
        before, excluded, restored = restore_norms(model, record, arm)
        report['restored_norms'] = restored
        report['frozen_state_count'] = len(before['state'])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if arm != 'norm400':
            report['retrieval'] = retrieval(model, tokenizer, norm, reference, manifest, output, arm, sha256(preparation))
        else:
            report['retrieval_reused_control'] = record['norm_retrieval_control']
        report['mmlu'] = mmlu_eval(model, tokenizer, mmlu, cases, output, record['mmlu_controls']['compressed'])
        require(frozen_inventory(model, excluded) == before, 'Inference/restoration changed a non-selected tensor or identity.')
        for key, saved in restored.items():
            value = model.get_parameter(key)
            require(tensor_digest(value) == saved['source_tensor_sha256']
                    and [id(value), value.data_ptr(), value._version] == saved['parameter_identity'],
                    'Inference changed a restored norm.')
        check_preparation(preparation)
        report.update(status='diagnostic_complete', exit_code=0, non_target_state_and_identity_unchanged=True,
            restored_norms_unchanged_after=True, input_pins_unchanged_after=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        write_json(output / 'report.json', report)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(**vars(parser.parse_args()))
