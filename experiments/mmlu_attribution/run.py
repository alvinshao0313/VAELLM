"""Run one paired MMLU restoration arm through the unchanged native evaluator."""
from contextlib import contextmanager
import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from experiments.longbench_v1.common import sha256, write_json
from experiments.longbench_attribution.restore import frozen_inventory, restore_targets
from experiments.mmlu_attribution.protocol import ARMS, case_identity, check_preparation, require, value_sha


@contextmanager
def observe_pair_encoding(observation):
    """Observe real superclass outputs; leave native encoding and likelihood untouched."""
    import lm_eval.models.huggingface as module
    original = module.HFLM

    class ObservedHFLM(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observation['hf_policy'] = {name: getattr(self, name) for name in
                ('backend', 'add_bos_token', 'prefix_token_id', 'max_length', 'truncation', 'logits_cache', 'batch_size')}
            observation['hf_policy']['tokenizer'] = {name: getattr(self.tokenizer, name) for name in
                ('bos_token_id', 'eos_token_id', 'pad_token_id', 'is_fast', 'padding_side')}

        def loglikelihood(self, requests, disable_tqdm=False):
            observation['request_order'] = [{'task': request.task_name, 'doc_id': request.doc_id,
                'choice_index': request.idx, 'arguments_sha256': value_sha(request.args)} for request in requests]
            return super().loglikelihood(requests, disable_tqdm=disable_tqdm)

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
            records = observation['pairs']
            for arguments, context, continuation in requests:
                key = value_sha(arguments)
                value = {'arguments_sha256': key, 'context_tokens': len(context),
                    'continuation_tokens': len(continuation), 'pair_tokens_sha256': value_sha([context, continuation]),
                    'native_left_truncated_tokens': max(0, len(context) + len(continuation) - (self.max_length + 1))}
                require(key not in records or records[key] == value, 'Native encoding differs for identical likelihood arguments.')
                records[key] = value
            return super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm, override_bs=override_bs)

        def _model_call(self, inps, attn_mask=None, labels=None):
            copied = inps.detach().cpu().contiguous()
            observation['batches'].append({'shape': list(inps.shape), 'dtype': str(inps.dtype),
                'input_ids_sha256': hashlib.sha256(copied.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'attention_mask_is_none': attn_mask is None, 'labels_is_none': labels is None})
            del copied
            return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

    module.HFLM = ObservedHFLM
    try:
        yield
    finally:
        module.HFLM = original


def load_model(arm, record, logger):
    hf_path = record['hf_path']
    policy = record['model_policy']
    if arm == 'dense':
        from rotation.model_utils import get_llama, get_qwen3
        loader = get_qwen3 if policy['model_type'] == 'qwen3' else get_llama
        model = loader(hf_path, None)
        model.eval().requires_grad_(False).to('cuda:0')
        report = {'loader': 'rotation.model_utils.' + loader.__name__, 'whole_model_dtype_cast': False}
    else:
        from train_utils.v6_model_loader import load_v6_model_checkpoint
        from tools.cat_eval import _prepare_model_for_eval
        model, meta, loaded = load_v6_model_checkpoint(record['source_checkpoint'], strict=True, map_location='cpu')
        require(meta['checkpoint_id'] == record['source_checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys,
                'Fresh compressed strict load failed.')
        model.eval().requires_grad_(False)
        warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logger, {})
        report = {'loader': 'v6_strict', 'checkpoint_id': meta['checkpoint_id'], 'warmup': warmup,
                  'persistent_tensors': len(model.state_dict())}
        require(report['persistent_tensors'] == record['expected_serialized_tensors'], 'Compressed source inventory changed.')
    model.config.use_cache = False
    require(model.config.model_type == policy['model_type'] and model.config.rope_scaling == policy['rope_scaling']
            and model.config.rope_theta == policy['rope_theta']
            and model.config._attn_implementation == policy['attn_implementation']
            and model.get_input_embeddings().weight.dtype == torch.bfloat16,
            'Original model-specific BF16/backend/RoPE policy changed.')
    rope = {name: str(value.dtype) for name, value in model.named_buffers() if 'inv_freq' in name}
    require(rope and set(rope.values()) == {'torch.float32'}, 'Original FP32 RoPE buffers must remain unrounded.')
    report.update(rope_buffers=rope, use_cache=False, parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
    return model, report


def verify_samples(raw_path, cases, pair_records):
    raw = json.loads(Path(raw_path).read_text())
    require(raw['sample_count'] == len(cases) == 1824, 'Missing complete diagnostic sample records.')
    expected = {(row['task'], row['doc_id']): row for row in cases}
    observed, compact = set(), []
    for row in raw['samples']:
        key = (row['task'], int(row['doc_id']))
        require(key in expected and key not in observed, 'Duplicate/unexpected native evaluation case.')
        sample = row['raw_sample']
        identity = case_identity(row['task'], row['doc_id'], sample['doc'], sample['arguments'])
        require(identity == expected[key], 'Runtime native doc/request arguments differ from the CPU preparation.')
        choices = [row['choice_scores'][label] for label in ('A', 'B', 'C', 'D')]
        require(all(math.isfinite(x) for x in choices) and row['target_index'] in range(4),
                'Missing finite native choice scores or official target.')
        observed.add(key)
        pair_keys = [value_sha(pair) for pair in identity['arguments']]
        require(all(pair in pair_records for pair in pair_keys), 'A request lacks actual HFLM pair encoding evidence.')
        target = row['target_index']
        compact.append({'task': key[0], 'doc_id': key[1], 'doc_sha256': identity['doc_sha256'],
            'arguments_sha256': identity['arguments_sha256'], 'target_index': target,
            'prediction_index': row['prediction_index'], 'correct': row['correct'], 'choice_scores': choices,
            'gold_margin': choices[target] - max(value for i, value in enumerate(choices) if i != target),
            'actual_pair_encodings': [pair_records[pair] for pair in pair_keys]})
    require(observed == set(expected), 'The seven arms must contain exactly the same paired documents.')
    return sorted(compact, key=lambda row: (row['task'], row['doc_id']))


def run(preparation, arm, output):
    record, cases = check_preparation(preparation)
    output = Path(output).resolve()
    require(arm in ARMS and not output.exists(), 'Use one declared arm and a fresh output.')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Bind exactly one root-assigned GPU.')
    for name, version in record['package_versions'].items():
        require(importlib.metadata.version(name) == version, 'Installed evaluation package changed.')
    output.mkdir(parents=True)
    report = {'status': 'running', 'arm': arm, 'protocol': record['protocol'],
        'preparation_sha256': sha256(preparation), 'source_checkpoint_id': record['source_checkpoint_id'],
        'pid': os.getpid(), 'start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'started_unix': time.time(), 'mixed_bf16_diagnostic_only': True, 'model_exported': False}
    write_json(output / 'report.json', report)
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(record['hf_path'], use_fast=True, local_files_only=True)
        model, report['load'] = load_model(arm, record, logging.getLogger(__name__))
        targets = record['selected_targets'][arm]
        require(targets == [name for name in record['architecture']['projection_shapes']
                            if name.rsplit('.', 1)[1] in ARMS[arm]], 'Restoration targets changed.')
        if targets:
            report['restoration'] = restore_targets(model, record['hf_path'], targets)
            before = report['restoration']['frozen_inventory']
        else:
            before = frozen_inventory(model, ())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        observation = {'pairs': {}, 'batches': [], 'request_order': []}
        from train_utils.eval_utils import run_lm_eval
        args = SimpleNamespace(tasks='mmlu', num_fewshot=0, batch_size=64, lm_limit=32,
            model_path=record['hf_path'], eval_log_dir=str(output), eval_run_ts='paired32',
            mmlu_debug_samples=1824, mmlu_debug_log_dir=str(output), mmlu_debug_run_ts='paired32')
        with observe_pair_encoding(observation):
            metrics = run_lm_eval(model, tokenizer, args)
        require(metrics['num_fewshot'] == 0 and metrics['limit'] == 32 and metrics['batch_size'] == 64
                and metrics['task_metric_keys']['mmlu'] == 'acc,none', 'Native evaluation protocol changed.')
        require(set(metrics['n_samples']) == set(record['subjects']), 'Missing MMLU subjects.')
        for name, subject in record['subjects'].items():
            require(metrics['n_samples'][name] == {'original': subject['original_samples'], 'effective': 32}
                    and metrics['n_shot'][name] == 0, 'Wrong diagnostic sample coverage/fewshot.')
        predictions = verify_samples(metrics['mmlu_debug']['path'], cases, observation['pairs'])
        correct = sum(row['correct'] for row in predictions)
        require(abs(correct / 1824 - metrics['task_metrics']['mmlu']) < 1e-12,
                'Native aggregate differs from the same selected per-example choices.')
        require(frozen_inventory(model, targets) == before, 'Inference changed a non-target tensor/buffer/identity.')
        check_preparation(preparation)
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'paired_choices.json', {'cases': predictions, 'actual_native_pair_observation': True})
        write_json(output / 'native_inputs.json', observation)
        report.update(status='diagnostic_complete', exit_code=0, correct=correct, sample_count=1824,
            accuracy_percent=correct * 100 / 1824, non_target_state_and_identity_unchanged=True,
            actual_unique_pair_encodings=len(observation['pairs']),
            paired_choices_sha256=sha256(output / 'paired_choices.json'), native_inputs_sha256=sha256(output / 'native_inputs.json'),
            metrics_sha256=sha256(output / 'metrics.json'),
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
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True, type=Path)
    parser.add_argument('--arm', choices=tuple(ARMS), required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(args.preparation, args.arm, args.output)
