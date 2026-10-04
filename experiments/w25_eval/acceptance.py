"""Qwen/Llama BF16 baselines; Llama-only strict low-bit relative acceptance."""
from __future__ import annotations

import json
import math
from pathlib import Path

from lm_eval.utils import simple_parse_args_string

from experiments.qwen3_w25.summarize_results import (
    DATASET_CONFIGS, MMLU_SAMPLE_COUNTS, SAMPLE_COUNTS, TASK_METRICS, _require, _score,
)
from experiments.w25_eval.model_inventory import model_inventory, sha256, verify_source_manifest
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import validate_v6_meta

INFERENCE_BACKENDS = {'llama': ('sdpa', 'rotation.model_utils.get_llama'),
                      'qwen3': ('flash_attention_2', 'rotation.model_utils.get_qwen3')}


def read_json(path):
    value = json.loads(Path(path).read_text())
    _require(isinstance(value, dict), f'Expected object: {path}')
    return value


def _check_goal_scope(model_type, paired):
    _require(not paired or model_type == 'llama',
             'The <=1 percentage-point paired goal is Llama-only; use summarize_mmlu_target.py for Qwen.')


def leaf_metrics(raw, counts, shots, configs):
    """Validate one of the exact MMLU/seven shards and return task scores."""
    leaves = set(counts)
    is_mmlu = leaves == set(MMLU_SAMPLE_COUNTS)
    _require(is_mmlu or leaves == set(SAMPLE_COUNTS), 'Require exactly 57 MMLU or seven standard leaves.')
    _require(set(configs) == leaves and set(shots) == leaves, 'Task configs/few-shot inventory differs from leaves.')
    scores, subjects = {}, {}
    for task in sorted(leaves):
        expected_count = (MMLU_SAMPLE_COUNTS if is_mmlu else SAMPLE_COUNTS)[task]
        _require(counts[task] == {'original': expected_count, 'effective': expected_count},
                 f'Non-full sample count: {task}')
        _require(shots.get(task) == 0, f'Nonzero/missing few-shot count: {task}')
        cfg = configs[task]
        expected_config = (('hails/mmlu_no_train', task.removeprefix('mmlu_'), 'test')
                           if is_mmlu else DATASET_CONFIGS[task])
        _require((cfg.get('dataset_path'), cfg.get('dataset_name'),
                  cfg.get('test_split') or cfg.get('validation_split')) == expected_config
                 and cfg.get('output_type') == 'multiple_choice', f'Wrong task configuration: {task}')
        key = 'acc,none' if is_mmlu else TASK_METRICS[task]
        value = _score(raw[task].get(key), f'{task}/{key}')
        (subjects if is_mmlu else scores)[task] = value
    if is_mmlu:
        scores['mmlu'] = math.fsum(subjects[t] * MMLU_SAMPLE_COUNTS[t] for t in subjects) / 14042
    return scores, subjects, 64 if is_mmlu else 32


def _group_agrees(scores, raw, groups):
    if 'mmlu' not in scores:
        return
    values = [g['mmlu'].get('acc,none') for g in (raw, groups) if 'mmlu' in g]
    _require(values, 'Missing MMLU group score.')
    for value in values:
        _require(math.isclose(_score(value, 'MMLU group'), scores['mmlu'], rel_tol=0, abs_tol=1e-12),
                 'MMLU group differs from all 57 subjects weighted by standard sample count.')


def _hf_identity(payload, inventory):
    config = payload['config']
    args = config.get('model_args')
    args = simple_parse_args_string(args) if isinstance(args, str) else args
    _require(isinstance(args, dict), 'HF raw must include actual model arguments.')
    base = Path(inventory['base_model_path'])
    _require(config.get('model') == 'hf' and Path(args.get('pretrained', '')).resolve() == base,
             'HF baseline points to a different model or backend.')
    _require(Path(args.get('tokenizer', str(base))).resolve() == base
             and args.get('use_fast_tokenizer') is True and args.get('add_bos_token') is False,
             'HF baseline must explicitly use the same fast tokenizer without automatic BOS.')
    _require(args.get('dtype') in ('bfloat16', 'bf16') and config.get('model_dtype') == 'torch.bfloat16'
             and config.get('model_num_parameters') == inventory['dense_parameter_elements'],
             'Baseline must be the actual complete, unquantized BF16 model.')
    _require(args.get('use_cache') is False, 'Baseline must disable KV cache.')
    _require(args.get('attn_implementation') == INFERENCE_BACKENDS[inventory['model_type']][0],
             'Baseline attention backend differs from the actual model-specific loader policy.')
    _require(not any(args.get(k) for k in ('peft', 'delta', 'autogptq', 'gptqmodel', 'load_in_4bit',
                                          'load_in_8bit', 'quantization_config', 'gguf_file', 'parallelize')),
             'Quantized/adapter/parallel baseline is outside this protocol.')
    for key in ('chat_template', 'chat_template_sha', 'system_instruction', 'system_instruction_sha'):
        _require(key in payload and payload[key] is None, f'Baseline must use bare benchmark prompts: {key}')
    _require(payload.get('fewshot_as_multiturn') is False, 'Unexpected multiturn prompting.')
    for key in ('bos', 'eos'):
        _require(payload.get(f'tokenizer_{key}_token', [None, None])[1]
                 == str(inventory[f'tokenizer_{key}_token_id']), f'Baseline tokenizer {key} ID differs.')


def collect_results(paths, inventory, *, checkpoint=None):
    _require(len(paths) == 2, 'Exactly two result shards are required.')
    scores, subjects, sources, versions = {}, {}, [], {}
    for path in paths:
        path = Path(path).resolve()
        payload = read_json(path)
        if checkpoint is None:
            _hf_identity(payload, inventory)
            config = payload['config']
            _require('limit' in config and config['limit'] is None, 'Baseline limit must be null.')
            raw, groups = payload['results'], payload.get('groups', {})
            counts, shots, configs = payload['n-samples'], payload['n-shot'], payload['configs']
            actual_batch = config['batch_size']
            version = payload['versions']
        else:
            directory, meta = checkpoint
            _require(Path(payload['checkpoint_dir']).resolve() == directory
                     and payload['checkpoint_id'] == meta['checkpoint_id'], 'Wrong compressed model identity.')
            _require(Path(payload['base_model_path']).resolve() == Path(inventory['base_model_path']),
                     'Compressed tokenizer/base differs from the baseline.')
            _require(payload.get('checkpoint_loader') == 'v6' and payload.get('strict_load') is True
                     and payload.get('adapter_dir') is None and payload.get('eval_hif4_act') is False,
                     'Require strict independent weight-only v6 evaluation without adapter.')
            _require(payload.get('tokenizer_class') == inventory['tokenizer_class']
                     and payload.get('tokenizer_is_fast') is True, 'Compressed tokenizer class/fast mode differs.')
            result = payload['evals']['lm_eval']
            _require(result.get('num_fewshot') == 0 and 'limit' in result and result['limit'] is None,
                     'Compressed evaluation must be complete zero-shot.')
            raw, groups = result['raw_results'], result.get('group_results', {})
            counts, shots, configs = result['n_samples'], result['n_shot'], result['configs']
            actual_batch = result['batch_size']
            version = result['versions']
        current, sub, required_batch = leaf_metrics(raw, counts, shots, configs)
        _require(str(actual_batch) == str(required_batch), 'Require explicit actual MMLU64/seven32 batch size.')
        _require(not set(scores) & set(current), 'Duplicate result shard.')
        _group_agrees(current, raw, groups)
        if checkpoint is not None:
            _require(set(result['tasks']) == set(current) and len(result['tasks']) == len(current)
                     and set(result['task_metrics']) == set(current)
                     and set(result['task_metric_keys']) == set(current), 'Cat summary task inventory differs.')
            for task, value in current.items():
                _require(result['task_metric_keys'][task] == TASK_METRICS[task]
                         and math.isclose(_score(result['task_metrics'][task], task), value,
                                          rel_tol=0, abs_tol=1e-12), 'Cat summary differs from raw metrics.')
        scores.update(current)
        subjects.update(sub)
        versions.update({task: version[task] for task in counts})
        sources.append({'path': str(path), 'sha256': sha256(path), 'tasks': list(current),
                        'leaf_count': len(counts), 'sample_count': sum(v['effective'] for v in counts.values()),
                        'batch_size': required_batch,
                        'format': 'standard_hf_cli' if checkpoint is None else 'cat_eval_v6'})
    _require(set(scores) == set(TASK_METRICS) and len(subjects) == 57, 'Incomplete eight-task evaluation.')
    _require(sum(s['leaf_count'] for s in sources) == 64
             and sum(s['sample_count'] for s in sources) == 24742, 'Incomplete full evaluation coverage.')
    return {'task_metrics': scores, 'task_metric_keys': TASK_METRICS,
            'mean_percent': math.fsum(scores[t] for t in TASK_METRICS) * 100 / 8,
            'mmlu_subject_metrics': subjects, 'versions': versions, 'input_results': sources,
            'leaf_count': 64, 'sample_count': 24742}


def validate_checkpoint(directory, inventory):
    directory = Path(directory).resolve()
    meta = validate_v6_meta(read_json(directory / 'checkpoint_meta.json'), expected_kind='final_model')
    expected = inventory['projection_shapes']
    _require(Path(meta['base_model_path']).resolve() == Path(inventory['base_model_path']), 'Wrong v6 base model.')
    _require(set(meta['compressed_targets']) == set(expected) and not meta['pending_dense_targets']
             and not meta['skip_targets'], 'All real model projections must be compressed.')
    for spec in meta['converted_modules']:
        name = spec['name']
        _require([spec['out_features'], spec['in_features']] == expected[name], f'Wrong projection shape: {name}')
        _require(not spec.get('always_use_original') and not spec.get('has_original_weight')
                 and spec.get('low_rank_a') is None and spec.get('low_rank_b') is None,
                 f'No dense bypass or external LoRA allowed: {name}')
        dims = spec.get('stage_codebook_dims') or [spec['codebook_dim']]
        _require(all(type(d) is int and d > 0 for d in dims), f'Invalid codebook dimensions: {name}')
        for axis in ('input', 'output'):
            indices = spec.get(f'protected_{axis}_indices')
            if indices is not None:
                _require(all(math.prod(indices['shape']) % d == 0 for d in dims),
                         f'Protected channels must be codebook_dim multiples: {name}')
    audit = audit_checkpoint(directory, max_bpw=2.5, max_lora_rank=0)
    _require(audit['compressed_target_count'] == len(expected)
             and audit['original_linear_weights'] == inventory['original_linear_weights'],
             'Measured storage inventory differs from the actual architecture.')
    _require(not any(audit['components'].get(k, {}).get('storage_bits', 0)
                     for k in ('lora', 'residual_lora', 'dense_weights')), 'Forbidden LoRA/dense projection storage.')
    return directory, meta, audit


def summarize(base_model_path, source_manifest, baseline_results, *, checkpoint_dir=None, compressed_results=None):
    _require((checkpoint_dir is None) == (compressed_results is None), 'Supply checkpoint and compressed shards together.')
    inventory = model_inventory(base_model_path)
    _check_goal_scope(inventory['model_type'], checkpoint_dir is not None)
    source = verify_source_manifest(source_manifest, inventory)
    baseline = collect_results(baseline_results, inventory)
    report = {'status': 'baseline_verified', 'quality_target_evaluated': False,
              'goal_scope': 'Llama relative mean drop <=1pp; Qwen baseline mode does not evaluate its accuracy goal.',
              'architecture': inventory, 'source_identity': source,
              'inference_backend': {
                  'expected': INFERENCE_BACKENDS[inventory['model_type']][0],
                  'policy_source': INFERENCE_BACKENDS[inventory['model_type']][1],
                  'policy_source_sha256': sha256(Path(__file__).resolve().parents[2] / 'rotation/model_utils.py'),
                  'baseline_model_args_verified': True,
                  'compressed_raw_backend_observed': False,
                  'compressed_backend_evidence_limit': 'cat_eval raw does not store backend. Strict v6/base identity uses the existing model-specific loader policy; no raw backend field is inferred.',
                  'meta_inventory_sdpa_is_shape_only': True},
              'baseline': {'representation': 'unquantized_bfloat16', **baseline},
              'protocol': {'num_fewshot': 0, 'limit': None, 'mmlu_batch_size': 64,
                           'seven_batch_size': 32, 'leaf_count': 64, 'sample_count': 24742,
                           'mmlu_aggregation': '57 subjects weighted by standard test sample counts',
                           'max_absolute_mean_drop_percentage_points': 1.0,
                           'user_linear_bpw_strictly_less_than': 2.5, 'lora_rank': 0}}
    if checkpoint_dir is None:
        return report
    directory, meta, audit = validate_checkpoint(checkpoint_dir, inventory)
    compressed = collect_results(compressed_results, inventory, checkpoint=(directory, meta))
    _require(compressed['versions'] == baseline['versions'], 'Task versions differ from the dense baseline.')
    drop = baseline['mean_percent'] - compressed['mean_percent']
    accuracy = drop <= 1.0
    budget = audit['budget_passed'] and audit['user_linear_bpw'] < 2.5
    report.update(status='passed' if accuracy and budget else 'not_met', passed=accuracy and budget,
                  quality_target_evaluated=True, compressed=compressed, checkpoint_id=meta['checkpoint_id'],
                  checkpoint_dir=str(directory), checkpoint_meta_sha256=sha256(directory / 'checkpoint_meta.json'),
                  bit_budget=audit, mean_drop_percentage_points=drop,
                  task_delta_percentage_points={t: 100 * (compressed['task_metrics'][t] - baseline['task_metrics'][t])
                                                for t in TASK_METRICS},
                  accuracy_passed=accuracy, budget_passed=budget,
                  failure_reasons=([] if accuracy else ['Eight-task mean drop exceeds 1 percentage point.'])
                  + ([] if budget else ['Actual user Linear bpw must be strictly below 2.5 and storage contract valid.'])
                  + audit['contract_violations'])
    return report
