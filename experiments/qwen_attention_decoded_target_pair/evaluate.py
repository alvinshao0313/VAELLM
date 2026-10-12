"""Existing actual-union INT4 codec, replay, native group8 and complete eight tasks."""
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from experiments.qwen_attention_decoded_target_pair.common import read, check, _require
from experiments.qwen_attention_int4_pair.common import SHARDS, verify_shard, summarize
from experiments.qwen_attention_int4_pair.combination import sidecar
from experiments.protected_int4_quality.protection import replay, immutable_inventory, verify_native_cache
from experiments.longbench_v1.common import write_json


def evaluate(model, meta, record, endpoint):
    from tools.cat_eval import _prepare_model_for_eval
    from train_utils.eval_utils import run_lm_eval
    output = Path(record['output_dir'])
    runtime, payload = sidecar(model.state_dict(), meta, record, endpoint['actual_native_union_budget'])
    replay_report, frozen_cpu = replay(model, runtime)
    model.eval().requires_grad_(False); model.config.use_cache = False
    report = {'status': 'simulation', 'arm': record['arm'], 'base_checkpoint_id': record['base_checkpoint_id'],
        'endpoint_id': endpoint['endpoint_id'], 'model_exported': False, 'native_int4_checkpoint': False,
        'goal_acceptance': False, 'payload': payload, 'replay': replay_report,
        'warmup': _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})}
    policy = record['model_policy']
    _require(model.config.model_type == policy['model_type']
             and model.config._attn_implementation == policy['attn_implementation']
             and model.config.rope_scaling == policy['rope_scaling']
             and model.config.rope_theta == policy['rope_theta']
             and model.get_input_embeddings().weight.dtype == torch.bfloat16, 'Native backend/precision differs.')
    rope = {key: str(value.dtype) for key, value in model.named_buffers() if 'inv_freq' in key}
    _require(rope and set(rope.values()) == {'torch.float32'}, 'FP32 RoPE changed.')
    frozen_gpu = immutable_inventory(model, runtime['selected_targets'])
    _require(frozen_cpu['state'] == frozen_gpu['state'] and frozen_cpu['buffers'] == frozen_gpu['buffers'],
             'Transfer changed frozen persistent state.')
    report['native_cache_validation'] = verify_native_cache(model, runtime)
    account = payload['accounting']
    expanded = (account['expanded_simulation_linear_bits'] - account['source_user_linear_bits']
                + account['replaced_native_qvalues_scales_bits'])
    _require(report['native_cache_validation']['expanded_protected_weight_storage_bits'] == expanded,
             'Actual expanded runtime storage differs.')
    tokenizer = AutoTokenizer.from_pretrained(record['hf_path'], use_fast=True, local_files_only=True)
    ids = tokenizer('A short test sentence.', return_tensors='pt', add_special_tokens=False)['input_ids'][:, :4]
    _require(ids.shape == (1, 4), 'Require the native four-token integration probe.')
    with torch.inference_mode():
        logits = model(input_ids=ids.to('cuda:0'), use_cache=False).logits
    _require(logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all()), 'Native BF16 integration failed.')
    report['integration_probe'] = {'input_shape': list(ids.shape), 'output_shape': list(logits.shape),
        'dtype': str(logits.dtype), 'finite': True, 'model_policy': policy, 'rope_buffers': rope}
    del ids, logits, frozen_cpu
    write_json(output / 'report.json', report)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for shard, tasks, batch in SHARDS:
        args = SimpleNamespace(tasks=tasks, num_fewshot=0, batch_size=batch, lm_limit=None,
            model_path=record['hf_path'], eval_log_dir=str(output), eval_run_ts=shard, mmlu_debug_samples=0)
        metrics = run_lm_eval(model, tokenizer, args)
        verify_shard(metrics, read(record['control_result_paths'][shard])['evals']['lm_eval'], batch)
        report.setdefault('completed_shards', []).append(shard); write_json(output / 'report.json', report)
    _require(immutable_inventory(model, runtime['selected_targets']) == frozen_gpu,
             'Inference changed frozen state or runtime identity.')
    report['final_native_cache_validation'] = verify_native_cache(model, runtime)
    check(record)
    report.update(status='diagnostic_complete', **summarize(output, record), all_non_protection_state_unchanged=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        longbench_status='not_evaluated', nine_item_goal_status='not_evaluated')
    report['primary_comparison'] = 'decoded minus NEW hf target with identical fixed plan and RNG boundary'
    write_json(output / 'report.json', report)
    return report
