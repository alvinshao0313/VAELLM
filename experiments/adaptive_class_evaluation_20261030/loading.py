"""Load the real mixed model and preserve native group8 prewarming."""
import torch

from experiments.adaptive_native_model_20261029 import checkpoint
from experiments.adaptive_class_evaluation_20261030 import bindings as b


def load_model(config, logger):
    b.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one assigned visible GPU is required.')
    b.check_config(config)
    torch.cuda.reset_peak_memory_stats()
    record, policy = config['model'], config['model_policy']
    model, meta, loaded, budget = checkpoint.load(record['checkpoint'])
    b.require(not loaded.missing_keys and not loaded.unexpected_keys and meta['checkpoint_id'] == record['checkpoint_id']
              and all(budget[key] == record['mixed_runtime']['bit_budget'][key] for key in b.BUDGET_FIELDS),
              'Fresh strict mixed load or its measured storage differs.')
    tools_path = b.RUNTIME / 'tools/cat_eval.py'
    tools = b.module(tools_path, '_adaptive_exact_native_cat_eval_tools')
    warmup = tools._prepare_model_for_eval(model, config['device'], 8, logger, {})
    b.require(warmup['total'] == warmup['warmed'] == len(meta['native_modules']) and not warmup['failed'],
              'All actual native modules must complete the original group8 prewarm.')
    b.require(model.config.model_type == policy['model_type'] and model.config.rope_scaling == policy['rope_scaling']
              and model.config.rope_theta == policy['rope_theta']
              and model.config._attn_implementation == policy['attn_implementation']
              and model.get_input_embeddings().weight.dtype == torch.bfloat16, 'Original BF16/RoPE/backend policy differs.')
    audit = dict(checkpoint_id=meta['checkpoint_id'], checkpoint_format=meta['format'], strict_missing=[], strict_unexpected=[],
        persistent_tensors=len(model.state_dict()), bit_budget=budget, warmup=warmup, model_policy=policy,
        checkpoint_files_sha256=record['checkpoint_files_sha256'],
        evaluation_tools=dict(path=str(tools_path), sha256=b.sha256(tools_path)),
        compact_consumer_classes={item['name']: type(model.get_submodule(item['name'])).__name__ for item in meta['compact_modules']},
        parameter_dtypes=sorted({str(value.dtype) for value in model.parameters()}),
        rope_buffers={name: dict(dtype=str(value.dtype), shape=list(value.shape))
                      for name, value in model.named_buffers() if 'inv_freq' in name},
        attn_implementation=model.config._attn_implementation, rope_scaling=model.config.rope_scaling,
        load_peak_allocated_bytes=torch.cuda.max_memory_allocated(), load_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    b.require(all(row['dtype'] == 'torch.float32' for row in audit['rope_buffers'].values()), 'Original RoPE buffers must remain FP32.')
    return model, audit
