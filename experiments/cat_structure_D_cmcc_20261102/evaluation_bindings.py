"""Native-v6 consumers for the original decoder-width D comparison."""
from pathlib import Path
import sys

from . import bindings as study
from experiments.adaptive_class_evaluation_20261030 import bindings as common
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as shared
from experiments.longbench_v1 import models

FORMAT = 'vaellm_model_checkpoint_v6'


def source(directory, report_path):
    from .source import source as actual_source
    binding, meta = actual_source(directory, report_path)
    common.require(not binding['validation_only'] and meta['format'] == FORMAT
        and meta['checkpoint_kind'] == 'final_model' and meta['converted_module_count'] == 252
        and not meta['pending_dense_targets'] and binding['bit_budget']['budget_passed'],
        'Only the completed formal original-D native endpoint can enter complete quality evaluation.')
    return binding, meta


def check_config(config):
    models.check_config(config)
    record = config['model']
    common.require(record['kind'] == 'compressed' and record['checkpoint_format'] == FORMAT
        and record['recompression']['max_linear_bpw'] == 2.6
        and record['recompression']['max_lora_rank'] == 0,
        'Require the actual native rank0/logical2.6 model and unchanged official protocol.')
    common.check_hashes(record['checkpoint_files_sha256'])
    common.check_hashes(record['recompression']['source_sha256'])
    meta = common.read(Path(record['checkpoint']) / 'checkpoint_meta.json')
    common.require(meta['format'] == FORMAT and meta['checkpoint_id'] == record['checkpoint_id'],
        'The bound native checkpoint identity changed.')


def make_config(binding, pins):
    reference = common.read(common.REFERENCE)
    config = models.make_config(reference['archive']['path'], kind='compressed', hf_path=common.BASE,
        checkpoint=binding['checkpoint_dir'], mode='full')
    common.require(all(config[key] == reference[key] for key in
        ('schema', 'mode', 'protocol', 'model_policy', 'prewarm_group_size', 'device')),
        'Keep canonical BF16/FA2/FP32-RoPE/group8/32K generation conditions.')
    config['model'].update(checkpoint_format=FORMAT,
        checkpoint_files_sha256=binding['checkpoint_files_sha256'],
        recompression=dict(max_linear_bpw=2.6, max_lora_rank=0,
            bit_budget=binding['bit_budget'], source_sha256=pins))
    check_config(config)
    return config


def setup(support):
    shared.setup(support)
    original_pins = common.source_pins

    def source_pins(binding):
        return {**original_pins(binding), **study.pins(),
            str(study.REPO / 'experiments/cat_recompression/longbench.py'):
                study.sha(study.REPO / 'experiments/cat_recompression/longbench.py')}

    def command(component, *arguments):
        return [str(common.PYTHON), '-B', '-u', '-m',
            'experiments.cat_structure_D_cmcc_20261102.evaluate',
            '--support', str(support), '--component', component, *arguments]

    common.FORMAT = FORMAT
    common.source, common.source_pins, common.command = source, source_pins, command
    common.make_config, common.check_config = make_config, check_config


def load_model(config, logger):
    import torch
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    from train_utils.checkpoint_bpw import audit_state_dict
    check_config(config)
    common.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
        'Use exactly one assigned visible GPU for the native consumer.')
    torch.cuda.reset_peak_memory_stats()
    record = config['model']
    model, meta, loaded = load_v6_model_checkpoint(record['checkpoint'],
        base_model_path=record['hf_path'], map_location='cpu', strict=True)
    common.require(not loaded.missing_keys and not loaded.unexpected_keys
        and meta['checkpoint_id'] == record['checkpoint_id'] and meta['lora_config'] is None
        and not meta['pending_dense_targets'] and meta['converted_module_count'] == 252,
        'Require complete fresh strict native v6 without adapters or pending dense modules.')
    budget = audit_state_dict(meta, model.state_dict(), max_bpw=2.6, max_lora_rank=0)
    common.require(budget['budget_passed'] and not budget['contract_violations']
        and not budget['unclassified_target_keys']
        and all(budget[key] == record['recompression']['bit_budget'][key] for key in
            ('components', 'user_linear_bits', 'packed_linear_bits', 'outside_linear_bits',
             'original_linear_weights', 'code_padding_bits')),
        'The actual native storage ledger differs from the completed endpoint.')
    model.eval().requires_grad_(False)
    tools = common.module(study.SNAPSHOT / 'tools/cat_eval.py', '_original_D_exact_native_prewarm')
    warmup = tools._prepare_model_for_eval(model, config['device'], 8, logger, {})
    policy = config['model_policy']
    common.require(model.config.model_type == policy['model_type']
        and model.config.rope_scaling == policy['rope_scaling']
        and model.config.rope_theta == policy['rope_theta']
        and model.config._attn_implementation == policy['attn_implementation']
        and model.get_input_embeddings().weight.dtype == torch.bfloat16,
        'Keep the original BF16/attention/RoPE inference policy.')
    audit = dict(checkpoint_id=meta['checkpoint_id'], strict_missing=[], strict_unexpected=[],
        persistent_tensors=len(model.state_dict()), bit_budget=budget, warmup=warmup,
        model_policy=policy, rope_buffers={name:dict(dtype=str(value.dtype), shape=list(value.shape))
            for name,value in model.named_buffers() if 'inv_freq' in name},
        load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        load_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        evaluation_tools=dict(path=str(study.SNAPSHOT/'tools/cat_eval.py'),
            sha256=study.sha(study.SNAPSHOT/'tools/cat_eval.py')))
    common.require(audit['persistent_tensors'] == 3423 and warmup['warmed'] == warmup['total'] == 252
        and not warmup['failed'] and all(item['dtype'] == 'torch.float32'
            for item in audit['rope_buffers'].values()),
        'Require all252 native prewarms, complete3423 state and original FP32 RoPE.')
    audit.update(checkpoint_format=FORMAT,
        checkpoint_files_sha256=config['model']['checkpoint_files_sha256'])
    return model, audit
