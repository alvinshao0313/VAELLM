"""Explicit native-v6 plus lossless-Rice7 bundle; no unpacked indices are saved."""
from __future__ import annotations

import copy
from pathlib import Path

import sparse_common as c
from torch import nn

t = c.torch


def rice_state(values, names):
    return {f'blocks.{index}.{key}': values['modules'][name][key]
            for index, name in enumerate(names) for key in c.FIELDS}


class RiceState(nn.Module):
    def __init__(self, values, names):
        super().__init__()
        self.blocks = nn.ModuleList()
        for name in names:
            block = nn.Module()
            for key in c.FIELDS:
                block.register_buffer(key, t.empty_like(values['modules'][name][key]), persistent=True)
            self.blocks.append(block)


def package(directory, meta):
    values = t.load(Path(directory) / 'rice7_payload.pt', map_location='cpu', weights_only=True, mmap=True)
    c.require(values['schema'] == c.FORMAT + '_payload'
              and values['checkpoint_id'] == meta['checkpoint_id']
              and values['source_checkpoint_id'] == meta['source_checkpoint_id']
              and values['recipe'] == c.RECIPE and sorted(values['modules']) == meta['rice_module_names'],
              'The actual native and Rice shards must have the same new identity.')
    dtypes = (t.uint16, t.int32, t.uint8, t.float32, t.float32, t.uint8, t.uint32, t.uint8)
    for name, fields in values['modules'].items():
        c.require(set(fields) == set(c.FIELDS)
                  and all(fields[key].dtype == dtype for key, dtype in zip(c.FIELDS, dtypes, strict=True))
                  and fields['qvalues'].shape == (251659,) and fields['rice7_header'].shape == (16,)
                  and fields['block_ptr'].shape == fields['rice7_block_offsets'].shape
                  and fields['scales'].shape == fields['zero_points'].shape == fields['active_block_ids'].shape,
                  'Fixed abs005 Rice field schema differs: ' + name)
        for tensor in fields.values():
            c.compact_bytes(tensor)
    return values


def manifest(directory):
    root = Path(directory).resolve()
    meta, base = c.read(root / 'checkpoint_meta.json'), c.read(root / 'base/checkpoint_meta.json')
    c.require(meta['format'] == c.FORMAT and meta['checkpoint_kind'] == 'final_model'
              and meta['base_tensors'] == 3423 and meta['rice_tensors'] == 864 and meta['native_tensors'] == 4287
              and meta['unpacked_local_indices_persisted'] is False and meta['recipe'] == c.RECIPE
              and base['checkpoint_id'] == meta['checkpoint_id'] != meta['source_checkpoint_id']
              and base['converted_modules'] == meta['converted_modules']
              and base['lora_config'] is None and not base['pending_dense_targets'] and not base['skip_targets'],
              'Require the explicit complete raw-native/Rice bundle.')
    c.require(sorted(row['name'] for row in base['converted_modules'] if '.mlp.' in row['name'])
              == meta['rice_module_names'] and len(meta['rice_module_names']) == 108,
              'Native MLP inventory differs.')
    for relative, record in meta['files'].items():
        path = root / relative
        c.require(not Path(relative).is_absolute() and '..' not in Path(relative).parts
                  and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                  'Missing/nonlocal bundle file: ' + relative)
        found = c.binding(path)
        c.require({key: found[key] for key in ('bytes', 'sha256')} == record,
                  'Local bundle file changed: ' + relative)
    return root, meta, base


def ledger(base_meta, original, values, identifier):
    from train_utils.checkpoint_bpw import audit_state_dict
    result = copy.deepcopy(audit_state_dict(base_meta, original, max_bpw=2.6, max_lora_rank=0))
    c.require(len(original) == 3423 and result['budget_passed'] and not result['contract_violations']
              and not result['unclassified_target_keys'],
              'The whole actual raw native source must have a valid rank0 ledger.')
    per_module = {name: sum(c.compact_bytes(value) for value in fields.values())
                  for name, fields in values['modules'].items()}
    size = sum(per_module.values())
    result['components']['rice7_sparse_residual'] = dict(budget_bits=8 * size, storage_bits=8 * size,
        unique_storages=864, dtypes=sorted({str(value.dtype).removeprefix('torch.')
            for fields in values['modules'].values() for value in fields.values()}))
    for name, count in per_module.items():
        for key in ('budget_bits', 'storage_bits'):
            result['per_target'][name][key] += 8 * count
    for key in ('user_linear_bits', 'packed_linear_bits'):
        result[key] += 8 * size
    for prefix in ('user', 'packed'):
        result[prefix + '_linear_bpw'] = result[prefix + '_linear_bits'] / result['original_linear_weights']
    result['full_model_tensor_bytes'] += size
    result['checkpoint_id'] = identifier
    result['budget_passed'] = result['user_linear_bits'] * 5 <= result['original_linear_weights'] * 13
    return result


@t.no_grad()
def load_cpu_bundle(directory):
    c.native_context()
    root, meta, base_meta = manifest(directory)
    values = package(root, meta)
    from transformers import AutoConfig, Qwen3ForCausalLM
    from rotation.model_utils import temporarily_disable_torch_weight_init
    from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model
    config = AutoConfig.from_pretrained(str(root), local_files_only=True)
    config._attn_implementation = 'eager'
    with temporarily_disable_torch_weight_init():
        model = Qwen3ForCausalLM(config)
    model.seqlen = 2048
    model.config._attn_implementation = 'flash_attention_2'
    model, actual, loaded = load_v6_full_checkpoint_into_model(model, str(root / 'base'),
        expected_kind='final_model', map_location='cpu', strict=True)
    c.require(actual == base_meta and not loaded.missing_keys and not loaded.unexpected_keys,
              'The actual local native base must load strict.')
    original = t.load(root / 'base/pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    c.exact(model.state_dict(), original, 'Fresh native3423')
    carrier = RiceState(values, meta['rice_module_names'])
    flat = rice_state(values, meta['rice_module_names'])
    status = carrier.load_state_dict(flat, strict=True)
    c.require(not status.missing_keys and not status.unexpected_keys, 'Rice864 strict load failed.')
    c.exact(carrier.state_dict(), flat, 'Fresh Rice864')
    model.add_module('rice7_payload', carrier)
    expected = {**original, **{'rice7_payload.' + key: value for key, value in flat.items()}}
    c.exact(model.state_dict(), expected, 'Fresh joint4287')
    budget = ledger(base_meta, original, values, meta['checkpoint_id'])
    model.eval().requires_grad_(False)
    return model, meta, dict(checkpoint_id=meta['checkpoint_id'], checkpoint_loader=c.FORMAT,
        strict_missing=[], strict_unexpected=[], persistent_tensors=4287, native_main_banks=504,
        base_strict_tensors=3423, rice_strict_tensors=864, bit_budget=budget,
        HF_pretrained_weights_loaded=False, external_source_fallback=False,
        loaded_weight_files=[str(root / 'base/pytorch_model.bin'), str(root / 'rice7_payload.pt')])


@t.no_grad()
def prepare_model(model, directory, meta, logger):
    from tools.cat_eval import _prepare_model_for_eval
    from litebsq.sparse_residual import decode_blocked_quantized_sparse_residual
    from experiments.longbench_v1.protocol import model_policy
    c.require(t.cuda.is_available() and t.cuda.device_count() == 1, 'Use one explicitly assigned GPU.')
    warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logger, {})
    c.require(warmup['warmed'] == warmup['total'] == 252 and not warmup['failed'] and not warmup['skipped'],
              'Consume all252 native projections.')
    rice = c.module(Path(directory) / 'rice7.py', '_raw_sparse_local_rice7')
    records = {}
    for index, name in enumerate(meta['rice_module_names']):
        layer = model.get_submodule(name)
        fields = {key: getattr(model.rice7_payload.blocks[index], key).detach().cpu() for key in c.FIELDS}
        c.require(fields['scales'].dtype == fields['zero_points'].dtype == t.float32,
                  'Keep the original FP32 sparse scale and zero point.')
        decoded = {key: fields[key] for key in rice.UNCHANGED}
        decoded['local_indices'] = rice.decode(fields)
        rows, cols, patch = decode_blocked_quantized_sparse_residual(**decoded,
            out_features=layer.out_features, in_features=layer.in_features, block_rows=256, block_cols=256,
            index_bits=8, value_bits=8, value_dtype=t.bfloat16, device='cuda:0')
        original = layer._cached_weight
        protected = layer.protected_input_indices.to('cuda:0')
        c.require(original.dtype == patch.dtype == t.bfloat16 and patch.numel() == 251659
                  and not t.isin(cols, protected).any().item(), 'Patch count/protection differs: ' + name)
        value = original.clone()
        value.index_put_((rows, cols), patch, accumulate=True)
        c.require(t.equal(value[:, protected], original[:, protected]), 'Protected columns changed: ' + name)
        layer._cached_weight = value
        records[name] = c.tensor_record(value)
    policy = model_policy(meta['base_model_path'])
    c.require(model.config.model_type == policy['model_type'] and model.config.rope_theta == policy['rope_theta']
              and model.config.rope_scaling == policy['rope_scaling']
              and model.config._attn_implementation == policy['attn_implementation']
              and model.get_input_embeddings().weight.dtype == t.bfloat16, 'Original inference policy changed.')
    return warmup, records
