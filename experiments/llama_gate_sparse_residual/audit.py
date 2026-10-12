"""Exact frozen-source/payload accounting and fresh native BF16 local forward."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import read
from experiments.llama_gate_sparse_residual.prepare import BASE, BASE_BITS, DENOMINATOR, PROBE, check
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from litebsq.vae_linear_prewarm import prime_named_vae_linear_cache
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import load_v6_meta
from train_utils.v6_model_loader import load_v6_model_checkpoint

FIELDS = ('active_block_ids', 'block_ptr', 'local_indices', 'qvalues', 'scales', 'zero_points')


def state(path):
    meta = load_v6_meta(str(path))
    return meta, torch.load(Path(path) / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)


def frozen_and_budget(meta, current, record):
    old_meta, original = state(BASE)
    _require(len(original) == 2979 and set(original) <= set(current), 'Original complete state inventory differs.')
    _assert_state_equal({key: current[key].detach().cpu() for key in original}, original, 'e767 frozen tensors')
    expected = {name + '.sparse_residual_' + field for name in record['targets'] for field in FIELDS}
    _require(set(current) - set(original) == expected, 'Only the six native buffers per selected gate may be added.')
    prior = {item['name']: item for item in old_meta['converted_modules']}
    actual = {item['name']: item for item in meta['converted_modules']}
    _require(set(prior) == set(actual) and meta['compressed_targets'] == old_meta['compressed_targets']
             and not meta['pending_dense_targets'] and not meta['skip_targets'] and meta['lora_config'] is None,
             'The complete compressed topology changed.')
    payload = {}
    for name, item in actual.items():
        remove = (lambda spec: {key: value for key, value in spec.items() if not key.startswith('sparse_residual_')})
        _require(remove(item) == remove(prior[name]), 'Non-sparse module metadata changed: ' + name)
        if name not in record['targets']:
            _require(item == prior[name], 'Unselected module metadata changed: ' + name)
            continue
        values = {field: current[name + '.sparse_residual_' + field] for field in FIELDS}
        n = values['qvalues'].numel(); blocks = values['active_block_ids'].numel()
        _require(item['sparse_residual_format'] == 'blocked_quantized'
                 and item['sparse_residual_index_bits'] == item['sparse_residual_value_bits'] == 8
                 and item['sparse_residual_block_rows'] == item['sparse_residual_block_cols'] == 256
                 and 0 < n <= record['budget_estimate']['maximum_nnz_per_gate'] and 0 < blocks <= 896
                 and values['active_block_ids'].dtype == torch.uint16
                 and values['block_ptr'].dtype == torch.int32 and values['block_ptr'].numel() == blocks + 1
                 and values['local_indices'].dtype == values['qvalues'].dtype == torch.uint8
                 and values['local_indices'].numel() == 2 * n
                 and values['scales'].dtype == values['zero_points'].dtype == torch.float32
                 and values['scales'].numel() == values['zero_points'].numel() == blocks,
                 'Native blocked index8/value8 payload differs: ' + name)
        size = sum(value.untyped_storage().nbytes() for value in values.values())
        _require(size == 3 * n + 14 * blocks + 4, 'Native serialized codec overhead differs: ' + name)
        payload[name] = {'nnz': n, 'active_blocks': blocks, 'bytes': size}
    budget = audit_state_dict(meta, current, max_bpw=2.5, max_lora_rank=0)
    total = sum(item['bytes'] for item in payload.values())
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['original_linear_weights'] == DENOMINATOR and budget['compressed_target_count'] == 224
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == BASE_BITS + 8 * total,
             'Actual native checkpoint storage exceeds the recipe or budget.')
    old_components = record['base_budget']['components']; new_components = budget['components']
    _require(set(new_components) == set(old_components) | {'sparse_residual'}
             and new_components['sparse_residual']['storage_bits'] == 8 * total
             and all(new_components[key]['storage_bits'] == value['storage_bits'] for key, value in old_components.items()),
             'Existing code/decoder/protection/rotation storage changed.')
    return {'original_frozen_tensors': len(original), 'added_tensors': len(expected),
            'saved_tensors': len(current), 'payload': payload, 'bit_budget': budget}


def local_forward(module, x):
    module.clear_decoded_weight_cache()
    prime_named_vae_linear_cache([(PROBE, module)], compute_device=torch.device('cuda:0'),
                                dtype=torch.bfloat16, group_size=8)
    with torch.inference_mode():
        result = module(x.to('cuda:0')).detach().cpu()
    _require(result.dtype == torch.bfloat16 and tuple(result.shape) == (4, 14336)
             and bool(torch.isfinite(result).all()), 'Native BF16 gate forward produced invalid output.')
    return result


def verify(record):
    check(record); output = Path(record['output_dir']); checkpoint = output / 'final_model'
    meta, saved = state(checkpoint); report = frozen_and_budget(meta, saved, record)
    model, fresh_meta, _ = load_v6_model_checkpoint(str(checkpoint), strict=True, map_location='cpu')
    model.eval().requires_grad_(False)
    compared = _assert_state_equal(model.state_dict(), saved, 'fresh strict native reload')
    probe = torch.load(output / 'local_probe.pt', weights_only=True, map_location='cpu')
    module = model.get_submodule(PROBE).to('cuda:0')
    after = local_forward(module, probe['input'])
    _require(torch.equal(after, probe['after']), 'Fresh native BF16 local output differs from saved probe.')
    module.clear_decoded_weight_cache(); module.to('cpu')
    _assert_state_equal(model.state_dict(), saved, 'after native BF16 forward')
    _require(not any(parameter.requires_grad for parameter in model.parameters()), 'Unexpected trainable parameter.')
    report.update(status='verified', scope=record['scope'], checkpoint_id=fresh_meta['checkpoint_id'],
        base_checkpoint_id=record['base_checkpoint_id'], strict_reloaded_tensors=compared,
        trainable_parameters=0, residual_target=record['residual_target'], decode_policy=record['decode_policy'],
        native_before_after_local_forward=read(output / 'build_report.json')['local_forward'],
        fresh_native_bf16_local_forward={'shape': list(after.shape), 'dtype': str(after.dtype),
            'finite': True, 'matches_saved_probe': True, 'input': 'fixed seed31 four CPU Gaussian vectors, BF16'},
        full_model_hybrid_evaluation=False, local_probe_sha256=sha256(output / 'local_probe.pt'),
        metadata_sha256=sha256(checkpoint / 'checkpoint_meta.json'))
    write_json(output / 'audit.json', report)
    print({'status': report['status'], 'frozen': 2979, 'added': report['added_tensors'],
           'user_linear_bpw': report['bit_budget']['user_linear_bpw'], 'fresh_bf16_forward': True}, flush=True)
