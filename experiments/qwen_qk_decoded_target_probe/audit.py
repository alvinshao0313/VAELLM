"""Native save/fresh and grouped-decode audit for the isolated Q/K replacement."""
from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.full_rht_gate_pair.common import read
from experiments.longbench_v1.common import sha256
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.qwen_qk_decoded_target_probe.core import NAMES, PROTECTION
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import audit_checkpoint
from litebsq.vae_linear_prewarm import decode_named_vae_linear_weights
from train_utils.checkpoint_v6 import load_v6_meta, save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def decoded_records(model, device):
    """Use the same category/name ordering and group8 as native deployment."""
    layers = [(name, model.get_submodule(name)) for name in NAMES]
    with torch.no_grad():
        values = decode_named_vae_linear_weights(layers, dtype=torch.bfloat16,
            group_size=8, compute_device=device, respect_cache_policy=False)
    records = {}
    for item in values:
        expected = (4096 if item.name.endswith('.q_proj') else 1024, 4096)
        _require(item.decoded_weight.dtype == torch.bfloat16 and tuple(item.decoded_weight.shape) == expected
                 and bool(torch.isfinite(item.decoded_weight).all()), 'Invalid native original-coordinate weight.')
        records[item.name] = tensor_record(item.decoded_weight)
    _require(set(records) == set(NAMES), 'Incomplete native72 decode.')
    del values, item
    for _, layer in layers:
        layer.clear_decoded_weight_cache(); layer.to('cpu')
    del layers, layer
    gc.collect(); torch.cuda.empty_cache()
    return records


def save_and_audit(model, base_state, record, output, device):
    started = time.time(); output = Path(output); checkpoint = output / 'final_model'
    _require(not checkpoint.exists() and len(base_state) == 3423, 'Use a new final checkpoint and complete f927 state.')
    model.to('cpu').eval()
    prefixes = tuple(name + '.' for name in NAMES)
    state = model.state_dict()
    local = {key: value for key, value in state.items() if key.startswith(prefixes)}
    outside = {key: value for key, value in state.items() if not key.startswith(prefixes)}
    source_outside = {key: value for key, value in base_state.items() if not key.startswith(prefixes)}
    _require(len(state) == 3423 and len(local) == 936 and len(outside) == 2487
             and set(state) == set(base_state), 'Unexpected complete or selected persistent inventory.')
    _assert_state_equal(outside, source_outside, 'Untouched180 projections and f927 backbone')
    geometry = read(record['source_geometry_path'])
    protection, signs = {}, {}
    kind = 'f927' if record['target_kind'] == 'decoded' else 'hf_fresh_block32'
    for name in NAMES:
        expected = geometry['projections'][name]
        protection[name] = {field: tensor_record(state[name + '.' + field]) for field in PROTECTION}
        for field in PROTECTION:
            _require(protection[name][field] == expected['protection'][field][kind],
                     'Native protection does not match the actual target: ' + name + '/' + field)
        signs[name] = {field: tensor_record(state[name + '.weight_rotation.' + field])
                       for field in ('input_signs', 'output_signs')}
        for field in signs[name]:
            _require(signs[name][field] == expected['signs'][field]['f927']
                     == expected['signs'][field]['hf_fresh_block32'], 'Rotation signs changed: ' + name)
        for key, value in local.items():
            if key.startswith(name + '.') and value.is_floating_point():
                _require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()),
                         'Native deployment payload must be finite BF16: ' + key)
    before = decoded_records(model, device)
    _assert_state_equal(model.state_dict(), state, 'Grouped decoding must preserve persistent trained state')
    source_meta = load_v6_meta(record['base_dir'])
    provenance = {'source_checkpoint_dir': record['base_dir'], 'source_checkpoint_id': record['base_checkpoint_id'],
        'target_kind': record['target_kind'], 'target_materialization': 'training.json',
        'block_size': record['block_size'], 'categories': ['q_proj', 'k_proj'],
        'replaced_targets': list(NAMES), 'steps_per_stage': record['steps_per_stage'],
        'source_geometry_sha256': sha256(record['source_geometry_path']),
        'purpose': 'Isolated fresh native CAT target adapter, inference-only full checkpoint; not CAT resume.'}
    tokenizer = AutoTokenizer.from_pretrained(source_meta['base_model_path'], use_fast=True, local_files_only=True)
    saved = save_v6_full_checkpoint(model, str(checkpoint), checkpoint_kind='final_model',
        compressed_targets=source_meta['compressed_targets'], pending_dense_targets=(), skip_targets=(),
        train_mode='none', after_category_mode='none', lora_config=None,
        base_model_path=source_meta['base_model_path'], tokenizer=tokenizer,
        extra_meta={'decoded_target_adapter': provenance})
    _require(saved['checkpoint_id'] != record['base_checkpoint_id']
             and saved['meta_payload'].get('immutable_resume_contract') is None,
             'A new inference identity must not pretend to resume old CAT.')
    serialized = torch.load(saved['state_dict'], map_location='cpu', weights_only=True, mmap=True)
    serialized_count = _assert_state_equal(serialized, state, 'Native complete saved state')
    reloaded, meta, _ = load_v6_model_checkpoint(str(checkpoint), map_location='cpu', strict=True,
                                               expected_kind='final_model')
    reloaded_count = _assert_state_equal(reloaded.state_dict(), state, 'Independent native v6 fresh reload')
    specs = {s['name']: s for s in meta['converted_modules']}
    old_specs = {s['name']: s for s in source_meta['converted_modules']}
    _require(set(specs) == set(old_specs) and len(specs) == 252, 'Complete native projection inventory changed.')
    for name in specs:
        expected = deepcopy(old_specs[name])
        if name in NAMES:
            expected['weight_rotation']['block_size'] = record['block_size']
        _require(specs[name] == expected, 'Unexpected native deployment spec change: ' + name)
    after = decoded_records(reloaded, device)
    _require(before == after, 'Actual72 BF16 group8 decoded weights changed on fresh reload.')
    _assert_state_equal(reloaded.state_dict(), serialized, 'Reloaded group8 must preserve serialized state')
    budget = audit_checkpoint(checkpoint, max_bpw=2.5, max_lora_rank=0)
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == record['expected_user_linear_bits'] == 17065672704
             and budget['original_linear_weights'] == 6945767424 and budget['compressed_target_count'] == 252
             and budget['code_padding_bits'] == 0 and budget['components'] == record['budget_components'],
             'Native actual storage differs from the unchanged same-rate contract.')
    _require(budget['components']['decoder']['dtypes'] == ['bfloat16'], 'Unexpected FP32 deployment decoder storage.')
    for field in ('lora', 'residual_lora', 'dense_weights'):
        _require(budget['components'].get(field, {}).get('storage_bits', 0) == 0, 'Unexpected auxiliary storage.')
    del reloaded, serialized
    gc.collect(); torch.cuda.empty_cache()
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'source_checkpoint_id': record['base_checkpoint_id'],
        'replaced_projection_count': 72, 'preserved_projection_count': 180,
        'replaced_local_tensor_count': 936, 'frozen_complement_tensor_count': 2487,
        'frozen_complement_exact': True, 'serialized_tensors_compared': serialized_count,
        'reloaded_tensors_compared': reloaded_count, 'native_group8_reload_exact': True,
        'native_group8_full_weight_records': before, 'protection_tensor_records': protection,
        'rotation_sign_records': signs, 'bit_budget': budget, 'provenance': provenance,
        'elapsed_seconds': time.time() - started,
        'peak_allocated_bytes_after_save_decode': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes_after_save_decode': torch.cuda.max_memory_reserved(device)}
