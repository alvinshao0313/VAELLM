"""Persist only attention tensors; rebuild the actual native source union strictly."""
from copy import deepcopy
import gc
from pathlib import Path
import time
import uuid

import torch

from experiments.qwen_attention_decoded_target_pair.common import BITS, NAMES, PROTECTION, read, _require
from experiments.longbench_attribution.restore import tensor_digest
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.rotation_factor_probe.capture import tensor_record
from litebsq.vae_linear_prewarm import decode_named_vae_linear_weights
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import (
    _collect_vae_linear_specs, _rebuild_converted_modules, refresh_vae_linear_runtime_after_state_load,
)
from train_utils.distill_precision import install_precision_runtime, restore_parameter_dtypes
from train_utils.v6_model_loader import load_v6_model_checkpoint


def decoded_records(model, record, device):
    layers = [(name, model.get_submodule(name)) for name in NAMES]
    with torch.no_grad():
        values = decode_named_vae_linear_weights(layers, dtype=torch.bfloat16,
            group_size=8, compute_device=device, respect_cache_policy=False)
    records = {}
    for item in values:
        _require(item.decoded_weight.dtype == torch.bfloat16
                 and list(item.decoded_weight.shape) == record['architecture']['projection_shapes'][item.name]
                 and bool(torch.isfinite(item.decoded_weight).all()), 'Invalid native group8 attention weight.')
        records[item.name] = tensor_record(item.decoded_weight)
    _require(set(records) == set(NAMES), 'Incomplete native144 group8 decode.')
    del values, item
    for _, layer in layers:
        layer.clear_decoded_weight_cache(); layer.to('cpu')
    del layers, layer
    gc.collect(); torch.cuda.empty_cache()
    return records


def save_and_reconstruct(model, base_state, record, training, device):
    started = time.time(); output = Path(record['output_dir'])
    archive = output / 'attention_state.pt'; meta_path = output / 'native_union_meta.json'
    _require(not archive.exists() and not meta_path.exists(), 'Never overwrite an endpoint archive.')
    model.to('cpu').eval().requires_grad_(False)
    for name in NAMES:
        model.get_submodule(name).clear_decoded_weight_cache()
    state = model.state_dict(); prefixes = tuple(name + '.' for name in NAMES)
    selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
    outside = {key: value for key, value in state.items() if not key.startswith(prefixes)}
    source_outside = {key: value for key, value in base_state.items() if not key.startswith(prefixes)}
    _require(set(state) == set(base_state) and len(state) == 3423 and len(selected) == 1872
             and len(outside) == 1551, 'Wrong actual attention/complement inventory.')
    _assert_state_equal(outside, source_outside, 'Frozen108 MLP plus147 backbone tensors')
    source_meta = read(Path(record['base_dir']) / 'checkpoint_meta.json')
    actual_specs = _collect_vae_linear_specs(model)
    old_specs = {spec['name']: spec for spec in source_meta['converted_modules']}
    specs = {spec['name']: spec for spec in actual_specs}
    _require(set(specs) == set(old_specs) and len(specs) == 252, 'Complete projection inventory changed.')
    protection, signs = {}, {}
    for name, spec in specs.items():
        expected = deepcopy(old_specs[name])
        if name in NAMES:
            bits = BITS[name.rsplit('.', 1)[1]]
            for packed_field, decoder_field in (('stage_vq_weights', 'stage_decoders'), ('vq_weights', 'decoders')):
                for packed, decoder in zip(expected[packed_field], expected[decoder_field], strict=True):
                    packed['logical_shape'][-1] = bits; packed['shape'][-1] = bits // 8
                    decoder['in_dim'] = bits
            protection[name] = {field: tensor_record(state[name + '.' + field]) for field in PROTECTION}
            _require(protection[name] == {field: training['materialization']['targets'][name][field]
                                         for field in PROTECTION}, 'Native protection differs from actual target: ' + name)
            signs[name] = {field: tensor_record(state[name + '.weight_rotation.' + field])
                           for field in ('input_signs', 'output_signs')}
            _require(signs[name] == record['source_plan'][name]['signs'], 'Actual fixed f927 signs changed: ' + name)
        _require(spec == expected, 'Unexpected actual native geometry: ' + name)
    _require(all(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all())
                 for value in selected.values() if value.is_floating_point()), 'Invalid BF16 endpoint parameters.')
    identity = str(uuid.uuid4())
    meta = dict(source_meta, checkpoint_id=identity, converted_modules=actual_specs,
                donor_checkpoint_id=identity, immutable_resume_contract=None)
    budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=0)
    _require(budget['user_linear_bits'] == 17813569536 and budget['original_linear_weights'] == 6945767424
             and budget['contract_violations'] == ['User Linear bpw 2.564665421 exceeds 2.5.']
             and not budget['budget_passed'] and not budget['unclassified_target_keys']
             and budget['code_padding_bits'] == 0, 'Unexpected native intermediate budget or contract.')
    before = decoded_records(model, record, device)
    _assert_state_equal(model.state_dict(), state, 'Native group8 decode preserved trained state')
    torch.save({'schema': 'isolated_attention_tensor_endpoint_v1', 'endpoint_id': identity,
                'source_checkpoint_id': record['base_checkpoint_id'], 'target_kind': record['target_kind'],
                'state': selected}, archive)
    # This is internal reconstruction metadata, deliberately not checkpoint_meta.json:
    # the native INT8 union exceeds2.5 and is never presented as a deployable product.
    write_json(meta_path, meta)
    saved = torch.load(archive, map_location='cpu', weights_only=True, mmap=True)
    _require(saved['endpoint_id'] == identity and saved['source_checkpoint_id'] == record['base_checkpoint_id']
             and saved['target_kind'] == record['target_kind'], 'Endpoint source identity changed.')
    _assert_state_equal(saved['state'], selected, 'Saved1872 attention tensors')
    union = {**source_outside, **saved['state']}
    fresh, fresh_meta, loaded = load_v6_model_checkpoint(record['base_dir'], map_location='cpu', strict=True,
                                                        expected_kind='final_model')
    _require(fresh_meta['checkpoint_id'] == record['base_checkpoint_id']
             and not loaded.missing_keys and not loaded.unexpected_keys, 'Fresh original source reconstruction failed.')
    _rebuild_converted_modules(fresh, [specs[name] for name in NAMES])
    restore_parameter_dtypes(fresh, union)
    incompatible = fresh.load_state_dict(union, strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, 'Native union strict load failed.')
    install_precision_runtime(fresh); refresh_vae_linear_runtime_after_state_load(fresh)
    fresh.eval().requires_grad_(False)
    fresh_count = _assert_state_equal(fresh.state_dict(), union, 'Fresh native3423 source union')
    _require(_collect_vae_linear_specs(fresh) == actual_specs, 'Native reconstruction changed actual specs.')
    after = decoded_records(fresh, record, device)
    _require(before == after, 'Actual144 BF16 group8 weights differ after endpoint/native reconstruction.')
    _assert_state_equal(fresh.state_dict(), union, 'Fresh group8 decode preserved union')
    _require(audit_state_dict(meta, fresh.state_dict(), max_bpw=2.5, max_lora_rank=0) == budget,
             'Actual fresh native union storage changed.')
    report = {'status': 'verified_isolated_intermediate', 'endpoint_id': identity,
        'source_checkpoint_id': record['base_checkpoint_id'], 'endpoint_sha256': sha256(archive),
        'native_union_meta_sha256': sha256(meta_path), 'serialized_attention_tensors': len(selected),
        'fresh_union_tensors_compared': fresh_count, 'frozen_complement_tensors': len(outside),
        'replaced_attention': 144, 'preserved_mlp': 108, 'preserved_backbone': 147,
        'source_union_tensor_sha256': {key: tensor_digest(value) for key, value in union.items()},
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'native_group8_before_after_exact': True, 'native_group8_weight_records': before,
        'actual_native_union_budget': budget, 'native_checkpoint_exported': False,
        'boundary': 'Only attention endpoint tensors plus metadata are saved. Strict native reconstruction uses '
                    'the unchanged f927 complement. The over-budget INT8 union is an isolated intermediate; '
                    'the native2.5 assembler guard was neither called nor weakened.',
        'elapsed_seconds': time.time() - started}
    del saved, union, state, selected, outside, source_outside
    gc.collect(); torch.cuda.empty_cache()
    return fresh, meta, report
