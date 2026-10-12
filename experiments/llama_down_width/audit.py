"""Audit the fresh Llama D16/K24 H256 down donor and paired H128 evidence."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.full_rht_gate_pair.audit import results as _full8_results
from experiments.full_rht_gate_pair.common import EXPECTED_IDS, read
from experiments.longbench_v1.common import sha256
from experiments.qwen3_w25.assemble_categories import _require
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import load_v6_meta


def check_donor(directory, record):
    directory = Path(directory)
    meta = load_v6_meta(str(directory))
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent,
             'Native normalized output does not name its actual run.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Native CAT parameters differ from CPU preparation.')
    architecture = record['architecture']; layers = architecture['num_hidden_layers']
    _require(record['model_type'] == architecture['model_type'] == 'llama'
             and record['categories'] == ['down_proj'] and record['codebook_bits'] == {'down_proj': 24}
             and record['dimension'] == 16 and record['block_size'] == 32 and record['hidden_dim'] == 256
             and record['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and layers == 32 and architecture['projection_count'] == layers * 7,
             'Require the defined original-source Llama down width pair.')
    names = {name for name in architecture['projection_shapes'] if name.endswith('.down_proj')}
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(len(names) == layers and set(meta['compressed_targets']) == set(specs) == names
             and meta['compression_categories'] == meta['completed_categories'] == ['down_proj']
             and meta['after_category_mode'] == 'none', 'Donor must contain the complete fresh down category only.')
    for name, spec in specs.items():
        out_features, in_features = architecture['projection_shapes'][name]
        blocks = out_features * (in_features - 256) // 16
        _require([out_features, in_features] == [4096, 14336]
                 and [spec['out_features'], spec['in_features']] == [out_features, in_features]
                 and spec['compressed_in_features'] == in_features - 256
                 and spec['compressed_out_features'] == out_features and spec['transpose']
                 and spec['codebook_dim'] == 16 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [16, 16] and spec['parallel_parts'] == 1,
                 'Down geometry changed: ' + name)
        _require(len(spec['stage_vq_weights']) == len(spec['stage_decoders']) == 2,
                 'Expected exactly two packed residual stages: ' + name)
        for packed in spec['stage_vq_weights']:
            _require(packed == {'shape': [blocks, 1, 3], 'dtype': 'uint8',
                     'storage_format': 'bitpack_u8', 'logical_shape': [blocks, 1, 24],
                     'logical_dtype': 'bool', 'pack_bits': 8}, 'Packed down code geometry changed: ' + name)
        _require(spec['protected_input_indices'] == {'shape': [256], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [256, out_features], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [256], 'dtype': 'bfloat16'}
                 and spec['protected_output_indices'] is None
                 and spec['protected_channel_quant_format'] == 'int8', 'Protection changed: ' + name)
        _require(spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == record['block_size'], 'Wrong actual rotation: ' + name)
        _require(all(stage['in_dim'] == 24 and stage['out_dim'] == 16 and stage['hidden_dim'] == 256
                     and stage['num_res_blocks'] == 0 and stage['norm_type'] == 'layer'
                     and stage['activation_type'] == 'swish' and stage['decoder_type'] == 'symmetric'
                     and stage['param_dtype'] == 'bfloat16' for stage in spec['stage_decoders']),
                 'Decoder architecture differs: ' + name)
    steps = record['steps_per_stage']; log = (Path(record['output_dir']) / 'train.log').read_text()
    _require(steps == 10000, 'Require 10000 native updates per residual stage.')
    for stage in (1, 2):
        _require(f'[down_proj.L0-{layers-1}/stage{stage}] step={steps}/{steps} ' in log,
                 'Native down training did not finish both declared stages.')
    return meta, names


def check_assembly(directory, donor, record):
    directory = Path(directory)
    meta = load_v6_meta(str(directory)); report = read(directory / 'assembly_report.json')
    donor_meta, names = check_donor(donor, record)
    historical_path = Path(record['historical_control_audit_path'])
    historical = read(historical_path)
    _require(historical['status'] == 'verified'
             and historical['checkpoint_id'] == record['historical_control_checkpoint_id']
             == '8a4633a7-93bd-429a-8a0e-a519a2ae30ae'
             and historical['base_checkpoint_id'] == record['base_checkpoint_id']
             and historical['down_count'] == len(names), 'Historical H128/D16 control audit changed.')
    count = historical['source_union_save_fresh_tensors']
    source = report['sources']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id']
             and source['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and source['categories'] == ['down_proj'] and set(source['replaced_targets']) == names,
             'Assembled checkpoint does not come from the actual paired sources.')
    _require(report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == count,
             'Native source-union/save/fresh strict inventory is incomplete.')
    budget = report['bit_budget']; bits, weights = budget['user_linear_bits'], budget['original_linear_weights']
    _require(budget['budget_passed'] and bits * 2 < weights * 5 and not budget['contract_violations']
             and not budget['unclassified_target_keys']
             and weights == record['architecture']['original_linear_weights'] == 6979321856
             and bits == record['expected_user_linear_bits'] == 17100816384
             and budget['compressed_target_count'] == record['architecture']['projection_count']
             and budget['components']['decoder']['dtypes'] == ['bfloat16'] and budget['code_padding_bits'] == 0,
             'Actual assembled tensor storage differs from the approved integer budget.')
    old_components = historical['bit_budget']['components']
    _require(set(budget['components']) == set(old_components), 'Unexpected persistent component inventory.')
    for key, value in budget['components'].items():
        delta = 5636096 if key == 'decoder' else 0
        _require(value['storage_bits'] == old_components[key]['storage_bits'] + delta,
                 'Unexpected component storage change: ' + key)
    for key in ('lora', 'residual_lora', 'dense_weights'):
        _require(budget['components'].get(key, {}).get('storage_bits', 0) == 0, 'Unexpected payload: ' + key)
    state = torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == count, 'Fresh persistent inventory changed after assembly.')
    protection = {name: {field: tensor_record(state[name + '.' + field]) for field in
        ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')}
        for name in sorted(names)}
    signs = {name: {key: tensor_record(value) for key, value in state.items()
                    if key.startswith(name + '.') and key.endswith(('.input_signs', '.output_signs'))}
             for name in sorted(names)}
    _require(all(len(value) == 2 for value in signs.values()), 'Expected two persistent sign vectors per down.')
    _require(protection == historical['protection_tensor_records']
             and signs == historical['rotation_sign_records'],
             'Fresh protection/sign tensors differ from the recorded H128/D16 control.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'],
        'donor_checkpoint_id': donor_meta['checkpoint_id'], 'base_checkpoint_id': record['base_checkpoint_id'],
        'bit_budget': budget, 'source_union_save_fresh_tensors': len(state),
        'down_count': len(names), 'replaced_projection_count': len(names),
        'preserved_projection_count': record['architecture']['projection_count'] - len(names),
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'historical_control_checkpoint_id': historical['checkpoint_id'],
        'historical_control_audit_sha256': sha256(historical_path),
        'protection_and_signs_exact_historical_control': True,
        'historical_control_weights_loaded': False,
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(Path(donor).parent / 'normalized_cat_runtime_args.json')}


def results(directory, record):
    result = _full8_results(directory, record)
    result.pop('rotation_pair_comparison')
    result.update(width_comparison='pending_historical_control_comparison',
                  comparison_boundary=record['comparison_boundary'])
    return result
