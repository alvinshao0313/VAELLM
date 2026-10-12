"""Audit the fresh Qwen O donor, exact source union, and complete full8 results."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import read
from experiments.full_rht_gate_pair.audit import results as _full8_results
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
    architecture, categories = record['architecture'], record['categories']
    _require(record['model_type'] == 'qwen3' and categories == ['o_proj']
             and architecture['num_hidden_layers'] == 36 and architecture['projection_count'] == 252,
             'Expected the explicit complete Qwen O pair.')
    _require(record['arm'] in ('control', 'candidate')
             and record['codebook_bits'] == {'o_proj': 48 if record['arm'] == 'control' else 56},
             'Expected fresh O48 control or O56 candidate.')
    names = {name for name in architecture['projection_shapes'] if name.endswith('.o_proj')}
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(len(names) == 36 and set(meta['compressed_targets']) == set(specs) == names
             and meta['compression_categories'] == meta['completed_categories'] == categories
             and meta['after_category_mode'] == 'none', 'Donor must contain only the 36 fresh O projections.')
    bits = record['codebook_bits']['o_proj']
    for name, spec in specs.items():
        out_features, in_features = architecture['projection_shapes'][name]
        blocks = out_features * (in_features - 64) // 32
        _require(out_features == in_features == 4096 and blocks == 516096
                 and [spec['out_features'], spec['in_features']] == [out_features, in_features]
                 and spec['compressed_in_features'] == in_features - 64
                 and spec['compressed_out_features'] == out_features and spec['transpose']
                 and spec['codebook_dim'] == 32 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32] and spec['parallel_parts'] == 1,
                 'O geometry changed: ' + name)
        _require(len(spec['stage_vq_weights']) == len(spec['stage_decoders']) == 2,
                 'Expected exactly two packed residual stages: ' + name)
        for packed in spec['stage_vq_weights']:
            _require(packed == {'shape': [blocks, 1, bits // 8], 'dtype': 'uint8',
                     'storage_format': 'bitpack_u8', 'logical_shape': [blocks, 1, bits],
                     'logical_dtype': 'bool', 'pack_bits': 8}, 'Packed O code geometry changed: ' + name)
        _require(spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [64, out_features], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and spec['protected_output_indices'] is None
                 and spec['protected_channel_quant_format'] == 'int8', 'Protection changed: ' + name)
        _require(spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == record['block_size'] == 32,
                 'Wrong native rotation: ' + name)
        _require(all(stage['in_dim'] == bits and stage['out_dim'] == 32 and stage['hidden_dim'] == 128
                     and stage['num_res_blocks'] == 0 and stage['norm_type'] == 'layer'
                     and stage['activation_type'] == 'swish' and stage['decoder_type'] == 'symmetric'
                     and stage['param_dtype'] == 'bfloat16' for stage in spec['stage_decoders']),
                 'Decoder architecture differs: ' + name)
    steps = record['steps_per_stage']
    _require(steps == 10000, 'This pair requires the complete 10000-step schedule per stage.')
    log = (Path(record['output_dir']) / 'train.log').read_text()
    for stage in (1, 2):
        _require(f'[o_proj.L0-35/stage{stage}] step={steps}/{steps} ' in log,
                 'Native training did not finish the declared O stage.')
    return meta, names


def check_assembly(directory, donor, record):
    directory = Path(directory)
    meta = load_v6_meta(str(directory)); report = read(directory / 'assembly_report.json')
    donor_meta, names = check_donor(donor, record)
    source = report['sources']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id']
             and source['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and source['categories'] == record['categories'] and set(source['replaced_targets']) == names,
             'Assembled checkpoint does not come from the actual paired sources.')
    _require(report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 3423,
             'Native source-union/save/fresh strict inventory is incomplete.')
    budget = report['bit_budget']
    expected_bits = 17065672704 if record['arm'] == 'control' else 17364123648
    _require(budget['budget_passed'] and not budget['contract_violations']
             and not budget['unclassified_target_keys']
             and budget['original_linear_weights'] == record['architecture']['original_linear_weights']
             and budget['user_linear_bits'] == record['expected_user_linear_bits'] == expected_bits
             and budget['user_linear_bits'] * 2 <= budget['original_linear_weights'] * 5
             and budget['compressed_target_count'] == 252,
             'Actual assembled storage differs from the approved at-most-2.5 budget.')
    _require(budget['components']['decoder']['dtypes'] == ['bfloat16'],
             'All deployed decoder parameters must retain BF16 storage.')
    for key in ('lora', 'residual_lora', 'dense_weights'):
        _require(budget['components'].get(key, {}).get('storage_bits', 0) == 0, 'Unexpected payload: ' + key)
    state = torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == 3423, 'Fresh persistent inventory changed after assembly.')
    protection = {name: {field: tensor_record(state[name + '.' + field]) for field in
        ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')}
        for name in sorted(names)}
    signs = {name: {key: tensor_record(value) for key, value in state.items()
                    if key.startswith(name + '.') and key.endswith(('.input_signs', '.output_signs'))}
             for name in sorted(names)}
    _require(all(len(value) == 2 for value in signs.values()), 'Expected two persistent sign vectors per projection.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'],
        'donor_checkpoint_id': donor_meta['checkpoint_id'], 'base_checkpoint_id': record['base_checkpoint_id'],
        'bit_budget': budget, 'source_union_save_fresh_tensors': len(state),
        'replaced_projection_count': len(names), 'preserved_projection_count': 216,
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(Path(donor).parent / 'normalized_cat_runtime_args.json')}


def results(directory, record):
    # Reuse complete 64-leaf/24742-sample evaluation and exact source protocol checks.
    result = _full8_results(directory, record)
    result.pop('rotation_pair_comparison')
    result.update(o_rate_pair_comparison='pending_other_arm',
                  comparison_boundary=record['comparison_boundary'])
    return result
