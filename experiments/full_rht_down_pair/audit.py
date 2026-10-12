"""Audit native complete-down donors, source unions and full-eight results."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.full_rht_gate_pair.audit import results
from experiments.full_rht_gate_pair.common import EXPECTED_BITS, EXPECTED_IDS, read
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
    architecture = record['architecture']; model_type = record['model_type']
    _require(model_type in ('qwen3', 'llama') and record['categories'] == ['down_proj']
             and record['codebook_bits'] == {'down_proj': 48}
             and record['base_checkpoint_id'] == EXPECTED_IDS[model_type]
             and record['block_size'] in (0, 32), 'Require the defined native down rotation pair.')
    layers, in_features, blocks = (36, 12288, 1540096) if model_type == 'qwen3' else (32, 14336, 1802240)
    _require(architecture['num_hidden_layers'] == layers and architecture['projection_count'] == layers * 7,
             'Actual model projection inventory differs.')
    names = {name for name in architecture['projection_shapes'] if name.endswith('.down_proj')}
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(len(names) == layers and set(meta['compressed_targets']) == set(specs) == names
             and meta['compression_categories'] == meta['completed_categories'] == ['down_proj']
             and meta['after_category_mode'] == 'none', 'Donor must contain the complete fresh down category only.')
    for name, spec in specs.items():
        _require(architecture['projection_shapes'][name] == [4096, in_features]
                 and [spec['out_features'], spec['in_features']] == [4096, in_features]
                 and spec['compressed_in_features'] == in_features - 256
                 and spec['compressed_out_features'] == 4096 and spec['transpose']
                 and spec['codebook_dim'] == 32 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32] and spec['parallel_parts'] == 1,
                 'Down geometry changed: ' + name)
        _require(len(spec['stage_vq_weights']) == len(spec['stage_decoders']) == 2,
                 'Expected exactly two packed residual stages: ' + name)
        for packed in spec['stage_vq_weights']:
            _require(packed == {'shape': [blocks, 1, 6], 'dtype': 'uint8',
                     'storage_format': 'bitpack_u8', 'logical_shape': [blocks, 1, 48],
                     'logical_dtype': 'bool', 'pack_bits': 8}, 'Packed down code geometry changed: ' + name)
        _require(spec['protected_input_indices'] == {'shape': [256], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [256, 4096], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [256], 'dtype': 'bfloat16'}
                 and spec['protected_output_indices'] is None
                 and spec['protected_channel_quant_format'] == 'int8', 'Protection changed: ' + name)
        _require(spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == record['block_size'], 'Wrong actual rotation: ' + name)
        _require(all(stage['in_dim'] == 48 and stage['out_dim'] == 32 and stage['hidden_dim'] == 128
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
    source = report['sources']; count = 3423 if record['model_type'] == 'qwen3' else 2979
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id']
             and source['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and source['categories'] == ['down_proj'] and set(source['replaced_targets']) == names,
             'Assembled checkpoint does not come from the actual paired sources.')
    _require(report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == count,
             'Native source-union/save/fresh strict inventory is incomplete.')
    budget = report['bit_budget']; bits, weights = budget['user_linear_bits'], budget['original_linear_weights']
    width_passed = bits * 2 < weights * 5 if record['model_type'] == 'llama' else bits * 2 <= weights * 5
    _require(budget['budget_passed'] and width_passed and not budget['contract_violations']
             and not budget['unclassified_target_keys']
             and weights == record['architecture']['original_linear_weights']
             and bits == record['expected_user_linear_bits'] == EXPECTED_BITS[record['model_type']]
             and budget['compressed_target_count'] == record['architecture']['projection_count']
             and budget['components']['decoder']['dtypes'] == ['bfloat16'],
             'Actual assembled tensor storage differs from the approved budget.')
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
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'],
        'donor_checkpoint_id': donor_meta['checkpoint_id'], 'base_checkpoint_id': record['base_checkpoint_id'],
        'bit_budget': budget, 'source_union_save_fresh_tensors': len(state),
        'down_count': len(names), 'replaced_projection_count': len(names),
        'preserved_projection_count': record['architecture']['projection_count'] - len(names),
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(Path(donor).parent / 'normalized_cat_runtime_args.json')}
