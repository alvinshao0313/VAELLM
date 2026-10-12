"""Audit the actual 72-projection Q/K donor and native f927 source union."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, read
from experiments.full_rht_gate_pair.audit import results as _full8_results
from experiments.longbench_v1.common import sha256
from experiments.qwen3_w25.assemble_categories import _require
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import load_v6_meta


def check_donor(directory, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent,
             'Native normalized output does not name its actual run.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Native parameters differ from CPU preparation.')
    architecture, categories = record['architecture'], record['categories']
    _require(record['model_type'] == 'qwen3' and categories == ['q_proj', 'k_proj']
             and architecture['num_hidden_layers'] == 36 and architecture['projection_count'] == 252
             and record['arm'] in ('block32', 'full')
             and record['block_size'] == (32 if record['arm'] == 'block32' else 0)
             and record['codebook_bits'] == {'q_proj': 32, 'k_proj': 32}, 'Wrong Q/K arm contract.')
    names = {name for name in architecture['projection_shapes'] if name.rsplit('.', 1)[-1] in categories}
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(len(names) == 72 and set(meta['compressed_targets']) == set(specs) == names
             and meta['compression_categories'] == meta['completed_categories'] == categories
             and meta['after_category_mode'] == 'none', 'Donor must contain exactly the complete fresh Q/K categories.')
    for name, spec in specs.items():
        q = name.endswith('.q_proj'); out = 4096 if q else 1024; groups = out * 4032 // 32
        _require(architecture['projection_shapes'][name] == [out, 4096]
                 and spec['out_features'] == out and spec['in_features'] == 4096
                 and spec['compressed_out_features'] == out and spec['compressed_in_features'] == 4032
                 and spec['transpose'] is q and spec['codebook_dim'] == 32
                 and spec['residual_stages'] == 2 and spec['stage_codebook_dims'] == [32, 32]
                 and spec['parallel_parts'] == 1, 'Q/K geometry changed: ' + name)
        _require(len(spec['stage_vq_weights']) == len(spec['stage_decoders']) == 2, 'Wrong stage count.')
        for packed in spec['stage_vq_weights']:
            _require(packed == {'shape': [groups, 1, 4], 'dtype': 'uint8', 'storage_format': 'bitpack_u8',
                     'logical_shape': [groups, 1, 32], 'logical_dtype': 'bool', 'pack_bits': 8},
                     'Packed Q/K code geometry changed: ' + name)
        _require(spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [64, out], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and spec['protected_output_indices'] is None
                 and spec['protected_channel_quant_format'] == 'int8', 'Protection changed: ' + name)
        _require(spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == record['block_size'], 'Wrong native rotation.')
        _require(all(stage['in_dim'] == stage['out_dim'] == 32 and stage['hidden_dim'] == 128
                     and stage['num_res_blocks'] == 0 and stage['norm_type'] == 'layer'
                     and stage['activation_type'] == 'swish' and stage['decoder_type'] == 'symmetric'
                     and stage['param_dtype'] == 'bfloat16' for stage in spec['stage_decoders']),
                 'Decoder architecture differs: ' + name)
    _require(record['steps_per_stage'] == 10000, 'Require the complete 10000-step schedule per stage.')
    log = (Path(record['output_dir']) / 'train.log').read_text()
    for category in categories:
        for stage in (1, 2):
            _require(f'[{category}.L0-35/stage{stage}] step=10000/10000 ' in log,
                     'Native training did not finish the declared Q/K stage.')
    return meta, names


def check_assembly(directory, donor, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    report = read(directory / 'assembly_report.json'); donor_meta, names = check_donor(donor, record)
    source = report['sources']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id'] == EXPECTED_IDS['qwen3']
             and source['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and source['categories'] == record['categories'] and set(source['replaced_targets']) == names,
             'Native source union differs from the actual paired sources.')
    _require(report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 3423,
             'Native source-union/save/fresh inventory is incomplete.')
    budget = report['bit_budget']
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['original_linear_weights'] == 6945767424 and budget['compressed_target_count'] == 252
             and budget['user_linear_bits'] == record['expected_user_linear_bits'] == 17065672704
             and budget['packed_linear_bits'] == 17065672704 and budget['code_padding_bits'] == 0
             and budget['components'] == record['budget_estimate']['components'],
             'Actual assembled storage differs from the same-rate f927 budget.')
    _require(budget['components']['decoder']['dtypes'] == ['bfloat16'], 'Decoder storage must remain BF16.')
    for key in ('lora', 'residual_lora', 'dense_weights'):
        _require(budget['components'].get(key, {}).get('storage_bits', 0) == 0, 'Unexpected payload: ' + key)
    state = torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    local_keys = [key for key in state if any(key.startswith(name + '.') for name in names)]
    _require(len(state) == 3423 and len(local_keys) == 936, 'Actual Q/K persistent inventory changed.')
    protection = {name: {field: tensor_record(state[name + '.' + field]) for field in
        ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')} for name in sorted(names)}
    signs = {name: {key: tensor_record(value) for key, value in state.items()
                    if key.startswith(name + '.') and key.endswith(('.input_signs', '.output_signs'))}
             for name in sorted(names)}
    _require(all(len(value) == 2 for value in signs.values()), 'Expected two sign vectors per projection.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'],
        'donor_checkpoint_id': donor_meta['checkpoint_id'], 'base_checkpoint_id': record['base_checkpoint_id'],
        'bit_budget': budget, 'source_union_save_fresh_tensors': len(state),
        'replaced_projection_count': 72, 'preserved_projection_count': 180,
        'replaced_local_tensor_count': 936, 'frozen_complement_tensor_count': 2487,
        'frozen_evidence': 'Native assembler forms the expected source union and checks all3423 in-memory, serialized and fresh-reloaded tensors exactly.',
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(Path(donor).parent / 'normalized_cat_runtime_args.json')}


def results(directory, record):
    result = _full8_results(directory, record)
    result.pop('rotation_pair_comparison')
    result.update(qk_rotation_pair_comparison='pending_other_arm', comparison_boundary=record['comparison_boundary'])
    return result
