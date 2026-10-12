"""Audit independent K40 donors and their native two-stage source union."""
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import read
from experiments.longbench_v1.common import sha256
from experiments.qwen3_w25.assemble_categories import _require
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import load_v6_meta


TOTAL_BITS = 17102536704
INTERMEDIATE_BITS = 16176644096


def persistent_records(state, names):
    protection = {name: {key: tensor_record(state[name + '.' + key]) for key in
        ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')}
        for name in sorted(names)}
    signs = {name: {key: tensor_record(value) for key, value in state.items()
                    if key.startswith(name + '.') and key.endswith(('.input_signs', '.output_signs'))}
             for name in sorted(names)}
    _require(all(len(value) == 2 for value in signs.values()), 'Missing native rotation signs.')
    return protection, signs


def check_donor(directory, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    category = record['category']; geometry = record['architecture']['projection_shapes']
    _require(record['model_type'] == 'llama' and category in ('gate_proj', 'down_proj')
             and record['steps_per_stage'] == 10000 and record['block_size'] == 32
             and record['category_index'] == 0 and record['shuffle_seeds'] == [31, 32],
             'Require an independent complete Llama K40 category.')
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent,
             'Native runtime output identity differs.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Native recipe differs from CPU preparation.')
    names = {name for name in geometry if name.endswith('.' + category)}
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(len(names) == 32 and set(specs) == set(meta['compressed_targets']) == names
             and meta['compression_categories'] == meta['completed_categories'] == [category]
             and meta['after_category_mode'] == 'none'
             and meta['base_model_path'] == record['architecture']['base_model_path'],
             'Donor inventory or original model differs.')
    protection_count = 64 if category == 'gate_proj' else 256
    for name, spec in specs.items():
        out_dim, in_dim = geometry[name]; groups = out_dim * (in_dim - protection_count) // 32
        _require(groups == (1806336 if category == 'gate_proj' else 1802240)
                 and [spec['out_features'], spec['in_features']] == [out_dim, in_dim]
                 and spec['compressed_out_features'] == out_dim
                 and spec['compressed_in_features'] == in_dim - protection_count
                 and spec['transpose'] == (category == 'down_proj')
                 and spec['codebook_dim'] == 32 and spec['stage_codebook_dims'] == [32, 32]
                 and spec['residual_stages'] == 2 and spec['parallel_parts'] == 1,
                 'Unexpected protected matrix geometry: ' + name)
        packed = {'shape': [groups, 1, 5], 'dtype': 'uint8', 'storage_format': 'bitpack_u8',
                  'logical_shape': [groups, 1, 40], 'logical_dtype': 'bool', 'pack_bits': 8}
        _require(spec['stage_vq_weights'] == [packed, packed], 'Expected two K40 packed banks.')
        _require(spec['protected_input_indices'] == {'shape': [protection_count], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [protection_count, out_dim], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [protection_count], 'dtype': 'bfloat16'}
                 and spec['protected_output_indices'] is None
                 and spec['protected_channel_quant_format'] == 'int8'
                 and spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == 32, 'Protection/rotation geometry changed.')
        _require(len(spec['stage_decoders']) == 2 and all(
            stage['in_dim'] == 40 and stage['out_dim'] == 32 and stage['hidden_dim'] == 128
            and stage['num_res_blocks'] == 0 and stage['norm_type'] == 'layer'
            and stage['activation_type'] == 'swish' and stage['decoder_type'] == 'symmetric'
            and stage['param_dtype'] == 'bfloat16' for stage in spec['stage_decoders']),
            'Decoder differs from the BF16 H128/R0 contract.')
    log = (Path(record['output_dir']) / 'train.log').read_text()
    for stage in (1, 2):
        _require(f'[{category}.L0-31/stage{stage}] step=10000/10000 ' in log,
                 'The actual native stage did not complete 10000 updates.')
    return meta, names


def donor_audit(directory, record):
    directory = Path(directory); meta, names = check_donor(directory, record)
    state = torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    protection, signs = persistent_records(state, names)
    historical = read(record['historical_category_audit'])
    _require(protection == historical['protection_tensor_records']
             and signs == historical['rotation_sign_records'],
             'Fresh K40 protection/signs differ from the matched historical category.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'category': record['category'],
        'projection_count': len(names), 'category_index': 0, 'shuffle_seeds': [31, 32],
        'actual_steps_per_stage': 10000, 'protection_tensor_records': protection,
        'rotation_sign_records': signs, 'historical_protection_sign_exact': True,
        'native_normalized_sha256': sha256(directory.parent / 'normalized_cat_runtime_args.json'),
        'checkpoint_meta_sha256': sha256(directory / 'checkpoint_meta.json'),
        'quality_evaluated': False}


def check_assembly(output, record):
    output = Path(output); final = output / 'final_model'; intermediate = output / 'down_base'
    previous_id = record['base_checkpoint_id']; audits = {}
    for category, directory, expected_bits in (
            ('down_proj', intermediate, INTERMEDIATE_BITS), ('gate_proj', final, TOTAL_BITS)):
        assembly = read(directory / 'assembly_report.json'); meta = load_v6_meta(str(directory))
        source = assembly['sources']; donor = record['donors'][category]
        names = {name for name in record['architecture']['projection_shapes'] if name.endswith('.' + category)}
        _require(assembly['status'] == 'verified' and assembly['checkpoint_id'] == meta['checkpoint_id']
                 and source['base_checkpoint_id'] == previous_id
                 and source['donor_checkpoint_id'] == donor['checkpoint_id']
                 and source['categories'] == [category] and set(source['replaced_targets']) == names
                 and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared'] == 2979,
                 'Native category union/save/fresh reload differs.')
        budget = assembly['bit_budget']
        _require(budget['budget_passed'] and not budget['contract_violations']
                 and not budget['unclassified_target_keys'] and budget['user_linear_bits'] == expected_bits
                 and budget['original_linear_weights'] == 6979321856
                 and 2 * expected_bits < 5 * 6979321856 and budget['compressed_target_count'] == 224
                 and budget['code_padding_bits'] == 0
                 and budget['components']['decoder']['dtypes'] == ['bfloat16'],
                 'Actual native storage differs from the declared integer budget.')
        _require(all(budget['components'].get(key, {}).get('storage_bits', 0) == 0
                     for key in ('lora', 'residual_lora', 'dense_weights')), 'Unexpected extra branch/storage.')
        audits[category] = {'checkpoint_id': meta['checkpoint_id'], 'bit_budget': budget,
                           'assembly_report_sha256': sha256(directory / 'assembly_report.json')}
        previous_id = meta['checkpoint_id']
    state = torch.load(final / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    names = {name for name in record['architecture']['projection_shapes']
             if name.endswith(('.gate_proj', '.down_proj'))}
    protection, signs = persistent_records(state, names)
    for category, donor in record['donors'].items():
        proof = read(donor['audit_path']); selected = [name for name in names if name.endswith('.' + category)]
        _require({name: protection[name] for name in selected} == proof['protection_tensor_records']
                 and {name: signs[name] for name in selected} == proof['rotation_sign_records'],
                 'Composition changed audited donor protection/signs.')
    source = read(record['source_budget_path'])['bit_budget']
    for key, value in source['components'].items():
        expected = value['storage_bits'] + (2097152 if key == 'code' else 0)
        _require(budget['components'][key]['storage_bits'] == expected,
                 'Actual component budget differs: ' + key)
    _require(len(state) == 2979 and len(names) == 64, 'Final complete Llama inventory differs.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'],
        'base_checkpoint_id': record['base_checkpoint_id'], 'replaced_projection_count': 64,
        'preserved_projection_count': 160, 'source_union_save_fresh_tensors': 2979,
        'two_native_unions': audits, 'bit_budget': budget,
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'historical_protection_sign_exact': True, 'rank_zero': True}
