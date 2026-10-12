"""Audit actual native geometry, preserved source union, and paired calibration."""
from pathlib import Path

import torch

from experiments.llama_gate_up_rate_pair.common import (TOTAL_BITS, DENOMINATOR,
    BASE_ID, read, _require)
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state


def tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return tensor_record(value)
    if isinstance(value, dict):
        return {k: tensor_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_tree(v) for v in value]
    _require(value is None or isinstance(value, (str, int, float, bool)), 'Unexpected calibration state type.')
    return value


def donor(directory, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent,
             'Actual native output does not identify its run.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Actual native recipe differs from preparation.')
    categories = record['categories']
    _require(categories in (['gate_proj', 'up_proj'], ['up_proj'])
             and record['block_size'] in (0, 32)
             and (categories == ['up_proj'] or record['block_size'] == 32)
             and (categories != ['up_proj'] or record['codebook_bits'] == {'up_proj': 32}),
             'Require the defined allocation pair or fixed-K single-up rotation pair.')
    names = {n for n in record['architecture']['projection_shapes'] if n.rsplit('.', 1)[1] in categories}
    specs = {s['name']: s for s in meta['converted_modules']}
    _require(len(names) == 32 * len(categories) and set(specs) == set(meta['compressed_targets']) == names
             and meta['completed_categories'] == meta['compression_categories'] == categories
             and meta['after_category_mode'] == 'none' and meta['lora_config'] is None
             and meta['base_model_path'] == record['architecture']['base_model_path'], 'Wrong donor identity/range.')
    for name, s in specs.items():
        k = record['codebook_bits'][name.rsplit('.', 1)[1]]
        _require([s['out_features'], s['in_features']] == [14336, 4096] and not s['transpose']
                 and s['compressed_in_features'] == 4032 and s['compressed_out_features'] == 14336
                 and s['codebook_dim'] == 32 and s['stage_codebook_dims'] == [32, 32]
                 and s['residual_stages'] == 2 and s['parallel_parts'] == 1
                 and len(s['stage_decoders']) == len(s['stage_vq_weights']) == 2,
                 'Wrong native geometry: ' + name)
        _require(all(p == {'shape': [1806336, 1, k // 8], 'dtype': 'uint8',
                    'storage_format': 'bitpack_u8', 'logical_shape': [1806336, 1, k],
                    'logical_dtype': 'bool', 'pack_bits': 8} for p in s['stage_vq_weights']), 'Wrong packed K: ' + name)
        _require(all(d['in_dim'] == k and d['out_dim'] == 32 and d['hidden_dim'] == 128
                 and d['num_res_blocks'] == 0 and d['param_dtype'] == 'bfloat16'
                 and d['norm_type'] == 'layer' and d['activation_type'] == 'swish'
                 and d['decoder_type'] == 'symmetric' for d in s['stage_decoders']), 'Wrong decoder: ' + name)
        _require(s['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and s['protected_input_qvalues'] == {'shape': [64, 14336], 'dtype': 'uint8'}
                 and s['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and s['protected_output_indices'] is None and s['protected_channel_quant_format'] == 'int8'
                 and s['weight_rotation']['type'] == 'two_sided_hadamard'
                 and s['weight_rotation']['block_size'] == record['block_size'], 'Wrong protection/rotation: ' + name)
    log = (Path(record['output_dir']) / 'train.log').read_text()
    for category in categories:
        for stage in (1, 2):
            _require(f'[{category}.L0-31/stage{stage}] step=10000/10000 ' in log, 'Missing full stage completion.')
    return meta, names


def assembly(directory, donor_dir, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    dm, names = donor(donor_dir, record); report = read(directory / 'assembly_report.json')
    source = report['sources']; budget = report['bit_budget']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id'] == BASE_ID
             and source['donor_checkpoint_id'] == dm['checkpoint_id'] and source['categories'] == record['categories']
             and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 2979,
             'Wrong source union/save/fresh evidence.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
             and budget['original_linear_weights'] == DENOMINATOR and budget['code_padding_bits'] == 0
             and budget['components'] == record['expected_budget_components']
             and budget['compressed_target_count'] == 224, 'Actual packed/decoder/protection budget differs.')
    state = torch.load(directory / meta['state_dict_file'], weights_only=True, map_location='cpu', mmap=True)
    _require(len(state) == 2979, 'Wrong actual saved state inventory.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protection = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {k: tensor_record(v) for k, v in state.items()
                  if k.startswith(n + '.') and k.endswith(('.input_signs', '.output_signs'))} for n in sorted(names)}
    _require(all(len(v) == 2 for v in signs.values()), 'Incomplete actual rotation signs.')
    runtime = load_v6_cat_runtime_state(str(donor_dir), required=True)
    activation = runtime['activation_runtime']; cache = activation['cache']
    _require(runtime['global_adaptive_plan'] is None and activation['dataset'] == cache['dataset'] == 'alpaca=1'
             and activation['nsamples'] == cache['nsamples'] == 512 and activation['seed'] == cache['seed'] == 31
             and activation['seqlen'] == cache['seqlen'] == 256 and len(cache['input_ids']) == 512
             and names <= set(activation['stats_by_linear']), 'Actual one-shot HF calibration differs.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': BASE_ID, 'bit_budget': budget, 'source_union_save_fresh_tensors': 2979,
        'replaced_projection_count': len(names), 'preserved_projection_count': 224 - len(names),
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'calibration_cache': tensor_tree(cache), 'calibration_stats': tensor_tree(activation['stats_by_linear']),
        'global_adaptive_plan': None,
        'actual_native_normalized_sha256': sha256(Path(donor_dir).parent / 'normalized_cat_runtime_args.json'),
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'cat_runtime_state_sha256': sha256(Path(donor_dir) / 'cat_runtime_state.pt')}
