"""Audit the single D16/K16 up donor and its e767 source union."""
from pathlib import Path

import torch

from experiments.llama_gate_up_rate_pair.common import BASE_ID, DENOMINATOR, read, _require
from experiments.llama_gate_up_rate_pair.audit import tensor_tree
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import load_v6_meta, load_v6_cat_runtime_state

CONTROL_ID = 'd086ab8d-7d8c-4f07-9231-e7f00e4cdbf5'
TOTAL_BITS = 17090969600


def donor(directory, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent, 'Wrong actual native output.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Actual recipe differs from CPU preparation.')
    names = {n for n in record['architecture']['projection_shapes'] if n.endswith('.up_proj')}
    specs = {s['name']: s for s in meta['converted_modules']}
    _require(len(names) == 32 and set(meta['compressed_targets']) == set(specs) == names
             and meta['compression_categories'] == meta['completed_categories'] == ['up_proj']
             and meta['after_category_mode'] == 'none' and meta['lora_config'] is None
             and meta['base_model_path'] == record['architecture']['base_model_path'], 'Wrong native donor range/source.')
    for name, s in specs.items():
        _require([s['out_features'], s['in_features']] == [14336, 4096] and not s['transpose']
                 and [s['compressed_out_features'], s['compressed_in_features']] == [14336, 4032]
                 and s['codebook_dim'] == 16 and s['stage_codebook_dims'] == [16, 16]
                 and s['residual_stages'] == 2 and s['parallel_parts'] == 1
                 and len(s['stage_decoders']) == len(s['stage_vq_weights']) == 2, 'Wrong D16 geometry: ' + name)
        _require(all(p == {'shape': [3612672, 1, 2], 'dtype': 'uint8', 'storage_format': 'bitpack_u8',
                    'logical_shape': [3612672, 1, 16], 'logical_dtype': 'bool', 'pack_bits': 8}
                    for p in s['stage_vq_weights']), 'Wrong native packed coverage: ' + name)
        _require(all(d['in_dim'] == d['out_dim'] == 16 and d['hidden_dim'] == 128
                 and d['num_res_blocks'] == 0 and d['param_dtype'] == 'bfloat16' and d['norm_type'] == 'layer'
                 and d['activation_type'] == 'swish' and d['decoder_type'] == 'symmetric'
                 for d in s['stage_decoders']), 'Wrong decoder: ' + name)
        _require(s['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and s['protected_input_qvalues'] == {'shape': [64, 14336], 'dtype': 'uint8'}
                 and s['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and s['protected_output_indices'] is None and s['protected_channel_quant_format'] == 'int8'
                 and s['weight_rotation']['type'] == 'two_sided_hadamard'
                 and s['weight_rotation']['block_size'] == 32, 'Wrong protection/rotation: ' + name)
    log = (Path(record['output_dir']) / 'train.log').read_text()
    for stage in (1, 2):
        _require(f'[up_proj.L0-31/stage{stage}] step=10000/10000 ' in log, 'Missing complete native stage.')
    return meta, names


def assembly(directory, native, record):
    directory = Path(directory); meta = load_v6_meta(str(directory)); dm, names = donor(native, record)
    report = read(directory / 'assembly_report.json'); source = report['sources']; budget = report['bit_budget']
    control = read(Path(record['historical_control_dir']) / 'audit.json')
    _require(control['status'] == 'verified' and control['checkpoint_id'] == CONTROL_ID
             and control['base_checkpoint_id'] == BASE_ID, 'Wrong completed D32 control.')
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id'] == BASE_ID
             and source['donor_checkpoint_id'] == dm['checkpoint_id'] and source['categories'] == ['up_proj']
             and set(source['replaced_targets']) == names
             and report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 2979,
             'Wrong complete e767 source union/save/fresh evidence.')
    _require(budget['budget_passed'] and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
             and budget['original_linear_weights'] == DENOMINATOR and budget['code_padding_bits'] == 0
             and budget['components'] == record['expected_budget_components']
             and budget['compressed_target_count'] == 224, 'Actual six-component native storage differs.')
    state = torch.load(directory / meta['state_dict_file'], weights_only=True, map_location='cpu', mmap=True)
    _require(len(state) == 2979, 'Wrong saved persistent inventory.')
    fields = ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')
    protected = {n: {f: tensor_record(state[n + '.' + f]) for f in fields} for n in sorted(names)}
    signs = {n: {k: tensor_record(v) for k, v in state.items()
                  if k.startswith(n + '.') and k.endswith(('.input_signs', '.output_signs'))} for n in sorted(names)}
    runtime = load_v6_cat_runtime_state(str(native), required=True); activation = runtime['activation_runtime']
    cache = activation['cache']; actual_cache = tensor_tree(cache)
    _require(runtime['global_adaptive_plan'] is None and activation['dataset'] == cache['dataset'] == 'alpaca=1'
             and activation['nsamples'] == cache['nsamples'] == len(cache['input_ids']) == 512
             and activation['seqlen'] == cache['seqlen'] == 256 and activation['seed'] == cache['seed'] == 31
             and names <= set(activation['stats_by_linear']), 'Wrong actual HF calibration.')
    _require(protected == control['protection_tensor_records'] and signs == control['rotation_sign_records']
             and all(len(v) == 2 for v in signs.values()) and actual_cache == control['calibration_cache'],
             'Actual protection/sign/calibration inputs differ from the matched D32 control.')
    stats = tensor_tree(activation['stats_by_linear'])
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'], 'donor_checkpoint_id': dm['checkpoint_id'],
        'base_checkpoint_id': BASE_ID, 'historical_control_checkpoint_id': CONTROL_ID, 'bit_budget': budget,
        'source_union_save_fresh_tensors': 2979, 'replaced_projection_count': 32, 'preserved_projection_count': 192,
        'protection_tensor_records': protected, 'rotation_sign_records': signs, 'calibration_cache': actual_cache,
        'calibration_stats': stats, 'calibration_stats_exact_control': stats == control['calibration_stats'],
        'actual_native_normalized_sha256': sha256(Path(native).parent / 'normalized_cat_runtime_args.json'),
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'cat_runtime_state_sha256': sha256(Path(native) / 'cat_runtime_state.pt'),
        'historical_control_audit_sha256': sha256(Path(record['historical_control_dir']) / 'audit.json')}
