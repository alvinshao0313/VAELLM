"""Validate actual donor geometry, source union, and complete paired task results."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_BITS, SHARDS, read
from experiments.longbench_v1.common import sha256
from experiments.qwen3_w25.assemble_categories import _require
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from train_utils.checkpoint_v6 import load_v6_meta


def check_donor(directory, record):
    directory = Path(directory)
    meta = load_v6_meta(str(directory))
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent,
             'Native normalized output does not name its actual run.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Native CAT parameters differ from CPU preparation.')
    architecture = record['architecture']
    names = {name for name in architecture['projection_shapes'] if name.endswith('.gate_proj')}
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(set(meta['compressed_targets']) == set(specs) == names
             and meta['compression_categories'] == meta['completed_categories'] == ['gate_proj']
             and meta['after_category_mode'] == 'none', 'Donor must contain the complete fresh gate category only.')
    for name, spec in specs.items():
        out_features, in_features = architecture['projection_shapes'][name]
        _require([spec['out_features'], spec['in_features']] == [out_features, in_features]
                 and spec['compressed_in_features'] == in_features - 64
                 and spec['compressed_out_features'] == out_features and not spec['transpose']
                 and spec['codebook_dim'] == 32 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32] and spec['parallel_parts'] == 1,
                 'Gate geometry changed: ' + name)
        _require(spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [64, out_features], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and spec['protected_output_indices'] is None
                 and spec['protected_channel_quant_format'] == 'int8', 'Protection changed: ' + name)
        _require(spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == record['block_size'], 'Wrong actual rotation: ' + name)
        _require(all(stage['in_dim'] == stage['out_dim'] == 32 and stage['hidden_dim'] == 128
                     and stage['num_res_blocks'] == 0 and stage['norm_type'] == 'layer'
                     and stage['activation_type'] == 'swish' and stage['decoder_type'] == 'symmetric'
                     and stage['param_dtype'] == 'bfloat16' for stage in spec['stage_decoders']),
                 'Decoder architecture differs: ' + name)
    log = (Path(record['output_dir']) / 'train.log').read_text()
    layers, steps = architecture['num_hidden_layers'], record['steps_per_stage']
    for stage in (1, 2):
        _require(f'[gate_proj.L0-{layers-1}/stage{stage}] step={steps}/{steps} ' in log,
                 'Native training did not finish both declared stages.')
    return meta, names


def check_assembly(directory, donor, record):
    directory = Path(directory)
    meta = load_v6_meta(str(directory))
    report = read(directory / 'assembly_report.json')
    donor_meta, names = check_donor(donor, record)
    source = report['sources']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id']
             and source['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and source['categories'] == ['gate_proj'] and set(source['replaced_targets']) == names,
             'Assembled checkpoint does not come from the actual paired sources.')
    _require(report['serialized_tensors_compared'] == report['reloaded_tensors_compared']
             and report['serialized_tensors_compared'] > len(names), 'Native save/fresh strict audit is incomplete.')
    budget = report['bit_budget']
    _require(budget['budget_passed'] and not budget['contract_violations']
             and not budget['unclassified_target_keys']
             and budget['original_linear_weights'] == record['architecture']['original_linear_weights']
             and budget['user_linear_bits'] == EXPECTED_BITS[record['model_type']]
             and budget['compressed_target_count'] == record['architecture']['projection_count'],
             'Actual assembled tensor storage differs from the approved budget.')
    if record['model_type'] == 'llama':
        _require(budget['user_linear_bpw'] < 2.5, 'Llama must remain strictly below 2.5 bits.')
    for key in ('lora', 'residual_lora', 'dense_weights'):
        _require(budget['components'].get(key, {}).get('storage_bits', 0) == 0, 'Unexpected payload: ' + key)
    state = torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == report['reloaded_tensors_compared'], 'Fresh state count changed after assembly.')
    protection = {name: {field: tensor_record(state[name + '.' + field]) for field in
        ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')}
        for name in sorted(names)}
    # Rotation signs are persistent metadata; equal signs alone do not imply equal transforms.
    signs = {name: {key: tensor_record(value) for key, value in state.items()
                    if key.startswith(name + '.') and key.endswith(('.input_signs', '.output_signs'))}
             for name in sorted(names)}
    _require(all(len(value) == 2 for value in signs.values()), 'Expected two persistent rotation sign vectors per gate.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'],
        'donor_checkpoint_id': donor_meta['checkpoint_id'], 'base_checkpoint_id': record['base_checkpoint_id'],
        'bit_budget': budget, 'source_union_save_fresh_tensors': len(state),
        'gate_count': len(names), 'preserved_projection_count': record['architecture']['projection_count'] - len(names),
        'protection_tensor_records': protection, 'rotation_sign_records': signs,
        'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(Path(donor).parent / 'normalized_cat_runtime_args.json')}


def results(directory, record):
    directory = Path(directory)
    checkpoint = directory / 'final_model'
    meta = load_v6_meta(str(checkpoint))
    paths = [directory / (name + '.json') for name, _, _ in SHARDS]
    result = collect_results(paths, record['architecture'], checkpoint=(checkpoint, meta))
    base = Path(record['base_dir'])
    control = collect_results(record['control_result_paths'], record['architecture'],
                              checkpoint=(base, load_v6_meta(str(base))))
    for path, reference in zip(paths, record['control_result_paths'], strict=True):
        current = read(path)['evals']['lm_eval']; old = read(reference)['evals']['lm_eval']
        for key in ('configs', 'versions', 'n_shot', 'n_samples'):
            _require(current[key] == old[key], 'Reference task protocol changed: ' + key)
    return {'status': 'complete_full8', 'checkpoint_id': meta['checkpoint_id'], **result,
        'base_mean_percent': control['mean_percent'],
        'mean_delta_from_base_pp': result['mean_percent'] - control['mean_percent'],
        'mmlu_correct': round(result['task_metrics']['mmlu'] * 14042),
        'task_delta_from_base_pp': {key: 100 * (value - control['task_metrics'][key])
                                   for key, value in result['task_metrics'].items()},
        'rotation_pair_comparison': 'pending_other_arm',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'goal_acceptance': False}
