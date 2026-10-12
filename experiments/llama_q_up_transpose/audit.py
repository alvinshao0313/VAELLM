"""Audit actual single-category deployment and compare each fresh orientation pair."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.audit import results
from experiments.full_rht_gate_pair.common import EXPECTED_IDS, SHARDS, check, read
from experiments.full_rht_gate_pair.compare import delta, score
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import load_v6_meta


def check_donor(directory, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    actual = read(directory.parent / 'normalized_cat_runtime_args.json')
    _require(Path(actual['cat_args']['output_dir']).resolve() == directory.parent,
             'Native normalized output does not name its actual run.')
    actual['cat_args']['output_dir'] = str(Path(record['output_dir']) / 'donor')
    _require(actual == record['normalized_expected'], 'Actual native parameters differ from preparation.')
    architecture = record['architecture']; layers = architecture['num_hidden_layers']
    _require(record['categories'] in (['q_proj'], ['up_proj']), 'Require one declared category.')
    category = record['categories'][0]
    _require(record['model_type'] == architecture['model_type'] == 'llama'
             and record['codebook_bits'] == {category: 32} and record['dimension'] == 32
             and record['block_size'] == 32 and type(record['transpose']) is bool
             and record['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and layers == 32 and architecture['projection_count'] == 224
             and meta['base_model_path'] == architecture['base_model_path'],
             'Require the original-source Llama q/up D32 recipe.')
    names = {name for name in architecture['projection_shapes'] if name.endswith('.' + category)}
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(len(names) == layers and set(meta['compressed_targets']) == set(specs) == names
             and meta['compression_categories'] == meta['completed_categories'] == [category]
             and meta['after_category_mode'] == 'none', 'Donor must contain the complete fresh category only.')
    expected_shape = [4096, 4096] if category == 'q_proj' else [14336, 4096]
    for name, spec in specs.items():
        out_features, in_features = architecture['projection_shapes'][name]
        blocks = out_features * (in_features - 64) // 32
        _require([out_features, in_features] == expected_shape
                 and [spec['out_features'], spec['in_features']] == expected_shape
                 and spec['compressed_in_features'] == in_features - 64
                 and spec['compressed_out_features'] == out_features and spec['transpose'] is record['transpose']
                 and spec['codebook_dim'] == 32 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32] and spec['parallel_parts'] == 1,
                 'Projection geometry changed: ' + name)
        _require(len(spec['stage_vq_weights']) == len(spec['stage_decoders']) == 2,
                 'Expected exactly two packed residual stages: ' + name)
        for packed in spec['stage_vq_weights']:
            _require(packed == {'shape': [blocks, 1, 4], 'dtype': 'uint8',
                     'storage_format': 'bitpack_u8', 'logical_shape': [blocks, 1, 32],
                     'logical_dtype': 'bool', 'pack_bits': 8}, 'Packed code geometry changed: ' + name)
        _require(spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [64, out_features], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and spec['protected_output_indices'] is None
                 and spec['protected_channel_quant_format'] == 'int8', 'Protection changed: ' + name)
        _require(spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == 32, 'Wrong actual rotation: ' + name)
        _require(all(stage['in_dim'] == stage['out_dim'] == 32 and stage['hidden_dim'] == 128
                     and stage['num_res_blocks'] == 0 and stage['norm_type'] == 'layer'
                     and stage['activation_type'] == 'swish' and stage['decoder_type'] == 'symmetric'
                     and stage['param_dtype'] == 'bfloat16' for stage in spec['stage_decoders']),
                 'Decoder architecture differs: ' + name)
    steps = record['steps_per_stage']; log = (Path(record['output_dir']) / 'train.log').read_text()
    _require(steps == 10000, 'Require10000 native updates per residual stage.')
    for stage in (1, 2):
        _require(f'[{category}.L0-{layers-1}/stage{stage}] step={steps}/{steps} ' in log,
                 'Native single-category training did not finish both stages.')
    return meta, names


def check_assembly(directory, donor, record):
    directory = Path(directory); meta = load_v6_meta(str(directory))
    report = read(directory / 'assembly_report.json'); donor_meta, names = check_donor(donor, record)
    category = record['categories'][0]
    reference = read(record['storage_reference_audit_path'])
    _require(reference['status'] == 'verified'
             and reference['checkpoint_id'] == record['storage_reference_checkpoint_id']
             == '0ad677ce-a49a-423e-a20c-50f8aa1ef710'
             and reference['base_checkpoint_id'] == record['base_checkpoint_id'], 'Storage reference changed.')
    source = report['sources']
    _require(report['status'] == 'verified' and report['checkpoint_id'] == meta['checkpoint_id']
             and source['base_checkpoint_id'] == record['base_checkpoint_id']
             and source['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and source['categories'] == [category] and set(source['replaced_targets']) == names,
             'Assembly differs from the actual source union.')
    _require(report['serialized_tensors_compared'] == report['reloaded_tensors_compared'] == 2979,
             'Source-union/save/fresh strict inventory is incomplete.')
    budget = report['bit_budget']; bits, weights = budget['user_linear_bits'], budget['original_linear_weights']
    _require(budget['budget_passed'] and bits * 2 < weights * 5 and not budget['contract_violations']
             and not budget['unclassified_target_keys'] and weights == 6979321856
             and bits == budget['packed_linear_bits'] == record['expected_user_linear_bits'] == 17100439552
             and budget['compressed_target_count'] == record['architecture']['projection_count'] == 224
             and budget['components'] == reference['bit_budget']['components']
             and budget['components']['decoder']['dtypes'] == ['bfloat16'] and budget['code_padding_bits'] == 0,
             'Actual final storage differs from the fixed integer budget.')
    for key in ('lora', 'residual_lora', 'dense_weights'):
        _require(budget['components'].get(key, {}).get('storage_bits', 0) == 0, 'Unexpected payload: ' + key)
    state = torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    _require(len(state) == 2979, 'Persistent inventory changed after assembly.')
    protection = {name: {field: tensor_record(state[name + '.' + field]) for field in
        ('protected_input_indices', 'protected_input_qvalues', 'protected_input_scales')}
        for name in sorted(names)}
    signs = {name: {key: tensor_record(value) for key, value in state.items()
                    if key.startswith(name + '.') and key.endswith(('.input_signs', '.output_signs'))}
             for name in sorted(names)}
    _require(all(len(value) == 2 for value in signs.values()), 'Expected two persistent sign vectors per projection.')
    return {'status': 'verified', 'checkpoint_id': meta['checkpoint_id'],
        'donor_checkpoint_id': donor_meta['checkpoint_id'], 'base_checkpoint_id': record['base_checkpoint_id'],
        'category': category, 'transpose': record['transpose'], 'bit_budget': budget,
        'source_union_save_fresh_tensors': len(state), 'replaced_projection_count': len(names),
        'preserved_projection_count': 224 - len(names), 'protection_tensor_records': protection,
        'rotation_sign_records': signs, 'assembly_report_sha256': sha256(directory / 'assembly_report.json'),
        'actual_native_normalized_sha256': sha256(Path(donor).parent / 'normalized_cat_runtime_args.json')}


def completed_arm(directory, transpose):
    directory = Path(directory).resolve(); record = read(directory / 'launch.json')
    _require(record['status'] == 'completed' and record['exit_code'] == 0
             and record['full_evaluation'] is True and record['steps_per_stage'] == 10000
             and record['transpose'] is transpose and Path(record['output_dir']).resolve() == directory,
             'Require a completed formal single-category orientation arm.')
    for name in ('train', 'assembly', 'mmlu', 'seven'):
        _require(record['stages'][name]['exit_code'] == 0
                 and int((directory / (name + '.exit')).read_text()) == 0, 'Incomplete stage: ' + name)
    for name in ('dispatcher', 'outer'):
        _require(int((directory / (name + '.exit')).read_text()) == 0, 'Incomplete launcher: ' + name)
    check(record)
    donor = Path(record['donor_checkpoint_dir']); donor_meta, names = check_donor(donor, record)
    checkpoint = directory / 'final_model'; meta = load_v6_meta(str(checkpoint))
    audit = read(directory / 'audit.json'); assembly = read(checkpoint / 'assembly_report.json')
    _require(audit['status'] == assembly['status'] == 'verified'
             and audit['checkpoint_id'] == assembly['checkpoint_id'] == meta['checkpoint_id'] == record['checkpoint_id']
             and audit['donor_checkpoint_id'] == donor_meta['checkpoint_id'] == record['donor_checkpoint_id']
             and audit['base_checkpoint_id'] == record['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and audit['category'] == record['categories'][0] and audit['transpose'] is transpose
             and audit['replaced_projection_count'] == 32 and audit['preserved_projection_count'] == 192
             and meta['base_model_path'] == donor_meta['base_model_path'] == record['architecture']['base_model_path'],
             'Completed source, donor, or deployed identity differs.')
    _require(audit['assembly_report_sha256'] == sha256(checkpoint / 'assembly_report.json')
             and audit['actual_native_normalized_sha256'] == sha256(donor.parent / 'normalized_cat_runtime_args.json')
             and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared']
             == audit['source_union_save_fresh_tensors'] == 2979, 'Saved strict deployment evidence changed.')
    source = assembly['sources']; budget = audit['bit_budget']
    _require(source['base_checkpoint_id'] == record['base_checkpoint_id']
             and source['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and source['categories'] == record['categories'] and set(source['replaced_targets']) == names
             and budget == assembly['bit_budget'] and budget['budget_passed']
             and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17100439552
             and budget['original_linear_weights'] == 6979321856 and budget['compressed_target_count'] == 224,
             'Actual assembly or integer budget differs from the approved source union.')
    _require(set(audit['protection_tensor_records']) == set(audit['rotation_sign_records']) == names,
             'Incomplete protection/sign inventory.')
    paths = [directory / (name + '.json') for name, _, _ in SHARDS]
    result, shards = score(paths, record['architecture'], checkpoint)
    summary = read(directory / 'full8_summary.json')
    _require(sha256(directory / 'full8_summary.json') == record['full8_summary_sha256']
             and summary['checkpoint_id'] == meta['checkpoint_id'] and summary['task_metrics'] == result['task_metrics']
             and summary['integer_counts'] == result['integer_counts']
             and summary['mean_percent'] == result['mean_percent'] == record['mean_percent']
             and summary['mmlu_correct'] == record['mmlu_correct'] == result['integer_counts']['mmlu']['correct'],
             'Stored full8 summary differs from complete raw results.')
    evidence = [directory / name for name in ('launch.json', 'audit.json', 'full8_summary.json',
        'train.log', 'train.exit', 'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')]
    evidence += [*paths, checkpoint / 'checkpoint_meta.json', checkpoint / 'assembly_report.json',
                 donor / 'checkpoint_meta.json', donor.parent / 'normalized_cat_runtime_args.json']
    return record, audit, result, shards, {str(path): sha256(path) for path in evidence}


def normalized_pair_recipe(record):
    value = copy.deepcopy(record['normalized_expected']); category = record['categories'][0]
    for section in ('canonical_config', 'cat_args'):
        _require(value[section]['output_dir'] == str(Path(record['output_dir']) / 'donor'), 'Wrong paired output.')
        value[section]['output_dir'] = '<paired-output>'
    for section in (value['canonical_config']['core_template'], value['cat_args']):
        selected = section['transpose_modules'].split(',')
        _require((category in selected) is record['transpose'], 'Wrong actual paired orientation.')
        section['transpose_modules'] = ','.join(name for name in selected if name != category)
    return value


def compare(nontranspose, transpose, output):
    output = Path(output).resolve()
    _require(not output.exists() and not output.is_symlink(), 'Do not overwrite a completed pair comparison.')
    f, fa, fs, fr, fe = completed_arm(nontranspose, False)
    t, ta, ts, tr, te = completed_arm(transpose, True)
    for key in ('model_type', 'categories', 'architecture', 'base_dir', 'base_checkpoint_id', 'control_result_paths',
                'cwd', 'source_sha256', 'input_sha256', 'protected_stats', 'environment', 'protocol'):
        _require(f[key] == t[key], 'Matched pair source/protocol differs: ' + key)
    _require(normalized_pair_recipe(f) == normalized_pair_recipe(t),
             'Pair native recipe differs beyond output and the selected category orientation.')
    _require(fa['protection_tensor_records'] == ta['protection_tensor_records']
             and fa['rotation_sign_records'] == ta['rotation_sign_records'],
             'Pair protection/signs differ; this is not an isolated orientation comparison.')
    for key in ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits', 'components'):
        _require(fa['bit_budget'][key] == ta['bit_budget'][key], 'Pair actual storage differs: ' + key)
    base = Path(f['base_dir']); control, cr = score([Path(p) for p in f['control_result_paths']], f['architecture'], base)
    for label, rows in (('transpose', tr), ('original3577', cr)):
        for current, reference in zip(rows, fr, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == reference[key], label + ' task protocol differs: ' + key)
    report = {'status': 'verified_pair', 'model_type': 'llama', 'category': f['categories'][0],
        'primary_comparison': 'fresh_transpose_minus_fresh_nontranspose', 'transpose_vs_nontranspose': delta(ts, fs),
        'secondary_vs_original3577': {'nontranspose': delta(fs, control), 'transpose': delta(ts, control)},
        'arms': {'nontranspose': fs, 'transpose': ts, 'original3577': control},
        'pair_checks': {'same_native_recipe_except_output_and_selected_orientation': True,
            'same_base_hf_source_and_calibration': True, 'independent_single_category_runs': True,
            'actual_steps_per_stage': 10000, 'stages': 2, 'complete_category_count': 32,
            'preserved_other_projection_count': 192, 'source_union_save_fresh_tensors': 2979,
            'protection_tensor_records_exact': True, 'rotation_sign_records_exact': True,
            'same_raw_task_protocol': True, 'leaf_count': 64, 'sample_count': 24742,
            'mmlu_subjects': 57, 'mmlu_samples': 14042, 'batch_sizes': {'mmlu': 64, 'seven': 32},
            'num_fewshot': 0, 'limit': None},
        'bit_budget': {key: ta['bit_budget'][key] for key in
                       ('user_linear_bits', 'user_linear_bpw', 'packed_linear_bits', 'components')},
        'evidence_sha256': {**fe, **te, str(base / 'checkpoint_meta.json'): sha256(base / 'checkpoint_meta.json'),
                           **{p: sha256(p) for p in f['control_result_paths']}},
        'comparison_source_sha256': sha256(Path(__file__)),
        'quality_boundary': 'Negative results are valid. Do not infer additive cross-category gains or whole-model pure CAT.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    write_json(output, report)
    print({'status': report['status'], 'category': report['category'],
           'transpose_minus_nontranspose_mean_pp': report['transpose_vs_nontranspose']['mean_delta_pp'],
           'sha256': sha256(output)}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nontranspose', type=Path, required=True)
    parser.add_argument('--transpose', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.nontranspose, args.transpose, args.output)
