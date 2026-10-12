"""CPU comparison of the defined fresh Llama OV and Qwen O rate pairs."""
from __future__ import annotations

import argparse
import copy
import inspect
import os
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, SHARDS, check, read
from experiments.full_rht_gate_pair.compare import delta, score
from experiments.llama_ov_bits_pair.audit import check_donor as llama_donor
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen_o_bits_pair.audit import check_donor as qwen_donor
from train_utils.checkpoint_v6 import load_v6_meta


CONTRACTS = {
    'llama': {'categories': ['v_proj', 'o_proj'], 'check_donor': llama_donor,
              'codebook_bits': {'control': {'v_proj': 32, 'o_proj': 48},
                                'candidate': {'v_proj': 40, 'o_proj': 56}},
              'total_bits': {'control': 17100439552, 'candidate': 17432838144},
              'replaced': 64, 'preserved': 160, 'state_count': 2979},
    'qwen3': {'categories': ['o_proj'], 'check_donor': qwen_donor,
              'codebook_bits': {'control': {'o_proj': 48}, 'candidate': {'o_proj': 56}},
              'total_bits': {'control': 17065672704, 'candidate': 17364123648},
              'replaced': 36, 'preserved': 216, 'state_count': 3423},
}


def normalized_recipe(record):
    """Remove only the actual output and category-K mirrors in the native recipe."""
    value = copy.deepcopy(record['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        _require(value[section]['output_dir'] == str(Path(record['output_dir']) / 'donor'),
                 'Normalized output differs from this arm: ' + section)
        value[section]['output_dir'] = '<paired-output>'
    for category, bits in record['codebook_bits'].items():
        for section in ('canonical_config', 'vae_args'):
            values = value[section]['codebook_bits']['by_category']
            _require(values[category] == bits, 'Normalized category K differs: ' + category)
            values[category] = '<paired-K>'
        resolved = value['resolved_category_runtime'][category]
        _require(resolved['codebook_bits'] == bits, 'Resolved category K differs: ' + category)
        resolved['codebook_bits'] = '<paired-K>'
    return value


def arm(directory, label):
    directory = Path(directory).resolve(); record = read(directory / 'launch.json')
    _require(record['model_type'] in CONTRACTS and record['arm'] == label,
             'Require one of the defined Llama OV or Qwen O arms.')
    contract = CONTRACTS[record['model_type']]
    _require(record['status'] == 'completed' and record['exit_code'] == 0
             and record['full_evaluation'] is True and record['steps_per_stage'] == 10000
             and record['block_size'] == 32 and record['categories'] == contract['categories']
             and record['codebook_bits'] == contract['codebook_bits'][label]
             and Path(record['output_dir']).resolve() == directory,
             f'{directory}: require the completed formal {label} arm.')
    for name in ('train', 'assembly', 'mmlu', 'seven'):
        _require(record['stages'][name]['exit_code'] == 0
                 and int((directory / (name + '.exit')).read_text()) == 0,
                 f'{directory}: incomplete/failed stage {name}.')
    for name in ('dispatcher', 'outer'):
        _require(int((directory / (name + '.exit')).read_text()) == 0,
                 f'{directory}: {name} has not exited successfully.')
    check(record)
    donor = Path(record['donor_checkpoint_dir'])
    donor_meta, names = contract['check_donor'](donor, record)
    checkpoint = directory / 'final_model'; meta = load_v6_meta(str(checkpoint))
    audit = read(directory / 'audit.json'); assembly = read(checkpoint / 'assembly_report.json')
    _require(record['base_checkpoint_id'] == EXPECTED_IDS[record['model_type']]
             and audit['status'] == assembly['status'] == 'verified'
             and audit['checkpoint_id'] == assembly['checkpoint_id'] == meta['checkpoint_id']
             and audit['donor_checkpoint_id'] == record['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and audit['base_checkpoint_id'] == record['base_checkpoint_id'],
             'Source, donor or deployed identity differs.')
    _require(meta['base_model_path'] == donor_meta['base_model_path']
             == record['architecture']['base_model_path'], 'Actual native HF model differs.')
    _require(audit['assembly_report_sha256'] == sha256(checkpoint / 'assembly_report.json')
             and audit['actual_native_normalized_sha256'] == sha256(donor.parent / 'normalized_cat_runtime_args.json')
             and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared']
             == audit['source_union_save_fresh_tensors'] == contract['state_count'],
             'Saved native audit evidence changed.')
    sources = assembly['sources']
    _require(sources['base_checkpoint_id'] == record['base_checkpoint_id']
             and sources['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and sources['categories'] == contract['categories'] and set(sources['replaced_targets']) == names
             and len(names) == audit['replaced_projection_count'] == contract['replaced']
             and audit['preserved_projection_count'] == contract['preserved'],
             'Assembly does not replace exactly the declared complete categories.')
    budget = audit['bit_budget']
    bits, weights = budget['user_linear_bits'], budget['original_linear_weights']
    width_passed = bits * 2 < weights * 5 if record['model_type'] == 'llama' else bits * 2 <= weights * 5
    _require(budget == assembly['bit_budget'] and budget['budget_passed'] and width_passed
             and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and bits == record['expected_user_linear_bits'] == contract['total_bits'][label]
             and weights == record['architecture']['original_linear_weights']
             and budget['compressed_target_count'] == record['architecture']['projection_count']
             and budget['components']['decoder']['dtypes'] == ['bfloat16'],
             'Actual storage differs from the approved arm budget.')
    for key in ('lora', 'residual_lora', 'dense_weights'):
        _require(budget['components'].get(key, {}).get('storage_bits', 0) == 0, 'Unexpected payload: ' + key)
    _require(set(audit['protection_tensor_records']) == set(audit['rotation_sign_records']) == names,
             'Incomplete category protection/sign inventory.')
    for name in names:
        _require(set(audit['protection_tensor_records'][name]) ==
                 {'protected_input_indices', 'protected_input_qvalues', 'protected_input_scales'}
                 and len(audit['rotation_sign_records'][name]) == 2,
                 'Incomplete protected payload/sign records: ' + name)
    paths = [directory / (name + '.json') for name, _, _ in SHARDS]
    result, shards = score(paths, record['architecture'], checkpoint)
    summary = read(directory / 'full8_summary.json')
    _require(sha256(directory / 'full8_summary.json') == record['full8_summary_sha256']
             and summary['checkpoint_id'] == meta['checkpoint_id']
             and summary['task_metrics'] == result['task_metrics']
             and summary['mean_percent'] == result['mean_percent'] == record['mean_percent']
             and summary['mmlu_correct'] == record['mmlu_correct'] == result['integer_counts']['mmlu']['correct'],
             'Stored complete summary differs from actual raw scores.')
    result['elapsed_seconds'] = record['finished_unix'] - record['started_unix']
    evidence = [directory / name for name in ('launch.json', 'audit.json', 'full8_summary.json',
        'train.log', 'train.exit', 'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')]
    evidence += [*paths, checkpoint / 'checkpoint_meta.json', checkpoint / 'assembly_report.json',
                 donor / 'checkpoint_meta.json', donor.parent / 'normalized_cat_runtime_args.json']
    return record, audit, result, shards, {str(path): sha256(path) for path in evidence}


def compare(control, candidate, output):
    output = Path(output).resolve()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Run this postprocessor with CUDA_VISIBLE_DEVICES empty.')
    _require(not output.exists() and not output.is_symlink(), 'Do not overwrite a comparison.')
    c, ca, cs, cr, ce = arm(control, 'control')
    n, na, ns, nr, ne = arm(candidate, 'candidate')
    for key in ('model_type', 'categories', 'architecture', 'base_dir', 'base_checkpoint_id',
                'control_result_paths', 'cwd', 'source_sha256', 'input_sha256', 'protected_stats',
                'environment', 'protocol', 'comparison_boundary'):
        _require(c[key] == n[key], 'Pair source/protocol differs: ' + key)
    _require(normalized_recipe(c) == normalized_recipe(n),
             'Pair native recipe differs beyond output directory and category K.')
    _require(ca['protection_tensor_records'] == na['protection_tensor_records'],
             'Pair protected indices/qvalues/scales differ; rate is not the only variable.')
    _require(ca['rotation_sign_records'] == na['rotation_sign_records'],
             'Pair persistent rotation signs differ despite matched seed/geometry.')
    cb, nb = ca['bit_budget'], na['bit_budget']
    _require(cb['original_linear_weights'] == nb['original_linear_weights']
             and set(cb['components']) == set(nb['components']), 'Pair storage inventory differs.')
    for key in set(cb['components']) - {'code', 'decoder'}:
        _require(cb['components'][key] == nb['components'][key], 'Unchanged storage component differs: ' + key)
    base = Path(c['base_dir']); base_meta = load_v6_meta(str(base))
    _require(base_meta['checkpoint_id'] == c['base_checkpoint_id'], 'Original compressed base identity changed.')
    baseline, br = score([Path(p) for p in c['control_result_paths']], c['architecture'], base)
    for label, rows in (('candidate', nr), ('base', br)):
        for current, reference in zip(rows, cr, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == reference[key], f'{label} benchmark protocol differs: {key}')
    budget_keys = ('original_linear_weights', 'user_linear_bits', 'user_linear_bpw', 'packed_linear_bits', 'components')
    dependencies = [Path(__file__), Path(inspect.getfile(score)), Path(inspect.getfile(llama_donor)),
                    Path(inspect.getfile(qwen_donor))]
    report = {'status': 'verified_pair', 'model_type': c['model_type'], 'categories': c['categories'],
        'primary_comparison': 'candidate_minus_fresh_control', 'candidate_vs_control': delta(ns, cs),
        'secondary_vs_original_compressed_base': {'control': delta(cs, baseline), 'candidate': delta(ns, baseline)},
        'arms': {'control': cs, 'candidate': ns, 'base': baseline},
        'codebook_bits': {'control': c['codebook_bits'], 'candidate': n['codebook_bits']},
        'pair_checks': {'same_native_recipe_except_output_and_category_K': True,
            'same_base_hf_source_and_calibration': True, 'actual_steps_per_stage': 10000, 'stages': 2,
            'replaced_projection_count': ca['replaced_projection_count'],
            'protection_tensor_records_exact': True, 'rotation_sign_records_exact': True,
            'same_raw_task_protocol': True, 'leaf_count': 64, 'sample_count': 24742,
            'mmlu_subjects': 57, 'mmlu_samples': 14042, 'batch_sizes': {'mmlu': 64, 'seven': 32},
            'num_fewshot': 0, 'limit': None},
        'bit_budgets': {'control': {key: cb[key] for key in budget_keys},
                        'candidate': {key: nb[key] for key in budget_keys}},
        'storage_delta': {'user_linear_bits': nb['user_linear_bits'] - cb['user_linear_bits'],
                          'user_linear_bpw': nb['user_linear_bpw'] - cb['user_linear_bpw']},
        'evidence_sha256': {**ce, **ne, str(base / 'checkpoint_meta.json'): sha256(base / 'checkpoint_meta.json'),
                           **{p: sha256(p) for p in c['control_result_paths']}},
        'comparison_source_sha256': {str(p): sha256(p) for p in dependencies},
        'comparison_boundary': c['comparison_boundary'],
        'quality_boundary': 'A valid negative comparison is saved successfully; no MMLU or eight-task threshold is imposed.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    _require(not torch.cuda.is_initialized(), 'CPU comparison initialized CUDA.')
    write_json(output, report)
    print({'status': report['status'], 'model_type': c['model_type'],
           'candidate_minus_control_mean_pp': report['candidate_vs_control']['mean_delta_pp'],
           'candidate_minus_control_mmlu_correct': report['candidate_vs_control']['correct_delta']['mmlu'],
           'output': str(output), 'sha256': sha256(output)}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.control, args.candidate, args.output)
