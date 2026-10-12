"""CPU-only comparison of completed fresh block32/full down-category runs."""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch

from experiments.full_rht_down_pair.audit import check_donor
from experiments.full_rht_gate_pair.common import EXPECTED_BITS, EXPECTED_IDS, SHARDS, check, read
from experiments.full_rht_gate_pair.compare import delta, normalized_recipe, score
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.checkpoint_v6 import load_v6_meta


def arm(directory, block_size):
    directory = Path(directory).resolve(); record = read(directory / 'launch.json')
    _require(record['status'] == 'completed' and record['exit_code'] == 0
             and record['full_evaluation'] is True and record['steps_per_stage'] == 10000
             and record['block_size'] == block_size and record['categories'] == ['down_proj']
             and record['codebook_bits'] == {'down_proj': 48}
             and record['model_type'] in EXPECTED_IDS and Path(record['output_dir']).resolve() == directory,
             'Require one completed native 10000/stage down arm of the requested rotation.')
    for name in ('train', 'assembly', 'mmlu', 'seven'):
        _require(record['stages'][name]['exit_code'] == 0
                 and int((directory / (name + '.exit')).read_text()) == 0, 'Failed/incomplete stage: ' + name)
    for name in ('dispatcher', 'outer'):
        _require(int((directory / (name + '.exit')).read_text()) == 0, 'Source process has not exited: ' + name)
    check(record)
    donor = Path(record['donor_checkpoint_dir']); donor_meta, names = check_donor(donor, record)
    checkpoint = directory / 'final_model'; meta = load_v6_meta(str(checkpoint))
    audit = read(directory / 'audit.json'); assembly = read(checkpoint / 'assembly_report.json')
    model = record['model_type']; layers = 36 if model == 'qwen3' else 32
    state_count = 3423 if model == 'qwen3' else 2979
    _require(record['base_checkpoint_id'] == EXPECTED_IDS[model]
             and audit['status'] == assembly['status'] == 'verified'
             and audit['checkpoint_id'] == assembly['checkpoint_id'] == meta['checkpoint_id']
             and audit['base_checkpoint_id'] == record['base_checkpoint_id']
             and audit['donor_checkpoint_id'] == record['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and meta['base_model_path'] == donor_meta['base_model_path'] == record['architecture']['base_model_path'],
             'Source, donor, original HF or deployed UUID differs.')
    _require(audit['assembly_report_sha256'] == sha256(checkpoint / 'assembly_report.json')
             and audit['actual_native_normalized_sha256'] == sha256(donor.parent / 'normalized_cat_runtime_args.json')
             and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared']
             == audit['source_union_save_fresh_tensors'] == state_count,
             'Completed native source-union/save/fresh evidence differs.')
    sources = assembly['sources']
    _require(sources['base_checkpoint_id'] == record['base_checkpoint_id']
             and sources['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and sources['categories'] == ['down_proj'] and set(sources['replaced_targets']) == names
             and len(names) == audit['down_count'] == audit['replaced_projection_count'] == layers
             and audit['preserved_projection_count'] == layers * 6,
             'The assembly did not replace exactly the complete down category.')
    budget = audit['bit_budget']; bits = budget['user_linear_bits']; weights = budget['original_linear_weights']
    width_passed = 2 * bits < 5 * weights if model == 'llama' else 2 * bits <= 5 * weights
    _require(budget == assembly['bit_budget'] and budget['budget_passed'] and width_passed
             and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and bits == record['expected_user_linear_bits'] == EXPECTED_BITS[model]
             and weights == record['architecture']['original_linear_weights']
             and budget['compressed_target_count'] == layers * 7
             and budget['components']['decoder']['dtypes'] == ['bfloat16'], 'Actual source storage differs.')
    _require(all(budget['components'].get(key, {}).get('storage_bits', 0) == 0
                 for key in ('lora', 'residual_lora', 'dense_weights')), 'Unexpected added payload.')
    _require(set(audit['protection_tensor_records']) == set(audit['rotation_sign_records']) == names,
             'Incomplete down protection/sign records.')
    for name in names:
        _require(set(audit['protection_tensor_records'][name]) ==
                 {'protected_input_indices', 'protected_input_qvalues', 'protected_input_scales'}
                 and len(audit['rotation_sign_records'][name]) == 2, 'Incomplete down persistent records: ' + name)
    paths = [directory / (name + '.json') for name, _, _ in SHARDS]
    result, shards = score(paths, record['architecture'], checkpoint)
    summary = read(directory / 'full8_summary.json')
    _require(sha256(directory / 'full8_summary.json') == record['full8_summary_sha256']
             and summary['checkpoint_id'] == meta['checkpoint_id']
             and summary['task_metrics'] == result['task_metrics']
             and summary['mean_percent'] == result['mean_percent'] == record['mean_percent']
             and summary['mmlu_correct'] == record['mmlu_correct'] == result['integer_counts']['mmlu']['correct'],
             'Stored summary differs from the complete 64-leaf raw evaluation.')
    evidence = [directory / name for name in ('launch.json', 'audit.json', 'full8_summary.json',
        'train.log', 'train.exit', 'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')]
    evidence += [*paths, checkpoint / 'checkpoint_meta.json', checkpoint / 'assembly_report.json',
                 donor / 'checkpoint_meta.json', donor.parent / 'normalized_cat_runtime_args.json']
    return record, audit, result, shards, {str(path): sha256(path) for path in evidence}


def compare(block32, full, output):
    output = Path(output).resolve()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Use CUDA_VISIBLE_DEVICES empty for this CPU postprocessor.')
    _require(not output.exists() and not output.is_symlink(), 'Do not overwrite an existing comparison.')
    b, ba, bs, br, be = arm(block32, 32)
    f, fa, fs, fr, fe = arm(full, 0)
    for key in ('model_type', 'categories', 'codebook_bits', 'architecture', 'base_dir', 'base_checkpoint_id',
                'control_result_paths', 'cwd', 'source_sha256', 'input_sha256', 'protected_stats',
                'environment', 'protocol', 'comparison_boundary'):
        _require(b[key] == f[key], 'Pair source/protocol differs: ' + key)
    _require(normalized_recipe(b) == normalized_recipe(f),
             'Pair recipe differs beyond output directory and rotation block size.')
    _require(ba['protection_tensor_records'] == fa['protection_tensor_records'],
             'Pair protected indices/qvalues/scales differ; this is not an isolated rotation comparison.')
    _require(ba['rotation_sign_records'] == fa['rotation_sign_records'],
             'Pair persistent sign records differ despite the same seed/geometry.')
    for key in ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits', 'components'):
        _require(ba['bit_budget'][key] == fa['bit_budget'][key], 'Pair actual storage differs: ' + key)
    base = Path(b['base_dir']); base_meta = load_v6_meta(str(base))
    _require(base_meta['checkpoint_id'] == b['base_checkpoint_id'], 'Original compressed base UUID changed.')
    control, cr = score([Path(p) for p in b['control_result_paths']], b['architecture'], base)
    for label, rows in (('full', fr), ('base', cr)):
        for current, reference in zip(rows, br, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == reference[key], label + ' benchmark protocol differs: ' + key)
    source_paths = [Path(__file__), Path(inspect.getfile(score)), Path(inspect.getfile(check_donor))]
    report = {'status': 'verified_pair', 'model_type': b['model_type'], 'categories': ['down_proj'],
        'primary_comparison': 'full_minus_fresh_block32', 'full_vs_block32': delta(fs, bs),
        'secondary_vs_original_compressed_base': {'block32': delta(bs, control), 'full': delta(fs, control)},
        'arms': {'block32': bs, 'full': fs, 'base': control},
        'pair_checks': {'same_native_recipe_except_output_and_block_size': True,
            'same_base_hf_source_and_calibration': True, 'actual_steps_per_stage': 10000, 'stages': 2,
            'complete_down_count': ba['down_count'], 'protection_tensor_records_exact': True,
            'rotation_sign_records_exact': True, 'same_actual_integer_storage': True,
            'same_raw_task_protocol': True, 'leaf_count': 64, 'sample_count': 24742,
            'mmlu_subjects': 57, 'mmlu_samples': 14042, 'batch_sizes': {'mmlu': 64, 'seven': 32},
            'num_fewshot': 0, 'limit': None},
        'bit_budget': {key: fa['bit_budget'][key] for key in
                       ('original_linear_weights', 'user_linear_bits', 'user_linear_bpw', 'packed_linear_bits', 'components')},
        'evidence_sha256': {**be, **fe, str(base / 'checkpoint_meta.json'): sha256(base / 'checkpoint_meta.json'),
                           **{p: sha256(p) for p in b['control_result_paths']}},
        'comparison_source_sha256': {str(p): sha256(p) for p in source_paths},
        'comparison_boundary': b['comparison_boundary'],
        'quality_boundary': 'A valid negative comparison is saved successfully; no MMLU or eight-task threshold is imposed.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    _require(not torch.cuda.is_initialized(), 'CPU postprocessor initialized CUDA.')
    write_json(output, report)
    print({'status': report['status'], 'model_type': b['model_type'],
           'full_minus_block32_mean_pp': report['full_vs_block32']['mean_delta_pp'],
           'full_minus_block32_mmlu_correct': report['full_vs_block32']['correct_delta']['mmlu'],
           'output': str(output), 'sha256': sha256(output)}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block32', type=Path, required=True)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.block32, args.full, args.output)
