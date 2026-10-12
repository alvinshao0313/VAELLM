"""CPU-only comparison of completed fresh block32/full gate-category runs."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from experiments.full_rht_gate_pair.audit import check_donor
from experiments.full_rht_gate_pair.common import EXPECTED_BITS, EXPECTED_IDS, SHARDS, check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.acceptance import TASK_METRICS, collect_results
from train_utils.checkpoint_v6 import load_v6_meta


def normalized_recipe(record):
    """Remove only the two intended pair differences, including their mirrors."""
    value = copy.deepcopy(record['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        _require(value[section]['output_dir'] == str(Path(record['output_dir']) / 'donor'),
                 'Normalized output differs from this arm: ' + section)
        value[section]['output_dir'] = '<paired-output>'
    for section in (value['canonical_config']['core_template'], value['vae_args']):
        _require(section['weight_rotation_block_size'] == record['block_size'],
                 'Normalized rotation differs from this arm.')
        section['weight_rotation_block_size'] = '<paired-block-size>'
    return value


def score(paths, architecture, checkpoint):
    meta = load_v6_meta(str(checkpoint))
    result = collect_results(paths, architecture, checkpoint=(checkpoint, meta))
    counts, shards = {}, []
    for path in paths:
        raw = read(path)['evals']['lm_eval']; shards.append(raw)
        for task, count in raw['n_samples'].items():
            metric = 'acc,none' if task.startswith('mmlu_') else TASK_METRICS[task]
            total = count['effective']; exact = raw['raw_results'][task][metric] * total
            correct = round(exact)
            _require(math.isclose(exact, correct, rel_tol=0, abs_tol=1e-8),
                     'Noninteger raw correct count: ' + task)
            counts[task] = {'correct': correct, 'total': total, 'metric': metric}
    subjects = [task for task in counts if task.startswith('mmlu_')]
    _require(len(counts) == 64 and len(subjects) == 57
             and sum(row['total'] for row in counts.values()) == 24742
             and sum(counts[task]['total'] for task in subjects) == 14042,
             'Require all 64 leaves, including 57 complete MMLU subjects.')
    counts['mmlu'] = {'correct': sum(counts[task]['correct'] for task in subjects),
                      'total': 14042, 'metric': 'weighted_57_subject_acc'}
    _require(math.isclose(counts['mmlu']['correct'] / 14042,
                         result['task_metrics']['mmlu'], rel_tol=0, abs_tol=1e-12),
             'MMLU integer total differs from the native score.')
    result.update(checkpoint_id=meta['checkpoint_id'], integer_counts=counts)
    return result, shards


def arm(directory, block_size):
    directory = Path(directory).resolve(); record = read(directory / 'launch.json')
    _require(record['status'] == 'completed' and record['exit_code'] == 0
             and record['full_evaluation'] is True and record['steps_per_stage'] == 10000
             and record['block_size'] == block_size
             and Path(record['output_dir']).resolve() == directory,
             f'{directory}: require the completed formal block_size={block_size}, 10000/stage arm.')
    for name in ('train', 'assembly', 'mmlu', 'seven'):
        _require(record['stages'][name]['exit_code'] == 0
                 and int((directory / (name + '.exit')).read_text()) == 0,
                 f'{directory}: incomplete/failed stage {name}.')
    for name in ('dispatcher', 'outer'):
        _require(int((directory / (name + '.exit')).read_text()) == 0,
                 f'{directory}: {name} has not exited successfully.')
    check(record)
    donor = Path(record['donor_checkpoint_dir'])
    donor_meta, names = check_donor(donor, record)  # Actual normalized argv and both 10000/10000 log entries.
    checkpoint = directory / 'final_model'; meta = load_v6_meta(str(checkpoint))
    audit = read(directory / 'audit.json'); assembly = read(checkpoint / 'assembly_report.json')
    _require(record['model_type'] in EXPECTED_IDS
             and record['base_checkpoint_id'] == EXPECTED_IDS[record['model_type']]
             and audit['status'] == assembly['status'] == 'verified'
             and audit['checkpoint_id'] == assembly['checkpoint_id'] == meta['checkpoint_id']
             and audit['donor_checkpoint_id'] == record['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and audit['base_checkpoint_id'] == record['base_checkpoint_id'],
             f'{directory}: source, donor or deployed identity differs.')
    _require(meta['base_model_path'] == donor_meta['base_model_path']
             == record['architecture']['base_model_path'], 'Actual native HF model differs.')
    _require(audit['assembly_report_sha256'] == sha256(checkpoint / 'assembly_report.json')
             and audit['actual_native_normalized_sha256'] == sha256(donor.parent / 'normalized_cat_runtime_args.json')
             and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared']
             == audit['source_union_save_fresh_tensors'], 'Saved native audit evidence changed.')
    sources = assembly['sources']
    _require(sources['base_checkpoint_id'] == record['base_checkpoint_id']
             and sources['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and sources['categories'] == ['gate_proj'] and set(sources['replaced_targets']) == names,
             'Assembly does not replace exactly the actual complete gate category.')
    budget = audit['bit_budget']
    _require(budget == assembly['bit_budget'] and budget['budget_passed']
             and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['user_linear_bits'] == EXPECTED_BITS[record['model_type']]
             and budget['compressed_target_count'] == record['architecture']['projection_count'],
             'Actual storage budget differs from the approved source union.')
    _require(set(audit['protection_tensor_records']) == set(audit['rotation_sign_records']) == names
             and audit['gate_count'] == len(names), 'Incomplete gate protection/sign inventory.')
    for name in names:
        fields = audit['protection_tensor_records'][name]
        _require(set(fields) == {'protected_input_indices', 'protected_input_qvalues', 'protected_input_scales'}
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
    evidence = [directory / name for name in ('launch.json', 'audit.json', 'full8_summary.json',
        'train.log', 'train.exit', 'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')]
    evidence += [*paths, checkpoint / 'checkpoint_meta.json', checkpoint / 'assembly_report.json',
                 donor / 'checkpoint_meta.json', donor.parent / 'normalized_cat_runtime_args.json']
    return record, audit, result, shards, {str(path): sha256(path) for path in evidence}


def delta(candidate, reference):
    return {'mean_delta_pp': candidate['mean_percent'] - reference['mean_percent'],
        'mmlu_delta_pp': 100 * (candidate['task_metrics']['mmlu'] - reference['task_metrics']['mmlu']),
        'task_delta_pp': {task: 100 * (candidate['task_metrics'][task] - reference['task_metrics'][task])
                          for task in TASK_METRICS},
        'correct_delta': {task: candidate['integer_counts'][task]['correct']
                         - reference['integer_counts'][task]['correct'] for task in TASK_METRICS},
        'mmlu_subject_correct_delta': {task: candidate['integer_counts'][task]['correct']
            - reference['integer_counts'][task]['correct'] for task in candidate['mmlu_subject_metrics']}}


def compare(block32, full, output):
    output = Path(output).resolve()
    _require(not output.exists() and not output.is_symlink(), 'Do not overwrite a comparison.')
    b, ba, bs, br, be = arm(block32, 32)
    f, fa, fs, fr, fe = arm(full, 0)
    for key in ('model_type', 'architecture', 'base_dir', 'base_checkpoint_id', 'control_result_paths',
                'cwd', 'source_sha256', 'input_sha256', 'protected_stats', 'environment', 'protocol'):
        _require(b[key] == f[key], 'Pair source/protocol differs: ' + key)
    _require(normalized_recipe(b) == normalized_recipe(f),
             'Pair native recipe differs beyond output directory and rotation block size.')
    _require(ba['protection_tensor_records'] == fa['protection_tensor_records'],
             'Pair protected indices/qvalues/scales differ; this is not an isolated rotation comparison.')
    _require(ba['rotation_sign_records'] == fa['rotation_sign_records'],
             'Pair persistent rotation signs differ despite the matched seed/geometry.')
    base = Path(b['base_dir']); base_meta = load_v6_meta(str(base))
    _require(base_meta['checkpoint_id'] == b['base_checkpoint_id'], 'Original compressed base identity changed.')
    control, cr = score([Path(p) for p in b['control_result_paths']], b['architecture'], base)
    for label, rows in (('full', fr), ('base', cr)):
        for current, reference in zip(rows, br, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == reference[key], f'{label} benchmark protocol differs: {key}')
    for key in ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits', 'components'):
        _require(ba['bit_budget'][key] == fa['bit_budget'][key], 'Pair storage accounting differs: ' + key)
    report = {'status': 'verified_pair', 'model_type': b['model_type'],
        'primary_comparison': 'full_minus_fresh_block32', 'full_vs_block32': delta(fs, bs),
        'secondary_vs_original_compressed_base': {'block32': delta(bs, control), 'full': delta(fs, control)},
        'arms': {'block32': bs, 'full': fs, 'base': control},
        'pair_checks': {'same_native_recipe_except_output_and_block_size': True,
            'same_base_hf_source_and_calibration': True, 'actual_steps_per_stage': 10000, 'stages': 2,
            'complete_gate_count': ba['gate_count'], 'protection_tensor_records_exact': True,
            'rotation_sign_records_exact': True, 'rotation_note': 'Equal signs do not imply equal transforms.',
            'same_raw_task_protocol': True, 'leaf_count': 64, 'sample_count': 24742,
            'mmlu_subjects': 57, 'mmlu_samples': 14042, 'batch_sizes': {'mmlu': 64, 'seven': 32},
            'num_fewshot': 0, 'limit': None},
        'bit_budget': {key: fa['bit_budget'][key] for key in
                       ('user_linear_bits', 'user_linear_bpw', 'packed_linear_bits', 'components')},
        'evidence_sha256': {**be, **fe, str(base / 'checkpoint_meta.json'): sha256(base / 'checkpoint_meta.json'),
                           **{p: sha256(p) for p in b['control_result_paths']}},
        'comparison_source_sha256': sha256(Path(__file__)),
        'quality_boundary': 'A valid negative comparison is saved successfully; no accuracy pass threshold is imposed.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    write_json(output, report)
    print({'status': report['status'], 'model_type': b['model_type'],
           'full_minus_block32_mean_pp': report['full_vs_block32']['mean_delta_pp'],
           'full_minus_block32_mmlu_correct': report['full_vs_block32']['correct_delta']['mmlu'],
           'output': str(output), 'sha256': sha256(output)}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block32', type=Path, required=True, help='Completed block32 run directory.')
    parser.add_argument('--full', type=Path, required=True, help='Completed full-RHT run directory.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.block32, args.full, args.output)
