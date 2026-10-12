"""CPU comparison of the two completed D16 arms and the retained D32 evidence."""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import SHARDS, check, read
from experiments.full_rht_gate_pair.compare import delta, normalized_recipe, score
from experiments.llama_down_dimension.audit import check_donor
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.checkpoint_v6 import load_v6_meta


def arm(directory, block):
    directory = Path(directory).resolve(); record = read(directory / 'launch.json')
    _require(record['status'] == 'completed' and record['exit_code'] == 0
             and record['model_type'] == 'llama' and record['dimension'] == 16
             and record['block_size'] == block and record['full_evaluation'] is True
             and Path(record['output_dir']).resolve() == directory,
             'Require a completed native Llama D16 arm of the requested block size.')
    for stage in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'):
        _require(int((directory / (stage + '.exit')).read_text()) == 0, 'Incomplete stage: ' + stage)
        if stage in record['stages']:
            _require(record['stages'][stage]['exit_code'] == 0, 'Failed recorded stage: ' + stage)
    check(record)
    donor = Path(record['donor_checkpoint_dir']); donor_meta, names = check_donor(donor, record)
    checkpoint = directory / 'final_model'; meta = load_v6_meta(str(checkpoint))
    audit = read(directory / 'audit.json'); assembly = read(checkpoint / 'assembly_report.json')
    historical_path = Path(record['historical_control_audit_path']); historical = read(historical_path)
    _require(audit['status'] == assembly['status'] == historical['status'] == 'verified'
             and audit['checkpoint_id'] == assembly['checkpoint_id'] == meta['checkpoint_id'] == record['checkpoint_id']
             and audit['donor_checkpoint_id'] == donor_meta['checkpoint_id'] == record['donor_checkpoint_id']
             and audit['base_checkpoint_id'] == record['base_checkpoint_id'] == historical['base_checkpoint_id']
             and meta['base_model_path'] == donor_meta['base_model_path'] == record['architecture']['base_model_path'],
             'Native model, donor, original HF or historical source identity differs.')
    _require(audit['assembly_report_sha256'] == sha256(checkpoint / 'assembly_report.json')
             and audit['actual_native_normalized_sha256'] == sha256(donor.parent / 'normalized_cat_runtime_args.json')
             and audit['historical_control_audit_sha256'] == sha256(historical_path)
             and audit['historical_control_checkpoint_id'] == record['historical_control_checkpoint_id']
             == historical['checkpoint_id']
             and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared']
             == audit['source_union_save_fresh_tensors'] == 2979,
             'Native source-union/save/reload or historical evidence changed.')
    sources = assembly['sources']
    _require(sources['base_checkpoint_id'] == record['base_checkpoint_id']
             and sources['donor_checkpoint_id'] == donor_meta['checkpoint_id']
             and sources['categories'] == ['down_proj'] and set(sources['replaced_targets']) == names
             and audit['down_count'] == audit['replaced_projection_count'] == len(names) == 32
             and audit['preserved_projection_count'] == 192,
             'The assembly must replace exactly the 32 down projections.')
    budget = audit['bit_budget']
    _require(budget == assembly['bit_budget'] and budget['budget_passed']
             and budget['user_linear_bits'] == record['expected_user_linear_bits'] == 17095180288
             and budget['original_linear_weights'] == 6979321856
             and 2 * budget['user_linear_bits'] < 5 * budget['original_linear_weights']
             and not budget['contract_violations'] and not budget['unclassified_target_keys']
             and budget['compressed_target_count'] == 224 and budget['code_padding_bits'] == 0
             and budget['components']['decoder']['dtypes'] == ['bfloat16'], 'Actual D16 storage differs.')
    _require(set(budget['components']) == set(historical['bit_budget']['components']), 'Component inventory changed.')
    for key, value in budget['components'].items():
        _require(value['storage_bits'] == historical['bit_budget']['components'][key]['storage_bits']
                 - (5259264 if key == 'decoder' else 0), 'Unexpected storage change: ' + key)
    _require(audit['protection_and_signs_exact_historical_control']
             and audit['historical_control_weights_loaded'] is False
             and audit['protection_tensor_records'] == historical['protection_tensor_records']
             and audit['rotation_sign_records'] == historical['rotation_sign_records']
             and set(audit['protection_tensor_records']) == set(audit['rotation_sign_records']) == names,
             'Protection or rotation signs differ from the historical D32 control.')
    paths = [directory / (name + '.json') for name, _, _ in SHARDS]
    result, raw = score(paths, record['architecture'], checkpoint)
    summary = read(directory / 'full8_summary.json')
    _require(sha256(directory / 'full8_summary.json') == record['full8_summary_sha256']
             and summary['checkpoint_id'] == result['checkpoint_id']
             and summary['task_metrics'] == result['task_metrics']
             and summary['integer_counts'] == result['integer_counts']
             and summary['mean_percent'] == record['mean_percent'] == result['mean_percent']
             and summary['mmlu_correct'] == record['mmlu_correct'] == result['integer_counts']['mmlu']['correct'],
             'Stored complete summary differs from raw evaluation.')
    evidence = [directory / name for name in ('launch.json', 'audit.json', 'full8_summary.json', 'train.log',
        'train.exit', 'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')]
    evidence += [*paths, checkpoint / 'checkpoint_meta.json', checkpoint / 'assembly_report.json',
                 donor / 'checkpoint_meta.json', donor.parent / 'normalized_cat_runtime_args.json']
    return record, audit, result, raw, {str(path): sha256(path) for path in evidence}


def compare(block32, block64, output):
    output = Path(output).resolve()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Use CPU only.')
    _require(not output.exists() and not output.is_symlink(), 'Do not overwrite an existing comparison.')
    b, ba, bs, br, be = arm(block32, 32); f, fa, fs, fr, fe = arm(block64, 64)
    for key in ('model_type', 'dimension', 'categories', 'codebook_bits', 'architecture', 'base_dir',
                'base_checkpoint_id', 'control_result_paths', 'historical_control_dir',
                'historical_control_checkpoint_id', 'historical_control_result_paths',
                'historical_control_audit_path', 'source_sha256', 'input_sha256', 'protected_stats',
                'environment', 'protocol', 'comparison_boundary'):
        _require(b[key] == f[key], 'Pair identity/protocol differs: ' + key)
    _require(normalized_recipe(b) == normalized_recipe(f), 'Pair recipe differs beyond output/block size.')
    for key in ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits', 'components', 'code_padding_bits'):
        _require(ba['bit_budget'][key] == fa['bit_budget'][key], 'Pair actual integer storage differs: ' + key)
    _require(ba['protection_tensor_records'] == fa['protection_tensor_records']
             and ba['rotation_sign_records'] == fa['rotation_sign_records'], 'Pair protection/signs differ.')
    historical_dir = Path(b['historical_control_dir']); base = Path(b['base_dir'])
    hs, hr = score([Path(p) for p in b['historical_control_result_paths']], b['architecture'], historical_dir / 'final_model')
    oscore, oraw = score([Path(p) for p in b['control_result_paths']], b['architecture'], base)
    _require(hs['checkpoint_id'] == b['historical_control_checkpoint_id']
             and oscore['checkpoint_id'] == b['base_checkpoint_id'], 'Reference checkpoint identity changed.')
    for label, rows in (('block64', fr), ('historical_d32_block32', hr), ('original3577', oraw)):
        for current, reference in zip(rows, br, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == reference[key], label + ' raw protocol differs: ' + key)
    primary = {name: delta(value, hs) for name, value in (('block32', bs), ('block64', fs))}
    for name, value in primary.items():
        _require(read(Path(b['output_dir'] if name == 'block32' else f['output_dir']) / 'full8_summary.json')
                 ['primary_vs_historical_d32_block32'] == value, 'Historical comparison summary changed.')
    reference_paths = [*map(Path, b['historical_control_result_paths']), *map(Path, b['control_result_paths']),
        Path(b['historical_control_audit_path']), historical_dir / 'final_model/checkpoint_meta.json',
        base / 'checkpoint_meta.json']
    sources = [Path(__file__), Path(inspect.getfile(score)), Path(inspect.getfile(check_donor))]
    report = {'status': 'verified_pair', 'model_type': 'llama', 'categories': ['down_proj'],
        'primary_vs_historical_d32_block32': primary, 'block64_vs_block32': delta(fs, bs),
        'secondary_vs_original_compressed_base': {'block32': delta(bs, oscore), 'block64': delta(fs, oscore)},
        'arms': {'block32': bs, 'block64': fs, 'historical_d32_block32': hs, 'base': oscore},
        'bit_budget': ba['bit_budget'], 'historical_bit_budget': read(b['historical_control_audit_path'])['bit_budget'],
        'pair_checks': {'same_recipe_except_output_and_block_size': True, 'protection_and_signs_exact': True,
            'steps_per_stage': 10000, 'stages': 2, 'historical_weights_loaded': False,
            'saved_and_reloaded_tensors': 2979, 'replaced_projections': 32, 'preserved_projections': 192,
            'same_raw_protocol': True, 'leaf_count': 64, 'sample_count': 24742, 'mmlu_subjects': 57,
            'mmlu_samples': 14042, 'actual_batch_sizes': {'mmlu': 64, 'seven': 32}, 'limit': None},
        'comparison_boundary': b['comparison_boundary'],
        'quality_boundary': 'Negative quality results are valid; no MMLU/eight-task acceptance threshold.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'evidence_sha256': {**be, **fe, **{str(p): sha256(p) for p in reference_paths}},
        'comparison_source_sha256': {str(p): sha256(p) for p in sources}}
    _require(not torch.cuda.is_initialized(), 'CPU comparison initialized CUDA.')
    write_json(output, report)
    print({'status': report['status'], 'output': str(output), 'sha256': sha256(output),
           'block64_vs_block32': report['block64_vs_block32'], 'primary': primary}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block32', type=Path, required=True)
    parser.add_argument('--block64', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.block32, args.block64, args.output)
