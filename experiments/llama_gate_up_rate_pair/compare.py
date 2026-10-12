"""Compare two completed arms using actual payload identities and all64 leaves."""
import argparse
from pathlib import Path

from experiments.llama_gate_up_rate_pair.common import BITS, SHARDS, check, read, paired_recipe, _require
from experiments.llama_gate_up_rate_pair.audit import donor
from experiments.full_rht_gate_pair.compare import score, delta
from experiments.longbench_v1.common import sha256, write_json


def completed(directory, arm, *, expected_bits=None):
    directory = Path(directory).resolve(); record = read(directory / 'launch.json'); check(record)
    bits = BITS[arm] if expected_bits is None else expected_bits
    _require(record['arm'] == arm and record['codebook_bits'] == bits and record['status'] == 'completed'
             and record['exit_code'] == 0 and Path(record['output_dir']) == directory, 'Require completed matched arm.')
    for name in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'):
        _require(int((directory / (name + '.exit')).read_text()) == 0, 'Arm did not naturally complete: ' + name)
    native = Path(record['donor_checkpoint_dir']); dm, _ = donor(native, record)
    audit = read(directory / 'audit.json'); summary = read(directory / 'full8_summary.json')
    final = directory / 'final_model'; assembly = read(final / 'assembly_report.json')
    result, raw = score([directory / (n + '.json') for n, _, _ in SHARDS], record['architecture'], final)
    _require(audit['status'] == assembly['status'] == 'verified'
             and result['checkpoint_id'] == audit['checkpoint_id'] == assembly['checkpoint_id'] == record['checkpoint_id']
             and audit['donor_checkpoint_id'] == dm['checkpoint_id'] == record['donor_checkpoint_id']
             and audit['assembly_report_sha256'] == sha256(final / 'assembly_report.json')
             and audit['actual_native_normalized_sha256'] == sha256(native.parent / 'normalized_cat_runtime_args.json')
             and audit['cat_runtime_state_sha256'] == sha256(native / 'cat_runtime_state.pt')
             and summary['audit_sha256'] == sha256(directory / 'audit.json')
             and record['full8_summary_sha256'] == sha256(directory / 'full8_summary.json')
             and summary['mean_percent'] == result['mean_percent'] == record['mean_percent']
             and summary['integer_counts'] == result['integer_counts'], 'Completed evidence changed.')
    evidence = [directory / n for n in ('launch.json', 'audit.json', 'full8_summary.json', 'train.log',
        'train.exit', 'assembly.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit', 'mmlu.json', 'seven.json')]
    evidence += [final / 'checkpoint_meta.json', final / 'assembly_report.json', native / 'checkpoint_meta.json',
                 native.parent / 'normalized_cat_runtime_args.json', native / 'cat_runtime_state.pt']
    return record, audit, result, raw, {str(p): sha256(p) for p in evidence}


def compare(control, candidate, output):
    output = Path(output).resolve(); _require(not output.exists(), 'Do not overwrite comparison.')
    c, ca, cs, cr, ce = completed(control, 'control'); t, ta, ts, tr, te = completed(candidate, 'candidate')
    for key in ('cwd', 'base_dir', 'base_checkpoint_id', 'architecture', 'control_result_paths', 'categories',
                'source_sha256', 'input_sha256', 'protected_stats', 'environment', 'protocol', 'seed_boundary'):
        _require(c[key] == t[key], 'Pair source/protocol mismatch: ' + key)
    _require(paired_recipe(c) == paired_recipe(t), 'Pair recipe differs beyond K allocation and output.')
    for key in ('protection_tensor_records', 'rotation_sign_records', 'calibration_cache', 'global_adaptive_plan'):
        _require(ca[key] == ta[key], 'Pair actual protected payload/sign/calibration differs: ' + key)
    for key in ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits', 'code_padding_bits', 'components'):
        _require(ca['bit_budget'][key] == ta['bit_budget'][key], 'Pair actual storage differs: ' + key)
    _require(ca['source_union_save_fresh_tensors'] == ta['source_union_save_fresh_tensors'] == 2979,
             'Pair strict union inventory differs.')
    for left, right in zip(cr, tr, strict=True):
        for key in ('configs', 'versions', 'n_shot', 'n_samples'):
            _require(left[key] == right[key], 'Pair raw protocol differs: ' + key)
    reference, _ = score([Path(p) for p in c['control_result_paths']], c['architecture'], Path(c['base_dir']))
    report = {'status': 'verified_pair', 'primary_comparison': 'fresh_K40_24_minus_fresh_K32_32',
        'candidate_minus_control': delta(ts, cs), 'arms': {'control': cs, 'candidate': ts, 'e767': reference},
        'secondary_vs_e767': {'control': delta(cs, reference), 'candidate': delta(ts, reference)},
        'pair_checks': {'native_recipe_except_K_output': True, 'same_actual512_calibration_inputs': True,
            'same_actual64_protection_indices_values_scales': True, 'same_actual128_rotation_signs': True,
            'calibration_stats_tensor_records_exact': ca['calibration_stats'] == ta['calibration_stats'],
            'full8_leaves': 64, 'samples': 24742, 'strict_saved_fresh_state_tensors': 2979,
            'preserved_projections': 160, 'protected_encoding': 'INT8'},
        'bit_budget': ta['bit_budget'], 'comparison_boundary': c['comparison_boundary'],
        'initialization_boundary': c['seed_boundary']['initialization'], 'evidence_sha256': {**ce, **te},
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'comparison_source_sha256': sha256(Path(__file__))}
    write_json(output, report)
    print({'mean_delta_pp': report['candidate_minus_control']['mean_delta_pp'], 'sha256': sha256(output)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); compare(args.control, args.candidate, args.output)
