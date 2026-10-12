"""Compare completed fixed-K up full/block32 arms; e767 remains secondary."""
import argparse
from pathlib import Path

from experiments.llama_gate_up_rate_pair.common import BASE_ID, TOTAL_BITS, SHARDS, _require
from experiments.llama_gate_up_rate_pair.compare import completed
from experiments.full_rht_gate_pair.compare import normalized_recipe, score, delta
from experiments.longbench_v1.common import sha256, write_json


def compare(block32, full, output):
    output = Path(output).resolve(); _require(not output.exists(), 'Do not overwrite comparison.')
    b, ba, bs, br, be = completed(block32, 'control', expected_bits={'up_proj': 32})
    f, fa, fs, fr, fe = completed(full, 'candidate', expected_bits={'up_proj': 32})
    _require(b['block_size'] == 32 and f['block_size'] == 0 and b['categories'] == f['categories'] == ['up_proj']
             and b['base_checkpoint_id'] == f['base_checkpoint_id'] == BASE_ID, 'Wrong rotation pair or source.')
    for key in ('cwd', 'base_dir', 'architecture', 'control_result_paths', 'source_sha256', 'input_sha256',
                'protected_stats', 'environment', 'protocol', 'seed_boundary', 'expected_budget_components'):
        _require(b[key] == f[key], 'Pair source/protocol mismatch: ' + key)
    _require(normalized_recipe(b) == normalized_recipe(f), 'Pair differs beyond block size/output.')
    for key in ('protection_tensor_records', 'rotation_sign_records', 'calibration_cache', 'global_adaptive_plan'):
        _require(ba[key] == fa[key], 'Actual paired calibration/protection/signs differ: ' + key)
    for audit in (ba, fa):
        _require(audit['replaced_projection_count'] == 32 and audit['preserved_projection_count'] == 192
                 and audit['source_union_save_fresh_tensors'] == 2979
                 and audit['bit_budget']['user_linear_bits'] == audit['bit_budget']['packed_linear_bits'] == TOTAL_BITS
                 and audit['bit_budget']['components'] == b['expected_budget_components'], 'Actual source union/budget differs.')
    reference, rr = score([Path(p) for p in b['control_result_paths']], b['architecture'], Path(b['base_dir']))
    for rows in (fr, rr):
        for current, control in zip(rows, br, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == control[key], 'Raw complete evaluation protocol differs: ' + key)
    report = {'status': 'verified_pair', 'primary_comparison': 'fresh_full_minus_fresh_block32',
        'full_minus_block32': delta(fs, bs), 'arms': {'block32': bs, 'full': fs, 'e767': reference},
        'secondary_vs_e767': {'block32': delta(bs, reference), 'full': delta(fs, reference)},
        'pair_checks': {'native_recipe_except_block_output': True, 'same_actual512_calibration_inputs': True,
            'same_actual32_protection_indices_values_scales': True, 'same_actual64_rotation_signs': True,
            'calibration_stats_tensor_records_exact': ba['calibration_stats'] == fa['calibration_stats'],
            'full8_leaves': 64, 'samples': 24742,
            'strict_saved_fresh_state_tensors': 2979, 'preserved_projections': 192, 'protected_encoding': 'INT8'},
        'bit_budget': fa['bit_budget'], 'comparison_boundary': b['comparison_boundary'],
        'initialization_boundary': b['seed_boundary']['initialization'], 'evidence_sha256': {**be, **fe},
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'comparison_source_sha256': sha256(Path(__file__))}
    write_json(output, report)
    print({'mean_delta_pp': report['full_minus_block32']['mean_delta_pp'], 'sha256': sha256(output)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block32', type=Path, required=True)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); compare(args.block32, args.full, args.output)
