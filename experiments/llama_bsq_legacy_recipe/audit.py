"""Reuse native gate audits and compare the four-coefficient BSQ recipe."""
from __future__ import annotations

from pathlib import Path

from experiments.full_rht_gate_pair.audit import check_assembly as native_assembly, check_donor, results as native_results
from experiments.full_rht_gate_pair.common import SHARDS, read
from experiments.full_rht_gate_pair.compare import delta, score
from experiments.longbench_v1.common import sha256
from experiments.qwen3_w25.assemble_categories import _require


def check_assembly(directory, donor, record):
    audit = native_assembly(directory, donor, record)
    historical_path = Path(record['historical_control_audit_path']); historical = read(historical_path)
    _require(historical['status'] == 'verified'
             and historical['checkpoint_id'] == record['historical_control_checkpoint_id']
             == '0ad677ce-a49a-423e-a20c-50f8aa1ef710'
             and audit['base_checkpoint_id'] == historical['base_checkpoint_id']
             == '35776715-2707-45a9-9fb4-2da1efef3f0e'
             and audit['source_union_save_fresh_tensors'] == historical['source_union_save_fresh_tensors'] == 2979
             and audit['gate_count'] == historical['gate_count'] == 32
             and audit['preserved_projection_count'] == historical['preserved_projection_count'] == 192,
             'The matched native gate source/saved inventory changed.')
    for key in ('protection_tensor_records', 'rotation_sign_records'):
        _require(audit[key] == historical[key], 'The BSQ coefficient recipe changed protected payload/signs: ' + key)
    for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components', 'code_padding_bits'):
        _require(audit['bit_budget'][key] == historical['bit_budget'][key], 'The paired storage budget changed: ' + key)
    _require(audit['bit_budget']['user_linear_bits'] == 17100439552
             and audit['bit_budget']['original_linear_weights'] == 6979321856
             and audit['bit_budget']['components']['decoder']['dtypes'] == ['bfloat16'], 'Wrong actual integer budget.')
    audit.update(historical_control_checkpoint_id=historical['checkpoint_id'],
        historical_control_audit_sha256=sha256(historical_path), historical_control_weights_loaded=False,
        protection_and_signs_exact_historical_control=True, same_actual_integer_storage_as_control=True,
        bsq_control=record['bsq_control'], bsq_candidate=record['bsq_candidate'])
    return audit


def results(directory, record, audit):
    directory = Path(directory)
    summary = native_results(directory, record)
    current, raw = score([directory / (name + '.json') for name, _, _ in SHARDS],
                         record['architecture'], directory / 'final_model')
    control, old_raw = score([Path(p) for p in record['historical_control_result_paths']],
        record['architecture'], Path(record['historical_control_dir']) / 'final_model')
    _require(control['checkpoint_id'] == record['historical_control_checkpoint_id'], 'Matched control identity differs.')
    for left, right in zip(raw, old_raw, strict=True):
        for key in ('configs', 'versions', 'n_shot', 'n_samples'):
            _require(left[key] == right[key], 'The paired full8 protocol differs: ' + key)
    summary.update(integer_counts=current['integer_counts'], user_linear_bpw=audit['bit_budget']['user_linear_bpw'],
        legacy_recipe_vs_current_recipe=delta(current, control), historical_control_checkpoint_id=control['checkpoint_id'],
        bsq_recipe_comparison='complete', comparison_boundary=record['comparison_boundary'],
        bsq_control=record['bsq_control'], bsq_candidate=record['bsq_candidate'])
    summary.pop('rotation_pair_comparison', None)
    return summary
