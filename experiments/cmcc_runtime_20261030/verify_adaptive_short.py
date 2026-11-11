"""Verify two naturally completed real short branches before formal execution."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest)


def branch(report_path, arm):
    report_path = Path(report_path).resolve()
    report = read(report_path)
    require(report['status'] == 'completed_short_whole_class_pipeline' and report['exit_code'] == 0
        and report['phase'] == 'short' and report['arm'] == arm,
        'Require this arm\'s actually completed short pipeline: ' + arm)
    require(report['actual_recovery_steps_per_category'] == 2 and report['planned_recovery_steps_per_category'] == 500
        and report['actual_native_fit_updates_per_stage'] == 2 and report['planned_native_fit_updates_per_stage'] == 10000,
        'Short validation must preserve the original formal schedules and real short exposure.')
    categories = ['gate_proj', 'up_proj', 'down_proj']
    require([row['category'] for row in report['stages']] == categories, 'Require all three actual recovery categories.')
    fits, strict_bindings = {}, []
    for index, row in enumerate(report['stages']):
        recovery = row['recovery']
        require(recovery['actual_steps'] == 2 and recovery['schedule_max_steps'] == 500
            and recovery['frozen_persistent_fields_exact'] and recovery['updates'][-1]['step'] == 2
            and recovery['updates'][-1]['changed_master_elements'] > 0,
            'Require two real updates and unchanged frozen fields: ' + row['category'])
        audits = recovery['save_audits']
        require(audits and audits[0]['actual_step'] == 1
            and all(item['master_and_Adam_objects_values_unchanged'] and item['scheduler_and_RNG_unchanged']
                and item['persisted_FP32_masters_exact'] and item['original_HF_optimizer_scheduler_RNG_trainer_state_saved']
                for item in audits), 'Actual save/return-to-training evidence is incomplete.')
        merge = recovery['merge']
        expected_native, expected_dense = 144 + index * 36, 72 - index * 36
        require(merge['actual_dense_merges'] == expected_dense and merge['BF16_dense_merge_rule_exact']
            and merge['native_decoder_finalized'] == expected_native
            and [len(merge['inventories'][key]) for key in ('native', 'compact', 'pending_dense')]
                == [expected_native, 36, expected_dense], 'Actual merged category inventory differs.')
        strict_path = Path(row['strict_reload_report'])
        strict = read(strict_path)
        require(strict['status'] == 'strict_full_state_and_BF16_logits_verified' and strict['exit_code'] == 0
            and strict['same_input_logits_exact'] and strict['checkpoint_id'] == row['after_recovery_checkpoint_id']
            and strict['deployment_digest'] == row['deployment_digest']
            and strict['effective_weight_files'] == [str(Path(row['deployment_path']) / 'pytorch_model.bin')],
            'Actual independent strict/BF16 same-input deployment evidence differs.')
        strict_bindings.append(binding(strict_path))
        if index < 2:
            fit = row['next_live_category_fit']
            category = categories[index + 1]
            require(fit['category'] == category and fit['actual_stages'] == 2
                and fit['actual_updates_per_stage'] == 2 and fit['original_M36_pooled_BSQ']
                and fit['vector_batch_per_layer'] == 8192 and fit['gate_fields_exact']
                and fit['fresh_VAE_initialization_seed'] == 31
                and fit['shuffle_seed'] == (500031 if category == 'up_proj' else 600031),
                'Keep the actual original live next-category fitting contract.')
            rows = fit['actual_optimizer_initializations']
            require(len(rows) == 2 and [item['actual_stage_index'] for item in rows] == [0, 1]
                and all(item['seed'] == 31 and item['parameter_elements'] > 0 for item in rows),
                'Both actual original VAE optimizer initializations must be observed.')
            fits[category] = rows
    meta = read(Path(report['final_model']) / 'adaptive_model_meta.json')
    require(meta['checkpoint_id'] == report['final_checkpoint_id']
        and len(meta['native_modules']) == 216 and len(meta['compact_modules']) == 36
        and not meta['pending_dense_modules'] and report['final_budget']['budget_passed']
        and report['final_budget']['final_budget_applicable']
        and report['final_strict_reload_report'] == report['stages'][-1]['strict_reload_report'],
        'Final real216 native+36 compact+0 dense budget/identity is incomplete.')
    return report, fits, dict(report=binding(report_path), checkpoint_id=meta['checkpoint_id'],
        final_model=report['final_model'], final_budget=report['final_budget'], strict_reload_reports=strict_bindings)


def verify(dynamic_report, rice_report):
    dynamic, dynamic_fits, dynamic_binding = branch(dynamic_report, 'dynamic')
    rice, rice_fits, rice_binding = branch(rice_report, 'rice')
    require(dynamic['source_raw_manifest_sha256'] == rice['source_raw_manifest_sha256']
        and dynamic['source_after_o'] == rice['source_after_o'], 'Comparison arms must use the same actual raw/live parent.')
    fields = ('actual_stage_index', 'initial_parameter_digest', 'parameter_elements', 'parameter_dtypes',
              'parameter_shapes', 'initial_learning_rates', 'seed')
    pairs = {}
    for category in ('up_proj', 'down_proj'):
        rows = []
        for left, right in zip(dynamic_fits[category], rice_fits[category], strict=True):
            require(all(left[key] == right[key] for key in fields),
                'Actual paired VAE initialization differs: ' + category + ' stage' + str(left['actual_stage_index']))
            rows.append({key: left[key] for key in fields if key != 'parameter_shapes'})
        pairs[category] = rows
    return dict(status='verified_actual_short_pair', exit_code=0, branches={'dynamic': dynamic_binding, 'rice': rice_binding},
        actual_up_down_two_stage_initializations_exact=pairs,
        boundary='Real short pipeline/state/strict BF16 finite-output and common VAE initialization evidence only; no quality or exact-resume claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dynamic-report', type=Path, required=True)
    parser.add_argument('--rice-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Never overwrite pair-verification evidence.')
    result = verify(args.dynamic_report, args.rice_report)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], exit_code=0)), flush=True)


if __name__ == '__main__':
    main()
