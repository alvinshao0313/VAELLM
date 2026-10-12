"""CPU comparison of the two completed, fixed-plan target-source arms."""
import argparse
import os
from pathlib import Path

import torch

from experiments.qwen_attention_decoded_target_pair.common import (
    ARMS, CATEGORIES, NAMES, read, check, paired_recipe, _require)
from experiments.qwen_attention_int4_pair.common import summarize
from experiments.longbench_v1.common import write_json, sha256


def compare(directory):
    directory = Path(directory).resolve(); output = directory / 'comparison.json'
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(), 'Fresh CPU-only comparison.')
    arms, evidence = {}, {}
    for arm in ARMS:
        folder = directory / arm; record = read(folder / 'launch.json'); check(record)
        _require(record['arm'] == arm and record['status'] == 'completed' and record['exit_code'] == 0,
                 'Require two actually completed arms.')
        for stage in ('train', 'endpoint', 'simulation', 'run', 'outer'):
            _require((folder / (stage + '.exit')).read_text().strip() == '0', 'Nonzero actual exit: ' + arm + '/' + stage)
        for name, digest in record['evidence_sha256'].items():
            _require(sha256(folder / name) == digest, 'Completed evidence changed.')
        training = read(folder / 'training.json'); endpoint = read(folder / 'endpoint_audit.json')
        report = read(folder / 'report.json'); summary = summarize(folder, record)
        _require(endpoint['status'] == 'verified_isolated_intermediate'
                 and endpoint['native_group8_before_after_exact'] and endpoint['fresh_union_tensors_compared'] == 3423
                 and endpoint['serialized_attention_tensors'] == 1872 and endpoint['frozen_complement_tensors'] == 1551
                 and report['status'] == 'diagnostic_complete'
                 and summary['mean_percent'] == report['mean_percent']
                 and summary['task_metrics'] == report['task_metrics'], 'Endpoint or raw full-eight evidence differs.')
        for category in CATEGORIES:
            _require(training['categories'][category]['steps_per_stage'] == 10000
                     and training['categories'][category]['stages'] == 2, 'Incomplete native schedule.')
        arms[arm] = (record, training, endpoint, report, summary)
        for name in ('launch.prepared.json', 'launch.json', 'training.json', 'endpoint_audit.json', 'report.json',
                     'lm_eval_results_mmlu.json', 'lm_eval_results_seven.json'):
            evidence[str(folder / name)] = sha256(folder / name)
    left, right = arms['hf'], arms['decoded']
    _require(left[0]['source_sha256'] == right[0]['source_sha256']
             and left[0]['input_sha256'] == right[0]['input_sha256']
             and left[0]['source_plan'] == right[0]['source_plan']
             and paired_recipe(left[0]) == paired_recipe(right[0])
             and left[1]['post_materialization_seed31_rng'] == right[1]['post_materialization_seed31_rng'],
             'Target comparison is not source/recipe/plan/reset matched.')
    for category in CATEGORIES:
        for field in ('entry_rng', 'exit_rng', 'shuffle_seeds'):
            _require(left[1]['categories'][category][field] == right[1]['categories'][category][field],
                     'Category random consumption differs: ' + category + '/' + field)
    prefixes = tuple(name + '.' for name in NAMES)
    for name in NAMES:
        _require(left[2]['protection_tensor_records'][name]['protected_input_indices']
                 == right[2]['protection_tensor_records'][name]['protected_input_indices']
                 and left[2]['rotation_sign_records'][name] == right[2]['rotation_sign_records'][name],
                 'Fixed indices/signs differ across targets.')
    frozen = [{k: v for k, v in arm[2]['source_union_tensor_sha256'].items() if not k.startswith(prefixes)}
              for arm in (left, right)]
    _require(frozen[0] == frozen[1] and len(frozen[0]) == 1551, 'Source complement differs across arms.')
    # Check native storage fields; unique endpoint checkpoint IDs are intentionally different.
    for field in ('user_linear_bits', 'original_linear_weights', 'components', 'contract_violations', 'code_padding_bits'):
        _require(left[2]['actual_native_union_budget'][field] == right[2]['actual_native_union_budget'][field],
                 'Native geometry/storage differs across targets.')
    for field in ('candidate_projected_linear_bits', 'expanded_simulation_linear_bits', 'original_linear_weights'):
        _require(left[3]['payload']['accounting'][field] == right[3]['payload']['accounting'][field],
                 'Actual INT4 geometry/storage differs across targets.')
    delta = right[4]['mean_percent'] - left[4]['mean_percent']
    result = {'status': 'verified_target_source_pair', 'evidence_sha256': evidence,
        'same_fixed_indices_signs_recipe_and_rng': True, 'frozen_complement_tensors': 1551,
        'protected_payload_values_required_equal': False,
        'arms': {arm: {'mean_percent': value[4]['mean_percent'], 'task_metrics': value[4]['task_metrics'],
            'delta_vs_native_f927_pp': value[4]['delta_vs_native_f927_pp'],
            'actual_storage': value[3]['payload']['accounting']} for arm, value in arms.items()},
        'decoded_minus_new_hf_pp': delta,
        'task_delta_pp': {key: 100 * (right[4]['task_metrics'][key] - left[4]['task_metrics'][key])
                          for key in left[4]['task_metrics']},
        'int4_decision_boundary': right[0]['int4_decision_boundary'],
        'historical_evidence': right[0]['historical_evidence'],
        'native_int4_checkpoint': False, 'goal_acceptance': False,
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    _require(not torch.cuda.is_initialized(), 'CPU comparison initialized CUDA.')
    write_json(output, result); print({'status': result['status'], 'decoded_minus_new_hf_pp': delta,
                                     'comparison_sha256': sha256(output)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory', type=Path, required=True)
    compare(parser.parse_args().directory)
