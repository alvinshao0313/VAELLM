"""Prepare matched official retrieval400 runs for D1200 and its unchanged sparse-patch union."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.longbench_candidate_probe.prepare_llama_best1600_d16 import DENSE, ROOT, R2, TASKS, read, require
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.protocol import generation_options
from experiments.longbench_v1.runner import read_input
from experiments.longbench_v1.scoring import check_prediction, score_example

DECODER = R2 / 'general_lm_llama_d16_combo1600_s2000/step_1200'
COMBO = R2 / 'llama_decoder1200_sparse075_combo'
REFERENCE = R2 / 'llama_best1600_d16_retrieval400'
BASE_ID = 'c74df41b-81d2-4176-a7a0-862b96b0ed6b'
COMBO_ID = '1ef53b97-abe9-48ff-a7f3-8c2e0a9d4fbd'
E767_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
PATCH_ID = '646d95f4-cafe-4eb6-8270-71c4324432d4'
REVIEWS = {BASE_ID: 'ec4308a3c201ed82e465e88dc6f1dc477d0c8f5c82f4de027736f427545233fc',
           COMBO_ID: '4caac497fc7c721cda3fb93b6be322e7dbf6bb06b4975c2b8ff231e4a6bfc456'}


def merge(*groups):
    result = {}
    for group in groups:
        for path, digest in group.items():
            require(path not in result or result[path] == digest, 'Conflicting source/evidence identity: ' + path)
            result[path] = digest
    check_hashes(result)
    return result


def audit_endpoint(directory, uid):
    review, audit, summary, meta = (read(directory / name) for name in
        ('independent_review.json', 'audit.json', 'full8_summary.json', 'final_model/checkpoint_meta.json'))
    require(sha256(directory / 'independent_review.json') == REVIEWS[uid]
            and {x['checkpoint_id'] for x in (review, audit, summary, meta)} == {uid}
            and audit['status'] == 'verified' and summary['leaf_count'] == 64 and summary['sample_count'] == 24742,
            'Require the actual independently reviewed endpoint and complete eight-task evidence.')
    candidate = review['candidate'] if uid == COMBO_ID else review
    require(summary['mean_percent'] == candidate['mean_percent']
            and summary['task_metrics'] == candidate['task_metrics'], 'Reviewed full8 summary differs.')
    budget = audit['bit_budget']
    require(budget['budget_passed'] and not budget['contract_violations']
            and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0
            and budget['user_linear_bits'] == budget['packed_linear_bits']
            and budget['components']['decoder']['dtypes'] == ['bfloat16']
            and not budget['components'].get('residual_lora', {}).get('storage_bits', 0), 'Actual deployment budget differs.')
    if uid == COMBO_ID:
        launch = read(directory / 'launch.json')
        require(review['status'] == 'independent_cpu_verified' and launch['status'] == 'completed'
                and launch['exit_code'] == 0 and audit['base_checkpoint_id'] == BASE_ID
                and audit['patch_checkpoint_id'] == PATCH_ID and audit['residual_source_checkpoint_id'] == E767_ID
                and audit['saved_tensors'] == audit['strict_reloaded_tensors'] == 3171
                and audit['original_frozen_tensors'] == 2979 and audit['added_frozen_sparse_tensors'] == 192
                and audit['source_tensors_exact'] and audit['sparse_payload_exact'] and audit['native_metadata_union_exact']
                and audit['fresh_native_bf16_local_forward']['matches_saved_probe']
                and audit['trainable_parameters'] == 0 and not audit['hf_residual_recomputed']
                and budget['user_linear_bits'] == 17436621312, 'Require the unchanged D1200 plus fixed P_e767 union.')
        stages = ('build', 'verify', 'mmlu', 'seven', 'dispatcher', 'outer')
        extra = ['build_report.json', 'launch.prepared.json', 'launch.json']
        check_hashes(launch['input_sha256'])
        for stat in launch['protected_stats']:
            check_stat(stat)
        native = {'persistent_tensors': 3171, 'original_frozen_tensors': 2979, 'appended_sparse_tensors': 192,
                  'strict_fresh_reload': True, 'native_bf16_local_reload_exact': True, 'full8_group8_completed': True}
    else:
        launch = read(directory.parent / 'launch.prepared.json')
        callback = read(directory / 'step_export.json')
        require(review['status'] == 'verified' and callback['status'] == 'evaluated'
                and review['global_step'] == audit['selected_global_step'] == callback['global_step'] == 1200
                and audit['source_checkpoint_id'] == E767_ID
                and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 2979
                and audit['fresh_strict_reload_state_exact'] and audit['frozen_tensors_compared'] == 1635
                and audit['selected_decoder_tensor_count'] == 1344 and audit['packed_codes_unchanged']
                and audit['changed_code_bits'] == 0 and audit['rank_zero']
                and budget['user_linear_bits'] == 17095180288
                and callback['audit_sha256'] == sha256(directory / 'audit.json')
                and callback['full8_summary_sha256'] == sha256(directory / 'full8_summary.json'),
                'Require the independently exported D1200 deployment, not the historical training checkpoint.')
        stages = ('export', 'audit', 'mmlu', 'seven')
        extra = ['step_export.json', 'final_model/export_record.json']
        native = {'persistent_tensors': 2979, 'decoder_tensors': 1344, 'frozen_tensors': 1635,
                  'strict_fresh_reload': True, 'packed_code_xor': 0, 'full8_group8_completed': True}
    require(all((directory / (s + '.exit')).read_text().strip() == '0' for s in stages), 'A candidate stage failed.')
    paths = [directory / n for n in ['independent_review.json', 'audit.json', 'full8_summary.json',
        'final_model/checkpoint_meta.json', 'final_model/config.json', 'final_model/tokenizer_config.json',
        'mmlu.json', 'seven.json', *extra, *(s + '.exit' for s in stages)]]
    evidence = {str(p): sha256(p) for p in paths}
    for p, expected in review['evidence_sha256'].items():
        if p in evidence:
            require(evidence[p] == expected, 'An already reviewed artifact changed: ' + p)
    source_config = read(directory / 'final_model/config.json')
    source_tokenizer = read(directory / 'final_model/tokenizer_config.json')
    return {'directory': str(directory), 'checkpoint_dir': str(directory / 'final_model'), 'checkpoint_id': uid,
        'full8_mean_percent': summary['mean_percent'], 'native_proof': native, 'bit_budget': budget,
        'source_config_defaults': {k: source_config.get(k) for k in ('pad_token_id', 'eos_token_id', 'use_cache')},
        'source_tokenizer_pad_token': source_tokenizer.get('pad_token'),
        'weight_stat': file_stat(directory / 'final_model' / meta['state_dict_file']),
        'review_path': str(directory / 'independent_review.json')}, launch, evidence


def controls_and_inputs():
    dense_config, dense_scores, dense_review = (read(DENSE / n) for n in
        ('config.json', 'scores.json', 'score_verification.json'))
    reference_config, reference_scores, reference_review = (read(REFERENCE / n) for n in
        ('config.json', 'scores.json', 'review.json'))
    require(dense_review['status'] == reference_review['status'] == 'verified'
            and dense_review['score_exit_code'] == reference_review['execution_exit_code'] == 0
            and dense_review['all_prediction_raw_source_token_decode_generation_checks_passed']
            and reference_review['samples'] == 400 and reference_review['actual_input_tokens_and_original_ids_exact']
            and reference_review['generation_options_effective_and_decoded_tokens_verified']
            and dense_review['scores_sha256'] == sha256(DENSE / 'scores.json')
            and reference_review['scores_sha256'] == sha256(REFERENCE / 'scores.json'), 'Require verified original controls.')
    require(dense_review['complete_tasks'] == 21 and dense_review['sample_count'] == 4750
            and dense_review['longbench_task_macro_unrounded'] == dense_scores['task_macro_mean']
            and abs(dense_review['baseline_nine_mean'] - (8 * dense_review['baseline_eight_mean']
                + dense_scores['task_macro_mean']) / 9) < 1e-12
            and abs(dense_review['required_compressed_nine_minimum']
                - (dense_review['baseline_nine_mean'] - .8)) < 1e-12,
            'Use the verified full21 BF16 baseline and the current0.8-point nine-item target.')
    manifest, index = load_archive(dense_config['archive']['path'])
    tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer']['path'], use_fast=True, local_files_only=True)
    tokens = np.memmap(manifest['inputs']['path'], dtype='<u4', mode='r')
    acceptance = ROOT / 'result/llama31_8b_w25_20261004/baseline_bf16/acceptance.json'
    controls, evidence, generation = {}, {str(acceptance): dense_review['baseline_eight_acceptance_sha256']}, {}
    evidence.update({item['path']: item['sha256'] for item in dense_review['baseline_eight_raw_inputs']})
    for label, directory, config, scores, prediction_root in (
            ('bf16', DENSE, dense_config, dense_scores, DENSE),
            ('e767', REFERENCE, reference_config, reference_scores, REFERENCE / 'run')):
        require(all(config[k] == dense_config[k] for k in ('schema', 'mode', 'protocol', 'archive',
            'package_versions', 'model_policy', 'prewarm_group_size', 'device')), 'Historical controls use different protocols.')
        tasks = {}
        for task in TASKS:
            rows = [row for row in index if row['dataset'] == task]
            path = prediction_root / 'predictions' / (task + '.jsonl')
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows) == len(predictions) == 200 and [r['row_index'] for r in rows] == list(range(200))
                    and sha256(path) == scores['prediction_sha256'][str(path)], 'Require each complete original200 task.')
            values = []
            for pred, row in zip(predictions, rows):
                check_prediction(pred, row, sha256(directory / 'config.json'))
                options = generation_options(tokenizer, task, row['input_tokens'], row['max_new_tokens'], protocol=config['protocol'])
                require(row['max_new_tokens'] == 32 and pred['generation_options'] == pred['effective_generation_options'] == options
                        and options['use_cache'] and pred['use_model_defaults'] is False
                        and tokenizer.decode(pred['output_token_ids'], skip_special_tokens=True) == pred['pred'],
                        'Preserve native explicit generation, stopping and token decoding.')
                if label == 'bf16':
                    read_input(tokens, row)
                values.append(score_example(task, pred['pred'], row['answers'], row['all_classes']))
            actual = math.fsum(values) / 2
            expected = scores['tasks'][task]['score' if label == 'bf16' else 'score_percent']
            require(abs(actual - expected) < 1e-10, 'Official fractional control score differs.')
            tasks[task] = {'score_percent': actual, 'samples': 200}
            evidence[str(path)] = sha256(path)
            generation[task] = {'input_tokens_min': min(r['input_tokens'] for r in rows),
                'input_tokens_max': max(r['input_tokens'] for r in rows), 'max_new_tokens': 32,
                'original_input_slices_sha_verified': True, 'generation_options': options}
        controls[label] = {'directory': str(directory), 'prediction_root': str(prediction_root), 'tasks': tasks}
        for name in ('config.json', 'scores.json', 'score_verification.json' if label == 'bf16' else 'review.json'):
            path = directory / name; evidence[str(path)] = sha256(path)
    del tokens
    execution = read(reference_review['execution_report'])
    require(execution['status'] == 'partition_complete' and execution['exit_code'] == 0
            and execution['load_audit']['persistent_tensors'] == 2979
            and execution['load_audit']['model_policy'] == dense_config['model_policy']
            and reference_review['duration_seconds'] == execution['duration_seconds'], 'Reuse actual native535.8s retrieval evidence.')
    evidence[reference_review['execution_report']] = sha256(reference_review['execution_report'])
    return dense_config, controls, evidence, generation, reference_review, dense_review


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Use CUDA-hidden bitvae in the public source root.')
    torch.set_num_threads(4)
    outputs = {BASE_ID: R2 / 'llama_decoder1200_retrieval400', COMBO_ID: R2 / 'llama_decoder1200_sparse075_retrieval400'}
    require(all(not p.exists() for p in outputs.values()), 'Both preparations must use fresh independent outputs.')
    candidates = [audit_endpoint(DECODER, BASE_ID), audit_endpoint(COMBO, COMBO_ID)]
    dense, controls, common_evidence, geometry, reference, bf16 = controls_and_inputs()
    helper = ROOT / 'experiments/longbench_candidate_probe/prepare_llama_best1600_d16.py'
    prepared = []
    for candidate, launch, evidence in candidates:
        uid = candidate['checkpoint_id']; output = outputs[uid]
        config = make_config(dense['archive']['path'], kind='compressed', hf_path=dense['model']['hf_path'],
                             checkpoint=candidate['checkpoint_dir'], mode='full')
        require(all(config[k] == dense[k] for k in ('schema', 'mode', 'protocol', 'archive', 'package_versions',
                'model_policy', 'prewarm_group_size', 'device')) and config['model']['checkpoint_id'] == uid
                and config['model']['weight_stat'] == candidate['weight_stat'], 'Candidate identity or official protocol changed.')
        check_config(config)
        sources = merge(launch['source_sha256'], dense['source_sha256'], config['source_sha256'],
                        {str(Path(__file__).resolve()): sha256(__file__), str(helper): sha256(helper)})
        evidence = merge(evidence, common_evidence)
        required_lb = 9 * bf16['required_compressed_nine_minimum'] - 8 * candidate['full8_mean_percent']
        config_path = output / 'config.json'
        record = {'schema': 'llama_decoder1200_matched_retrieval400_v1', 'status': 'prepared_unbound', 'gpu': None,
            'source_root': str(ROOT), 'python_executable': sys.executable, 'checkpoint_id': uid,
            'config_path': str(config_path), 'output_dir': str(output / 'run'), 'datasets': list(TASKS),
            'sample_count': 400, 'task_samples': {t: 200 for t in TASKS}, 'source_sha256': sources,
            'evidence_sha256': evidence, 'source_bit_budget': candidate['bit_budget'], 'completed_candidate': candidate,
            'controls': deepcopy(controls), 'matched_pair': {key: str(value) for key, value in outputs.items()},
            'environment': {k: v for k, v in launch['environment'].items() if k != 'CUDA_VISIBLE_DEVICES'},
            'launcher_argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config', str(config_path),
                '--output', str(output / 'run'), '--datasets', *TASKS],
            'timeout_seconds': 900, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False, 'resume': False,
            'resource_evidence': {'reference_duration_seconds': reference['duration_seconds'],
                'reference_generation_peak_allocated_bytes': reference['generation_peak_allocated_bytes'],
                'reference_generation_peak_reserved_bytes': reference['generation_peak_reserved_bytes'],
                'wall_seconds_estimate': [500, 750], 'outer_cap_seconds': 900,
                'boundary': 'Estimate from same original Llama SDPA/group8/400 path; sparse adds42.68MB payload and native sparse matmul. '
                            'No additional GPU smoke test; parent must allocate a sufficient900+20s owner window.'},
            'input_geometry': geometry,
            'selection': 'All archived original rows0..199 of each official task, identical exact tokens; no filtering/resampling.',
            'scoring': 'Native check_prediction/score_example fractional score, taskwise200 and two-task macro; do not call full21 scorer.',
            'generation_default_boundary': 'Checkpoint pad_token_id/use_cache defaults differ and are recorded. Existing runner uses '
                'original HF tokenizer, explicit GenerationConfig including pad/EOS/use_cache, use_model_defaults=False and all-one input mask.',
            'validation_reuse': 'Existing complete native save/fresh/full8 audits of each actual endpoint; original e767535.8s native retrieval400. '
                'The ordinary LongBench loader still performs fresh strict loading, actual storage audit and group8 prewarm at execution.',
            'comparison_boundary': 'Compare D1200+fixed P_e767 against D1200 on the same400 inputs. Compare D1200 against e767 for recovery. '
                'Before both complete, do not attribute a combo change to sparse; no HF-minus-D1200 residual was fitted.',
            'nine_item_necessary_bound': {'bf16_nine_mean': bf16['baseline_nine_mean'],
                'minimum_nine_mean': bf16['required_compressed_nine_minimum'], 'candidate_eight_mean': candidate['full8_mean_percent'],
                'required_longbench21_mean': required_lb,
                'post_retrieval_upper_bound_formula': 'LB21_max=(1900+retrieval_en+retrieval_zh)/21; nine_max=(8*eight_mean+LB21_max)/9.',
                'interpretation': 'Only a necessary ceiling for this exact endpoint; no full21 or nine-item acceptance from two tasks.'},
            'scope': 'Two of21 official LongBench tasks only.', 'full_longbench_status': 'not_evaluated',
            'nine_item_goal_status': 'not_evaluated', 'model_mutations': False, 'model_exported': False,
            'cuda_initialized': torch.cuda.is_initialized()}
        require(not record['cuda_initialized'], 'CPU preparation initialized CUDA.')
        prepared.append((output, config, record))
    for output, config, record in prepared:
        output.mkdir(parents=True); write_json(output / 'config.json', config)
        record['config_sha256'] = sha256(output / 'config.json'); write_json(output / 'preparation.json', record)
        print(json.dumps({'status': record['status'], 'checkpoint_id': record['checkpoint_id'], 'output': str(output),
            'preparation_sha256': sha256(output / 'preparation.json'), 'config_sha256': record['config_sha256'],
            'source_count': len(record['source_sha256']), 'evidence_count': len(record['evidence_sha256']),
            'required_longbench21_mean': record['nine_item_necessary_bound']['required_longbench21_mean'],
            'cuda_initialized': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    prepare()
