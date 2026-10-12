"""Prepare all21 official tasks for the predetermined Llama QK-only step2000."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
S = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
TRAIN = R2 / 'general_lm_llama_fullqk_qkonly_s2000'
STEP = TRAIN / 'step_2000'
OUTPUT = R2 / 'llama_fullqk_qkonly2000_longbench21'
DENSE = PUBLIC / 'result/llama31_8b_longbench_20261004/dense'
ORIGINAL = R2 / 'llama_qk_full_retrieval400'
E767 = R2 / 'llama_best1600_d16_retrieval400'
ENDPOINT_ID = '131205df-7993-4f54-adf1-8903e0488aa7'
SOURCE_ID = '4e1cec35-be20-42d5-91e3-dcaebc1d0552'
RETRIEVAL = ('passage_retrieval_en', 'passage_retrieval_zh')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare():
    require(Path.cwd() == S and Path(__file__).resolve().is_relative_to(S)
            and os.environ.get('PYTHONPATH') == str(S) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'New CPU-only bitvae/S preparation required.')
    review = read(STEP / 'independent_review.json'); audit = read(STEP / 'audit.json')
    require(sha256(STEP / 'independent_review.json') ==
            '08e2be8b7b320e22f4f6ce40b54385a70a234154f957bdd3d492f5a9475363b8'
            and review['status'] == 'verified_independent_cpu_review' and review['global_step'] == 2000
            and review['checkpoint_id'] == ENDPOINT_ID and review['source_checkpoint_id'] == SOURCE_ID
            and review['whole_training']['status'] == 'completed'
            and review['whole_training']['completed_steps'] == [400, 800, 1200, 1600, 2000]
            and set(review['whole_training']['exit_codes'].values()) == {0}
            and audit['status'] == 'verified' and audit['checkpoint_id'] == ENDPOINT_ID
            and audit['selected_decoder_tensor_count'] == 384 and audit['frozen_tensors_compared'] == 2595
            and audit['code_bank_count'] == 448 and audit['changed_code_bits'] == 0
            and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 2979
            and audit['fresh_strict_reload_state_exact'] and audit['frozen_bytes_unchanged'] and audit['rank_zero'],
            'Require the naturally completed trajectory and exact native Llama subset export proof.')
    evidence = dict(review['evidence_sha256'])
    for path in (STEP / 'independent_review.json', STEP / 'final_model/config.json'):
        evidence[str(path)] = sha256(path)
    baseline, baseline_review, config0 = (read(DENSE / name) for name in
        ('scores.json', 'score_verification.json', 'config.json'))
    require(sha256(DENSE / 'scores.json') == baseline_review['scores_sha256'] ==
            'fd87e8b9c9937854806eb1adbfa4384482e0103df55d0105e82944fb72bdee3d'
            and sha256(DENSE / 'score_verification.json') ==
            '1e4812e04fa5587d0de135dadfec7c6b25938e6928c130b0a535b82817021090'
            and baseline['status'] == 'complete' and baseline_review['status'] == 'verified'
            and baseline['sample_count'] == baseline_review['sample_count'] == 4750
            and baseline_review['complete_tasks'] == len(baseline['tasks']) == 21
            and baseline_review['all_prediction_raw_source_token_decode_generation_checks_passed']
            and baseline['config_sha256'] == sha256(DENSE / 'config.json'), 'Require reviewed actual Llama BF16 full21.')
    for name in ('scores.json', 'score_verification.json', 'config.json'):
        path = DENSE / name; evidence[str(path)] = sha256(path)
    config = make_config(config0['archive']['path'], kind='compressed', hf_path=config0['model']['hf_path'],
                         checkpoint=STEP / 'final_model', mode='full')
    require(config['model']['checkpoint_id'] == ENDPOINT_ID
            and all(config[k] == config0[k] for k in ('schema', 'mode', 'protocol', 'archive',
                'package_versions', 'model_policy', 'prewarm_group_size', 'device')),
            'Keep actual original Llama SDPA/FP32RoPE/EOS/32K and all official generation caps.')
    check_config(config)
    manifest, index = load_archive(config['archive']['path'], check_token_file=False)
    require(len(index) == 4750 and all([r['row_index'] for r in index if r['dataset'] == task]
            == list(range(EXPECTED_COUNTS[task])) for task in DATASETS), 'Require exact archived21 tasks/4750 rows.')
    costs, peaks, raw_bytes = {}, {'allocated_bytes': 0, 'reserved_bytes': 0}, 0
    for task in DATASETS:
        path = DENSE / 'predictions' / (task + '.jsonl')
        require(sha256(path) == baseline['prediction_sha256'][str(path)], 'BF16 timing prediction changed.')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows) == EXPECTED_COUNTS[task] and all(row['dataset'] == task and row['row_index'] == i
                and row['config_sha256'] == baseline['config_sha256'] and row['diagnostic'] is False
                and row['elapsed_seconds'] > 0 for i, row in enumerate(rows)), 'Incomplete BF16 timing reference.')
        costs[task] = {'generation_seconds': math.fsum(row['elapsed_seconds'] for row in rows),
            'largest_sample_seconds': max(row['elapsed_seconds'] for row in rows)}
        for label in ('allocated_bytes', 'reserved_bytes'):
            peaks[label] = max(peaks[label], max(row['peak_' + label] for row in rows))
        raw_bytes += path.stat().st_size; evidence[str(path)] = sha256(path)
    local_controls, ratios = {}, []
    for label, directory, expected_id in (('source4e1', ORIGINAL, SOURCE_ID),
            ('e767_historical', E767, 'e767c232-56fd-4b03-a4ae-dd0781b6e047')):
        prior, prior_config = read(directory / 'review.json'), read(directory / 'config.json')
        require(prior['status'] == 'verified' and prior['checkpoint_id'] == expected_id
                and prior['samples'] == 400 and prior['execution_exit_code'] == 0
                and all(prior_config[k] == config[k] for k in ('protocol', 'archive', 'package_versions',
                    'model_policy', 'prewarm_group_size')), 'Local retrieval control differs.')
        if label == 'source4e1':
            require(sha256(directory / 'review.json') ==
                    '905de1d977bb1d9d7bea7ea4222be0117a10b19d9e01593e5bb42879c1191b19'
                    and config['source_sha256'] == prior_config['source_sha256'], 'Keep tested full-H252 S implementation.')
        local_controls[label] = {'checkpoint_id': expected_id, 'directory': str(directory), 'tasks': {}}
        for name in ('review.json', 'config.json'):
            path = directory / name; evidence[str(path)] = sha256(path)
        for task in RETRIEVAL:
            path = directory / 'run/predictions' / (task + '.jsonl')
            require(prior['evidence_sha256'][str(path)] == sha256(path), 'Reviewed local prediction changed.')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows) == 200, 'Incomplete local400 runtime reference.')
            seconds = math.fsum(row['elapsed_seconds'] for row in rows)
            ratios.append(seconds / costs[task]['generation_seconds'])
            local_controls[label]['tasks'][task] = {'samples': 200,
                'score_percent': prior['task_results'][task]['score_percent'], 'prediction_path': str(path),
                'generation_seconds': seconds, 'generation_ratio_to_BF16': ratios[-1]}
            evidence[str(path)] = sha256(path)
    factor = max(1.0, *ratios); jobs, merge_sources = [], {}
    config_path = OUTPUT / 'config.json'
    for number, task in enumerate(DATASETS, 1):
        directory = OUTPUT / 'partitions' / task
        jobs.append({'part': number, 'datasets': [task], 'samples': EXPECTED_COUNTS[task],
            'status': 'prepared_unbound', 'gpu': None, 'cwd': str(S), 'output_dir': str(directory),
            'bf16_generation_seconds': costs[task]['generation_seconds'],
            'bf16_largest_sample_seconds': costs[task]['largest_sample_seconds'],
            'estimated_generation_seconds': factor * costs[task]['generation_seconds'],
            'loading_allowance_seconds': 120, 'estimated_total_seconds': factor * costs[task]['generation_seconds'] + 120,
            'timeout_seconds': 10800, 'termination_grace_seconds': 20,
            'environment': {'PYTHONPATH': str(S), 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
            'argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config', str(config_path),
                     '--output', str(directory), '--datasets', task]})
        merge_sources[task] = str(directory / 'predictions' / (task + '.jsonl'))
    require(len(jobs) == 21 and sum(job['samples'] for job in jobs) == 4750, 'No omitted, reused or repeated task.')
    sources = {**config['source_sha256'], str(Path(__file__).resolve()): sha256(__file__)}
    public_source = PUBLIC / Path(__file__).resolve().relative_to(S)
    require(public_source.read_bytes() == Path(__file__).read_bytes(), 'Public/S new thin entry differs.')
    sources[str(public_source)] = sha256(public_source)
    for relative in ('experiments/longbench_v1/score.py', 'experiments/longbench_v1/scoring.py',
            'experiments/compare_longbench_v1.py', 'experiments/w25_eval/nine_task_acceptance.py',
            'experiments/w25_eval/summarize_nine_task.py'):
        path = PUBLIC / relative; sources[str(path)] = sha256(path)
    dense_paths = []
    for row in baseline_review['baseline_eight_raw_inputs']:
        require(sha256(row['path']) == row['sha256'], 'Original BF16 full8 changed.')
        dense_paths.append(row['path']); evidence[row['path']] = row['sha256']
    merged = OUTPUT / 'merged'
    score_argv = [sys.executable, '-m', 'experiments.longbench_v1.score', '--manifest', config['archive']['path'],
        '--prediction-dir', str(merged), '--output', str(merged / 'scores.json')]
    nine_argv = [sys.executable, '-m', 'experiments.w25_eval.summarize_nine_task', '--base_model_path',
        config['model']['hf_path'], '--checkpoint_dir', config['model']['checkpoint'], '--baseline_results', *dense_paths,
        '--compressed_results', str(STEP / 'mmlu.json'), str(STEP / 'seven.json'), '--longbench_manifest',
        config['archive']['path'], '--baseline_longbench_scores', str(DENSE / 'scores.json'),
        '--compressed_longbench_scores', str(merged / 'scores.json'), '--output', str(merged / 'nine_task_acceptance.json')]
    check_hashes(sources); check_hashes(evidence)
    record = {'schema': 'llama_fullqk_qkonly2000_longbench21_v1', 'status': 'prepared_unbound', 'gpu': None,
        'checkpoint_id': ENDPOINT_ID, 'source_root': str(S), 'cwd': str(S), 'python_executable': sys.executable,
        'config_path': str(config_path), 'source_sha256': sources, 'evidence_sha256': evidence,
        'new_prediction_samples': 4750, 'reused_prediction_samples': 0, 'final_sample_count': 4750,
        'partitions': jobs, 'source_bit_budget': audit['bit_budget'], 'completed_candidate': {
            'checkpoint_dir': str(STEP / 'final_model'), 'full8': review['result'], 'review_path': str(STEP / 'independent_review.json'),
            'native_subset_audit_reused': review['native_audit_reused'], 'source_checkpoint_id': SOURCE_ID},
        'full21_baseline': {'directory': str(DENSE), 'checkpoint_kind': 'BF16',
            'longbench_task_macro': baseline['task_macro_mean'], 'nine_task_mean': baseline_review['baseline_nine_mean']},
        'local_retrieval_controls': local_controls,
        'coverage_boundary': 'Only BF16 currently has a verified Llama complete21/4750 result in the inspected goal result roots. '
            'Original4e1 and e767 have only400 retrieval, so neither is a full21 compressed baseline.',
        'timing_reference_seconds': costs, 'bf16_generation_peak_bytes': peaks,
        'retrieval_observed_max_generation_ratio': factor, 'reference_prediction_bytes': raw_bytes,
        'new_disk_budget_bytes': 100 * 1024**2,
        'resource_boundary': 'Use actual BF16 whole-task generation times scaled by the largest observed compressed/BF16 '
            'retrieval ratio, plus120s load. This is a scheduling estimate, not a bound: other tasks/output lengths can differ. '
            'Reuse original fullQK strict2979/group8/SDPA/FP32RoPE/H252 and BF16 full21 protocol; no extra GPU short.',
        'scheduling': 'Root manually binds whole-task outputs to available cards or callback gaps. Long tasks can use GPU0; '
            'no GPU assignment, daemon, automatic retry or queue is made here.',
        'native_budget_resume': {'optional_flag': '--max-runtime-seconds', 'resume_flag': '--resume',
            'status_at_budget': 'partial_runtime_budget', 'preserve_same_config_output_datasets': True,
            'boundary': 'Native time budget checks between complete generations, including load time; allow in-flight '
                'generation and shutdown margin before the owner callback. Root may resume the same task/output only '
                'after the former process exits. Partial exit0 is not complete; no shortened caps or dropped rows.'},
        'merge': {'output_dir': str(merged), 'config_source': str(config_path), 'task_file_sources': merge_sources,
            'interface': 'After every task is naturally partition_complete/exit0 with exact configuration and canonical '
                'row coverage, copy unchanged config and21 complete JSONLs to merged/predictions; verify byte hashes before scoring.',
            'all_required_outputs_currently_exist': False},
        'score_argv': score_argv, 'nine_task_argv': nine_argv, 'scoring_cwd': str(PUBLIC),
        'scoring_environment': {'PYTHONPATH': str(PUBLIC), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
        'acceptance': {'longbench': 'Unrounded equally weighted21 official task scores.',
            'nine_terms': 'Eight downstream task scores plus full21 LongBench mean, divided by9.',
            'maximum_absolute_drop_points': 0.8, 'eight_task_hard_gate': False, 'mmlu_hard_gate': False,
            'source_boundary': 'Predict through unchanged S. Score and final acceptance through current public without obsolete MMLU65 gate.'},
        'nine_task_exit_semantics': {'0': 'goal passed', '1': 'valid full evaluation, goal not met', '2': 'invalid evidence'},
        'selection': 'Predetermined2000 endpoint; no best-step or task selection. All4750 original archived rows.',
        'queued': False, 'automatic_retry': False, 'automatic_resume': False, 'model_mutations': False,
        'full_longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated', 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    OUTPUT.mkdir(); write_json(config_path, config)
    record['config_sha256'] = sha256(config_path); write_json(OUTPUT / 'preparation.json', record)
    print({'status': record['status'], 'checkpoint_id': ENDPOINT_ID, 'tasks': len(jobs), 'new_samples': 4750,
        'sources': len(sources), 'inputs': len(evidence), 'preparation_sha256': sha256(OUTPUT / 'preparation.json'),
        'estimated_total_gpu_seconds': math.fsum(job['estimated_total_seconds'] for job in jobs)}, flush=True)


if __name__ == '__main__':
    prepare()
