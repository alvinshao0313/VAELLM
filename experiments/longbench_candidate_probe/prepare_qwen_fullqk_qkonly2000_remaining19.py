"""CPU preparation of the remaining official tasks for audited Qwen endpoint228a."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_candidate_probe.prepare_qwen_fullqk_qkonly2000 import (
    ENDPOINT_ID, OUTPUT as RETRIEVAL, PUBLIC, R2, S, STEP, read, require,
)
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS

OUTPUT = R2 / 'qwen_fullqk_qkonly2000_longbench21'
LB = PUBLIC / 'result/qwen3_8b_longbench_20261004'
REUSED = ('passage_retrieval_en', 'passage_retrieval_zh')


def prepare():
    require(Path.cwd() == S and Path(__file__).resolve().is_relative_to(S)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not OUTPUT.exists(), 'Use CUDA-hidden bitvae in S with a fresh output.')
    old, review = (read(RETRIEVAL / name) for name in ('preparation.json', 'review.json'))
    config_path = RETRIEVAL / 'config.json'; config = read(config_path); check_config(config)
    execution = read(review['execution_report']); scores = read(RETRIEVAL / 'scores.json')
    require(sha256(RETRIEVAL / 'review.json') ==
            '6d32a0b4fbc43e2f86d6a88497e6f15891a89ac4848ad88e37d5b04eec6c8cfc'
            and review['status'] == 'verified' and review['samples'] == 400
            and review['checkpoint_id'] == config['model']['checkpoint_id'] == ENDPOINT_ID
            and config['mode'] == 'full' and sha256(config_path) == old['config_sha256']
            and review['scores_sha256'] == sha256(RETRIEVAL / 'scores.json')
            and execution['status'] == 'partition_complete' and execution['exit_code'] == 0
            and execution['new_predictions'] == 400 and execution['already_present'] == 0
            and review['execution_exit_code'] == 0 and (RETRIEVAL / 'outer.exit').read_text().strip() == '0'
            and review['actual_input_tokens_and_original_ids_exact']
            and review['generation_options_effective_and_decoded_tokens_verified'],
            'Require the actual reviewed400 and exactly the same pinned full configuration.')
    sources = dict(old['source_sha256'])
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    sources[str(PUBLIC / Path(__file__).resolve().relative_to(S))] = sha256(PUBLIC / Path(__file__).resolve().relative_to(S))
    # The S snapshot intentionally remains frozen; public already implements the accepted nine-term rule.
    for relative in ('experiments/longbench_v1/score.py', 'experiments/longbench_v1/scoring.py',
                     'experiments/compare_longbench_v1.py', 'experiments/w25_eval/nine_task_acceptance.py',
                     'experiments/w25_eval/summarize_nine_task.py'):
        path = PUBLIC / relative; sources[str(path)] = sha256(path)
    evidence = {**old['evidence_sha256'], **review['evidence_sha256']}
    for name in ('preparation.json', 'review.json', 'scores.json', 'config.json', 'outer.exit'):
        path = RETRIEVAL / name; evidence[str(path)] = sha256(path)
    merge_sources = {}
    for task in REUSED:
        path = RETRIEVAL / 'run/predictions' / (task + '.jsonl')
        require(scores['prediction_sha256'][str(path)] == sha256(path), 'Reviewed retrieval prediction changed.')
        merge_sources[task] = str(path); evidence[str(path)] = sha256(path)
    reference_dirs = {'f927': LB / 'compressed', 'norm400': LB / 'compressed_norm400',
                      'l18': R2 / 'l18_gate_longbench21/merged'}
    costs, peaks, raw_bytes = {}, {}, {}
    for label, directory in reference_dirs.items():
        score = read(directory / 'scores.json'); reference = read(directory / 'config.json')
        require(score['status'] == 'complete' and score['sample_count'] == 4750
                and score['config_sha256'] == sha256(directory / 'config.json')
                and all(reference[k] == config[k] for k in ('protocol', 'archive', 'package_versions')),
                'Runtime reference must be a complete same-protocol Qwen run.')
        costs[label] = {}; peaks[label] = {'allocated_bytes': 0, 'reserved_bytes': 0}; raw_bytes[label] = 0
        for task in DATASETS:
            path = directory / 'predictions' / (task + '.jsonl')
            require(sha256(path) == score['prediction_sha256'][str(path)], 'Runtime prediction identity changed.')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows) == EXPECTED_COUNTS[task]
                    and all(row['dataset'] == task and row['row_index'] == i
                            and row['config_sha256'] == score['config_sha256']
                            and row['diagnostic'] is False and row['elapsed_seconds'] > 0
                            for i, row in enumerate(rows)), 'Incomplete canonical timing reference.')
            costs[label][task] = {'generation_seconds': sum(row['elapsed_seconds'] for row in rows),
                                 'largest_sample_seconds': max(row['elapsed_seconds'] for row in rows)}
            peaks[label]['allocated_bytes'] = max(peaks[label]['allocated_bytes'], *[row['peak_allocated_bytes'] for row in rows])
            peaks[label]['reserved_bytes'] = max(peaks[label]['reserved_bytes'], *[row['peak_reserved_bytes'] for row in rows])
            raw_bytes[label] += path.stat().st_size; evidence[str(path)] = sha256(path)
        for name in ('scores.json', 'config.json'):
            evidence[str(directory / name)] = sha256(directory / name)
    remaining = [task for task in DATASETS if task not in REUSED]
    jobs = []
    for index, task in enumerate(remaining, 1):
        directory = OUTPUT / 'partitions' / task
        conservative = max(costs[label][task]['generation_seconds'] for label in costs)
        jobs.append({'part': index, 'datasets': [task], 'samples': EXPECTED_COUNTS[task],
            'status': 'prepared_unbound', 'gpu': None, 'cwd': str(S), 'output_dir': str(directory),
            'generation_seconds_by_reference': {label: costs[label][task]['generation_seconds'] for label in costs},
            'largest_observed_sample_seconds': max(costs[label][task]['largest_sample_seconds'] for label in costs),
            'conservative_generation_seconds': conservative, 'loading_allowance_seconds': 120,
            'estimated_total_seconds': conservative + 120, 'timeout_seconds': 3600,
            'termination_grace_seconds': 20, 'environment': dict(old['environment']),
            'argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config', str(config_path),
                     '--output', str(directory), '--datasets', task]})
        merge_sources[task] = str(directory / 'predictions' / (task + '.jsonl'))
    require(len(jobs) == 19 and sum(j['samples'] for j in jobs) == 4350
            and set(merge_sources) == set(DATASETS) and len(merge_sources) == 21
            and all(j['estimated_total_seconds'] < j['timeout_seconds'] for j in jobs),
            'Require complete disjoint19 plus reused2, with plausible whole-task windows.')
    baseline = read(R2 / 'nine_task_baseline_1200/acceptance.json')
    dense_paths = []
    for row in baseline['downstream_baseline']['input_results']:
        require(sha256(row['path']) == row['sha256'], 'BF16 full8 source changed.')
        dense_paths.append(row['path']); evidence[row['path']] = row['sha256']
    for path in (R2 / 'nine_task_baseline_1200/acceptance.json', LB / 'dense/scores.json',
                 STEP / 'independent_review.json', STEP / 'mmlu.json', STEP / 'seven.json'):
        evidence[str(path)] = sha256(path)
    merged = OUTPUT / 'merged'
    score_argv = [sys.executable, '-m', 'experiments.longbench_v1.score', '--manifest', config['archive']['path'],
                  '--prediction-dir', str(merged), '--output', str(merged / 'scores.json')]
    nine_argv = [sys.executable, '-m', 'experiments.w25_eval.summarize_nine_task',
        '--base_model_path', config['model']['hf_path'], '--checkpoint_dir', config['model']['checkpoint'],
        '--baseline_results', *dense_paths, '--compressed_results', str(STEP / 'mmlu.json'), str(STEP / 'seven.json'),
        '--longbench_manifest', config['archive']['path'], '--baseline_longbench_scores', str(LB / 'dense/scores.json'),
        '--compressed_longbench_scores', str(merged / 'scores.json'), '--output', str(merged / 'nine_task_acceptance.json')]
    check_hashes(sources); check_hashes(evidence)
    record = {'schema': 'qwen_fullqk_qkonly2000_remaining19_v1', 'status': 'prepared_unbound', 'gpu': None,
        'checkpoint_id': ENDPOINT_ID, 'source_root': str(S), 'cwd': str(S), 'python_executable': sys.executable,
        'config_path': str(config_path), 'config_sha256': sha256(config_path), 'config_regenerated': False,
        'source_sha256': sources, 'evidence_sha256': evidence, 'reused_retrieval_review': str(RETRIEVAL / 'review.json'),
        'reused_prediction_samples': 400, 'new_prediction_samples': 4350, 'final_sample_count': 4750,
        'partitions': jobs, 'source_bit_budget': old['source_bit_budget'], 'native_load_audit_reused': review['native_load_audit'],
        'timing_reference_seconds': costs, 'generation_peak_bytes_by_reference': peaks,
        'reference_prediction_bytes': raw_bytes, 'new_disk_budget_bytes': 100 * 1024**2,
        'resource_boundary': 'Observed same-protocol complete f927/norm400/L18 tasks; use per-task maximum plus120s loading. '
            'Endpoint228a has measured H252 full-model loading from completed400. Output lengths and hardware contention can vary.',
        'scheduling': 'Root binds disjoint whole-task outputs manually to free GPU windows; no daemon, automatic queue or retry.',
        'native_budget_resume': {'optional_flag': '--max-runtime-seconds', 'resume_flag': '--resume',
            'status_at_budget': 'partial_runtime_budget', 'preserve_same_config_output_datasets': True,
            'boundary': 'Budget is checked between complete official generations, including load time. '
                'Allow the in-flight sample and cleanup before an owner callback. Partial exit0 is not complete; '
                'root explicitly resumes the same task/output after the process exits. No row cutoff or generation shortening.'},
        'merge': {'output_dir': str(merged), 'config_source': str(config_path), 'task_file_sources': merge_sources,
            'interface': 'After all19 are partition_complete/exit0 with exact config and canonical row coverage, '
                'copy the unchanged config and21 complete JSONLs once into merged/predictions; verify byte hashes before native scoring.',
            'do_not_generate_retrieval_again': True, 'all_required_outputs_currently_exist': False},
        'score_argv': score_argv, 'nine_task_argv': nine_argv, 'scoring_cwd': str(PUBLIC),
        'scoring_environment': {'PYTHONPATH': str(PUBLIC), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
        'acceptance': {'longbench': 'Unrounded equal mean of all21 official task scores, not category macro.',
            'nine_terms': 'Eight downstream task scores plus full21 LongBench macro, divided by9.',
            'maximum_absolute_drop_points': 0.8, 'eight_task_hard_gate': False, 'mmlu_hard_gate': False,
            'source_boundary': 'Predict using unchanged S; score/accept using public. S has the obsolete MMLU65 acceptance gate.'},
        'nine_task_exit_semantics': {'0': 'goal passed', '1': 'valid full evaluation, goal not met', '2': 'invalid evidence'},
        'queued': False, 'automatic_retry': False, 'automatic_resume': False, 'model_mutations': False,
        'full_longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated', 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    OUTPUT.mkdir(); write_json(OUTPUT / 'preparation.json', record)
    print({'status': record['status'], 'checkpoint_id': ENDPOINT_ID, 'tasks': len(jobs),
        'new_samples': 4350, 'reused_samples': 400, 'source_count': len(sources), 'evidence_count': len(evidence),
        'preparation_sha256': sha256(OUTPUT / 'preparation.json')}, flush=True)


if __name__ == '__main__':
    prepare()
