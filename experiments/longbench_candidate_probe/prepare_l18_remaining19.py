"""Prepare independent whole-task LongBench partitions; never schedule GPU work."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_candidate_probe.prepare_l18_gate import ROOT, R2, CHECKPOINT_ID, require
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS

LB = ROOT / 'result/qwen3_8b_longbench_20261004'
RETRIEVAL = R2 / 'l18_gate_retrieval400'
REUSED = ('passage_retrieval_en', 'passage_retrieval_zh')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT, 'Use the public source root and CUDA-hidden CPU preparation.')
    require(not any((output / name).exists() for name in ('preparation.json', 'partitions', 'merged')),
            'Use fresh partition and merged outputs.')
    old, review = read(RETRIEVAL / 'preparation.json'), read(RETRIEVAL / 'review.json')
    config_path = RETRIEVAL / 'config.json'; config = read(config_path); check_config(config)
    require(review['status'] == 'verified' and review['samples'] == 400
            and review['checkpoint_id'] == config['model']['checkpoint_id'] == CHECKPOINT_ID
            and config['mode'] == 'full' and sha256(config_path) == old['config_sha256']
            and review['execution_status'] == 'partition_complete' and review['execution_exit_code'] == 0,
            'Require the actual completed400 and exactly the same native full configuration.')
    check_hashes(old['source_sha256']); check_hashes(old['evidence_sha256']); check_hashes(review['evidence_sha256'])
    sources = dict(old['source_sha256'])
    for relative in ('experiments/longbench_candidate_probe/prepare_l18_remaining19.py',
                     'experiments/compare_longbench_v1.py', 'experiments/w25_eval/nine_task_acceptance.py',
                     'experiments/w25_eval/summarize_nine_task.py'):
        path = ROOT / relative; sources[str(path)] = sha256(path)
    evidence = {**old['evidence_sha256'], **review['evidence_sha256']}
    evidence.update({str(p): sha256(p) for p in (RETRIEVAL / 'preparation.json', RETRIEVAL / 'review.json', config_path)})
    costs = {}
    for kind in ('compressed', 'compressed_norm400'):
        directory = LB / kind; score = read(directory / 'scores.json'); control = read(directory / 'config.json')
        require(score['status'] == 'complete' and score['sample_count'] == 4750
                and score['config_sha256'] == sha256(directory / 'config.json')
                and all(control[key] == config[key] for key in ('protocol', 'archive', 'package_versions')),
                'Timing reference must be a complete same-protocol native run.')
        costs[kind] = {}
        for task in DATASETS:
            path = directory / 'predictions' / (task + '.jsonl')
            require(sha256(path) == score['prediction_sha256'][str(path)], 'Timing prediction identity changed.')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows) == EXPECTED_COUNTS[task]
                    and all(row['dataset'] == task and row['row_index'] == i
                            and row['config_sha256'] == score['config_sha256']
                            and row['elapsed_seconds'] > 0 for i, row in enumerate(rows)),
                    'Timing reference does not contain the complete canonical task.')
            costs[kind][task] = sum(row['elapsed_seconds'] for row in rows)
            evidence[str(path)] = sha256(path)
        for name in ('scores.json', 'config.json'):
            evidence[str(directory / name)] = sha256(directory / name)
    remaining = [task for task in DATASETS if task not in REUSED]
    conservative = {task: max(cost[task] for cost in costs.values()) for task in remaining}
    groups = [{'tasks': [task], 'seconds': conservative[task]} for task in ('gov_report', 'vcsum')]
    groups += [{'tasks': [], 'seconds': 0.0} for _ in range(6)]
    for task in sorted((t for t in remaining if t not in ('gov_report', 'vcsum')),
                       key=lambda t: (-conservative[t], t)):
        group = min(groups[2:], key=lambda g: g['seconds'])
        group['tasks'].append(task); group['seconds'] += conservative[task]
    # Put both long single tasks and the two longest ordinary groups in wave1.
    ordinary = sorted(groups[2:], key=lambda g: (-g['seconds'], tuple(g['tasks'])))
    ordered = groups[:2] + ordinary[:2] + ordinary[2:]
    tasks = [task for group in ordered for task in group['tasks']]
    require(len(tasks) == len(set(tasks)) == 19 and set(tasks) == set(remaining), 'Broken whole-task partition.')
    jobs = []
    for i, group in enumerate(ordered, 1):
        directory = output / 'partitions' / f'part{i:02d}'
        jobs.append({'part': i, 'suggested_wave': 1 if i <= 4 else 2,
            'status': 'prepared_unbound', 'gpu': None, 'output_dir': str(directory),
            'datasets': group['tasks'], 'samples': sum(EXPECTED_COUNTS[t] for t in group['tasks']),
            'generation_seconds_by_reference': {kind: sum(cost[t] for t in group['tasks']) for kind, cost in costs.items()},
            'conservative_generation_seconds': group['seconds'], 'loading_allowance_seconds': 120,
            'estimated_total_seconds': group['seconds'] + 120, 'timeout_seconds': 3600,
            'termination_grace_seconds': 20, 'environment': dict(old['environment']),
            'argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict',
                     '--config', str(config_path), '--output', str(directory), '--datasets', *group['tasks']]})
    require(all(job['estimated_total_seconds'] < 3600 for job in jobs), 'Estimated partition exceeds its requested window.')
    original = read(R2 / 'nine_task_baseline_1200/acceptance.json')
    dense_paths = [row['path'] for row in original['downstream_baseline']['input_results']]
    for row in original['downstream_baseline']['input_results']:
        require(sha256(row['path']) == row['sha256'], 'Paired BF16 full8 baseline changed.')
        evidence[row['path']] = row['sha256']
    dense_scores = LB / 'dense/scores.json'; evidence[str(dense_scores)] = sha256(dense_scores)
    evidence[str(R2 / 'nine_task_baseline_1200/acceptance.json')] = sha256(R2 / 'nine_task_baseline_1200/acceptance.json')
    merged = output / 'merged'
    merge_sources = {task: str(RETRIEVAL / 'run/predictions' / (task + '.jsonl')) for task in REUSED}
    for job in jobs:
        merge_sources.update({task: str(Path(job['output_dir']) / 'predictions' / (task + '.jsonl')) for task in job['datasets']})
    require(set(merge_sources) == set(DATASETS), 'Final merge must cover all21 exactly once.')
    score_argv = [sys.executable, '-m', 'experiments.longbench_v1.score', '--manifest', config['archive']['path'],
                  '--prediction-dir', str(merged), '--output', str(merged / 'scores.json')]
    candidate = Path(config['model']['checkpoint']).parent
    nine_argv = [sys.executable, '-m', 'experiments.w25_eval.summarize_nine_task',
        '--base_model_path', config['model']['hf_path'], '--checkpoint_dir', config['model']['checkpoint'],
        '--baseline_results', *dense_paths, '--compressed_results', str(candidate / 'mmlu.json'), str(candidate / 'seven.json'),
        '--longbench_manifest', config['archive']['path'], '--baseline_longbench_scores', str(dense_scores),
        '--compressed_longbench_scores', str(merged / 'scores.json'), '--output', str(merged / 'nine_task_acceptance.json')]
    check_hashes(sources); check_hashes(evidence)
    record = {'schema': 'l18_longbench_remaining19_preparation_v1', 'status': 'prepared_unbound', 'gpu': None,
        'checkpoint_id': CHECKPOINT_ID, 'source_root': str(ROOT), 'python_executable': sys.executable,
        'config_path': str(config_path), 'config_sha256': sha256(config_path), 'config_regenerated': False,
        'source_sha256': sources, 'evidence_sha256': evidence, 'reused_retrieval_review': str(RETRIEVAL / 'review.json'),
        'reused_prediction_samples': 400, 'new_prediction_samples': sum(EXPECTED_COUNTS[t] for t in remaining),
        'partitions': jobs, 'suggested_waves': [[j['part'] for j in jobs if j['suggested_wave'] == wave] for wave in (1, 2)],
        'scheduling': 'No queue or automatic next wave. Root binds each wave only after checking all active training/callback resources.',
        'timing_reference_seconds': costs, 'timing_estimate_boundary': 'Observed f927/norm generation sums; uses per-task maximum plus120s loading allowance. Changed output lengths may vary, so this is not a wall-time guarantee. Keep complete tasks and official generation budgets.',
        'merge': {'output_dir': str(merged), 'config_source': str(config_path), 'task_file_sources': merge_sources,
            'interface': 'No dedicated native merge CLI. After every assigned partition has partition_complete/exit0 and exact config SHA, copy config bytes and each complete task JSONL once into merged/predictions, require disjoint21-task coverage and identical bytes; then use native score CLI.',
            'do_not_generate_retrieval_again': True, 'all_required_outputs_currently_exist': False},
        'score_argv': score_argv, 'nine_task_argv': nine_argv,
        'nine_task_exit_semantics': {'0': 'goal passed', '1': 'valid complete evaluation, goal not met', '2': 'invalid evidence; stop'},
        'queued': False, 'automatic_retry': False, 'automatic_resume': False, 'model_mutations': False,
        'full_longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'cuda_initialized': torch.cuda.is_initialized()}
    require(not record['cuda_initialized'], 'CPU preparation unexpectedly initialized CUDA.')
    output.mkdir(parents=True, exist_ok=True); write_json(output / 'preparation.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); result = prepare(args.output_dir)
    print(json.dumps({'status': result['status'], 'partitions': len(result['partitions']),
                      'new_samples': result['new_prediction_samples'], 'reused_samples': result['reused_prediction_samples'],
                      'source_count': len(result['source_sha256']), 'evidence_count': len(result['evidence_sha256']),
                      'preparation_sha256': sha256(args.output_dir / 'preparation.json'),
                      'cuda_initialized': result['cuda_initialized']}, ensure_ascii=False))
