"""Bind completed native checkpoints to the original sixteen retrieval cases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.protocol import generation_options
from experiments.longbench_v1.runner import read_input
from experiments.longbench_v1.scoring import check_prediction, score_example
from experiments.w25_eval.acceptance import collect_results

ROOT = Path('/home/star/VAELLM')
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SNAPSHOT = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
QLB = ROOT / 'result/qwen3_8b_longbench_20261004'
LLB = ROOT / 'result/llama31_8b_longbench_20261004'
REFERENCE = QLB / 'retrieval_attribution/preparation.json'
BITS_ARMS = {
    'qwen3_o_control': (R2 / 'fresh_o_bits_pair/control', 17065672704),
    'qwen3_o_candidate': (R2 / 'fresh_o_bits_pair/candidate', 17364123648),
    'llama_ov_control': (ROOT / 'result/llama31_8b_w25_20261004/fresh_ov_bits_pair/control', 17100439552),
    'llama_ov_candidate': (ROOT / 'result/llama31_8b_w25_20261004/fresh_ov_bits_pair/candidate', 17432838144),
}
ARMS = ('qwen3_block32', 'qwen3_full', 'llama_block32', 'llama_full',
        'qwen3_ce8000_attention', *BITS_ARMS)
TASKS = ('passage_retrieval_en', 'passage_retrieval_zh')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_native(reference, rows):
    """Preserve Qwen's original selected IDs; never reselect by Llama token length."""
    by_id = {(row['dataset'], row['row_index'], row['_id']): row for row in rows}
    cases = []
    for original in reference['cases']:
        row = by_id[(original['dataset'], original['row_index'], original['_id'])]
        for key in ('source_line_sha256', 'prompt_text_sha256', 'answers', 'all_classes',
                    'length', 'max_new_tokens'):
            require(row[key] == original[key], 'Cross-model raw example changed: ' + key)
        cases.append(row)
    require(len(cases) == 16 and len({(r['dataset'], r['row_index']) for r in cases}) == 16
            and all(sum(r['dataset'] == task for r in cases) == 8 for task in TASKS),
            'Require exactly the existing eight English/eight Chinese examples.')
    return cases


def completed_source(arm):
    model_type = 'llama' if arm.startswith('llama_') else 'qwen3'
    if arm in BITS_ARMS:
        directory, expected_bits = BITS_ARMS[arm]
    else:
        directory = (R2 / 'best1200_ce8000_attention' if arm == 'qwen3_ce8000_attention'
                     else R2 / 'full_rht_gate_pair' / arm)
        expected_bits = 17100439552 if model_type == 'llama' else 17065672704
    launch = read(directory / 'launch.json')
    checkpoint = directory / 'final_model'
    meta = read(checkpoint / 'checkpoint_meta.json')
    assembly = read(checkpoint / 'assembly_report.json')
    summary = read(directory / 'full8_summary.json')
    inventory = launch['architecture']
    require(launch['status'] == 'completed' and launch['exit_code'] == 0
            and Path(launch['output_dir']) == directory and inventory['model_type'] == model_type,
            'Candidate full-eight run has not completed normally.')
    evidence = [directory / name for name in ('launch.json', 'full8_summary.json', 'mmlu.json',
        'seven.json', 'dispatcher.exit', 'outer.exit')]
    for stage, value in launch['stages'].items():
        path = directory / (stage + '.exit')
        require(value['exit_code'] == 0 and int(path.read_text()) == 0, 'Failed source stage: ' + stage)
        evidence.append(path)
    require(all(int((directory / name).read_text()) == 0 for name in ('dispatcher.exit', 'outer.exit')),
            'Source dispatcher/outer process has not exited successfully.')
    count = 2979 if model_type == 'llama' else 3423
    budget = assembly['bit_budget']
    require(assembly['status'] == 'verified'
            and assembly['serialized_tensors_compared'] == assembly['reloaded_tensors_compared'] == count
            and assembly['checkpoint_id'] == summary['checkpoint_id'] == meta['checkpoint_id']
            and assembly['sources']['base_checkpoint_id'] == launch['base_checkpoint_id']
            and meta['base_model_path'] == inventory['base_model_path']
            and budget['budget_passed'] and not budget['contract_violations']
            and not budget['unclassified_target_keys']
            and budget['user_linear_bits'] == expected_bits
            and budget['original_linear_weights'] == inventory['original_linear_weights']
            and budget['compressed_target_count'] == inventory['projection_count'],
            'Native source identity, strict assembly or integer budget differs.')
    require(2 * budget['user_linear_bits'] < 5 * budget['original_linear_weights'],
            'These fixed source checkpoints must remain below 2.5 bits.')
    require(all(budget['components'].get(key, {}).get('storage_bits', 0) == 0
                for key in ('lora', 'residual_lora', 'dense_weights')), 'Unexpected source payload.')
    scores = collect_results([directory / 'mmlu.json', directory / 'seven.json'], inventory,
                             checkpoint=(checkpoint, meta))
    require(scores['task_metrics'] == summary['task_metrics']
            and scores['mean_percent'] == summary['mean_percent'] == launch['mean_percent']
            and sha256(directory / 'full8_summary.json') == launch['full8_summary_sha256'],
            'Completed full-eight summary differs from the actual native raw results.')
    proof_path = directory / ('independent_review.json' if arm == 'qwen3_ce8000_attention' else 'audit.json')
    proof = read(proof_path)
    require(proof['status'] == 'verified' and proof['checkpoint_id'] == meta['checkpoint_id'],
            'Missing verified completed-checkpoint evidence.')
    evidence += [proof_path, checkpoint / 'checkpoint_meta.json', checkpoint / 'assembly_report.json']
    if arm in BITS_ARMS:
        pair_path = directory.parent / 'comparison.json'
        pair = read(pair_path)
        label = directory.name
        require(pair['status'] == 'verified_pair' and pair['model_type'] == model_type
                and pair['primary_comparison'] == 'candidate_minus_fresh_control'
                and pair['categories'] == launch['categories']
                and pair['codebook_bits'][label] == launch['codebook_bits']
                and pair['arms'][label]['checkpoint_id'] == meta['checkpoint_id']
                and pair['arms'][label]['task_metrics'] == scores['task_metrics']
                and launch['expected_user_linear_bits'] == expected_bits
                and proof['assembly_report_sha256'] == sha256(checkpoint / 'assembly_report.json')
                and all(budget[key] == value for key, value in pair['bit_budgets'][label].items()),
                'Fresh rate-pair identity or audited bit budget differs.')
        check_hashes(pair['evidence_sha256'])
        check_hashes(pair['comparison_source_sha256'])
        evidence += [pair_path, *map(Path, pair['evidence_sha256']),
                     *map(Path, pair['comparison_source_sha256'])]
    return checkpoint, budget, {str(p): sha256(p) for p in evidence}


def historical_controls(lbroot, cases, rows, config, reference):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['hf_path'], use_fast=True, local_files_only=True)
    controls, evidence = {}, {}
    for kind in (('dense', 'compressed') if lbroot == QLB else ('dense',)):
        path = lbroot / kind / 'config.json'
        old = read(path); config_sha = sha256(path); evidence[str(path)] = config_sha
        for key in ('archive', 'protocol', 'package_versions'):
            require(old[key] == config[key], 'Historical control protocol differs: ' + key)
        for key in ('hf_path', 'hf_metadata_sha256', 'hf_weight_stats'):
            require(old['model'][key] == config['model'][key], 'Historical HF identity differs: ' + key)
        require(old['model']['kind'] == kind and old['mode'] == 'full', 'Require real full-run controls.')
        by_id = {}
        for task in TASKS:
            path = lbroot / kind / 'predictions' / (task + '.jsonl')
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            task_rows = [row for row in rows if row['dataset'] == task]
            require(len(predictions) == len(task_rows) == 200, 'Historical retrieval control is incomplete.')
            evidence[str(path)] = sha256(path)
            for prediction, row in zip(predictions, task_rows, strict=True):
                check_prediction(prediction, row, config_sha)
                by_id[(row['dataset'], row['row_index'], row['_id'])] = prediction
        values = []
        for row in cases:
            predicted = by_id[(row['dataset'], row['row_index'], row['_id'])]
            options = generation_options(tokenizer, row['dataset'], row['input_tokens'],
                                         row['max_new_tokens'], protocol=config['protocol'])
            require(predicted['generation_options'] == predicted['effective_generation_options'] == options
                    and predicted['use_model_defaults'] is False
                    and predicted['input_tokens'] == row['input_tokens']
                    and len(predicted['output_token_ids']) == predicted['generated_tokens']
                    and tokenizer.decode(predicted['output_token_ids'], skip_special_tokens=True) == predicted['pred'],
                    'Historical generation settings or decoded output differ.')
            values.append({'source_prediction': predicted, 'official_score': score_example(
                row['dataset'], predicted['pred'], row['answers'], row['all_classes'])})
        if lbroot == QLB:
            require(values == reference['controls'][kind], 'Original Qwen selected control records changed.')
        controls[kind] = values
    return controls, evidence


def check_preparation(path):
    record = read(path)
    require(record['schema'] == 'rht_retrieval_probe_v1' and record['arm'] in ARMS
            and Path(record['source_root']) == SNAPSHOT, 'Unexpected retrieval preparation.')
    check_hashes(record['source_sha256']); check_hashes(record['evidence_sha256'])
    check_config(record['config'])
    for stat in record['archive_stats']:
        check_stat(stat)
    manifest, rows = load_archive(record['config']['archive']['path'], check_token_file=False)
    require(record['config']['mode'] == 'validation' and record['config']['model']['kind'] == 'compressed'
            and record['cases'] == select_native(read(REFERENCE), rows), 'Native selected inputs changed.')
    tokens = np.memmap(manifest['inputs']['path'], dtype='<u4', mode='r')
    for row in record['cases']:
        read_input(tokens, row)
    del tokens
    return record, manifest


def prepare(arm, output):
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Run CPU preparation with CUDA_VISIBLE_DEVICES empty.')
    require(Path(__file__).resolve().parents[2] == SNAPSHOT and Path.cwd() == SNAPSHOT,
            'Prepare using the physical snapshot and its native model policy.')
    output = Path(output).resolve()
    require(arm in ARMS and not output.exists(), 'Use one known arm and a new preparation path.')
    checkpoint, budget, evidence = completed_source(arm)
    reference = read(REFERENCE)
    lbroot = LLB if arm.startswith('llama_') else QLB
    archive = lbroot / 'data/manifest.json'
    manifest, rows = load_archive(archive)
    cases = select_native(reference, rows)
    config = make_config(archive, kind='compressed', hf_path=manifest['tokenizer']['path'],
                         checkpoint=checkpoint, mode='validation')
    controls, control_evidence = historical_controls(lbroot, cases, rows, config, reference)
    sources = dict(config['source_sha256'])
    for relative in ('experiments/rht_retrieval_probe/prepare.py', 'experiments/rht_retrieval_probe/run.py',
                     'experiments/longbench_attribution/restore.py', 'experiments/w25_eval/acceptance.py',
                     'experiments/qwen3_w25/summarize_results.py'):
        path = SNAPSHOT / relative; sources[str(path)] = sha256(path)
    evidence.update(control_evidence)
    evidence.update({str(REFERENCE): sha256(REFERENCE), str(archive): sha256(archive),
                     manifest['index']['path']: manifest['index']['sha256']})
    evidence.update({path: digest for path, digest in manifest['data']['raw_sha256'].items()
                     if Path(path).stem in TASKS})
    record = {'schema': 'rht_retrieval_probe_v1', 'status': 'prepared_unbound', 'gpu': None,
        'queued': False, 'automatic_retry': False, 'arm': arm, 'source_root': str(SNAPSHOT),
        'python_executable': sys.executable, 'output_dir': str(output.parent / 'run'),
        'config': config, 'cases': cases, 'controls': controls, 'source_sha256': sources,
        'evidence_sha256': evidence, 'archive_stats': [file_stat(manifest['inputs']['path'])],
        'environment': {'PYTHONPATH': str(SNAPSHOT), 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                        'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                        'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'},
        'launcher_argv': [sys.executable, '-m', 'experiments.rht_retrieval_probe.run', '--preparation', str(output)],
        'timeout_seconds': 900, 'termination_grace_seconds': 20, 'source_bit_budget': budget,
        'selection': 'Reuse the original Qwen sixteen (task,row_index,_id) identities in order; map into each native archive. No new quantile selection.',
        'input_tokens_range': [min(r['input_tokens'] for r in cases), max(r['input_tokens'] for r in cases)],
        'selected_truncated_count': sum(r['removed_tokens'] > 0 for r in cases),
        'length_boundary': '32768 is the shared total budget upper bound, not a claim that each selected example has 32K tokens.',
        'scope': 'Fixed sixteen-case retrieval diagnosis only; no full LongBench or nine-item acceptance.',
        'model_mutations': False, 'model_exported': False, 'goal_acceptance': False, 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    check_preparation(output)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.arm, args.output)
    print(json.dumps({'arm': args.arm, 'status': result['status'], 'cases': len(result['cases']),
                      'sha256': sha256(args.output)}, ensure_ascii=False))
