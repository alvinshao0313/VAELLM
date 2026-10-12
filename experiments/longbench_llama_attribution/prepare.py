"""Bind completed Llama D1200 retrieval controls to the existing sixteen cases."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.longbench_attribution.protocol import ARMS, TASKS
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.longbench_v1.protocol import generation_options
from experiments.longbench_v1.runner import read_input
from experiments.longbench_v1.scoring import check_prediction, score_example
from experiments.longbench_candidate_probe.prepare_llama_decoder1200_pair import audit_endpoint, DECODER, BASE_ID
from experiments.rht_retrieval_probe.prepare import select_native
from experiments.w25_eval.model_inventory import model_inventory

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
DENSE = ROOT / 'result/llama31_8b_longbench_20261004/dense'
CONTROL = R2 / 'llama_decoder1200_retrieval400'
CASES = R2 / 'attention_bits_retrieval16/llama_ov_control/preparation.json'
CASES_SHA = '6fc0623619c8ab1f49946c4d5a9aebe41f9383a0ac3196ec2b13c660a811df19'
PROFILES = {'coarse': ARMS, 'mlp': {'gate': ('gate_proj',), 'up': ('up_proj',), 'down': ('down_proj',)}}
COARSE = R2 / 'llama_d1200_retrieval_attribution'
COARSE_REVIEW_SHA = 'eb6ec0fb5c72ab70f3196869e57c429e133ea630e8e100638b9a294ed8d296c2'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_controls(config, rows, cases, tokenizer):
    evidence, controls = {}, {}
    for kind, directory, prediction_root, review_name in (
            ('dense', DENSE, DENSE, 'score_verification.json'),
            ('compressed', CONTROL, CONTROL / 'run', 'review.json')):
        old = read(directory / 'config.json')
        review, scores = read(directory / review_name), read(directory / 'scores.json')
        require(review['status'] == 'verified' and review['scores_sha256'] == sha256(directory / 'scores.json'),
                'Require completed independently verified retrieval controls.')
        if kind == 'dense':
            require(review['complete_tasks'] == 21 and review['sample_count'] == 4750
                    and review['score_exit_code'] == 0, 'Reuse the completed original BF16 full21 predictions.')
        else:
            execution = read(review['execution_report'])
            require(review['checkpoint_id'] == old['model']['checkpoint_id'] == BASE_ID
                    and old['model'] == config['model']
                    and review['samples'] == 400 and review['execution_exit_code'] == 0
                    and execution['status'] == 'partition_complete' and execution['exit_code'] == 0
                    and execution['load_audit']['checkpoint_id'] == BASE_ID
                    and execution['config_sha256'] == sha256(directory / 'config.json')
                    and execution['new_predictions'] + execution['already_present'] == 400
                    and execution['datasets'] == list(TASKS) and bool(review['terminal_exits'])
                    and all(value == 0 for value in review['terminal_exits'].values())
                    and review['evidence_sha256'][review['execution_report']] == sha256(review['execution_report'])
                    and review['actual_input_tokens_and_original_ids_exact']
                    and review['generation_options_effective_and_decoded_tokens_verified'],
                    'D1200 retrieval400 must finish and pass its independent review before preparation.')
            evidence[review['execution_report']] = sha256(review['execution_report'])
        require(old['model']['kind'] == kind and old['mode'] == 'full'
                and all(old[key] == config[key] for key in ('archive', 'protocol', 'package_versions', 'model_policy'))
                and all(old['model'][key] == config['model'][key]
                        for key in ('hf_path', 'hf_metadata_sha256', 'hf_weight_stats')),
                'Historical controls differ from the original Llama protocol or HF identity.')
        config_sha = sha256(directory / 'config.json')
        by_id = {}
        for task in TASKS:
            path = prediction_root / 'predictions' / (task + '.jsonl')
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            task_rows = [row for row in rows if row['dataset'] == task]
            require(len(predictions) == len(task_rows) == 200
                    and scores['prediction_sha256'][str(path)] == sha256(path), 'Require both complete official200 controls.')
            values = []
            for pred, row in zip(predictions, task_rows, strict=True):
                check_prediction(pred, row, config_sha)
                options = generation_options(tokenizer, task, row['input_tokens'], row['max_new_tokens'],
                                             protocol=config['protocol'])
                require(pred['generation_options'] == pred['effective_generation_options'] == options
                        and pred['use_model_defaults'] is False and row['max_new_tokens'] == 32
                        and len(pred['output_token_ids']) == pred['generated_tokens'] <= 32
                        and tokenizer.decode(pred['output_token_ids'], skip_special_tokens=True) == pred['pred'],
                        'Historical stopping, explicit generation or decoded output differs.')
                score = score_example(task, pred['pred'], row['answers'], row['all_classes'])
                values.append(score)
                by_id[(task, row['row_index'], row['_id'])] = {'source_prediction': pred, 'official_score': score}
            expected = scores['tasks'][task]['score' if kind == 'dense' else 'score_percent']
            require(abs(math.fsum(values) / 2 - expected) < 1e-10, 'Historical fractional scores differ.')
            evidence[str(path)] = sha256(path)
        controls[kind] = [by_id[(row['dataset'], row['row_index'], row['_id'])] for row in cases]
        for name in ('config.json', 'scores.json', review_name):
            path = directory / name
            evidence[str(path)] = sha256(path)
    return controls, evidence


def check_preparation(path):
    record = read(path)
    require(record['schema'] == 'llama_d1200_retrieval_restoration_v1'
            and record['config']['model']['checkpoint_id'] == BASE_ID
            and record['profile'] in PROFILES
            and record['arms'] == {key: list(value) for key, value in PROFILES[record['profile']].items()},
            'Wrong diagnostic identity.')
    check_hashes(record['source_sha256']); check_hashes(record['evidence_sha256'])
    check_config(record['config'])
    manifest, rows = load_archive(record['config']['archive']['path'])
    require(sha256(CASES) == CASES_SHA and record['cases'] == select_native(read(CASES), rows),
            'Keep the existing Llama sixteen IDs; do not reselect by Llama token length or predictions.')
    return record, manifest


def coarse_selection_evidence(cases):
    path = COARSE / 'independent_review.json'
    require(sha256(path) == COARSE_REVIEW_SHA, 'The completed coarse diagnosis review changed.')
    review = read(path)
    require(review['status'] == 'independent_cpu_verified' and review['checkpoint_id'] == BASE_ID
            and review['sample_count_per_arm'] == 16 and review['outer_and_all_workers_exited']
            and all(value == 0 for value in review['terminal_exits'].values()), 'Require the completed coarse diagnosis.')
    evidence = {str(path): COARSE_REVIEW_SHA}
    # Old runtime source pins describe the archived executed version, not today's profile extension.
    for arm in PROFILES['coarse']:
        path = COARSE / arm / 'predictions.jsonl'
        require(sha256(path) == review['evidence_sha256'][str(path)], 'Coarse raw predictions changed.')
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(raw) == len(cases) == 16 and all(
            value['checkpoint_id'] == BASE_ID and value['arm'] == arm and value['diagnostic'] is True
            and all(value[key] == row[key] for key in ('dataset', 'row_index', '_id', 'input_ids_sha256'))
            for value, row in zip(raw, cases, strict=True)), 'The MLP split must retain the actual coarse cases.')
        evidence[str(path)] = sha256(path)
    return evidence


def prepare(output, profile='coarse'):
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Prepare on CPU in the public bitvae workspace with CUDA hidden.')
    output = Path(output).resolve()
    require(profile in PROFILES, 'Unknown restoration profile.')
    arms = PROFILES[profile]
    require(not output.exists() and not any((output.parent / arm).exists() for arm in arms), 'Use fresh outputs.')
    # This required review intentionally blocks readiness while retrieval400 is running.
    require((CONTROL / 'review.json').is_file(), 'D1200 retrieval400 independent review is not yet available.')
    candidate, launch, evidence = audit_endpoint(DECODER, BASE_ID)
    old_config = read(CONTROL / 'config.json')
    config = make_config(old_config['archive']['path'], kind='compressed', hf_path=old_config['model']['hf_path'],
                         checkpoint=candidate['checkpoint_dir'], mode='validation')
    require(config['model_policy']['model_type'] == 'llama'
            and config['model']['checkpoint_id'] == BASE_ID, 'Require native Llama D1200.')
    manifest, rows = load_archive(config['archive']['path'])
    require(sha256(CASES) == CASES_SHA, 'Original Llama fixed-case provenance changed.')
    cases = select_native(read(CASES), rows)
    tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer']['path'], use_fast=True, local_files_only=True)
    tokens = np.memmap(manifest['inputs']['path'], dtype='<u4', mode='r')
    for row in cases:
        read_input(tokens, row)
    del tokens
    controls, control_evidence = read_controls(config, rows, cases, tokenizer)
    evidence.update(control_evidence); evidence[str(CASES)] = CASES_SHA
    if profile == 'mlp':
        evidence.update(coarse_selection_evidence(cases))
    architecture = model_inventory(config['model']['hf_path'])
    targets = {arm: [name for name in architecture['projection_shapes'] if name.rsplit('.', 1)[-1] in suffixes]
               for arm, suffixes in arms.items()}
    require(architecture['num_hidden_layers'] == 32 and {arm: len(names) for arm, names in targets.items()}
            == {arm: 32 * len(suffixes) for arm, suffixes in arms.items()}, 'Wrong target geometry.')
    sources = dict(config['source_sha256'])
    for relative in ('longbench_llama_attribution/prepare.py', 'longbench_llama_attribution/run.py',
                     'longbench_attribution/protocol.py', 'longbench_attribution/restore.py',
                     'rht_retrieval_probe/prepare.py', 'longbench_candidate_probe/prepare_llama_decoder1200_pair.py',
                     'longbench_candidate_probe/prepare_llama_best1600_d16.py', 'w25_eval/acceptance.py'):
        path = ROOT / 'experiments' / relative
        sources[str(path)] = sha256(path)
    record = {'schema': 'llama_d1200_retrieval_restoration_v1', 'status': 'prepared_unbound',
        'gpu': None, 'queued': False, 'automatic_retry': False, 'source_root': str(ROOT),
        'python_executable': sys.executable, 'config': config, 'architecture': architecture,
        'cases': cases, 'controls': controls, 'profile': profile, 'arms': arms, 'selected_targets': targets,
        'source_sha256': sources, 'evidence_sha256': evidence, 'completed_source': candidate,
        'output_root': str(output.parent), 'environment': launch['environment'],
        'selection': 'Reuse the existing eight EN/eight ZH original IDs, mapped to original Llama archived tokens; development diagnosis.',
        'scope': 'Three conditional HF-BF16 restorations on sixteen cases only; not a deployable budget-qualified candidate or full21/nine-item acceptance. Family restoration is not a per-bit optimization benefit.',
        'historical_selected_seconds': {kind: sum(x['source_prediction']['elapsed_seconds'] for x in values)
                                        for kind, values in controls.items()},
        'resources': {'whole_three_arm_seconds_estimate': [360, 600], 'suggested_outer_timeout_seconds': 900,
            'termination_grace_seconds': 20, 'gpu': 'One exclusively assigned A80080GB; three arms sequential and independently fresh.',
            'basis': 'Existing Qwen same three native restoration arms took107.54/108.98/86.41s; Llama same16 BF16 generation15.61s. Llama restoration total time is not yet measured.',
            'disk_provision_bytes': 268435456, 'checkpoint_written': False},
        'validation_reuse': 'Completed source native save/fresh/full8 and actual retrieval400; original Llama BF16 full21; existing Llama MMLU HF restoration. No duplicate source16/BF16 generation required.',
        'goal_acceptance': False, 'cuda_initialized': False}
    if profile == 'mlp':
        record['selection_evidence'] = {'review': str(COARSE / 'independent_review.json'),
            'sha256': COARSE_REVIEW_SHA,
            'reason': 'Coarse MLP restoration improved selected ZH from14.0625 to50; resolve gate/up/down conditional effects without changing the cases or source. Not independent additive contributions.'}
        record['resources'].update(whole_three_arm_seconds_estimate=[300, 480], suggested_outer_timeout_seconds=720,
            basis='Same Llama D1200 coarse diagnosis completed all3x16 in326.37s; individual arms113.34/108.62/75.95s. The32-projection MLP split is not yet measured; source load reserve73,660,366,848B remains the observed peak.')
    check_hashes(sources); check_hashes(evidence)
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.parent.mkdir(parents=True, exist_ok=True); write_json(output, record)
    check_preparation(output)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=tuple(PROFILES), default='coarse')
    args = parser.parse_args()
    value = prepare(args.output, args.profile)
    print(json.dumps({'status': value['status'], 'checkpoint_id': BASE_ID, 'cases': len(value['cases']),
                      'source_pins': len(value['source_sha256']), 'evidence_pins': len(value['evidence_sha256']),
                      'sha256': sha256(args.output), 'cuda_initialized': torch.cuda.is_initialized()}), flush=True)
