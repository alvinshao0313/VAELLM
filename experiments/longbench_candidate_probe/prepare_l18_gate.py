"""Bind the completed original-code L18 gate endpoint to the existing 16 cases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_attribution.protocol import check_preparation
from experiments.longbench_candidate_probe.prepare import check_candidate_preparation
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.longbench_v1.models import make_config
from experiments.w25_eval.acceptance import collect_results, read_json

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
CANDIDATE = R2 / 'gate_student_context_full8_r1/original_codes'
REFERENCE = ROOT / 'result/qwen3_8b_longbench_20261004/retrieval_attribution/preparation.json'
CHECKPOINT_ID = '289d4b3d-9c5e-4038-8a48-d67fe1f9194a'
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Hide CUDA for this CPU-only preparation.')
    require(Path.cwd() == ROOT, 'Use the public repository and its existing retrieval implementation.')
    output = Path(output).resolve()
    require(not output.exists() and not (output.parent / 'run').exists(), 'Use fresh diagnostic outputs.')
    checkpoint = CANDIDATE / 'final_model'
    launch_path, audit_path = (CANDIDATE / name for name in ('launch.json', 'audit.json'))
    summary_path, review_path = CANDIDATE / 'full8_summary.json', CANDIDATE.parent / 'review.json'
    launch, audit, summary, review = map(read_json, (launch_path, audit_path, summary_path, review_path))
    require(launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['dispatcher_exit_code'] == 0 and launch['arm'] == 'original_codes'
            and Path(launch['output_dir']) == CANDIDATE and Path(launch['final_model']) == checkpoint
            and launch['checkpoint_id'] == CHECKPOINT_ID and launch['source_checkpoint_id'] == SOURCE_ID,
            'Require the actual completed original-code L18 gate run.')
    exits = [CANDIDATE / (name + '.exit') for name in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer')]
    require(all(path.read_text().strip() == '0' for path in exits)
            and set(launch['stages']) == {'deploy', 'mmlu', 'seven'}
            and all(stage['exit_code'] == 0 for stage in launch['stages'].values()),
            'Every real source stage and terminal exit must be zero.')
    require(audit['status'] == 'verified' and audit['arm'] == 'original_codes'
            and audit['checkpoint_id'] == CHECKPOINT_ID and audit['source_checkpoint_id'] == SOURCE_ID
            and audit['patched_local_tensor_count'] == 13 and audit['frozen_outside_target_tensors'] == 3410
            and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423
            and audit['changed_code_bits'] == 0
            and all(audit[key] is True for key in ('fresh_strict_reload_state_exact', 'endpoint_state_exact',
                'frozen_outside_target_exact', 'frozen_local_nonmutable_exact', 'native_full_weight_reload_exact',
                'integer_budget_unchanged', 'rank_zero')),
            'Require the original-code endpoint and its completed strict/frozen native audit.')
    require(sha256(audit_path) == launch['audit_sha256'] == summary['audit_sha256']
            and sha256(summary_path) == launch['full8_summary_sha256'], 'Completed result identity changed.')
    require(review['status'] == 'verified' and review['source_checkpoint_id'] == SOURCE_ID,
            'Missing independent completed-source review.')
    arm_review = review['arms']['original_codes']
    require(arm_review['checkpoint_id'] == CHECKPOINT_ID and arm_review['changed_code_bits'] == 0
            and arm_review['whole_state_sha256_recomputed_exact'] == audit['checkpoint_state_sha256'],
            'Independent review belongs to a different endpoint.')
    check_hashes(arm_review['evidence_sha256'])
    check_stat(audit['checkpoint_state_stat'])
    require(Path(audit['checkpoint_state_stat']['path']) == checkpoint / 'pytorch_model.bin',
            'The audited stored weight is not this candidate.')
    budget = audit['bit_budget']
    require(budget['budget_passed'] and not budget['contract_violations']
            and budget['checkpoint_id'] == CHECKPOINT_ID and budget['compressed_target_count'] == 252
            and budget['user_linear_bits'] == 17065672704
            and budget['original_linear_weights'] == 6945767424
            and 2 * budget['user_linear_bits'] <= 5 * budget['original_linear_weights'],
            'Original integer storage contract changed.')
    reference, manifest = check_preparation(REFERENCE)
    config = make_config(reference['config']['archive']['path'], kind='compressed',
                         hf_path=manifest['tokenizer']['path'], checkpoint=checkpoint, mode='validation')
    require(config['model']['checkpoint_id'] == CHECKPOINT_ID
            and config['model']['metadata_sha256'] == audit['checkpoint_meta_sha256']
            and config['model']['weight_stat'] == audit['checkpoint_state_stat'],
            'Current native model metadata/stat differ from the completed audit.')
    meta = read_json(checkpoint / 'checkpoint_meta.json')
    scores = collect_results([CANDIDATE / 'mmlu.json', CANDIDATE / 'seven.json'], launch['architecture'],
                             checkpoint=(checkpoint, meta))
    require(summary['checkpoint_id'] == CHECKPOINT_ID and summary['status'] == 'full8_evaluation_verified'
            and scores['task_metrics'] == summary['task_metrics']
            and scores['input_results'] == summary['input_results']
            and scores['mean_percent'] == summary['mean_percent'] == launch['mean_percent']
            == arm_review['mean_percent'], 'Completed full-eight evidence differs from its actual raw results.')
    sources = dict(config['source_sha256'])
    sources.update(reference['source_sha256'])
    for relative in ('experiments/longbench_candidate_probe/prepare_l18_gate.py',
                     'experiments/longbench_candidate_probe/prepare.py',
                     'experiments/longbench_candidate_probe/run.py',
                     'experiments/w25_eval/acceptance.py', 'experiments/qwen3_w25/summarize_results.py'):
        path = ROOT / relative
        sources[str(path)] = sha256(path)
    evidence = dict(arm_review['evidence_sha256'])
    evidence.update({str(path): sha256(path) for path in [REFERENCE, review_path, *exits,
                    checkpoint / 'config.json']})
    record = {'schema': 'longbench_candidate_retrieval_probe_v1', 'status': 'prepared_not_scheduled',
        'gpu': None, 'queued': False, 'automatic_retry': False, 'source_root': str(ROOT),
        'python_executable': sys.executable, 'config': config, 'reference_preparation': str(REFERENCE),
        'cases': reference['cases'], 'selection': reference['selection'], 'source_sha256': sources,
        'evidence_sha256': evidence, 'source_bit_budget': budget,
        'completed_candidate': {'launch': str(launch_path), 'audit': str(audit_path), 'review': str(review_path),
            'checkpoint_id': CHECKPOINT_ID, 'source_checkpoint_id': SOURCE_ID,
            'terminal_exit_paths': [str(path) for path in exits], 'mean_percent': scores['mean_percent'],
            'mmlu_correct': arm_review['counts']['mmlu']['correct'], 'native_state_rehash_performed': False,
            'weight_identity': 'Reuses independently verified whole-state SHA and unchanged actual file stat.'},
        'output_dir': str(output.parent / 'run'), 'timeout_seconds': 900, 'termination_grace_seconds': 20,
        'environment': {'PYTHONPATH': str(ROOT), 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'},
        'launcher_argv': [sys.executable, '-m', 'experiments.longbench_candidate_probe.run',
            '--preparation', str(output), '--output', str(output.parent / 'run')],
        'scope': 'Fixed original sixteen-case retrieval diagnosis only; no full LongBench or nine-item acceptance.',
        'model_mutations': False, 'model_exported': False, 'goal_acceptance': False, 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    check_candidate_preparation(output)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({'status': result['status'], 'checkpoint_id': CHECKPOINT_ID,
                      'cases': len(result['cases']), 'sha256': sha256(args.output),
                      'cuda_initialized': torch.cuda.is_initialized()}, ensure_ascii=False))
