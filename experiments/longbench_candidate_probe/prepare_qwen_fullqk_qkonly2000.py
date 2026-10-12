"""Bind the predetermined full-QK, QK-only LM step2000 to official retrieval400."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
S = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
TRAIN = R2 / 'general_lm_qwen_fullqk_qkonly_s2000'
STEP = TRAIN / 'step_2000'
CONTROL = R2 / 'qwen_qk_rotation_retrieval400/full'
OUTPUT = R2 / 'qwen_fullqk_qkonly2000_retrieval400'
SOURCE_ID = '43b0c5a2-4ef7-4a00-a6d1-ed887be18d01'
ENDPOINT_ID = '228aa08f-9abf-4421-b907-073629e1045e'
TASKS = ('passage_retrieval_en', 'passage_retrieval_zh')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare():
    require(Path.cwd() == S and Path(__file__).resolve().is_relative_to(S)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not OUTPUT.exists(), 'Use CUDA-hidden bitvae and actual S cwd/PYTHONPATH; output must be new.')
    review = read(STEP / 'independent_review.json')
    require(sha256(STEP / 'independent_review.json') ==
            'c1151a73c33f44936c205dc7ed1eec8aa59b77bafe9f9b69ad230d8096b00f4c'
            and review['status'] == 'verified_independent_cpu_review' and review['global_step'] == 2000
            and review['checkpoint_id'] == ENDPOINT_ID and review['source_checkpoint_id'] == SOURCE_ID
            and review['whole_training']['status'] == 'completed'
            and review['whole_training']['completed_steps'] == [400, 800, 1200, 1600, 2000]
            and set(review['whole_training']['exit_codes'].values()) == {0}
            and all(p['retired'] for p in review['whole_training']['retired_processes'].values()),
            'Require the actual completed2000 export and independently reviewed complete trajectory.')
    inputs = dict(review['evidence_sha256'])
    for path in (STEP / 'independent_review.json', STEP / 'final_model/config.json'):
        inputs[str(path)] = sha256(path)
    check_hashes(inputs)
    launch, trained, audit = (read(path) for path in
        (TRAIN / 'launch.json', TRAIN / 'training.json', STEP / 'audit.json'))
    require(launch['status'] == trained['status'] == 'completed' and launch['exit_code'] == 0
            and launch['completed_steps'] == trained['callback_completed_steps'] == [400, 800, 1200, 1600, 2000]
            and all((TRAIN / (name + '.exit')).read_text().strip() == '0' for name in ('run', 'outer')),
            'Whole native run must finish naturally before retrieval preparation.')
    require(audit['status'] == 'verified' and audit['checkpoint_id'] == ENDPOINT_ID
            and audit['selected_decoder_tensor_count'] == 432 and audit['frozen_tensors_compared'] == 2991
            and audit['code_bank_count'] == 504 and audit['changed_code_bits'] == 0
            and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423
            and audit['fresh_strict_reload_state_exact'] and audit['frozen_bytes_unchanged'] and audit['rank_zero'],
            'Reuse the native exact subset export audit; no tensor reload in CPU preparation.')
    budget = audit['bit_budget']
    require(budget['user_linear_bits'] == budget['packed_linear_bits'] == 17065672704
            and budget['original_linear_weights'] == 6945767424 and budget['code_padding_bits'] == 0,
            'Require actual unchanged2.45698879076bpw/rank0.')
    old, config0, control_review, scores = (read(CONTROL / name) for name in
        ('preparation.json', 'config.json', 'review.json', 'scores.json'))
    require(sha256(CONTROL / 'review.json') ==
            '4818ddab48c11bb3115c958f1997d8982ee8e4a9966a61429260c099aed1c95e'
            and control_review['status'] == 'verified' and control_review['samples'] == 400
            and control_review['execution_exit_code'] == 0
            and control_review['checkpoint_id'] == scores['checkpoint_id'] == SOURCE_ID
            and control_review['paired_input_generation_options_exact']
            and control_review['decoded_output_tokens_exact']
            and old['config_sha256'] == scores['config_sha256'] == sha256(CONTROL / 'config.json')
            and control_review['evidence_sha256'][str(CONTROL / 'scores.json')] == sha256(CONTROL / 'scores.json'),
            'Require reviewed43b0 official400 primary control.')
    for name in ('preparation.json', 'config.json', 'review.json', 'scores.json'):
        path = CONTROL / name; inputs[str(path)] = sha256(path)
    controls = {'source43b0': {'directory': str(CONTROL), 'checkpoint_id': SOURCE_ID,
        'model_reload_required': False, 'tasks': {task: {'samples': 200,
            'score_percent': control_review['task_results'][task]['score_percent'],
            'prediction_path': str(CONTROL / 'run/predictions' / (task + '.jsonl'))} for task in TASKS}},
        'f927_historical': old['controls']['f927']}
    config = make_config(config0['archive']['path'], kind='compressed',
        hf_path=config0['model']['hf_path'], checkpoint=STEP / 'final_model', mode='full')
    require(config['model']['checkpoint_id'] == ENDPOINT_ID
            and all(config[k] == config0[k] for k in ('schema', 'mode', 'protocol', 'archive',
                'package_versions', 'model_policy', 'prewarm_group_size', 'device', 'source_sha256')),
            'Only the reviewed checkpoint may differ: retain exact S/H252 and original official generation policy.')
    check_config(config)
    for control in controls.values():
        directory = Path(control['directory']); historical = read(directory / 'config.json')
        require(all(historical[k] == config[k] for k in ('protocol', 'archive', 'package_versions')),
                'Historical400 controls must share actual archive and official protocol.')
        inputs[str(directory / 'config.json')] = sha256(directory / 'config.json')
        for task in TASKS:
            path = control['tasks'][task]['prediction_path']
            require(scores['prediction_sha256'][path] == sha256(path), 'Reviewed original400 prediction changed.')
            inputs[path] = sha256(path)
    _, index = load_archive(config['archive']['path'], check_token_file=False)
    rows = {task: [row for row in index if row['dataset'] == task] for task in TASKS}
    require(all([row['row_index'] for row in value] == list(range(200)) for value in rows.values()),
            'Use every original archived row0..199 in both tasks, without selection or resampling.')
    sources = {**config['source_sha256'], str(Path(__file__).resolve()): sha256(__file__)}
    check_hashes(sources); check_hashes(inputs)
    record = {'schema': 'qwen_fullqk_qkonly2000_retrieval400_v1', 'status': 'prepared_unbound', 'gpu': None,
        'source_root': str(S), 'cwd': str(S), 'python_executable': sys.executable,
        'checkpoint_id': ENDPOINT_ID, 'config_path': str(OUTPUT / 'config.json'),
        'output_dir': str(OUTPUT / 'run'), 'datasets': list(TASKS), 'sample_count': 400,
        'task_samples': {task: 200 for task in TASKS}, 'source_sha256': sources, 'evidence_sha256': inputs,
        'source_bit_budget': budget, 'controls': controls, 'primary_comparison': 'step2000_minus_source43b0',
        'completed_candidate': {'directory': str(TRAIN), 'checkpoint_dir': str(STEP / 'final_model'),
            'source_checkpoint_id': SOURCE_ID, 'full8': review['result'], 'versus_source': review['versus_fresh_source'],
            'whole_training': review['whole_training'], 'native_subset_audit': review['native_audit_reused'],
            'trajectory_review': str(STEP / 'independent_review.json')},
        'environment': {**old['environment'], 'PYTHONPATH': str(S)},
        'input_lengths': {task: {'min': min(row['input_tokens'] for row in value),
            'max': max(row['input_tokens'] for row in value)} for task, value in rows.items()},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config',
            str(OUTPUT / 'config.json'), '--output', str(OUTPUT / 'run'), '--datasets', *TASKS],
        'timeout_seconds': 900, 'termination_grace_seconds': 20, 'queued': False,
        'automatic_retry': False, 'resume': False, 'selection': 'Predetermined step2000; original archived200+200 rows. '
            'No eight-task peak selection, truncation change, or additional19-task evaluation.',
        'resource_evidence': {'same_protocol_source43b0_seconds': control_review['duration_seconds'],
            'wall_seconds_estimate': [420, 600], 'new_disk_budget_bytes': 100 * 1024**2,
            'boundary': 'Native single-GPU400 inference only; no export, training or additional short test.'},
        'scoring': old['scoring'], 'comparison_boundary': 'Primary is this exact43b0 training source; f927 is historical. '
            'The candidate changes only Q/K decoders during FineWeb recovery, with original code/sign/protection frozen.',
        'validation_reuse': 'Completed native3423 save/fresh,432 trainable decoder tensors,2991 frozen states/504 fixed banks. '
            'Retrieval uses audited step2000 export228a, not the distinct native whole-final ecd6 artifact. '
            'Original predictor retains strict load, actual bit budget, BF16/FP32RoPE/FA2 and group8 checks.',
        'scope': 'Only2/21 complete official retrieval tasks; no fullLongBench or nine-item acceptance claim.',
        'full_longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'model_mutations': False, 'model_exported': False, 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    OUTPUT.mkdir(); write_json(OUTPUT / 'config.json', config)
    record['config_sha256'] = sha256(OUTPUT / 'config.json'); write_json(OUTPUT / 'preparation.json', record)
    print({'status': record['status'], 'checkpoint_id': ENDPOINT_ID, 'source_count': len(sources),
        'input_count': len(inputs), 'preparation_sha256': sha256(OUTPUT / 'preparation.json')}, flush=True)


if __name__ == '__main__':
    prepare()
