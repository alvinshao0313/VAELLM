"""Bind a completed matched Q/K rotation arm to both complete retrieval tasks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
SNAPSHOT = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
PAIR = R2 / 'qwen_qk_rotation_pair'
PROTOCOL = R2 / 'l18_gate_retrieval400'
TASKS = ('passage_retrieval_en', 'passage_retrieval_zh')
IDS = {'block32': '1b5f063b-9b87-4495-b940-dcc3d0fa8118', 'full': '43b0c5a2-4ef7-4a00-a6d1-ed887be18d01'}
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def prepare(arm, output, comparison):
    require(ROOT == SNAPSHOT and Path.cwd() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Prepare with CUDA hidden in the immutable full-RHT snapshot, not the active public tree.')
    output, comparison = Path(output).resolve(), Path(comparison).resolve()
    require(arm in IDS and not output.exists(), 'Use one known arm and a fresh independent output.')
    pair = read(comparison)
    require(pair['status'] == 'verified_pair' and pair['model_type'] == 'qwen3'
            and pair['categories'] == ['q_proj', 'k_proj']
            and pair['primary_comparison'] == 'full_minus_new_fresh_block32'
            and all(pair['arms'][key]['checkpoint_id'] == value for key, value in IDS.items())
            and pair['arms']['base']['checkpoint_id'] == SOURCE_ID
            and pair['pair_checks'] == {'same_recipe_except_output_and_block': True, 'same_category_rng': True,
                'q_and_k_two_complete_10000_stages': True, 'protection_records_exact': True,
                'rotation_sign_records_exact': True, 'native_union_save_fresh_tensors': 3423,
                'leaf_count': 64, 'sample_count': 24742, 'mmlu_subjects': 57, 'mmlu_samples': 14042},
            'Require the actual completed matched full8 pair, including equal protection/sign records.')
    check_hashes(pair['evidence_sha256']); check_hashes(pair['comparison_source_sha256'])
    directory = PAIR / arm; checkpoint = directory / 'final_model'
    launch, audit, summary, meta, assembled = (read(directory / name) for name in
        ('launch.json', 'audit.json', 'full8_summary.json', 'final_model/checkpoint_meta.json', 'final_model/assembly_report.json'))
    require(launch['status'] == 'completed' and launch['exit_code'] == 0 and launch['arm'] == arm
            and launch['base_checkpoint_id'] == audit['base_checkpoint_id'] == SOURCE_ID
            and all(x['checkpoint_id'] == IDS[arm] for x in (launch, audit, summary, meta, assembled))
            and audit['status'] == assembled['status'] == 'verified'
            and audit['source_union_save_fresh_tensors'] == assembled['serialized_tensors_compared']
            == assembled['reloaded_tensors_compared'] == 3423
            and audit['replaced_projection_count'] == 72 and audit['preserved_projection_count'] == 180
            and audit['replaced_local_tensor_count'] == 936 and audit['frozen_complement_tensor_count'] == 2487
            and audit['assembly_report_sha256'] == sha256(checkpoint / 'assembly_report.json')
            and launch['full8_summary_sha256'] == sha256(directory / 'full8_summary.json')
            and pair['arms'][arm]['task_metrics'] == summary['task_metrics']
            and pair['arms'][arm]['mean_percent'] == summary['mean_percent'] == launch['mean_percent'],
            'Actual Q/K category assembly/full8 source differs from the verified pair.')
    require(all((directory / (stage + '.exit')).read_text().strip() == '0'
                for stage in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))
            and all(stage['exit_code'] == 0 for stage in launch['stages'].values()), 'Source stages must end successfully.')
    check_hashes(launch['source_sha256']); check_hashes(launch['input_sha256'])
    for stat in launch['protected_stats']: check_stat(stat)
    budget = audit['bit_budget']
    require(budget == assembled['bit_budget'] and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17065672704
            and budget['original_linear_weights'] == 6945767424 and budget['code_padding_bits'] == 0
            and budget['components'] == pair['bit_budget']['components'] and budget['budget_passed']
            and not budget['contract_violations'] and budget['components']['decoder']['dtypes'] == ['bfloat16']
            and all(budget['components'].get(key, {}).get('storage_bits', 0) == 0 for key in ('lora', 'residual_lora', 'dense_weights')),
            'Actual rank-zero same-rate storage differs.')
    previous, previous_review = read(PROTOCOL / 'preparation.json'), read(PROTOCOL / 'review.json')
    old_config = read(previous['config_path'])
    require(previous_review['status'] == 'verified' and previous_review['execution_status'] == 'partition_complete'
            and previous_review['execution_exit_code'] == 0 and sha256(previous['config_path']) == previous['config_sha256'],
            'Require the completed official400 protocol proof; no previous model reload is needed.')
    config = make_config(old_config['archive']['path'], kind='compressed', hf_path=launch['architecture']['base_model_path'],
                         checkpoint=checkpoint, mode='full')
    require(config['model']['checkpoint_id'] == IDS[arm]
            and all(config[k] == old_config[k] for k in ('schema', 'mode', 'protocol', 'archive', 'package_versions', 'model_policy', 'prewarm_group_size', 'device'))
            and all(config['model'][k] == old_config['model'][k] for k in ('kind', 'hf_path', 'hf_metadata_sha256', 'hf_weight_stats')),
            'Keep original official data/model policy; only checkpoint identity and snapshot source pins may differ.')
    check_config(config)
    _, index = load_archive(config['archive']['path'], check_token_file=False)
    rows = {task: [row for row in index if row['dataset'] == task] for task in TASKS}
    require(all(len(value) == 200 and [r['row_index'] for r in value] == list(range(200)) for value in rows.values()),
            'Use every original EN200/ZH200 row in archive order, without a selection.')
    sources = dict(launch['source_sha256'])
    for path, digest in config['source_sha256'].items():
        require(path not in sources or sources[path] == digest, 'Native training/generation versions conflict: ' + path)
        sources[path] = digest
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    evidence = {**launch['input_sha256'], **pair['evidence_sha256'], **pair['comparison_source_sha256']}
    for path in (comparison, PROTOCOL / 'preparation.json', PROTOCOL / 'review.json', Path(previous['config_path'])):
        evidence[str(path)] = sha256(path)
    controls = {key: previous['controls'][key] for key in ('bf16', 'f927')}
    for control in controls.values():
        root = Path(control['directory']); control_config = read(root / 'config.json')
        require(all(control_config[k] == config[k] for k in ('archive', 'protocol', 'package_versions')),
                'Historical control generation protocol differs.')
        for name in ('config.json', 'scores.json', *('predictions/' + task + '.jsonl' for task in TASKS)):
            path = root / name; digest = sha256(path)
            require(previous['evidence_sha256'].get(str(path)) == digest, 'Previously scored400 control changed: ' + str(path))
            evidence[str(path)] = digest
    check_hashes(sources); check_hashes(evidence)
    config_path = output / 'config.json'
    record = {'schema': 'qwen_qk_rotation_full_retrieval_preparation_v1', 'status': 'prepared_unbound', 'gpu': None,
        'arm': arm, 'source_root': str(ROOT), 'python_executable': sys.executable, 'checkpoint_id': IDS[arm],
        'config_path': str(config_path), 'output_dir': str(output / 'run'), 'datasets': list(TASKS),
        'sample_count': 400, 'task_samples': {task: 200 for task in TASKS},
        'selection': 'All original archived rows0..199 per task, no filtering/resampling.',
        'input_lengths': {task: {'min': min(r['input_tokens'] for r in value), 'max': max(r['input_tokens'] for r in value)} for task, value in rows.items()},
        'source_sha256': sources, 'evidence_sha256': evidence, 'source_bit_budget': budget,
        'completed_candidate': {'directory': str(directory), 'checkpoint_dir': str(checkpoint), 'checkpoint_id': IDS[arm],
            'base_checkpoint_id': SOURCE_ID, 'full8_mean_percent': summary['mean_percent'], 'matched_pair': str(comparison),
            'replaced_projections': 72, 'preserved_projections': 180, 'native_union_tensors': 3423},
        'controls': controls, 'primary_comparison': 'full_minus_new_fresh_block32',
        'comparison_boundary': 'The newblock32 candidate is the matched causal control; BF16/f927 are immutable secondary references. No E2E decoder/norm audit is assumed for category recompression.',
        'environment': {**previous['environment'], 'PYTHONPATH': str(ROOT)},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config', str(config_path),
                          '--output', str(output / 'run'), '--datasets', *TASKS],
        'timeout_seconds': 900, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False, 'resume': False,
        'resource_evidence': {'same_protocol400_seconds': [418.60447883605957, 420.7395203113556, 424.63357615470886],
            'boundary': 'Actual L18/D800/DN400 native400 histories; QK full400 runtime not yet measured. Existing snapshot fullgate16 covered full-RHT load/group8 generation.'},
        'scoring': previous['scoring'], 'scope': 'Two complete official LongBench v1 retrieval tasks; only2/21, no overall LongBench or nine-item acceptance.',
        'full_longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'model_mutations': False, 'model_exported': False, 'cuda_initialized': torch.cuda.is_initialized()}
    require(not record['cuda_initialized'], 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(config_path, config); record['config_sha256'] = sha256(config_path)
    write_json(output / 'preparation.json', record)
    print({'status': record['status'], 'arm': arm, 'source_count': len(sources), 'evidence_count': len(evidence),
        'preparation_sha256': sha256(output / 'preparation.json'), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(IDS), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, default=PAIR / 'comparison.json')
    args = parser.parse_args()
    prepare(args.arm, args.output_dir, args.comparison)
