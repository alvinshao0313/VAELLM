"""Prepare only the missing full-eight shards of existing Llama CAT stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from train_utils.checkpoint_v6 import validate_v6_meta

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'result/llama31_8b_w25_20261004'
STAGES = ('after_o_proj', 'after_gate_proj', 'after_up_proj', 'down_round_base')
IDENTITIES = ('7ea8108f-ff1d-4b9b-bc45-8db1b7f1cec4', '89044995-d6b4-4bc3-a832-440f9144e423',
              '6449d4db-4b14-4928-a3b1-05320c0fc3e2', 'bc6f897d-195b-40f8-8b84-6c2507102922')
SEVEN = 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(record):
    check_hashes(record['source_sha256'])
    check_hashes(record['input_sha256'])
    for item in record['checkpoints'].values():
        check_stat(item['weight_stat'])
        require(read(Path(item['path']) / 'checkpoint_meta.json')['checkpoint_id'] == item['checkpoint_id'],
                'Existing stage identity changed.')


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Prepare on CPU only.')
    require(not output.exists(), 'Use a fresh output directory.')
    previous_path = OLD / 'remaining_prefix_stage_mmlu/preparation.json'
    previous = read(previous_path)
    check_hashes(previous['source_sha256'])
    trajectory = Path(previous['checkpoints']['after_o_proj']['path']).parent
    final_acceptance = OLD / 'remaining_prefix_block32_s500_full8/acceptance.json'
    final = read(final_acceptance)
    inputs = {str(p): sha256(p) for p in (previous_path, final_acceptance,
        OLD / 'remaining_prefix_stage_mmlu/comparison.json',
        OLD / 'remaining_prefix_block32_s500_full8/independent_review.json',
        trajectory / 'normalized_cat_runtime_args.json')}
    checkpoints, commands = {}, []
    from tools.cat_eval import build_parser
    for index, (name, expected_id) in enumerate(zip(STAGES, IDENTITIES, strict=True)):
        directory = trajectory / name if name != 'down_round_base' else Path(previous['checkpoints'][name]['path'])
        meta_path = directory / 'checkpoint_meta.json'
        meta = validate_v6_meta(read(meta_path), expected_kind='round_base' if name == 'down_round_base' else 'category_boundary')
        count = 128 + 32 * index
        require(meta['checkpoint_id'] == expected_id and len(meta['compressed_targets']) == count
                and len(meta['pending_dense_targets']) == 224 - count and meta['lora_config'] is None,
                'Unexpected native stage topology or identity.')
        weight = directory / meta['state_dict_file']
        require(weight.is_file() and not weight.is_symlink(), 'The actual stage weight must still exist.')
        state = torch.load(weight, weights_only=True, map_location='cpu', mmap=True)
        require(len(state) == count * 13 + (224 - count) + 67
                and not any('lora_' in key for key in state), 'Unexpected serialized stage inventory.')
        del state
        reused = None if name == 'after_gate_proj' else OLD / 'remaining_prefix_stage_mmlu' / name / 'mmlu.json'
        if reused is not None:
            raw = read(reused); lm = raw['evals']['lm_eval']; old = previous['checkpoints'][name]
            require(raw['checkpoint_id'] == expected_id and raw['strict_load'] and raw['checkpoint_loader'] == 'v6'
                    and lm['num_fewshot'] == 0 and lm['limit'] is None and lm['batch_size'] == 64
                    and len(lm['n_samples']) == 57 and sum(v['effective'] for v in lm['n_samples'].values()) == 14042,
                    'Reused complete MMLU identity/protocol differs.')
            require(file_stat(weight) == {'path': str(weight), **old['state_stat']}
                    and sha256(meta_path) == old['metadata_sha256'], 'Existing MMLU source changed.')
            inputs[str(reused)] = sha256(reused)
        checkpoints[name] = {'path': str(directory), 'checkpoint_id': expected_id,
            'checkpoint_kind': meta['checkpoint_kind'], 'compressed_count': count, 'pending_dense_count': 224 - count,
            'weight_stat': file_stat(weight), 'mmlu_path': str(reused or output / name / 'mmlu.json'),
            'seven_path': str(output / name / 'seven.json'), 'reused_complete_mmlu': reused is not None}
        for p in (meta_path, directory / 'config.json'):
            inputs[str(p)] = sha256(p)
        for shard in (('mmlu', 'seven') if reused is None else ('seven',)):
            argv = [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir', str(directory), '--strict', '--eval_lm_eval',
                    '--eval_device', 'cuda:0', '--tasks', 'mmlu' if shard == 'mmlu' else SEVEN,
                    '--num_fewshot', '0', '--lm_batch_size', '64' if shard == 'mmlu' else '32',
                    '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
                    '--eval_log_dir', str(output / name / ('logs_' + shard)), '--output_json', str(output / name / (shard + '.json'))]
            build_parser().parse_args(argv[2:])
            commands.append({'name': name + '_' + shard, 'stage': name, 'shard': shard, 'argv': argv})
    for group in ('baseline', 'compressed'):
        for item in final[group]['input_results']:
            require(sha256(item['path']) == item['sha256'], 'Existing final/baseline raw changed.')
            inputs[item['path']] = item['sha256']
    inputs[str(Path(final['checkpoint_dir']) / 'checkpoint_meta.json')] = sha256(Path(final['checkpoint_dir']) / 'checkpoint_meta.json')
    sources = dict(previous['source_sha256'])
    extra = list(Path(__file__).parent.glob('*.py')) + [ROOT / path for path in (
        'experiments/full_rht_gate_pair/compare.py', 'experiments/full_rht_gate_pair/audit.py',
        'experiments/full_rht_gate_pair/common.py', 'experiments/qwen3_w25/assemble_categories.py',
        'experiments/qwen3_w25/summarize_results.py', 'experiments/w25_eval/acceptance.py',
        'experiments/longbench_v1/common.py')]
    sources.update({str(p): sha256(p) for p in extra})
    record = {'schema': 'llama_existing_prefix_full8_v1', 'status': 'prepared', 'gpu': None, 'queued': False,
        'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'architecture': final['architecture'], 'checkpoints': checkpoints, 'stage_commands': commands,
        'source_sha256': sources, 'input_sha256': inputs, 'final_acceptance_path': str(final_acceptance),
        'environment': {**previous['environment'], 'TRANSFORMERS_OFFLINE': '1'},
        'timeout_seconds': 1200, 'termination_grace_seconds': 20,
        'protocol': {'leaf_count': 64, 'samples': 24742, 'mmlu_subjects': 57, 'mmlu_samples': 14042,
                     'num_fewshot': 0, 'limit': None, 'mmlu_batch_size': 64, 'seven_batch_size': 32,
                     'chat_template': False, 'backend': 'sdpa', 'rope_dtype': 'torch.float32', 'prewarm_group_size': 8},
        'validation_reuse': 'Three unchanged stage weights already passed native mixed-dense strict full MMLU. The gate stage has the same native category-boundary topology; no new loader or GPU short test.',
        'scope': 'Existing cumulative CAT stages, not isolated category effects or deployment candidates; no training, export, or ninth-task acceptance.',
        'cuda_initialized': False}
    check(record)
    output.mkdir(parents=True)
    for name in STAGES:
        (output / name).mkdir()
    write_json(output / 'launch.prepared.json', record)
    write_json(output / 'launch.json', record)
    print(json.dumps({'status': 'prepared', 'commands': len(commands), 'source_pins': len(sources),
        'input_pins': len(inputs), 'sha256': sha256(output / 'launch.prepared.json'), 'cuda_initialized': torch.cuda.is_initialized()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    prepare(parser.parse_args().output_root)
