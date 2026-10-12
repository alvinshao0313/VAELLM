"""Prepare the existing Llama CAT recipe with only the requested category order changed."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'result/llama31_8b_w25_20261004/remaining_prefix_block32_s500'
ORDER = ['k_proj', 'v_proj', 'q_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj']


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(argv):
    cat, _, training, vae = parse_cat_runtime_args(argv)
    configs = resolve_category_runtime_configs(cat, vae, cat.compression_categories.split(','))
    return build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=configs)


def check(record):
    for group in ('source_sha256', 'input_sha256', 'model_metadata_sha256'):
        check_hashes(record[group])
    for item in record['input_data_stat'] + record['source_weight_stat']:
        check_stat(item)


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd().resolve() == ROOT and not output.exists(), 'Use a fresh CPU-only preparation in the repository.')
    control = read(CONTROL / 'launch.json')
    require(control['status'] == 'trained_pending_evaluation' and control['exit_code'] == 0
            and control['source']['all_official_weights_verified'], 'Require the completed original-HF control.')
    check_hashes(control['source_sha256']); check_hashes(control['input_evidence_sha256'])
    check_hashes(control['model_metadata_sha256'])
    for item in control['input_data_stat'] + control['source_weight_stat']:
        check_stat(item)
    require(sha256(control['script']) == control['script_sha256'], 'Original CAT shell changed.')
    old_prepared = read(CONTROL / 'normalized_prepared_args.json')
    actual_path = Path(control['final_checkpoint_dir']).parent / 'normalized_cat_runtime_args.json'
    actual = read(actual_path)
    original_argv = control['argv'][2:]
    require(normalized(original_argv) == old_prepared, 'Current native parser no longer reproduces the original recipe.')
    original_actual = copy.deepcopy(old_prepared)
    original_actual['cat_args']['output_dir'] = str(actual_path.parent)
    require(actual == original_actual, 'Original actual recipe differs beyond its native timestamped output.')
    argv = list(original_argv)
    for key, value in {'--output_dir': str(output), '--compression_categories': ','.join(ORDER)}.items():
        require(argv.count(key) == 1, 'Ambiguous original argument: ' + key)
        argv[argv.index(key) + 1] = value
    candidate = normalized(argv)
    expected = copy.deepcopy(old_prepared)
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output)
    expected['canonical_config']['compression_categories'] = ORDER
    expected['cat_args']['compression_categories'] = ','.join(ORDER)
    require(candidate == expected, 'Only output and the two order mirrors may differ.')
    require('--resume_from_checkpoint' not in argv and candidate['cat_args']['resume_from_checkpoint'] is None
            and candidate['cat_args']['transpose_modules'] == 'q_proj,v_proj,o_proj,down_proj'
            and candidate['canonical_config']['after_category_mode'] == 'remaining_lora_prefix_decoder',
            'Use fresh HF, original transpose and native merged-LoRA/prefix-decoder recovery.')
    validation_path = Path(control['diagnostic_audit']); validation = read(validation_path)
    require(validation['status'] == 'verified' and validation['diagnostic_not_formal_initial']
            and validation['quality_evaluated'] is False, 'The existing real seven-category Llama proof is required.')
    final_acceptance = CONTROL.parent / 'remaining_prefix_block32_s500_full8/acceptance.json'
    final = read(final_acceptance)
    require(final['checkpoint_id'] == '35776715-2707-45a9-9fb4-2da1efef3f0e', 'Wrong matched original control.')
    inputs = {**control['input_evidence_sha256'], **{str(p): sha256(p) for p in (
        CONTROL / 'launch.json', CONTROL / 'run.sh', CONTROL / 'normalized_prepared_args.json', actual_path,
        validation_path, final_acceptance, CONTROL.parent / 'remaining_prefix_block32_s500_full8/independent_review.json',
        Path(control['source']['source_manifest']), Path(control['final_checkpoint_dir']) / 'checkpoint_meta.json')}}
    for section in ('baseline', 'compressed'):
        for row in final[section]['input_results']:
            require(sha256(row['path']) == row['sha256'], 'A paired evaluation input changed.')
            inputs[row['path']] = row['sha256']
    sources = dict(control['source_sha256'])
    # Reuse the unchanged, already exercised native evaluation dependency set.
    evaluation_sources = read(CONTROL.parent / 'remaining_prefix_stage_mmlu/preparation.json')['source_sha256']
    check_hashes(evaluation_sources); sources.update(evaluation_sources)
    extra = list(Path(__file__).parent.glob('*.py')) + [ROOT / p for p in (
        'experiments/full_rht_gate_pair/compare.py', 'experiments/full_rht_gate_pair/audit.py',
        'experiments/full_rht_gate_pair/common.py', 'experiments/qwen3_w25/assemble_categories.py',
        'experiments/qwen3_w25/summarize_results.py', 'experiments/w25_eval/acceptance.py',
        'experiments/longbench_v1/common.py')]
    sources.update({str(p): sha256(p) for p in extra})
    record = {'schema': 'llama_fresh_cat_order_v1', 'status': 'prepared', 'gpu': None, 'queued': False,
        'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': candidate,
        'categories': ORDER, 'control_categories': old_prepared['canonical_config']['compression_categories'],
        'architecture': final['architecture'], 'control_final_acceptance': str(final_acceptance),
        'source_sha256': sources, 'input_sha256': inputs, 'model_metadata_sha256': control['model_metadata_sha256'],
        'input_data_stat': control['input_data_stat'], 'source_weight_stat': control['source_weight_stat'],
        'hf_source': control['source'], 'environment': control['environment'], 'geometry': control['geometry'],
        'expected_user_linear_bits': 17100439552, 'original_linear_weights': 6979321856,
        'estimated_bpw': 17100439552 / 6979321856,
        'timeout_seconds': 54000, 'termination_grace_seconds': 20,
        'resource_estimate': 'Original complete CAT required 9h54m; allocate one full GPU, never a callback gap. New order duration is not yet measured.',
        'pair_contract': {'original_46_training_source_hashes_unchanged': True,
            'current_native_parser_matches_control_preparation': True,
            'actual_control_diff_only_timestamped_cat_output': True,
            'candidate_changes_only_order_and_output_mirrors': True,
            'same_seed_does_not_mean_same_category_rng': 'Category positions, remaining dense targets, quantization inputs after earlier fusion, and recovery trajectories change as consequences of order.'},
        'validation_reuse': {'path': str(validation_path), 'new_gpu_short': False,
            'scope': 'Existing seven-category native Llama CAT, remaining dense rank12 fusion, prefix decoder recovery, final rank0/strict save and evaluation paths; no new algorithm or loader.'},
        'boundary_retention': 'Retain all seven native after_category checkpoints plus final_model for the explicitly planned category-trajectory comparison. Native CLI already saves these; no new save behavior.',
        'protocol': {'num_fewshot': 0, 'limit': None, 'mmlu_batch_size': 64, 'seven_batch_size': 32,
            'leaf_count': 64, 'sample_count': 24742, 'full8_after_native_train': True,
            'goal_acceptance': False, 'longbench_status': 'not_evaluated'}, 'cuda_initialized': False}
    check(record)
    output.mkdir(parents=True)
    write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'status': 'prepared', 'sha256': sha256(output / 'launch.prepared.json'), 'source_count': len(sources),
           'input_count': len(inputs), 'stats': len(record['input_data_stat']) + len(record['source_weight_stat']),
           'estimated_bpw': record['estimated_bpw'], 'cuda_initialized': torch.cuda.is_initialized()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    prepare(parser.parse_args().output_root)
