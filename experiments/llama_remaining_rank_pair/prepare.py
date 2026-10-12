"""Prepare matched native after-o continuations, changing only after-gate LoRA rank."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_order_control.prepare import CONTROL, ROOT, check, normalized, read, require
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity, restore_cat_runtime_state, validate_cat_runtime_identity
from train_utils.cat_train_runtime import load_cat_resume_distill_progress, resolve_cat_resume_source
from train_utils.checkpoint_v6 import load_v6_cat_runtime_state
from train_utils.config.targets import parse_skip_layers, parse_target_layers

ORDER = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
SOURCE_ID = '7ea8108f-ff1d-4b9b-bc45-8db1b7f1cec4'


def recipe(original, old, output, boundary, rank):
    argv = list(original)
    for key, value in {'--output_dir': str(output), '--lora_rank': f'default=12,after:gate_proj={rank}',
                       '--lora_alpha': f'default=24,after:gate_proj={rank * 2}'}.items():
        require(argv.count(key) == 1, 'Ambiguous original argument: ' + key)
        argv[argv.index(key) + 1] = value
    require('--resume_from_checkpoint' not in argv, 'Original control must start from HF.')
    argv += ['--resume_from_checkpoint', str(boundary)]
    value = normalized(argv); expected = deepcopy(old)
    for section in ('canonical_config', 'cat_args'):
        expected[section].update(output_dir=str(output), resume_from_checkpoint=str(boundary))
    expected['canonical_config']['explicit_cli_flags'] = sorted(
        expected['canonical_config']['explicit_cli_flags'] + ['--resume_from_checkpoint'])
    expected['canonical_config']['lora_rank']['by_after_category'] = {'gate_proj': rank}
    expected['canonical_config']['lora_alpha']['by_after_category'] = {'gate_proj': float(rank * 2)}
    require(value == expected, 'Resume recipe differs beyond output, source and after-gate rank/alpha.')
    cat, _, _, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ORDER)
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=resolved, compression_categories=ORDER,
        target_layers=parse_target_layers(cat.target_layers), skip_layers=parse_skip_layers(cat.skip_layers),
        transpose_modules=cat.transpose_modules.split(','))
    stages = {category: asdict(cat.resolve_after_category_config(category)) for category in ORDER}
    old_cat, _, _, _ = parse_cat_runtime_args(original)
    for category in ORDER:
        reference = asdict(old_cat.resolve_after_category_config(category))
        if category == 'gate_proj':
            reference['lora'].update(rank=rank, alpha=float(rank * 2))
        require(stages[category] == reference, 'Resolved recovery recipe changed: ' + category)
    return argv, value, identity, stages


def prepare(output):
    output = Path(output).resolve()
    require(Path.cwd().resolve() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not output.exists(), 'Use fresh CPU-only public preparation.')
    original = read(CONTROL / 'launch.json')
    require(original['status'] == 'trained_pending_evaluation' and original['exit_code'] == 0,
            'Require the completed original native CAT trajectory.')
    for key in ('source_sha256', 'input_evidence_sha256', 'model_metadata_sha256'):
        check_hashes(original[key])
    for item in original['input_data_stat'] + original['source_weight_stat']:
        check_stat(item)
    require(sha256(original['script']) == original['script_sha256'], 'Original script changed.')
    old = read(CONTROL / 'normalized_prepared_args.json'); argv = original['argv'][2:]
    require(normalized(argv) == old, 'Current native parser differs from the original recipe.')
    native = Path(original['final_checkpoint_dir']).parent
    actual = deepcopy(old); actual['cat_args']['output_dir'] = str(native)
    require(read(native / 'normalized_cat_runtime_args.json') == actual, 'Original actual recipe differs.')
    boundary = native / 'after_o_proj'; source = resolve_cat_resume_source(str(boundary))
    meta = source.model_checkpoint_meta
    require(source.source_kind == source.model_checkpoint_kind == 'category_boundary'
            and source.model_checkpoint_dir == str(boundary) and source.training_step_meta is None
            and meta['checkpoint_id'] == SOURCE_ID and meta['completed_categories'] == ORDER[:4]
            and len(meta['compressed_targets']) == 128 and len(meta['pending_dense_targets']) == 96
            and meta['lora_config'] is None and meta['base_model_path'] == original['source']['model_path']
            and sha256(boundary / 'checkpoint_meta.json') == 'b2d2243ebb487ee527664e1c61ebacddfb2911377601d5b44d0746fbd2071b35',
            'Require the actual after-o boundary with original HF teacher identity.')
    weight = boundary / meta['state_dict_file']; runtime = boundary / 'cat_runtime_state.pt'
    require(weight.stat().st_size == 13799867229 and runtime.stat().st_size == 21384882,
            'Source weight/runtime sizes differ from retained evidence.')
    progress = load_cat_resume_distill_progress(str(boundary))
    require(list(progress.completed_categories) == ORDER[:4] and progress.lora_round_idx == 4
            and progress.active_category is None and progress.training_step_checkpoint is None
            and list(progress.distill_stage_history) == meta['extra_meta']['distill_stage_history'],
            'Boundary resume must skip four completed categories and start fresh gate recovery.')
    payload = load_v6_cat_runtime_state(str(boundary), required=True)
    activation, plan, saved_identity = restore_cat_runtime_state(payload)
    require(activation is not None and plan is None and activation['cache'].nsamples == 512
            and len(activation['cache'].input_ids) == 512 and len(activation['stats_by_linear']) == 224,
            'Require the original full activation/channel planning state.')
    acceptance_path = CONTROL.parent / 'remaining_prefix_block32_s500_full8/acceptance.json'
    acceptance = read(acceptance_path)
    require(acceptance['checkpoint_id'] == '35776715-2707-45a9-9fb4-2da1efef3f0e', 'Wrong historical endpoint.')
    stage_proof = CONTROL.parent / 'remaining_prefix_stage_mmlu/preparation.json'
    sources = {**original['source_sha256'], **read(stage_proof)['source_sha256']}
    check_hashes(sources)
    for path in [*Path(__file__).parent.glob('*.py'), ROOT / 'experiments/llama_order_control/prepare.py',
                 *[ROOT / p for p in ('experiments/full_rht_gate_pair/compare.py', 'experiments/full_rht_gate_pair/audit.py',
                     'experiments/full_rht_gate_pair/common.py', 'experiments/qwen3_w25/assemble_categories.py',
                     'experiments/qwen3_w25/summarize_results.py', 'experiments/longbench_v1/common.py')]]:
        sources[str(path)] = sha256(path)
    inputs = dict(original['input_evidence_sha256'])
    for path in [CONTROL / 'launch.json', CONTROL / 'run.sh', CONTROL / 'normalized_prepared_args.json',
                 native / 'normalized_cat_runtime_args.json', acceptance_path, stage_proof,
                 CONTROL.parent / 'remaining_prefix_block32_s500_full8/independent_review.json',
                 native / 'final_model/checkpoint_meta.json', boundary / 'checkpoint_meta.json', runtime,
                 Path(original['diagnostic_audit'])]:
        inputs[str(path)] = sha256(path)
    for group in ('baseline', 'compressed'):
        for row in acceptance[group]['input_results']:
            require(sha256(row['path']) == row['sha256'], 'Historical full-eight evidence changed.')
            inputs[row['path']] = row['sha256']
    records = {}
    for rank in (12, 56):
        arm = output / f'rank{rank}'
        train, expected, identity, stages = recipe(argv, old, arm, boundary, rank)
        validate_cat_runtime_identity(saved_identity, identity)
        records[rank] = {'schema': 'llama_after_o_remaining_rank_v1', 'status': 'prepared_unbound', 'gpu': None,
            'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
            'output_dir': str(arm), 'training_argv': [sys.executable, 'tools/cat_train.py', *train],
            'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_remaining_rank_pair.run',
                              '--launch-record', str(arm / 'launch.json')],
            'normalized_expected': expected, 'resolved_recovery_stages': stages, 'after_gate_rank': rank,
            'categories': ORDER, 'new_categories': ORDER[4:], 'architecture': acceptance['architecture'],
            'source_boundary': str(boundary), 'source_checkpoint_id': SOURCE_ID,
            'source_distill_history': meta['extra_meta']['distill_stage_history'], 'saved_runtime_identity': identity,
            'control_final_acceptance': str(acceptance_path), 'source_sha256': sources, 'input_sha256': inputs,
            'model_metadata_sha256': original['model_metadata_sha256'], 'input_data_stat': original['input_data_stat'],
            'source_weight_stat': [*original['source_weight_stat'], file_stat(weight), file_stat(runtime)],
            'environment': original['environment'], 'expected_user_linear_bits': 17100439552,
            'original_linear_weights': 6979321856, 'estimated_bpw': 17100439552 / 6979321856,
            'timeout_seconds': 39600, 'termination_grace_seconds': 20,
            'resource_estimate': {'historical_gate_up_down_seconds': 25179,
                'new_arm_duration': 'Approximately 7 hours plus initialization/full8; rank56 not yet measured; 11-hour bound.',
                'per_arm_peak_generated_disk_bytes_estimate': 38000000000,
                'pair_disk_bytes_estimate': 76000000000, 'filesystem_free_at_preparation': shutil.disk_usage(output.parent).free,
                'gate_lora_parameter_count': 64 * (14336 + 4096) * rank,
                'rank56_added_fp32_parameter_gradient_adam_bytes_estimate': 64 * (14336 + 4096) * (56 - 12) * 16,
                'boundary': 'One dedicated A800 per arm. Rank56 has not had a GPU resource test; the added optimizer estimate excludes activation/workspace effects.'},
            'resume_evidence': {'source_kind': source.source_kind, 'native_identity_validated': True,
                'native_runtime_payload_loaded': True, 'calibration_samples': 512, 'calibration_linears': 224,
                'new_seed': 31, 'exact_continuous_rng_resume': False,
                'boundary': 'Native category-boundary restore preserves activation/channel planning and model state, but does not restore global RNG/old optimizer. Both arms restart seed31 at the same boundary. Original3577 is historical; rank12 continuation is the primary matched control.'},
            'protocol': {'num_fewshot': 0, 'limit': None, 'mmlu_batch_size': 64, 'seven_batch_size': 32,
                'leaf_count': 64, 'sample_count': 24742, 'full8_after_native_train': True,
                'goal_acceptance': False, 'longbench_status': 'not_evaluated'},
            'pair_contract': 'Only after:gate_proj temporary dense LoRA rank12/alpha24 versus rank56/alpha112. After up/down use rank12/alpha24; down has no remaining dense targets. All branches are merged; final external rank0. Initialization RNG consumption can differ as a consequence of rank.',
            'validation_reuse': {'path': original['diagnostic_audit'], 'new_gpu_short': False,
                'scope': 'Existing native seven-category Llama CAT/merged plain LoRA/prefix decoder/save+strict full8; actual CPU boundary and runtime identity verified here. No new algorithm, schema, or loader.'},
            'experience': 'Checkpoint lifecycle requires saved runtime state and real source identity; same seed boundary continuation is not identical to uninterrupted training. Rank56 historical DoRA results do not establish plain-LoRA gains.',
            'cuda_initialized': False}
        check(records[rank])
    a, b = [deepcopy(records[r]['normalized_expected']) for r in (12, 56)]
    for value in (a, b):
        for section in ('canonical_config', 'cat_args'):
            value[section]['output_dir'] = '<arm>'
        for key in ('lora_rank', 'lora_alpha'):
            value['canonical_config'][key]['by_after_category']['gate_proj'] = '<rank-variable>'
    require(a == b and not torch.cuda.is_initialized(), 'Pair differs beyond intended rank/alpha or initialized CUDA.')
    for rank, record in records.items():
        arm = Path(record['output_dir']); arm.mkdir(parents=True)
        write_json(arm / 'launch.prepared.json', record); write_json(arm / 'launch.json', record)
        print({'rank': rank, 'prepared_sha256': sha256(arm / 'launch.prepared.json'),
               'source_count': len(sources), 'input_count': len(inputs), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    prepare(parser.parse_args().output_root)
