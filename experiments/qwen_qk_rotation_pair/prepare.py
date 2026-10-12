"""CPU-bind a fresh Qwen Q/K block32 or full-RHT native CAT recipe."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_BITS, EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.full_rht_gate_pair.compare import normalized_recipe
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta

CATEGORIES = ['q_proj', 'k_proj']
BLOCKS = {'block32': 32, 'full': 0}
TEMPLATE = RESULT / 'full_rht_gate_pair/qwen3_block32/launch.json'


def recipe(template, output, block_size):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--compression_categories': ','.join(CATEGORIES),
                       '--weight_rotation_block_size': str(block_size)}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    _require(cat.compression_categories == 'q_proj,k_proj' and cat.target_layers == 'all'
             and cat.batch_size == 8192 and cat.linear_group_size == 36 and cat.seed == 31
             and cat.deterministic and cat.after_category_mode == 'none'
             and not cat.channel_refresh_after_category and cat.channel_axis == 'input'
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj'
             and training.bf16 and not training.fp16,
             'Native parser changed the fixed Q/K recipe.')
    for category, cfg in resolved.items():
        _require(cfg.steps == 10000 and cfg.codebook_bits == cfg.codebook_dim == 32
                 and cfg.residual_stages == 2 and cfg.base_ch == 128 and cfg.num_res_blocks == 0
                 and cfg.channel_protect_count == 64, 'Wrong resolved geometry: ' + category)
    return argv, normalized


def source_geometry(meta, budget):
    specs = [s for s in meta['converted_modules'] if s['name'].rsplit('.', 1)[-1] in CATEGORIES]
    _require(len(specs) == 72 and budget['checkpoint_id'] == meta['checkpoint_id']
             and budget['user_linear_bits'] == EXPECTED_BITS['qwen3']
             and budget['original_linear_weights'] == 6945767424
             and budget['code_padding_bits'] == 0, 'Unexpected measured f927 ledger.')
    for spec in specs:
        q = spec['name'].endswith('.q_proj'); out = 4096 if q else 1024
        groups = out * 4032 // 32
        _require(spec['in_features'] == 4096 and spec['out_features'] == out
                 and spec['compressed_in_features'] == 4032 and spec['compressed_out_features'] == out
                 and spec['transpose'] is q and spec['stage_codebook_dims'] == [32, 32]
                 and spec['residual_stages'] == 2 and spec['parallel_parts'] == 1
                 and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [64, out], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [64], 'dtype': 'bfloat16'}
                 and spec['protected_output_indices'] is None
                 and spec['weight_rotation']['block_size'] == 32, 'Wrong source Q/K geometry.')
        _require(len(spec['stage_vq_weights']) == len(spec['stage_decoders']) == 2, 'Wrong stage count.')
        for packed, decoder in zip(spec['stage_vq_weights'], spec['stage_decoders'], strict=True):
            _require(packed == {'shape': [groups, 1, 4], 'dtype': 'uint8', 'storage_format': 'bitpack_u8',
                     'logical_shape': [groups, 1, 32], 'logical_dtype': 'bool', 'pack_bits': 8}
                     and decoder['in_dim'] == decoder['out_dim'] == 32 and decoder['hidden_dim'] == 128
                     and decoder['num_res_blocks'] == 0 and decoder['param_dtype'] == 'bfloat16',
                     'Source code/decoder storage differs from this pair.')
    selected_bits = sum(int(budget['per_target'][s['name']]['budget_bits']) for s in specs)
    _require(selected_bits == 1604726784, 'Source Q/K storage changed.')
    return {'basis': 'Actual f927 metadata and measured integer ledger; candidate still needs native audit.',
        'replaced_projection_count': 72, 'preserved_projection_count': 180,
        'source_qk_bits': selected_bits, 'estimated_total_bits': budget['user_linear_bits'],
        'original_linear_weights': budget['original_linear_weights'],
        'estimated_bpw': budget['user_linear_bpw'], 'code_padding_bits': 0,
        'components': budget['components'],
        'reason': 'Same K/D/stages, BF16 decoder, protection and persistent sign element counts; Hadamard bases are nonpersistent framework constants.'}


def prepare(arm, output):
    output = Path(output).resolve(); block_size = BLOCKS[arm]
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template = read(TEMPLATE); check(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and template['model_type'] == 'qwen3' and template['block_size'] == 32
             and template['steps_per_stage'] == 10000 and template['full_evaluation'],
             'Require the completed reviewed Qwen native recipe.')
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS['qwen3'],
             'Assembly must retain the original f927 source.')
    ledger = RESULT / 'nine_task_baseline_1200/acceptance.json'
    budget = source_geometry(meta, read(ledger)['bit_budget'])
    validation = RESULT / 'full_rht_gate_pair/native_validation_r1/verification.json'
    proof = read(validation)
    _require(proof['status'] == 'verified' and proof['source_root'] == str(ROOT)
             and (validation.parent / 'run.exit').read_text().strip() == '0',
             'Require the existing native full-RHT proof.')
    check_hashes(proof['source_sha256'])
    o_review = RESULT / 'qwen_o_full_rht/independent_review.json'
    _require(read(o_review)['status'] == 'verified', 'Require the completed full 4096x4032 transpose path.')
    argv, normalized = recipe(template, output, block_size)
    _, paired_normalized = recipe(template, output, 32 if block_size == 0 else 0)
    _require(normalized_recipe({'normalized_expected': normalized, 'output_dir': str(output), 'block_size': block_size})
             == normalized_recipe({'normalized_expected': paired_normalized, 'output_dir': str(output),
                                   'block_size': 32 if block_size == 0 else 0}),
             'Native paired parsing differs beyond output and block-size mirrors.')
    sources = dict(template['source_sha256'])
    paths = [ROOT / 'experiments/full_rht_gate_pair/compare.py',
             *(ROOT / 'experiments/qwen_qk_rotation_pair').glob('*.py')]
    sources.update({str(p): sha256(p) for p in paths})
    history = ORIGINAL / 'result/qwen3_8b_w25_20260930/reconstruction_block128_p128_o0_d128/part_qk'
    timing_log = next(history.glob('*/linear_by_category.log'))
    inputs = dict(template['input_sha256'])
    inputs.update({str(p): sha256(p) for p in [TEMPLATE, ledger, validation, validation.parent / 'run.exit',
        o_review, o_review.parent / 'audit.json', o_review.parent / 'final_model/checkpoint_meta.json',
        timing_log, history / 'train.exit']})
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'qwen3', 'arm': arm, 'categories': CATEGORIES,
        'codebook_bits': {'q_proj': 32, 'k_proj': 32}, 'architecture': template['architecture'],
        'base_dir': str(base), 'base_checkpoint_id': meta['checkpoint_id'],
        'control_result_paths': template['control_result_paths'], 'block_size': block_size,
        'steps_per_stage': 10000, 'full_evaluation': True,
        'expected_user_linear_bits': EXPECTED_BITS['qwen3'], 'budget_estimate': budget,
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': template['protected_stats'],
        'snapshot_import_paths': template['snapshot_import_paths'], 'environment': template['environment'],
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.qwen_qk_rotation_pair.run',
                          '--launch-record', str(output / 'launch.json')],
        'category_rng': {'order': CATEGORIES, 'category_indices': [0, 1],
            'shuffle_seeds_by_stage': {'q_proj': [31, 32], 'k_proj': [100031, 100032]},
            'boundary': 'Same shapes/order/seeds across arms; no assertion of identical optimization trajectories.'},
        'validation_reuse': {'native_full_rht': str(validation), 'native_transpose_full': str(o_review),
            'no_new_gpu_short': True, 'historical_generated_weights_loaded': False},
        'scope': 'Fresh original-HF Q/K only, two categories in one CAT command; retain other180 and backbone from f927. No KD/LoRA or added FP32 deployment state.',
        'comparison_boundary': 'Full minus new fresh block32 is primary; f927 secondary. Reselect protection from the same original HF calibration, require actual pair protection/sign equality, do not impose f927 indices. The 12/16 QK oracle is not this method or a predicted gain.',
        'resource_estimate': {'historical_qk_cat_seconds': 1334.486, 'historical_condition': 'D32/K32/input128/block128, not this exact geometry.',
            'estimated_training_assembly_full8_seconds': [1850, 2160],
            'uncertainty': 'Full Q/K runtime is unmeasured; 2700 seconds is a resource bound, not a completion guarantee.'},
        'protocol': {**template['protocol'], 'longbench': 'not_evaluated', 'nine_item_goal': 'not_evaluated'},
        'future_retrieval': {'prepared': False, 'gpu': None, 'automatic_queue': False,
            'tasks': ['passage_retrieval_en', 'passage_retrieval_zh'], 'rows': 400,
            'timeout_seconds': 900, 'termination_grace_seconds': 20,
            'boundary': 'Separately bind official native protocol after full8; reuse BF16/f927 predictions. Only2/21 tasks.'},
        'cuda_initialized': False}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'arm': arm, 'status': record['status'], 'sources': len(sources), 'inputs': len(inputs),
           'estimated_bpw': budget['estimated_bpw'], 'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(BLOCKS), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); prepare(args.arm, args.output_dir)
