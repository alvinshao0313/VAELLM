"""CPU preparation of two existing D16/K24 Llama down-category settings."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_down_pair.compare import arm as old_arm
from experiments.full_rht_down_pair.prepare import source_geometry
from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _build_decoder_from_spec, load_v6_meta


CONTROL = RESULT / 'full_rht_down_pair/llama_block32'
EXPECTED_BITS = 17095180288


def recipe(template, output, block_size):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--codebook_bits': 'default=32,cat:down_proj=24',
                       '--codebook_dim': 'default=32,cat:down_proj=16',
                       '--weight_rotation_block_size': str(block_size)}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['down_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    for section in (expected['canonical_config'], expected['vae_args']):
        section['codebook_bits']['by_category']['down_proj'] = 24
        section['codebook_dim']['by_category'] = {'down_proj': 16}
    for section in (expected['canonical_config']['core_template'], expected['vae_args']):
        section['weight_rotation_block_size'] = block_size
    expected['resolved_category_runtime']['down_proj']['codebook_bits'] = 24
    expected['resolved_category_runtime']['down_proj']['codebook_dim'] = 16
    _require(normalized == expected,
             'Actual parser changed settings beyond output, down K/D and rotation block size.')
    _require(cat.target_layers == 'all' and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and training.bf16 and not training.fp16 and resolved['down_proj'].steps == 10000,
             'The inherited full-category native training recipe changed.')
    return argv, normalized


def estimate(meta, architecture, control_budget):
    source_geometry(meta, architecture)
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.down_proj')]
    first = specs[0]['stage_decoders'][0]
    _require(len(specs) == 32 and all(stage == first for s in specs for stage in s['stage_decoders']),
             'Expected 64 identical original down decoder specifications.')
    new = {**first, 'in_dim': 24, 'out_dim': 16}
    old_parameters = sum(p.numel() for p in _build_decoder_from_spec(first).parameters())
    new_parameters = sum(p.numel() for p in _build_decoder_from_spec(new).parameters())
    old_code_bits = sum(s['stage_vq_weights'][0]['logical_shape'][0] * 48 * 2 for s in specs)
    new_groups = 4096 * (14336 - 256) // 16
    new_code_bits = 32 * new_groups * 24 * 2
    removed = (old_parameters - new_parameters) * 64 * 16
    bits = control_budget['user_linear_bits'] - removed
    _require(old_parameters == 10656 and new_parameters == 5520 and removed == 5259264
             and new_groups == 3604480 and new_code_bits == old_code_bits
             and bits == EXPECTED_BITS and control_budget['original_linear_weights'] == 6979321856
             and 2 * bits < 5 * 6979321856, 'Actual source metadata/decoder budget differs.')
    return {'estimated_total_bits': bits, 'original_linear_weights': 6979321856,
        'estimated_bpw': bits / 6979321856, 'down_count': 32, 'preserved_projection_count': 192,
        'old_decoder_parameters_per_stage': old_parameters, 'new_decoder_parameters_per_stage': new_parameters,
        'decoder_bits_reduction': removed, 'estimated_decoder_bits': control_budget['components']['decoder']['storage_bits'] - removed,
        'old_and_new_down_code_bits': old_code_bits, 'new_groups_per_stage': new_groups,
        'other_storage_components_unchanged': True,
        'boundary': 'CPU estimate from real metadata and native decoder constructors; final assembled integer tensor audit remains required.'}


def prepare(block_size, output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and block_size in (32, 64)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template, audit, control, _, evidence = old_arm(CONTROL, 32)
    _require(template['model_type'] == 'llama' and template['categories'] == ['down_proj']
             and template['base_checkpoint_id'] == EXPECTED_IDS['llama'], 'Wrong completed down control.')
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == EXPECTED_IDS['llama'], 'Require fresh original 3577 assembly base.')
    budget = estimate(meta, template['architecture'], audit['bit_budget'])
    argv, normalized = recipe(template, output, block_size)
    sources = dict(template['source_sha256'])
    for path in [ROOT / 'experiments/full_rht_down_pair/compare.py',
                 ROOT / 'experiments/full_rht_gate_pair/compare.py',
                 *(ROOT / 'experiments/llama_down_dimension').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = {**template['input_sha256'], **evidence}
    parity_path = ORIGINAL / 'result/qwen3_8b_w25_20260930/validation/d16_fp32_packed_parity/run.json'
    parity = read(parity_path)
    _require(parity['status'] == 'passed' and parity['exit_code'] == 0
             and parity['case'] == {'K': 24, 'D': 16, 'H': 128, 'stages': 2,
                                   'parameter_dtype': 'float32', 'compute_dtype': 'bfloat16', 'group_size': 8},
             'The existing packed D16/K24 numerical proof differs.')
    for relative, digest in parity['source_sha256'].items():
        path = ROOT / relative
        _require(sha256(path) == digest, 'Existing packed proof source changed: ' + relative)
        sources[str(path)] = digest
    evidence[str(parity_path)] = sha256(parity_path)
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'categories': ['down_proj'], 'codebook_bits': {'down_proj': 24},
        'dimension': 16, 'architecture': template['architecture'], 'base_dir': str(base),
        'base_checkpoint_id': meta['checkpoint_id'], 'control_result_paths': template['control_result_paths'],
        'historical_control_dir': str(CONTROL), 'historical_control_checkpoint_id': control['checkpoint_id'],
        'historical_control_result_paths': [str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        'historical_control_audit_path': str(CONTROL / 'audit.json'),
        'block_size': block_size, 'steps_per_stage': 10000, 'full_evaluation': True,
        'expected_user_linear_bits': EXPECTED_BITS, 'budget_estimate': budget,
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': evidence, 'protected_stats': template['protected_stats'],
        'snapshot_import_paths': template['snapshot_import_paths'], 'environment': template['environment'],
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_down_dimension.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 'packed_d16_proof': str(parity_path),
            'existing_paths': 'Native per-category K/D parser, packed K24, D16 down and block32/64 paths already exercised. Historical control uses surviving meta/audit/normalized/raw only; its weights are not required or retrained.'},
        'scope': 'Fresh original-HF all32 down D16/K24/S2/H128R0/BF16/input256 INT8, MSE10Kx2/B8192/seed31; retain other192 and backbone from3577. No KD/LoRA.',
        'comparison_boundary': 'Primary versus completed fresh D32/K48/block32 down; secondary D16/block64 versus D16/block32. Fixed optimizer updates and vector batch; D16 samples half as many scalar weight values per update. Geometry changes initialization and block-shuffle RNG consumption, so equal seed is not an identical optimization trajectory.',
        'resource_estimate': {'seconds_limit': 2700, 'estimated_full_chain_seconds': [1500, 2100],
            'historical_control_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'uncertainty': 'Llama D16 full-category runtime is not yet measured; no wait/queue or automatic retry.'},
        'protocol': template['protocol'], 'cuda_initialized': False}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'block_size': block_size, 'status': record['status'], 'source_pins': len(sources),
           'input_pins': len(evidence), 'estimated_bpw': budget['estimated_bpw'],
           'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-size', type=int, choices=(32, 64), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.block_size, args.output_dir)
