"""Prepare fresh Llama gate D16/K16 at the existing two-bit code rate."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.full_rht_gate_pair.compare import arm as historical_arm
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _build_decoder_from_spec, load_v6_meta


CONTROL = RESULT / 'full_rht_gate_pair/llama_block32'
CONTROL_ID = '0ad677ce-a49a-423e-a20c-50f8aa1ef710'
EXPECTED_BITS = 17096228864
PARITY = ORIGINAL / 'result/qwen3_8b_w25_20260930/validation/d16_packed_parity/launch.json'


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--codebook_bits': 'default=32,cat:gate_proj=16',
                       '--codebook_dim': 'default=32,cat:gate_proj=16'}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    for field in ('codebook_bits', 'codebook_dim'):
        for section in ('canonical_config', 'vae_args'):
            expected[section][field]['by_category'] = {'gate_proj': 16}
        expected['resolved_category_runtime']['gate_proj'][field] = 16
    _require(normalized == expected, 'Actual parser differs beyond output and gate K/D.')
    cfg = resolved['gate_proj']
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
             and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.channel_axis == 'input' and cat.channel_quant == 'int8'
             and training.bf16 and not training.fp16 and cfg.steps == 10000
             and cfg.codebook_bits == cfg.codebook_dim == 16 and cfg.base_ch == 128
             and cfg.channel_protect_count == 64 and cfg.residual_stages == 2
             and cfg.num_res_blocks == 0 and cfg.decoder_type == 'symmetric',
             'The inherited complete gate-category recipe changed.')
    return argv, normalized


def estimate(meta, budget):
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    _require(meta['checkpoint_id'] == budget['checkpoint_id'] == CONTROL_ID and len(specs) == 32
             and budget['original_linear_weights'] == 6979321856
             and budget['user_linear_bits'] == 17100439552, 'Require the completed fresh D32 gate control.')
    first = specs[0]['stage_decoders'][0]
    for spec in specs:
        _require(spec['in_features'] == 4096 and spec['out_features'] == 14336
                 and spec['compressed_in_features'] == 4032 and spec['compressed_out_features'] == 14336
                 and not spec['transpose'] and spec['parallel_parts'] == 1 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32] and spec['weight_rotation']['block_size'] == 32
                 and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['stage_decoders'] == [first, first]
                 and all(p['shape'] == [1806336, 1, 4] and p['logical_shape'] == [1806336, 1, 32]
                         for p in spec['stage_vq_weights']), 'Historical gate geometry differs.')
    _require(first['in_dim'] == first['out_dim'] == 32 and first['hidden_dim'] == 128
             and first['num_res_blocks'] == 0 and first['param_dtype'] == 'bfloat16',
             'Historical decoder specification differs.')
    with torch.random.fork_rng(devices=[]):
        old_count = sum(p.numel() for p in _build_decoder_from_spec(first).parameters())
        new_count = sum(p.numel() for p in _build_decoder_from_spec({**first, 'in_dim': 16, 'out_dim': 16}).parameters())
    groups = 14336 * 4032 // 16
    old_code = sum(s['stage_vq_weights'][0]['logical_shape'][0] * 32 * 2 for s in specs)
    new_code = groups * 16 * 2 * 32
    removed = (old_count - new_count) * 32 * 2 * 16
    components = {key: value['storage_bits'] for key, value in budget['components'].items()}
    components['decoder'] -= removed; total = sum(components.values())
    _require(old_count == 8608 and new_count == 4496 and removed == 4210688
             and old_code == new_code == 3699376128 and groups == 3612672
             and 64 % 16 == 0 and 4032 % 32 == 14336 % 32 == 0
             and total == EXPECTED_BITS and 2 * total < 5 * 6979321856
             and budget['code_padding_bits'] == 0, 'Actual native decoder or packed storage calculation differs.')
    return {'estimated_total_bits': total, 'original_linear_weights': 6979321856,
        'estimated_bpw': total / 6979321856, 'estimated_component_bits': components,
        'old_decoder_parameters_per_stage': old_count, 'new_decoder_parameters_per_stage': new_count,
        'decoder_bits_reduction': removed, 'old_and_new_gate_code_bits': new_code,
        'new_groups_per_stage': groups, 'gate_count': 32, 'preserved_projection_count': 192,
        'boundary': 'Exact integer structure from actual control metadata/audit and native decoder constructors; final assembled tensor storage still requires audit. All main codes and protection/sign storage remain unchanged in size.'}


def prepare(output):
    output = Path(output).resolve(); feasibility = output / 'feasibility.json'
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and output.is_dir() and set(output.iterdir()) == {feasibility},
             'Use the CPU-only snapshot and a directory containing only the retained feasibility record.')
    prior = read(feasibility)
    _require(prior['exit_code'] == 0 and prior['result']['status'] == 'verified_readonly_cpu'
             and prior['result']['candidate_total_bits'] == EXPECTED_BITS
             and prior['result']['cuda_initialized'] is False, 'Require the original read-only CPU feasibility result.')
    template, audit, control, _, evidence = historical_arm(CONTROL, 32)
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(template['model_type'] == 'llama' and control['checkpoint_id'] == CONTROL_ID
             and meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS['llama'],
             'Use original3577 assembly and the completed fresh D32 gate control.')
    budget = estimate(load_v6_meta(str(CONTROL / 'final_model')), audit['bit_budget'])
    argv, normalized = recipe(template, output)
    proof = read(PARITY)
    _require(proof['status'] == 'passed' and proof['pytest_exit_code'] == proof['run_exit_code'] == 0
             and (PARITY.parent / 'run.exit').read_text().strip() == '0'
             and proof['command'][-2:] == [
                 'tests/test_fused_multistage_decoder_parity.py::test_d16_h128_bf16_native_and_group8_keep_gradients[16]',
                 'tests/test_fused_multistage_decoder_parity.py::test_d16_h128_bf16_native_and_group8_keep_gradients[24]'],
             'The existing K16/D16/BF16 two-stage group8 proof differs.')
    sources = dict(template['source_sha256'])
    for relative, digest in proof['source_sha256'].items():
        path = Path(relative)
        if path.is_absolute():
            path = path.relative_to(ORIGINAL)
        path = ROOT / path
        _require(sha256(path) == digest, 'Packed D16 proof source changed: ' + str(path))
        sources[str(path)] = digest
    for path in [ROOT / 'experiments/full_rht_gate_pair/compare.py',
                 *(ROOT / 'experiments/llama_gate_dimension').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = {**template['input_sha256'], **evidence}
    for path in (feasibility, PARITY, PARITY.parent / 'run.exit'):
        evidence[str(path)] = sha256(path)
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'categories': ['gate_proj'], 'codebook_bits': {'gate_proj': 16},
        'dimension': 16, 'hidden_dim': 128, 'block_size': 32, 'protected_input_count': 64,
        'architecture': template['architecture'], 'base_dir': str(base), 'base_checkpoint_id': meta['checkpoint_id'],
        'control_result_paths': template['control_result_paths'], 'historical_control_dir': str(CONTROL),
        'historical_control_checkpoint_id': CONTROL_ID,
        'historical_control_result_paths': [str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        'historical_control_audit_path': str(CONTROL / 'audit.json'),
        'steps_per_stage': 10000, 'full_evaluation': True, 'expected_user_linear_bits': EXPECTED_BITS,
        'budget_estimate': budget, 'training_argv': [sys.executable, 'tools/cat_train.py', *argv],
        'normalized_expected': normalized, 'source_sha256': sources, 'input_sha256': evidence,
        'protected_stats': template['protected_stats'], 'snapshot_import_paths': template['snapshot_import_paths'],
        'environment': template['environment'], 'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_dimension.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 'packed_d16_proof': str(PARITY),
            'existing_paths': 'Native per-category D16/K16 parser, two-stage BF16 packed/fused/group8 parity, original Qwen D16 gate checkpoint deployment and unchanged Llama native category assembly. Historical control uses retained metadata/raw/audit; no retired control weights are required.'},
        'scope': 'Fresh original-HF all32 gate D16/K16/S2/H128R0/BF16/block32/input64 INT8, MSE10Kx2/B8192/seed31. Preserve other192 and backbone from3577. Original global BSQ regularizer; no KD/LoRA.',
        'comparison_boundary': 'Primary versus completed fresh D32/K32 gate0ad; original3577 secondary. K and D halve together at fixed two-bit main rate. Fixed vector batch halves sampled scalar elements; decoder capacity and initialization/shuffle RNG consumption differ. This is not an equal-FLOP or identical-trajectory comparison, and no local-loss or downstream improvement is assumed.',
        'resource_estimate': {'seconds_limit': 2700,
            'historical_control_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'historical_control_stage_seconds': {name: item['finished_unix'] - item['started_unix']
                                               for name, item in template['stages'].items()},
            'uncertainty': 'Same32-module native category recipe and existing D16 deployment evidence; candidate wall time remains unmeasured. No queue or automatic retry.'},
        'protocol': template['protocol'], 'cuda_initialized': False}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(evidence),
           'estimated_bpw': budget['estimated_bpw'], 'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
