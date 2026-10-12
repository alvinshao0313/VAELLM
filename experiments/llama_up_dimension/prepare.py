"""CPU-only D16/K16 up preparation; reuse the completed matched D32 control."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_up_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE_ID, DENOMINATOR, check, read, _require
from experiments.llama_gate_up_rate_pair.compare import completed
from experiments.llama_up_dimension.audit import CONTROL_ID, TOTAL_BITS
from experiments.longbench_v1.common import sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import _build_decoder_from_spec, load_v6_meta


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Use a fresh CPU-only snapshot preparation.')
    control_dir = RESULT / 'llama_up_rotation_pair/block32'
    template, audit, control, _, evidence = completed(control_dir, 'control', expected_bits={'up_proj': 32})
    _require(control['checkpoint_id'] == CONTROL_ID and template['base_checkpoint_id'] == BASE_ID
             and template['categories'] == ['up_proj'] and template['block_size'] == 32,
             'Require the completed fresh up block32 control in e767 context.')
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'), '--codebook_bits': 'default=32,cat:up_proj=16',
                       '--codebook_dim': 'default=32,cat:up_proj=16'}.items():
        _require(argv.count(key) == 1, 'Ambiguous native argument: ' + key); argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['up_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'): expected[section]['output_dir'] = str(output / 'donor')
    for field in ('codebook_bits', 'codebook_dim'):
        for section in ('canonical_config', 'vae_args'): expected[section][field] = {'default': 32, 'by_category': {'up_proj': 16}}
        expected['resolved_category_runtime']['up_proj'][field] = 16
    _require(normalized == expected, 'Native recipe differs beyond up K/D and output.')
    cfg = resolved['up_proj']
    _require(cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.seed == 31 and cat.deterministic
             and cat.target_layers == 'all' and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint
             and not cat.channel_refresh_after_category and cat.channel_axis == 'input' and cat.channel_quant == 'int8'
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj' and training.bf16 and not training.fp16
             and cfg.steps == 10000 and cfg.codebook_bits == cfg.codebook_dim == 16 and cfg.residual_stages == 2
             and cfg.base_ch == 128 and cfg.num_res_blocks == 0 and cfg.channel_protect_count == 64, 'Wrong native candidate.')
    meta = load_v6_meta(str(control_dir / 'final_model')); budget = audit['bit_budget']
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.up_proj')]
    first = specs[0]['stage_decoders'][0]
    _require(len(specs) == 32 and all(s['stage_decoders'] == [first, first] for s in specs), 'Wrong control decoders.')
    with torch.random.fork_rng(devices=[]):
        old = sum(p.numel() for p in _build_decoder_from_spec(first).parameters())
        new = sum(p.numel() for p in _build_decoder_from_spec({**first, 'in_dim': 16, 'out_dim': 16}).parameters())
    removed = (old - new) * 64 * 16; groups = 14336 * 4032 // 16
    components = deepcopy(budget['components'])
    for field in ('budget_bits', 'storage_bits'): components['decoder'][field] -= removed
    _require(old == 8608 and new == 4496 and removed == 4210688 and groups == 3612672
             and groups * 16 * 2 * 32 == sum(s['stage_vq_weights'][0]['logical_shape'][0] * 32 * 2 for s in specs)
             and 64 % 16 == 4032 % 32 == 14336 % 32 == 0
             and sum(v['storage_bits'] for v in components.values()) == budget['user_linear_bits'] - removed == TOTAL_BITS,
             'Native integer geometry estimate differs.')
    sources = dict(template['source_sha256']); inputs = {**template['input_sha256'], **evidence}
    parity = ORIGINAL / 'result/qwen3_8b_w25_20260930/validation/d16_packed_parity/launch.json'
    proof = read(parity)
    _require(proof['status'] == 'passed' and proof['pytest_exit_code'] == proof['run_exit_code'] == 0
             and (parity.parent / 'run.exit').read_text().strip() == '0', 'Existing native D16 proof is incomplete.')
    for relative, digest in proof['source_sha256'].items():
        p = ROOT / relative; _require(sha256(p) == digest, 'Existing packed proof source changed: ' + relative)
        sources[str(p)] = digest
    for p in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'Public/snapshot source differs.')
    for p in (parity, parity.parent / 'run.exit', RESULT / 'llama_gate_dimension/review.json',
              RESULT / 'llama_down_dimension/comparison.json', control_dir / 'launch.prepared.json',
              RESULT / 'llama_up_rotation_pair/comparison.json'):
        inputs[str(p)] = sha256(p)
    dense_bytes = read(Path(meta['base_model_path']) / 'model.safetensors.index.json')['metadata']['total_size']
    names = {s['name'] for s in specs}; up_bits = sum(budget['per_target'][n]['budget_bits'] for n in names) - removed
    donor_bytes = int(dense_bytes - 32 * 14336 * 4096 * 2 + up_bits // 8)
    final_bytes = budget['full_model_tensor_bytes'] - removed // 8
    provision = 2 * donor_bytes + final_bytes + 2 * 1024**3
    record = {'schema': 'llama_fresh_up_d16_v1', 'status': 'prepared_unbound', 'gpu': None,
        'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(output), 'model_type': 'llama', 'categories': ['up_proj'], 'codebook_bits': {'up_proj': 16},
        'dimension': 16, 'transpose': False, 'block_size': 32, 'steps_per_stage': 10000,
        'base_dir': template['base_dir'], 'base_checkpoint_id': BASE_ID, 'architecture': template['architecture'],
        'historical_control_dir': str(control_dir), 'historical_control_checkpoint_id': CONTROL_ID,
        'control_result_paths': template['control_result_paths'], 'full_evaluation': True,
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': template['protected_stats'],
        'environment': template['environment'], 'protocol': template['protocol'], 'expected_user_linear_bits': TOTAL_BITS,
        'expected_budget_components': components,
        'budget_estimate': {'bits': TOTAL_BITS, 'original_linear_weights': DENOMINATOR, 'bpw': TOTAL_BITS / DENOMINATOR,
            'decoder_parameters_per_stage': [old, new], 'decoder_bits_reduction': removed,
            'new_groups_per_stage': groups, 'up_code_bits_unchanged': groups * 16 * 2 * 32, 'padding_bits': 0,
            'boundary': 'Native constructor/actual control geometry estimate; final assembled storage audit required.'},
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_up_dimension.run',
                          '--launch-record', str(output / 'launch.json')],
        'comparison_boundary': 'Primary candidate versus completed fresh d086 D32/K32 up, both with e767 other192/backbone including D16 down. Fixed2bit main code rate; B8192 fixed vector batch halves scalar exposure and changes decoder capacity/initialization/shuffle consumption. Same seed does not give an identical optimization trajectory or equal FLOPs. e767 is secondary; no KD/LoRA or LongBench/nine-item inference.',
        'validation_reuse': {'new_gpu_short': False, 'packed_d16_proof': str(parity),
            'scope': 'Existing D16/K16 BF16 native/group8 parity, Llama gate same14336x4032 geometry, D16 down and complete up/e767 native assembly.'},
        'resource_estimate': {'estimated_seconds': [1200, 1500], 'seconds_limit': 2700,
            'historical_up_d32_seconds': template['finished_unix'] - template['started_unix'],
            'historical_gate_d16_seconds': read(RESULT / 'llama_gate_dimension/review.json')['elapsed_seconds'],
            'single_arm_disk_provision_bytes': provision,
            'checkpoint_tensor_bytes': {'after_up_proj': donor_bytes, 'final_model': donor_bytes, 'assembled': final_bytes},
            'filesystem_free_at_preparation': shutil.disk_usage(output.parent.parent).free}, 'cuda_initialized': False}
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(output.parent.parent).free > provision,
             'CPU preparation initialized CUDA or insufficient output disk.')
    check(record); output.mkdir(parents=True)
    write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'source': len(sources),
           'input': len(inputs), 'stat': len(record['protected_stats']), 'bpw': TOTAL_BITS / DENOMINATOR,
           'disk_bytes': provision, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
