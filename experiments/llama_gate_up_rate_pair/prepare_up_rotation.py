"""Prepare the single-up block32/full pair in the fixed recovered e767 context."""
import argparse
import ast
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_up_rate_pair.common import (ROOT, ORIGINAL, RESULT, BASE, BASE_ID,
    TOTAL_BITS, DENOMINATOR, check, read, _require)
from experiments.llama_gate_up_rate_pair.prepare import recipe
from experiments.full_rht_gate_pair.compare import arm, normalized_recipe
from experiments.longbench_v1.common import file_stat, sha256, write_json


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Use a fresh CPU-only snapshot preparation.')
    template_dir = RESULT / 'full_rht_gate_pair/llama_block32'
    template, _, _, _, evidence = arm(template_dir, 32)
    meta = read(BASE / 'checkpoint_meta.json'); source_review = BASE.parent / 'review.json'
    review = read(source_review); budget = review['bit_budget']
    _require(meta['checkpoint_id'] == review['checkpoint_id'] == BASE_ID and review['status'] == 'verified'
             and meta['base_model_path'] == template['architecture']['base_model_path']
             and (BASE / meta['state_dict_file']).is_file()
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
             and budget['original_linear_weights'] == DENOMINATOR and budget['code_padding_bits'] == 0,
             'Require the retained e767 source and its completed native budget proof.')
    specs = {s['name']: s for s in meta['converted_modules']}
    names = {n for n in specs if n.endswith('.up_proj')}
    _require(len(specs) == 224 and len(names) == 32, 'Incomplete e767 projection inventory.')
    for name in names:
        s = specs[name]
        _require(not s['transpose'] and [s['out_features'], s['in_features']] == [14336, 4096]
                 and [s['compressed_out_features'], s['compressed_in_features']] == [14336, 4032]
                 and s['stage_codebook_dims'] == [32, 32] and s['parallel_parts'] == 1
                 and s['residual_stages'] == 2 and s['weight_rotation']['block_size'] == 32
                 and s['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and s['protected_channel_quant_format'] == 'int8'
                 and all(p['logical_shape'] == [1806336, 1, 32] and p['shape'] == [1806336, 1, 4]
                         for p in s['stage_vq_weights'])
                 and all(d['in_dim'] == d['out_dim'] == 32 and d['hidden_dim'] == 128
                         and d['num_res_blocks'] == 0 and d['param_dtype'] == 'bfloat16'
                         for d in s['stage_decoders']), 'Unexpected source up geometry: ' + name)
    down = [s for n, s in specs.items() if n.endswith('.down_proj')]
    _require(len(down) == 32 and all(s['stage_codebook_dims'] == [16, 16] and s['transpose']
             and s['protected_input_indices']['shape'] == [256]
             and all(d['in_dim'] == 24 for d in s['stage_decoders']) for s in down), 'e767 down differs.')
    validation_path = RESULT / 'full_rht_gate_pair/native_validation_r1/verification.json'
    validation = read(validation_path)
    _require(validation['status'] == 'verified' and validation['gate_models']['llama']['status'] == 'verified'
             and validation['gate_models']['llama']['gate_count'] == 32
             and '252' in validation['factor_records'], 'Missing existing same-geometry native full-RHT proof.')
    archive = RESULT / 'llama_gate_up_rate_pair/source_before_up_rotation_expansion/manifest.json'
    inputs = {**template['input_sha256'], **evidence}
    extra = [source_review, BASE / 'checkpoint_meta.json', BASE / 'config.json', BASE / 'assembly_report.json',
             *[BASE.parent / (n + '.json') for n in ('mmlu', 'seven')], validation_path, archive,
             RESULT / 'full_rht_gate_pair/llama_comparison.json',
             RESULT / 'full_rht_gate_pair/llama_full/launch.json',
             RESULT / 'llama_q_up_transpose/up_false/launch.json',
             RESULT / 'llama_q_up_transpose/up_pair_review.json',
             RESULT / 'llama_gate_up_rate_pair/comparison.json',
             RESULT / 'llama_d1200_retrieval_mlp_attribution/independent_review.json']
    inputs.update({str(p): sha256(p) for p in extra})
    sources = dict(template['source_sha256'])
    for p in [*sorted(Path(__file__).parent.glob('*.py')),
              *[ROOT / q for q in ('experiments/full_rht_gate_pair/compare.py',
                 'experiments/rotation_factor_probe/capture.py', 'experiments/w25_eval/acceptance.py')]]:
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'Public/snapshot source differs.')
    stats = [s for s in template['protected_stats'] if Path(s['path']).parent != Path(template['base_dir'])]
    stats.append(file_stat(BASE / meta['state_dict_file']))
    times = {label: read(RESULT / name) for label, name in {
        'up_block32': 'llama_q_up_transpose/up_false/launch.json',
        'gate_full_same_geometry': 'full_rht_gate_pair/llama_full/launch.json'}.items()}
    times = {label: {k: v['finished_unix'] - v['started_unix'] for k, v in row['stages'].items()}
             for label, row in times.items()}
    source_storage = read(BASE / 'assembly_report.json')['bit_budget']
    dense_bytes = read(Path(meta['base_model_path']) / 'model.safetensors.index.json')['metadata']['total_size']
    packed_bytes = sum(source_storage['per_target'][n]['budget_bits'] for n in names) // 8
    donor_bytes = int(dense_bytes - 32 * 14336 * 4096 * 2 + packed_bytes)
    provision = 2 * donor_bytes + source_storage['full_model_tensor_bytes'] + 2 * 1024**3
    records = {}
    for label, block in (('block32', 32), ('full', 0)):
        directory = output / label; selected = 'control' if block == 32 else 'candidate'
        argv, normalized = recipe(template, directory, selected, up_rotation_block=block)
        record = {'schema': 'llama_fresh_up_rotation_pair_v1', 'status': 'prepared_unbound', 'gpu': None,
            'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
            'output_dir': str(directory), 'arm': selected, 'model_type': 'llama', 'categories': ['up_proj'],
            'codebook_bits': {'up_proj': 32}, 'dimension': 32, 'transpose': False,
            'block_size': block, 'steps_per_stage': 10000, 'base_dir': str(BASE), 'base_checkpoint_id': BASE_ID,
            'architecture': template['architecture'],
            'control_result_paths': [str(BASE.parent / (n + '.json')) for n in ('mmlu', 'seven')],
            'training_argv': argv, 'normalized_expected': normalized, 'source_sha256': sources,
            'input_sha256': inputs, 'protected_stats': stats, 'environment': template['environment'],
            'protocol': template['protocol'], 'full_evaluation': True, 'expected_user_linear_bits': TOTAL_BITS,
            'expected_budget_components': budget['components'],
            'budget_estimate': {'bits': TOTAL_BITS, 'original_linear_weights': DENOMINATOR,
                'bpw': TOTAL_BITS / DENOMINATOR, 'padding_bits': 0,
                'scope': 'Aligned main codes/decoder/protection/signs retain sizes; actual final audit required.'},
            'timeout_seconds': 2700, 'termination_grace_seconds': 20,
            'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_up_rate_pair.run',
                              '--launch-record', str(directory / 'launch.json')],
            'seed_boundary': {'global_seed': 31, 'category_order': ['up_proj'], 'stage_shuffle_seeds': [[31, 32]],
                'initialization': 'Independent native seed31 single-category processes; same geometry and random-call dimensions, but no historical unsaved RNG snapshots are claimed.',
                'calibration': 'Original HF Alpaca512x256 seed31 once before up; no refresh. Compare actual512 inputs, stats, protection and signs across completed arms.'},
            'comparison_boundary': 'Primary fresh full minus fresh block32; only rotation block size changes. Both replace e767 recovered up by fresh HF encoding, retain192 other projections and all backbone including D16 down; e767 is secondary. No KD/LoRA. No full21/nine-item inference.',
            'experience': 'Uncovered Llama up full RHT plus two recovered cases in fixed16 conditional BF16-up diagnosis motivate exploration only. Earlier upTrue block32 lost0.26939pp in3577 context; gate40/up24 lost2.48792pp in e767 context. Neither oracle nor gate full-RHT proves up benefit.',
            'validation_reuse': {'new_gpu_short': False, 'same_geometry_full_gate': str(validation_path),
                'existing_paths': 'Native H252/full14336x4032, single-up10Kx2, strict architecture-aware assembler2979, complete64-leaf evaluation.'},
            'resource_estimate': {'historical_stage_seconds': times, 'estimated_seconds': [900, 1500],
                'seconds_limit': 2700, 'single_arm_disk_provision_bytes': provision,
                'checkpoint_tensor_bytes': {'after_up_proj': donor_bytes, 'final_model': donor_bytes,
                                            'assembled_final_model': source_storage['full_model_tensor_bytes']},
                'boundary': 'Up block32 previously1173s; same-geometry full gate1224s. Cold scheduling/hardware variation remains.',
                'filesystem_free_at_preparation': shutil.disk_usage(output.parent).free}, 'cuda_initialized': False}
        check(record); records[label] = record
    _require(normalized_recipe(records['block32']) == normalized_recipe(records['full'])
             and not torch.cuda.is_initialized(), 'Unexpected paired recipe difference or CUDA initialization.')
    _require(shutil.disk_usage(output.parent).free > 2 * provision, 'Insufficient disk for both independently retained arms.')
    for label, record in records.items():
        directory = output / label; directory.mkdir(parents=True)
        write_json(directory / 'launch.prepared.json', record); write_json(directory / 'launch.json', record)
        print({'arm': label, 'prepared_sha256': sha256(directory / 'launch.prepared.json'),
               'source': len(sources), 'input': len(inputs), 'stat': len(stats),
               'disk_bytes': provision, 'estimated_bpw': TOTAL_BITS / DENOMINATOR}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    prepare(parser.parse_args().output_root)
