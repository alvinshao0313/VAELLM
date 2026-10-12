"""CPU preparation only; both arms remain unbound until separately scheduled."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_up_rate_pair.common import (ROOT, ORIGINAL, RESULT, BASE, BASE_ID,
    CATEGORIES, BITS, TOTAL_BITS, DENOMINATOR, check, read, paired_recipe, _require)
from experiments.full_rht_gate_pair.compare import arm as completed_gate
from experiments.longbench_v1.common import file_stat, sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload


def recipe(template, output, arm, *, up_rotation_block=None):
    _require(up_rotation_block is None or up_rotation_block in (0, 32), 'Only the declared up rotation pair is supported.')
    categories = CATEGORIES if up_rotation_block is None else ['up_proj']
    bits = BITS[arm] if up_rotation_block is None else {'up_proj': 32}
    argv = list(template['training_argv'][2:])
    changes = {'--output_dir': str(output / 'donor'), '--compression_categories': ','.join(categories),
               '--codebook_bits': 'default=32,' + ','.join(f'cat:{c}={bits[c]}' for c in categories)}
    if up_rotation_block is not None:
        changes['--weight_rotation_block_size'] = str(up_rotation_block)
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous native argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, categories)
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    expected['canonical_config']['compression_categories'] = categories
    expected['cat_args']['compression_categories'] = ','.join(categories)
    for section in ('canonical_config', 'vae_args'):
        expected[section]['codebook_bits'] = {'default': 32, 'by_category': bits}
    if up_rotation_block is not None:
        expected['canonical_config']['core_template']['weight_rotation_block_size'] = up_rotation_block
        expected['vae_args']['weight_rotation_block_size'] = up_rotation_block
    cfg = deepcopy(expected['resolved_category_runtime']['gate_proj'])
    expected['resolved_category_runtime'] = {}
    for category in categories:
        expected['resolved_category_runtime'][category] = dict(cfg, category=category,
                                                              codebook_bits=bits[category])
    _require(normalized == expected, 'Native recipe changed beyond declared category, K, rotation block, and output.')
    _require(cat.seed == 31 and cat.deterministic and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.target_layers == 'all' and cat.after_category_mode == 'none'
             and not cat.channel_refresh_after_category and not cat.resume_from_checkpoint
             and cat.activation_calib_dataset == 'alpaca=1' and cat.activation_calib_nsamples == 512
             and cat.activation_calib_seqlen == 256 and cat.activation_calib_seed == 31
             and training.bf16 and not training.fp16
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj', 'Native scientific recipe differs.')
    for category, cfg in resolved.items():
        _require(cfg.steps == 10000 and cfg.codebook_dim == 32 and cfg.residual_stages == 2
                 and cfg.base_ch == 128 and cfg.num_res_blocks == 0 and cfg.channel_protect_count == 64
                 and cfg.codebook_bits == bits[category], 'Resolved category configuration differs.')
    return [sys.executable, 'tools/cat_train.py', *argv], normalized


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Use one fresh CPU-only snapshot preparation.')
    template_dir = RESULT / 'full_rht_gate_pair/llama_block32'
    template, _, _, _, evidence = completed_gate(template_dir, 32)
    meta = read(BASE / 'checkpoint_meta.json'); review_path = BASE.parent / 'review.json'
    reference = read(review_path); budget = reference['bit_budget']
    _require(meta['checkpoint_id'] == reference['checkpoint_id'] == BASE_ID
             and reference['status'] == 'verified' and reference['mean_percent'] == 62.12036318745571
             and meta['base_model_path'] == template['architecture']['base_model_path']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
             and budget['original_linear_weights'] == DENOMINATOR and budget['code_padding_bits'] == 0,
             'Require the actual e767 D16-down/FineWeb1600 source.')
    specs = {s['name']: s for s in meta['converted_modules']}
    changes = {}
    for category in CATEGORIES:
        selected = [s for s in specs.values() if s['name'].endswith('.' + category)]
        _require(len(selected) == 32, 'Incomplete source category.')
        code = decoder = 0
        for s in selected:
            _require(not s['transpose'] and s['in_features'] == 4096 and s['out_features'] == 14336
                     and s['compressed_in_features'] == 4032 and s['compressed_out_features'] == 14336
                     and s['stage_codebook_dims'] == [32, 32] and s['parallel_parts'] == 1
                     and s['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                     and s['weight_rotation']['block_size'] == 32, 'Unexpected source geometry.')
            for bank, dec in zip(s['stage_vq_weights'], s['stage_decoders'], strict=True):
                _require(bank['logical_shape'] == [1806336, 1, 32] and bank['shape'] == [1806336, 1, 4]
                         and dec['in_dim'] == dec['out_dim'] == 32 and dec['hidden_dim'] == 128
                         and dec['num_res_blocks'] == 0 and dec['param_dtype'] == 'bfloat16', 'Wrong packed geometry.')
                change = BITS['candidate'][category] - 32
                code += 1806336 * change; decoder += 128 * change * 16
        changes[category] = {'code_bits': code, 'decoder_bits': decoder}
    _require(sum(sum(v.values()) for v in changes.values()) == 0, 'Allocation must preserve total actual geometry.')
    down = [s for s in specs.values() if s['name'].endswith('.down_proj')]
    _require(len(down) == 32 and all(s['stage_codebook_dims'] == [16, 16] and s['transpose']
             and s['protected_input_indices']['shape'] == [256]
             and all(d['in_dim'] == 24 for d in s['stage_decoders']) for s in down), 'e767 down geometry differs.')
    sources = dict(template['source_sha256'])
    for path in [*sorted(Path(__file__).parent.glob('*.py')),
                 *[ROOT / p for p in ('experiments/full_rht_gate_pair/compare.py',
                    'experiments/rotation_factor_probe/capture.py', 'experiments/w25_eval/acceptance.py')]]:
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
        _require(sha256(ORIGINAL / path.relative_to(ROOT)) == sha256(path), 'Public/snapshot source differs.')
    inputs = {**template['input_sha256'], **evidence}
    extra = [template_dir / 'launch.json', review_path, BASE / 'checkpoint_meta.json', BASE / 'config.json',
             BASE / 'assembly_report.json', *[BASE.parent / (n + '.json') for n in ('mmlu', 'seven')],
             RESULT / 'llama_q_up_transpose/up_false/launch.json',
             RESULT / 'llama_gate_down_balanced_bits/combined/independent_review.json',
             RESULT / 'best1200_fresh_gate_up_k40_24/launch.json',
             ORIGINAL / 'result/qwen3_8b_w25_20260930/validation/mixed_k24_k56_d32_bf16/validation.json']
    inputs.update({str(p): sha256(p) for p in extra})
    stats = [*template['protected_stats'], file_stat(BASE / meta['state_dict_file'])]
    up = read(RESULT / 'llama_q_up_transpose/up_false/launch.json')
    times = {label: {k: v['finished_unix'] - v['started_unix'] for k, v in record['stages'].items()}
             for label, record in (('gate', template), ('up', up))}
    source_storage = read(BASE / 'assembly_report.json')['bit_budget']
    dense_bytes = read(Path(meta['base_model_path']) / 'model.safetensors.index.json')['metadata']['total_size']
    records = {}
    for arm in BITS:
        directory = output / arm; argv, normalized = recipe(template, directory, arm)
        remaining = dense_bytes; checkpoints = {}
        for category in CATEGORIES:
            names = [n for n in specs if n.endswith('.' + category)]
            packed = sum(source_storage['per_target'][n]['budget_bits'] for n in names)
            if arm == 'candidate': packed += sum(changes[category].values())
            remaining += int(packed / 8) - 32 * 14336 * 4096 * 2
            checkpoints['after_' + category] = remaining
        checkpoints['final_model'] = remaining
        provision = sum(checkpoints.values()) + source_storage['full_model_tensor_bytes'] + 2 * 1024**3
        record = {'schema': 'llama_fresh_gate_up_rate_pair_v1', 'status': 'prepared_unbound', 'gpu': None,
            'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
            'output_dir': str(directory), 'arm': arm, 'model_type': 'llama', 'categories': CATEGORIES,
            'codebook_bits': BITS[arm], 'dimension': 32, 'block_size': 32, 'steps_per_stage': 10000,
            'base_dir': str(BASE), 'base_checkpoint_id': BASE_ID, 'architecture': template['architecture'],
            'control_result_paths': [str(BASE.parent / (n + '.json')) for n in ('mmlu', 'seven')],
            'training_argv': argv, 'normalized_expected': normalized, 'source_sha256': sources,
            'input_sha256': inputs, 'protected_stats': stats, 'environment': template['environment'],
            'protocol': template['protocol'], 'full_evaluation': True, 'expected_user_linear_bits': TOTAL_BITS,
            'expected_budget_components': budget['components'],
            'budget_estimate': {'bits': TOTAL_BITS, 'original_linear_weights': DENOMINATOR,
                'bpw': TOTAL_BITS / DENOMINATOR, 'candidate_delta': changes, 'padding_bits': 0,
                'scope': 'Metadata arithmetic; final native tensor storage audit required. No external LoRA.'},
            'timeout_seconds': 3600, 'termination_grace_seconds': 20,
            'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_up_rate_pair.run',
                              '--launch-record', str(directory / 'launch.json')],
            'seed_boundary': {'global_seed': 31, 'category_order': CATEGORIES,
                'stage_shuffle_seeds': [[31, 32], [100031, 100032]],
                'initialization': 'Unchanged native global seed and RNG flow; K changes random consumption, including later up initialization.',
                'calibration': 'Each arm collects original HF Alpaca512x256 seed31 once before either category; refresh=false. Native CLI has no separate calibration-cache import. Actual caches/stats/protection/signs compared after completion.'},
            'comparison_boundary': 'Primary candidate minus new fresh control. Both replace recovered e767 gate/up with fresh HF encodings; e767 is secondary, not a matched allocation control. Preserve other160 projections/backbone, including32 D16K24 down. No new KD/LoRA, no resume identity changes.',
            'experience': 'Llama gate/down2.5 allocation gained1.26690pp in a different fresh pair; Qwen gate40/up24 lost1.74061pp. BF16 class sensitivity does not determine finite-rate gains. No guaranteed69 or LongBench/nine-item conclusion.',
            'validation_reuse': 'Native Llama gate/up nontranspose full8, Qwen K40/K24 native two-category packed chain, and existing strict architecture-generic assembler; no new algorithm/kernel/GPU short.',
            'resource_estimate': {'historical_seconds': times, 'estimated_seconds': [1500, 2100],
                'boundary': 'Historical gate train760s/up725s plus assembly43s/full8~406s gives~1934s, with duplicated initialization/calibration; two-category K40/24 runtime not yet measured.',
                'checkpoint_tensor_bytes': checkpoints, 'single_arm_disk_provision_bytes': provision,
                'filesystem_free_at_preparation': shutil.disk_usage(output.parent).free}, 'cuda_initialized': False}
        check(record); records[arm] = record
    _require(paired_recipe(records['control']) == paired_recipe(records['candidate'])
             and not torch.cuda.is_initialized(), 'Pair differs beyond intended K allocation/output or initialized CUDA.')
    for arm, record in records.items():
        directory = output / arm; directory.mkdir(parents=True)
        write_json(directory / 'launch.prepared.json', record); write_json(directory / 'launch.json', record)
        print({'arm': arm, 'prepared_sha256': sha256(directory / 'launch.prepared.json'),
               'source': len(sources), 'input': len(inputs), 'stat': len(stats),
               'disk_bytes': record['resource_estimate']['single_arm_disk_provision_bytes']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    prepare(parser.parse_args().output_root)
