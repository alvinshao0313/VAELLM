"""Prepare one native Llama down-only input-protection comparison on CPU."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check
from experiments.llama_down_dimension.compare import arm as historical_arm
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta


CONTROL = RESULT / 'llama_down_dimension/block32'
CONTROL_ID = '8a4633a7-93bd-429a-8a0e-a519a2ae30ae'
EXPECTED_BITS = 17263542272


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--channel_protect_count': 'default=512'}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['down_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    expected['canonical_config']['channel_protect_count_table'] = {'default': 512}
    expected['cat_args']['channel_protect_count'] = {'default': 512}
    expected['resolved_category_runtime']['down_proj']['channel_protect_count'] = 512
    _require(normalized == expected, 'Actual parser differs beyond output and input protection count.')
    cfg = resolved['down_proj']
    _require(cat.compression_categories == 'down_proj' and cat.target_layers == 'all'
             and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.channel_axis == 'input' and cat.channel_quant == 'int8'
             and training.bf16 and not training.fp16 and cfg.steps == 10000
             and cfg.codebook_bits == 24 and cfg.codebook_dim == 16 and cfg.base_ch == 128
             and cfg.channel_protect_count == 512
             and cfg.num_res_blocks == 0 and cfg.decoder_type == 'symmetric',
             'The inherited complete down-category recipe changed.')
    return argv, normalized


def estimate(meta, budget):
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.down_proj')]
    _require(meta['checkpoint_id'] == CONTROL_ID and len(specs) == 32
             and budget['checkpoint_id'] == CONTROL_ID
             and budget['original_linear_weights'] == 6979321856
             and budget['user_linear_bits'] == 17095180288, 'Require the completed D16 protection256 control.')
    components = {key: value['storage_bits'] for key, value in budget['components'].items()}
    changes = dict.fromkeys(components, 0)
    for spec in specs:
        _require(spec['in_features'] == 14336 and spec['out_features'] == 4096
                 and spec['compressed_in_features'] == 14080 and spec['compressed_out_features'] == 4096
                 and spec['transpose'] and spec['parallel_parts'] == 1 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [16, 16] and spec['weight_rotation']['block_size'] == 32
                 and spec['protected_input_indices'] == {'shape': [256], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [256, 4096], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [256], 'dtype': 'bfloat16'},
                 'Historical projection geometry or protection differs.')
        _require(all(p['logical_shape'] == [3604480, 1, 24] and p['shape'] == [3604480, 1, 3]
                     for p in spec['stage_vq_weights'])
                 and all(d['in_dim'] == 24 and d['out_dim'] == 16 and d['hidden_dim'] == 128
                         and d['num_res_blocks'] == 0 and d['param_dtype'] == 'bfloat16'
                         for d in spec['stage_decoders']), 'Historical stages differ.')
        additional = 512 - 256
        changes['code'] -= 4096 * additional // 16 * 24 * 2
        changes['protected_weights'] += additional * 4096 * 8
        changes['protected_scales'] += additional * 16
        changes['protected_indices'] += additional * 64
        changes['rotation'] -= additional * 8
    updated = {key: value + changes[key] for key, value in components.items()}
    groups = 4096 * (14336 - 512) // 16
    total = sum(updated.values())
    _require(groups == 3538944 and 512 % 16 == 0 and (14336 - 512) % 32 == 0
             and total == EXPECTED_BITS and 2 * total < 5 * 6979321856
             and changes['decoder'] == 0 and budget['code_padding_bits'] == 0,
             'Exact metadata-derived storage or block alignment differs.')
    return {'estimated_total_bits': total, 'original_linear_weights': 6979321856,
        'estimated_bpw': total / 6979321856, 'estimated_component_bits': updated,
        'component_delta_bits': changes, 'new_groups_per_stage': groups,
        'protected_input_count': 512, 'down_count': 32, 'preserved_projection_count': 192,
        'boundary': 'Integer structural estimate from actual control tensor audit and native metadata. Decoder unchanged; packed codes, INT8 values, BF16 scales, INT64 indices and INT8 rotation signs counted. Final saved model tensor audit remains required.'}


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template, audit, control, _, evidence = historical_arm(CONTROL, 32)
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(template['model_type'] == 'llama' and template['dimension'] == 16
             and control['checkpoint_id'] == CONTROL_ID
             and meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS['llama'],
             'Use original3577 for assembly and completed8a for the matched quality control.')
    budget = estimate(load_v6_meta(str(CONTROL / 'final_model')), audit['bit_budget'])
    argv, normalized = recipe(template, output)
    sources = dict(template['source_sha256'])
    for path in [ROOT / 'experiments/llama_down_dimension/compare.py',
                 *(ROOT / 'experiments/llama_down_protection').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = {**template['input_sha256'], **evidence}
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'categories': ['down_proj'], 'codebook_bits': {'down_proj': 24},
        'dimension': 16, 'hidden_dim': 128, 'block_size': 32, 'protected_input_count': 512,
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
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_down_protection.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 'source_control': str(CONTROL),
            'existing_paths': 'Use the completed native D16/K24/B8192/H128/block32 training, INT8 protection and save/fresh evaluation chain; changing this existing count preserves valid D16/block32 geometry. No new quantizer, decoder or runtime path.'},
        'scope': 'Fresh original-HF all32 down D16/K24/S2/H128R0/BF16/block32/input512 INT8, MSE10Kx2/B8192/seed31. Preserve other192 and backbone from3577. No KD/LoRA.',
        'comparison_boundary': 'Primary versus completed fresh down input256/D16/H128/block328a; original3577 secondary. Only protection count and output paths differ in canonical parameters. Different protection changes indices, remaining matrix, input rotation sign length, vector inventory and sampling trajectory; do not require old/new protection or signs to be equal. No claim that greater protection is necessarily beneficial.',
        'resource_estimate': {'seconds_limit': 2700,
            'historical_control_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'historical_control_stage_seconds': {name: item['finished_unix'] - item['started_unix']
                                               for name, item in template['stages'].items()},
            'uncertainty': 'Estimate from the actual same-geometry protection256 run; no queue, automatic retry or new GPU short.'},
        'protocol': template['protocol'], 'cuda_initialized': False}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(evidence),
           'estimated_bpw': budget['estimated_bpw'], 'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
