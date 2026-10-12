"""Prepare independent fresh Llama q/up orientation pairs in the original3577 context."""
from __future__ import annotations

import argparse
import copy
import importlib
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.full_rht_gate_pair.compare import arm
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta

TEMPLATE = RESULT / 'full_rht_gate_pair/llama_block32'
TEMPLATE_ID = '0ad677ce-a49a-423e-a20c-50f8aa1ef710'
EXPECTED_BITS = 17100439552
INVENTORY = RESULT / 'llama_q_up_transpose/control_inventory.json'


def recipe(template, output, category, transpose):
    _require(category in ('up_proj', 'q_proj') and type(transpose) is bool,
             'Only the two declared independent q/up orientation pairs are supported.')
    original = 'q_proj,v_proj,o_proj,down_proj'
    modules = original.split(',')
    if transpose and category not in modules:
        modules.insert(modules.index('down_proj'), category)
    elif not transpose and category in modules:
        modules.remove(category)
    selected = ','.join(modules)
    argv = list(template['training_argv'][2:])
    _require(argv[argv.index('--compression_categories') + 1] == 'gate_proj'
             and argv[argv.index('--transpose_modules') + 1] == original,
             'Require the existing fresh native gate template.')
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--compression_categories': category, '--transpose_modules': selected}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, [category])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    expected['canonical_config']['compression_categories'] = [category]
    expected['cat_args']['compression_categories'] = category
    expected['canonical_config']['core_template']['transpose_modules'] = selected
    expected['cat_args']['transpose_modules'] = selected
    configuration = expected['resolved_category_runtime'].pop('gate_proj')
    configuration['category'] = category
    expected['resolved_category_runtime'][category] = configuration
    _require(normalized == expected,
             'Only the category, its orientation, and output path may differ from the native template.')
    cfg = resolved[category]
    _require(cat.compression_categories == category and cat.target_layers == 'all'
             and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.seed == 31 and normalized['canonical_config']['data']['data_seed'] == 31
             and training.bf16 and not training.fp16
             and cfg.steps == 10000 and cfg.codebook_bits == cfg.codebook_dim == 32
             and cfg.residual_stages == 2 and cfg.base_ch == 128 and cfg.num_res_blocks == 0
             and cfg.channel_protect_count == 64, 'Native fixed single-category recipe differs.')
    return argv, normalized


def prepare(output, category, transpose):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template, audit, control, _, evidence = arm(TEMPLATE, 32)
    _require(audit['checkpoint_id'] == control['checkpoint_id'] == TEMPLATE_ID
             and template['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and audit['source_union_save_fresh_tensors'] == 2979,
             'Require the completed fresh native template in the original3577 context.')
    review_path = RESULT / 'full_rht_gate_pair/llama_comparison.json'
    review = read(review_path)
    _require(review['status'] == 'verified_pair' and review['model_type'] == 'llama'
             and review['arms']['block32'] == control
             and sha256(ROOT / 'experiments/full_rht_gate_pair/compare.py') == review['comparison_source_sha256'],
             'Existing native template independent review differs.')
    check_hashes(review['evidence_sha256'])
    inventory = read(INVENTORY)
    _require(inventory['status'] == 'searched' and not inventory['eligible_fresh_single_category_controls'],
             'Existing fresh q/up controls must be considered before preparing a new pair.')
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == EXPECTED_IDS['llama'] and (base / meta['state_dict_file']).is_file(),
             'Require the retained original3577 assembly base.')
    budget = audit['bit_budget']
    _require(budget['user_linear_bits'] == budget['packed_linear_bits'] == EXPECTED_BITS
             and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0
             and budget['budget_passed'] and not budget['contract_violations']
             and budget['components']['decoder']['dtypes'] == ['bfloat16'], 'Template actual budget differs.')
    names = {name for name in template['architecture']['projection_shapes'] if name.endswith('.' + category)}
    _require(len(names) == 32, 'Require all 32 actual q or up projections.')
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    expected_shape = [4096, 4096] if category == 'q_proj' else [14336, 4096]
    for name in names:
        shape = template['architecture']['projection_shapes'][name]; spec = specs[name]
        _require(shape == expected_shape and spec['in_features'] == shape[1]
                 and spec['out_features'] == shape[0] and spec['compressed_in_features'] == shape[1] - 64
                 and spec['compressed_out_features'] == shape[0] and spec['codebook_dim'] == 32
                 and spec['residual_stages'] == 2 and spec['parallel_parts'] == 1
                 and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['weight_rotation']['block_size'] == 32
                 and all(stage['in_dim'] == stage['out_dim'] == 32 and stage['hidden_dim'] == 128
                         and stage['num_res_blocks'] == 0 and stage['param_dtype'] == 'bfloat16'
                         for stage in spec['stage_decoders'])
                 and (shape[1] - 64) % 32 == shape[0] % 32 == 64 % 32 == 0,
                 'Base geometry must preserve integer storage under either orientation: ' + name)
    argv, normalized = recipe(template, output, category, transpose)
    sources = dict(template['source_sha256'])
    for path in [ROOT / 'experiments/full_rht_gate_pair/compare.py',
                 *(ROOT / 'experiments/llama_q_up_transpose').glob('*.py'),
                 *(ORIGINAL / 'experiments/llama_q_up_transpose').glob('*.py')]:
        sources[str(path)] = sha256(path)
    inputs = {**template['input_sha256'], **evidence, str(review_path): sha256(review_path),
              str(INVENTORY): sha256(INVENTORY)}
    import_paths = dict(template['snapshot_import_paths'])
    for module in ('experiments.llama_q_up_transpose.prepare', 'experiments.llama_q_up_transpose.run',
                   'experiments.llama_q_up_transpose.audit'):
        actual = Path(importlib.import_module(module).__file__).resolve()
        _require(actual == ROOT / (module.replace('.', '/') + '.py'), 'New module escaped the frozen snapshot.')
        import_paths[module] = str(actual)
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'categories': [category], 'codebook_bits': {category: 32},
        'dimension': 32, 'transpose': transpose, 'architecture': template['architecture'],
        'base_dir': str(base), 'base_checkpoint_id': meta['checkpoint_id'],
        'control_result_paths': template['control_result_paths'],
        'storage_reference_audit_path': str(TEMPLATE / 'audit.json'),
        'storage_reference_checkpoint_id': TEMPLATE_ID,
        'block_size': 32, 'steps_per_stage': 10000, 'full_evaluation': True,
        'expected_user_linear_bits': EXPECTED_BITS,
        'budget_estimate': {'estimated_total_bits': EXPECTED_BITS, 'original_linear_weights': 6979321856,
            'estimated_bpw': EXPECTED_BITS / 6979321856,
            'same_vector_count_per_linear_per_stage': expected_shape[0] * (expected_shape[1] - 64) // 32,
            'boundary': 'Equal aligned vector/decoder/protection sizes; actual final tensor audit required.'},
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': template['protected_stats'],
        'snapshot_import_paths': import_paths, 'environment': template['environment'],
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_q_up_transpose.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 'native_template': str(TEMPLATE),
            'template_review': str(review_path), 'control_search': str(INVENTORY),
            'existing_paths': 'Native category/transpose CLI, D32/BF16 packed deployment, assembler2979, complete64-leaf full8.'},
        'scope': 'Independent fresh HF single q/up category, all32 K32/D32/S2/H128R0/block32/input64, MSE10Kx2/B8192/seed31; retain3577 other192/backbone; no new KD/LoRA.',
        'comparison_boundary': 'Compare False/True within the same fresh category. Each arm starts a separate native process; do not train q/up together. Equal random-call dimensions do not prove identical unsaved initialization RNG snapshots. The3577 complement includes earlier recovery, so this is not whole-model pure CAT. The gate template is a pipeline/storage reference, not a q/up quality control.',
        'experience': {'reference': str(ORIGINAL / 'docs/experience/lessons/compression_choices.md'),
            'prior': 'Llama gate True gained1.25784373pp against freshFalse; matched down False also improved its own context.',
            'difference': 'Neither result determines q/up orientation or additive gains.',
            'action': 'Separate fresh pairs, exact protection/sign comparison, complete full8; no LongBench/nine-item inference.'},
        'resource_estimate': {'seconds_limit': 2700, 'estimated_full_chain_seconds': [1200, 2100],
            'native_gate_chain_seconds': template['finished_unix'] - template['started_unix'],
            'uncertainty': 'q is smaller and up matches gate geometry; orientation-specific time is unmeasured.'},
        'protocol': template['protocol'], 'cuda_initialized': False}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'category': category, 'transpose': transpose,
           'source_pins': len(sources), 'input_pins': len(inputs),
           'estimated_bpw': record['budget_estimate']['estimated_bpw'],
           'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', choices=('up_proj', 'q_proj'), required=True)
    parser.add_argument('--transpose', choices=('false', 'true'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output_dir, args.category, args.transpose == 'true')
