"""Prepare one native unrotated gate arm against the completed fresh block32 control."""
from __future__ import annotations

import argparse
import copy
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

CONTROL = RESULT / 'full_rht_gate_pair/llama_block32'
CONTROL_ID = '0ad677ce-a49a-423e-a20c-50f8aa1ef710'
CONTROL_BITS = 17100439552
REMOVED_SIGN_BITS = 32 * (14336 + 4032) * 8
EXPECTED_BITS = CONTROL_BITS - REMOVED_SIGN_BITS
TRANSPOSE_MODULES = 'q_proj,v_proj,o_proj,down_proj'


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    _require(argv.count('--weight_rotation') == 1
             and argv[argv.index('--weight_rotation') + 1] == 'two_sided'
             and argv[argv.index('--transpose_modules') + 1] == TRANSPOSE_MODULES,
             'Require the original nontransposed block32 gate recipe.')
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--weight_rotation': 'none'}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    expected['canonical_config']['core_template']['weight_rotation'] = 'none'
    expected['vae_args']['weight_rotation'] = 'none'
    _require(normalized == expected, 'Only output path and weight_rotation may change; every BSQ/geometry/training field must match.')
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
             and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.seed == 31 and cat.transpose_modules == TRANSPOSE_MODULES
             and training.bf16 and not training.fp16 and resolved['gate_proj'].steps == 10000
             and vae.weight_rotation == 'none' and vae.weight_rotation_block_size == 32,
             'Native complete single-category recipe differs.')
    return argv, normalized


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template, audit, control, _, evidence = arm(CONTROL, 32)
    _require(audit['checkpoint_id'] == control['checkpoint_id'] == CONTROL_ID
             and template['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and control['integer_counts']['mmlu']['correct'] == 6436
             and control['mean_percent'] == 58.65337216978529
             and audit['source_union_save_fresh_tensors'] == 2979,
             'Require the completed fresh K32/D32/block32 nontransposed gate control.')
    review_path = RESULT / 'full_rht_gate_pair/llama_comparison.json'; review = read(review_path)
    _require(review['status'] == 'verified_pair' and review['model_type'] == 'llama'
             and review['arms']['block32'] == control
             and review['pair_checks']['protection_tensor_records_exact']
             and review['pair_checks']['rotation_sign_records_exact'], 'Control independent review differs.')
    _require(sha256(ROOT / 'experiments/full_rht_gate_pair/compare.py')
             == review['comparison_source_sha256'], 'Control comparison source differs.')
    check_hashes(review['evidence_sha256'])
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == EXPECTED_IDS['llama'] and (base / meta['state_dict_file']).is_file(),
             'Require the retained original3577 assembly base, not a later recovery variant.')
    budget = audit['bit_budget']
    _require(budget['user_linear_bits'] == budget['packed_linear_bits'] == CONTROL_BITS
             and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0
             and budget['budget_passed'] and not budget['contract_violations']
             and budget['components']['decoder']['dtypes'] == ['bfloat16'], 'Measured control budget differs.')
    signs = audit['rotation_sign_records']
    _require(len(signs) == 32 and all(len(row) == 2 for row in signs.values()),
             'Require both actual historical sign vectors for every gate.')
    removed_bits = 0
    for name, row in signs.items():
        _require(set(row) == {name + '.weight_rotation.input_signs', name + '.weight_rotation.output_signs'},
                 'Historical sign names differ.')
        for key, value in row.items():
            expected_shape = [4032] if key.endswith('.input_signs') else [14336]
            _require(value['dtype'] == 'torch.int8' and value['shape'] == expected_shape,
                     'Historical sign shape/dtype differs: ' + key)
            removed_bits += value['shape'][0] * 8
    _require(removed_bits == REMOVED_SIGN_BITS and (4096 - 64) % 32 == 14336 % 32 == 0,
             'Require the actual removable sign inventory and aligned gate vectors.')
    components = copy.deepcopy(budget['components'])
    for key in ('budget_bits', 'storage_bits'):
        components['rotation'][key] -= removed_bits
    components['rotation']['unique_storages'] -= 64
    _require(sum(row['budget_bits'] for row in components.values()) == EXPECTED_BITS,
             'Expected component accounting does not sum to the reduced budget.')
    argv, normalized = recipe(template, output)
    sources = dict(template['source_sha256'])
    for path in [ROOT / 'experiments/full_rht_gate_pair/compare.py',
                 *(ROOT / 'experiments/llama_gate_no_rotation').glob('*.py'),
                 *(ORIGINAL / 'experiments/llama_gate_no_rotation').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = {**template['input_sha256'], **evidence, str(review_path): sha256(review_path)}
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'categories': ['gate_proj'], 'codebook_bits': {'gate_proj': 32},
        'dimension': 32, 'transpose': False, 'weight_rotation': 'none', 'architecture': template['architecture'], 'base_dir': str(base),
        'base_checkpoint_id': meta['checkpoint_id'], 'control_result_paths': template['control_result_paths'],
        'historical_control_dir': str(CONTROL), 'historical_control_checkpoint_id': CONTROL_ID,
        'historical_control_result_paths': [str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        'historical_control_audit_path': str(CONTROL / 'audit.json'),
        'block_size': 32, 'steps_per_stage': 10000, 'full_evaluation': True,
        'expected_user_linear_bits': EXPECTED_BITS, 'expected_components': components,
        'expected_serialized_tensors': 2915, 'removed_rotation_sign_bits': removed_bits,
        'removed_rotation_sign_tensors': 64,
        'budget_estimate': {'estimated_total_bits': EXPECTED_BITS, 'original_linear_weights': 6979321856,
            'estimated_bpw': EXPECTED_BITS / 6979321856, 'same_vector_count_per_linear_per_stage': 1806336,
            'boundary': 'Historical native measured components minus exactly64 gate sign tensors; main codes, decoder, protection indices/q/scales unchanged in size. Final native tensor audit is required.'},
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': evidence, 'protected_stats': template['protected_stats'],
        'snapshot_import_paths': template['snapshot_import_paths'], 'environment': template['environment'],
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_no_rotation.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 'matched_native_control': str(CONTROL),
            'control_review': str(review_path), 'historical_control_weights_required': False,
            'existing_paths': 'Existing weight_rotation=none, native D32 packed training/deployment and strict source-union assembler. No new GPU mechanism short test.'},
        'scope': 'Fresh HF all32 gate K32/D32/S2/H128R0/rotation none/input64/BF16, transpose False, MSE10Kx2/B8192/seed31; retain3577 other192 and backbone; no new KD/LoRA.',
        'comparison_boundary': 'Only weight_rotation two_sided to none; block_size32 remains an inactive CLI field. No gate sign tensors are created. Protection indices/q/scales must match the historical control exactly. Same seed/order and optimizer budget do not prove identical training RNG trajectories; historical initialization RNG was not saved. The fixed3577 complement contains prior recovery; this is not a whole-model pure-CAT comparison.',
        'experience': {'reference': str(ORIGINAL / 'docs/experience/lessons/compression_choices.md'),
            'prior': 'Existing Llama controls compare block32/full rotation or transpose, not rotation none.',
            'difference': 'Isolate the missing no-rotation gate control without asymmetric/R1/old-BSQ changes.',
            'action': 'Compare complete matched eight-task results; verify protection exact and64 sign tensors absent. No LongBench or nine-item conclusion.'},
        'resource_estimate': {'seconds_limit': 2700, 'estimated_full_chain_seconds': [1200, 2100],
            'matched_control_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'uncertainty': 'No-rotation full-chain runtime is unmeasured; no automatic retry or queue.'},
        'protocol': template['protocol'], 'cuda_initialized': False}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(evidence),
           'estimated_bpw': record['budget_estimate']['estimated_bpw'],
           'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output_dir)
