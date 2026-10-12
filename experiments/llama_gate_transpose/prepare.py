"""Prepare one native transposed gate arm against the completed fresh control."""
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
EXPECTED_BITS = 17100439552
TRANSPOSE_MODULES = 'q_proj,v_proj,o_proj,gate_proj,down_proj'


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    _require(argv.count('--transpose_modules') == 1
             and argv[argv.index('--transpose_modules') + 1] == 'q_proj,v_proj,o_proj,down_proj',
             'The matched control must use the original nontransposed gate recipe.')
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--transpose_modules': TRANSPOSE_MODULES}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    expected['canonical_config']['core_template']['transpose_modules'] = TRANSPOSE_MODULES
    expected['cat_args']['transpose_modules'] = TRANSPOSE_MODULES
    _require(normalized == expected, 'Only output path and gate transpose selection may change.')
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
             and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.seed == 31 and cat.transpose_modules == TRANSPOSE_MODULES
             and training.bf16 and not training.fp16 and resolved['gate_proj'].steps == 10000,
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
    _require(budget['user_linear_bits'] == budget['packed_linear_bits'] == EXPECTED_BITS
             and budget['original_linear_weights'] == 6979321856 and budget['code_padding_bits'] == 0
             and budget['budget_passed'] and not budget['contract_violations']
             and budget['components']['decoder']['dtypes'] == ['bfloat16'], 'Measured control budget differs.')
    # Both remaining axes divide D32/block32; transposition adds no tail vectors or padding.
    _require((4096 - 64) % 32 == 14336 % 32 == 0 and 64 % 32 == 0,
             'Protection and both vector orientations must remain aligned.')
    argv, normalized = recipe(template, output)
    sources = dict(template['source_sha256'])
    for path in [ROOT / 'experiments/full_rht_gate_pair/compare.py',
                 *(ROOT / 'experiments/llama_gate_transpose').glob('*.py'),
                 *(ORIGINAL / 'experiments/llama_gate_transpose').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = {**template['input_sha256'], **evidence, str(review_path): sha256(review_path)}
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'categories': ['gate_proj'], 'codebook_bits': {'gate_proj': 32},
        'dimension': 32, 'transpose': True, 'architecture': template['architecture'], 'base_dir': str(base),
        'base_checkpoint_id': meta['checkpoint_id'], 'control_result_paths': template['control_result_paths'],
        'historical_control_dir': str(CONTROL), 'historical_control_checkpoint_id': CONTROL_ID,
        'historical_control_result_paths': [str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        'historical_control_audit_path': str(CONTROL / 'audit.json'),
        'block_size': 32, 'steps_per_stage': 10000, 'full_evaluation': True,
        'expected_user_linear_bits': EXPECTED_BITS,
        'budget_estimate': {'estimated_total_bits': EXPECTED_BITS, 'original_linear_weights': 6979321856,
            'estimated_bpw': EXPECTED_BITS / 6979321856, 'same_vector_count_per_linear_per_stage': 1806336,
            'boundary': 'Same D32 decoder/vector count and protected payload; final integer tensor audit required.'},
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': evidence, 'protected_stats': template['protected_stats'],
        'snapshot_import_paths': template['snapshot_import_paths'], 'environment': template['environment'],
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_transpose.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 'matched_native_control': str(CONTROL),
            'control_review': str(review_path), 'historical_control_weights_required': False,
            'existing_paths': 'Existing transpose_modules, native D32 packed training/deployment, assembler2979 and complete64-leaf evaluation.'},
        'scope': 'Fresh HF all32 gate K32/D32/S2/H128R0/block32/input64/BF16, MSE10Kx2/B8192/seed31; retain3577 other192 and backbone; no new KD/LoRA.',
        'comparison_boundary': 'Only gate transpose False toTrue. Native single-category/group32 order and random-call dimensions are retained; historical initialization RNG snapshots were not saved or compared. Orientation changes VQ vectors after protection/rotation and does not imply equal errors or optimization trajectories. The fixed3577 complement contains prior recovery; this is not a whole-model pure-CAT comparison.',
        'experience': {'reference': str(ORIGINAL / 'docs/experience/lessons/compression_choices.md'),
            'prior': 'Matched Qwen gate transpose hurt its mean; Llama down nontranspose helped its own matched mean.',
            'difference': 'Llama gate32/block32/3577 context differs from both prior geometries; do not transfer rankings.',
            'action': 'Keep protection/signs exact, compare complete matched eight-task results, and infer no LongBench or nine-item score.'},
        'resource_estimate': {'seconds_limit': 2700, 'estimated_full_chain_seconds': [1200, 2100],
            'matched_control_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'uncertainty': 'Orientation-specific runtime is unmeasured; no automatic retry or queue.'},
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
