"""CPU preparation of one fresh Qwen O48 full-RHT category replacement."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check
from experiments.full_rht_gate_pair.compare import normalized_recipe
from experiments.llama_ov_bits_pair.compare import arm as historical_arm
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen_o_bits_pair.prepare import estimate
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta


CONTROL = RESULT / 'fresh_o_bits_pair/control'
EXPECTED_BITS = 17065672704


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'),
                       '--weight_rotation_block_size': '0'}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['o_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    for section in (expected['canonical_config']['core_template'], expected['vae_args']):
        section['weight_rotation_block_size'] = 0
    _require(normalized == expected,
             'Actual native parser changed settings beyond output and rotation block size.')
    _require(cat.target_layers == 'all' and cat.batch_size == 8192 and cat.linear_group_size == 36
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and training.bf16 and not training.fp16 and resolved['o_proj'].steps == 10000
             and resolved['o_proj'].codebook_bits == 48 and resolved['o_proj'].codebook_dim == 32,
             'The inherited full-category O48 recipe changed.')
    return argv, normalized


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a new output and CPU-only snapshot invocation.')
    template, audit, control, _, evidence = historical_arm(CONTROL, 'control')
    _require(template['model_type'] == 'qwen3' and template['categories'] == ['o_proj']
             and template['codebook_bits'] == {'o_proj': 48} and template['block_size'] == 32
             and template['base_checkpoint_id'] == EXPECTED_IDS['qwen3'],
             'Require the completed fresh O48/block32 historical control.')
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == EXPECTED_IDS['qwen3'], 'Assembly base must be fresh f927.')
    budget = estimate(meta, {'o_proj': 48}, audit['bit_budget'])
    _require(budget['estimated_total_bits'] == EXPECTED_BITS
             and budget['code_delta_bits'] == budget['decoder_delta_bits'] == 0,
             'Same-rate rotation must not add code or decoder storage.')
    argv, normalized = recipe(template, output)
    sources = dict(template['source_sha256'])
    dependencies = ['experiments/full_rht_gate_pair/compare.py',
        'experiments/llama_ov_bits_pair/compare.py', 'experiments/llama_ov_bits_pair/audit.py',
        'experiments/qwen_o_bits_pair/prepare.py', 'experiments/qwen_o_bits_pair/audit.py']
    for path in [*(ROOT / relative for relative in dependencies),
                 *(ROOT / 'experiments/qwen_o_full_rht').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = {**template['input_sha256'], **evidence}
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'qwen3', 'arm': 'full', 'categories': ['o_proj'], 'codebook_bits': {'o_proj': 48},
        'architecture': template['architecture'], 'base_dir': str(base),
        'base_checkpoint_id': meta['checkpoint_id'], 'control_result_paths': template['control_result_paths'],
        'historical_control_dir': str(CONTROL), 'historical_control_checkpoint_id': control['checkpoint_id'],
        'historical_control_result_paths': [str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        'historical_control_audit_path': str(CONTROL / 'audit.json'),
        'block_size': 0, 'steps_per_stage': 10000, 'full_evaluation': True,
        'expected_user_linear_bits': EXPECTED_BITS, 'budget_estimate': budget,
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': evidence, 'protected_stats': template['protected_stats'],
        'snapshot_import_paths': template['snapshot_import_paths'], 'environment': template['environment'],
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.qwen_o_full_rht.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False,
            'existing_paths': 'Original O48/D32 native control and verified full-RHT gate/down paths cover the 4032/4096 factors, BF16 native packed decode, and transpose/inverse. Reuse the pinned snapshot CPU and native_validation_r1 proofs.',
            'historical_weights_loaded': False},
        'scope': 'Fresh original-HF all36 O48/D32/S2/H128R0/BF16/input64 INT8, full two-sided RHT, MSE10Kx2/B8192/seed31; retain other216 and backbone from f927. No KD/LoRA.',
        'comparison_boundary': 'Primary full versus completed fresh O48/block32; f927 secondary. Only native output path and block-size mirrors change. Protection/sign tensors must match exactly. Joint OV retrieval attribution does not identify O alone.',
        'resource_estimate': {'seconds_limit': 2700,
            'historical_control_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'historical_control_stage_seconds': {name: item['finished_unix'] - item['started_unix']
                                                for name, item in template['stages'].items()},
            'uncertainty': 'Actual O48/full-RHT full-category timing is not measured; the bound is not a runtime guarantee. No queue or automatic retry.'},
        'protocol': template['protocol'], 'cuda_initialized': False}
    _require(normalized_recipe(record) == normalized_recipe(template), 'Only output/block may differ.')
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(evidence),
           'estimated_bpw': budget['estimated_bpw'],
           'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output_dir)
