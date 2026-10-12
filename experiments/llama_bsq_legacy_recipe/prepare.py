"""Prepare the existing four-parameter BSQ recipe on matched fresh Llama gate."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_BITS, EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.full_rht_gate_pair.compare import arm
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta

CONTROL = RESULT / 'full_rht_gate_pair/llama_block32'
CONTROL_ID = '0ad677ce-a49a-423e-a20c-50f8aa1ef710'
OLD_VALUES = {'lfq_weight': 2.5, 'commitment_loss_weight': .25, 'inv_temperature': 100., 'entropy_loss_weight': .01}
NEW_VALUES = {'lfq_weight': 5., 'commitment_loss_weight': .075, 'inv_temperature': 200., 'entropy_loss_weight': 1e-4}


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    changes = {'--output_dir': str(output / 'donor'), **{'--' + key: str(value) for key, value in NEW_VALUES.items()}}
    for key, value in OLD_VALUES.items():
        _require(template['normalized_expected']['vae_args'][key] == value, 'Unexpected control BSQ value: ' + key)
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous native argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    for key, value in NEW_VALUES.items():
        expected['vae_args'][key] = value
        section = 'core_template' if key == 'inv_temperature' else 'vae_opt_template'
        expected['canonical_config'][section][key] = value
    _require(normalized == expected, 'Only output and the four BSQ coefficients may change, including their mirrors.')
    _require(cat.target_layers == 'all' and cat.batch_size == 8192 and cat.linear_group_size == 32
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj'
             and training.bf16 and not training.fp16 and vae.lr == .003
             and resolved['gate_proj'].steps == 10000 and vae.new_quant,
             'Keep the native fresh full-category recipe and gate transpose=False.')
    return argv, normalized


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template, audit, control, _, evidence = arm(CONTROL, 32)
    _require(template['model_type'] == 'llama' and audit['checkpoint_id'] == control['checkpoint_id'] == CONTROL_ID
             and template['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and control['integer_counts']['mmlu']['correct'] == 6436
             and control['mean_percent'] == 58.65337216978529
             and audit['source_union_save_fresh_tensors'] == 2979
             and audit['gate_count'] == 32 and audit['preserved_projection_count'] == 192,
             'Require the completed matched fresh gate control, never an E2E or per-model-entropy endpoint.')
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == EXPECTED_IDS['llama'] and (base / meta['state_dict_file']).is_file(),
             'The original3577 assembly base must remain available.')
    argv, normalized = recipe(template, output)
    sources = dict(template['source_sha256'])
    for path in [ROOT / 'experiments/full_rht_gate_pair/compare.py',
                 *(ROOT / 'experiments/llama_bsq_legacy_recipe').glob('*.py'),
                 *(ORIGINAL / 'experiments/llama_bsq_legacy_recipe').glob('*.py')]:
        sources[str(path)] = sha256(path)
    record = {key: copy.deepcopy(template[key]) for key in (
        'cwd', 'architecture', 'base_dir', 'base_checkpoint_id', 'control_result_paths',
        'block_size', 'steps_per_stage', 'full_evaluation', 'expected_user_linear_bits',
        'protected_stats', 'snapshot_import_paths', 'environment', 'protocol')}
    record.update(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        python_executable=sys.executable, output_dir=str(output), model_type='llama', categories=['gate_proj'],
        training_argv=[sys.executable, 'tools/cat_train.py', *argv], normalized_expected=normalized,
        source_sha256=sources, input_sha256={**template['input_sha256'], **evidence},
        historical_control_dir=str(CONTROL), historical_control_checkpoint_id=CONTROL_ID,
        historical_control_audit_path=str(CONTROL / 'audit.json'),
        historical_control_result_paths=[str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        bsq_control=OLD_VALUES, bsq_candidate=NEW_VALUES,
        budget_estimate={'estimated_total_bits': EXPECTED_BITS['llama'], 'original_linear_weights': 6979321856,
            'estimated_bpw': EXPECTED_BITS['llama'] / 6979321856,
            'boundary': 'Only training regularizer coefficients change; final native tensor storage remains mandatory.'},
        timeout_seconds=2700, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_bsq_legacy_recipe.run',
                       '--launch-record', str(output / 'launch.json')],
        validation_reuse={'new_gpu_short': False, 'matched_native_control': str(CONTROL),
            'native_tensor_count': 2979, 'full8_leaves': 64, 'existing_cli_parameters_only': True},
        scope='Fresh original-HF all32 gate/K32D32/S2/H128R0/block32/input64/BF16/B8192/10Kx2/seed31/LR.003; no KD/LoRA. Native pooled BSQ entropy unchanged.',
        comparison_boundary='Four-parameter BSQ recipe bundle, not a single-factor attribution. Primary is the matched fresh0ad677ce gate control; original3577 is secondary. No inference from reduced entropy/auxiliary loss alone.',
        experience={'reference': str(ORIGINAL / 'docs/experience/lessons/output_losses.md'),
            'prior': 'Llama per-Linear marginal entropy changed the algorithm and had mixed full8 results; it did not test these four coefficients. Qwen lfq1/gamma0=0 are not Llama controls.',
            'action': 'Keep native pooled entropy/STE, rotation, geometry, LR and budget fixed; compare complete eight tasks without a score threshold.'},
        resource_estimate={'matched_control_full_chain_seconds': template['finished_unix'] - template['started_unix'],
            'seconds_limit': 2700, 'boundary': 'One explicit assigned GPU window; no queue or automatic retry.'},
        cuda_initialized=False)
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record); output.mkdir(parents=True)
    write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(record['input_sha256']),
           'preparation_sha256': sha256(output / 'launch.prepared.json'), 'cuda_initialized': False}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
