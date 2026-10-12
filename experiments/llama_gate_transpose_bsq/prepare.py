"""Prepare fresh transposed Llama gate with the already measured four BSQ coefficients."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.full_rht_gate_pair.compare import score
from experiments.llama_bsq_legacy_recipe.prepare import NEW_VALUES, OLD_VALUES
from experiments.llama_gate_transpose.audit import check_donor
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta

CONTROL = RESULT / 'llama_gate_transpose/transpose'
BSQ_CONTROL = RESULT / 'llama_bsq_legacy_recipe/gate'


def recipe(template, output):
    argv = list(template['training_argv'][2:])
    _require(template['transpose'] is True, 'Use the completed fresh transposed gate as the primary control.')
    for key, value in OLD_VALUES.items():
        _require(template['normalized_expected']['vae_args'][key] == value, 'Unexpected primary BSQ value: ' + key)
    for key, value in {'--output_dir': str(output / 'donor'),
                       **{'--' + key: str(value) for key, value in NEW_VALUES.items()}}.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited native argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'): expected[section]['output_dir'] = str(output / 'donor')
    for key, value in NEW_VALUES.items():
        expected['vae_args'][key] = value
        expected['canonical_config']['core_template' if key == 'inv_temperature' else 'vae_opt_template'][key] = value
    _require(normalized == expected, 'Only output and the four existing BSQ coefficients may differ from gateTrue.')
    cfg = resolved['gate_proj']
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,gate_proj,down_proj'
             and cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.seed == 31
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cfg.steps == 10000 and cfg.codebook_bits == cfg.codebook_dim == 32
             and cfg.residual_stages == 2 and cfg.base_ch == 128 and cfg.num_res_blocks == 0
             and cfg.channel_protect_count == 64 and vae.weight_rotation == 'two_sided'
             and vae.weight_rotation_block_size == 32 and vae.lr == .003 and vae.new_quant
             and training.bf16 and not training.fp16, 'Keep the exact fresh native gateTrue recipe.')
    return argv, normalized


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get('PYTHONPATH') == str(ROOT)
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    template = read(CONTROL / 'launch.json'); check(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and template['base_checkpoint_id'] == EXPECTED_IDS['llama']
             and all(int((CONTROL / (name + '.exit')).read_text()) == 0
                     for name in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'Require the completed formal gateTrue control.')
    donor, names = check_donor(Path(template['donor_checkpoint_dir']), template)
    audit = read(CONTROL / 'audit.json'); meta = load_v6_meta(str(CONTROL / 'final_model'))
    current, _ = score([CONTROL / (name + '.json') for name in ('mmlu', 'seven')],
                        template['architecture'], CONTROL / 'final_model')
    review_path = CONTROL / 'independent_review.json'; review = read(review_path)
    _require(review['status'] == 'independent_cpu_verified' and review['candidate'] == current
             and audit['status'] == 'verified' and audit['checkpoint_id'] == current['checkpoint_id'] == meta['checkpoint_id']
             and audit['donor_checkpoint_id'] == donor['checkpoint_id']
             and current['mean_percent'] == 59.91121590186278 and current['integer_counts']['mmlu']['correct'] == 6638
             and audit['source_union_save_fresh_tensors'] == 2979 and len(names) == 32,
             'Primary control raw scores or deployed source evidence differs.')
    check_hashes(review['evidence_sha256'])
    bsq_review_path = BSQ_CONTROL / 'independent_review.json'; bsq_review = read(bsq_review_path)
    _require(bsq_review['status'] == 'verified' and bsq_review['source_checkpoint_id'] == EXPECTED_IDS['llama']
             and bsq_review['bsq_control'] == OLD_VALUES and bsq_review['bsq_candidate'] == NEW_VALUES
             and bsq_review['mean_percent'] == 59.212918930137484 and bsq_review['mmlu_correct'] == 6609,
             'Require the measured separate native four-coefficient BSQ recipe.')
    check_hashes(bsq_review['evidence_sha256'])
    base = Path(template['base_dir']); base_meta = load_v6_meta(str(base))
    budget = audit['bit_budget']
    _require(base_meta['checkpoint_id'] == EXPECTED_IDS['llama'] and (base / base_meta['state_dict_file']).is_file()
             and budget['budget_passed'] and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17100439552
             and budget['original_linear_weights'] == 6979321856 and not budget['contract_violations']
             and budget['code_padding_bits'] == 0, 'Original3577 or actual control storage changed.')
    argv, normalized = recipe(template, output)
    sources = dict(template['source_sha256'])
    for path in [ROOT / 'experiments/llama_bsq_legacy_recipe/prepare.py',
                 *(ROOT / 'experiments/llama_gate_transpose_bsq').glob('*.py'),
                 *(ORIGINAL / 'experiments/llama_gate_transpose_bsq').glob('*.py')]:
        sources[str(path)] = sha256(path)
    evidence = [CONTROL / name for name in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json')]
    evidence += [review_path, bsq_review_path, CONTROL / 'final_model/checkpoint_meta.json',
                 CONTROL / 'final_model/assembly_report.json']
    record = {key: deepcopy(template[key]) for key in ('cwd', 'architecture', 'base_dir', 'base_checkpoint_id',
        'control_result_paths', 'block_size', 'steps_per_stage', 'full_evaluation', 'expected_user_linear_bits',
        'protected_stats', 'snapshot_import_paths', 'environment', 'protocol', 'dimension', 'transpose',
        'codebook_bits', 'historical_control_dir', 'historical_control_checkpoint_id',
        'historical_control_audit_path', 'historical_control_result_paths')}
    record.update(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        python_executable=sys.executable, output_dir=str(output), model_type='llama', categories=['gate_proj'],
        training_argv=[sys.executable, 'tools/cat_train.py', *argv], normalized_expected=normalized,
        source_sha256=sources, input_sha256={**template['input_sha256'], **{str(path): sha256(path) for path in evidence}},
        primary_control_dir=str(CONTROL), primary_control_checkpoint_id=current['checkpoint_id'],
        primary_control_result_paths=[str(CONTROL / (name + '.json')) for name in ('mmlu', 'seven')],
        primary_control_audit_path=str(CONTROL / 'audit.json'),
        bsq_control=OLD_VALUES, bsq_candidate=NEW_VALUES,
        budget_estimate={'estimated_total_bits': 17100439552, 'original_linear_weights': 6979321856,
            'estimated_bpw': 17100439552 / 6979321856,
            'boundary': 'Training coefficients only; actual saved tensor audit remains required.'},
        timeout_seconds=2700, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_gate_transpose_bsq.run',
                       '--launch-record', str(output / 'launch.json')],
        validation_reuse={'new_gpu_short': False, 'matched_native_gate_true': str(CONTROL),
            'separate_bsq_recipe': str(BSQ_CONTROL), 'existing_cli_parameters_only': True},
        scope='Fresh HF all32 gate transposeTrue/K32D32/S2/H128R0/block32/input64/BF16/B8192/10Kx2/seed31/LR.003; retain3577 other192/backbone; no KD/LoRA.',
        comparison_boundary='Primary is fresh gateTrue59.91121590 with old BSQ coefficients. The four-coefficient bundle is one tested recipe, not isolated coefficient attribution. Separate gate-transpose and BSQ gains do not imply additive combined gains. The original3577 complement includes earlier recovery.',
        experience={'reference': str(ORIGINAL / 'docs/experience/lessons/compression_choices.md'),
            'prior': 'Separate gateTrue and native old-BSQ coefficients each improved its matched freshFalse control.',
            'action': 'Keep gateTrue geometry/rotation/protection/seed/budget; compare complete eight tasks without assuming additive gains.'},
        resource_estimate={'seconds_limit': 2700,
            'matched_gate_true_chain_seconds': template['finished_unix'] - template['started_unix'],
            'boundary': 'One explicitly bound GPU; no automatic queue or retry.'}, cuda_initialized=False)
    check(record); _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'source_pins': len(sources), 'input_pins': len(record['input_sha256']),
           'estimated_bpw': record['budget_estimate']['estimated_bpw'],
           'prepared_sha256': sha256(output / 'launch.prepared.json'), 'cuda_initialized': False}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
