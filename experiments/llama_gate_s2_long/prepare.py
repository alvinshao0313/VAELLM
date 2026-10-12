"""Prepare fresh K32/S2 gate with 20000 native updates per stage."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_small_rate_pair.common import ROOT, ORIGINAL, RESULT, BASE_ID, DENOMINATOR, check, read, _require
from experiments.llama_gate_s2_long.audit import TOTAL_BITS, CONTROL_ID, S4_ID
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_v1.common import sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload

CONTROL = RESULT / 'llama_gate_small_rate_pair/control'
S4 = RESULT / 'llama_gate_stage4/s4'


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Require fresh CPU-only preparation.')
    template = read(CONTROL / 'launch.json'); check(template)
    previous = {}
    for directory, identity in ((CONTROL, CONTROL_ID), (S4, S4_ID)):
        launch = read(directory / 'launch.json'); proof = read(directory / 'audit.json'); summary = read(directory / 'full8_summary.json')
        result, _ = score([directory / (n + '.json') for n in ('mmlu', 'seven')], template['architecture'], directory / 'final_model')
        _require(launch['status'] == 'completed' and launch['exit_code'] == 0
                 and launch['checkpoint_id'] == proof['checkpoint_id'] == result['checkpoint_id'] == identity
                 and launch['base_checkpoint_id'] == BASE_ID and proof['status'] == 'verified'
                 and all((directory / (s + '.exit')).read_text().strip() == '0'
                         for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))
                 and result['integer_counts'] == summary['integer_counts'] and result['mean_percent'] == launch['mean_percent']
                 and summary['audit_sha256'] == sha256(directory / 'audit.json')
                 and launch['full8_summary_sha256'] == sha256(directory / 'full8_summary.json'), 'Completed control evidence differs.')
        previous[str(directory)] = result
    s4_review = read(S4 / 'independent_review.json')
    _require(s4_review['status'] == 'independent_cpu_verified' and s4_review['checkpoint_id'] == S4_ID, 'Require independent completed S4 evidence.')
    argv = list(template['training_argv'][2:])
    for key, value in {'--output_dir': str(output / 'donor'), '--vae_steps': 'default=20000'}.items():
        _require(argv.count(key) == 1, 'Ambiguous native argument: ' + key); argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv); resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'): expected[section]['output_dir'] = str(output / 'donor')
    expected['canonical_config']['vae_steps'] = {'default': 20000}
    expected['resolved_category_runtime']['gate_proj']['steps'] = 20000
    _require(normalized == expected, 'Recipe changes beyond native stage length and output.')
    cfg = resolved['gate_proj']
    _require(cat.seed == 31 and cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.target_layers == 'all'
             and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint and not cat.channel_refresh_after_category
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj' and cfg.steps == 20000
             and cfg.codebook_bits == cfg.codebook_dim == 32 and cfg.residual_stages == 2
             and cfg.channel_protect_count == 64 and cfg.base_ch == 128 and cfg.num_res_blocks == 0
             and training.bf16 and not training.fp16, 'Fixed scientific conditions changed.')
    budget = read(CONTROL / 'audit.json')['bit_budget']; s4_budget = read(S4 / 'audit.json')['bit_budget']
    _require(budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
             and budget['code_padding_bits'] == 0 and budget['original_linear_weights'] == DENOMINATOR
             and s4_budget['user_linear_bits'] - TOTAL_BITS == 4620288, 'Original same-geometry budget differs.')
    sources = dict(template['source_sha256']); inputs = dict(template['input_sha256'])
    for p in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
        _require(sha256(ORIGINAL / p.relative_to(ROOT)) == sha256(p), 'Public/snapshot source differs.')
    for directory in (CONTROL, S4):
        native = Path(read(directory / 'launch.json')['donor_checkpoint_dir'])
        for p in [*[directory / n for n in ('launch.json', 'audit.json', 'full8_summary.json', 'mmlu.json', 'seven.json')],
                  directory / 'final_model/checkpoint_meta.json', directory / 'final_model/assembly_report.json',
                  native.parent / 'normalized_cat_runtime_args.json', native / 'checkpoint_meta.json']:
            inputs[str(p)] = sha256(p)
    inputs[str(S4 / 'independent_review.json')] = sha256(S4 / 'independent_review.json')
    provision = template['resource_estimate']['disk_provision_bytes']
    record = {'schema': 'llama_gate_s2_long_v1', 'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'model_type': 'llama',
        'categories': ['gate_proj'], 'codebook_bits': 32, 'residual_stages': 2, 'transpose': False, 'block_size': 32,
        'steps_per_stage': 20000, 'total_stage_updates': 40000, 'full_evaluation': True, 'base_dir': template['base_dir'],
        'base_checkpoint_id': BASE_ID, 'architecture': template['architecture'], 'historical_control_dir': str(CONTROL),
        'historical_control_checkpoint_id': CONTROL_ID, 'equal_update_s4_dir': str(S4), 'equal_update_s4_checkpoint_id': S4_ID,
        'control_result_paths': template['control_result_paths'], 'training_argv': [sys.executable, 'tools/cat_train.py', *argv],
        'normalized_expected': normalized, 'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': template['protected_stats'],
        'environment': template['environment'], 'protocol': template['protocol'], 'expected_budget_components': deepcopy(budget['components']),
        'expected_user_linear_bits': TOTAL_BITS, 'expected_state_tensors': 2979, 'expected_code_banks': 448,
        'budget_estimate': {'user_linear_bits': TOTAL_BITS, 'packed_linear_bits': TOTAL_BITS, 'bpw': TOTAL_BITS / DENOMINATOR, 'code_padding_bits': 0},
        'timeout_seconds': 3600, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_s2_long.run', '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'new_gpu_short': False, 'deleted_control_weights_required': False,
            'scope': 'Existing identical K32/D32/S2 geometry and complete native CAT/assembly/full8 proofs; only per-stage length changes. No S4 E2E exporter use.'},
        'comparison_boundary': 'Primary fresh K32/D32/S2 20000 per stage minus completed same recipe10000 per stage: original native cosine horizon extends and later stage RNG/residual targets change, not exact continuation. Secondary versus K16/D32/S4 at10000 per stage matches40000 stage updates and main2bit code rate, but K/S factorization, decoder capacity/storage, per-stage LR horizons and per-update FLOPs differ; not a pure stage-count or equal-compute causal comparison. Same freshHF/e767192/False/input64/block32. e767 additional reference; no LongBench conclusion.',
        'resource_estimate': {'estimated_seconds': [1800, 2500], 'seconds_limit': 3600, 'single_arm_disk_provision_bytes': provision,
            'historical_s2_seconds': template['finished_unix'] - template['started_unix'],
            'historical_s4_seconds': read(S4 / 'launch.json')['finished_unix'] - read(S4 / 'launch.json')['started_unix'],
            'filesystem_free_at_preparation': shutil.disk_usage(output.parent.parent).free,
            'boundary': 'Original S2 total1220s/train769s; S4 total1789s/train1340s. Two20000-step K32 stages not yet measured.'},
        'cuda_initialized': False}
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(output.parent.parent).free > provision, 'CUDA initialized or insufficient disk.')
    check(record); output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'source': len(sources), 'input': len(inputs),
           'stat': len(record['protected_stats']), 'bits': TOTAL_BITS, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
