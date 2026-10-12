"""Prepare native after-gate3000, matched to the completed rank56-500 restart."""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import sys

import torch

from experiments.llama_order_control.prepare import ROOT, check, normalized, read, require
from experiments.llama_remaining_rank_pair.prepare import ORDER, SOURCE_ID
from experiments.full_rht_gate_pair.compare import score, delta
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity, restore_cat_runtime_state, validate_cat_runtime_identity
from train_utils.cat_train_runtime import load_cat_resume_distill_progress, resolve_cat_resume_source
from train_utils.checkpoint_v6 import load_v6_cat_runtime_state
from train_utils.config.targets import parse_skip_layers, parse_target_layers

PAIR = ROOT / 'result/qwen3_8b_mmlu65_20261004/llama_remaining_rank_pair'
CONTROL = PAIR / 'rank56'
CONTROL_ID = 'c5d0f0f5-eb6a-4cd8-b865-fc30a17b2e3b'


def prepare(output):
    output = Path(output).resolve()
    require(Path.cwd().resolve() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not output.exists(), 'Use a fresh CPU-only public preparation.')
    control = read(CONTROL / 'launch.json'); check(control)
    require(control['status'] == 'completed' and control['exit_code'] == 0 and control['after_gate_rank'] == 56
            and all((CONTROL / (s + '.exit')).read_text().strip() == '0'
                    for s in ('train', 'mmlu', 'seven', 'summary', 'run', 'outer'))
            and (PAIR / 'comparison.exit').read_text().strip() == '0', 'Require natural rank56 and comparison completion.')
    pair = read(PAIR / 'comparison.json'); check_hashes(pair['evidence_sha256'])
    summary = read(CONTROL / 'full8_summary.json'); native = Path(control['final_checkpoint_dir']).parent
    measured, _ = score([CONTROL / (s + '.json') for s in ('mmlu', 'seven')], control['architecture'], native / 'final_model')
    require(pair['status'] == 'matched_full8_rank_pair_verified' and measured == pair['rank56']
            and measured['checkpoint_id'] == control['checkpoint_id'] == summary['checkpoint_id'] == CONTROL_ID
            and measured['mean_percent'] == summary['mean_percent'] == control['mean_percent']
            and measured['integer_counts'] == summary['integer_counts']
            and pair['rank56_minus_rank12'] == delta(pair['rank56'], pair['rank12'])
            and sha256(CONTROL / 'full8_summary.json') == control['full8_summary_sha256'], 'Wrong actual matched small evidence.')
    expected_actual = deepcopy(control['normalized_expected']); expected_actual['cat_args']['output_dir'] = str(native)
    require(read(native / 'normalized_cat_runtime_args.json') == expected_actual, 'Actual500 recipe changed.')
    argv = list(control['training_argv'][2:])
    for key, value in {'--output_dir': str(output), '--steps': 'default=500,after:gate_proj=3000'}.items():
        require(argv.count(key) == 1, 'Ambiguous original option: ' + key); argv[argv.index(key) + 1] = value
    expected = deepcopy(control['normalized_expected'])
    for section in ('cat_args', 'canonical_config'): expected[section]['output_dir'] = str(output)
    expected['canonical_config']['steps']['by_after_category'] = {'gate_proj': 3000}
    actual = normalized(argv); require(actual == expected, 'Native parsing changed beyond output and after-gate budget.')
    cat, _, _, vae = parse_cat_runtime_args(argv)
    stages = json.loads(json.dumps({category: asdict(cat.resolve_after_category_config(category)) for category in ORDER}))
    old_stages = deepcopy(control['resolved_recovery_stages']); old_stages['gate_proj']['opt']['steps'] = 3000
    require(stages == old_stages and stages['gate_proj']['lora']['rank'] == 56
            and stages['gate_proj']['lora']['alpha'] == 112 and cat.seed == stages['gate_proj']['data']['data_seed'] == 31,
            'Resolved recovery differs beyond gate3000.')
    boundary = Path(control['source_boundary']); source = resolve_cat_resume_source(str(boundary)); meta = source.model_checkpoint_meta
    require(source.source_kind == source.model_checkpoint_kind == 'category_boundary' and source.training_step_meta is None
            and source.model_checkpoint_dir == str(boundary) and meta['checkpoint_id'] == SOURCE_ID
            and meta['completed_categories'] == ORDER[:4] and len(meta['compressed_targets']) == 128
            and len(meta['pending_dense_targets']) == 96 and meta['lora_config'] is None,
            'Must restart the original after-o, never a rank56-produced boundary.')
    progress = load_cat_resume_distill_progress(str(boundary))
    require(list(progress.completed_categories) == ORDER[:4] and progress.lora_round_idx == 4
            and progress.active_category is None and progress.training_step_checkpoint is None
            and list(progress.distill_stage_history) == control['source_distill_history'], 'Wrong native resume progress.')
    payload = load_v6_cat_runtime_state(str(boundary), required=True)
    activation, plan, saved = restore_cat_runtime_state(payload)
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=resolve_category_runtime_configs(cat, vae, ORDER), compression_categories=ORDER,
        target_layers=parse_target_layers(cat.target_layers), skip_layers=parse_skip_layers(cat.skip_layers),
        transpose_modules=cat.transpose_modules.split(','))
    validate_cat_runtime_identity(saved, identity)
    require(identity == control['saved_runtime_identity'] and activation is not None and plan is None
            and activation['cache'].nsamples == len(activation['cache'].input_ids) == 512
            and len(activation['stats_by_linear']) == 224, 'Native runtime/activation identity changed.')
    log_path = native / 'linear_by_category.log'; sections = re.split(r'=== Category: (\w+) \(', log_path.read_text())
    require(sections[1::2] == ORDER[4:], 'Wrong measured continuation log.')
    runtimes, trainable, boundary_bytes = {}, {}, {}
    for category, text in zip(ORDER[4:], sections[2::2], strict=True):
        match = re.findall(r'LoRA train: step=500 train_loss=.*?train_runtime=([0-9.]+)', text)
        require(len(match) == 1, 'Missing measured500 timing: ' + category); runtimes[category] = float(match[0])
        counts = re.findall(r"parameter_counts=\{'lora': (\d+), 'decoder': (\d+)\}", text)
        require(len(counts) == 1, 'Missing actual FP32 component count: ' + category)
        trainable[category] = sum(map(int, counts[0]))
        boundary_bytes[category] = sum(p.stat().st_size for p in (native / ('after_' + category)).rglob('*') if p.is_file())
    measured_seconds = control['finished_unix'] - control['started_unix']
    extrapolated = measured_seconds + 5 * runtimes['gate_proj']
    g, u, d = [boundary_bytes[c] for c in ORDER[4:]]
    cg, cu, cd = [12 * trainable[c] for c in ORDER[4:]]  # FP32 trainable state + two FP32 Adam moments.
    disk_phases = {'gate_checkpoint_write_two_steps': g + 2 * cg,
        'gate_boundary_commit': 2 * g + cg,
        'up_checkpoint_write_old_gate_round_retained': 2 * g + u + cg + 2 * cu,
        'up_boundary_commit_before_old_round_prune': 2 * g + 2 * u + cg + cu,
        'down_checkpoint_write_old_up_round_retained': g + 2 * u + d + cu + 2 * cd,
        'down_boundary_commit_before_old_round_prune': g + 2 * u + 2 * d + cu + cd,
        'final_with_last_down_round': g + u + 3 * d + cd}
    provision = 40000000000
    require(max(disk_phases.values()) + 2 * 1024**3 < provision, 'Recompute provision for changed native geometry.')
    sources = dict(control['source_sha256']); inputs = dict(control['input_sha256'])
    for p in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
    evidence = [PAIR / 'comparison.json', PAIR / 'comparison.exit', *[CONTROL / f for f in
        ('launch.prepared.json', 'launch.json', 'full8_summary.json', 'mmlu.json', 'seven.json',
         'train.exit', 'mmlu.exit', 'seven.exit', 'summary.exit', 'run.exit', 'outer.exit')],
        native / 'normalized_cat_runtime_args.json', native / 'final_model/checkpoint_meta.json', log_path,
        *[native / ('after_' + c) / 'checkpoint_meta.json' for c in ORDER[4:]]]
    inputs.update({str(p): sha256(p) for p in evidence})
    fields = ('cwd', 'environment', 'architecture', 'categories', 'new_categories', 'source_boundary', 'source_checkpoint_id',
        'source_distill_history', 'control_final_acceptance', 'model_metadata_sha256', 'input_data_stat', 'source_weight_stat',
        'expected_user_linear_bits', 'original_linear_weights', 'estimated_bpw', 'resume_evidence', 'protocol', 'validation_reuse')
    record = {k: deepcopy(control[k]) for k in fields}
    record.update(schema='llama_remaining_gate3000_v1', status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        python_executable=sys.executable, output_dir=str(output), training_argv=[sys.executable, 'tools/cat_train.py', *argv],
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.llama_remaining_gate_steps.run', '--launch-record', str(output / 'launch.json')],
        normalized_expected=actual, resolved_recovery_stages=stages, after_gate_rank=56, after_gate_steps=3000,
        saved_runtime_identity=identity, source_sha256=sources, input_sha256=inputs,
        matched_control_launch=str(CONTROL / 'launch.json'), matched_control_checkpoint_id=CONTROL_ID,
        matched_control_result_paths=[str(CONTROL / (s + '.json')) for s in ('mmlu', 'seven')],
        matched_control_small_evidence_only=True, timeout_seconds=75600, termination_grace_seconds=20,
        pair_contract='Primary after-gate3000 versus actual rank56-500, both restart the same original after_o at seed31. Only after:gate steps and output change; up/down remain500/rank12, gate rank56/alpha112. Same public native implementation and fixed full8. Deployment external rank0.',
        exposure_boundary={'gate_sample_presentations': [2000, 12000], 'all_three_category_presentations': [6000, 16000],
            'gate_warmup_steps': [5, 30], 'maximum_gate_input_token_slots': [2048000, 12288000],
            'description': 'Lazy five-source weighted stream at B4/GAS1/dynamic max1024. Longer gate budget reads further stream data and stretches cosine/warmup; not six passes of fixed2000 examples, not exact optimizer/RNG resume, not same-data isolated updates. Later categories restart their original dataset iteration.'},
        resource_estimate={'actual_rank56_500_chain_seconds': measured_seconds, 'actual_recovery_seconds': runtimes,
            'same_mean_gate_time_extrapolated_chain_seconds': extrapolated, 'estimated_chain_seconds_range': [extrapolated, extrapolated * 1.25],
            'seconds_limit': 75600, 'per_arm_peak_generated_disk_bytes_estimate': provision,
            'measured_control_boundary_directory_bytes': boundary_bytes, 'actual_fp32_trainable_parameters': trainable,
            'phase_tensor_archive_byte_estimates': disk_phases, 'archive_workspace_headroom_bytes': 2 * 1024**3,
            'filesystem_free_at_preparation': shutil.disk_usage(PAIR.parent).free,
            'save_steps': 250, 'save_total_limit': 1, 'gate_optimizer_parameter_count': 64 * (14336 + 4096) * 56,
            'boundary': 'Round bases are saved before stage adapters and approximated by same-geometry observed boundary directories. Include previous round until new boundary commit, two step checkpoints during write, FP32 state+Adam, and2GiB archive/runtime/workspace headroom. These are estimates, not a measured peak; control payload sizes are observations, not future input dependencies. Step count increases saved checkpoint churn, not retained count. Longer stream may change token lengths/runtime. Root rechecks actual free disk and GPU5 owner completion before binding; no queue.'},
        experience='User asked whether500-step recovery is insufficient. Existing native per-after steps preserves cross-category identity; no new algorithm or GPU smoke needed. Rank56 versus12 yielded+.35694pp/M+166 but was mixed across tasks and does not predict long-budget gains.',
        cuda_initialized=False)
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print({'prepared_sha256': sha256(output / 'launch.prepared.json'), 'sources': len(sources), 'inputs': len(inputs),
        'source_stats': len(record['source_weight_stat']), 'data_stats': len(record['input_data_stat']),
        'expected_bpw': record['estimated_bpw'], 'extrapolated_seconds': extrapolated, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
