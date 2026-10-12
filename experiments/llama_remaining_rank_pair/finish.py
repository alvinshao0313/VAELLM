"""Verify the resumed native trajectory and compare its complete eight tasks."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import re

import torch

from experiments.full_rht_gate_pair.compare import delta, score
from experiments.llama_remaining_rank_pair.prepare import ORDER, check, read, require
from experiments.longbench_v1.common import sha256, write_json
from experiments.w25_eval.acceptance import validate_checkpoint
from train_utils.checkpoint_v6 import validate_v6_meta


def verify_native(record, directory):
    directory = Path(directory).resolve(); native = directory.parent
    expected = deepcopy(record['normalized_expected']); expected['cat_args']['output_dir'] = str(native)
    require(read(native / 'normalized_cat_runtime_args.json') == expected, 'Actual native recipe differs.')
    meta = validate_v6_meta(read(directory / 'checkpoint_meta.json'), expected_kind='final_model')
    require(meta['completed_categories'] == meta['compression_categories'] == ORDER and meta['lora_config'] is None
            and len(meta['compressed_targets']) == 224 and not meta['pending_dense_targets'],
            'Final seven-category topology or external rank differs.')
    history = meta['extra_meta']['distill_stage_history']
    require(len(history) == 7 and history[:4] == record['source_distill_history'], 'Resumed history prefix changed.')
    boundaries = {}
    for index, category in enumerate(ORDER[4:], start=4):
        row = history[index]; rank = record['after_gate_rank'] if index == 4 else 12
        require(row['category'] == category and row['did_train'] and row['mode'] == 'remaining_lora_prefix_decoder'
                and row['newly_compressed_target_count'] == 32 and row['decoder_target_count'] == (index + 1) * 32
                and row['remaining_lora_target_count'] == (6 - index) * 32 and row['current_lora_target_count'] == 0
                and row['resolved_distill_lr'] == 1e-4 and row['resolved_decoder_lr'] == 1e-5,
                'Actual resumed recovery target inventory or LR differs.')
        path = native / ('after_' + category)
        boundary = validate_v6_meta(read(path / 'checkpoint_meta.json'), expected_kind='category_boundary')
        require(boundary['completed_categories'] == ORDER[:index + 1]
                and len(boundary['compressed_targets']) == (index + 1) * 32
                and len(boundary['pending_dense_targets']) == (6 - index) * 32
                and boundary['lora_config'] is None and (path / boundary['state_dict_file']).is_file()
                and boundary['extra_meta']['recovery_lora_config'] == {'rank': rank, 'alpha': float(rank * 2), 'dropout': .1},
                'Required native boundary or actual recovery rank/alpha differs.')
        boundaries[category] = {'path': str(path), 'checkpoint_id': boundary['checkpoint_id'],
            'metadata_sha256': sha256(path / 'checkpoint_meta.json'), 'remaining_lora_count': (6 - index) * 32,
            'temporary_recovery_rank': rank}
    log = (native / 'linear_by_category.log').read_text()
    sections = re.split(r'=== Category: (\w+) \(', log)
    require(sections[1::2] == ORDER[4:] and 'Restored exact cross-category CAT runtime state: activation=True global_plan=False' in log,
            'Native resume did not restore runtime state or skip completed categories.')
    for category, text in zip(ORDER[4:], sections[2::2], strict=True):
        for stage in (1, 2):
            require(f'[{category}.L0-31/stage{stage}] step=10000/10000' in text,
                    'Incomplete real VAE training: ' + category)
        require('LoRA train: step=500 train_loss=' in text, 'Incomplete native 500-step recovery: ' + category)
    return meta, boundaries


def finish(record_path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Finish on CPU only.')
    record = read(record_path); output = Path(record['output_dir']); check(record)
    directory = Path(record['final_checkpoint_dir']); meta, boundaries = verify_native(record, directory)
    _, _, budget = validate_checkpoint(directory, record['architecture'])
    require(budget['user_linear_bits'] == budget['packed_linear_bits'] == record['expected_user_linear_bits']
            and budget['original_linear_weights'] == record['original_linear_weights'] and budget['budget_passed']
            and not budget['contract_violations'] and 2 * budget['user_linear_bits'] < 5 * budget['original_linear_weights'],
            'Actual merged endpoint must remain strictly below 2.5bpw.')
    candidate, raw = score([output / 'mmlu.json', output / 'seven.json'], record['architecture'], directory)
    old = read(record['control_final_acceptance'])
    historical, old_raw = score([Path(row['path']) for row in old['compressed']['input_results']],
                                record['architecture'], Path(old['checkpoint_dir']))
    for a, b in zip(raw, old_raw, strict=True):
        for key in ('configs', 'versions', 'n_shot', 'n_samples'):
            require(a[key] == b[key], 'Actual full8 protocol differs: ' + key)
    summary = {'status': 'full8_resume_rank_verified', **candidate, 'bit_budget': budget,
        'after_gate_rank': record['after_gate_rank'], 'source_checkpoint_id': record['source_checkpoint_id'],
        'native_category_boundaries': boundaries, 'vs_historical3577': delta(candidate, historical),
        'historical_mean_percent': historical['mean_percent'], 'baseline_mean_percent': old['baseline']['mean_percent'],
        'actual_normalized_sha256': sha256(directory.parent / 'normalized_cat_runtime_args.json'),
        'checkpoint_meta_sha256': sha256(directory / 'checkpoint_meta.json'),
        'prepared_sha256': sha256(output / 'launch.prepared.json'), 'pair_contract': record['pair_contract'],
        'resume_evidence': record['resume_evidence'], 'protocol': record['protocol'], 'cuda_initialized': False,
        'goal_acceptance': False, 'longbench_status': 'not_evaluated'}
    check(record); require(not (output / 'full8_summary.json').exists(), 'Do not overwrite actual results.')
    write_json(output / 'full8_summary.json', summary)
    print({'checkpoint_id': meta['checkpoint_id'], 'mean_percent': candidate['mean_percent'],
        'mmlu_correct': candidate['integer_counts']['mmlu']['correct'], 'bpw': budget['user_linear_bpw']}, flush=True)


def compare(root):
    root = Path(root).resolve(); records = [read(root / f'rank{r}/launch.json') for r in (12, 56)]
    scores, raws, pins = [], [], {}
    for record in records:
        output = Path(record['output_dir']); check(record)
        require(record['status'] == 'completed' and record['exit_code'] == 0
                and int((output / 'run.exit').read_text()) == 0, 'Both actual arms must complete.')
        verify_native(record, Path(record['final_checkpoint_dir']))
        value, raw = score([output / 'mmlu.json', output / 'seven.json'], record['architecture'], Path(record['final_checkpoint_dir']))
        summary = read(output / 'full8_summary.json')
        require(value['mean_percent'] == summary['mean_percent'] and value['integer_counts'] == summary['integer_counts']
                and sha256(output / 'full8_summary.json') == record['full8_summary_sha256'], 'Actual scores or summary changed.')
        scores.append(value); raws.append(raw)
        for name in ('launch.prepared.json', 'launch.json', 'full8_summary.json', 'mmlu.json', 'seven.json', 'run.exit'):
            pins[str(output / name)] = sha256(output / name)
    for key in ('source_checkpoint_id', 'source_weight_stat', 'source_sha256', 'input_sha256', 'environment', 'protocol', 'saved_runtime_identity'):
        require(records[0][key] == records[1][key], 'Pair input or implementation differs: ' + key)
    a, b = [deepcopy(record['normalized_expected']) for record in records]
    for value in (a, b):
        for section in ('canonical_config', 'cat_args'):
            value[section]['output_dir'] = '<arm>'
        for key in ('lora_rank', 'lora_alpha'):
            value['canonical_config'][key]['by_after_category']['gate_proj'] = '<rank-variable>'
    require(a == b, 'Pair differs beyond after-gate rank/alpha.')
    for left, right in zip(*raws, strict=True):
        for key in ('configs', 'versions', 'n_shot', 'n_samples'):
            require(left[key] == right[key], 'Matched full8 protocol differs: ' + key)
    require(not (root / 'comparison.json').exists(), 'Do not overwrite a comparison.')
    write_json(root / 'comparison.json', {'status': 'matched_full8_rank_pair_verified', 'rank12': scores[0], 'rank56': scores[1],
        'rank56_minus_rank12': delta(scores[1], scores[0]), 'evidence_sha256': pins,
        'goal_acceptance': False, 'longbench_status': 'not_evaluated'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch-record', type=Path); group.add_argument('--compare-root', type=Path)
    args = parser.parse_args()
    finish(args.launch_record) if args.launch_record else compare(args.compare_root)
