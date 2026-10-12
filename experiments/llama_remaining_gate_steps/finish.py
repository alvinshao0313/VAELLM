"""Verify the resumed native trajectory and compare the after-gate3000 budget with rank56-500."""
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
        steps = record['resolved_recovery_stages'][category]['opt']['steps']
        require(steps == (3000 if category == 'gate_proj' else 500)
                and f'LoRA train: step={steps} train_loss=' in text, 'Incomplete declared native recovery: ' + category)
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
    control = read(record['matched_control_launch'])
    matched, matched_raw = score(record['matched_control_result_paths'], record['architecture'], Path(control['final_checkpoint_dir']))
    require(matched['checkpoint_id'] == record['matched_control_checkpoint_id'], 'Wrong rank56-500 control identity.')
    for a, b in zip(raw, matched_raw, strict=True):
        for key in ('configs', 'versions', 'n_shot', 'n_samples'):
            require(a[key] == b[key], 'Primary matched full8 protocol differs: ' + key)
    summary = {'status': 'full8_resume_budget_verified', 'primary_3000_minus_500': delta(candidate, matched),
        'matched_control': matched, **candidate, 'bit_budget': budget,
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch-record', type=Path, required=True)
    finish(parser.parse_args().launch_record)
