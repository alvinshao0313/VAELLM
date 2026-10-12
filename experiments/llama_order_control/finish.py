"""Verify the real native CAT endpoint and aggregate its matched full-eight evaluation."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.compare import delta, score
from experiments.llama_order_control.prepare import ORDER, check, read, require
from experiments.longbench_v1.common import sha256, write_json
from experiments.w25_eval.acceptance import validate_checkpoint
from train_utils.checkpoint_v6 import validate_v6_meta


def verify_native(record, directory):
    directory = Path(directory).resolve(); native = directory.parent
    expected = copy.deepcopy(record['normalized_expected'])
    expected['cat_args']['output_dir'] = str(native)
    require(read(native / 'normalized_cat_runtime_args.json') == expected, 'Actual native configuration differs from the prepared order-only recipe.')
    meta = validate_v6_meta(read(directory / 'checkpoint_meta.json'), expected_kind='final_model')
    require(meta['completed_categories'] == meta['compression_categories'] == ORDER and meta['lora_config'] is None,
            'Final categories/order or external LoRA topology differ.')
    history = meta['extra_meta']['distill_stage_history']
    require(len(history) == 7, 'Missing native category recovery history.')
    boundaries = {}
    for index, category in enumerate(ORDER):
        row = history[index]
        require(row['category'] == category and row['did_train'] and row['mode'] == 'remaining_lora_prefix_decoder'
                and row['newly_compressed_target_count'] == 32 and row['decoder_target_count'] == (index + 1) * 32
                and row['remaining_lora_target_count'] == (6 - index) * 32,
                'Native prefix/remaining-dense recovery inventory differs.')
        path = native / ('after_' + category)
        boundary = validate_v6_meta(read(path / 'checkpoint_meta.json'), expected_kind='category_boundary')
        require(boundary['completed_categories'] == ORDER[:index + 1]
                and len(boundary['compressed_targets']) == (index + 1) * 32
                and len(boundary['pending_dense_targets']) == (6 - index) * 32
                and (path / boundary['state_dict_file']).is_file(), 'Required original category checkpoint is missing or incomplete.')
        boundaries[category] = {'path': str(path), 'checkpoint_id': boundary['checkpoint_id'],
            'metadata_sha256': sha256(path / 'checkpoint_meta.json'), 'compressed_count': (index + 1) * 32,
            'pending_dense_count': (6 - index) * 32}
    return meta, boundaries


def finish(record_path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Finish on CPU only.')
    record = read(record_path); output = Path(record['output_dir']); check(record)
    directory = Path(record['final_checkpoint_dir'])
    _, boundaries = verify_native(record, directory)
    directory, meta, budget = validate_checkpoint(directory, record['architecture'])
    require(budget['user_linear_bits'] == budget['packed_linear_bits'] == record['expected_user_linear_bits']
            and budget['original_linear_weights'] == record['original_linear_weights'] and budget['budget_passed']
            and not budget['contract_violations'] and 2 * budget['user_linear_bits'] < 5 * budget['original_linear_weights'],
            'Actual integer storage must match the order-invariant budget and strict <2.5bpw.')
    candidate, raw = score([output / 'mmlu.json', output / 'seven.json'], record['architecture'], directory)
    source = read(record['control_final_acceptance'])
    control, old_raw = score([Path(row['path']) for row in source['compressed']['input_results']],
        record['architecture'], Path(source['checkpoint_dir']))
    for a, b in zip(raw, old_raw, strict=True):
        for field in ('configs', 'versions', 'n_shot', 'n_samples'):
            require(a[field] == b[field], 'Actual paired full-eight protocol differs: ' + field)
    summary = {'status': 'full8_order_comparison_verified', **candidate, 'bit_budget': budget,
        'control_checkpoint_id': control['checkpoint_id'], 'control_mean_percent': control['mean_percent'],
        'vs_original_order': delta(candidate, control), 'baseline_mean_percent': source['baseline']['mean_percent'],
        'native_category_boundaries': boundaries, 'pair_contract': record['pair_contract'],
        'actual_normalized_sha256': sha256(directory.parent / 'normalized_cat_runtime_args.json'),
        'checkpoint_meta_sha256': sha256(directory / 'checkpoint_meta.json'),
        'prepared_sha256': sha256(output / 'launch.prepared.json'), 'source_sha256': record['source_sha256'],
        'protocol': record['protocol'], 'goal_acceptance': False, 'longbench_status': 'not_evaluated',
        'scope': 'Fresh original-HF category-order comparison; accuracy is evidence, never an execution-success gate.',
        'cuda_initialized': False}
    check(record); require(not (output / 'full8_summary.json').exists(), 'Do not overwrite completed evidence.')
    write_json(output / 'full8_summary.json', summary)
    print({'checkpoint_id': meta['checkpoint_id'], 'mean_percent': candidate['mean_percent'],
        'mmlu_correct': candidate['integer_counts']['mmlu']['correct'], 'bpw': budget['user_linear_bpw'],
        'delta_pp': summary['vs_original_order']['mean_delta_pp']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    finish(parser.parse_args().launch_record)
