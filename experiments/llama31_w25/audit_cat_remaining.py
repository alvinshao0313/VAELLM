"""Audit the completed Llama CAT resource diagnostic, without quality evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import time

import torch

from experiments.llama31_w25.validate_cat_remaining import check_inputs, read_json
from experiments.w25_eval.acceptance import validate_checkpoint
from experiments.w25_eval.model_inventory import CATEGORIES, model_inventory, sha256
from rotation.model_utils import get_model
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import (
    load_v6_full_checkpoint_into_model, load_v6_training_model_state,
    load_v6_training_step_meta, resolve_training_step_round_base_ref,
)


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(record):
    check_inputs(record)
    resources = read_json(record['training_resource_report'])
    require(resources['status'] == 'completed' and resources['exit_code'] == 0,
            'Native CAT training did not finish successfully.')
    roots = list(Path(record['cat_output_root']).glob('*/normalized_cat_runtime_args.json'))
    require(len(roots) == 1, 'Require one actual native CAT run.')
    run = roots[0].parent
    native = read_json(roots[0])
    prepared = record['normalized_prepared_args']
    prepared['cat_args']['output_dir'] = str(run)
    require(native == prepared, 'Actual normalized CAT configuration differs from preparation.')
    architecture = model_inventory(record['source']['model_path'])
    final, meta, budget = validate_checkpoint(run / 'final_model', architecture)
    require(budget['budget_passed'] and budget['user_linear_bpw'] < 2.5,
            'Actual final checkpoint must remain strictly below 2.5 bpw.')
    require(meta['completed_categories'] == list(CATEGORIES), 'All seven categories must finish in order.')
    history = meta['extra_meta']['distill_stage_history']
    require(len(history) == len(CATEGORIES), 'Missing actual category recovery history.')
    layers = architecture['num_hidden_layers']
    history_summary = []
    for i, (category, row) in enumerate(zip(CATEGORIES, history)):
        selected = {name for name in architecture['projection_shapes'] if name.rsplit('.', 1)[1] in CATEGORIES[:i + 1]}
        remaining = set(architecture['projection_shapes']) - selected
        require(row['category'] == category and row['did_train']
                and row['mode'] == 'remaining_lora_prefix_decoder'
                and row['newly_compressed_target_count'] == layers
                and row['decoder_target_count'] == len(selected)
                and set(row['decoder_targets']) == selected
                and row['remaining_lora_target_count'] == len(remaining)
                and set(row['remaining_lora_targets']) == remaining
                and row['current_lora_target_count'] == 0, 'Wrong actual prefix/remaining stage history.')
        history_summary.append({key: row[key] for key in ('category', 'did_train', 'newly_compressed_target_count',
                                                        'decoder_target_count', 'remaining_lora_target_count')})
    log_path = run / 'linear_by_category.log'
    log = log_path.read_text()
    sections = re.split(r'=== Category: (\w+) \(', log)
    require(sections[1::2] == list(CATEGORIES), 'Missing/reordered native category log sections.')
    stage_logs = []
    for category, text in zip(sections[1::2], sections[2::2]):
        vae_rows = re.findall(r'\[(' + category + r'\.L0-31/stage[12])\] step=(\d+)/4 loss=([^ ]+)', text)
        require(Counter(tag for tag, step, loss in vae_rows) == {f'{category}.L0-31/stage1': 4,
                                                               f'{category}.L0-31/stage2': 4},
                f'Missing real VAE updates for {category}.')
        require(all(math.isfinite(float(loss)) for _, _, loss in vae_rows), 'Non-finite VAE loss.')
        kd_rows = re.findall(r'LoRA train: step=([1-4]) loss=([^ ]+)', text)
        require([int(step) for step, _ in kd_rows] == [1, 2, 3, 4]
                and all(math.isfinite(float(loss)) for _, loss in kd_rows), f'Missing/non-finite real KD updates: {category}.')
        tokens = re.findall(r'LoRA token stats: step=([1-4]) window_optimizer_steps=1 avg_prompt_tokens=([^ ]+) avg_response_tokens=([^ ]+) global_samples=(\d+)', text)
        require([int(x[0]) for x in tokens] == [1, 2, 3, 4] and all(int(x[3]) == 4 for x in tokens),
                f'Missing real B4 token windows for {category}.')
        stage_logs.append({'category': category, 'vae_updates_per_stage': 4, 'kd_optimizer_steps': 4,
                           'token_windows': [{'step': int(s), 'avg_prompt_tokens': float(p),
                                              'avg_response_tokens': float(r), 'samples': int(n)} for s, p, r, n in tokens]})
    checkpoint = run / 'training_rounds/0006_down_proj/trainer_state/checkpoint-4'
    state = read_json(checkpoint / 'trainer_state.json')
    require(state['global_step'] == state['max_steps'] == 4, 'Final native KD step must be four.')
    step_meta = load_v6_training_step_meta(str(checkpoint))
    source, base_meta = resolve_training_step_round_base_ref(str(checkpoint), step_meta)
    contract = step_meta['immutable_resume_contract']
    require(contract['active_category'] == 'down_proj' and not contract['lora_target_names']
            and set(contract['decoder_target_names']) == set(architecture['projection_shapes'])
            and step_meta['lora_config'] is None, 'Last down stage must train all decoders without remaining LoRA.')
    training, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
    require(manifest and all(row['component_class'] == 'decoder' for row in manifest)
            and all(t.dtype == torch.float32 for t in training.values()), 'Expected saved FP32 decoder-only last stage.')
    before = torch.load(Path(source) / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    saved = torch.load(final / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    require(set(before) == set(saved) and set(training) <= set(saved), 'Last-stage state key inventory changed.')
    fp32_changes = bf16_changes = 0
    for name, value in training.items():
        require(saved[name].dtype == torch.bfloat16 and torch.equal(saved[name], value.to(torch.bfloat16)),
                f'Decoder export differs from saved FP32 BF16 projection: {name}')
        fp32_changes += int(torch.count_nonzero(value != before[name].float()))
        bf16_changes += int(torch.count_nonzero(saved[name] != before[name]))
    require(fp32_changes > 0, 'No actual FP32 decoder update in final recovery stage.')
    frozen = set(saved) - set(training)
    require(all(torch.equal(saved[name], before[name]) for name in frozen), 'Last-stage nondecoder state changed.')
    code_count = 0
    for spec in meta['converted_modules']:
        for name, code in _code_specs(spec):
            require(saved[name].dtype == torch.uint8 and code['logical_shape'][-1] % 8 == 0
                    and torch.equal(saved[name], before[name]), 'Packed code state/layout changed in last KD stage.')
            code_count += 1
    require(code_count == 2 * architecture['projection_count'], 'Expected two packed banks per projection.')
    require(all(not any(marker in name for marker in ('.lora_A.', '.lora_B.', 'low_rank_a', 'low_rank_b', '_residual_lora_'))
                for name in saved), 'A live LoRA branch remains in the final checkpoint.')
    dense = get_model(record['source']['model_path'])
    dense_state = dense.state_dict()
    backbone = {key for key in dense_state if not any(key.startswith(name + '.') for name in architecture['projection_shapes'])}
    require(all(torch.equal(dense_state[key], saved[key]) for key in backbone), 'Frozen HF backbone changed during CAT.')
    del dense_state
    fresh, fresh_meta, load = load_v6_full_checkpoint_into_model(dense, str(final), expected_kind='final_model', strict=True)
    actual = fresh.state_dict()
    require(not load.missing_keys and not load.unexpected_keys and set(actual) == set(saved)
            and all(actual[key].dtype == saved[key].dtype and torch.equal(actual[key], saved[key]) for key in saved),
            'Fresh native strict load differs from every saved tensor.')
    require(fresh_meta['checkpoint_id'] == meta['checkpoint_id'], 'Fresh checkpoint identity differs.')
    check_inputs(record)
    return {'status': 'verified', 'quality_evaluated': False, 'diagnostic_not_formal_initial': True,
            'run_dir': str(run), 'checkpoint_id': meta['checkpoint_id'], 'architecture': architecture,
            'bit_budget': budget, 'category_history': history_summary, 'actual_training_logs': stage_logs,
            'last_stage': {'global_step': 4, 'decoder_fp32_changed_elements': fp32_changes,
                'decoder_bf16_changed_elements': bf16_changes, 'decoder_bf16_projection_exact': True,
                'frozen_tensors': len(frozen), 'code_banks_frozen': code_count, 'code_xor': 0},
            'fresh_strict_state': {'tensor_count': len(saved), 'all_tensors_exact': True,
                                  'original_hf_backbone_tensors_exact': len(backbone), 'lora_rank': 0},
            'training_resources': resources, 'evidence_sha256': {str(p): sha256(p) for p in
                (roots[0], log_path, final / 'checkpoint_meta.json', checkpoint / 'trainer_state.json',
                 checkpoint / 'checkpoint_meta.json', Path(source) / 'checkpoint_meta.json')},
            'limits': ['Earlier round states are pruned by the original save_total_limit=1 path; only their ordinary logs/history are checked.',
                       'No resume test or full downstream quality evaluation. Four-step dynamic padding does not establish a worst-case 1024-token memory bound.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    args = parser.parse_args()
    record = read_json(args.launch)
    output = Path(record['audit_report'])
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    started = time.monotonic()
    result = audit(record)
    result['audit_elapsed_seconds'] = time.monotonic() - started
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'bpw': result['bit_budget']['user_linear_bpw'],
                      'full_state_tensors': result['fresh_strict_state']['tensor_count']}))


if __name__ == '__main__':
    main()
