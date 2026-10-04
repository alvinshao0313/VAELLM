"""Check the existing canonical SFT path with actual Llama/Qwen tokenizers.

This diagnostic changes no tokenizer templates, training code, or loss masks.
Its independent oracle uses explicit role/end markers in the complete render.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

import torch
from transformers import AutoTokenizer

from e2e_common.chat_template_utils import render_messages
from train_utils.config.configs import DistillDataConfig
from train_utils.distill_data import (
    build_distill_data_collator, build_distill_dataset, encode_canonical_record,
)


CASES = {
    'single_turn': [('user', 'Which integer follows forty?'), ('assistant', 'Forty-one.')],
    'explicit_system': [('system', 'Give a concise answer.'),
                        ('user', 'Which color is a clear daytime sky?'), ('assistant', 'Blue.')],
    'two_turn': [('user', 'Which integer precedes nine?'), ('assistant', 'Eight.'),
                 ('user', 'Which integer follows ten?'), ('assistant', 'Eleven.')],
}


def check_tokenizer(path: str, family: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = {}
    for name, turns in CASES.items():
        messages = [{'role': role, 'content': content} for role, content in turns]
        rendered = render_messages(messages, tokenizer, add_generation_prompt=False)
        tokens = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = tokens['input_ids']
        expected_mask = [False] * len(ids)
        cursor = 0
        turn_rows = []
        for role, content in turns:
            header = (f'<|start_header_id|>{role}<|end_header_id|>\n\n' if family == 'llama'
                      else f'<|im_start|>{role}\n')
            ending = '<|eot_id|>' if family == 'llama' else '<|im_end|>\n'
            start = rendered.index(header, cursor) + len(header)
            end_start = rendered.index(ending, start)
            end = end_start + len(ending)
            cursor = end
            content_start = rendered.index(content, start, end_start)
            content_positions = [i for i, (left, right) in enumerate(tokens['offset_mapping'])
                                 if right > content_start and left < content_start + len(content)]
            end_positions = [i for i, (left, right) in enumerate(tokens['offset_mapping'])
                             if right > end_start and left < end]
            assert content_positions and end_positions
            if role == 'assistant':
                for i, (left, right) in enumerate(tokens['offset_mapping']):
                    if right > left and right > start and left < end:
                        expected_mask[i] = True
            turn_rows.append({'role': role, 'content_positions': content_positions,
                              'ending_positions': end_positions})
        expected_ids = list(ids)
        if expected_ids[-1] == tokenizer.eos_token_id:
            expected_mask[-1] = True
        else:
            expected_ids.append(tokenizer.eos_token_id)
            expected_mask.append(True)
        actual = encode_canonical_record({'messages': messages}, tokenizer,
                                         text_format='edgerazor_messages', text_field='text',
                                         task='sft', model_max_length=1024)
        assert actual is not None, (family, name, 'unexpected skipped sample')
        assert actual['input_ids'].tolist() == expected_ids, (family, name, 'input mismatch')
        labels = actual['labels'].tolist()
        expected_labels = [token if keep else -100 for token, keep in zip(expected_ids, expected_mask)]
        assert labels == expected_labels, (family, name, 'canonical labels differ from complete-render oracle')
        for row in turn_rows:
            trainable = row['role'] == 'assistant'
            assert all((labels[i] != -100) == trainable for i in row['content_positions'])
            assert all((labels[i] != -100) == trainable for i in row['ending_positions'])
        results[name] = {'tokens': len(expected_ids), 'response_tokens': sum(expected_mask),
                         'input_sha256': hashlib.sha256(actual['input_ids'].numpy().tobytes()).hexdigest(),
                         'labels_sha256': hashlib.sha256(actual['labels'].numpy().tobytes()).hexdigest(),
                         'exact_full_render_mask': True, 'turns': turn_rows}
    config = DistillDataConfig(dataset_mix='vaellm_eval_task=1', dataset_task='sft',
                               model_max_length=1024, seed=0, data_seed=0)
    bundle = build_distill_dataset(config, tokenizer)
    samples = list(islice(iter(bundle.train_dataset), 3))
    assert len(samples) == 3
    originals = [{key: value.clone() for key, value in sample.items()} for sample in samples]
    batch = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)(samples)
    for i, sample in enumerate(originals):
        length = sample['input_ids'].numel()
        assert torch.equal(batch['input_ids'][i, :length], sample['input_ids'])
        assert torch.equal(batch['labels'][i, :length], sample['labels'])
        assert (batch['labels'][i, length:] == -100).all()
        assert (batch['attention_mask'][i, :length] == 1).all()
        assert (batch['attention_mask'][i, length:] == 0).all()
        assert int((sample['labels'][:-1] != -100).sum()) > 0
        assert int(sample['labels'][-1]) == tokenizer.eos_token_id
    builder = {'dataset_mix': config.dataset_mix, 'source_stats': bundle.source_stats,
               'is_iterable': bundle.is_iterable, 'group_by_length': bundle.group_by_length,
               'sample_tokens': [s['input_ids'].numel() for s in originals],
               'sample_response_tokens': [int((s['labels'] != -100).sum()) for s in originals],
               'batch_shape': list(batch['input_ids'].shape),
               'collator_preserves_labels_and_masks_padding': True}
    return {'tokenizer_path': path, 'tokenizer_class': type(tokenizer).__name__,
            'cases': results, 'actual_builder_collator': builder}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-tokenizer', required=True)
    parser.add_argument('--qwen-tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = {'status': 'verified', 'weights_loaded': False,
              'models': {'llama': check_tokenizer(args.llama_tokenizer, 'llama'),
                         'qwen': check_tokenizer(args.qwen_tokenizer, 'qwen')},
              'cuda_initialized': torch.cuda.is_initialized(),
              'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                  ('train_utils/distill_data.py', 'e2e_common/data.py',
                   'e2e_common/chat_template_utils.py', __file__)},
              'scope': 'Existing canonical encoder; legacy TRL helper is not used by current CAT/E2E entrypoints.',
              'training_code_modified': False}
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Verified Llama and Qwen canonical masks: three cases each; no shared source changes.')


if __name__ == '__main__':
    main()
