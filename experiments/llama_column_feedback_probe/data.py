"""Fresh, fixed probe splits using existing Alpaca and native FineWeb encoders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.distill_data import build_distill_data_collator, encode_canonical_record

SEED, MAX_LENGTH = 1060616, 256
TRAIN_DOCUMENTS, HELDOUT_DOCUMENTS = 128, 32
FINEWEB_START, FINEWEB_STOP = 700000, 700032
DOMAINS = ('train', 'alpaca_hold32', 'fineweb_hold32')


def build_data(tokenizer, fineweb_path):
    train, heldout, alpaca = build_calibration_batches(tokenizer, seed=SEED,
        train_documents=TRAIN_DOCUMENTS, heldout_documents=HELDOUT_DOCUMENTS, max_length=MAX_LENGTH)
    samples, fineweb = [], []
    with Path(fineweb_path).open('rb') as stream:
        for index, line in enumerate(stream):
            if index < FINEWEB_START:
                continue
            if index >= FINEWEB_STOP:
                break
            raw = json.loads(line)
            require(set(raw) == {'text'} and isinstance(raw['text'], str) and raw['text'].strip(),
                    'A fixed FineWeb row is invalid; no skipping or substitution.')
            sample = encode_canonical_record(raw, tokenizer, text_format='text', text_field='text',
                                             task='lm', model_max_length=MAX_LENGTH)
            require(sample is not None and len(sample['input_ids']) >= 2
                    and torch.equal(sample['labels'], sample['input_ids']), 'Invalid native causal LM sample.')
            samples.append(sample)
            fineweb.append({'row_index': index, 'raw_line_sha256': hashlib.sha256(line).hexdigest(),
                'raw_text_sha256': hashlib.sha256(raw['text'].encode()).hexdigest(),
                'encoded': {key: tensor_record(value) for key, value in sample.items()},
                'valid_tokens': len(sample['input_ids'])})
    require(len(samples) == HELDOUT_DOCUMENTS, 'The exact predetermined FineWeb rows are required.')
    collator = build_distill_data_collator(tokenizer, model_max_length=MAX_LENGTH, dynamic_padding=False)
    collated = collator(samples)
    batches = {'train': train, 'alpaca_hold32': heldout,
               'fineweb_hold32': {key: collated[key] for key in ('input_ids', 'attention_mask')}}
    tokens = {}
    for domain, batch in batches.items():
        size = TRAIN_DOCUMENTS if domain == 'train' else HELDOUT_DOCUMENTS
        require(all(tuple(value.shape) == (size, MAX_LENGTH) for value in batch.values()),
                'All domains must use the same fixed B1/256 native forward shape.')
        tokens[domain] = [tensor_record(ids[mask.bool()])['sha256'] for ids, mask in
                          zip(batch['input_ids'], batch['attention_mask'])]
        require(len(set(tokens[domain])) == len(tokens[domain]), 'Duplicate token sequences in a fixed split.')
    for i, left in enumerate(DOMAINS):
        for right in DOMAINS[i + 1:]:
            require(not set(tokens[left]) & set(tokens[right]), 'Calibration and observation token sequences overlap.')
    require(not {row['group_sha256'] for row in alpaca['train']['documents']}
            & {row['group_sha256'] for row in alpaca['heldout']['documents']},
            'Alpaca instruction/input groups cross calibration and holdout.')
    report = {'seed': SEED, 'max_length': MAX_LENGTH, 'train_documents': TRAIN_DOCUMENTS,
              'holdout_documents_per_domain': HELDOUT_DOCUMENTS, 'alpaca': alpaca, 'fineweb': fineweb,
              'fineweb_file': str(Path(fineweb_path).resolve()), 'fineweb_physical_rows': [FINEWEB_START, FINEWEB_STOP],
              'token_sha256': tokens, 'token_batches': {domain: {key: tensor_record(value) for key, value in batch.items()}
                                                     for domain, batch in batches.items()},
              'boundary': 'Holdouts are newly fixed for this probe and excluded from its calibration and code selection. '
                          'Historical recovery/pretraining non-exposure is not established. No heldout chooses codes, '
                          'damping, feedback groups, iterations, or stopping.'}
    return batches, report
