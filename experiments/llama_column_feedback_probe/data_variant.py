"""Fixed FineWeb calibration and old/new observations for the existing column probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from experiments.llama_column_feedback_probe.data import MAX_LENGTH, SEED
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.distill_data import build_distill_data_collator, encode_canonical_record

FINEWEB_RANGES = {'train': (650000, 650128), 'fineweb_hold32': (700000, 700032),
                  'fineweb_observe32': (700032, 700064)}
OLD_DOMAINS = ('alpaca_hold32', 'fineweb_hold32')
NEW_DOMAINS = ('alpaca_observe32', 'fineweb_observe32')


def build_variant_data(tokenizer, previous_batches, previous_data):
    train, heldout, alpaca = build_calibration_batches(tokenizer, seed=SEED,
        train_documents=128, heldout_documents=64, max_length=MAX_LENGTH)
    require(all(torch.equal(train[key], previous_batches['train'][key]) for key in train),
            'The same-seed Alpaca first128 no longer exactly match; stop without selecting alternatives.')
    require(all(torch.equal(heldout[key][:32], previous_batches['alpaca_hold32'][key]) for key in heldout),
            'The same-seed Alpaca first held32 no longer exactly match; stop without selecting alternatives.')
    require(alpaca['train']['documents'] == previous_data['alpaca']['train']['documents']
            and alpaca['heldout']['documents'][:32] == previous_data['alpaca']['heldout']['documents'],
            'Original Alpaca row identities or grouping changed.')
    all_alpaca = alpaca['train']['documents'] + alpaca['heldout']['documents']
    require(len({row['group_sha256'] for row in all_alpaca}) == 192,
            'Original Alpaca calibration and old/new observations overlap in instruction/input groups.')
    samples = {domain: [] for domain in FINEWEB_RANGES}
    fineweb = {domain: [] for domain in FINEWEB_RANGES}
    fineweb_path = Path(previous_data['fineweb_file'])
    with fineweb_path.open('rb') as stream:
        for index, line in enumerate(stream):
            if index >= max(stop for _, stop in FINEWEB_RANGES.values()):
                break
            selected = [domain for domain, (start, stop) in FINEWEB_RANGES.items() if start <= index < stop]
            if not selected:
                continue
            require(len(selected) == 1, 'Fixed FineWeb physical ranges overlap.')
            domain = selected[0]
            raw = json.loads(line)
            require(set(raw) == {'text'} and isinstance(raw['text'], str) and raw['text'].strip(),
                    'A predetermined FineWeb row is invalid; no skipping or replacement.')
            sample = encode_canonical_record(raw, tokenizer, text_format='text', text_field='text',
                                             task='lm', model_max_length=MAX_LENGTH)
            require(sample is not None and len(sample['input_ids']) >= 2
                    and torch.equal(sample['labels'], sample['input_ids']), 'Invalid fixed native LM sample.')
            samples[domain].append(sample)
            fineweb[domain].append({'row_index': index, 'raw_line_sha256': hashlib.sha256(line).hexdigest(),
                'raw_text_sha256': hashlib.sha256(raw['text'].encode()).hexdigest(),
                'normalized_text_group_sha256': hashlib.sha256(' '.join(raw['text'].split()).encode()).hexdigest(),
                'encoded': {key: tensor_record(value) for key, value in sample.items()},
                'valid_tokens': len(sample['input_ids'])})
    collator = build_distill_data_collator(tokenizer, model_max_length=MAX_LENGTH, dynamic_padding=False)
    collated = {}
    for domain, (start, stop) in FINEWEB_RANGES.items():
        require(len(samples[domain]) == stop - start, 'Every predetermined FineWeb row is required: ' + domain)
        value = collator(samples[domain])
        collated[domain] = {key: value[key] for key in ('input_ids', 'attention_mask')}
    require(all(torch.equal(value, previous_batches['fineweb_hold32'][key])
                for key, value in collated['fineweb_hold32'].items()), 'Old FineWeb observations changed.')
    require([{key: value for key, value in row.items() if key != 'normalized_text_group_sha256'}
             for row in fineweb['fineweb_hold32']] == previous_data['fineweb'], 'Old FineWeb source rows changed.')
    fineweb_groups = [row['normalized_text_group_sha256'] for rows in fineweb.values() for row in rows]
    require(len(set(fineweb_groups)) == len(fineweb_groups),
            'Fixed FineWeb calibration and observations overlap in normalized text; no replacement.')
    batches = {'train': collated['train'],
        'alpaca_hold32': {key: value.clone() for key, value in previous_batches['alpaca_hold32'].items()},
        'fineweb_hold32': {key: value.clone() for key, value in previous_batches['fineweb_hold32'].items()},
        'alpaca_observe32': {key: value[32:].clone() for key, value in heldout.items()},
        'fineweb_observe32': collated['fineweb_observe32']}
    tokens, counts = {}, {}
    for domain, batch in {**batches, 'previous_alpaca_train_reference_only': train}.items():
        expected = 128 if domain in ('train', 'previous_alpaca_train_reference_only') else 32
        require(all(tuple(value.shape) == (expected, MAX_LENGTH) for value in batch.values()),
                'Keep original B1/256 native forward shapes: ' + domain)
        tokens[domain] = [tensor_record(ids[mask.bool()])['sha256'] for ids, mask in
                          zip(batch['input_ids'], batch['attention_mask'])]
        require(len(set(tokens[domain])) == expected, 'Duplicate fixed token sequences: ' + domain)
        valid = int(batch['attention_mask'].sum())
        counts[domain] = {'documents': expected, 'valid_tokens': valid, 'causal_tokens': valid - expected}
    domains = list(tokens)
    for i, left in enumerate(domains):
        for right in domains[i + 1:]:
            require(not set(tokens[left]) & set(tokens[right]),
                    'Fixed calibration/observation/reference token sequences overlap: ' + left + '/' + right)
    report = {'seed': SEED, 'max_length': MAX_LENGTH, 'train_documents': 128,
        'holdout_documents_per_domain': 32, 'development_domains': list(OLD_DOMAINS),
        'new_observation_domains': list(NEW_DOMAINS), 'calibration_domain': 'FineWeb',
        'alpaca': alpaca, 'alpaca_roles': {'train': 'Previous128 calibration, reference-only and not run this variant.',
            'heldout_first32': 'Reused development observations.', 'heldout_last32': 'New fixed observations.'},
        'fineweb': fineweb, 'fineweb_file': str(fineweb_path),
        'fineweb_physical_rows': {domain: list(pair) for domain, pair in FINEWEB_RANGES.items()},
        'token_sha256': {domain: tokens[domain] for domain in batches}, 'counts': counts,
        'token_batches': {domain: {key: tensor_record(value) for key, value in batch.items()}
                         for domain, batch in batches.items()},
        'split_checks': {'old_alpaca128_and_held32_exact': True, 'old_fineweb32_exact': True,
            'alpaca_instruction_input_groups_disjoint': True, 'fineweb_normalized_text_groups_disjoint': True,
            'all_domains_and_previous_alpaca_train_token_sequences_disjoint': True,
            'heuristic_replacement_or_posthoc_skipping': False},
        'boundary': 'Old two hold32 sets are reused development observations after v2 results were read. '
                    'New Alpaca32 and FineWeb32 are fixed before this variant and excluded from all its code selection. '
                    'Non-exposure in all historical recovery/pretraining is not established. Calibration domain and '
                    'effective token count both change; this is not a domain-only single-variable contrast.'}
    return batches, report
