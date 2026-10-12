"""Fixed old Alpaca heldout and physical FineWeb rows for read-only tail scoring."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.gate_student_context_probe.data import (
    ROOT, SOURCE_ID, TARGET, prepare_data as prepare_student_data,
)
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.distill_data import build_distill_data_collator, encode_canonical_record


DOMAINS = ('alpaca_held32', 'fineweb_rows500000_500031')
ROW_START, ROW_STOP, MAX_LENGTH = 500000, 500032, 256
DATA = ROOT / 'data/fineweb_edu_10bt_20261005'
MANIFEST_SHA = 'b229051e173a1f9d37138d102ce8fa6658350a3ba4096be0750a96b1df43b6a8'
STATISTICS_SHA = '377b249f16dc0be228b9e3713332207697dc49eb7f4c6859dd9fa09a0e4226dd'


def _bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def _fineweb_batch(tokenizer, path):
    samples, documents = [], []
    with path.open('rb') as stream:
        for row_index, line in enumerate(stream):
            if row_index < ROW_START:
                continue
            if row_index >= ROW_STOP:
                break
            row = json.loads(line)
            require(set(row) == {'text'} and isinstance(row['text'], str) and row['text'].strip(),
                    'A preselected FineWeb row is invalid; do not skip or replace it.')
            sample = encode_canonical_record(row, tokenizer, text_format='text', text_field='text',
                                             task='lm', model_max_length=MAX_LENGTH)
            require(sample is not None and torch.equal(sample['labels'], sample['input_ids'])
                    and sample['attention_mask'].eq(1).all(), 'Native LM encoding or labels changed.')
            raw_ids = tokenizer(row['text'].strip(), add_special_tokens=False,
                                return_attention_mask=False, return_token_type_ids=False)['input_ids']
            had_eos = bool(raw_ids) and raw_ids[-1] == tokenizer.eos_token_id
            payload = raw_ids[:-1] if had_eos else raw_ids
            require(sample['input_ids'].tolist() == payload[:MAX_LENGTH - 1] + [tokenizer.eos_token_id],
                    'Native LM truncate/terminal-EOS behavior differs.')
            length = sample['input_ids'].numel()
            require(2 <= length <= MAX_LENGTH, 'A selected document has no valid causal label.')
            documents.append({'row_index': row_index, 'jsonl_line_sha256': _bytes_sha(line),
                'raw_text_utf8_sha256': _bytes_sha(row['text'].encode('utf-8')),
                'canonical_text_utf8_sha256': _bytes_sha(row['text'].strip().encode('utf-8')),
                'canonical_strip_changed': row['text'] != row['text'].strip(),
                'raw_token_count': len(raw_ids), 'had_terminal_eos': had_eos,
                'truncated_payload_tokens': max(0, len(payload) - (MAX_LENGTH - 1)),
                'valid_tokens': length, 'causal_tokens': length - 1,
                'encoded_tensors': {key: tensor_record(value) for key, value in sample.items()}})
            samples.append(sample)
    require(len(samples) == 32 and [d['row_index'] for d in documents] == list(range(ROW_START, ROW_STOP)),
            'The exact 32 physical FineWeb rows are required, without filtering.')
    collator = build_distill_data_collator(tokenizer, model_max_length=MAX_LENGTH, dynamic_padding=False)
    batch = collator(samples)
    active = batch['attention_mask'].bool()
    require(all(tuple(batch[key].shape) == (32, MAX_LENGTH) for key in ('input_ids', 'attention_mask', 'labels'))
            and batch['labels'][~active].eq(-100).all()
            and torch.equal(batch['labels'][active], batch['input_ids'][active]),
            'Fixed padding must preserve native LM labels and valid-token identities.')
    return batch, documents, Path(inspect.getfile(type(collator))).resolve()


def prepare_data(base_root):
    """Return CPU batches and the existing HF L18 archive; never capture or fit."""
    base = Path(base_root).resolve()
    source_data, previous, archive = prepare_student_data(base)
    require(source_data['checkpoint_id'] == SOURCE_ID and source_data['target'] == TARGET,
            'Use the original f927 L18 diagnostic source.')
    heldout = previous['heldout']
    require({key: tensor_record(value) for key, value in heldout.items()}
            == source_data['token_batches']['heldout'], 'The old held32 token batch changed.')
    manifest_path, statistics_path = DATA / 'manifest.json', DATA / 'canonical_statistics.json'
    require(sha256(manifest_path) == MANIFEST_SHA and sha256(statistics_path) == STATISTICS_SHA,
            'The original FineWeb conversion/canonical evidence changed.')
    manifest, statistics = (json.loads(path.read_text()) for path in (manifest_path, statistics_path))
    require(manifest['status'] == 'ready_for_canonical_cpu_statistics'
            and manifest['counts']['rows'] == 726000 and manifest['exact_text_roundtrip_each_row']
            and statistics['status'] == 'verified' and not statistics['cuda_initialized'],
            'Require the complete, unchanged original text shard and its native CPU evidence.')
    path = Path(manifest['jsonl_path']).resolve()
    require(path == DATA / 'train.jsonl' and path.stat().st_size == manifest['jsonl_bytes'],
            'The original JSONL identity or size differs.')
    # The running experiment already bound the large file after full SHA validation.
    # Read only its immutable data-stat value; do not pin its mutable launch file.
    general_launch = base / 'general_lm_recovery_s2000/qwen3/launch.json'
    original_stat = json.loads(general_launch.read_text())['input_data_stat']
    require(original_stat == [file_stat(path)], 'FineWeb differs from the already bound original data file.')
    tokenizer = AutoTokenizer.from_pretrained(source_data['model_path'], local_files_only=True,
                                             use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    fineweb, fine_documents, collator_source = _fineweb_batch(tokenizer, path)
    batches = dict(zip(DOMAINS, (heldout, fineweb)))
    documents = {DOMAINS[0]: source_data['documents']['heldout'], DOMAINS[1]: fine_documents}
    lengths = {name: batch['attention_mask'].sum(1).tolist() for name, batch in batches.items()}
    require(all(len(lengths[name]) == 32 and lengths[name] == [d['valid_tokens'] for d in documents[name]]
                and tuple(batch['input_ids'].shape) == (32, MAX_LENGTH)
                and all(value.device.type == 'cpu' for value in batch.values())
                for name, batch in batches.items()), 'Document boundaries, batch shape or CPU placement differ.')
    sources = dict(source_data['source_sha256'])
    for source in (Path(__file__).resolve(), ROOT / 'train_utils/distill_data.py',
                   ROOT / 'e2e_common/data.py', collator_source):
        digest = sha256(source)
        require(str(source) not in sources or sources[str(source)] == digest, 'Existing source pin changed.')
        sources[str(source)] = digest
    inputs = dict(source_data['input_sha256'])
    inputs.update({str(manifest_path): MANIFEST_SHA, str(statistics_path): STATISTICS_SHA})
    record = {'format': 'gate_tail_objective_data_v1', 'status': 'cpu_ready',
        'source_data': source_data, 'source_sha256': sources, 'input_sha256': inputs,
        'input_data_stat': original_stat, 'domains': list(DOMAINS), 'target': TARGET,
        'documents': documents, 'document_lengths': lengths,
        'token_batches': {name: {key: tensor_record(value) for key, value in batch.items()}
                          for name, batch in batches.items()},
        'valid_tokens': {name: sum(values) for name, values in lengths.items()},
        'causal_tokens': {name: sum(values) - len(values) for name, values in lengths.items()},
        'fineweb_source': {key: manifest[key] for key in ('repo', 'revision', 'config', 'split',
            'original_shard', 'jsonl_path', 'jsonl_sha256', 'jsonl_bytes', 'parquet_sha256')},
        'protocol': {'max_length': MAX_LENGTH, 'documents_per_domain': 32,
            'alpaca': 'Exact old held32 native LM batch, unchanged formatting/order/tokens; previously used development documents.',
            'fineweb': 'Physical JSONL rows 500000..500031, zero-based, fixed before scoring; no filtering, packing or shuffle.',
            'encoding': 'Native encode_canonical_record(lm,text); strip text, no BOS/template, at most 255 payload tokens plus terminal EOS.',
            'padding': 'Right to 256; original Alpaca padding unchanged; native FineWeb collator labels padding -100.',
            'scoring': 'Both domains use input_ids[:,1:] and attention_mask[:,1:] for all nonpadding causal positions; no prompt mask.',
            'usage': 'Endpoint measurement only; no updates or parameter selection from these results.',
            'independence_boundary': 'No claim of exclusion from original pretraining or the concurrent general-LM training sampler.'}}
    check_hashes(sources); check_hashes(inputs); check_stat(original_stat[0])
    return record, batches, archive
