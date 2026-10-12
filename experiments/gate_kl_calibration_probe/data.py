"""Fixed 128-document calibration variants and disjoint local observation domains."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from datasets import Dataset
import torch
from transformers import AutoTokenizer

from experiments.gate_tail_objective_probe.data import (
    DATA, MAX_LENGTH, ROOT, prepare_data as prepare_tail_data,
)
from experiments.longbench_v1.common import check_hashes, check_stat, sha256
from experiments.qwen3_w25.calibration import _sha256, build_calibration_batches
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.distill_data import (
    _instruction_prompt_and_response, build_distill_data_collator, encode_canonical_record,
)

DOMAINS = ('alpaca128', 'alpaca64_fineweb64', 'train32', 'alpaca_held32',
           'fineweb_rows500000_500031', 'alpaca_confirmation32', 'fineweb_rows500032_500063')
KEYS = ('input_ids', 'attention_mask')


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def text_identities(raw, rendered):
    return {'raw_text_utf8_sha256': text_sha(raw), 'rendered_utf8_sha256': text_sha(rendered),
            'raw_whitespace_normalized_sha256': text_sha(' '.join(raw.split())),
            'rendered_whitespace_normalized_sha256': text_sha(' '.join(rendered.split()))}


def take(batch, indices):
    return {key: value[indices].clone() for key, value in batch.items() if key in KEYS}


def concat(*batches):
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in KEYS}


def fineweb_batches(tokenizer, sampler_rows, old_batch, old_documents):
    """Read only fixed physical rows; reject invalid rows instead of replacing them."""
    path = DATA / 'train.jsonl'
    train_indices = [row['row_index'] for row in sampler_rows]
    new_indices = list(range(500032, 500064))
    require(len(train_indices) == len(set(train_indices)) == 64
            and not set(train_indices) & set(range(500000, 500064)), 'FineWeb train/observation indices overlap.')
    selected = set(train_indices + list(range(500000, 500064)))
    last_selected = max(selected); originals = {}; line_hashes = {}
    with path.open('rb') as stream:
        for index, line in enumerate(stream):
            if index in selected:
                row = json.loads(line)
                require(set(row) == {'text'} and isinstance(row['text'], str) and row['text'].strip(),
                        'A fixed FineWeb document is invalid; do not filter or replace it.')
                originals[index] = row
                line_hashes[index] = hashlib.sha256(line).hexdigest()
            if index >= last_selected:
                break
    require(set(originals) == selected, 'Missing selected FineWeb physical rows.')
    samples, documents = {}, {}
    for index in train_indices + new_indices:
        row = originals[index]; rendered = row['text'].strip()
        sample = encode_canonical_record(row, tokenizer, text_format='text', text_field='text',
                                         task='lm', model_max_length=MAX_LENGTH)
        require(sample is not None and torch.equal(sample['labels'], sample['input_ids'])
                and sample['attention_mask'].eq(1).all(), 'FineWeb native LM labels/encoding changed.')
        ids = tokenizer(rendered, add_special_tokens=False, return_attention_mask=False,
                        return_token_type_ids=False)['input_ids']
        had_eos = bool(ids) and ids[-1] == tokenizer.eos_token_id
        payload = ids[:-1] if had_eos else ids
        require(sample['input_ids'].tolist() == payload[:255] + [tokenizer.eos_token_id],
                'Native LM max-minus-one plus terminal-EOS semantics changed.')
        size = sample['input_ids'].numel()
        require(2 <= size <= MAX_LENGTH, 'A fixed document has no causal label.')
        samples[index] = sample
        documents[index] = {'source': 'fineweb', 'row_index': index,
            'document_identity': f'fineweb:{index}', 'jsonl_line_sha256': line_hashes[index],
            **text_identities(row['text'], rendered), 'token_sha256': _sha256(sample['input_ids'].tolist()),
            'valid_tokens': size, 'causal_tokens': size - 1, 'raw_token_count': len(ids),
            'had_terminal_eos': had_eos, 'truncated_payload_tokens': max(0, len(payload) - 255)}
    for saved in sampler_rows:
        current = documents[saved['row_index']]
        require(current['raw_text_utf8_sha256'] == saved['text_sha256']
                and current['raw_token_count'] == saved['raw_token_length']
                and current['had_terminal_eos'] == saved['had_terminal_eos'],
                'Native seed0 sampler row identity differs from its existing1024 CPU evidence.')
    collator = build_distill_data_collator(tokenizer, model_max_length=MAX_LENGTH, dynamic_padding=False)
    groups = []
    for indices in (train_indices, new_indices):
        batch = collator([samples[index] for index in indices]); mask = batch['attention_mask'].bool()
        require(batch['labels'][~mask].eq(-100).all()
                and torch.equal(batch['labels'][mask], batch['input_ids'][mask]), 'LM padding labels changed.')
        groups.append(({key: batch[key] for key in KEYS}, [documents[i] for i in indices]))
    # Reuse the old encoded32, while reading their text in this same physical
    # pass so cross-domain normalized-text checks also cover these documents.
    old = []
    for position, item in enumerate(old_documents):
        length = item['valid_tokens']; row = dict(item); raw_text = originals[item['row_index']]['text']
        require(text_sha(raw_text) == item['raw_text_utf8_sha256']
                and text_sha(raw_text.strip()) == item['canonical_text_utf8_sha256']
                and line_hashes[item['row_index']] == item['jsonl_line_sha256'], 'Old FineWeb32 raw identity changed.')
        row.update(source='fineweb', document_identity=f"fineweb:{item['row_index']}",
                   token_sha256=_sha256(old_batch['input_ids'][position, :length].tolist()))
        row.update(text_identities(raw_text, raw_text.strip()))
        old.append(row)
    return groups[0], groups[1], old, Path(inspect.getfile(type(collator))).resolve()


def prepare_data(base_root=ROOT / 'result/qwen3_8b_mmlu65_20261004'):
    """CPU replay only; every train/evaluation identity is fixed before scoring."""
    old_data, old_batches, archive = prepare_tail_data(base_root)
    source = old_data['source_data']
    tokenizer = AutoTokenizer.from_pretrained(source['model_path'], local_files_only=True,
                                             use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    first, new_alpaca, alpaca = build_calibration_batches(tokenizer, seed=137,
        train_documents=192, heldout_documents=32, max_length=MAX_LENGTH)
    all_alpaca = concat(first, new_alpaca)
    original_documents = source['documents']['train'] + source['documents']['heldout'] + source['documents']['replay32']
    documents = alpaca['train']['documents'] + alpaca['heldout']['documents']
    require(len(documents) == 224 and all(all(item[key] == previous[key] for key in item)
            for item, previous in zip(documents[:96], original_documents, strict=True)),
            'The first96 accepted Alpaca identities must exactly replay the existing development pool.')
    for start, split in ((0, 'train'), (32, 'heldout'), (64, 'replay32')):
        require({key: tensor_record(value[start:start+32]) for key, value in all_alpaca.items()}
                == source['token_batches'][split], 'The original96 native token batches changed.')
    raw = Dataset.from_file(alpaca['source_arrow']); alpaca_documents = []
    for position, (item, ids) in enumerate(zip(documents, all_alpaca['input_ids'], strict=True)):
        row = dict(raw[item['row_index']]); prompt, response = _instruction_prompt_and_response(row, text_format='alpaca')
        require(_sha256([row[key] for key in ('instruction', 'input', 'output')]) == item['content_sha256']
                and _sha256(ids[:item['valid_tokens']].tolist()) == item['token_sha256'], 'Actual Alpaca content/tokens changed.')
        alpaca_documents.append({**item, 'source': 'alpaca', 'accepted_position': position,
            'document_identity': f"alpaca:{item['row_index']}", 'causal_tokens': item['valid_tokens'] - 1,
            **text_identities(str(row['text']), prompt + response)})
    statistics = json.loads((DATA / 'canonical_statistics.json').read_text())
    sampler = statistics['sampler']; sampler_rows = statistics['models']['qwen3']['sampler_first8000']['rows'][:64]
    require(sampler['class'] == 'accelerate.data_loader.SeedableRandomSampler' and sampler['data_seed'] == 0
            and sampler['epoch'] == 0 and sampler['world_size'] == 1 and not sampler['group_by_length']
            and not sampler['replacement'] and sampler['native_trainer_use_seedable_sampler'], 'Fixed original seed0 sampler evidence differs.')
    (fw_train, fw_train_docs), (fw_new, fw_new_docs), fw_old_docs, collator_source = fineweb_batches(
        tokenizer, sampler_rows, old_batches['fineweb_rows500000_500031'],
        old_data['documents']['fineweb_rows500000_500031'])
    pure_positions = list(range(32)) + list(range(96, 192)); mixed_positions = pure_positions[:64]
    pure = take(all_alpaca, pure_positions); mixed = concat(take(all_alpaca, mixed_positions), fw_train)
    batches = dict(zip(DOMAINS, (pure, mixed, take(all_alpaca, list(range(32))),
        {k: old_batches['alpaca_held32'][k] for k in KEYS},
        {k: old_batches['fineweb_rows500000_500031'][k] for k in KEYS}, new_alpaca, fw_new)))
    records = dict(zip(DOMAINS, ([alpaca_documents[i] for i in pure_positions],
        [alpaca_documents[i] for i in mixed_positions] + fw_train_docs, alpaca_documents[:32],
        alpaca_documents[32:64], fw_old_docs, alpaca_documents[192:224], fw_new_docs)))
    unique = alpaca_documents + fw_train_docs + fw_old_docs + fw_new_docs
    token_seen, text_seen = {}, {}
    require(len(unique) == 352 and len({d['document_identity'] for d in unique}) == 352, 'Unique registry identity overlap.')
    for item in unique:
        identity = item['document_identity']; token = item['token_sha256']
        require(token not in token_seen, 'Effective token sequence collides across fixed unique documents.')
        token_seen[token] = identity
        hashes = {value for key, value in item.items() if key in (
            'raw_text_utf8_sha256', 'rendered_utf8_sha256', 'raw_whitespace_normalized_sha256',
            'rendered_whitespace_normalized_sha256')}
        require(not hashes & text_seen.keys(), 'Raw/rendered/normalized text collides across fixed unique documents.')
        text_seen.update({value: identity for value in hashes})
    lengths = {name: batch['attention_mask'].sum(1).tolist() for name, batch in batches.items()}
    counts = dict(zip(DOMAINS, (128, 128, 32, 32, 32, 32, 32)))
    require(all(set(batch) == set(KEYS) and all(tuple(t.shape) == (counts[name],256) and t.device.type == 'cpu' for t in batch.values())
            and lengths[name] == [d['valid_tokens'] for d in records[name]] for name, batch in batches.items()), 'Fixed256 native batch boundaries differ.')
    causal = {name: sum(value) - len(value) for name, value in lengths.items()}
    require(causal['alpaca128'] == 18582 and causal['alpaca64_fineweb64'] == 25718
            and sum(d['causal_tokens'] for d in fw_train_docs) == 15829, 'Actual fixed128 token counts differ from CPU design.')
    sources = dict(old_data['source_sha256'])
    for path in (Path(__file__).resolve(), ROOT/'experiments/qwen3_w25/calibration.py',
                 ROOT/'train_utils/distill_data.py', ROOT/'e2e_common/data.py', collator_source):
        value = sha256(path); require(str(path) not in sources or sources[str(path)] == value, 'Existing source changed.')
        sources[str(path)] = value
    inputs = dict(old_data['input_sha256']); check_hashes(sources); check_hashes(inputs)
    for stat in old_data['input_data_stat']: check_stat(stat)
    record = {'format': 'gate_kl_calibration_data_v1', 'status': 'cpu_ready',
        'source_data': source, 'old_data': old_data, 'source_sha256': sources, 'input_sha256': inputs,
        'input_data_stat': old_data['input_data_stat'], 'target': source['target'], 'domains': list(DOMAINS),
        'training_domains': list(DOMAINS[:2]), 'observation_domains': list(DOMAINS[2:]),
        'documents': records, 'document_lengths': lengths, 'causal_tokens': causal,
        'valid_tokens': {name: sum(values) for name, values in lengths.items()},
        'token_batches': {name: {k: tensor_record(v) for k,v in batch.items()} for name,batch in batches.items()},
        'fineweb_sampler': {'evidence_path': str(DATA/'canonical_statistics.json'), 'sampler': sampler,
            'selected_first64_physical_rows': [r['row_index'] for r in sampler_rows], 'selection_filtering': False},
        'token_weight': {'alpaca128': {'alpaca': 1.0, 'fineweb': 0.0}, 'alpaca64_fineweb64':
            {'alpaca': 9889/25718, 'fineweb': 15829/25718},
            'denominator': 'All nonpadding causal tokens, not an equal-document/source mean.'},
        'deduplication': {'unique_documents_checked':352, 'alpaca_accepted_pool':224, 'fineweb_train_old_new': [64,32,32],
            'unique_effective_token_sequences':352, 'raw_rendered_and_whitespace_text_no_collision':True,
            'allowed_reuse': 'Identical fixed Alpaca64 shared between training arms; train32 repeats first32 for endpoint observation only.',
            'registry': unique, 'collision_policy': 'Stop; never skip/reorder/replace a fixed document.'},
        'protocol': {'seed':137,'max_length':256,'microbatch_size':1,'padding':'Original right-padded native256 shapes.',
            'alpaca128_positions': pure_positions, 'mixed_alpaca_positions':mixed_positions,
            'new_alpaca_observation_positions':list(range(192,224)), 'new_fineweb_observation_rows':list(range(500032,500064)),
            'no_prior_heldout_or_replay_in_training':True,
            'encoding':'Alpaca native LM rendered prompt+response; FineWeb native plain-text LM, at most255 payload tokens plus EOS; no packing/chat conversion.',
            'boundary':'Local calibration/observation only, not a downstream benchmark. Old observations are already-used development data; new rows are disjoint from this local pool only, with no pretraining/general-LM exposure exclusion claim.'}}
    return record, batches, archive
