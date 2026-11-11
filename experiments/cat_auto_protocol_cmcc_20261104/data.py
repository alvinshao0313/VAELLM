"""Train-row identities around the canonical sampler, renderer and collator."""
from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .contracts import canonical_json, digest, require
from .identity import semantic_seed

MIX = {'edgerazor_ii_7m': .341, 'edgerazor_ii_gen': .067,
       'edgerazor_tulu': .028, 'edgerazor_am': .064, 'vaellm_eval_task': .500}
ROW_KEY = '__cat_auto_row__'
DATA_VERSION = 'canonical_indexed_artifact_rows_v0'
INPUT_FIELDS = ('input_ids', 'attention_mask', 'labels', 'position_ids', 'cache_position')


def file_sha256(path):
    path = Path(path)
    before = path.stat()
    hasher = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
        (after.st_size, after.st_mtime_ns, after.st_ino), 'Source changed while hashing: ' + str(path))
    return hasher.hexdigest()


def tensor_digest(values):
    """Bind values and padding, independently of the CPU/CUDA location."""
    hasher = sha256()
    for key in sorted(values):
        value = values[key].detach().cpu().contiguous()
        header = canonical_json(dict(key=key, shape=list(value.shape), dtype=str(value.dtype)))
        hasher.update(header.encode('utf-8'))
        hasher.update(value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def batch_digest(inputs):
    keys = ('input_ids', 'attention_mask', 'labels')
    require(all(key in inputs for key in keys), 'Bind causal tokens, attention and response labels together.')
    require(not (set(inputs) - set(INPUT_FIELDS)), 'The native data adapter received unsupported model input fields.')
    return tensor_digest({key: inputs[key] for key in INPUT_FIELDS if key in inputs})


def tokenizer_manifest(tokenizer):
    require(tokenizer.is_fast and tokenizer.eos_token_id is not None,
        'The v0 data adapter requires the actual fast tokenizer and terminal EOS.')
    return dict(backend_sha256=sha256(tokenizer.backend_tokenizer.to_str().encode('utf-8')).hexdigest(),
        chat_template_digest=digest(tokenizer.chat_template),
        eos_token_id=int(tokenizer.eos_token_id), pad_token_id=tokenizer.pad_token_id,
        added_vocabulary_digest=digest(tokenizer.get_added_vocab()),
        padding_side=str(tokenizer.padding_side), canonical_format='vaellm_distill_canonical_v1')


@dataclass
class BoundSource:
    raw: Any
    preset: Any
    identity: dict

    @property
    def alias(self):
        return str(self.preset.alias)

    def row_reference(self, index, record):
        require(0 <= index < len(self.raw), 'Raw row index lies outside its bound source.')
        source_id = digest(self.identity)
        namespace = digest(dict(version='artifact_split_config_rows_v0',
            artifact_sha256=self.identity['artifact_sha256'], split=self.identity['split'],
            config=self.identity['config']))
        return dict(record_id=digest(dict(record_namespace=namespace, raw_index=index)),
            record_namespace=namespace,
            source_id=source_id, alias=self.alias, raw_index=index, raw_digest=digest(record))


def bind_source(raw, preset, *, revision, license_record, provenance_record, artifact_digest):
    require(preset.train_split in ('train', 'auxiliary_train'), 'Only train/auxiliary_train may be bound.')
    require(isinstance(revision, str) and bool(revision) and license_record and provenance_record,
        'Record the actual source revision, license and split provenance before use.')
    require(isinstance(artifact_digest, str) and len(artifact_digest) == 64
        and all(c in '0123456789abcdef' for c in artifact_digest), 'Source needs an explicit SHA256 binding.')
    require(len(raw) > 0 and isinstance(raw._fingerprint, str), 'Use the actual map-style HF train source.')
    require(raw._indices is None, 'Bind original artifact rows before any Dataset.select/shuffle/filter index view.')
    identity = dict(version=DATA_VERSION, uri=str(preset.path), config=preset.config,
        split=str(preset.train_split), revision=revision, license=license_record,
        provenance=provenance_record, artifact_sha256=artifact_digest,
        fingerprint=raw._fingerprint, rows=len(raw), text_format=str(preset.text_format),
        text_field=str(preset.text_field))
    return BoundSource(raw, preset, identity)


def load_local_source(alias, *, revision, license_record, provenance_record, expected_sha256):
    """Load only a registered local train file; never download or infer provenance."""
    from e2e_common.data import DATASET_MIX_SOURCE_PRESETS, _load_preset_raw_datasets, _resolve_dataset_path

    require(alias in DATASET_MIX_SOURCE_PRESETS, 'Unknown source alias: ' + alias)
    preset = DATASET_MIX_SOURCE_PRESETS[alias]
    path = Path(_resolve_dataset_path(preset.path))
    require(path.is_file(), 'The local source is missing; no network fallback: ' + str(path))
    actual = file_sha256(path)
    require(actual == expected_sha256, 'Train source SHA256 differs from its recorded artifact: ' + str(path))
    raw, _unused_eval = _load_preset_raw_datasets(preset)
    return bind_source(raw, preset, revision=revision, license_record=license_record,
        provenance_record=provenance_record, artifact_digest=actual)


class _TaggedRows(Dataset):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source.raw)

    def __getitem__(self, index):
        record = dict(self.source.raw[int(index)])
        require(ROW_KEY not in record, 'The raw source contains the reserved provenance field.')
        return {**record, ROW_KEY: self.source.row_reference(int(index), record)}


def raw_stream(sources, weights, seed):
    """Use the existing permutation and weighted indexed scheduler without reseeding globals."""
    from e2e_common.lazy_datasets import _IndexedMixedRawStream, _PermutedRawDatasetView, _build_permutation_indices

    require(sources and len(sources) == len(weights), 'Explicit nonempty sources and aligned weights required.')
    require(len({source.alias for source in sources}) == len(sources), 'Repeated source alias.')
    require(all(float(weight) > 0 for weight in weights), 'Source weights must be positive.')
    raw = [_TaggedRows(source) for source in sources]
    permutations = [_build_permutation_indices(len(value), seed=int(seed)) for value in raw]
    if len(raw) == 1:
        return _PermutedRawDatasetView(raw[0], permutations[0])
    total = sum(float(weight) for weight in weights)
    return _IndexedMixedRawStream(raw, permutations, [float(weight) / total for weight in weights], seed=int(seed))


class BoundCanonicalIterable(IterableDataset):
    def __init__(self, sources, tokenizer, *, weights, seed, excluded_ids=(), trace=False):
        super().__init__()
        self.sources = tuple(sources)
        self.raw_dataset = raw_stream(sources, weights, seed)
        self.presets = {source.alias: source.preset for source in sources}
        self.tokenizer = tokenizer
        self.excluded_ids = frozenset(excluded_ids)
        self.trace = bool(trace)

    def __iter__(self):
        from e2e_common.lazy_datasets import _iter_raw_records_for_worker
        from train_utils.distill_data import encode_canonical_record

        for tagged in _iter_raw_records_for_worker(self.raw_dataset):
            record = dict(tagged)
            ref = record.pop(ROW_KEY)
            if ref['record_id'] in self.excluded_ids:
                continue
            preset = self.presets[ref['alias']]
            sample = encode_canonical_record(record, self.tokenizer, text_format=preset.text_format,
                text_field=preset.text_field, task='sft', model_max_length=1024)
            # This is the canonical renderer-returned None path, not another filter.
            if sample is not None:
                yield {**sample, ROW_KEY: ref} if self.trace else sample


class _TraceCollator:
    def __init__(self, tokenizer):
        from train_utils.distill_data import build_distill_data_collator
        self.collator = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)

    def __call__(self, rows):
        return dict(inputs=self.collator([{key: value for key, value in row.items() if key != ROW_KEY}
            for row in rows]), records=[row[ROW_KEY] for row in rows])


def kd_data_seed(model_identity, category_id, search_seed):
    return semantic_seed(model_identity, category_id, 'recovery', 'data', search_seed)


def trace_updates(sources, tokenizer, *, model_identity, category_id, search_seed,
                  steps=1000, workers=16):
    require(steps in (500, 1000), 'Reserve a legal original schedule prefix, not a rescaled trajectory.')
    require(workers >= 0, 'Invalid DataLoader worker count.')
    weights = [MIX[source.alias] for source in sources]
    seed = kd_data_seed(model_identity, category_id, search_seed)
    dataset = BoundCanonicalIterable(sources, tokenizer, weights=weights, seed=seed, trace=True)
    loader = DataLoader(dataset, batch_size=4, num_workers=workers, collate_fn=_TraceCollator(tokenizer),
        generator=torch.Generator().manual_seed(seed), pin_memory=False)
    iterator = iter(loader)
    batches = []
    try:
        for index in range(steps):
            batch = next(iterator)
            require(len(batch['records']) == 4, 'The actual update stream ended before its reserved prefix.')
            batches.append(dict(update_index=index, input_sha256=batch_digest(batch['inputs']),
                shape=list(batch['inputs']['input_ids'].shape), records=batch['records']))
    finally:
        if workers:
            iterator._shutdown_workers()
    return dict(category_id=category_id, data_seed=seed, schedule_reserved=steps, batch=4,
        max_length=1024, workers=workers, group_by_length=False, dynamic_padding=True,
        batches=batches, token_order_digest=digest(batches))


def calibration_blocks(source, tokenizer, *, blocks=512, length=256):
    """The original seed31 text stream and packing, with raw-row contributors."""
    from e2e_common.chat_template_utils import render_messages
    from e2e_common.data import _normalize_edgerazor_messages, _record_to_text

    require(blocks > 0 and length > 0, 'Positive packed calibration dimensions required.')
    stream = raw_stream([source], [1.], 31)
    buffer, spans, result, contributors = [], deque(), [], []
    for tagged in stream:
        record = dict(tagged)
        ref = record.pop(ROW_KEY)
        if source.preset.text_format == 'edgerazor_messages':
            messages = _normalize_edgerazor_messages(record)
            if messages is None:
                continue
            text = render_messages(messages, tokenizer).strip()
        else:
            value = _record_to_text(record, text_field=source.preset.text_field,
                text_format=source.preset.text_format)
            text = '' if value is None else str(value).strip()
        if not text:
            continue
        tokens = tokenizer(text + '\n\n', add_special_tokens=False, return_attention_mask=False,
            return_token_type_ids=False)['input_ids']
        if not tokens:
            continue
        buffer.extend(int(value) for value in tokens)
        spans.append([len(tokens), ref])
        while len(buffer) >= length and len(result) < blocks:
            result.append(torch.tensor(buffer[:length], dtype=torch.long).unsqueeze(0))
            del buffer[:length]
            remaining, refs = length, []
            while remaining:
                count, origin = spans[0]
                refs.append(origin)
                used = min(count, remaining)
                remaining -= used
                if used == count:
                    spans.popleft()
                else:
                    spans[0][0] -= used
            contributors.append(refs)
        if len(result) == blocks:
            break
    require(len(result) == blocks, 'The actual calibration source has insufficient packed tokens.')
    return result, dict(seed=31, blocks=blocks, seqlen=length, records_by_block=contributors,
        token_digest=tensor_digest({'input_ids': torch.cat(result)}))


def select_proposal(sources, tokenizer, *, model_identity, search_seed, excluded_ids, count=512):
    require(count == 512, 'The v0 common proposal-select set contains exactly512 records.')
    weights = [MIX[source.alias] for source in sources]
    seed = semantic_seed(model_identity, '__shared__', 'proposal', 'data', search_seed)
    dataset = BoundCanonicalIterable(sources, tokenizer, weights=weights, seed=seed,
        excluded_ids=excluded_ids, trace=True)
    selected, seen = [], set()
    for sample in dataset:
        ref = sample[ROW_KEY]
        if ref['record_id'] in seen:
            continue
        seen.add(ref['record_id'])
        selected.append(sample)
        if len(selected) == count:
            break
    require(len(selected) == count, 'Not enough distinct legal train records remain for proposal-select.')
    counts = Counter(sample[ROW_KEY]['alias'] for sample in selected)
    signature = [dict(record=sample[ROW_KEY], input_sha256=batch_digest(
        {key: value.unsqueeze(0) for key, value in sample.items() if key != ROW_KEY})) for sample in selected]
    return selected, dict(version=DATA_VERSION, data_seed=seed, records=signature,
        source_counts=dict(counts), count=count, halves=[[0, 256], [256, 512]],
        max_length=1024, token_order_digest=digest(signature),
        aggregation='sum(token_KL * (response_mask + .3*prompt_mask)) / sum(mask_weight); source weights are in sampling only')


def freeze_data_plan(sources, calibration_source, tokenizer, *, model_identity, category_ids, search_seed):
    require(tuple(source.alias for source in sources) == tuple(MIX),
        'Keep the original five-source enumeration as well as its weights; source order is part of the indexed sampler.')
    require(len(category_ids) == len(set(category_ids)) and category_ids, 'Freeze unique semantic category IDs.')
    calibration, calibration_manifest = calibration_blocks(calibration_source, tokenizer)
    updates = {category: trace_updates(sources, tokenizer, model_identity=model_identity,
        category_id=category, search_seed=search_seed) for category in category_ids}
    excluded = {ref['record_id'] for refs in calibration_manifest['records_by_block'] for ref in refs}
    excluded.update(ref['record_id'] for update in updates.values() for batch in update['batches'] for ref in batch['records'])
    proposal, proposal_manifest = select_proposal(sources, tokenizer, model_identity=model_identity,
        search_seed=search_seed, excluded_ids=excluded)
    require(not ({sample[ROW_KEY]['record_id'] for sample in proposal} & excluded), 'Train/proposal ID overlap.')
    manifest = dict(version=DATA_VERSION, model_identity=model_identity, search_seed=search_seed,
        tokenizer=tokenizer_manifest(tokenizer), sources=[source.identity for source in sources],
        calibration_source=calibration_source.identity, source_weights=MIX,
        calibration=calibration_manifest, updates=updates, proposal=proposal_manifest,
        reservation='All semantic categories reserve the original T1000 data prefix before offline search; unused tails do not count as actual updates.')
    return dict(manifest=manifest, data_identity=digest(manifest), calibration=calibration,
        proposal=proposal, excluded_record_ids=sorted(excluded))


def build_update_bundle(prepared, sources, tokenizer, category_id):
    """The native trainer consumes the same frozen stream, without provenance input keys."""
    from train_utils.distill_data import DistillDatasetBundle

    manifest = prepared['manifest']
    require(manifest['version'] == DATA_VERSION and digest(manifest) == prepared['data_identity'],
        'Prepared source plan identity or record namespace changed.')
    require(tuple(source.alias for source in sources) == tuple(MIX)
        and [source.identity for source in sources] == manifest['sources'],
        'Rebuild the actual original five sources from their root-frozen provenance.')
    require(tokenizer_manifest(tokenizer) == manifest['tokenizer'], 'Native tokenizer/template changed after data freeze.')
    update = manifest['updates'][category_id]
    require(update['data_seed'] == kd_data_seed(manifest['model_identity'], category_id, manifest['search_seed'])
        and update['workers'] == 16 and update['schedule_reserved'] == 1000,
        'All legal500/1000 trajectories share the same original16-worker reserved prefix.')
    held_out = {row['record']['record_id'] for row in manifest['proposal']['records']}
    dataset = BoundCanonicalIterable(sources, tokenizer, weights=[MIX[source.alias] for source in sources],
        seed=update['data_seed'], excluded_ids=held_out, trace=False)
    return DistillDatasetBundle(train_dataset=dataset, eval_dataset=None,
        dataset_mix_spec=','.join(f'{alias}={weight:.3f}' for alias, weight in MIX.items()),
        source_stats=[dict(alias=source.alias, weight=MIX[source.alias], actual_rows=len(source.raw),
            identity=source.identity, is_iterable=True) for source in sources],
        is_iterable=True, cache_key=('cat_auto', prepared['data_identity'], category_id, update['data_seed']),
        group_by_length=False)


class UpdateObserver:
    """Called before native compute_loss changes inputs; no extra model input keys."""
    def __init__(self, update_manifest):
        self.manifest = update_manifest
        self.observed = []

    def __call__(self, inputs, update_index):
        index = int(update_index)
        require(index == len(self.observed) and index < len(self.manifest['batches']),
            'Actual update order left its root-frozen source prefix.')
        expected = self.manifest['batches'][index]
        actual = batch_digest(inputs)
        require(actual == expected['input_sha256'], 'Native tokens/labels/padding differ from the frozen data plan.')
        self.observed.append(dict(update_index=index, input_sha256=actual,
            record_ids=[row['record_id'] for row in expected['records']]))
        return self.observed[-1]

    def report(self):
        return dict(actual_updates=len(self.observed), actual_record_presentations=4 * len(self.observed),
            token_order_digest=digest(self.observed), group_by_length=False, rows=self.observed)
