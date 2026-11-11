"""Unchanged H96 train inputs, stored once for detached block reconstruction."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import torch

from experiments.mlp_learnability_cmcc_20261030.data import legacy
from experiments.muon_compact_cmcc_20261030 import inputs as original


SHORT_GROUPS = (2, 10)


@dataclass(frozen=True)
class InputEntry:
    entry_id: str
    domain: str
    case_ids: tuple[str, ...]
    inputs: dict[str, torch.Tensor]
    valid_mask: torch.Tensor

    @property
    def valid_tokens(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def padded_tokens(self) -> int:
        return self.valid_mask.numel()


@dataclass(frozen=True)
class EntryGroup:
    group_index: int
    entry_indices: tuple[int, ...]
    valid_tokens: int

    def denominator(self, hidden_size: int) -> int:
        return self.valid_tokens * hidden_size


@dataclass(frozen=True)
class EntryPool:
    entries: tuple[InputEntry, ...]
    groups: tuple[EntryGroup, ...]

    def group(self, index: int) -> EntryGroup:
        for group in self.groups:
            if group.group_index == index:
                return group
        raise ValueError(f'Group {index} is not in this fixed input pool.')

    def select_groups(self, indices) -> EntryPool:
        indices = tuple(indices)
        if not indices or len(indices) != len(set(indices)):
            raise ValueError('Select nonempty, distinct original group indices.')
        selected = tuple(self.group(index) for index in indices)
        used = sorted({entry for group in selected for entry in group.entry_indices})
        remap = {old: new for new, old in enumerate(used)}
        return EntryPool(tuple(self.entries[index] for index in used),
            tuple(EntryGroup(group.group_index, tuple(remap[index] for index in group.entry_indices),
                             group.valid_tokens) for group in selected))


def summary(pool: EntryPool) -> dict:
    """Count the actual CPU entries and original group presentations separately."""
    unique = Counter()
    presentations = Counter()
    unique_valid = Counter()
    for entry in pool.entries:
        unique[entry.domain] += len(entry.case_ids)
        unique_valid[entry.domain] += entry.valid_tokens
    groups = []
    for group in pool.groups:
        entries = [pool.entries[index] for index in group.entry_indices]
        for entry in entries:
            presentations[entry.domain] += len(entry.case_ids)
        groups.append(dict(group_index=group.group_index, entry_indices=list(group.entry_indices),
            entries=len(entries), valid_tokens=group.valid_tokens, M_shape=list(entries[0].valid_mask.shape),
            QA_max_length=max(entry.valid_mask.shape[1] for entry in entries[1:]),
            denominator_hidden4096=group.denominator(4096)))
    padded = sum(entry.padded_tokens for entry in pool.entries)
    return dict(group_indices=[group.group_index for group in pool.groups], groups=groups,
        entries=len(pool.entries), M32_entries=sum(entry.domain == 'mmlu' for entry in pool.entries),
        QA_B1_entries=sum(entry.domain != 'mmlu' for entry in pool.entries),
        unique_cases=dict(unique), presentations=dict(presentations), unique_valid_tokens=dict(unique_valid),
        unique_total_valid_tokens=sum(unique_valid.values()), unique_padded_tokens=padded,
        group_presented_valid_tokens=sum(group.valid_tokens for group in pool.groups),
        two_BF16_hidden_stream_bytes=padded * 4096 * 2 * 2,
        shared_BF16_RoPE_bytes=padded * 128 * 2 * 2)


def _load_original():
    recipe = legacy.aligned.reference.original_record()
    manifest_path = original.HOT / 'dataset_prepared.json'
    control_path = legacy.R / 'qwen_teacher_scope_qa128_20261011/dataset_prepared.json'
    hot_path = legacy.R / 'qwen_hotpot_train_cpu_20261011/prepared.json'
    manifests = [legacy.read(path) for path in (manifest_path, control_path, hot_path)]
    paths = [original.checked(manifest['dataset_binding']) for manifest in manifests]
    dataset, control, hot = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    properties = legacy.aligned.reference.data.validate(dataset, (control, hot, manifests[1], manifests[2]))
    legacy.require(dataset['groups'] == recipe['groups'] == manifests[0]['groups']
        and dataset['train_ids'] == recipe['train_ids'] == manifests[0]['train_ids']
        and dataset['evidence'] == manifests[0]['data_contract']
        and properties == manifests[0]['CPU_validation']['data_properties'],
        'Changed the original H96 train identities, order, inputs or source evidence.')
    bindings = {str(path): legacy.binding(path) for path in (manifest_path, control_path, hot_path, *paths)}
    return dataset, bindings, properties


def load(pad_token_id: int | None = None) -> tuple[EntryPool, dict]:
    """Load original input datasets only; no cached teacher logits or trajectories."""
    dataset, bindings, properties = _load_original()
    if pad_token_id is None:
        value = legacy.read(legacy.HF / 'config.json').get('pad_token_id')
        pad_token_id = 0 if value is None else value
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError('pad_token_id must be the native nonnegative integer ID.')
    cases = {case['case_id']: case for case in dataset['training_cases']}
    entries = []
    for group in dataset['groups']:
        selected = [cases[key] for key in group['case_ids'][:32]]
        inputs, _options = legacy.mmlu.batch_inputs(selected, pad_token_id)
        entries.append(InputEntry(f"M32/group{group['group_index']}", 'mmlu',
            tuple(case['case_id'] for case in selected), inputs, inputs['attention_mask'].bool()))
    qa_indices = {}
    for case in dataset['training_cases'][1024:]:
        ids = case['input_ids'].unsqueeze(0)
        mask = torch.ones_like(ids)
        qa_indices[case['case_id']] = len(entries)
        entries.append(InputEntry(case['case_id'], case['domain'], (case['case_id'],),
            dict(input_ids=ids, attention_mask=mask), mask.bool()))
    groups = []
    for group in dataset['groups']:
        indices = (group['group_index'], *(qa_indices[key] for key in group['case_ids'][32:]))
        groups.append(EntryGroup(group['group_index'], indices, sum(entries[index].valid_tokens for index in indices)))
    pool = EntryPool(tuple(entries), tuple(groups))
    full, short = summary(pool), summary(pool.select_groups(SHORT_GROUPS))
    legacy.require(full['entries'] == 160 and full['M32_entries'] == 32 and full['QA_B1_entries'] == 128
        and short['entries'] == 18 and max(group['M_shape'][1] for group in short['groups']) == 958
        and max(group['QA_max_length'] for group in short['groups']) == 10593,
        'The actual fixed H96 grouping or resource shapes changed; do not truncate or resample.')
    identities = {case['case_id']: dict(domain=case['domain'], input_signature=case['input_signature'],
        official_split=case['source']['official_split']) for case in dataset['training_cases']}
    identity_sha = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    evidence = dict(schema='BR_H96_original_train_entries_v1', status='verified_original_inputs_only',
        bindings=bindings, original_data_properties=properties, input_identity_sha256=identity_sha,
        full=full, short=short, pad_token_id=pad_token_id,
        objective='sum squared block-output error / (sum actual valid tokens in all nine entries * hidden_size)',
        domain_weights='No E-stage .5 domain weights; no mean of nine separately normalized entry MSEs.',
        capture='32 complete native left-padded M32 entries with explicit position_ids;128 original QA B1 TF entries.',
        reuse='Each original QA entry is stored once and referenced by its two original groups.',
        source='Original M1024 auxiliary_train/train, Qasper32 train and Hotpot96 distractor/train; no new selection.',
        teacher_logits_loaded=False, boundary='CPU input identity and resource accounting only; no model execution or quality claim.')
    return pool, evidence
