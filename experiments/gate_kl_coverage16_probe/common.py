"""Fixed f927/mixed128 sixteen-group coverage contrast; no general search API."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.gate_group_accept_probe.core import BANKS, VECTORS_PER_GROUP, apply_choices
from experiments.gate_student_context_probe.data import ROOT, SOURCE, SOURCE_ID, TARGET
from experiments.longbench_v1.common import check_hashes, check_stat
from experiments.rotation_factor_probe.capture import require

BASE = ROOT / 'result/qwen3_8b_mmlu65_20261004'
ARM = 'alpaca64_fineweb64'
GROUPS = 16
CHUNK = 4 * VECTORS_PER_GROUP
VECTORS = GROUPS * VECTORS_PER_GROUP


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    check_hashes(record['source_sha256'])
    check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)


def choice_inventory(choices):
    require(choices.dtype == torch.uint8 and tuple(choices.shape) == (VECTORS,)
            and int(choices.max()) <= 64, 'Require exactly the first16 groups of native D32/S2 choices.')
    rows = []
    for group in range(GROUPS):
        values = choices[group * VECTORS_PER_GROUP:(group + 1) * VECTORS_PER_GROUP]
        rows.append({'group': group, 'vector_start': group * VECTORS_PER_GROUP,
            'selected_vectors': int((values > 0).sum()),
            'stage_xor': [int(((values > stage * 32) & (values <= (stage + 1) * 32)).sum())
                          for stage in range(2)]})
    return rows


def replay_codes(original, choices):
    inventory = choice_inventory(choices)
    require(len(original) == 13 and all(tuple(original[key].shape) == (1548288, 1, 4)
            and original[key].dtype == torch.uint8 for key in BANKS), 'Original f927 gate topology differs.')
    result = dict(original)
    for stage, key in enumerate(BANKS):
        result[key] = original[key].clone()
        apply_choices(result[key][:VECTORS], choices.to(result[key].device), stage)
    return result, [sum(row['stage_xor'][stage] for row in inventory) for stage in range(2)]
