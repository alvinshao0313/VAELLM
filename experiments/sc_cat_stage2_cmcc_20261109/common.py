"""Frozen identities and finite-batch settings for the independent second stage."""
import json
from pathlib import Path
import random
import numpy as np
import torch
from .prepare import ROOT, read, require, sha, binding


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed():
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)


def bound(reference):
    require(sha(reference['path']) == reference['sha256'], 'Changed binding: ' + reference['path'])
    return Path(reference['path'])


def load_plan(path):
    plan = read(path)
    for name, digest in plan['source_sha256'].items():
        require(sha(name) == digest, 'Changed running source: ' + name)
    return plan


def code_pins():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT / 'experiments/sc_cat_recovery_cmcc_20261108' / name
              for name in ('mutable.py', 'protection.py', 'optimizer.py', 'settings.py')]
    files += [ROOT / 'train_utils' / name for name in
              ('checkpoint_v6.py', 'distill_loss_core.py', 'distill_data.py', 'distill_precision.py',
               'distill_decoder.py', 'model_level_trainables.py', 'model_level_checkpoint_state.py')]
    return {str(p): sha(p) for p in files}
