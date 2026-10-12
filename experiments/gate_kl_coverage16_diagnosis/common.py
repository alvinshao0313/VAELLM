"""Fixed no-training diagnosis of existing f927 sixteen-group endpoints."""
import os
from pathlib import Path
import sys

from experiments.gate_kl_coverage16_probe.common import (
    ROOT, BASE, SOURCE, SOURCE_ID, TARGET, BANKS, GROUPS, VECTORS, read, check, replay_codes,
)
from experiments.longbench_v1.common import sha256
from experiments.rotation_factor_probe.capture import require

ARM = 'alpaca64_fineweb64'
POPULATIONS = (ARM, 'train32', 'alpaca_held32', 'fineweb_rows500000_500031',
               'alpaca_confirmation32', 'fineweb_rows500032_500063')


def bound(record):
    root = Path(record['output_dir'])
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == ROOT == Path(record['cwd'])
            and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and not record['queued'] and not record['automatic_retry'], 'Require one root-bound diagnosis.')
    prepared = read(root / 'launch.prepared.json')
    require(record['preparation_sha256'] == sha256(root / 'launch.prepared.json')
            and all(record.get(k) == v for k, v in prepared.items() if k not in ('status', 'gpu')),
            'Bound scientific configuration differs from CPU preparation.')
    require(all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
            'Prepared numerical/offline environment changed.')
    check(record)
