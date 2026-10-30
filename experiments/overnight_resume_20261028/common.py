"""Fixed identities and immutable prefix checks for seven reboot-interrupted runs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / 'result/qwen3_8b_mmlu65_20261004'
FOLLOWUP = TOPIC / 'qwen_recompression_followup_20261026'
HERE = Path(__file__).resolve().parent
OUTPUT = TOPIC / 'qwen_overnight_resume_20261028'
SNAPSHOT = TOPIC / 'rotation_factor_probe/source_snapshot_20261005_01'
# Set the model package context before importing the root file's native loader.
sys.path.insert(0, str(SNAPSHOT))
spec = importlib.util.spec_from_file_location('overnight_resume_native_longbench', ROOT / 'experiments/cat_recompression/longbench.py')
longbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(longbench)
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.protocol import DATASETS
from experiments.longbench_v1.scoring import check_prediction

sys.path.insert(0, str(ROOT / 'experiments/night_sequence_20261026'))
import recovery_eval
from recovery_core import check_gpu, read, require, stat

JOBS = {
    'C06': dict(gpu='2', checkpoint_id='a8bb3da6-0f63-44d0-8f38-40768fe289dc', rows=1690,
        output=TOPIC / 'qwen_night_endpoint_evaluations_20261027/C06/full9'),
    'G06': dict(gpu='6', checkpoint_id='956eb2cf-f96b-4e75-8e87-d587221617c2', rows=1710,
        output=TOPIC / 'qwen_night_endpoint_evaluations_20261027/G06/full9'),
    'C09': dict(gpu='1', checkpoint_id='726b7b36-bb95-4418-b626-cc029b4f0aec', rows=1698,
        output=FOLLOWUP / 'qwen_recompression_recovery_tail_20261027/C/09_joint1200/evaluation_full9'),
    'G09': dict(gpu='3', checkpoint_id='e8e5efd4-655e-462c-b6e0-fedf4e2e0346', rows=1743,
        output=FOLLOWUP / 'qwen_recompression_recovery_tail_20261027/G/09_joint1200/evaluation_full9'),
    'Gce10': dict(gpu='0', checkpoint_id='bde0db08-b61b-436c-999d-83e67d96abe1', rows=1768,
        output=FOLLOWUP / 'repaired_recovery/G/ce10/evaluation_full8'),
    'Gce50': dict(gpu='7', checkpoint_id='3b82e64b-28b7-45a3-942e-296134daabb6', rows=1813,
        output=FOLLOWUP / 'recovery/G/ce50/evaluation_full8'),
    'D05': dict(gpu='5', checkpoint_id='398dd76d-df23-40c3-9a99-a76ace5c23e7', rows=863,
        output=FOLLOWUP / 'qwen_recompression_d_recovery_20261027/05_ce50_attention_ce10_mlp/evaluation_full9'),
}


def check_imports():
    import experiments
    import rotation.hadamard_utils
    import train_utils.v6_model_loader
    modules = (experiments, rotation.hadamard_utils, train_utils.v6_model_loader)
    paths = {module.__name__: str(Path(module.__file__).resolve()) for module in modules}
    require(all(Path(path).is_relative_to(SNAPSHOT) for path in paths.values()),
            'The real model import context must use the verified extended-Hadamard snapshot.')
    return paths


def prediction_prefixes(output, config, *, validate_rows=True):
    _, rows = load_archive(config['archive']['path'])
    by_dataset = {name: [row for row in rows if row['dataset'] == name] for name in DATASETS}
    paths = sorted((output / 'longbench/predictions').glob('*.jsonl'))
    require({path.stem for path in paths} == set(DATASETS), 'Require all21 existing prediction prefix files.')
    config_sha = sha256(output / 'longbench_config.json')
    prefixes = {}
    for path in paths:
        payload = path.read_bytes()
        require(not payload or payload.endswith(b'\n'), 'Incomplete JSONL tail: ' + str(path))
        predictions = [json.loads(line) for line in payload.splitlines()]
        reference = by_dataset[path.stem]
        require(len(predictions) <= len(reference), 'Too many existing predictions: ' + str(path))
        if validate_rows:
            for pred, row in zip(predictions, reference):
                check_prediction(pred, row, config_sha)
        prefixes[str(path)] = dict(bytes=len(payload), rows=len(predictions), expected=len(reference),
            sha256=hashlib.sha256(payload).hexdigest())
    return prefixes


def check_prefix_bytes(prefixes, *, exact_length=False):
    for filename, prefix in prefixes.items():
        path = Path(filename)
        with path.open('rb') as stream:
            payload = stream.read(prefix['bytes'])
            require(len(payload) == prefix['bytes'] and hashlib.sha256(payload).hexdigest() == prefix['sha256'],
                    'An original prediction prefix byte changed: ' + filename)
            if exact_length:
                require(not stream.read(1), 'An existing prediction file changed before resume: ' + filename)
