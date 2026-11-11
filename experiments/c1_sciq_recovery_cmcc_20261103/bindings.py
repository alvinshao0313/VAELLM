"""Bind one data-only variant to the original C1/Rice/Muon implementation."""
from copy import deepcopy
from pathlib import Path

from experiments.rice_muon_cmcc_20261031.bindings import *
from experiments.rice_muon_cmcc_20261031 import bindings as original

HERE = Path(__file__).resolve().parent
OUTPUT = ARTIFACTS / 'c1_sciq_recovery_cmcc_20261103'
EVIDENCE = TOPIC / 'c1_sciq_recovery_cmcc_20261103'
ORIGINAL_PREPARED = original.OUTPUT / 'C1_H96_Muon32/prepared.json'
PREPARED = OUTPUT / 'prepared.json'
DATA_ROOT = Path('/data1/cmcc/VAELLM/data/allenai_sciq_2c94ad3e1aafab77146f384e23536f97a4849815')
REVISION = '2c94ad3e1aafab77146f384e23536f97a4849815'
PARENT_ID = '744e3b98-8a09-4a1d-8b30-013db5d73d44'
CONTROL_ID = '3ebcc40c-e572-41db-96c9-1dc58be564a9'
PHASES = dict(formal32=list(range(32)))
CONTRACT = {**deepcopy(original.CONTRACT),
    'budget': 'Original C1 raw fresh; all252 D/R0; M1024=aux512+SciQ512; Q32 twice/H96 twice; original500 prefix32; seed137/1xLR.',
    'lineage': 'Independent fresh original C1 raw BF16 masters/optimizer/kappa; never continue any recovered endpoint.',
    'M_data': 'Each original M32 keeps its first16 auxiliary cases and replaces last16 by consecutive fixed-hash SciQ train cases; QA8 exact.',
    'M_target': 'Retained aux targets reused exactly; SciQ original HF BF16 FA2/FP32RoPE B1 last-query native4 softmax, existing .5teacher+.5gold helper once.',
    'QA_target': 'Unchanged current-student greedy OPD and original full-HF full-vocabulary KL; actual generated supervision measured.',
    'control': 'Completed original C1 Muon32, nine64.89598790075989/M61.57954707306651/LB43.91154134; no retraining.'}


def configure():
    original.configure()
    mature.STEPS = deepcopy(PHASES)


def code_bindings():
    paths = set(HERE.glob('*.py'))
    paths.update((REPO / 'e2e_common/data.py',
        TOPIC / 'qwen_mmlu_causal_readout/prepare_data.py',
        TOPIC / 'qwen_causal_capability_probe/data_adapter.py',
        TOPIC / 'qwen_mmlu_target_mix32/targets.py'))
    return {**original.code_bindings(), **{str(path.resolve()): binding(path) for path in paths}}


def verified_Rice_short(_prepared_path):
    return original.verified_Rice_short(ORIGINAL_PREPARED)


def read_prepared(path):
    path = Path(path).resolve()
    value = read(path)
    require(path == PREPARED and value['schema'] == 'C1_SciQ_data_Muon_prepared_v1'
        and value['status'] == 'CPU_source_data_and_serial_HF_cache_verified'
        and value['contract'] == CONTRACT and value['group_indices'] == GROUPS,
        'Use only the reviewed fixed SciQ512/aux512 preparation.')
    for item in value['bindings'].values():
        require(binding(item['path']) == item, 'Prepared source/input changed: ' + item['path'])
    old = original.read_prepared(ORIGINAL_PREPARED)
    expected = deepcopy(old['record'])
    dataset = inputs.read_dataset()
    expected.update(groups=dataset['groups'], train_ids=dataset['train_ids'])
    require(value['original_prepared'] == binding(ORIGINAL_PREPARED)
        and value['source_binding'] == old['source_binding']
        and value['source_binding']['checkpoint_id'] == PARENT_ID
        and value['source_budget'] == old['source_budget'] and value['record'] == expected
        and value['Rice_lifecycle_evidence'] == verified_Rice_short(path),
        'Same actual raw C1, all-D geometry and completed original Rice lifecycle are required.')
    manifest = read(OUTPUT / 'teacher/cache.json')
    require(value['teacher_cache'] == binding(OUTPUT / 'teacher/cache.json')
        and manifest['status'] == 'verified_serial_HF_SciQ512_once_mixed'
        and manifest['teacher_fully_released'], 'Require completed serial HF cache and teacher release.')
    return value


from . import data as inputs
