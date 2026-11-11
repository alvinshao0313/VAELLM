"""Isolated C1 target study; reuse the completed Rice/Muon lifecycle unchanged."""
from copy import deepcopy
from pathlib import Path

from experiments.rice_muon_cmcc_20261031 import bindings as original
from experiments.muon_compact_cmcc_20261030 import rollout, motion

torch, core, checkpoint = original.torch, original.core, original.checkpoint
read, require, binding = original.read, original.require, original.binding
inputs, lifecycle, optimizer, mature = original.inputs, original.lifecycle, original.optimizer, original.mature
HF, PYTHON, REPO, SUPPORT = original.HF, original.PYTHON, original.REPO, original.SUPPORT
CompactRVQLinear = mature.CompactRVQLinear
HERE = Path(__file__).resolve().parent
OUTPUT = original.ARTIFACTS / 'c1_m_target_muon_cmcc_20261102'
REVISION = 'revision_group_fix_20261102'
RUN_OUTPUT = OUTPUT / REVISION
EVIDENCE = original.TOPIC / 'c1_m_target_muon_cmcc_20261102' / REVISION
ORIGINAL_PREPARED = original.OUTPUT / 'C1_H96_Muon32/prepared.json'
CPU_SOURCE_PREPARED = OUTPUT / 'prepared.json'
PREPARED = RUN_OUTPUT / 'prepared.json'
GROUPS = list(range(32))
TARGET_ARMS = ('full_teacher', 'pure_gold')
ARMS = (*TARGET_ARMS, 'conditional_lr_half', 'conditional_lr_double')
LR_MULTIPLIERS = dict(full_teacher=1., pure_gold=1., conditional_lr_half=.5, conditional_lr_double=2.)
STEPS = dict(formal32=GROUPS)
CONTRACT = {**deepcopy(original.CONTRACT),
    'budget': 'Original C1 raw; all252 D; M1024 once/Q32 twice/H96 twice; original500 prefix32; seed137/R0.',
    'lineage': 'Each arm independently loads the original C1 raw BF16 checkpoint; fresh masters/optimizer/kappa; no continuation.',
    'M_full_teacher': 'M domain .5 times mean[.5 KL(full-vocab F||S)+.5 CE(gold|native4)].',
    'M_pure_gold': 'M domain .5 times mean[CE(gold|native4)]; QA domain and online full-HF targets remain unchanged.',
    'M_LR_arms': 'Original cached conditional4 mixed target and weighted_loss unchanged; all Muon/aux Adam scheduled LRs multiplied by .5 or2 after original schedule, kappa calibration unchanged.',
    'control': 'Completed C1 Muon32 conditional4 teacher/gold mixture; reuse its complete-nine result, no retraining.'}


def configure():
    original.configure()
    mature.STEPS = deepcopy(STEPS)


def code_bindings():
    paths = set(HERE.glob('*.py'))
    return {**original.code_bindings(), **{str(path.resolve()): binding(path) for path in paths}}


def read_prepared(path):
    path = Path(path).resolve()
    value = read(path)
    require(value['schema'] == 'C1_M_target_Muon_prepared_v1'
        and value['status'] == 'CPU_prepared_reused_Rice_lifecycle'
        and value['contract'] == CONTRACT and value['group_indices'] == GROUPS,
        'Use this fixed two-arm C1 preparation.')
    for item in value['bindings'].values():
        require(binding(item['path']) == item, 'Prepared input/source changed: ' + item['path'])
    old = read(CPU_SOURCE_PREPARED)
    require(old['schema'] == 'C1_M_target_Muon_prepared_v1'
        and old['status'] == 'CPU_prepared_reused_Rice_lifecycle', 'The original completed CPU source proof is required.')
    for name, item in old['bindings'].items():
        if not Path(name).is_relative_to(HERE):
            require(binding(item['path']) == item, 'Original shared source/input changed: ' + item['path'])
    parent, _ = original.source(old['source_binding']['checkpoint_dir'], old['source_binding']['pipeline_report'])
    require(value['original_prepared'] == binding(ORIGINAL_PREPARED)
        and value['source_binding'] == old['source_binding']
        and value['source_binding'] == parent and value['CPU_source_preparation'] == binding(CPU_SOURCE_PREPARED)
        and value['source_budget'] == old['source_budget'] and value['record'] == old['record']
        and value['Rice_lifecycle_evidence'] == original.verified_Rice_short(ORIGINAL_PREPARED),
        'Reuse only the same actual raw C1 and its completed Rice lifecycle proof.')
    return value


def load_full_targets(prepared):
    manifest = read(OUTPUT / 'teacher/cache.json')
    require(manifest['status'] == 'verified_full_HF_M1024_last_query'
        and manifest['source_checkpoint_id'] == prepared['record']['source_checkpoint_id']
        and manifest['original_prepared'] == binding(ORIGINAL_PREPARED)
        and manifest['teacher_fully_released'], 'Require the complete independent full-HF M cache.')
    require(binding(manifest['payload']['path']) == manifest['payload'], 'Full-M teacher payload changed.')
    payload = torch.load(manifest['payload']['path'], map_location='cpu', weights_only=True, mmap=True)
    require(payload['teacher_identity'] == prepared['record']['teacher_identity']
        and len(payload['targets']) == 1024, 'Full-M teacher identity or coverage changed.')
    return payload['targets'], manifest
