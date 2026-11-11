"""One same-C0 M coverage contrast; established numerical helpers stay read-only."""
from pathlib import Path

from experiments.mlp_learnability_formal_cmcc_20261101.bindings import *
from experiments.mlp_learnability_formal_cmcc_20261101 import bindings as control

HERE = Path(__file__).resolve().parent
ROOT = ARTIFACTS/'mlp_m4096_cmcc_20261101'
EVIDENCE = TOPIC/'mlp_m4096_cmcc_20261101'
DATA = ROOT/'data'
CONTRACT = {**control.CONTRACT,
    'groups_per_cycle':list(range(128)), 'cycles':1,
    'M_unique':4096, 'QA_unique':128,
    'M_coverage':'Only same SHA256 causal_output_fit/137/ auxiliary_train prefix1024 repeated4 -> prefix4096 once',
    'QA_order':'Original32 complete QA8 group suffixes repeated4, exact inputs/targets/order/weights',
    'trainable':'All108 fixed-code MLP decoders; attention/protection/codes/non-MLP frozen',
    'input_tokens':'Actual complete128 exposure measured in data_contract.json; different from control',
    'answer_positions':10944}
from . import data


def groups(short=False):
    require(short is False, 'This coverage comparison has no new GPU short path.')
    return list(range(128))


def snapshot_steps(short=False):
    require(short is False, 'This coverage comparison has no new GPU short path.')
    return [64,128]


def data_binding():
    return binding(DATA/'data_contract.json')


def resources(devices):
    require(devices==['2','3'],'This coverage training/fresh uses only assigned physical GPUs2/3.')
    return control.resources(devices)


def code_bindings():
    paths=list(HERE.glob('*.py'))
    if (DATA/'data_contract.json').exists():
        paths.extend(DATA/name for name in ('data_contract.json','dataset_prepared.json','dataset.pt','teacher.json','new_M3072.pt'))
    return {**control.code_bindings(), **{str(path.resolve()):sha(path) for path in paths}}
