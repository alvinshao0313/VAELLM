"""The one fixed C0 all-MLP learning comparison; numerical helpers remain shared."""
import os
from pathlib import Path
import sys
import subprocess
import time

from experiments.mlp_learnability_cmcc_20261030 import bindings as core
from experiments.mlp_learnability_cmcc_20261030 import data, export, lifecycle as life, free_dense
from experiments.mlp_learnability_cmcc_20261030 import run_short
from experiments.cmcc_runtime_20261030 import sequence as executor
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, TOPIC, PYTHON, REPO

torch, checkpoint, HF = core.torch, core.checkpoint, core.HF
require, read, write, sha, digest, binding = core.require, core.read, core.write, core.sha, core.digest, core.file_binding
HERE = Path(__file__).resolve().parent
ROOT = ARTIFACTS/'mlp_learnability_formal_cmcc_20261101'
EVIDENCE = TOPIC/'mlp_learnability_formal_cmcc_20261101'
SUPPORT = ARTIFACTS/'adaptive_class_cmcc_20261030/evaluation_support'
C0 = ARTIFACTS/'fresh_fixed2_cmcc_20261030/single_20261030_v3_C0/final_model'
C0_ID = 'cbb15c9f-1c1f-4394-817c-65c226d76ce7'
BASE_LR = dict(decoder=1e-6, dense=1e-5)
CONTRACT = dict(seed=137, optimizer='AdamW', betas=[.9,.95], eps=1e-8, weight_decay=0., foreach=False,
    clip=1.5, original_schedule_steps=500, base_lr=BASE_LR, warmup_steps=0,
    target='Original static H96 TF: once-mixed M and unchanged full-F gold-prefix QA; no OPD',
    groups_per_cycle=list(range(32)), cycles=4, steps=128, midpoint=64,
    M_presentations=4096, QA_presentations=1024, input_tokens=4212624, answer_positions=10944,
    trainable='All108 MLP D or same-parent BF16 nonprotected dense columns; other fields frozen',
    snapshots='Independent BF16 copies; preserve live master/Adam/LR and Python/NumPy/CPU/CUDA RNG')


def gpus(value):
    values = value.split(',')
    require(len(values) == 2 and len(set(values)) == 2 and all(item in tuple(map(str,range(8))) for item in values),
        'Choose exactly two distinct physical GPU indices.')
    return values


def resources(devices):
    require(Path(sys.executable).resolve() == PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(devices), 'Use caller bitvae and the assigned physical pair.')
    found = subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True)
    inventory = {line.split(',')[0].strip():line.split(',')[1].strip() for line in found.stdout.splitlines()}
    require(set(devices) <= set(inventory), 'An assigned physical GPU is absent.')
    selected = {gpu:inventory[gpu] for gpu in devices}
    active = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True)
    occupants = [line for line in active.stdout.splitlines()
        if line.split(',')[0].strip() in selected.values() and int(line.split(',')[1].strip()) != os.getpid()]
    require(not occupants, 'Assigned GPUs have other compute processes; stop without retry: '+repr(occupants))
    checked = dict(checked_unix=time.time(),gpu_uuids=selected,other_compute_processes=occupants,
        only_self_PID_excluded=os.getpid())
    require(torch.cuda.device_count() == 2, 'The existing18/18 layer MP requires two visible devices.')
    return checked


def source_C0(directory=C0):
    parent = core.source(directory, prototype_only=False)
    require(parent['checkpoint_id'] == C0_ID and parent['path'] == str(C0.resolve(strict=True))
        and parent['source_format'] == 'mixed' and not parent['prototype_only'], 'Use only the actual completed common C0.')
    return parent


def code_bindings():
    return {**run_short.code_bindings(), **{str(path.resolve()):sha(path) for path in HERE.glob('*.py')}}


def groups(short):
    return [2,10] if short else list(range(32))*4


def snapshot_steps(short):
    return [1,2] if short else [64,128]
