"""Read-only S4 parent, immutable H96 targets and the fixed two-card contract."""
from pathlib import Path
import os
import subprocess
import sys
import time

from experiments.progressive_residual_bsq_20261030 import bindings as parent
from experiments.mlp_learnability_cmcc_20261030 import data, lifecycle as old_life, run_short

import torch
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as native
from train_utils.distill_precision import install_precision_runtime

HERE = Path(__file__).resolve().parent
REPO, HF, PYTHON = parent.ROOT, parent.HF, parent.PYTHON
BASE = parent.OUTPUT_ROOT / 'independent_stage1_20261031_2157'
SOURCE = BASE / 'formal/native/data1_cmcc_models_Qwen_Qwen3-8B_20261031_220056/after_down_proj'
SOURCE_ID = 'b12014e7-f931-4502-86e3-bca93085eaf3'
OUTPUT = parent.OUTPUT_ROOT / 'recovery_compare_20261101'
CATEGORIES = tuple(parent.CATEGORIES)
require, read, write, sha = parent.require, parent.s.read, parent.s.write, parent.s.sha
digest = data.runtime.digest
CONTRACT = dict(steps=56, steps_per_segment=8, groups=list(range(32))+list(range(24)),
    seed=137, lr=1e-6, schedule_steps=500, warmup=0, optimizer='AdamW', betas=[.9,.95],
    eps=1e-8, weight_decay=0., foreach=False, joint_clip=1.5,
    optimizer_across_segments='continuous; newly active leaves start at zero, existing masters/moments retained',
    target='Original H96 static TF: once-mixed M(.5F+.5gold) plus gold-prefix full-vocabulary QA',
    trainable='Only prefix native S4 decoders; codes/protection/rotation/nonprojection/future dense frozen',
    terminal='Full S4 throughout; prefix decoder training set grows once per segment',
    sequential='Install next original S4 bank category into the original CAT BF16 factory; keep earlier recovery',
    endpoint_readout='Fixed small train probe, full MMLU B64 and seven B32; no LongBench in this screening pair')


def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def source():
    result = read(BASE / 'formal/result.json')
    request = read(BASE / 'formal/request.json')
    require(read(BASE / 'run.json')['status'] == 'completed_formal_S4_first_stage'
        and result['checkpoint_id'] == SOURCE_ID and result['checkpoint']['path'] == str(SOURCE)
        and read(BASE / 'formal/fit.json')['recoveries'] == 0, 'Use the completed original no-recovery S4 parent.')
    parent.s.check_pins(request['source_sha256'])
    event = parent.boundary(SOURCE)
    require(event == {key:result['checkpoint'][key] for key in event}, 'Original S4 parent identity changed.')
    return dict(**event, formal_request=binding(BASE / 'formal/request.json'),
        original_result=binding(BASE / 'formal/result.json'), original_source_sha256=request['source_sha256'])


def pins():
    return {**parent.source_pins(), **run_short.code_bindings(),
        **{str(p):sha(p) for p in sorted(HERE.glob('*.py'))}}


def resources():
    require(Path(sys.executable).resolve() == PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '6,7', 'Use caller bitvae and exactly GPU6/7.')
    inventory = subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    selected = {line.split(',')[0].strip():line.split(',')[1].strip()
        for line in inventory.splitlines() if line.split(',')[0].strip() in ('6','7')}
    apps = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    occupants = [line for line in apps.splitlines()
        if line.split(',')[0].strip() in selected.values() and int(line.split(',')[1]) != os.getpid()]
    require(len(selected) == 2 and not occupants, 'GPU6/7 occupied; stop without waiting or retry: '+repr(occupants))
    require(torch.cuda.device_count() == 2, 'Two visible devices required for the real18/18 map.')
    return dict(checked_unix=time.time(), selected=selected, other_compute_processes=occupants)


def decoder_name(name):
    return '._parallel_stage_decoder.' in name


def category(name):
    return name.split('._parallel_stage_decoder.',1)[0].rsplit('.',1)[-1]


def segments(short):
    # Short keeps real peak groups and the actual stage1->2 activation, then full S4.
    return [(1,[2]), (2,[10]), (7,[2,10])] if short else [
        (index+1,CONTRACT['groups'][index*8:(index+1)*8]) for index in range(7)]
