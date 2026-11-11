"""Immutable upstream, truthful local provenance, and single-card resources."""
from pathlib import Path
import os
import subprocess
import sys
import time

from experiments.s4_skip_up_20261103 import bindings as prior

base,old,parent,data=prior.base,prior.old,prior.parent,prior.data
torch,require,read,write,sha=prior.torch,prior.require,prior.read,prior.write,prior.sha
ROOT,HF,PYTHON,CATEGORIES=prior.ROOT,prior.HF,prior.PYTHON,prior.CATEGORIES
HERE=Path(__file__).resolve().parent
UPSTREAM=prior.OUTPUT/'trial_20261103_1730'
SOURCE=UPSTREAM/'formal/nodes/down_proj/pre/model'
SOURCE_ID='fa50bf40-c682-4910-8e72-d024f6b678e4'
CONTROL=UPSTREAM/'formal/nodes/down_proj/post/model'
CONTROL_ID='a6a04164-2782-4b19-965f-410198c3ddde'
OUTPUT=prior.OUTPUT/'diagnosis_20261104'
ALGORITHM='same_parent_S4_down_recovery_diagnosis'
ARMS={'m_only':dict(gpu='6',base_lr=1e-6,include_QA=False),
      'joint_half_lr':dict(gpu='7',base_lr=5e-7,include_QA=True)}
CONTRACT=dict(algorithm=ALGORITHM,scope='Fixed-upstream local recovery comparison; not a new BF16 full compression chain',
    parent_id=SOURCE_ID,control_id=CONTROL_ID,updates=128,schedule_steps=500,seed=137,
    optimizer='fresh AdamW',betas=[.9,.95],eps=1e-8,weight_decay=0.,foreach=False,clip=1.5,
    M_weight=.5,QA_weight=.5,M_batch=32,QA_documents=8,groups=list(range(128)),
    trainable='All252 installed S4 projection decoders; all other persistent fields frozen',
    masters='FP32 parameters/gradients/Adam moments; original BF16 compute and deployment',
    observations=[0,64,128],early_groups=[2,10,34,42],late_groups=[66,74,98,106],
    heldout='128 official same-source auxiliary_train cases by existing stable ID hash excluding training4096; never train',
    selection='Fixed128 endpoint; no quality early stop, rollback or adaptive extension',
    placement='One complete unchanged model per assigned GPU; no batch/length/parameter reduction')
binding,event,initialize,environment=prior.binding,prior.event,prior.initialize,prior.environment
restore_rope_alias=prior.restore_rope_alias


def pins():
    upstream=read(UPSTREAM/'plan.json')['source_sha256'];parent.s.check_pins(upstream)
    return {**upstream,**{str(p):sha(p) for p in sorted(HERE.glob('*.py'))}}


def resources(gpu):
    initialize()
    require(gpu in ('6','7') and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
        and torch.cuda.device_count()==1,'Exactly the assigned single GPU6 or7 in bitvae is required.')
    uuids=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    selected={a.strip():v.strip() for a,v in (row.split(',') for row in uuids.splitlines())}[gpu]
    rows=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    others=[row for row in rows.splitlines() if row.split(',')[0].strip()==selected
        and int(row.split(',')[1])!=os.getpid()]
    require(not others,'Assigned GPU has an existing process; stop without retry: '+repr(others))
    return dict(physical_gpu=gpu,uuid=selected,other_processes=others,checked_unix=time.time())


def memory():
    return dict(allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0),
        peak_allocated=torch.cuda.max_memory_allocated(0),peak_reserved=torch.cuda.max_memory_reserved(0))


def source():
    value=event(SOURCE);require(value['checkpoint_id']==SOURCE_ID,'Exact same-parent down/pre UUID required.')
    require(read(UPSTREAM/'run.json')['exit_code']==0,'The upstream candidate must be complete.')
    return value
