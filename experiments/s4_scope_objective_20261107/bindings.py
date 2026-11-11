"""One immutable down/pre parent and four explicitly authorized recipes."""
import os
import subprocess
import time
from pathlib import Path
from experiments.s4_recovery_screen_20261105 import bindings as prior

torch,require,read,write,sha=prior.torch,prior.require,prior.read,prior.write,prior.sha
ROOT,HF,PYTHON,CATEGORIES=prior.ROOT,prior.HF,prior.PYTHON,prior.CATEGORIES
SOURCE,SOURCE_ID=prior.SOURCE,prior.SOURCE_ID
old,parent=prior.old,prior.parent
binding,event,source=prior.binding,prior.event,prior.source
initialize,environment,memory=prior.initialize,prior.environment,prior.memory
restore_rope_alias=prior.restore_rope_alias
PARENT_EIGHT=prior.PARENT_EIGHT
HERE=Path(__file__).resolve().parent
OUTPUT=prior.OUTPUT.parent/'scope_objective_20261107'
RUN=OUTPUT/'trial_20261108_four_arms128_v3'
ALGORITHM='S4_same_parent_scope_objective_128'
BPW=4.094031292459239
ARMS={
    'A':dict(gpu='4',protection=None,objective='forward',leaves=1512),
    'B':dict(gpu='5',protection=None,objective='adaptive',leaves=1512),
    'C':dict(gpu='6',protection='scale',objective='forward',leaves=1764),
    'D':dict(gpu='7',protection='integer',objective='forward',leaves=1764)}
CONTRACT=dict(algorithm=ALGORITHM,parent_id=SOURCE_ID,updates=128,groups=list(range(128)),
    endpoints=[64,128],seed=137,optimizer='fresh AdamW',betas=[.9,.95],eps=1e-8,
    weight_decay=0.,foreach=False,clip=1.5,schedule_steps=500,
    decoder_lr=5e-7,log_scale_lr=1e-3,integer_lr=.03,
    M_batch=32,QA_documents=8,M_weight=.5,QA_weight=.5,teacher_gold_mixture=[.5,.5],
    source_weights=dict(webinstruct=.25,medmcqa=.25,QA=.5),
    adaptive='batch mean raw teacher entropy/log4; bidirectional KL against once-mixed target; QA unchanged',
    protection='native INT8 q/BF16 scales; fixed indices; separate q or log-scale with all252 decoders',
    precision='FP32 masters/gradients/Adam; BF16 compute and native deploy',
    selection='Complete128 and both64/128 eight-task endpoints; no quality early stop, LongBench or successor',
    early_groups=prior.CONTRACT['early_groups'],late_groups=prior.CONTRACT['late_groups'],
    expected_bpw=BPW,scope='Local down/pre recovery; not a BF16 full-chain result')
REFERENCE_VALIDATION=prior.RUN/'validation/result.json'


def pins():
    return {**prior.pins(),**{str(p):sha(p) for p in sorted([*HERE.glob('*.py'),*HERE.glob('*.sh')])}}


def check_request(request):
    require(request['arm'] in ARMS and request['recipe']==ARMS[request['arm']]
        and request['contract']==CONTRACT and request['parent']==source(),'Changed recipe or parent.')
    require(request['groups']==([2,9] if request['validation_only'] and request['arm'] in ('C','D')
        else [9] if request['validation_only'] else list(range(128))), 'Changed group order.')
    parent.s.check_pins(request['source_sha256'])


def resources(gpu):
    initialize()
    require(gpu in ('4','5','6','7') and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
        and torch.cuda.device_count()==1,'Use exactly the assigned single GPU4..7 in bitvae.')
    uuids=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    selected={a.strip():v.strip() for a,v in (row.split(',') for row in uuids.splitlines())}[gpu]
    rows=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    others=[row for row in rows.splitlines() if row.split(',')[0].strip()==selected
        and int(row.split(',')[1])!=os.getpid()]
    require(not others,'Assigned GPU occupied; no retry: '+repr(others))
    return dict(physical_gpu=gpu,uuid=selected,other_processes=others,checked_unix=time.time())
