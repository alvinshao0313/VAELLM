"""Exact QK/VO subsets of one frozen WebMed prefix and common S4 parent."""
from pathlib import Path
import os
import subprocess
import time
from experiments.s4_webmed_recovery_20261105 import bindings as prior
from experiments.s4_recovery_screen_20261105 import bindings as control

torch,require,read,write,sha=prior.torch,prior.require,prior.read,prior.write,prior.sha
ROOT,HF,PYTHON,CATEGORIES=prior.ROOT,prior.HF,prior.PYTHON,prior.CATEGORIES
SOURCE,SOURCE_ID=prior.SOURCE,prior.SOURCE_ID
old,parent=prior.old,prior.parent
binding,event,source=prior.binding,prior.event,prior.source
initialize,environment,memory=prior.initialize,prior.environment,prior.memory
restore_rope_alias=prior.restore_rope_alias
HERE=Path(__file__).resolve().parent
OUTPUT=prior.OUTPUT.parent/'scope_refine_20261106'
RUN=OUTPUT/'trial_20261106_qk_vo64'
ALGORITHM='same_parent_S4_WebMed_fixed64_QK_VO_decoder_scope'
ARMS={'qk':dict(gpu='4',categories=['q_proj','k_proj'],projections=72,leaves=432),
      'vo':dict(gpu='5',categories=['v_proj','o_proj'],projections=72,leaves=432)}
REFERENCE_VALIDATION=control.RUN/'validation/result.json'
CONTRACT=dict(prior.CONTRACT,algorithm=ALGORITHM,updates=64,groups=list(range(64)),observations=[0,64],
    trainable='exact QK72/432 or VO72/432; frozen180 projections preserve input backward; all252 nativeS4',
    selection='One fixed64 endpoint; eight tasks only; no32 readout, LB, early stop, or automatic128',
    resume='Named FP32 masters/Adam/RNG/cursor retained at64; no continuation beyond64 in this batch',
    M_sources='Original pinned WebMed groups0..63:1024/source once; originalQA128 each4times',
    diagnostic_semantics='early trained by64; late groups66/74/98/106 untrained by64; heldout never train')
LEARNING_CONTRACT=dict(seed=137,optimizer='AdamW',base_lr=5e-7,betas=[.9,.95],eps=1e-8,
    weight_decay=0.,foreach=False,clip=1.5,schedule_steps=500,warmup=0,
    M_batch=32,QA_documents=8,M_weight=.5,QA_weight=.5,teacher_gold_mixture=[.5,.5],
    target='Original WebMed conditional4 once-mixed, original gold-prefix full-vocabulary static QA KL',
    precision='FP32 masters/grad/Adam; native BF16 compute/deploy',all252_nativeS4=True)
PARENT_EIGHT={'mmlu':SOURCE.parent/'evaluation/mmlu.json',
    'seven':prior.prior.OUTPUT/'trial_20261104_1940/m_only/zero_parent/evaluation/seven.json'}


def pins():
    return {**control.pins(),**{str(p):sha(p) for p in sorted([*HERE.glob('*.py'),*HERE.glob('*.sh')])}}


def dependency(arm):
    require(arm in ARMS,'Unknown predeclared exact scope.')
    return dict(policy='user_started_QK_VO_fresh64_20261106',
        parent=source(),reference_resume_validation=binding(REFERENCE_VALIDATION))


def resources(gpu):
    initialize()
    require(gpu in ('4','5','6') and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
        and torch.cuda.device_count()==1,'Exactly assigned GPU4/5 or validation GPU6 in bitvae required.')
    uuids=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    selected={a.strip():v.strip() for a,v in (row.split(',') for row in uuids.splitlines())}[gpu]
    rows=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    others=[row for row in rows.splitlines() if row.split(',')[0].strip()==selected
        and int(row.split(',')[1])!=os.getpid()]
    require(not others,'Assigned GPU has an existing process; stop without retry: '+repr(others))
    return dict(physical_gpu=gpu,uuid=selected,other_processes=others,checked_unix=time.time())
