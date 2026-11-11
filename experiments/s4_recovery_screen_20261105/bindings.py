"""One frozen WebMed prefix, common parent, and two predeclared scopes."""
from pathlib import Path
from experiments.s4_webmed_recovery_20261105 import bindings as prior

torch,require,read,write,sha=prior.torch,prior.require,prior.read,prior.write,prior.sha
ROOT,HF,PYTHON,CATEGORIES=prior.ROOT,prior.HF,prior.PYTHON,prior.CATEGORIES
SOURCE,SOURCE_ID=prior.SOURCE,prior.SOURCE_ID
old,parent=prior.old,prior.parent
binding,event,source=prior.binding,prior.event,prior.source
initialize,environment,resources,memory=prior.initialize,prior.environment,prior.resources,prior.memory
restore_rope_alias=prior.restore_rope_alias
HERE=Path(__file__).resolve().parent
OUTPUT=prior.OUTPUT.parent/'scope_screen_20261105'
RUN=OUTPUT/'trial_20261105_scope64'
ALGORITHM='same_parent_S4_WebMed_fixed64_decoder_scope'
ARMS={'all':dict(gpu='6',stage=7,projections=252,leaves=1512),
      'attention':dict(gpu='7',stage=4,projections=144,leaves=864)}
CONTRACT=dict(prior.CONTRACT,algorithm=ALGORITHM,updates=64,groups=list(range(64)),observations=[0,64],
    trainable='all252/1512 or attention_qkvo144/864; all252 remain nativeS4',
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
    return {**prior.pins(),**{str(p):sha(p) for p in sorted([*HERE.glob('*.py'),*HERE.glob('*.sh')])}}


def dependency(arm):
    require(arm in ARMS,'Unknown predeclared scope.')
    if arm=='attention':
        return dict(policy='user_authorized_parallel_attention_20261105',old_WebMed_completion_required=False)
    report=read(prior.RUN/'webmed/run.json')
    require(report.get('exit_code')==0 and report['status']=='complete_fixed_local_comparison_arm',
        'Formal screen waits for the original WebMed128 plus full nine to finish; no polling or retry.')
    result=read(prior.RUN/'webmed/formal/full_nine/result.json')
    trained=read(prior.RUN/'webmed/formal/training.json')
    require(trained['exit_code']==0 and trained['actual_updates']==128
        and result['checkpoint_id']==trained['final']['checkpoint']['checkpoint_id'],
        'Original WebMed complete endpoint/UUID required before formal scope runs.')
    return dict(run=binding(prior.RUN/'webmed/run.json'),result=binding(prior.RUN/'webmed/formal/full_nine/result.json'))
