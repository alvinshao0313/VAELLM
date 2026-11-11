"""Original S4 parent and exact QK scope; fixed source loss weights only."""
from pathlib import Path
from experiments.s4_scope_refine_20261106 import bindings as previous

torch,require,read,write,sha=previous.torch,previous.require,previous.read,previous.write,previous.sha
ROOT,HF,PYTHON,CATEGORIES=previous.ROOT,previous.HF,previous.PYTHON,previous.CATEGORIES
SOURCE,SOURCE_ID=previous.SOURCE,previous.SOURCE_ID
old,parent,prior=previous.old,previous.parent,previous.prior
binding,event,source=previous.binding,previous.event,previous.source
initialize,environment,memory=previous.initialize,previous.environment,previous.memory
restore_rope_alias=previous.restore_rope_alias
HERE=Path(__file__).resolve().parent
OUTPUT=previous.OUTPUT.parent/'qk_source_weights_20261107'
RUN=OUTPUT/'trial_20261107_web3_med3'
ALGORITHM='same_parent_S4_QK_fixed64_source_loss_weights'
ARMS={'web3':dict(gpu='4',alpha=.5,source_weights=dict(webinstruct=.375,medmcqa=.125,QA=.5),projections=72,leaves=432),
      'med3':dict(gpu='5',alpha=.5,source_weights=dict(webinstruct=.125,medmcqa=.375,QA=.5),projections=72,leaves=432)}
CONTROL_ID='40f76c27-8f58-4bf8-b448-08af2ea439c9'
BASE_CONTRACT=dict(previous.CONTRACT,algorithm=ALGORITHM,control_id=CONTROL_ID,
    trainable='exact QK72/432 from the original down/pre; frozen180 retain input backward; all252 nativeS4',
    variable='Web/Med source-mean loss weights only; inputs, order, targets and Web16/Med16/QA8 unchanged',
    selection='Two fixed64 endpoints vs existing QK equal-source .5 control; no automatic128, LongBench, dynamic alpha or new data')
PARENT_EIGHT=previous.PARENT_EIGHT
CONTROL_ROOT=previous.RUN/'qk'
CONTROL_EIGHT={name:CONTROL_ROOT/'post/evaluation'/(name+'.json') for name in ('mmlu','seven')}
QK_VALIDATION=previous.RUN/'validation_qk/result.json'


def contract(arm):
    weights=ARMS[arm]['source_weights']
    return dict(BASE_CONTRACT,teacher_alpha=.5,teacher_gold_mixture=[.5,.5],source_weights=weights,
        per_case_weights=dict(webinstruct=weights['webinstruct']/16,medmcqa=weights['medmcqa']/16,QA=.5/8),
        objective='sum(source mean KL * source weight); no extra batch division or knowledge factor')


def learning_contract(arm):
    return dict(previous.LEARNING_CONTRACT,teacher_alpha=.5,source_weights=ARMS[arm]['source_weights'],
        per_case_weights=contract(arm)['per_case_weights'])


def pins():
    return {**previous.pins(),**{str(p):sha(p) for p in sorted([*HERE.glob('*.py'),*HERE.glob('*.sh')])}}


def resources(gpu):
    require(gpu in ('4','5'),'This source-weight batch only uses assigned GPU4/5.')
    return previous.resources(gpu)


def dependency(arm):
    require(arm in ARMS,'Unknown predeclared source-weight arm.')
    return dict(policy='user_started_QK_source_loss_weights_20261107',parent=source(),
        reused_QK_validation=binding(QK_VALIDATION),control_result=binding(CONTROL_ROOT/'result.json'))
