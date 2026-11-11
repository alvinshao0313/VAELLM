"""Original S4 parent, fixed QK consumer, and explicit per-arm mixture contracts."""
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
OUTPUT=previous.OUTPUT.parent/'qk_alpha_20261107'
RUN=OUTPUT/'trial_20261107_alpha25_75'
ALGORITHM='same_parent_S4_QK_fixed64_teacher_alpha'
ARMS={'alpha25':dict(gpu='4',alpha=.25,projections=72,leaves=432),
      'alpha75':dict(gpu='5',alpha=.75,projections=72,leaves=432)}
CONTROL_ID='40f76c27-8f58-4bf8-b448-08af2ea439c9'
BASE_CONTRACT=dict(previous.CONTRACT,algorithm=ALGORITHM,control_id=CONTROL_ID,
    trainable='exact QK72/432 from the original down/pre; frozen180 retain input backward; all252 nativeS4',
    variable='Knowledge teacher alpha only; raw teacher mixed once with gold; QA unchanged',
    selection='Two fixed64 endpoints vs existing QK alpha.5; no automatic128, LongBench, data or dynamic alpha')
PARENT_EIGHT=previous.PARENT_EIGHT
CONTROL_ROOT=previous.RUN/'qk'
CONTROL_EIGHT={name:CONTROL_ROOT/'post/evaluation'/(name+'.json') for name in ('mmlu','seven')}
QK_VALIDATION=previous.RUN/'validation_qk/result.json'


def contract(arm):
    alpha=ARMS[arm]['alpha']
    return dict(BASE_CONTRACT,teacher_alpha=alpha,teacher_gold_mixture=[alpha,1-alpha])


def learning_contract(arm):
    alpha=ARMS[arm]['alpha']
    return dict(previous.LEARNING_CONTRACT,teacher_alpha=alpha,teacher_gold_mixture=[alpha,1-alpha],
        target='Raw original WebMed conditional4 once-mixed in log space; unchanged gold-prefix static full-vocabulary QA KL')


def pins():
    return {**previous.pins(),**{str(p):sha(p) for p in sorted([*HERE.glob('*.py'),*HERE.glob('*.sh')])}}


def resources(gpu):
    require(gpu in ('4','5'),'This alpha batch only uses assigned GPU4/5.')
    return previous.resources(gpu)


def dependency(arm):
    require(arm in ARMS,'Unknown predeclared teacher alpha.')
    return dict(policy='user_started_QK_alpha25_75_20261107',parent=source(),
        reused_QK_validation=binding(QK_VALIDATION),control_result=binding(CONTROL_ROOT/'result.json'))
