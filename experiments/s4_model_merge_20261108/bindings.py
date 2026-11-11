"""Fixed donors, three global ratios and original complete-eight protocol."""
from pathlib import Path
from experiments.s4_scope_objective_20261107 import bindings as prior

torch,require,read,write,sha=prior.torch,prior.require,prior.read,prior.write,prior.sha
ROOT,HF,PYTHON,CATEGORIES=prior.ROOT,prior.HF,prior.PYTHON,prior.CATEGORIES
SOURCE,SOURCE_ID,BPW=prior.SOURCE,prior.SOURCE_ID,prior.BPW
old,parent=prior.old,prior.parent
binding,event,source=prior.binding,prior.event,prior.source
initialize,environment,memory,resources=prior.initialize,prior.environment,prior.memory,prior.resources
restore_rope_alias=prior.restore_rope_alias
HERE=Path(__file__).resolve().parent
OUTPUT=prior.OUTPUT.parent/'model_merge_20261108'
RUN=OUTPUT/'trial_20261108_D64_D128'
ALGORITHM='S4_D64_D128_FP32_parameter_model_merge'
DONOR_ROOT=prior.RUN/'D'
DONORS={64:dict(checkpoint_id='4ea78091-808c-439b-9d5e-b7e5f5e2230b',correct=10083),
        128:dict(checkpoint_id='5e292933-9317-474a-b3b5-80c3b34215e7',correct=10065)}
ARMS={'r25':dict(gpu='4',ratio_late=.25),'r50':dict(gpu='5',ratio_late=.5),'r75':dict(gpu='6',ratio_late=.75)}
CONTRACT=dict(algorithm=ALGORITHM,ratios_late=[.25,.5,.75],
    formula='FP32 (1-r)*master64+r*master128 for all1512 decoder and252 protected integer proxy leaves',
    deploy='BF16 decoder; round(clamp(q,-128,127)) signedINT8 storedUINT8; existing scale/index and mainBSQ codes exact',
    observations='Native independent fresh and original complete eight; reuse donor evaluations',
    no_training=True,no_optimizer_merge=True,selection='Only three predeclared global ratios; no per-layer/per-item search',
    expected_bpw=BPW,acceptance='Screen only; final M>=10111 and nine_mean>67.64714492 not yet accepted',
    generalization='Same seed/same trajectory; existing benchmark screening is not independent generalization evidence; fixed rule requires new seed/model/bitwidth checks')


def donor_directory(step):
    return DONOR_ROOT/f'step_{step:06d}'


def pins():
    return {**prior.pins(),**{str(p):sha(p) for p in sorted([*HERE.glob('*.py'),*HERE.glob('*.sh')])}}


def check(request):
    require(request['arm'] in ARMS and request['recipe']==ARMS[request['arm']] and request['contract']==CONTRACT
        and request['parent']==source(),'Merge contract/source changed.')
    parent.s.check_pins(request['source_sha256'])
