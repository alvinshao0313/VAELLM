"""Preserve the original64 identity; bind the separately authorized continuation."""
from pathlib import Path
from experiments.s4_recovery_screen_20261105 import bindings as prior

torch,require,read,write,sha=prior.torch,prior.require,prior.read,prior.write,prior.sha
ROOT,HF,PYTHON,CATEGORIES=prior.ROOT,prior.HF,prior.PYTHON,prior.CATEGORIES
SOURCE,SOURCE_ID=prior.SOURCE,prior.SOURCE_ID
old,parent=prior.old,prior.parent
binding,event,source=prior.binding,prior.event,prior.source
initialize,environment,resources,memory=prior.initialize,prior.environment,prior.resources,prior.memory
restore_rope_alias=prior.restore_rope_alias
ARMS,LEARNING_CONTRACT,PARENT_EIGHT=prior.ARMS,prior.LEARNING_CONTRACT,prior.PARENT_EIGHT
HERE=Path(__file__).resolve().parent
OUTPUT=prior.OUTPUT.parent/'attention128_20261106'
RUN=OUTPUT/'trial_20261106_attention64_to128'
PREDECESSOR=prior.RUN/'attention'
PREDECESSOR_ID='40b00612-b701-45ea-bf30-e922b8c1876e'
PREDECESSOR_STATE=PREDECESSOR/'training/training_state.pt'
ALGORITHM='same_parent_S4_attention64_to128_exact_named_continuation'
CONTRACT=dict(algorithm=ALGORITHM,parent_id=SOURCE_ID,predecessor_id=PREDECESSOR_ID,
    start_cursor=64,stop_cursor=128,executed_groups=list(range(64,128)),
    optimizer='Continue original FP32 masters and complete AdamW step64, RNG and cursor',
    schedule='Original500 steps; no reset; step65 LR4.36e-7, step128 LR3.73e-7',
    trainable='Original attention q/k/v/o144 projections /864 leaves; all252 remain nativeS4',
    observations=[128],diagnostics='Actual cursor determines early/late consumption; heldout never train',
    selection='One128 endpoint and complete eight B64/B32; no LongBench or automatic successor')


def pins():
    return {str(path):sha(path) for path in sorted([*HERE.glob('*.py'),*HERE.glob('*.sh')])}


def check_request(request):
    extension=request['continuation']
    require(extension['contract']==CONTRACT,'Continuation execution contract changed.')
    parent.s.check_pins(request['source_sha256']);parent.s.check_pins(extension['source_sha256'])
    require(extension['original_request']==binding(PREDECESSOR/'request.json')
        and extension['predecessor_state']==binding(PREDECESSOR_STATE)
        and extension['predecessor_result']==binding(PREDECESSOR/'result.json'),
        'Original request/state/result binding changed.')
    original=read(PREDECESSOR/'request.json')
    require({key:value for key,value in request.items() if key!='continuation'}==original,
        'The original request and source identity must remain unchanged.')
    require(request['arm']=='attention' and request['gpu']=='7' and not request['validation_only']
        and request['groups']==list(range(128)) and request['stop_step']==64
        and request['contract']==prior.CONTRACT and request['learning_contract']==LEARNING_CONTRACT
        and request['parent']==source(),'Original attention learning/source contract differs.')
