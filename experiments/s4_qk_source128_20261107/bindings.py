"""Preserve both original learning identities and bind this bounded continuation."""
from pathlib import Path
from experiments.s4_scope_refine_20261106 import bindings as equal
from experiments.s4_qk_source_weights_20261107 import bindings as weighted

torch,require,read,write,sha=equal.torch,equal.require,equal.read,equal.write,equal.sha
ROOT,HF,PYTHON,CATEGORIES=equal.ROOT,equal.HF,equal.PYTHON,equal.CATEGORIES
SOURCE,SOURCE_ID=equal.SOURCE,equal.SOURCE_ID
old,parent=equal.old,equal.parent
binding,event,source=equal.binding,equal.event,equal.source
initialize,environment,resources,memory=equal.initialize,equal.environment,equal.resources,equal.memory
restore_rope_alias=equal.restore_rope_alias
PARENT_EIGHT=equal.PARENT_EIGHT
HERE=Path(__file__).resolve().parent
OUTPUT=equal.OUTPUT.parent/'qk_source128_20261107'
RUN=OUTPUT/'trial_20261107_equal_web3_64to128'
ALGORITHM='S4_QK_equal_and_Web3_exact64_to128'
ARMS={
    'qk':dict(gpu='4',predecessor=equal.RUN/'qk',checkpoint_id=weighted.CONTROL_ID,
        source_weights=dict(webinstruct=.25,medmcqa=.25,QA=.5),projections=72),
    'web3':dict(gpu='5',predecessor=weighted.RUN/'web3',checkpoint_id='93deb5f1-7847-4546-9f2c-f6dce6585029',
        source_weights=dict(webinstruct=.375,medmcqa=.125,QA=.5),projections=72)}
CONTRACT=dict(algorithm=ALGORITHM,parent_id=SOURCE_ID,start_cursor=64,stop_cursor=128,
    executed_groups=list(range(64,128)),optimizer='Continue actual FP32 masters, AdamW step64, RNG and cursor',
    schedule='Original500; step65 LR4.36e-7; step128 LR3.73e-7; no reset',
    trainable='QK72 projections/432 leaves; all252 nativeS4',
    observations=[128],progress_update=96,
    selection='Two complete128 eight-task endpoints; no LongBench or automatic successor',
    exposure='Each source2048 once cumulatively; QA128 each8; steps and new knowledge coverage both increase')


def predecessor_state(arm):
    return ARMS[arm]['predecessor']/'training/training_state.pt'


def pins():
    paths=[*HERE.glob('*.py'),*HERE.glob('*.sh'),
        ROOT/'experiments/s4_attention128_20261106/core.py',
        ROOT/'experiments/s4_qk_source_weights_20261107/result.py']
    return {str(path):sha(path) for path in sorted(paths)}


def check_request(request):
    arm=request['arm'];spec=ARMS[arm];extension=request['continuation']
    require(extension['contract']==CONTRACT and extension['gpu']==spec['gpu'], 'Continuation contract/GPU changed.')
    parent.s.check_pins(request['source_sha256']);parent.s.check_pins(extension['source_sha256'])
    require(extension['original_request']==binding(spec['predecessor']/'request.json')
        and extension['predecessor_state']==binding(predecessor_state(arm))
        and extension['predecessor_result']==binding(spec['predecessor']/'result.json'), 'Original64 binding changed.')
    original=read(spec['predecessor']/'request.json')
    require({k:v for k,v in request.items() if k!='continuation'}==original,
        'Original learning request must remain unchanged, including original GPU metadata.')
    require(request['parent']==source() and not request['validation_only']
        and request['groups']==list(range(128)) and request['stop_step']==64,
        'Original parent/group/stop identity changed.')
    expected_contract=equal.CONTRACT if arm=='qk' else weighted.contract(arm)
    expected_learning=equal.LEARNING_CONTRACT if arm=='qk' else weighted.learning_contract(arm)
    require(request['contract']==expected_contract and request['learning_contract']==expected_learning,
        'Original learning conditions changed.')
