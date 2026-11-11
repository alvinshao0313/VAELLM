"""Protect the selected merge and its eight scores; evaluate only full LongBench."""
from pathlib import Path
import sys
from experiments.s4_model_merge_20261108 import bindings as prior

torch,require,read,write,sha=prior.torch,prior.require,prior.read,prior.write,prior.sha
ROOT,HF,PYTHON,CATEGORIES=prior.ROOT,prior.HF,prior.PYTHON,prior.CATEGORIES
old,parent=prior.old,prior.parent
binding,event=prior.binding,prior.event
initialize,environment,memory,resources=prior.initialize,prior.environment,prior.memory,prior.resources
restore_rope_alias=prior.restore_rope_alias
HERE=Path(__file__).resolve().parent
SOURCE_ROOT=prior.RUN/'r75'
SOURCE=SOURCE_ROOT/'post/model'
SOURCE_ID='838d2db4-48e7-48cd-b299-fc69790e7a35'
OUTPUT=prior.OUTPUT.parent/'merge75_longbench_20261108'
RUN=OUTPUT/'trial_20261108_complete21'
BPW=prior.BPW
GPUS=['4','5','6','7']
TERMINAL=ROOT/'result/qwen3_8b_mmlu65_20261004/llama_fullqk_qkonly2000_longbench21'
HELPERS=ROOT/'result/qwen3_8b_mmlu65_20261004/qwen_aux_retention32_batch_20261011/evaluate.py'
PARTITIONS=ROOT/'experiments/cat_aligned_evaluation_20261028/evaluate.py'
REFERENCE=prior.OUTPUT.parent/'webmed_20261105/trial_20261105_1223/webmed/formal/full_nine/longbench'
CONTRACT=dict(scope='Selected S4 model merge local quality acceptance; not cross-seed/model/bitwidth or BF16 full-chain proof',
    checkpoint_id=SOURCE_ID,ratio_late=.75,reuse_eight=True,datasets=21,samples=4750,
    partition='Existing aligned canonical LB_GROUPS round-robin across GPU4..7',
    no_training=True,no_checkpoint_changes=True,no_new_ratios=True,
    M_correct_minimum=10111,nine_mean_strict_minimum=67.64714491790848,
    expected_BPW=BPW,generalization='Unverified; same-seed same-trajectory merge screening; keep warning')


def partition_api():
    directory=str(parent.HERE)
    if directory not in sys.path:sys.path.insert(0,directory)
    return parent.module(PARTITIONS,'_merge75_existing_canonical_partitions')


def helper():
    result=parent.module(HELPERS,'_merge75_existing_outer_execution')
    result.S=ROOT;result.PYTHON=PYTHON
    return result


def pins():
    return {**prior.pins(),**{str(p):sha(p) for p in [*HERE.glob('*.py'),PARTITIONS,HELPERS,
        TERMINAL/'bind_terminal.py',TERMINAL/'merge_full21.py']}}


def check(request):
    require(request['contract']==CONTRACT and request['checkpoint']==event(SOURCE)
        and request['checkpoint']['checkpoint_id']==SOURCE_ID,'Selected native checkpoint changed.')
    parent.s.check_pins(request['source_sha256'])
    for record in request['eight'].values():require(binding(record['path'])==record,'Protected eight-task artifact changed.')
