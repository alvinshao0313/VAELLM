"""One isolated Web2048/Med2048 transfer from the exact S4 down/pre parent."""
from pathlib import Path
from experiments.s4_down_recovery_diagnosis_20261104 import bindings as prior

base,old,parent=prior.base,prior.old,prior.parent
torch,require,read,write,sha=prior.torch,prior.require,prior.read,prior.write,prior.sha
ROOT,HF,PYTHON,CATEGORIES=prior.ROOT,prior.HF,prior.PYTHON,prior.CATEGORIES
SOURCE,SOURCE_ID,UPSTREAM=prior.SOURCE,prior.SOURCE_ID,prior.UPSTREAM
CONTROL=prior.OUTPUT/'trial_20261104_1940/joint_half_lr/formal/post/model'
CONTROL_ID='8a874b00-a3ce-44f5-b869-7af1579cf988'
HERE=Path(__file__).resolve().parent
OUTPUT=prior.prior.OUTPUT/'webmed_20261105'
RUN=OUTPUT/'trial_20261105_1223'
DATA=RUN/'data'
RAW=Path('/data1/cmcc/VAELLM/data/s4_webmed_sources_20261105')
ALGORITHM='same_parent_S4_Web2048_Med2048_recovery'
CONTRACT=dict(prior.CONTRACT,algorithm=ALGORITHM,control_id=CONTROL_ID,
    M_sources='WebInstruct2048+MedMCQA2048 unique once; fixed hash causal_output_fit/137/',
    per_group_sources=dict(webinstruct=16,medmcqa=16,QA=8),
    teacher_gold_mixture=[.5,.5],base_lr=5e-7,
    heldout='64/source pinned official train, excluding all2048/source; gold only, never train',
    diagnostics='early128 and late128 contain64/source; heldout128 contains64/source',
    updates=128,observations=[0,64,128])
ARMS={'webmed':dict(gpu='6',base_lr=5e-7,include_QA=True)}
binding,event,initialize,environment=prior.binding,prior.event,prior.initialize,prior.environment
restore_rope_alias,resources,memory,source=prior.restore_rope_alias,prior.resources,prior.memory,prior.source


def pins():
    result=prior.pins()
    result.update({str(path):sha(path) for path in sorted([*HERE.glob('*.py'),*HERE.glob('*.sh')])})
    helpers=ROOT/'result/qwen3_8b_mmlu65_20261004'
    for path in [helpers/'qwen_webinstruct_train_inventory_20261015/web_inputs.py',
                 helpers/'qwen_webinstruct_train_inventory_20261015/inventory.py',
                 helpers/'qwen_webinstruct_train_inventory_20261015/download_train.py',
                 helpers/'qwen_open_mcqa_coverage64_20261014/coverage_inputs.py']:
        result[str(path)]=sha(path)
    if DATA.exists():
        for name in ['dataset_prepared.json','dataset.pt','heldout.pt','teacher.json','raw_teacher.pt','data_contract.json']:
            path=DATA/name
            if path.exists():result[str(path)]=sha(path)
    return result


from . import data
