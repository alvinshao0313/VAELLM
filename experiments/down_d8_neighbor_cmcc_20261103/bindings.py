"""One D8/K12/B32768 fork; source and existing D16/32 controls are read-only."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path

from experiments.down_geometry_baseline_cmcc_20261103 import bindings as base
from experiments.cmcc_runtime_20261030 import sequence as executor
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS,PYTHON,REPO,TOPIC
from . import consumer

HERE=Path(__file__).resolve().parent
OUTPUT,RESULT=ARTIFACTS/HERE.name,TOPIC/HERE.name
PARENT,CONTROL,C0,SUPPORT,SNAPSHOT=base.PARENT,base.CONTROL,base.C0,base.SUPPORT,base.SNAPSHOT
PREPARED=RESULT/'prepared.json'
D16_OUTPUT=ARTIFACTS/'down_geometry_baseline_cmcc_20261103/d16/formal'
D16_RESULT=base.RESULT/'allocator_repair/d16'
D16_ID='02292fb6-5041-4f0d-bbb3-9827ec5f3a1f'
GPU=1
SETTINGS=dict(gpu=1,dim=8,bits=12,batch=32768,transpose=True)
ARMS=dict(d8=SETTINGS)
read,write,sha,require,torch,c,checkpoint,recovery=base.read,base.write,base.sha,base.require,base.torch,base.c,base.checkpoint,base.recovery
CONTRACT=dict(schema='same_parent_down_D8_K12_neighbor_v1',parent='5a94d57f-d3b9-4c97-adef-a099d659eb6a',
    main_control=D16_ID,reference_control='ed507bcb-9b83-4e4c-9859-daaef44d80d2',settings=SETTINGS,
    residual_stages=2,updates_per_stage=10000,M=36,encoder=128,decoder=128,residual_blocks=0,
    input_protection=256,rotation_block=32,seed=31,shuffle_seed=600031,
    optimizer='Original BF16 native AdamW(.9,.95)/eps1e-8/WD0/base.003/linear10000; unchanged original losses and pooled BSQ.',
    exposure='B*D=262144 scalar exposure only; K/BSQ scales and compute differ. Not pure-D or equal-FLOPs/equal physical storage.',
    budget='short2x2<=1800s; fresh formal2x10000+deployment/fresh<=5400s; complete raw8<=5400s; fixed train readout<=1800s.',
    logical_limit=2.6,packed_limit_is_acceptance=False,recovery_or_full9_budget=0,
    automatic_retry=False,automatic_resume=False)
EFFECTIVE=consumer.install()


def pins():
    paths=set(HERE.glob('*.py'))
    paths.update((consumer.ORIGINAL,Path(torch.__file__).parent.parent/'triton/backends/nvidia/compiler.py'))
    return {**base.pins(),**{str(path.resolve()):sha(path) for path in paths}}


def check_pins(values):
    executor.check_pins(values)


def sources():
    return base.sources()


def control():
    report=read(D16_OUTPUT/'report.json')
    summary=read(D16_RESULT/'raw_full8/full8_summary.json')
    readout=read(D16_RESULT/'raw_readout.json')
    require(report['status']=='completed_formal_raw_down' and report['exit_code']==0
        and report['raw_checkpoint_id']==summary['checkpoint_id']==readout['checkpoint_id']==D16_ID
        and summary['leaf_count']==64 and summary['sample_count']==24742
        and summary['mean_percent']==67.01259272688102,
        'Reuse only completed D16 raw/full8/readout as the main control, without reevaluation or importing down state.')
    paths=(D16_OUTPUT/'report.json',D16_OUTPUT/'raw_model'/checkpoint.META_FILE,
        D16_RESULT/'raw_full8/full8_summary.json',D16_RESULT/'raw_readout.json')
    return dict(checkpoint_id=D16_ID,report=str(paths[0]),summary=str(paths[2]),readout=str(paths[3]),
        evidence_sha256={str(path):sha(path) for path in paths},mean_percent=summary['mean_percent'],budget=report['raw_budget'])


def recipe(output):
    parsed,_identity,baseline=base.recipe('control',output)
    cat,hf,training,vae,cfgs,runtime=parsed
    cfgs=dict(cfgs)
    cfgs['down_proj']=replace(cfgs['down_proj'],codebook_dim=8,codebook_bits=12)
    identity=base.build_cat_cross_category_runtime_identity(cat_args=cat,vae_args=vae,resolved_category_cfgs=cfgs,
        compression_categories=recovery.CATEGORIES,target_layers='all',skip_layers=(),transpose_modules=cat.transpose_modules.split(','))
    identity['experiment_runtime']=deepcopy(baseline['experiment_runtime'])
    identity['experiment_runtime']['category_vector_batches']['down_proj']=32768
    expected=deepcopy(baseline)
    expected['resolved_category_runtime']['down_proj'].update(codebook_dim=8,codebook_bits=12)
    expected['experiment_runtime']['category_vector_batches']['down_proj']=32768
    require(identity==expected and 32768*8==262144,'Only this fixed D/K/B neighbor may change the original recipe.')
    stage=recovery.canonical.resolve_cat_after_category_stage(cat,training,category='down_proj',round_idx=6)
    actual=json.loads(json.dumps(asdict(stage)))
    reference=read(C0/'report.json')['stages'][2]['original_stage']
    actual.pop('output_dir');reference.pop('output_dir')
    require(actual==reference and vae.optimizer.lower()=='adamw' and vae.beta1==.9 and vae.beta2==.95
        and vae.lr==.003 and vae.weight_decay==0 and vae.vae_weight_dtype=='bf16'
        and cfgs['down_proj'].steps==10000 and cfgs['down_proj'].residual_stages==2,
        'Preserve all original live-source fitting/optimizer/data and recovery configuration fields.')
    return (cat,hf,training,vae,cfgs,runtime),identity,baseline


def cpu_environment():
    base.cpu_environment()


def prepared():
    value=read(PREPARED)
    require(value['status']=='CPU_verified_D8_parent_geometry_and_two_tile_extension' and value['contract']==CONTRACT,
        'Require this new-shape CPU proof.')
    check_pins(value['source_sha256'])
    require(sources()==value['source_binding'] and control()==value['D16_control'], 'The actual source or control evidence changed.')
    return value


def require_core_review():
    review=read(RESULT/'core_review.json')
    require(review['status']=='approved_GPU_short' and review['prepared_sha256']==sha(PREPARED)
        and review['effective_consumer']==consumer.contract(),'The explicit Astra kernel/core review must release this exact CPU-bound source before GPU.')
    return review
