"""Fixed D16 geometry, semantic VAE owners and the original500 suffix contract."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from experiments.down_geometry_baseline_cmcc_20261103 import bindings as base
from experiments.cmcc_runtime_20261030 import sequence as executor
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS,PYTHON,REPO,TOPIC

HERE=Path(__file__).resolve().parent
OUTPUT,RESULT=ARTIFACTS/HERE.name,TOPIC/HERE.name
PREPARED=RESULT/'prepared.json'
PARENT,CONTROL,C0,SUPPORT,SNAPSHOT=base.PARENT,base.CONTROL,base.C0,base.SUPPORT,base.SNAPSHOT
GPU=1
SETTINGS=dict(gpu=1,dim=16,bits=24,batch=16384,transpose=True)
ARMS=dict(d16_muon=SETTINGS)
D16_RAW_REPORT=ARTIFACTS/'down_geometry_baseline_cmcc_20261103/d16/formal/report.json'
D16_RECOVERY_REPORT=ARTIFACTS/'down_geometry_baseline_cmcc_20261103/d16/recovery/report.json'
MUV_RECOVERY_REPORT=ARTIFACTS/'cat_down_vae_muon_closure_cmcc_20261103/formal/report.json'
C0_ACCEPTANCE=TOPIC/'fresh_fixed2_cmcc_20261030/single_20261030_v3/longbench/merged/nine_task_acceptance.json'
REFERENCE_ARGS=C0/'06_down_proj/trainer_state/checkpoint-500/training_args.bin'
ORIGINAL_MUV_OPTIMIZER=REPO/'experiments/cat_down_vae_muon_cmcc_20261103/optimizer.py'
read,write,sha,require,torch,c,checkpoint,recovery=base.read,base.write,base.sha,base.require,base.torch,base.c,base.checkpoint,base.recovery
CONTRACT=dict(schema='D16_VAE_Muon_original_Adam500_full9_v1',parent='5a94d57f-d3b9-4c97-adef-a099d659eb6a',
    settings=SETTINGS,M=36,residual_stages=2,updates_per_stage=10000,E=128,D=128,residual_blocks=0,
    VAE_parameter_elements=397728,VAE_matrix_elements=368640,VAE_auxiliary_elements=29088,
    VAE_optimizer=dict(momentum=.95,nesterov=True,NS_steps=5,NS_dtype='torch.bfloat16',momentum_dtype='torch.float32',
        NS_coefficients=[3.4445,-4.7750,2.0315],shape_scale='.2*sqrt(max(rows,cols))',
        parameter_gradient_dtype='torch.bfloat16',persistent_FP32_parameter_master=False,one_BF16_writeback=True,
        auxiliary_AdamW=dict(betas=[.9,.95],eps=1e-8,weight_decay=0.,base_lr=.003),
        kappa='Per-stage first real gradient, one common E+D BF16 Adam displacement match, max2 upper doublings/12 bisections/relative error<=.01; no transplanted D32 kappa.'),
    seed=31,shuffle_seed=600031,protection=256,rotation_block=32,pooled_BSQ_unchanged=True,
    suffix=dict(optimizer='torch.optim.AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,decoder_lr=1e-5,
        steps=500,batch=4,max_length=1024,dynamic_padding=True,stage_seed=37,data_seed=31,
        clip=1.5,scheduler='cosine',warmup_ratio=.01,save_steps=250,LoRA=0,FP32_projection_D=252),
    budget=dict(short_seconds=1800,fit_fresh_seconds=5400,Adam500_fresh_seconds=43200,full9_seconds=28800),
    raw_full8_or_readout_budget=0,automatic_retry=False,automatic_resume=False,
    authorization='Root explicitly authorized this single missing cell tonight before the three existing nine-task results finish; original evaluations unchanged.')


def pins():
    paths=set(HERE.glob('*.py'))
    paths.update((ORIGINAL_MUV_OPTIMIZER,ORIGINAL_MUV_OPTIMIZER.with_name('bindings.py'),D16_RAW_REPORT,REFERENCE_ARGS))
    paths.update(REPO/'experiments/adaptive_class_evaluation_20261030'/name for name in
        ('bindings.py','loading.py','full8.py','evaluate.py','summarize.py','predict.py','merge.py'))
    return {**base.pins(),**{str(path.resolve()):sha(path) for path in paths}}


def check_pins(values):
    executor.check_pins(values)


def sources():
    return base.sources()


def controls():
    raw=read(D16_RAW_REPORT)
    d16,muv,c0=read(D16_RECOVERY_REPORT),read(MUV_RECOVERY_REPORT),read(C0_ACCEPTANCE)
    require(raw['status']=='completed_formal_raw_down' and raw['exit_code']==0
        and raw['raw_checkpoint_id']=='02292fb6-5041-4f0d-bbb3-9827ec5f3a1f'
        and raw['fit']['actual_updates_per_stage']==10000 and raw['fit']['vector_batch_per_layer']==16384
        and d16['status']=='completed_own_down500' and d16['recovery']['actual_steps']==500
        and muv['status']=='completed_VAE_Muon_original_Adam500_and_fresh' and muv['recovery']['actual_steps']==500
        and c0['checkpoint_id']=='cbb15c9f-1c1f-4394-817c-65c226d76ce7',
        'Reuse the genuine D16 initialization/structure and three fixed final identities; never their down weights/states.')
    files=(D16_RAW_REPORT,D16_RECOVERY_REPORT,MUV_RECOVERY_REPORT,C0_ACCEPTANCE)
    return dict(D16_raw_checkpoint_id=raw['raw_checkpoint_id'],D16_raw_budget=raw['raw_budget'],
        D16_initializations=raw['fit']['actual_optimizer_initializations'],
        final_ids=dict(C0=c0['checkpoint_id'],D16=d16['final_checkpoint_id'],MuV=muv['final_checkpoint_id']),
        evidence_sha256={str(path):sha(path) for path in files},
        acceptance_paths=dict(C0=str(C0_ACCEPTANCE),
            D16=str(base.RESULT/'allocator_repair/d16/final_full9/longbench/merged/nine_task_acceptance.json'),
            MuV=str(TOPIC/'cat_down_vae_muon_closure_cmcc_20261103/evaluation/longbench/merged/nine_task_acceptance.json')))


def recipe(output):
    parsed,identity,baseline=base.recipe('d16',output)
    cfg=parsed[4]['down_proj']
    require(cfg.codebook_dim==16 and cfg.codebook_bits==24 and cfg.steps==10000 and cfg.residual_stages==2
        and cfg.base_ch==128 and cfg.num_res_blocks==0 and cfg.channel_protect_count==256
        and cfg.weight_rotation_block_size==32 and parsed[3].optimizer.lower()=='adamw'
        and parsed[3].beta1==.9 and parsed[3].beta2==.95 and parsed[3].lr==.003
        and parsed[3].vae_weight_dtype=='bf16' and parsed[3].weight_decay==0,
        'Use the actual D16/B16384 geometry and original BF16 reference optimizer; only matrix fitting optimizer is replaced.')
    return parsed,identity,baseline


def stage_args(parsed):
    from experiments.cat_down_vae_muon_cmcc_20261103.bindings import stage_args as original_args
    return original_args(parsed)


def recovery_recipe(output):
    parsed,identity,baseline=recipe(output)
    stage=recovery.canonical.resolve_cat_after_category_stage(parsed[0],parsed[2],category='down_proj',round_idx=6)
    actual=json.loads(json.dumps(asdict(stage)));expected=read(C0/'report.json')['stages'][2]['original_stage']
    actual.pop('output_dir');expected.pop('output_dir')
    require(actual==expected,'The full original down500 stage/data/optimizer contract may differ only by independent output_dir.')
    return parsed,identity,baseline,stage


def cpu_environment():
    base.cpu_environment()


def prepared():
    value=read(PREPARED)
    require(value['status']=='CPU_verified_D16_Muon_shape_batch_initialization_and_suffix' and value['contract']==CONTRACT,
        'Require this actual D16 owner/initialization/explicit batch CPU proof.')
    check_pins(value['source_sha256'])
    require(sources()==value['source_binding'] and controls()==value['controls'],'Actual parent or fixed control identities changed.')
    return value


def raw_source():
    report=read(OUTPUT/'formal/report.json')
    meta=read(OUTPUT/'formal/raw_model'/checkpoint.META_FILE)
    strict=read(OUTPUT/'formal/strict_reload.json')
    require(report['status']=='completed_D16_VAE_Muon_formal_raw' and report['exit_code']==0
        and report['phase']=='formal' and report['validation_only'] is False and report['contract']==CONTRACT
        and report['raw_checkpoint_id']==meta['checkpoint_id']==strict['checkpoint_id']
        and report['fit']['actual_updates_per_stage']==10000
        and report['fit']['original_D16_flat_initialization_matches']==[True,True]
        and report['packed_codes']['all_packed_bits_roundtrip_exact'] and strict['same_input_logits_exact']
        and strict['exit_code']==0 and Path(report['raw_model'])==OUTPUT/'formal/raw_model',
        'Only this new genuine fresh D16-Muon S2x10K/raw/fresh can feed its original Adam500 suffix.')
    return report,meta
