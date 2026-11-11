"""Bind one completed VAE-Muon raw to the unchanged C0 category500 contract."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from experiments.fresh_fixed2_cmcc_20261030 import sources as original
from experiments.down_geometry_baseline_cmcc_20261103 import bindings as base
from experiments.cmcc_runtime_20261030 import sequence as executor
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC
from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_vector_rvq import common as c

HERE=Path(__file__).resolve().parent
OUTPUT,RESULT=ARTIFACTS/HERE.name,TOPIC/HERE.name
PREPARED=RESULT/'prepared.json'
V_OUTPUT=ARTIFACTS/'cat_down_vae_muon_cmcc_20261103'
V_RESULT=TOPIC/'cat_down_vae_muon_cmcc_20261103'
RAW=V_OUTPUT/'formal/raw_model'
RAW_ID='c5ccf174-0c56-4f90-9e29-7d4ba63d2e23'
C0=base.C0
CONTROL_ID='cbb15c9f-1c1f-4394-817c-65c226d76ce7'
CONTROL_ACCEPTANCE=TOPIC/'fresh_fixed2_cmcc_20261030/single_20261030_v3/longbench/merged/nine_task_acceptance.json'
REFERENCE_ARGS=C0/'06_down_proj/trainer_state/checkpoint-500/training_args.bin'
SUPPORT=base.SUPPORT
GPU=2
torch,read,write,sha,require=c.torch,c.read,c.write,c.sha,c.require
CONTRACT=dict(schema='VAE_Muon_original_Adam500_closure_v1',parent=RAW_ID,control=CONTROL_ID,
    category='down_proj',projection_decoders=252,trainable_leaves=1728,trainable_elements=5336064,
    optimizer=dict(cls='torch.optim.AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,decoder_lr=1e-5,fresh=True),
    precision='Original all252 FP32 decoder masters; BF16 compute and deployment; LoRA0.',
    schedule=dict(steps=500,type='cosine',warmup_ratio=.01,save_steps=250,max_grad_norm=1.5),
    stage_seed=37,data_seed=31,batch=4,model_max_length=1024,dynamic_padding=True,
    data_target='Unchanged canonical original five-source train, teacher, losses and reduction.',
    source_scope='Original C0 BF16 -> common after_o -> gate -> after_up -> VAE-Muon S2x10K raw; no D16, Rice or category-Muon optimizer.',
    completion='One original Adam500 plus deployment/fresh within12h; same final UUID complete nine within8h, regardless of intermediate quality.',
    budget_after_closure=0,automatic_retry=False,automatic_resume=False)


def normalized(value):
    return json.loads(json.dumps(value))


def pins():
    paths=set(HERE.glob('*.py'))
    paths.update(REPO/'experiments/adaptive_class_evaluation_20261030'/name for name in
        ('bindings.py','loading.py','full8.py','evaluate.py','summarize.py','predict.py','merge.py'))
    paths.update((Path(recovery.__file__).with_name('training_checkpoint.py'),REFERENCE_ARGS,CONTROL_ACCEPTANCE))
    return {**original.source_pins(),**{str(path.resolve()):sha(path) for path in paths}}


def check_pins(values):
    executor.check_pins(values)


def source():
    upstream=base.sources()
    raw,run=read(V_OUTPUT/'formal/report.json'),read(V_RESULT/'run.json')
    meta=read(RAW/checkpoint.META_FILE)
    strict=read(V_OUTPUT/'formal/strict_reload.json')
    full8=read(V_RESULT/'raw_full8/full8_summary.json')
    readout=read(V_RESULT/'raw_readout.json')
    control=read(CONTROL_ACCEPTANCE)
    require(raw['status']=='completed_VAE_Muon_formal_raw' and raw['exit_code']==0
        and raw['phase']=='formal' and not raw['validation_only'] and raw['source_binding']==upstream
        and run['status']=='completed_VAE_Muon_raw_eight_and_readout' and run['exit_code']==0
        and all(row['exit_code']==0 for row in run['stages'].values())
        and raw['raw_checkpoint_id']==meta['checkpoint_id']==strict['checkpoint_id']==full8['checkpoint_id']==readout['checkpoint_id']==RAW_ID
        and Path(raw['raw_model']).resolve()==RAW and strict['same_input_logits_exact']
        and strict['status']=='strict_full_state_and_BF16_logits_verified' and strict['exit_code']==0
        and full8['leaf_count']==64 and full8['sample_count']==24742
        and raw['fit']['actual_updates_per_stage']==10000 and raw['fit']['actual_stages']==2
        and [len(meta[key]) for key in ('native_modules','compact_modules','pending_dense_modules')]==[216,36,0],
        'Require the completed actual VAE-Muon S2x10K raw/fresh/full8/readout, never short or another geometry.')
    require(sha(RAW/checkpoint.STATE_FILE)==meta['state_sha256']
        and read(C0/'report.json')['final_checkpoint_id']==CONTROL_ID
        and control['checkpoint_id']==CONTROL_ID and control['compressed_nine_task_score']==63.58416595020284,
        'The actual V raw payload or unique original AdamW500/full9 control changed.')
    paths=(V_OUTPUT/'formal/report.json',V_OUTPUT/'formal/strict_reload.json',V_RESULT/'run.json',
        V_RESULT/'raw_full8/full8_summary.json',V_RESULT/'raw_readout.json',CONTROL_ACCEPTANCE,C0/'report.json')
    return dict(raw_directory=str(RAW),raw_checkpoint_id=RAW_ID,raw_metadata_sha256=sha(RAW/checkpoint.META_FILE),
        raw_state_sha256=meta['state_sha256'],upstream_binding=upstream,
        evidence_sha256={str(path):sha(path) for path in paths},control_checkpoint_id=CONTROL_ID,
        control_acceptance=str(CONTROL_ACCEPTANCE),control_nine_task_score=control['compressed_nine_task_score'])


def recipe(output):
    common=original.bind(base.COMMON)
    cat,hf,training,vae,cfgs,runtime=original.recipe(output,common)
    stage=recovery.canonical.resolve_cat_after_category_stage(cat,training,category='down_proj',round_idx=6)
    actual,expected=normalized(asdict(stage)),read(C0/'report.json')['stages'][2]['original_stage']
    actual.pop('output_dir');expected.pop('output_dir')
    require(actual==expected and stage.stage_seed==37 and stage.config.data.data_seed==31
        and stage.config.opt.steps==500 and stage.config.opt.batch_size==4
        and stage.config.data.model_max_length==1024 and stage.config.data.dynamic_padding
        and stage.config.opt.optim=='adamw_torch' and stage.config.opt.resolved_decoder_lr()==1e-5
        and stage.config.opt.max_grad_norm==1.5 and stage.config.opt.warmup_ratio==.01
        and stage.config.opt.lr_scheduler_type=='cosine' and stage.save_steps==250,
        'All original stage/data/loss/optimizer fields must match after JSON normalization except independent output_dir.')
    return cat,hf,training,vae,runtime,stage


def cpu_environment():
    executor.cpu_environment()
    require(not torch.cuda.is_initialized(),'CPU preparation must not initialize CUDA.')


def prepared():
    value=read(PREPARED)
    require(value['status']=='verified_CPU_V_parent_and_original_Adam500_contract' and value['contract']==CONTRACT,
        'Require this exact approved closure CPU proof.')
    check_pins(value['source_sha256'])
    require(source()==value['source_binding'],'Bound VAE raw/control evidence changed.')
    return value
