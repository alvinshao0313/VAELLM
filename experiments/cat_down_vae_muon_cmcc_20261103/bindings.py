"""One optimizer-only fork of the genuine C0 live-down VAE fit."""
from copy import deepcopy
from pathlib import Path
import os
import sys

from experiments.down_geometry_baseline_cmcc_20261103 import bindings as base
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC

HERE=Path(__file__).resolve().parent
OUTPUT,RESULT=ARTIFACTS / HERE.name,TOPIC / HERE.name
PARENT,CONTROL,SUPPORT,SNAPSHOT=base.PARENT,base.CONTROL,base.SUPPORT,base.SNAPSHOT
GPU=2
read,write,sha,require,torch=base.read,base.write,base.sha,base.require,base.torch
checkpoint,recovery,c=base.checkpoint,base.recovery,base.c
CONTROL_RESULT=TOPIC / 'down_geometry_baseline_cmcc_20261103/allocator_repair/control'
CONTRACT=dict(schema='cat_down_vae_muon_bf16_v1',scope='VAE encoder and decoder fitting only',
    dimensions=dict(D=32,K=48,M=36,batch=8192,residual_stages=2,updates_per_stage=10000,transpose=True),
    parameter_dtype='torch.bfloat16',gradient_dtype='torch.bfloat16',persistent_FP32_parameter_master=False,
    momentum=.95,nesterov=True,NS_steps=5,NS_dtype='torch.bfloat16',momentum_dtype='torch.float32',
    NS_coefficients=[3.4445,-4.7750,2.0315],shape_scale='.2*sqrt(max(rows,cols))',
    auxiliary_AdamW=dict(betas=[.9,.95],eps=1e-8,weight_decay=0,base_lr=.003),
    external_schedule='Original linear 10000, no warmup; scheduled base LR multiplied by fixed stage kappa inside matrix step.',
    kappa=dict(per_stage=1,first_actual_gradient=True,reference='Native BF16 AdamW actual matrix displacement',
        initial_interval='[0,2*kappa0]',maximum_upper_doublings=2,maximum_bisections=12,target_relative_error=.01),
    seed=31,shuffle_seed=600031,automatic_recovery_or_full9=False)


def pins():
    return {**base.pins(),**{str(path):sha(path) for path in sorted(HERE.glob('*.py'))}}


def check_pins(values):
    base.check_pins(values)


def sources():
    binding=base.sources()
    old=read(base.RESULT / 'prepared.json')
    require(old['status']=='cpu_verified' and old['source_binding']==binding,
        'Reuse only the already verified exact same after-up/live-down/frozen-prefix evidence.')
    summary=read(CONTROL_RESULT / 'raw_full8/full8_summary.json')
    require(summary['checkpoint_id']==binding['raw_control']['checkpoint_id']
        and summary['leaf_count']==64 and summary['sample_count']==24742,
        'The original raw AdamW control requires its already completed complete-eight evidence.')
    return binding,dict(live_down_source_digest=old['live_down_source_digest'],frozen_prefix_digest=old['frozen_prefix_digest'],
        reused_CPU_proof=str(base.RESULT / 'prepared.json'),reused_CPU_proof_sha256=sha(base.RESULT / 'prepared.json'),
        control_full8=str(CONTROL_RESULT / 'raw_full8/full8_summary.json'),
        control_full8_sha256=sha(CONTROL_RESULT / 'raw_full8/full8_summary.json'),
        control_readout=str(CONTROL_RESULT / 'raw_readout.json'),control_readout_sha256=sha(CONTROL_RESULT / 'raw_readout.json'))


def recipe(output):
    parsed,identity,baseline=base.recipe('control',output)
    require(identity==baseline and parsed[4]['down_proj'].steps==10000,
        'The complete native recipe stays unchanged; the actual factory replacement is recorded separately.')
    vae=parsed[3]
    require(vae.optimizer.lower()=='adamw' and vae.beta1==.9 and vae.beta2==.95
        and vae.lr==.003 and vae.weight_decay==0 and vae.vae_weight_dtype=='bf16',
        'The native BF16 AdamW reference/auxiliary optimizer changed.')
    return parsed,identity


def stage_args(parsed):
    args=deepcopy(parsed[3])
    cfg=parsed[4]['down_proj']
    for name,value in dict(parallel_layers=36,residual_stages=2,codebook_bits=cfg.codebook_bits,
        codebook_dim=cfg.codebook_dim,base_ch=cfg.base_ch,num_res_blocks=cfg.num_res_blocks,
        norm_type=cfg.norm_type,activation_type=cfg.activation_type,decoder_type=cfg.decoder_type,
        decoder_base_ch=cfg.decoder_base_ch,decoder_num_res_blocks=cfg.decoder_num_res_blocks,
        recon_loss_type=cfg.recon_loss_type).items():
        setattr(args,name,value)
    recovery.native.apply_autoencoder_arch_defaults(args)
    return args


def cpu_environment():
    require(Path(sys.executable).resolve()==PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),
        'Use caller bitvae and hidden CUDA for the CPU task.')
