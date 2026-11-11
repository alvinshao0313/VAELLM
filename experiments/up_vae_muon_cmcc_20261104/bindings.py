"""Actual C0 after-gate/up-raw identities; no copied model or new predecessor."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import sys

from experiments.fresh_fixed2_cmcc_20261030 import sources as original
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC
from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_vector_rvq import common as c
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state, build_cat_cross_category_runtime_identity

HERE = Path(__file__).resolve().parent
OUTPUT, RESULT = ARTIFACTS / HERE.name, TOPIC / HERE.name
COMMON = ARTIFACTS / 'fresh_common_cmcc_20261030/formal'
C0 = ARTIFACTS / 'fresh_fixed2_cmcc_20261030/single_20261030_v3_C0'
PARENT, CONTROL = C0 / '04_gate_proj/after_recovery', C0 / '05_up_proj/round_base'
SUPPORT = ARTIFACTS / 'adaptive_class_cmcc_20261030/evaluation_support'
MATURE_OPTIMIZER = REPO / 'experiments/cat_down_vae_muon_cmcc_20261103/optimizer.py'
read, write, sha, require, torch = c.read, c.write, c.sha, c.require, original.torch
ARMS = dict(muon=dict(gpu=1, dim=32, bits=32, batch=8192), d16=dict(gpu=3, dim=16, bits=16, batch=16384))
CONTROL_GPU = 5
CONTRACT = dict(category='up_proj', source_after_gate='dafee7a6-2301-47a6-9412-d6dfaa1b61ed',
    raw_control='44378f2a-a34a-457a-8384-3dab3fcef13b', M=36, S=2, E=128, decoder=128,
    decoder_type='symmetric', residual_blocks=0, norm='layer', activation='swish',
    protection_input64_INT8=True, rotation_block=32, transpose=False,
    seed=31, shuffle_seed=500031, formal_updates_per_stage=10000,
    short_actual_updates_per_stage=2, short_original_schedule_max=10000,
    VAE_AdamW=dict(lr=.003, betas=[.9,.95], eps=1e-8, weight_decay=0),
    Muon='Mature BF16 NS5/mu.95/Nesterov; real stage Adam-displacement kappa <=1%, max2 doublings/12 bisections',
    matrix_elements=589824, auxiliary_elements=29952, parameters=619776,
    fixed_train_readout='Original common first8x256 train inputs; not holdout',
    GPU_hours=dict(muon=4, control=1.5, d16=4), automatic_recovery_or_full9=False, automatic_retry=False)


def check_pins(values):
    for path, value in values.items():
        require(sha(path) == value, 'Bound implementation changed: ' + path)


def pins():
    paths = set(HERE.glob('*.py')) | {MATURE_OPTIMIZER}
    paths.update(REPO / 'experiments/adaptive_class_evaluation_20261030' / name for name in
        ('bindings.py','loading.py','full8.py','evaluate.py','summarize.py'))
    paths.update(REPO / 'experiments/adaptive_class_20261030' / name for name in ('continuation.py','initialization.py','run_pipeline.py'))
    paths.add(REPO / 'experiments/down_geometry_baseline_cmcc_20261103/validation.py')
    return {**original.source_pins(), **{str(path): sha(path) for path in sorted(paths)}}


def event(directory):
    meta = read(directory / checkpoint.META_FILE)
    require(meta['format'] == checkpoint.FORMAT and meta['schema_version'] == 2
        and sha(directory / checkpoint.STATE_FILE) == meta['state_sha256'], 'Real C0 source weight/config identity changed.')
    return dict(path=str(directory), checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=sha(directory / checkpoint.META_FILE), state_sha256=meta['state_sha256'])


def sources():
    common = original.bind(COMMON)
    report = read(C0 / 'report.json')
    require(report['status'] == 'completed_fresh_C0_fixed2_pipeline' and report['exit_code'] == 0
        and report['source_binding'] == common, 'Use the complete actual original C0 chain.')
    parent, control = event(PARENT), event(CONTROL)
    require(parent['checkpoint_id'] == report['stages'][0]['after_recovery_checkpoint_id'] == CONTRACT['source_after_gate']
        and control['checkpoint_id'] == report['stages'][1]['round_base_checkpoint_id'] == CONTRACT['raw_control'],
        'Use the real after-gate and original raw-up, not a later or other-parent donor.')
    for directory, counts in ((PARENT,[144,36,72]), (CONTROL,[180,36,36])):
        meta = read(directory / checkpoint.META_FILE)
        require([len(meta[key]) for key in ('native_modules','compact_modules','pending_dense_modules')] == counts,
            'Preserve the actual partial native/compact/future-dense topology.')
    fit = report['stages'][0]['next_live_category_fit']
    require(fit['category'] == 'up_proj' and fit['source_target_digest'] == '171c0538a534bea092270d93a90ffc4927140a0611b4970790ea4557f037d3d1'
        and fit['fresh_VAE_initialization_seed'] == 31 and fit['shuffle_seed'] == 500031
        and fit['actual_updates_per_stage'] == 10000 and fit['vector_batch_per_layer'] == 8192,
        'The actual original up fitting recipe or live source changed.')
    return dict(common=common, after_gate=parent, raw_control=control, original_up_fit=fit,
        c0_report=str(C0 / 'report.json'), c0_report_sha256=sha(C0 / 'report.json'))


def recipe(arm, output):
    cat,hf,training,vae,cfgs,runtime = original.recipe(output, original.bind(COMMON))
    baseline = deepcopy(runtime['runtime_identity'])
    settings = dict(dim=32,bits=32,batch=8192) if arm == 'control' else ARMS[arm]
    cfgs = dict(cfgs)
    cfgs['up_proj'] = replace(cfgs['up_proj'], codebook_dim=settings['dim'], codebook_bits=settings['bits'])
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=recovery.CATEGORIES,
        target_layers='all', skip_layers=(), transpose_modules=cat.transpose_modules.split(','))
    identity['experiment_runtime'] = deepcopy(baseline['experiment_runtime'])
    identity['experiment_runtime']['category_vector_batches']['up_proj'] = settings['batch']
    expected = deepcopy(baseline)
    expected['resolved_category_runtime']['up_proj'].update(codebook_dim=settings['dim'],codebook_bits=settings['bits'])
    expected['experiment_runtime']['category_vector_batches']['up_proj'] = settings['batch']
    require(identity == expected, 'Change only the selected up D/K/B, with the optimizer factory separate.')
    cfg = cfgs['up_proj']
    require(cfg.steps == 10000 and cfg.residual_stages == 2 and cfg.base_ch == 128
        and cfg.decoder_type == 'symmetric' and cfg.num_res_blocks == 0
        and cfg.channel_protect_count == 64 and cfg.weight_rotation_block_size == 32
        and 'up_proj' not in cat.transpose_modules.split(',') and cat.seed == training.seed == 31
        and vae.optimizer == 'adamw' and vae.beta1 == .9 and vae.beta2 == .95 and vae.lr == .003
        and vae.weight_decay == 0 and vae.vae_weight_dtype == 'bf16'
        and settings['dim'] * settings['batch'] == 262144, 'Keep the actual original symmetric up/BSQ/Adam/precision contract.')
    return (cat,hf,training,vae,cfgs,runtime), identity, baseline


def stage_args(parsed):
    args, cfg = deepcopy(parsed[3]), parsed[4]['up_proj']
    for key,value in dict(parallel_layers=36,residual_stages=2,codebook_dim=cfg.codebook_dim,
        codebook_bits=cfg.codebook_bits,base_ch=cfg.base_ch,num_res_blocks=cfg.num_res_blocks,
        decoder_type=cfg.decoder_type,decoder_base_ch=cfg.decoder_base_ch,
        decoder_num_res_blocks=cfg.decoder_num_res_blocks,norm_type=cfg.norm_type,
        activation_type=cfg.activation_type,recon_loss_type=cfg.recon_loss_type).items():
        setattr(args,key,value)
    recovery.native.apply_autoencoder_arch_defaults(args)
    return args


def cpu_environment():
    require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Use caller bitvae and hidden CUDA for CPU stages.')
