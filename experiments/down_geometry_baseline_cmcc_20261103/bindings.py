"""Matched down-only forks from the completed C0 live after-up boundary."""
from copy import deepcopy
from dataclasses import asdict, replace
import os
from pathlib import Path
import sys

from experiments.fresh_fixed2_cmcc_20261030 import sources as original
from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC
from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_vector_rvq import common as c
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity

HERE = Path(__file__).resolve().parent
OUTPUT, RESULT = ARTIFACTS / HERE.name, TOPIC / HERE.name
COMMON = ARTIFACTS / 'fresh_common_cmcc_20261030/formal'
C0 = ARTIFACTS / 'fresh_fixed2_cmcc_20261030/single_20261030_v3_C0'
PARENT, CONTROL = C0 / '05_up_proj/after_recovery', C0 / '06_down_proj/round_base'
SUPPORT = ARTIFACTS / 'adaptive_class_cmcc_20261030/evaluation_support'
SNAPSHOT = original.common.s.SNAPSHOT
ARMS = dict(d16=dict(gpu=0, dim=16, bits=24, batch=16384, transpose=True),
    transpose_false=dict(gpu=2, dim=32, bits=48, batch=8192, transpose=False))
CONTROL_SETTINGS = dict(dim=32, bits=48, batch=8192, transpose=True)
read, write, sha, require, torch = c.read, c.write, c.sha, c.require, original.torch


def pins():
    files = set(HERE.glob('*.py'))
    files.update(REPO / 'experiments/adaptive_class_evaluation_20261030' / name for name in
        ('bindings.py', 'loading.py', 'full8.py', 'evaluate.py', 'summarize.py', 'predict.py', 'merge.py'))
    return {**original.source_pins(), **{str(path): sha(path) for path in sorted(files)}}


def check_pins(values):
    for path, digest in values.items():
        require(sha(path) == digest, 'Bound source changed: ' + path)


def event(directory):
    meta = read(directory / checkpoint.META_FILE)
    require(meta['format'] == checkpoint.FORMAT and meta['schema_version'] == 2
        and sha(directory / checkpoint.STATE_FILE) == meta['state_sha256'], 'Actual mixed source payload changed.')
    return dict(path=str(directory), checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=sha(directory / checkpoint.META_FILE), state_sha256=meta['state_sha256'])


def sources():
    common = original.bind(COMMON)
    report = read(C0 / 'report.json')
    require(report['status'] == 'completed_fresh_C0_fixed2_pipeline' and report['exit_code'] == 0
        and report['source_binding'] == common and len(report['stages']) == 3,
        'Require the completed genuine C0 chain from this common BF16 predecessor.')
    parent, control = event(PARENT), event(CONTROL)
    require(parent['checkpoint_id'] == report['stages'][1]['after_recovery_checkpoint_id']
        == '5a94d57f-d3b9-4c97-adef-a099d659eb6a'
        and control['checkpoint_id'] == report['stages'][2]['round_base_checkpoint_id']
        == 'ed507bcb-9b83-4e4c-9859-daaef44d80d2'
        and report['stages'][1]['recovery']['actual_steps'] == 500,
        'The actual C0 after-up/raw-down parent UUID differs.')
    meta = read(PARENT / checkpoint.META_FILE)
    require([len(meta[key]) for key in ('native_modules', 'compact_modules', 'pending_dense_modules')] == [180,36,36]
        and all(item['name'].endswith('.down_proj') for item in meta['pending_dense_modules']),
        'After-up requires native180/compact36/live-dense-down36.')
    fit = report['stages'][1]['next_live_category_fit']
    require(fit['category'] == 'down_proj' and fit['fresh_VAE_initialization_seed'] == 31
        and fit['shuffle_seed'] == 600031 and fit['actual_updates_per_stage'] == 10000
        and fit['vector_batch_per_layer'] == 8192 and len(fit['actual_optimizer_initializations']) == 2,
        'Original raw-down control did not use the fixed live-source fitting contract.')
    return dict(common=common, after_up=parent, raw_control=control,
        c0_report=str(C0 / 'report.json'), c0_report_sha256=sha(C0 / 'report.json'), original_down_fit=fit,
        fixed_predecessor='Original BF16 -> common after_o -> raw2 B gate -> C0 gate500 -> own up S2x10K/500')


def recipe(arm, output):
    binding = original.bind(COMMON)
    cat, hf, training, vae, cfgs, runtime = original.recipe(output, binding)
    baseline = deepcopy(runtime['runtime_identity'])
    settings = CONTROL_SETTINGS if arm == 'control' else ARMS[arm]
    cfgs = dict(cfgs)
    cfgs['down_proj'] = replace(cfgs['down_proj'], codebook_dim=settings['dim'], codebook_bits=settings['bits'])
    transpose = cat.transpose_modules.split(',')
    require('down_proj' in transpose, 'Original C0 must transpose down.')
    if not settings['transpose']:
        cat.transpose_modules = ','.join(name for name in transpose if name != 'down_proj')
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=recovery.CATEGORIES,
        target_layers='all', skip_layers=(), transpose_modules=cat.transpose_modules.split(','))
    identity['experiment_runtime'] = deepcopy(baseline['experiment_runtime'])
    identity['experiment_runtime']['category_vector_batches']['down_proj'] = settings['batch']
    expected=deepcopy(baseline)
    expected['resolved_category_runtime']['down_proj'].update(codebook_dim=settings['dim'],codebook_bits=settings['bits'])
    expected['experiment_runtime']['category_vector_batches']['down_proj']=settings['batch']
    if not settings['transpose']:
        expected['transpose_modules'].remove('down_proj')
    require(identity == expected,'Effective runtime changed beyond the approved down-only factor.')
    cfg = cfgs['down_proj']
    require(cfg.residual_stages == 2 and cfg.steps == 10000 and cfg.base_ch == 128
        and cfg.codebook_dim == settings['dim'] and cfg.codebook_bits == settings['bits']
        and cfg.channel_protect_count == 256 and cfg.weight_rotation_block_size == 32
        and cfg.num_res_blocks == 0 and cat.seed == training.seed == 31
        and settings['batch'] * settings['dim'] == 262144,
        'Change only the approved same-rate down geometry/batch or transpose factor.')
    return (cat,hf,training,vae,cfgs,runtime), identity, baseline


def cpu_environment():
    require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
        'Use caller-activated bitvae and hidden CUDA for the CPU task.')
