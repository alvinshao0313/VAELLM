"""Same live D16 control; override only the current down VAE learning rate."""
from copy import deepcopy
from pathlib import Path

from experiments.down_geometry_baseline_cmcc_20261103 import bindings as frozen
from experiments.down_geometry_baseline_cmcc_20261103.bindings import (
    ARTIFACTS, TOPIC, REPO, PYTHON, PARENT, C0, SUPPORT, SNAPSHOT, c, original,
    checkpoint, recovery, torch, read, write, sha, require, check_pins,
    cpu_environment, restore_cat_runtime_state, build_cat_cross_category_runtime_identity,
)

HERE = Path(__file__).resolve().parent
OUTPUT, RESULT = ARTIFACTS / HERE.name, TOPIC / HERE.name
ARMS = dict(half=dict(gpu=2, dim=16, bits=24, batch=16384, transpose=True, lr=.0015),
    double=dict(gpu=4, dim=16, bits=24, batch=16384, transpose=True, lr=.006))
CONTROL = frozen.OUTPUT / 'd16/formal/raw_model'
CONTROL_REPORT = frozen.OUTPUT / 'd16/formal/report.json'
CONTROL_FULL8 = frozen.RESULT / 'allocator_repair/d16/raw_full8/full8_summary.json'
CONTROL_READOUT = frozen.RESULT / 'allocator_repair/d16/raw_readout.json'
ORIGINAL_PREPARED = frozen.RESULT / 'prepared.json'
INPUT_SHA = 'c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98'


def sources():
    source = frozen.sources()
    report = read(CONTROL_REPORT)
    event = frozen.event(CONTROL)
    require(report['status'] == 'completed_formal_raw_down' and report['exit_code'] == 0
        and report['source_binding'] == source and report['raw_checkpoint_id'] == event['checkpoint_id']
        == '02292fb6-5041-4f0d-bbb3-9827ec5f3a1f', 'Bind the actual same-parent D16 control.')
    summary, readout = read(CONTROL_FULL8), read(CONTROL_READOUT)
    require(summary['checkpoint_id'] == readout['checkpoint_id'] == event['checkpoint_id']
        and summary['leaf_count'] == 64 and summary['sample_count'] == 24742
        and readout['input_ids_sha256'] == INPUT_SHA, 'Reuse the completed D16 full8/readout without a new control run.')
    return dict(source, raw_control=event, comparison_D16_fit_report=str(CONTROL_REPORT),
        comparison_D16_fit_report_sha256=sha(CONTROL_REPORT),
        override_scope='Only current down VAE fit. Frozen historical prefix retains its original source identity/LR.')


def pins():
    paths = [*HERE.glob('*.py'), CONTROL_REPORT, CONTROL_FULL8, CONTROL_READOUT, ORIGINAL_PREPARED]
    return {**frozen.pins(), **{str(path): sha(path) for path in paths}}


def recipe(arm, output):
    require(arm in ARMS, 'Use only the two registered LR points.')
    parsed, old_identity, baseline = frozen.recipe('d16', output)
    cat, hf, training, vae, cfgs, runtime = parsed
    vae.lr = ARMS[arm]['lr']
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=recovery.CATEGORIES,
        target_layers='all', skip_layers=(), transpose_modules=cat.transpose_modules.split(','))
    identity['experiment_runtime'] = deepcopy(old_identity['experiment_runtime'])
    expected = deepcopy(old_identity)
    expected['vae_shared']['lr'] = ARMS[arm]['lr']
    require(identity == expected and cfgs['down_proj'].weight_rotation_block_size == 32,
        'Only the active down fit LR may change; keep the mature D16/B16384/block32 recipe.')
    return (cat, hf, training, vae, cfgs, runtime), identity, baseline


def initialization_reference():
    rows = read(CONTROL_REPORT)['fit']['actual_optimizer_initializations']
    require(len(rows) == 2 and all(row['parameter_elements'] == 397728
        and row['parameter_dtypes'] == ['torch.bfloat16'] for row in rows), 'Bind actual D16 stage initialization evidence.')
    return rows
