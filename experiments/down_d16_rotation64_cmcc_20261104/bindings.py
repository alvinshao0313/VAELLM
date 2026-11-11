"""Bind the genuine D16 control and change only its down rotation block."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from experiments.down_geometry_baseline_cmcc_20261103 import bindings as frozen
from experiments.down_geometry_baseline_cmcc_20261103.bindings import (
    ARTIFACTS, TOPIC, REPO, PYTHON, PARENT, C0, SUPPORT, SNAPSHOT, c, original,
    checkpoint, recovery, torch, read, write, sha, require, check_pins,
    cpu_environment, restore_cat_runtime_state, build_cat_cross_category_runtime_identity,
)

HERE = Path(__file__).resolve().parent
OUTPUT, RESULT = ARTIFACTS / HERE.name, TOPIC / HERE.name
ARM = 'block64'
ARMS = {ARM: dict(gpu=4, dim=16, bits=24, batch=16384, transpose=True)}
CONTROL = frozen.OUTPUT / 'd16/formal/raw_model'
CONTROL_REPORT = frozen.OUTPUT / 'd16/formal/report.json'
CONTROL_FULL8 = frozen.RESULT / 'allocator_repair/d16/raw_full8/full8_summary.json'
CONTROL_READOUT = frozen.RESULT / 'allocator_repair/d16/raw_readout.json'
ORIGINAL_PREPARED = frozen.RESULT / 'prepared.json'
INPUT_SHA = 'c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98'


def sources():
    source = frozen.sources()
    reference = read(CONTROL_REPORT)
    event = frozen.event(CONTROL)
    require(reference['status'] == 'completed_formal_raw_down' and reference['exit_code'] == 0
        and reference['raw_checkpoint_id'] == event['checkpoint_id'] == '02292fb6-5041-4f0d-bbb3-9827ec5f3a1f'
        and reference['source_binding'] == source, 'Bind the completed D16 control from the same after-up parent.')
    summary, readout = read(CONTROL_FULL8), read(CONTROL_READOUT)
    require(summary['checkpoint_id'] == readout['checkpoint_id'] == event['checkpoint_id']
        and summary['leaf_count'] == 64 and summary['sample_count'] == 24742
        and readout['input_ids_sha256'] == INPUT_SHA, 'Reuse only the complete existing D16 full8/train readout.')
    return dict(source, raw_control=event, comparison_control_fit_report=str(CONTROL_REPORT),
        comparison_control_fit_report_sha256=sha(CONTROL_REPORT),
        comparison='Same live D16/K24/B16384; only down two-sided Hadamard block32→64')


def pins():
    paths = [*HERE.glob('*.py'), CONTROL_REPORT, CONTROL_FULL8, CONTROL_READOUT, ORIGINAL_PREPARED]
    return {**frozen.pins(), **{str(path): sha(path) for path in paths}}


def recipe(arm, output):
    require(arm == ARM, 'Only the registered block64 factor is enabled.')
    parsed, old_identity, baseline = frozen.recipe('d16', output)
    cat, hf, training, vae, cfgs, runtime = parsed
    cfgs = dict(cfgs)
    cfgs['down_proj'] = replace(cfgs['down_proj'], weight_rotation_block_size=64)
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=recovery.CATEGORIES,
        target_layers='all', skip_layers=(), transpose_modules=cat.transpose_modules.split(','))
    identity['experiment_runtime'] = deepcopy(old_identity['experiment_runtime'])
    expected = deepcopy(old_identity)
    expected['resolved_category_runtime']['down_proj']['weight_rotation_block_size'] = 64
    require(identity == expected, 'Resolved recipe changed beyond down block32→64.')
    cfg = cfgs['down_proj']
    require(cfg.steps == 10000 and cfg.codebook_dim == 16 and cfg.codebook_bits == 24
        and cfg.residual_stages == 2 and cfg.channel_protect_count == 256
        and cfg.weight_rotation_block_size == 64, 'Preserve all actual D16 control geometry and exposure.')
    return (cat, hf, training, vae, cfgs, runtime), identity, baseline


def initialization_reference():
    rows = read(CONTROL_REPORT)['fit']['actual_optimizer_initializations']
    require(len(rows) == 2 and all(row['parameter_elements'] == 397728
        and row['parameter_dtypes'] == ['torch.bfloat16'] for row in rows),
        'Bind both real D16/B16384 stage initializations.')
    return rows
