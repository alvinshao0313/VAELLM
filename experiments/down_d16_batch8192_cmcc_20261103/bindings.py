"""One fixed D16/K24 fit: B16384 -> B8192, with its genuine live parent."""
from copy import deepcopy
from pathlib import Path

from experiments.down_geometry_baseline_cmcc_20261103 import bindings as frozen
from experiments.down_geometry_baseline_cmcc_20261103.bindings import (
    ARTIFACTS, TOPIC, REPO, PYTHON, PARENT, CONTROL, C0, SUPPORT, SNAPSHOT,
    c, original, checkpoint, recovery, torch, read, write, sha, require,
    sources, check_pins, cpu_environment, restore_cat_runtime_state,
)

HERE = Path(__file__).resolve().parent
OUTPUT, RESULT = ARTIFACTS / HERE.name, TOPIC / HERE.name
ARM = 'batch8192'
ARMS = {ARM: dict(gpu=4, dim=16, bits=24, batch=8192, transpose=True)}
CONTROL_REPORT = frozen.OUTPUT / 'd16/formal/report.json'
CONTROL_FULL8 = frozen.RESULT / 'allocator_repair/d16/raw_full8/full8_summary.json'
CONTROL_READOUT = frozen.RESULT / 'allocator_repair/d16/raw_readout.json'
ORIGINAL_PREPARED = frozen.RESULT / 'prepared.json'
INPUT_SHA = 'c36ee8ce92ae468aa19634f7e3286e179a523e33a5f06b3fc502e29d7f4aee98'


def pins():
    paths = [*HERE.glob('*.py'), HERE / 'README.md', HERE / 'run.sh', CONTROL_REPORT,
        CONTROL_FULL8, CONTROL_READOUT, ORIGINAL_PREPARED]
    return {**frozen.pins(), **{str(path): sha(path) for path in paths}}


def recipe(arm, output):
    require(arm == ARM, 'Only the reviewed D16/K24/B8192 point is supported.')
    parsed, identity, baseline = frozen.recipe('d16', output)
    expected = deepcopy(identity)
    identity['experiment_runtime']['category_vector_batches']['down_proj'] = 8192
    expected['experiment_runtime']['category_vector_batches']['down_proj'] = 8192
    require(identity == expected and parsed[-2]['down_proj'].steps == 10000,
        'Keep the original complete D16 recipe and 10K schedule.')
    return parsed, identity, baseline


def initialization_reference():
    report = read(CONTROL_REPORT)
    require(report['status'] == 'completed_formal_raw_down' and report['exit_code'] == 0
        and report['raw_checkpoint_id'] == '02292fb6-5041-4f0d-bbb3-9827ec5f3a1f'
        and report['settings'] == dict(gpu=0, dim=16, bits=24, batch=16384, transpose=True),
        'Use the completed original D16 fit as the batch-only control.')
    rows = report['fit']['actual_optimizer_initializations']
    require(len(rows) == 2 and all(row['parameter_elements'] == 397728
        and row['parameter_dtypes'] == ['torch.bfloat16'] for row in rows),
        'Bind both actual original D16 stage initializations.')
    return rows
