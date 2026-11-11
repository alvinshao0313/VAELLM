"""Immutable source/config pins for the full36, twelve-independent-group stage."""
from pathlib import Path
import os
import sys

from experiments.down_tail23_short_cmcc_20261104 import bindings as short
from experiments.adaptive_native_model_20261029 import checkpoint

c, torch = short.c, short.torch
HERE, REPO = Path(__file__).resolve().parent, short.REPO
OUTPUT = short.ARTIFACTS / HERE.name
RESULT = REPO / 'result/qwen3_8b_mmlu65_20261004' / HERE.name
PREPARED = RESULT / 'prepared.json'
SHORT_RUN = short.OUTPUT / 'gpu_pair_20261104_1044'
RAW, LIVE, RAW_ID, LIVE_ID = short.RAW, short.LIVE, short.RAW_ID, short.LIVE_ID
GROUPS = [[f'model.layers.{index}.mlp.down_proj' for index in range(start, start + 3)] for start in range(0, 36, 3)]
NAMES = [name for group in GROUPS for name in group]
CAP, N = short.CAP, short.N
SUPPORT = short.original.SUPPORT
SCHEMA = 'down_tail24_full_raw_stage_v1'


def cpu_environment():
    short.cpu_environment()


def gpu_environment():
    c.require(Path(sys.executable).resolve() == short.original.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True',
        'Use caller bitvae/GPU2 and the original expandable_segments:True allocator.')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def pins():
    paths = set(HERE.glob('*.py')) | set(short.HERE.glob('*.py'))
    paths.update((Path(short.original.__file__), Path(short.original.original.__file__)))
    for folder in ('adaptive_native_model_20261029', 'adaptive_class_evaluation_20261030'):
        paths.update((REPO / 'experiments' / folder).glob('*.py'))
    for folder in ('litebsq', 'rotation', 'train_utils', 'e2e_common'):
        paths.update((checkpoint.RUNTIME / folder).rglob('*.py'))
    paths.update((checkpoint.RUNTIME / 'tools/cat_eval.py', REPO / 'tools/cat_eval.py'))
    for relative in ('experiments/adaptive_grouped_fit_20261029/grouped.py',
        'experiments/adaptive_vector_rvq/common.py', 'experiments/adaptive_vector_rvq/codec.py',
        'experiments/adaptive_vector_rvq/length_codec_probe.py', 'experiments/cmcc_runtime_20261030/paths.py',
        'experiments/cmcc_runtime_20261030/adaptive_evaluation.py',
        'experiments/night_sequence_20261026/queue_wait.py'):
        paths.add(REPO / relative)
    return {str(path.resolve()): c.sha(path) for path in sorted(paths)}


def short_data_pins():
    paths = [SHORT_RUN / name for name in ('report.json', 'capture_inputs.pt', 'route_h.pt')]
    paths.extend(SHORT_RUN / f'model.layers.{layer}.mlp.down_proj.K24.pt' for layer in (0, 18, 35))
    return {str(path.resolve()): c.sha(path) for path in paths}


def native_reference():
    path = short.original.RESULT / 'allocator_repair/d16/raw_full8/full8_summary.json'
    record = c.read(path)
    c.require(record['checkpoint_id'] == RAW_ID and record['mean_percent'] == 67.01259272688102
        and record['leaf_count'] == 64 and record['sample_count'] == 24742,
        'Require the actual complete native D16 raw full8 reference.')
    return dict(path=str(path), sha256=c.sha(path), checkpoint_id=record['checkpoint_id'],
        mean_percent=record['mean_percent'], leaf_count=64, sample_count=24742)


def check_pins(values):
    for path, digest in values.items():
        c.require(c.sha(path) == digest, 'Bound source changed: ' + path)


def source(*, initial=False):
    if initial:
        raw, live, _three, manifest = short.sources()
    else:
        prepared = c.read(PREPARED)
        check_pins(prepared['source_sha256'])
        manifest = prepared['source_binding']
        for path, key in ((RAW, 'raw'), (LIVE, 'live')):
            c.require(c.sha(path / checkpoint.META_FILE) == manifest[key]['metadata_sha256']
                and c.stat(path / checkpoint.STATE_FILE) == manifest[key]['state'], 'The immutable bound source changed.')
        raw = torch.load(RAW / checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
        live = torch.load(LIVE / checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    meta = c.read(RAW / checkpoint.META_FILE)
    specs = sorted((item for item in meta['native_modules'] if item['name'].endswith('.down_proj')),
        key=lambda item: int(item['name'].split('.')[2]))
    c.require([item['name'] for item in specs] == NAMES, 'Require precisely36 numerically ordered down modules.')
    manifest = {**manifest, 'names': NAMES, 'groups': GROUPS,
        'layer_selection': 'All36 numeric layers in12 consecutive independent groups of3',
        'updates_per_layer': 10000, 'K': 24, 'fresh_init_and_shuffle_seed_per_layer': 2031}
    return raw, live, specs, manifest


def calibration():
    original = short.original
    bound = original.original.bind(original.COMMON)
    runtime = original.original.recipe(OUTPUT / 'recipe_identity_only', bound)[-1]
    cache = runtime['activation_runtime']['cache']
    c.require(cache['nsamples'] == len(cache['input_ids']) == 512 and cache['seqlen'] == 256,
        'Reuse the recorded original Alpaca train512x256 cache.')
    return torch.cat([value.reshape(1, 256) for value in cache['input_ids'][:16]], dim=0)
