"""Bind the genuine raw D16 prefix and its unrecovered live after-up targets."""
from pathlib import Path
import os
import sys

from experiments.down_geometry_baseline_cmcc_20261103 import bindings as original
from experiments.adaptive_native_model_20261029 import native_prefix

c, torch = original.c, original.torch
HERE = Path(__file__).resolve().parent
REPO, ARTIFACTS = original.REPO, original.ARTIFACTS
RAW = ARTIFACTS / 'down_geometry_baseline_cmcc_20261103/d16/formal/raw_model'
LIVE = original.PARENT
OUTPUT = ARTIFACTS / HERE.name
RAW_ID = '02292fb6-5041-4f0d-bbb3-9827ec5f3a1f'
LIVE_ID = '5a94d57f-d3b9-4c97-adef-a099d659eb6a'
N, DIM, BATCH = 3080192, 16, 16384
BASE_PAYLOAD, EXTRA_PER_LAYER, CAP = 19570496, 2462620, 22033116


def cpu_environment():
    c.require(Path(sys.executable).resolve() == original.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
        'Use caller-activated bitvae and hidden CUDA; this preparation never authorizes a GPU.')


def sources():
    raw_meta, live_meta = (c.read(path / original.checkpoint.META_FILE) for path in (RAW, LIVE))
    c.require(raw_meta['checkpoint_id'] == RAW_ID and live_meta['checkpoint_id'] == LIVE_ID,
        'Require genuine raw D16 and the common live after-up parent, never recovered down weights.')
    for directory, meta in ((RAW, raw_meta), (LIVE, live_meta)):
        c.require(c.sha(directory / original.checkpoint.STATE_FILE) == meta['state_sha256'],
            'Actual source weight file disagrees with its bound metadata SHA256: ' + str(directory))
    specs = sorted((row for row in raw_meta['native_modules'] if row['name'].endswith('.down_proj')),
        key=lambda row: int(row['name'].split('.')[2]))
    c.require(len(specs) == 36 and len(live_meta['pending_dense_modules']) == 36,
        'The source requires all36 native down modules and all36 genuine pending live down weights.')
    selected = [specs[index] for index in (0, len(specs) // 2, len(specs) - 1)]
    for row in specs:
        c.require(row['codebook_dim'] == DIM and row['stage_codebook_dims'] == [DIM, DIM]
            and row['residual_stages'] == 2 and row['transpose'] and row['parallel_parts'] == 1
            and all(stage['in_dim'] == 24 and stage['out_dim'] == DIM for stage in row['stage_decoders']),
            'Only the actual D16/K24 two-stage transpose source is supported by this short card.')
    raw = torch.load(RAW / original.checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    live = torch.load(LIVE / original.checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    manifest = dict(raw=dict(path=str(RAW), checkpoint_id=RAW_ID,
        metadata_sha256=c.sha(RAW / original.checkpoint.META_FILE), state=c.stat(RAW / original.checkpoint.STATE_FILE),
        bound_state_sha256=raw_meta['state_sha256']), live=dict(path=str(LIVE), checkpoint_id=LIVE_ID,
        metadata_sha256=c.sha(LIVE / original.checkpoint.META_FILE), state=c.stat(LIVE / original.checkpoint.STATE_FILE),
        bound_state_sha256=live_meta['state_sha256']),
        layer_selection='Sort down modules by numeric layer; indices0,floor(L/2),L-1; short prototype only',
        names=[row['name'] for row in selected], vector_batch_per_layer=BATCH,
        original_payload_bytes=BASE_PAYLOAD, additional_bytes_per_layer=EXTRA_PER_LAYER, payload_cap_per_layer=CAP,
        full_projection_weights=6945767424, base_logical_bits=17349760512,
        total_new_integer_bytes=88654348, layers_cannot_borrow_budget=True)
    return raw, live, selected, manifest


def prefix(raw, spec, manifest):
    name = spec['name']
    fields = {key[len(name) + 1:]: value for key, value in raw.items() if key.startswith(name + '.')}
    bundle, lengths, resolved = native_prefix.from_native(spec, fields,
        dict(raw=manifest['raw'], source_module=name, source_relative_digest=c.digest(fields)))
    c.require(resolved['n_vectors'] == N and resolved['vector_dim'] == DIM,
        'Real protected/transpose geometry changed the approved vector count.')
    return bundle, lengths, resolved, fields


def arguments(bits, dtype='bf16'):
    c.require(bits in (8, 24), 'The short card contains only K8 and K24 third stages.')
    args = c.arguments(dtype)
    args.codebook_dim, args.codebook_bits = DIM, bits
    args.base_ch = args.encoder_base_ch = args.decoder_base_ch = 128
    args.num_res_blocks = args.encoder_num_res_blocks = args.decoder_num_res_blocks = 0
    args.decoder_type, args.norm_type, args.activation_type = 'symmetric', 'layer', 'swish'
    return args


def pins():
    from tools import cat_eval
    from e2e_common import proxy_trainables

    paths = set(HERE.glob('*.py'))
    paths.update((Path(original.__file__), Path(original.original.__file__)))
    paths.update(Path(module.__file__) for module in (cat_eval, proxy_trainables))
    for folder, names in (
        ('adaptive_native_model_20261029', ('native_prefix.py', 'spec.py', 'objective.py', 'consumer.py', 'codec.py', 'checkpoint.py')),
        ('adaptive_grouped_fit_20261029', ('grouped.py',)),
        ('adaptive_vector_rvq', ('common.py', 'codec.py', 'length_codec_probe.py')),
    ):
        paths.update(REPO / 'experiments' / folder / name for name in names)
    paths.update(Path(module.__file__) for module in sys.modules.values()
        if getattr(module, '__name__', '').split('.')[0] in ('litebsq', 'rotation', 'train_utils', 'e2e_common')
        and getattr(module, '__file__', None))
    return {str(path.resolve()): c.sha(path) for path in sorted(paths)}
