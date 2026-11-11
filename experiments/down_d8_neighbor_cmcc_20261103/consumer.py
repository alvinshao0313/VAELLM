"""Bind the copied original fused consumer with only its two OUT tiles widened."""
import hashlib
import importlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ORIGINAL=REPO/'result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot/litebsq/fused_multistage_decoder.py'
EFFECTIVE=HERE/'effective_fused_multistage_decoder.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return dict(schema='D8_masked_fused_OUT_tile_v1',base_kernel_path=str(ORIGINAL),base_kernel_sha256=sha(ORIGINAL),
        effective_kernel_path=str(EFFECTIVE),effective_kernel_sha256=sha(EFFECTIVE),
        launch_only_change='Two BLOCK_OUT=max(16,min(next_pow2(OUT),64)); logical OUT and all load/store masks, strides and arithmetic unchanged.',
        registration='Process-local registered N module public decode functions plus vae_linear imported functions refer to this effective file; N module identity/namespace remains intact.',
        public_snapshot_modified=False,scope='Isolated D8 candidate consumers, not public entry support or a decoder dimension change.')


def install():
    original=ORIGINAL.read_text(encoding='utf-8')
    before='BLOCK_OUT = min(_next_pow2(OUT), 64)'
    after='BLOCK_OUT = max(16, min(_next_pow2(OUT), 64))'
    if original.count(before)!=2 or EFFECTIVE.read_text(encoding='utf-8')!=original.replace(before,after):
        raise ValueError('Effective fused source must differ only at the two original launcher OUT tile expressions.')
    effective=importlib.import_module('experiments.down_d8_neighbor_cmcc_20261103.effective_fused_multistage_decoder')
    registered=importlib.import_module('litebsq.fused_multistage_decoder')
    linear=importlib.import_module('litebsq.vae_linear')
    for name in ('fused_decode_packed_symmetric_decoder','fused_decode_batched_same_shape_decoders','packed_symmetric_decoder_supports_fuse'):
        setattr(registered,name,getattr(effective,name))
    for name in ('_TRITON_AVAILABLE','fused_decode_packed_symmetric_decoder','packed_symmetric_decoder_supports_fuse'):
        setattr(linear,name,getattr(effective,name))
    if importlib.import_module('litebsq.fused_multistage_decoder').fused_decode_batched_same_shape_decoders is not effective.fused_decode_batched_same_shape_decoders:
        raise ValueError('The effective group-prewarm import did not bind to the isolated masked kernel.')
    return effective


def require_meta(meta):
    expected=contract()
    if meta['runtime'].get('effective_consumer')!=expected or meta['provenance'].get('effective_consumer')!=expected:
        raise ValueError('This D8 checkpoint must explicitly bind its isolated effective consumer in runtime and provenance.')
    install()
    return expected
