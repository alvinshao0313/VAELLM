"""Persist real per-stage short codes and BF16 decoders; count payload separately."""
from io import BytesIO
from pathlib import Path

from . import bindings as b
from experiments.adaptive_native_model_20261029.codec import decode_lengths
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.native_prefix import inventory
from experiments.adaptive_native_model_20261029.spec import resolve

SCHEMA = 'down_tail23_short_v1'


def _tensor_leaves(value):
    if isinstance(value, b.torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)


def measure(bundle):
    lengths = decode_lengths(bundle['length_frame'], bundle['n_vectors'])
    resolved = resolve(bundle, lengths)
    tensors = inventory(bundle)
    b.c.require({id(value) for value in _tensor_leaves(bundle)} == {id(value) for value in tensors.values()},
        'Every persistent tensor must appear in the charged inventory, including future metadata fields.')
    pointers = [value.untyped_storage().data_ptr() for value in tensors.values() if value.numel()]
    b.c.require(len(set(pointers)) == len(pointers), 'Persist independent compact tensors so container aliasing cannot hide payload bytes.')
    sizes = {name: b.c.compact(value) for name, value in tensors.items()}
    counts = [int((lengths >= index + 1).sum()) for index in range(len(bundle['stages']))]
    bits = [int(stage['decoder_spec']['in_dim']) for stage in bundle['stages']]
    codes = sum(sizes[f'stage{index}/codes'] for index in range(len(bits)))
    logical = sum(k * rows for k, rows in zip(bits, counts, strict=True))
    payload = sum(sizes.values())
    return dict(payload_tensor_bytes=payload, tensor_bytes=sizes, stage_bits=bits, stage_active_rows=counts,
        code_bytes=codes, logical_code_bits=logical, packed_code_bits=codes * 8,
        code_padding_bits=codes * 8 - logical,
        decoder_bytes=sum(size for name, size in sizes.items() if '/decoder/' in name),
        fixed_bytes=sum(size for name, size in sizes.items() if name.startswith('fixed/')),
        length_frame_bytes=sizes['length_frame'], length_frame_header_bytes=29,
        resolved=resolved, cap_scope='All deployed tensor payload including actual route frame; serialization/metadata separate')


def serialize(bundle):
    measured = measure(bundle)
    stream = BytesIO()
    b.torch.save(dict(schema=SCHEMA, bundle=bundle, resolved=measured['resolved']), stream)
    return stream.getvalue()


def save(bundle, path):
    path = Path(path)
    measured = measure(bundle)
    data = serialize(bundle)
    with path.open('xb') as stream:
        stream.write(data)
    measured.update(actual_file_bytes=len(data), serialization_and_metadata_bytes=len(data) - measured['payload_tensor_bytes'])
    return measured


def load(path):
    saved = b.torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    b.c.require(saved['schema'] == SCHEMA, 'Require this isolated actual compact artifact.')
    bundle = saved['bundle']
    lengths = decode_lengths(bundle['length_frame'], bundle['n_vectors'])
    b.c.require(resolve(bundle, lengths) == saved['resolved'], 'Saved per-stage geometry disagrees with actual codes/routes.')
    module = CompactRVQLinear(bundle, lengths).eval().requires_grad_(False)
    return bundle, lengths, module
