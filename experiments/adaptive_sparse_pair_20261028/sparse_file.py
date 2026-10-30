"""One charged16KiB header, all fixed2 state, and the native six-tensor sparse codec."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import torch

from experiments.adaptive_vector_rvq import codec, common as c

MAGIC = b'AVRQSP1\0'
FIELDS = ('active_block_ids', 'block_ptr', 'local_indices', 'qvalues', 'scales', 'zero_points')
SPEC = ('format', 'index_bits', 'value_bits', 'block_rows', 'block_cols')
DTYPES = {**codec.DTYPES, str(torch.uint16): torch.uint16, str(torch.int32): torch.int32}


def inventory(source, payload):
    return {**codec.tensor_inventory(source),
        **{'sparse/' + name: payload['sparse_residual_' + name] for name in FIELDS}}


def save(source, payload, path):
    path = Path(path)
    c.require(source['mode'] == 'fixed2' and len(source['stages']) == 2,
        'Persist the full actual fixed2 baseline, without charging any old state away.')
    tensors = inventory(source, payload)
    table, offset = {}, c.HEADER_BYTES
    for name, tensor in tensors.items():
        size = c.compact(tensor)
        table[name] = dict(dtype=str(tensor.dtype), shape=list(tensor.shape), offset=offset, bytes=size)
        offset += size
    meta = dict(schema='adaptive_sparse_pair_fixed2_native_sparse_v1', mode='fixed2',
        n_vectors=source['n_vectors'], geometry=source['geometry'], tensors=table,
        stages=[{name: stage[name] for name in ('decoder_spec', 'normalization')} for stage in source['stages']],
        sparse={name: payload['sparse_residual_' + name] for name in SPEC},
        accumulation='BF16 saved stage decode/add, inverse rotation/protection; native sparse decode to BF16 then index_put accumulate=True')
    header = json.dumps(meta, separators=(',', ':'), allow_nan=False).encode('utf-8')
    c.require(len(header) + 12 <= c.HEADER_BYTES, 'All actual metadata must fit the charged16KiB header.')
    with path.open('xb') as stream:
        stream.write(struct.pack('<8sI', MAGIC, len(header)))
        stream.write(header)
        stream.write(bytes(c.HEADER_BYTES - 12 - len(header)))
        for tensor in tensors.values():
            stream.write(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    sparse_bytes = sum(c.compact(payload['sparse_residual_' + name]) for name in FIELDS)
    c.require(path.stat().st_size == offset == codec.ledger(source)['total_bytes'] + sparse_bytes,
        'The actual sparse file must include its entire base and native six-tensor payload.')
    return dict(total_bytes=offset, header_bytes=c.HEADER_BYTES,
        fixed2_payload_bytes=codec.ledger(source)['all_tensor_bytes'], sparse_payload_bytes=sparse_bytes,
        tensors=table)


def load(path):
    path = Path(path)
    with path.open('rb') as stream:
        magic, size = struct.unpack('<8sI', stream.read(12))
        c.require(magic == MAGIC and size <= c.HEADER_BYTES - 12, 'Invalid pair sparse file header.')
        meta = json.loads(stream.read(size))
        c.require(meta['schema'] == 'adaptive_sparse_pair_fixed2_native_sparse_v1'
            and meta['mode'] == 'fixed2' and len(meta['stages']) == 2
            and not any(stream.read(c.HEADER_BYTES - 12 - size)), 'Invalid pair schema or charged padding.')
        tensors, offset = {}, c.HEADER_BYTES
        for name, row in meta['tensors'].items():
            c.require(row['offset'] == offset and row['dtype'] in DTYPES, 'Invalid contiguous tensor table.')
            raw = bytearray(stream.read(row['bytes']))
            c.require(len(raw) == row['bytes'], 'Truncated actual pair sparse payload.')
            dtype = DTYPES[row['dtype']]
            tensor = torch.frombuffer(raw, dtype=dtype).clone().reshape(row['shape']) if raw else torch.empty(row['shape'], dtype=dtype)
            c.require(c.compact(tensor) == row['bytes'], 'Loaded tensor shape/bytes changed.')
            tensors[name], offset = tensor, offset + row['bytes']
        c.require(not stream.read(1) and offset == path.stat().st_size, 'Uncharged trailing bytes.')
    stages = []
    for index, stage in enumerate(meta['stages']):
        prefix = f'stage{index}/decoder/'
        stages.append({**stage, 'codes': tensors[f'stage{index}/codes'],
            'decoder_state': {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}})
    source = dict(mode='fixed2', n_vectors=meta['n_vectors'], geometry=meta['geometry'], stages=stages,
        fixed={name[len('fixed/'):]: value for name, value in tensors.items() if name.startswith('fixed/')})
    payload = {'sparse_residual_' + name: value for name, value in meta['sparse'].items()}
    payload.update({'sparse_residual_' + name: tensors['sparse/' + name] for name in FIELDS})
    c.require(set(tensors) == set(inventory(source, payload)), 'Unexpected or omitted persistent tensor fields.')
    return source, payload
