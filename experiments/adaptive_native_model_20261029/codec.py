"""Read the real historical compact input using its decoder-derived code shape."""
import json
from pathlib import Path
import struct
import zlib

import torch

from experiments.adaptive_vector_rvq import codec as old_codec
from experiments.adaptive_vector_rvq.common import HEADER_BYTES, compact, require
from experiments.adaptive_vector_rvq.length_codec_probe import HEADER, MAGIC as LENGTH_MAGIC, CODEC_ZLIB
from .spec import resolve


def decode_lengths(frame, n):
    require(frame.dtype == torch.uint8 and frame.ndim == 1, 'The persisted length frame must be a byte vector.')
    raw_frame = frame.cpu().numpy().tobytes()
    require(len(raw_frame) >= HEADER.size, 'Truncated route frame.')
    magic, kind, recorded_n, raw_size, checksum = HEADER.unpack(raw_frame[:HEADER.size])
    require(magic == LENGTH_MAGIC and kind == CODEC_ZLIB and recorded_n == n, 'Actual route frame identity disagrees.')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(raw_frame[HEADER.size:]) + decoder.flush()
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
            and len(raw) == raw_size == (n + 3) // 4 and zlib.crc32(raw) == checksum,
            'Invalid route frame length/checksum or trailing bytes.')
    return old_codec.unpack_lengths(torch.frombuffer(bytearray(raw), dtype=torch.uint8), n)


def load_input(path):
    path = Path(path)
    with path.open('rb') as stream:
        magic, size = struct.unpack('<8sI', stream.read(12))
        require(magic == b'AVRQL01\0' and size <= HEADER_BYTES - 12, 'Expected the actual standalone dynamic3 input.')
        meta = json.loads(stream.read(size))
        require(meta['schema'] == 'adaptive_vector_rvq_zlib_l0_three_stage_v1'
                and not any(stream.read(HEADER_BYTES - 12 - size)), 'Invalid actual compact schema/header padding.')
        tensors, offset = {}, HEADER_BYTES
        for name, item in meta['tensors'].items():
            require(item['offset'] == offset and item['dtype'] in old_codec.DTYPES, 'Noncontiguous compact tensor table.')
            raw = bytearray(stream.read(item['bytes']))
            require(len(raw) == item['bytes'], 'Truncated compact payload.')
            dtype = old_codec.DTYPES[item['dtype']]
            tensor = torch.frombuffer(raw, dtype=dtype).clone().reshape(item['shape']) if raw else torch.empty(item['shape'], dtype=dtype)
            require(compact(tensor) == item['bytes'], 'Actual tensor shape/bytes disagree.')
            tensors[name], offset = tensor, offset + item['bytes']
        require(not stream.read(1) and offset == path.stat().st_size, 'Uncharged trailing compact bytes.')
    stages = []
    for index, record in enumerate(meta['stages']):
        prefix = f'stage{index}/decoder/'
        stages.append({**record, 'codes': tensors[f'stage{index}/codes'],
                       'decoder_state': {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}})
    bundle = dict(n_vectors=meta['n_vectors'], geometry=meta['geometry'], stages=stages,
                  fixed={name[len('fixed/'):]: value for name, value in tensors.items() if name.startswith('fixed/')},
                  length_frame=tensors['length_frame'])
    expected = {'length_frame', *(f'fixed/{name}' for name in bundle['fixed'])}
    for index, stage in enumerate(stages):
        expected.add(f'stage{index}/codes')
        expected.update(f'stage{index}/decoder/{name}' for name in stage['decoder_state'])
    require(set(tensors) == expected, 'Unexpected or omitted compact persistent fields.')
    lengths = decode_lengths(bundle['length_frame'], bundle['n_vectors'])
    return bundle, lengths, resolve(bundle, lengths)
