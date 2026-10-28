"""Stage-major short codes, tightly packed two-bit lengths, and a standalone codec."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

from .common import BITS, CHUNK, DIM, HEADER_BYTES, compact, require
from litebsq.autoencoder import Decoder
from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool

MAGIC = b'AVRVQ1\0\0'
DTYPES = {str(value): value for value in (torch.float32, torch.float64, torch.bfloat16, torch.uint8, torch.int8, torch.int64)}


def pack_lengths(lengths):
    require(lengths.device.type == 'cpu' and lengths.ndim == 1 and bool(((lengths >= 1) & (lengths <= 4)).all()), 'Lengths must be in1..4.')
    values = lengths.to(torch.uint8) - 1
    padded = torch.zeros(((values.numel() + 3) // 4) * 4, dtype=torch.uint8)
    padded[:values.numel()] = values
    return (padded.reshape(-1, 4).to(torch.int64) << torch.tensor([0, 2, 4, 6])).sum(-1).to(torch.uint8).contiguous()


def unpack_lengths(packed, n):
    require(packed.dtype == torch.uint8 and packed.ndim == 1 and packed.numel() == (n + 3) // 4, 'Invalid tight count stream.')
    values = ((packed[:, None].to(torch.int64) >> torch.tensor([0, 2, 4, 6])) & 3).reshape(-1)
    require(not bool(values[n:].any()), 'Nonzero padding in the two-bit count stream.')
    return (values[:n] + 1).to(torch.uint8).contiguous()


def decoder_spec(decoder):
    return {key: getattr(decoder, key) for key in ('in_dim', 'out_dim', 'hidden_dim', 'num_res_blocks', 'norm_type', 'activation_type', 'decoder_type')}


def restore_decoder(stage, device='cpu'):
    state = stage['decoder_state']
    dtype = next(iter(state.values())).dtype
    require(dtype in (torch.float32, torch.bfloat16) and all(value.dtype == dtype for value in state.values()), 'Uniform FP32-contract or BF16-deployment decoder required.')
    model = Decoder(**stage['decoder_spec'], use_checkpoint=False, num_models=1).extract_single(0).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    model._q_scale_fused = True
    return model.eval().requires_grad_(False)


@torch.no_grad()
def add_stage(prefix, stage, active, device='cpu', chunk=CHUNK):
    require(active.numel() == stage['codes'].shape[0] and stage['codes'].dtype == torch.uint8
        and tuple(stage['codes'].shape) == (active.numel(), BITS // 8), 'Save exactly the active stage rows, without padded vector tails.')
    decoder = restore_decoder(stage, device)
    dtype = next(decoder.parameters()).dtype
    require(prefix.dtype == dtype and prefix.shape[1] == DIM, 'Use the same deployment accumulation precision in both arms.')
    for start in range(0, active.numel(), chunk):
        ids = active[start:start + chunk]
        codes = stage['codes'][start:start + ids.numel()].to(device)
        bits = unpack_uint8_tensor_to_bool(codes, logical_shape=(ids.numel(), BITS))
        values = decoder(bits.to(dtype=dtype).unsqueeze(1)).squeeze(1).cpu()
        require(bool(torch.isfinite(values).all()), 'Nonfinite deployed stage reconstruction.')
        # Disabled vectors never enter this decoder, including its fused mean/bias.
        prefix.index_copy_(0, ids, prefix.index_select(0, ids) + values)
    return prefix


def lengths(bundle):
    n = bundle['n_vectors']
    if bundle['mode'] == 'fixed2':
        require('lengths' not in bundle and len(bundle['stages']) == 2, 'Fixed2 has implicit length2 and no count stream.')
        return torch.full((n,), 2, dtype=torch.uint8)
    require(bundle['mode'] == 'adaptive' and len(bundle['stages']) == 4, 'Adaptive deploys four available residual stages.')
    return unpack_lengths(bundle['lengths'], n)


@torch.no_grad()
def decode(bundle, device='cpu', chunk=CHUNK):
    sizes = lengths(bundle)
    dtype = next(iter(bundle['stages'][0]['decoder_state'].values())).dtype
    output = torch.zeros((bundle['n_vectors'], DIM), dtype=dtype)
    for index, stage in enumerate(bundle['stages']):
        active = torch.nonzero(sizes >= index + 1, as_tuple=False).flatten()
        add_stage(output, stage, active, device, chunk)
    return output


def tensor_inventory(bundle):
    tensors = {f'fixed/{name}': value for name, value in bundle['fixed'].items()}
    if bundle['mode'] == 'adaptive':
        tensors['lengths'] = bundle['lengths']
    for index, stage in enumerate(bundle['stages']):
        tensors[f'stage{index}/codes'] = stage['codes']
        tensors.update({f'stage{index}/decoder/{name}': value for name, value in stage['decoder_state'].items()})
    return tensors


def ledger(bundle):
    tensors = tensor_inventory(bundle)
    sizes = {name: compact(value) for name, value in tensors.items()}
    counts = lengths(bundle)
    require(sum(stage['codes'].shape[0] for stage in bundle['stages']) == int(counts.long().sum()), 'Stored rows must equal the actual lengths sum.')
    decoder = sum(size for name, size in sizes.items() if '/decoder/' in name)
    return dict(total_bytes=HEADER_BYTES + sum(sizes.values()), header_bytes=HEADER_BYTES,
        code_bytes=sum(size for name, size in sizes.items() if name.endswith('/codes')),
        count_bytes=sizes.get('lengths', 0), decoder_bytes=decoder,
        fixed_bytes=sum(size for name, size in sizes.items() if name.startswith('fixed/')),
        all_tensor_bytes=sum(sizes.values()), container='actual standalone binary including fixed header and all payload bytes')


def save(bundle, path):
    path = Path(path)
    require(not path.exists(), 'Never overwrite an actual codec.')
    tensors = tensor_inventory(bundle)
    records, offset = {}, HEADER_BYTES
    for name, value in tensors.items():
        size = compact(value)
        records[name] = dict(dtype=str(value.dtype), shape=list(value.shape), offset=offset, bytes=size)
        offset += size
    metadata = {key: bundle[key] for key in ('mode', 'n_vectors', 'geometry')}
    metadata.update(schema='adaptive_vector_rvq_codec_v1', stages=[{key: stage[key] for key in ('decoder_spec', 'normalization')} for stage in bundle['stages']],
        tensors=records, accumulation='stage order; each saved decoder and each addition use the persisted decoder dtype', decoder_input='bool01; q_scale and stage normalization already fused')
    header = json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode('utf-8')
    require(len(header) + 12 <= HEADER_BYTES, 'Codec metadata exceeds the fixed charged header.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(struct.pack('<8sI', MAGIC, len(header)))
        stream.write(header)
        stream.write(bytes(HEADER_BYTES - 12 - len(header)))
        for value in tensors.values():
            stream.write(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    require(path.stat().st_size == ledger(bundle)['total_bytes'], 'Actual standalone bytes disagree with the complete ledger.')
    return ledger(bundle)


def load(path):
    path = Path(path)
    with path.open('rb') as stream:
        magic, size = struct.unpack('<8sI', stream.read(12))
        require(magic == MAGIC and size <= HEADER_BYTES - 12, 'Invalid codec header.')
        meta = json.loads(stream.read(size))
        require(meta['schema'] == 'adaptive_vector_rvq_codec_v1' and not any(stream.read(HEADER_BYTES - 12 - size)), 'Invalid header schema/padding.')
        tensors, expected = {}, HEADER_BYTES
        for name, item in meta['tensors'].items():
            require(item['offset'] == expected and item['dtype'] in DTYPES, 'Invalid/noncontiguous tensor table.')
            raw = bytearray(stream.read(item['bytes']))
            require(len(raw) == item['bytes'], 'Truncated actual codec payload.')
            dtype = DTYPES[item['dtype']]
            value = torch.frombuffer(raw, dtype=dtype).clone().reshape(item['shape']) if raw else torch.empty(item['shape'], dtype=dtype)
            require(compact(value) == item['bytes'], 'Tensor byte count disagrees with its geometry.')
            tensors[name] = value
            expected += item['bytes']
        require(not stream.read(1) and expected == path.stat().st_size, 'Unexpected uncharged trailing codec bytes.')
    stages = []
    for index, record in enumerate(meta['stages']):
        prefix = f'stage{index}/decoder/'
        stages.append({**record, 'codes': tensors[f'stage{index}/codes'],
            'decoder_state': {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}})
    bundle = dict(mode=meta['mode'], n_vectors=meta['n_vectors'], geometry=meta['geometry'], stages=stages,
        fixed={name[len('fixed/'):]: value for name, value in tensors.items() if name.startswith('fixed/')})
    if bundle['mode'] == 'adaptive':
        bundle['lengths'] = tensors['lengths']
    require(ledger(bundle)['total_bytes'] == path.stat().st_size, 'Fresh codec full ledger changed.')
    return bundle
