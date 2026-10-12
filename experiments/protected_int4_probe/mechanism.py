"""CPU packing, scale-axis and protected-subspace inverse checks; no model load."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from experiments.protected_int4_probe.codec import (decode_output_scale, encode_output_scale,
    fixed_signs, hadamard_last, pack_int4, require, rotate, unpack_int4)
from litebsq.protected_channel_quant import decode_protected_channel_weight, encode_protected_channel_weight


def checks():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Hide CUDA.')
    values = torch.cartesian_prod(torch.arange(-8, 8, dtype=torch.int8), torch.arange(-8, 8, dtype=torch.int8))
    packed = pack_int4(values)
    require(torch.equal(unpack_int4(packed, 2), values), 'All256 signed nibble pairs must round-trip exactly.')
    oracle = ((values[:, 0].to(torch.int16) & 15) + 16 * (values[:, 1].to(torch.int16) & 15)).to(torch.uint8)
    require(torch.equal(packed[:, 0], oracle), 'Byte layout differs from independent arithmetic oracle.')
    odd = torch.tensor([[-8, -1, 0, 1, 7], [7, 1, 0, -1, -8]], dtype=torch.int8)
    odd_packed = pack_int4(odd)
    require(torch.equal(unpack_int4(odd_packed, 5), odd) and bool((odd_packed[:, -1] >> 4 == 0).all()),
            'Odd row padding or signed extremes differ.')
    dirty = odd_packed.clone(); dirty[0, -1] |= 16
    rejected = False
    try:
        unpack_int4(dirty, 5)
    except ValueError:
        rejected = True
    require(rejected, 'Nonzero padding must fail instead of being silently ignored.')
    boundaries = torch.tensor([[0., 0., 0., 0.], [-8., -7., 0., 7.], [7., -7., 7., -7.]])
    for bits in (4, 8):
        payload = encode_output_scale(boundaries, bits=bits)
        restored = decode_output_scale(payload)
        require(payload.scales.dtype == torch.bfloat16 and payload.scales[0].item() == 1
                and bool((restored[0] == 0).all()) and bool(torch.isfinite(restored).all()), 'Zero/extreme row failed.')
    p8 = encode_output_scale(boundaries, bits=8)
    native_q, native_s = encode_protected_channel_weight(boundaries, quant_format='int8')
    native_w = decode_protected_channel_weight(native_q, native_s, quant_format='int8', dtype=torch.bfloat16)
    require(torch.equal(p8.values, native_q) and torch.equal(p8.scales, native_s)
            and torch.equal(decode_output_scale(p8), native_w), 'Output-row INT8 diagnostic differs from native row codec.')
    odd_payload = encode_output_scale(torch.zeros(2, 5), bits=4)
    require(odd_payload.storage()['padding_bits'] == 8 and odd_payload.storage()['total_bits'] == 80,
            'Odd padding and BF16 scale storage must both be charged.')
    generator = torch.Generator(device='cpu').manual_seed(2718)
    rotations = {}
    for size in (1, 2, 8, 64, 256):
        h = torch.ones((1, 1), dtype=torch.float64)
        while h.shape[0] < size:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        h /= size ** .5
        require(torch.allclose(h.T @ h, torch.eye(size, dtype=torch.float64), rtol=0, atol=2e-14),
                'Dense Sylvester oracle is not orthonormal.')
        weight = torch.randn((3, size), generator=generator, dtype=torch.float64)
        signs = fixed_signs(size, 31)
        forward = rotate(weight, signs); inverse = rotate(forward, signs, inverse=True)
        require(torch.allclose(forward, (weight * signs) @ h, rtol=0, atol=2e-14)
                and torch.allclose(inverse, weight, rtol=0, atol=2e-14), 'RHT axis or inverse order is incorrect.')
        require(torch.allclose(hadamard_last(weight), weight @ h, rtol=0, atol=2e-14), 'Butterfly/dense mismatch.')
        p4 = encode_output_scale(weight.float(), bits=4, use_rotation=True)
        require(p4.signs.dtype == torch.int8 and torch.equal(p4.signs, signs)
                and p4.storage()['components']['signs']['storage_bits'] == size * 8,
                'Persist and account actual signs; seed metadata does not replace their storage.')
        rotations[str(size)] = {'inverse_max_abs_fp64': float((inverse - weight).abs().max()),
            'sign_storage_bits': size * 8, 'unquantized_energy_relative_error':
                float(abs(forward.square().sum() / weight.square().sum() - 1))}
    return {'status': 'mechanism_verified', 'signed_nibble_pairs': 256, 'odd_padding_bits': 8,
        'zero_rows_and_signed_extremes': 'passed', 'output_scale_int8_exact_native_row_codec': True,
        'rotation': rotations, 'cuda_initialized': torch.cuda.is_initialized(),
        'boundary': 'Packing/inverse/scale mechanics only; no model or downstream quality claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a fresh report path.')
    report = checks(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
