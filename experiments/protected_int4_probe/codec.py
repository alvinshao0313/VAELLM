"""Isolated CPU output-scale codecs for a fixed protected W[out, input_subset]."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pack_int4(values):
    """Pack two signed two's-complement nibbles per byte, low nibble first."""
    require(values.device.type == 'cpu' and values.dtype == torch.int8 and values.ndim == 2
            and values.numel() > 0, 'Pack a nonempty CPU int8 matrix.')
    require(bool(((values >= -8) & (values <= 7)).all()), 'Signed INT4 values must be in [-8,7].')
    nibbles = torch.bitwise_and(values.to(torch.int16), 15).to(torch.uint8)
    if values.shape[1] % 2:
        nibbles = torch.cat((nibbles, torch.zeros((values.shape[0], 1), dtype=torch.uint8)), dim=1)
    return (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()


def unpack_int4(packed, columns):
    require(packed.device.type == 'cpu' and packed.dtype == torch.uint8 and packed.ndim == 2
            and isinstance(columns, int) and columns > 0 and packed.shape[1] == (columns + 1) // 2,
            'Invalid CPU INT4 payload geometry.')
    if columns % 2:
        require(bool((packed[:, -1] >> 4 == 0).all()), 'Odd-column padding nibble must be zero.')
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    signed = torch.where(nibbles >= 8, nibbles.to(torch.int16) - 16, nibbles.to(torch.int16))
    return signed[:, :columns].to(torch.int8).contiguous()


def hadamard_last(value):
    """Normalized Sylvester transform on the last axis; no padded dimensions."""
    require(value.device.type == 'cpu' and value.dtype in (torch.float32, torch.float64)
            and value.ndim >= 1, 'Hadamard requires a CPU float32/float64 tensor.')
    size = value.shape[-1]
    require(size > 0 and size & (size - 1) == 0, 'Protected subspace must have power-of-two width.')
    transformed = value.clone()
    width = 1
    while width < size:
        blocks = transformed.reshape(*value.shape[:-1], -1, 2, width)
        left, right = blocks[..., 0, :].clone(), blocks[..., 1, :].clone()
        transformed = torch.stack((left + right, left - right), dim=-2).reshape(value.shape)
        width *= 2
    return (transformed / math.sqrt(size)).contiguous()


def fixed_signs(columns, seed=31):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return (torch.randint(0, 2, (columns,), generator=generator, dtype=torch.int8) * 2 - 1).contiguous()


def rotate(weight, signs, *, inverse=False):
    require(signs.device.type == 'cpu' and signs.dtype == torch.int8 and signs.ndim == 1
            and signs.numel() == weight.shape[-1] and bool(((signs == 1) | (signs == -1)).all()),
            'Store one explicit signed int8 sign per protected input coordinate.')
    # Forward W D H; inverse Z H D. D and H generally do not commute.
    return (hadamard_last(weight) * signs).contiguous() if inverse else hadamard_last(weight * signs)


@dataclass(frozen=True)
class Payload:
    values: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int]
    bits: int
    signs: torch.Tensor | None = None

    def tensors(self):
        return {'values': self.values, 'scales': self.scales,
                **({'signs': self.signs} if self.signs is not None else {})}

    def storage(self):
        components = {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                      'storage_bits': tensor.untyped_storage().nbytes() * 8}
                      for name, tensor in self.tensors().items()}
        padding = self.shape[0] * (self.shape[1] % 2) * 4 if self.bits == 4 else 0
        return {'components': components, 'total_bits': sum(v['storage_bits'] for v in components.values()),
                'logical_value_bits': math.prod(self.shape) * self.bits, 'padding_bits': padding,
                'scale_count': self.shape[0], 'scale_axis': 'original_output',
                'zero_point_bits': 0, 'hadamard_matrix_bits': 0,
                'sign_storage': 'explicit int8 per protected input coordinate; no cross-projection sharing'}


def encode_output_scale(weight, *, bits, use_rotation=False, seed=31):
    require(weight.device.type == 'cpu' and weight.ndim == 2 and weight.numel() > 0
            and weight.is_floating_point() and bool(torch.isfinite(weight).all()),
            'Encode finite nonempty CPU W[out,protected_input].')
    require(bits in (4, 8), 'Only the diagnostic INT8 or candidate INT4 is defined.')
    target = weight.detach().float().contiguous()
    signs = fixed_signs(target.shape[1], seed) if use_rotation else None
    if signs is not None:
        target = rotate(target, signs)
    maximum = target.abs().amax(dim=1)
    qmax, qmin = (7, -8) if bits == 4 else (127, -128)
    scales = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum)).to(torch.bfloat16).contiguous()
    require(bool(torch.isfinite(scales).all()) and bool((scales > 0).all()),
            'Scale overflow/underflow is unsupported; do not silently change the rule.')
    quantized = torch.round(target / scales.float().unsqueeze(1)).clamp(qmin, qmax).to(torch.int8)
    values = pack_int4(quantized) if bits == 4 else quantized.view(torch.uint8).contiguous()
    return Payload(values, scales, tuple(target.shape), bits, signs)


def decode_output_scale(payload, *, dtype=torch.bfloat16):
    require(payload.bits in (4, 8) and payload.values.device.type == payload.scales.device.type == 'cpu'
            and payload.values.dtype == torch.uint8 and payload.scales.dtype == torch.bfloat16
            and payload.scales.shape == (payload.shape[0],), 'Invalid output-scale payload.')
    if payload.bits == 4:
        signed = unpack_int4(payload.values, payload.shape[1])
    else:
        require(tuple(payload.values.shape) == payload.shape, 'INT8 payload shape differs.')
        signed = payload.values.view(torch.int8)
    require(tuple(signed.shape) == payload.shape, 'Decoded matrix shape differs.')
    decoded = signed.float() * payload.scales.float().unsqueeze(1)
    if payload.signs is not None:
        decoded = rotate(decoded, payload.signs, inverse=True)
    return decoded.to(dtype).contiguous()
