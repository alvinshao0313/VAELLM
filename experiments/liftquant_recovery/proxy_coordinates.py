"""Training-only coordinates calibrated to the existing decoder's bit sensitivity.

p = s * (b - 1/2), b(p) = clamp(round(p/s + 1/2), 0, 1).
The fixed s is measured before training; it is NOT a new model parameter.
"""
import math

import torch

from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool


@torch.no_grad()
def measure_scale(packed, logical_shape, decode, rows=256):
    if len(logical_shape) != 3 or logical_shape[1] != 1 or not 0 < rows <= 512:
        raise ValueError('Expected one code bank and a bounded sensitivity sample.')
    count = min(rows, logical_shape[0])
    if count == 0:
        raise ValueError('Cannot calibrate an empty code bank.')
    indices = torch.linspace(0, logical_shape[0]-1, count, device=packed.device).round().long()
    sample = packed.index_select(0, indices).contiguous()
    bits = unpack_uint8_tensor_to_bool(sample, logical_shape=(count, 1, logical_shape[-1]))
    reference = decode(sample).float()
    bit_mse = []
    for start in range(0, logical_shape[-1], 8):
        stop = min(start + 8, logical_shape[-1])
        flipped = bits.unsqueeze(0).expand(stop-start, -1, -1, -1).clone()
        for i, bit in enumerate(range(start, stop)):
            flipped[i, :, :, bit].logical_not_()
        shape = ((stop-start)*count, 1, logical_shape[-1])
        trial = decode(pack_bool_tensor_to_uint8(flipped.reshape(shape), logical_shape=shape)).float()
        diff = trial.reshape(stop-start, count, 1, -1) - reference.unsqueeze(0)
        bit_mse.append(diff.square().mean(dim=(1, 2, 3)).cpu())
    rms_by_bit = torch.cat(bit_mse).sqrt()
    scale = rms_by_bit.square().mean().sqrt().item()
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f'Nonpositive/nonfinite decoder sensitivity: {scale}')
    return dict(scale=scale, rows=count, row_indices=indices.cpu().tolist(),
                single_bit_rms=rms_by_bit.tolist(), initial_margin=scale/2,
                definition='RMS decoded-weight change across sampled rows, all single-bit flips and output coordinates',
                fixed_during_training=True, saved_in_model=False)


def hard_bits(proxy, scale):
    # Keep the exact round-to-even tie rule and saturation of the specified STE.
    return (proxy / scale + 0.5).round().clamp_(0, 1).bool()


def ste_gate(proxy, scale):
    rounded = (proxy / scale + 0.5).round()
    return (rounded >= 0) & (rounded <= 1)


def dense_ste(proxy, scale):
    normalized = proxy / scale + 0.5
    rounded = normalized + (normalized.round() - normalized).detach()
    return rounded.clamp(0, 1)


def motion_budget(initial_lr, steps, margin):
    if min(initial_lr, steps, margin) <= 0:
        raise ValueError('Expected a positive lr, step budget and margin.')
    b1, b2 = .9, .999
    ratio_bound = math.sqrt((1-b1)**2 / ((1-b2)*(1-b1*b1/b2)))
    # Standard cosine upper-bounds the pinned upstream step()+get_lr() sequence.
    lr_sum_upper = initial_lr * (.525*steps + .475)
    bound = lr_sum_upper * ratio_bound
    return dict(steps=steps, initial_lr=initial_lr, margin=margin,
                adam_total_motion_upper_bound=bound,
                upper_bound_to_margin=bound/margin,
                structurally_unreachable=bound < margin,
                constant_gradient_motion_to_margin_upper=lr_sum_upper/margin,
                meaning='A bound above the margin permits crossing; it does not guarantee a flip or improvement.')
