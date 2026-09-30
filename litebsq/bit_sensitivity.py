"""Fixed decoder sensitivity from deterministic single-bit perturbations."""
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
