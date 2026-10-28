"""Original-coordinate matrix and fixed FineWeb-train local-output observations."""
from __future__ import annotations

import torch
from safetensors import safe_open

from . import common as c
from litebsq.protected_channel_quant import decode_protected_channel_weight
from rotation.weight_preprocess import build_weight_rotation


@torch.no_grad()
def original_weight(bundle, vectors, device):
    geometry, fixed = bundle['geometry'], bundle['fixed']
    shape = geometry['compressed_shape']
    rotated = vectors.reshape(shape[1], shape[0]).t().contiguous().to(device)
    transform = build_weight_rotation(*shape, geometry['rotation']).to(device)
    transform.input_signs.copy_(fixed['rotation_input_signs'].to(device))
    transform.output_signs.copy_(fixed['rotation_output_signs'].to(device))
    decoded = transform(rotated, inverse=True)
    keep = torch.ones(geometry['original_shape'][1], dtype=torch.bool, device=device)
    ids = fixed['protected_input_indices'].to(device)
    keep[ids] = False
    result = torch.empty(geometry['original_shape'], dtype=torch.bfloat16, device=device)
    result[:, keep] = decoded
    protected = decode_protected_channel_weight(fixed['protected_input_qvalues'], fixed['protected_input_scales'], quant_format='int8', dtype=torch.bfloat16, device=torch.device(device))
    result[:, ids] = protected.t()
    c.require(bool(torch.isfinite(result).all()), 'Nonfinite actual original-coordinate BF16 matrix.')
    return result


@torch.no_grad()
def compare(bundle, vectors, activation_record, device):
    geometry = bundle['geometry']
    index = c.read(c.HF / 'model.safetensors.index.json')['weight_map']
    key = geometry['source_key']
    with safe_open(str(c.HF / index[key]), framework='pt', device='cpu') as stream:
        reference = stream.get_tensor(key).to(device)
    reconstructed = original_weight(bundle, vectors, device)
    squared = (reference.double() - reconstructed.double()).square().sum()
    denominator = reference.double().square().sum()
    result = dict(original_BF16_weight_MSE=float(squared / reference.numel()), original_BF16_weight_NMSE=float(squared / denominator))
    if 'input' in activation_record:
        c.require(c.stat(activation_record['path']) == activation_record['input'], 'The fixed output observation inputs changed.')
        archive = torch.load(activation_record['path'], map_location='cpu', weights_only=True, mmap=True)
        x = archive['inputs'][geometry['name']]
        c.require(x.dtype == torch.bfloat16 and tuple(x.shape) == (30675, 12288), 'Actual FineWeb train inputs changed.')
        error_sum, reference_sum, count = 0., 0., 0
        for start in range(0, len(x), 256):
            block = x[start:start + 256].to(device)
            actual = torch.nn.functional.linear(block, reconstructed)
            expected = torch.nn.functional.linear(block, reference)
            c.require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), 'Nonfinite actual local BF16 outputs.')
            error_sum += float((actual.double() - expected.double()).square().sum())
            reference_sum += float(expected.double().square().sum())
            count += expected.numel()
        result.update(local_output_MSE=error_sum / count, local_output_NMSE=error_sum / reference_sum,
            observation_tokens=len(x), observation='Complete existing FineWeb-Edu train inputs in original HF context; local down-linear outputs only, not blind heldout/end-model; inputs never enter training/routing.')
    return result
