"""Device-independent input-energy checks and matched local BF16 readouts."""
from __future__ import annotations

from pathlib import Path

import torch

from experiments.adaptive_vector_rvq import codec, common as c
from rotation.weight_preprocess import _transform_last, build_weight_rotation

HERE = Path(__file__).resolve().parent
OUTPUT = Path('/data/star/VAELLM/adaptive_layer_transfer_20261028')
LAYERS = (18, 35)


def check_prepared(directory):
    record = c.read(Path(directory) / 'prepared.json')
    c.require(record['status'] == 'completed'
        and record['schema'] == 'adaptive_layer_transfer_20261028_inputs_v1',
        'Use the completed independent L18/L35 CPU preparation.')
    for path, expected in record['source_sha256'].items():
        c.require(c.sha(path) == expected, 'Bound source/input changed: ' + path)
    for item in record['input_stats']:
        c.require(c.stat(item['path']) == item, 'Bound input identity changed: ' + item['path'])
    return record


def activation_inputs(archive, manifest, name):
    lengths, documents = archive['document_lengths'], manifest['documents']
    c.require(len(lengths) == len(documents) == 128
        and lengths == [row['valid_tokens'] for row in documents]
        and [row['row_index'] for row in documents] == list(range(700000, 700128)),
        'Preserve the exact original contiguous train documents and order.')
    x = archive['inputs'][name]
    split = sum(lengths[:64])
    c.require(split == 15183 and sum(lengths[64:]) == 15492
        and sum(lengths) == len(x) == 30675 and x.dtype == torch.bfloat16
        and tuple(x.shape) == (30675, 12288), 'Use the fixed train 64/64 document split.')
    return x, split


@torch.no_grad()
def input_energy(source, target, prefix, x, split, e2, device):
    geometry, fixed = source['geometry'], source['fixed']
    out_features, in_features = geometry['compressed_shape']
    c.require(geometry['transpose'] and [out_features, in_features] == [4096, 12032],
        'Use the original complete down transpose geometry.')
    keep = torch.ones(geometry['original_shape'][1], dtype=torch.bool)
    keep[fixed['protected_input_indices']] = False
    kept = torch.nonzero(keep, as_tuple=False).flatten()
    factor = out_features // c.DIM
    c.require(len(kept) == in_features and factor == 128
        and in_features * factor == len(target), 'Preserve input-row/vector ordering.')
    transform = build_weight_rotation(out_features, in_features, geometry['rotation'])
    transform.input_signs.copy_(fixed['rotation_input_signs'])
    transform.output_signs.copy_(fixed['rotation_output_signs'])
    c.require(transform.input_block == transform.output_block == 32,
        'Preserve the specified two-sided block32 rotation.')
    sin, sout = transform.input_signs.float(), transform.output_signs[:32].float()
    ids = torch.arange(in_features) * factor
    error = (target.index_select(0, ids).float() - prefix.index_select(0, ids).float()).t().contiguous()
    xb = x[:64].index_select(1, kept).float()
    xr = _transform_last(xb * sin, block_size=32, factor=transform.input_factor,
        had=transform.input_had, inverse=False)
    original_error = _transform_last(error, block_size=32, factor=transform.input_factor,
        had=transform.input_had, inverse=True) * sin
    original_error = _transform_last(original_error.t(), block_size=32,
        factor=transform.output_factor, had=transform.output_had, inverse=True).t() * sout[:, None]
    lhs = _transform_last((xb @ original_error.t()) * sout, block_size=32,
        factor=transform.output_factor, had=transform.output_had, inverse=False)
    rhs = xr @ error.t()
    relative = float(torch.linalg.vector_norm((lhs - rhs).double()) / torch.linalg.vector_norm(rhs.double()))
    c.require(relative <= 2e-5, 'Real FP32 bilinear rotation direction/sign identity failed.')
    energy = torch.zeros(in_features, dtype=torch.float64, device=device)
    transform = transform.to(device)
    signs = sin.to(device)
    for start in range(0, split, 256):
        values = x[start:min(start + 256, split)].index_select(1, kept).to(device=device, dtype=torch.float32)
        rotated = _transform_last(values * signs, block_size=32, factor=transform.input_factor,
            had=transform.input_had, inverse=False)
        energy += rotated.double().square().sum(0)
    h = (energy / split).cpu()
    c.require(bool(torch.isfinite(h).all()) and bool((h >= 0).all()) and float(h.mean()) > 0,
        'Calibration second moments must be finite/nonnegative with a positive global mean.')
    normalized = h / h.mean()
    weights = normalized.repeat_interleave(factor)
    column_energy = torch.empty(in_features, dtype=torch.float64)
    for start in range(0, in_features, 128):
        end = min(start + 128, in_features)
        residual = target[start * factor:end * factor].double() - prefix[start * factor:end * factor].double()
        column_energy[start:end] = residual.reshape(end - start, out_features).square().sum(-1)
    left = float((weights * e2 * c.DIM).sum())
    right = float((normalized * column_energy).sum())
    layout_relative = abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)
    c.require(layout_relative < 1e-12, 'Input-row second moments must repeat128 in vector order.')
    checks = dict(bilinear_FP32_relative_L2=relative, bilinear_relative_L2_tolerance=2e-5,
        tolerance_basis='FP32 signed FWHT and12032-term dot reductions, before BF16 inverse rounding',
        weighted_vector_matrix_relative_error=layout_relative, vectors_per_input_row=factor,
        input_sign_before_forward_hadamard=True, original_unprotected_columns_ascending=True,
        no_epsilon_or_clipping=True, calibration_tokens=split)
    return h, normalized, weights, kept, checks


@torch.no_grad()
def decode_compact(bundle, lengths, device):
    vectors = torch.zeros((bundle['n_vectors'], c.DIM), dtype=torch.bfloat16)
    for index, stage in enumerate(bundle['stages'], start=1):
        active = torch.nonzero(lengths >= index, as_tuple=False).flatten()
        codec.add_stage(vectors, stage, active, device)
    return vectors


@torch.no_grad()
def output_metrics(weight, reference, x, split, device):
    result = {}
    for name, begin, end in [('calibration64', 0, split), ('heldout_train64', split, len(x))]:
        error_sum, target_sum, elements = 0., 0., 0
        for start in range(begin, end, 256):
            values = x[start:min(start + 256, end)].to(device)
            actual = torch.nn.functional.linear(values, weight)
            expected = torch.nn.functional.linear(values, reference)
            c.require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
                'Nonfinite actual native BF16 Linear outputs.')
            error_sum += float((actual.double() - expected.double()).square().sum())
            target_sum += float(expected.double().square().sum())
            elements += expected.numel()
        result[name] = dict(tokens=end - begin, output_elements=elements, squared_error_sum=error_sum,
            target_energy_sum=target_sum, MSE=error_sum / elements, NMSE=error_sum / target_sum)
    a, b = result['calibration64'], result['heldout_train64']
    error, energy = a['squared_error_sum'] + b['squared_error_sum'], a['target_energy_sum'] + b['target_energy_sum']
    count = a['output_elements'] + b['output_elements']
    result['whole128'] = dict(tokens=len(x), output_elements=count, squared_error_sum=error,
        target_energy_sum=energy, MSE=error / count, NMSE=error / energy,
        aggregation='Sum both groups numerator/target energy/count, never average NMSE.')
    squared = (weight.double() - reference.double()).square().sum()
    result.update(original_BF16_weight_MSE=float(squared / reference.numel()),
        original_BF16_weight_NMSE=float(squared / reference.double().square().sum()))
    return result
