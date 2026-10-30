"""Live-target geometry and errors for the actual compact BF16 prefix."""
import math

import torch

from experiments.adaptive_vector_rvq.codec import restore_decoder
from experiments.adaptive_vector_rvq.common import require
from litebsq.bitpack import unpack_uint8_tensor_to_bool
from rotation.weight_preprocess import _transform_last, build_weight_rotation
from .codec import decode_lengths
from .spec import resolve


def _resolved(bundle):
    lengths = decode_lengths(bundle['length_frame'], int(bundle['n_vectors']))
    require(1 <= len(bundle['stages']) <= 3, 'Use one to three actual residual stages.')
    return lengths, resolve(bundle, lengths)


def _rotation(bundle, resolved, device):
    rotation = build_weight_rotation(*resolved['geometry']['compressed_shape'], resolved['geometry']['rotation'])
    require(rotation is not None, 'The actual imported rotation must be explicit.')
    rotation = rotation.to(device)
    for axis in ('input', 'output'):
        getattr(rotation, axis + '_signs').copy_(bundle['fixed'][f'rotation_{axis}_signs'].to(device))
    rotation.validate_state()
    return rotation


def _kept(bundle, resolved, device):
    keep = torch.ones(resolved['geometry']['original_shape'][1], dtype=torch.bool, device=device)
    keep[bundle['fixed']['protected_input_indices'].to(device)] = False
    return torch.nonzero(keep, as_tuple=False).flatten()


@torch.no_grad()
def live_vectors(bundle, live_weight):
    """Rotate a caller-bound live weight with the existing protection and signs.

    This does not choose protected coordinates, requantize them, or infer a target
    from a decoder/HF checkpoint. Source identity and train split belong to callers.
    """
    _, resolved = _resolved(bundle)
    require(live_weight.is_floating_point() and tuple(live_weight.shape) == tuple(resolved['geometry']['original_shape'])
            and bool(torch.isfinite(live_weight).all()), 'Provide the finite actual live matrix in its original geometry.')
    rotation = _rotation(bundle, resolved, live_weight.device)
    weight = live_weight.index_select(1, _kept(bundle, resolved, live_weight.device)).float()
    rotated = rotation(weight)
    if resolved['geometry']['transpose']:
        rotated = rotated.t().contiguous()
    return rotated.reshape(resolved['n_vectors'], resolved['vector_dim']).contiguous()


@torch.no_grad()
def input_energy(bundle, inputs, *, chunk=256):
    """Normalize E[(forward_Hadamard(x_unprotected * input_sign))**2]."""
    _, resolved = _resolved(bundle)
    require(isinstance(chunk, int) and not isinstance(chunk, bool) and chunk > 0,
            'Input-energy chunk must be a positive integer.')
    require(inputs.is_floating_point() and inputs.ndim == 2 and len(inputs) > 0
            and inputs.shape[1] == resolved['geometry']['original_shape'][1],
            'Provide actual valid-token input rows in the original input geometry.')
    rotation = _rotation(bundle, resolved, inputs.device)
    kept = _kept(bundle, resolved, inputs.device)
    energy = torch.zeros(len(kept), dtype=torch.float64, device=inputs.device)
    for start in range(0, len(inputs), chunk):
        values = inputs[start:start + chunk].index_select(1, kept).float()
        require(bool(torch.isfinite(values).all()), 'Nonfinite live calibration input.')
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            rotated = _transform_last(values * rotation.input_signs.float(), block_size=rotation.input_block,
                factor=rotation.input_factor, had=rotation.input_had, inverse=False)
        energy += rotated.double().square().sum(0)
    h = energy / len(inputs)
    require(bool(torch.isfinite(h).all()) and float(h.mean()) > 0, 'Live rotated inputs have no finite positive energy.')
    return (h / h.mean()).cpu().contiguous()


@torch.no_grad()
def decode_prefixes(bundle, *, start=0, stop=None, chunk=8192, device='cpu'):
    """Decode full-bank prefixes, optionally only a contiguous range of real rows.

    Each decoder operation and addition matches CompactRVQLinear. Keep only one
    layer's prefixes at a time; routed banks cannot supply missing full-bank rows.
    """
    lengths, resolved = _resolved(bundle)
    n, dim = resolved['n_vectors'], resolved['vector_dim']
    stop = n if stop is None else stop
    require(all(isinstance(value, int) and not isinstance(value, bool) for value in (start, stop, chunk))
            and 0 <= start < stop <= n and chunk > 0, 'Invalid actual prefix row range or chunk.')
    require(bool((lengths == len(bundle['stages'])).all()), 'Prefix errors require complete banks for every candidate stage.')
    prefix = torch.zeros((stop - start, dim), dtype=torch.bfloat16)
    prefixes = []
    for stage in bundle['stages']:
        decoder = restore_decoder(stage, device)
        k = int(stage['decoder_spec']['in_dim'])
        for offset in range(start, stop, chunk):
            packed = stage['codes'][offset:min(offset + chunk, stop)].to(device)
            bits = unpack_uint8_tensor_to_bool(packed, logical_shape=(len(packed), k))
            values = decoder(bits.to(torch.bfloat16).unsqueeze(1)).squeeze(1)
            require(values.dtype == torch.bfloat16 and bool(torch.isfinite(values).all()),
                    'Actual prefix decoding must remain finite BF16.')
            local = offset - start
            prefix[local:local + len(values)] = prefix[local:local + len(values)] + values.cpu()
        prefixes.append(prefix.clone())
    return prefixes


@torch.no_grad()
def prefix_errors(target, prefixes, h, geometry, source_variance, *, start=0, chunk=8192):
    """Return CPU FP64 per-vector weighted errors in prefix order.

    For t=v*D+d, input coordinates are t % I_c without transpose and
    t // O_c with transpose. No vector-mean h approximation is used.
    """
    require(target.device.type == 'cpu' and target.dtype == torch.float32 and target.ndim == 2
            and min(target.shape) > 0, 'Use finite FP32 live rotated target vectors on CPU.')
    require(isinstance(prefixes, (tuple, list)) and 1 <= len(prefixes) <= 3
            and all(value.device.type == 'cpu' and value.dtype == torch.bfloat16 and value.shape == target.shape
                    for value in prefixes), 'Provide one to three actual same-layout BF16 prefixes.')
    require(isinstance(geometry['transpose'], bool), 'The real flatten/transpose layout must be explicit.')
    shape = tuple(geometry['compressed_shape'])
    require(len(shape) == 2 and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape),
            'Positive compressed matrix geometry is required.')
    out_features, in_features = shape
    n, dim = target.shape
    require(out_features * in_features % dim == 0
            and all(isinstance(value, int) and not isinstance(value, bool) for value in (start, chunk))
            and start >= 0 and start + n <= out_features * in_features // dim and chunk > 0,
            'Target rows must map to complete real D-vectors in the declared geometry.')
    require(h.device.type == 'cpu' and h.is_floating_point() and tuple(h.shape) == (in_features,)
            and bool(torch.isfinite(h).all()) and bool((h >= 0).all())
            and math.isclose(float(h.double().mean()), 1., rel_tol=1e-6, abs_tol=1e-12),
            'Provide the finite nonnegative mean-normalized live input second moments.')
    require(isinstance(source_variance, (int, float)) and not isinstance(source_variance, bool)
            and math.isfinite(source_variance) and source_variance > 0, 'Provide the positive full live-source variance.')
    errors = torch.empty((n, len(prefixes)), dtype=torch.float64)
    for offset in range(0, n, chunk):
        end = min(n, offset + chunk)
        coordinates = torch.arange((start + offset) * dim, (start + end) * dim, dtype=torch.int64)
        inputs = coordinates // out_features if geometry['transpose'] else coordinates % in_features
        weights = h.index_select(0, inputs).double().reshape(end - offset, dim)
        values = target[offset:end].float()
        require(bool(torch.isfinite(values).all()), 'Nonfinite live target vectors.')
        for index, prefix in enumerate(prefixes):
            require(bool(torch.isfinite(prefix[offset:end]).all()), 'Nonfinite actual BF16 prefix.')
            # The deployed BF16 prefix is first promoted to FP32, as for stage3's residual.
            residual = (values - prefix[offset:end].float()).double()
            errors[offset:end, index] = (weights * residual.square()).sum(1) / (dim * source_variance)
    require(bool(torch.isfinite(errors).all()), 'Nonfinite normalized prefix errors.')
    return errors
