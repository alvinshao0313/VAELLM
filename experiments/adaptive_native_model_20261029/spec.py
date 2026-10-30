"""Resolve the actual deployed compact geometry; no Qwen/D/K constants."""
from copy import deepcopy

import torch

from experiments.adaptive_vector_rvq.common import require
from litebsq.bitpack import build_bitpack_u8_spec


def resolve(bundle, lengths):
    geometry = deepcopy(bundle['geometry'])
    original = tuple(int(value) for value in geometry['original_shape'])
    compressed = tuple(int(value) for value in geometry['compressed_shape'])
    require(len(original) == len(compressed) == 2 and min(*original, *compressed) > 0,
            'Positive real original/compressed matrix geometry is required.')
    fixed = bundle['fixed']
    required = {'protected_input_indices', 'protected_input_qvalues', 'protected_input_scales',
                'rotation_input_signs', 'rotation_output_signs'}
    require(set(fixed) == required, 'This isolated consumer supports the actual INT8 input-protection payload only.')
    ids = fixed['protected_input_indices']
    require(ids.dtype == torch.int64 and ids.ndim == 1 and len(ids.unique()) == len(ids)
            and bool(((ids >= 0) & (ids < original[1])).all()), 'Invalid protected input coordinates.')
    require(compressed == (original[0], original[1] - len(ids)), 'Protection and compressed source geometry disagree.')
    require(tuple(fixed['protected_input_qvalues'].shape) == (len(ids), original[0])
            and fixed['protected_input_qvalues'].dtype == torch.uint8
            and tuple(fixed['protected_input_scales'].shape) == (len(ids),)
            and fixed['protected_input_scales'].dtype == torch.bfloat16,
            'Preserve the actual INT8 protected-value/scale orientation and precision.')
    for axis, size in (('input', compressed[1]), ('output', compressed[0])):
        signs = fixed[f'rotation_{axis}_signs']
        require(signs.dtype == torch.int8 and tuple(signs.shape) == (size,)
                and bool(((signs == 1) | (signs == -1)).all()), 'Invalid actual rotation signs.')
    stages, dim = [], None
    for index, stage in enumerate(bundle['stages']):
        decoder = deepcopy(stage['decoder_spec'])
        d, k = int(decoder['out_dim']), int(decoder['in_dim'])
        require(d > 0 and k > 0 and (dim is None or d == dim), 'All stages must reconstruct the same vector geometry.')
        dim = d
        rows = int((lengths >= index + 1).sum())
        code_spec = build_bitpack_u8_spec(logical_shape=(rows, k))
        codes = stage['codes']
        require(codes.dtype == torch.uint8 and list(codes.shape) == code_spec['shape'],
                'Actual code rows/ceil(K/8) disagree with the routes and decoder in_dim.')
        if k % 8 and codes.numel():
            require(not bool((codes[:, -1].to(torch.int64) >> (k % 8)).any()), 'Nonzero packed-code tail padding.')
        require(stage['decoder_state'] and all(tensor.dtype == torch.bfloat16 for tensor in stage['decoder_state'].values()),
                'Import the actual BF16 deployment decoder; FP32 masters are runtime-only.')
        stages.append(dict(decoder_spec=decoder, normalization=deepcopy(stage['normalization']),
                           codes=code_spec, decoder_tensor_names=list(stage['decoder_state']),
                           hidden_dim=int(decoder['hidden_dim'])))
    require(stages and compressed[0] * compressed[1] % dim == 0, 'Real compressed geometry must divide into complete D-vectors.')
    n = compressed[0] * compressed[1] // dim
    require(int(bundle['n_vectors']) == n and lengths.dtype == torch.uint8 and tuple(lengths.shape) == (n,)
            and bool(((lengths >= 1) & (lengths <= len(stages))).all()), 'Routing N/length range disagrees with the source geometry.')
    # Both layouts flatten their actual 2D matrix; only the transpose changes restoration.
    require(isinstance(geometry['transpose'], bool), 'The actual flatten/transpose rule must be explicit.')
    return dict(geometry=geometry, n_vectors=n, vector_dim=dim, stages=stages,
                fixed_names=sorted(required), vector_padding=0,
                encoder='not deployed; encoder hidden width is not inferred from decoder width',
                accumulation='active rows only; stage-ordered BF16 decoder outputs and BF16 additions')
