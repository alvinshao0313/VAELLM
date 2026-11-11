"""Unchanged compact Muon math, isolated from that experiment's N-runtime imports.

Source: experiments/muon_compact_cmcc_20261030/optimizer.py, functions
stages/zeropower/direction. CPU checks compare their ASTs after b.require→require.
"""
import math
import time

import torch

from experiments.cat_auto_protocol_cmcc_20261104.contracts import require


def stages(value, count):
    if value.ndim == 2:
        require(count == 1, 'An actual independent nn.Linear leaf is one matrix.')
        return value.unsqueeze(0)
    require(value.ndim == 3 and value.shape[-1] == 1 and count > 0 and value.shape[0] % count == 0,
        'Muon matrices must have the explicit source stage layout.')
    return value[...,0].reshape(count, value.shape[0]//count, value.shape[1])


def zeropower(value):
    require(value.ndim == 3, 'NS receives separate stage matrices.')
    x = value.to(torch.bfloat16)
    transpose = x.size(-2) > x.size(-1)
    if transpose: x = x.mT
    x = x/(x.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(5):
        a = x@x.mT
        polynomial = -4.7750*a+2.0315*(a@a)
        x = 3.4445*x+polynomial@x
    if transpose: x = x.mT
    require(torch.isfinite(x).all().item(), 'Nonfinite stage-local BF16 NS update.')
    return x


def direction(gradient, momentum, count):
    momentum.lerp_(gradient, .05)
    nesterov = gradient.lerp(momentum, .95)
    matrix = stages(nesterov, count)
    if gradient.is_cuda: torch.cuda.synchronize()
    started = time.perf_counter()
    orthogonalized = zeropower(matrix)
    elapsed = time.perf_counter()-started
    result = orthogonalized.float()*(.2*math.sqrt(max(matrix.shape[-2:])))
    return result.reshape_as(gradient), elapsed
