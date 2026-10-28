"""Exact binary-payload budget and nested stable residual routing."""
from __future__ import annotations

import math

import torch

from .common import BITS, HEADER_BYTES, require


def counts_for_budget(n, decoder_bytes, fixed_bytes=0):
    require(type(n) is int and n > 0 and decoder_bytes > 0, 'Positive real vector count/decoder size required.')
    row_bytes = BITS // 8
    control = HEADER_BYTES + fixed_bytes + 2 * decoder_bytes + 2 * n * row_bytes
    count_bytes = (n + 3) // 4
    base = HEADER_BYTES + fixed_bytes + 4 * decoder_bytes + n * row_bytes + count_bytes
    extra = (control - base) // row_bytes
    require(0 <= extra < 3 * n, 'The fixed2 budget cannot fund mandatory stage1, count bits and four decoders.')
    low, high = 0., 1.
    for _ in range(80):
        middle = (low + high) / 2
        if sum(math.floor(n * middle ** power) for power in (1, 2, 3)) <= extra:
            low = middle
        else:
            high = middle
    values = [math.floor(n * low ** power) for power in (1, 2, 3)]
    rounding_additions = [0, 0, 0]
    # Coincident floor breakpoints can leave one or two rows; fill earlier stages first.
    while sum(values) < extra:
        for index in range(3):
            upper = n if index == 0 else values[index - 1]
            if values[index] < upper:
                values[index] += 1
                rounding_additions[index] += 1
                break
        else:
            raise ValueError('Nested budget rounding could not place an available row.')
    require(n >= values[0] >= values[1] >= values[2] >= 0, 'Active quotas must form a prefix.')
    adaptive = base + row_bytes * sum(values)
    require(0 <= control - adaptive < row_bytes, 'Use the complete fixed2 budget to within one packed code row.')
    return dict(vector_count=n, decoder_bytes=decoder_bytes, fixed_bytes=fixed_bytes, header_bytes=HEADER_BYTES,
        count_bytes=count_bytes, control_bytes=control, adaptive_bytes=adaptive, unused_bytes=control - adaptive,
        extra_code_rows=extra, rho=low, active_counts=[n, *values], floor_rounding_additions=rounding_additions)


def select_active(target, prefix, previous, count, source_variance, chunk=65536):
    require(0 <= count <= previous.numel() and source_variance > 0, 'Invalid nested active quota/variance.')
    require(previous.dtype == torch.int64 and previous.device.type == 'cpu'
        and (previous.numel() < 2 or bool((previous[1:] > previous[:-1]).all())), 'Previous IDs must be unique and ascending.')
    scores = torch.empty(previous.numel(), dtype=torch.float64)
    for start in range(0, previous.numel(), chunk):
        ids = previous[start:start + chunk]
        error = target.index_select(0, ids).double() - prefix.index_select(0, ids).double()
        scores[start:start + ids.numel()] = error.square().mean(-1) / source_variance
    require(bool(torch.isfinite(scores).all()), 'Nonfinite actual deployed-prefix routing error.')
    ranking = torch.argsort(scores, descending=True, stable=True)
    chosen = previous.index_select(0, ranking[:count]).sort().values.contiguous()
    threshold = float(scores[ranking[count - 1]]) if count else None
    return chosen, dict(selected=count, previous_count=previous.numel(), normalized_MSE_threshold=threshold,
        source_variance=source_variance, tie_break='ascending original vector index', score='actual cumulative deployed-prefix residual MSE / fixed original source variance')
