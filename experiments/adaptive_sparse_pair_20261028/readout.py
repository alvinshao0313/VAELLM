"""One CPU BF16 target output per batch shared by all three actual endpoints."""
from __future__ import annotations

import time

import torch

from experiments.adaptive_vector_rvq import common as c


@torch.no_grad()
def matched_outputs(weights, reference, x, split, emit):
    results = {name: {} for name in weights}
    batches = 0
    for group, begin, end in [('calibration64', 0, split), ('heldout_train64', split, len(x))]:
        values = {name: [0., 0] for name in weights}
        target_energy = 0.
        for start in range(begin, end, 256):
            started = time.perf_counter()
            block = x[start:min(start + 256, end)]
            expected = torch.nn.functional.linear(block, reference)
            c.require(bool(torch.isfinite(expected).all()), 'Nonfinite actual HF BF16 CPU output.')
            target_energy += float(expected.double().square().sum())
            for name, weight in weights.items():
                actual = torch.nn.functional.linear(block, weight)
                c.require(bool(torch.isfinite(actual).all()), 'Nonfinite actual BF16 CPU endpoint output.')
                values[name][0] += float((actual.double() - expected.double()).square().sum())
                values[name][1] += expected.numel()
            batches += 1
            if batches == 1:
                elapsed = time.perf_counter() - started
                total_batches = (split + 255) // 256 + (len(x) - split + 255) // 256
                emit(dict(event='first_formal_BF16_CPU_batch_complete', batch_tokens=len(block),
                    seconds=elapsed, estimated_total_batch_seconds=total_batches * elapsed,
                    formal_batches=total_batches, target_F_linear_calls_per_batch=1,
                    endpoint_F_linear_calls_per_batch=len(weights)))
        for name, (squared, count) in values.items():
            results[name][group] = dict(tokens=end - begin, output_elements=count,
                squared_error_sum=squared, target_energy_sum=target_energy,
                MSE=squared / count, NMSE=squared / target_energy)
        emit(dict(event='matched_output_group_complete', group=group,
            results={name: results[name][group] for name in weights}))
    for name in weights:
        a, b = results[name]['calibration64'], results[name]['heldout_train64']
        error = a['squared_error_sum'] + b['squared_error_sum']
        energy = a['target_energy_sum'] + b['target_energy_sum']
        count = a['output_elements'] + b['output_elements']
        results[name]['whole128'] = dict(tokens=len(x), output_elements=count,
            squared_error_sum=error, target_energy_sum=energy, MSE=error / count, NMSE=error / energy,
            aggregation='Sum both groups numerator/target energy/count, never average NMSE.')
    return results
