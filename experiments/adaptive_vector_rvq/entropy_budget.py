"""At most four actual-zlib budget iterations; no global entropy-cost optimum claim."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from . import codec, common as c
from .length_codec_probe import probe


@torch.no_grad()
def allocate(errors, max_rows):
    c.require(errors.device.type == 'cpu' and errors.dtype == torch.float64
        and errors.ndim == 2 and 2 <= errors.shape[1] <= 4
        and bool(torch.isfinite(errors).all()) and bool((errors >= 0).all()),
        'Use finite nonnegative actual/explicitly ideal FP64 prefix errors.')
    n, stages = errors.shape
    c.require(n <= max_rows <= stages * n, 'Invalid prefix-row budget.')
    costs = torch.arange(1, stages + 1, dtype=torch.float64)
    best_dual = -math.inf

    def evaluate(value):
        nonlocal best_dual
        minima, selected = (errors + value * costs).min(dim=1)
        lengths = (selected + 1).to(torch.uint8)
        rows = int(lengths.long().sum())
        primal = float(errors.gather(1, selected[:, None]).sum())
        dual = float(minima.sum()) - value * max_rows
        best_dual = max(best_dual, dual)
        return lengths, rows, primal

    lengths, rows, primal = evaluate(0.)
    if rows <= max_rows:
        return lengths, dict(lambda_value=0., code_rows=rows, NMSE=primal / n,
            fixed_row_budget_dual_NMSE=best_dual / n, bisection_iterations=0)
    high = max(0., float(((errors[:, :1] - errors[:, 1:]) / costs[1:].sub(1)).max()))
    high += max(1., abs(high)) * 16 * torch.finfo(torch.float64).eps
    _, rows, _ = evaluate(high)
    c.require(rows == n, 'Upper lambda must select stage1 everywhere.')
    low = 0.
    for _ in range(60):
        middle = (low + high) / 2
        _, rows, _ = evaluate(middle)
        if rows > max_rows:
            low = middle
        else:
            high = middle
    lengths, rows, primal = evaluate(high)
    c.require(rows <= max_rows and primal + n * 1e-12 >= best_dual, 'Fixed-row feasibility/weak duality failed.')
    return lengths, dict(lambda_value=high, code_rows=rows, NMSE=primal / n,
        fixed_row_budget_dual_NMSE=best_dual / n, bisection_iterations=60)


def iterate(errors, budget, output, *, original_elements, actual_stages, ideal_stages):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    n = len(errors)
    current = n + budget['extra_code_rows']
    base = budget['fixed_bytes'] + budget['header_bytes'] + 4 * budget['decoder_bytes']
    limit = budget['control_bytes']
    rounds, seen, chosen, chosen_lengths = [], set(), None, None
    for index in range(1, 5):
        seen.add(current)
        lengths, solution = allocate(errors, current)
        packed = codec.pack_lengths(lengths)
        frame = probe(packed.numpy().tobytes(), lengths, f'round_{index:02d}', output, original_elements)
        total = base + 6 * solution['code_rows'] + frame['compressed_total_actual_bytes']
        next_rows = (limit - base - frame['compressed_total_actual_bytes']) // 6
        row = dict(round=index, max_code_rows=current, **solution, length_frame=frame,
            complete_accounting_bytes=total, budget_bytes=limit, feasible=total <= limit,
            remaining_bytes=limit - total, next_max_code_rows=next_rows,
            accounting_only=True, complete_model_file_written=False)
        rounds.append(row)
        if row['feasible']:
            chosen, chosen_lengths = row, lengths
        if next_rows == current or next_rows in seen:
            stop = 'row_budget_unchanged' if next_rows == current else 'row_budget_repeated'
            break
        current = next_rows
    else:
        stop = 'four_round_limit'
    c.require(chosen is not None, 'No feasible candidate in the predeclared four iterations.')
    report = dict(status='completed', rounds=rounds, chosen_round=chosen['round'], chosen=chosen,
        stop_reason=stop, maximum_rounds=4, actual_stages=actual_stages, ideal_stages=ideal_stages,
        reserved_decoder_stages=4, actual_decoder_stages=actual_stages,
        unmaterialized_decoder_reservation_bytes=(4 - actual_stages) * budget['decoder_bytes'],
        fixed_bytes=budget['fixed_bytes'], original_model_header_bytes=budget['header_bytes'],
        reserved_decoder_bytes=4 * budget['decoder_bytes'],
        boundary='Each lambda solves only a fixed-code-row separable allocation. Zlib length cost is remeasured after each route; this bounded iteration is not a global optimum or global lower bound for nonseparable compressed costs.')
    c.write(output / 'report.json', report)
    return chosen_lengths, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not args.output_dir.exists(),
        'Hide CUDA and use an independent output directory.')
    torch.set_num_threads(4)
    root, output = args.root_dir.resolve(), args.output_dir.resolve()
    prepared = c.check_prepared(root / 'prepared')
    layer = prepared['layers']['0']
    errors_path = root / 'threshold_diagnostic/per_vector_errors.pt'
    diagnostic_path = errors_path.with_name('report.json')
    diagnostic = c.read(diagnostic_path)
    c.require(diagnostic['status'] == 'completed' and c.sha(errors_path) == diagnostic['per_vector']['sha256']
        and diagnostic['target_digest'] == layer['target_digest'], 'Use bound real L0 e1/e2.')
    data = torch.load(errors_path, map_location='cpu', weights_only=True)
    variance = layer['source_variance']
    errors = torch.stack((data['e1_MSE'] / variance, data['e2_MSE'] / variance,
        torch.zeros_like(data['e1_MSE']), torch.zeros_like(data['e1_MSE'])), dim=1)
    paths = [Path(__file__), c.HERE / 'common.py', c.HERE / 'codec.py', c.HERE / 'length_codec_probe.py',
        root / 'prepared/prepared.json', errors_path, diagnostic_path]
    record = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv],
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]),
        source_sha256={str(path.resolve()): c.sha(path) for path in paths},
        started_unix=time.time(), used_cuda=False, cpu_threads=4, deployment_model=False,
        model_boundary='Actual frozen BF16 stages1/2; hypothetical stages3/4 have zero error. Only actual label frames are written.')
    output.mkdir(parents=True)
    c.write(output / 'report.json', record)
    try:
        _, result = iterate(errors, layer['budget'], output / 'ideal_zero_tail',
            original_elements=math.prod(layer['geometry']['original_shape']), actual_stages=2, ideal_stages=[3, 4])
        expected_m = [3008917, 3041863, 3042143, 3042144]
        expected_z = [187350, 185666, 185659, 185661]
        expected_e = [.07693826130372873, .0743240976796646, .07430217879741507, .07430210052393818]
        c.require([row['max_code_rows'] for row in result['rounds']] == expected_m
            and [row['length_frame']['compressed_total_actual_bytes'] for row in result['rounds']] == expected_z
            and all(abs(row['NMSE'] - ref) < 1e-12 for row, ref in zip(result['rounds'], expected_e, strict=True)),
            'Independent Astra CPU recomputation disagrees with actual artifacts.')
        for filename, expected in record['source_sha256'].items():
            c.require(c.sha(filename) == expected, 'Bound source/input changed: ' + filename)
        record.update(status='completed', exit_code=0, source_unchanged=True, independent_reference_matches=True,
            fixed2_NMSE=float(errors[:, 1].mean()), result=result,
            ideal_point_better_than_fixed2=result['chosen']['NMSE'] < float(errors[:, 1].mean()))
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        c.write(output / 'report.json', record)
    print(dict(status=record['status'], chosen=result['chosen'], fixed2_NMSE=record['fixed2_NMSE']), flush=True)


if __name__ == '__main__':
    main()
