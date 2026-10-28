"""CPU lower bound for any frozen-stage1/2 prefix bank, with ideal zero-error tails."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from . import codec
from . import common as c


@torch.no_grad()
def allocate(errors, max_code_rows):
    c.require(errors.device.type == 'cpu' and errors.dtype == torch.float64
        and errors.ndim == 2 and errors.shape[1] == 4 and bool(torch.isfinite(errors).all())
        and bool((errors >= 0).all()), 'Four finite nonnegative actual/ideal FP64 prefix errors are required.')
    n = len(errors)
    c.require(n <= max_code_rows <= 4 * n, 'Invalid complete-model prefix-code budget.')
    costs = torch.arange(1, 5, dtype=torch.float64)
    trace = []
    best_dual = -math.inf

    def evaluate(value, label):
        nonlocal best_dual
        minima, selected = (errors + value * costs).min(dim=1)
        lengths = selected + 1  # First minimum gives the shorter-prefix tie break.
        rows = int(lengths.sum())
        primal = float(errors.gather(1, selected[:, None]).sum())
        dual = float(minima.sum()) - value * max_code_rows
        best_dual = max(best_dual, dual)
        trace.append(dict(label=label, lambda_value=value, code_rows=rows, feasible=rows <= max_code_rows,
            primal_NMSE=primal / n, dual_NMSE=dual / n))
        return lengths.to(torch.uint8), rows, primal

    lengths, rows, primal = evaluate(0., 'zero_lambda')
    if rows <= max_code_rows:
        return lengths, dict(lambda_value=0., code_rows=rows, primal_sum=primal,
            dual_sum=best_dual, trace=trace, bisection_iterations=0)
    high = max(0., float(((errors[:, :1] - errors[:, 1:]) / costs[1:].sub(1)).max()))
    high += max(1., abs(high)) * 16 * torch.finfo(torch.float64).eps
    _, rows, _ = evaluate(high, 'all_stage1_bracket')
    c.require(rows == n, 'The upper lambda bracket must choose stage1 for every vector.')
    low = 0.
    for iteration in range(60):
        middle = (low + high) / 2
        _, rows, _ = evaluate(middle, f'bisection_{iteration + 1}')
        if rows > max_code_rows:
            low = middle
        else:
            high = middle
    lengths, rows, primal = evaluate(high, 'final_feasible_high')
    c.require(rows <= max_code_rows and primal + n * 1e-12 >= best_dual, 'Feasibility or numerical weak-duality contract failed.')
    return lengths, dict(lambda_value=high, code_rows=rows, primal_sum=primal,
        dual_sum=best_dual, trace=trace, bisection_iterations=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--fixed2', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'This bound needs only CPU; explicitly hide CUDA.')
    c.require(not args.output_dir.exists(), 'Keep the ideal-bound report in a new independent output directory.')
    torch.set_num_threads(4)
    prepared = c.check_prepared(args.prepared_dir.resolve())
    layer = prepared['layers']['0']
    diagnostic_path = args.diagnostic_dir / 'report.json'
    diagnostic = c.read(diagnostic_path)
    errors_path = Path(diagnostic['per_vector']['path'])
    c.require(diagnostic['status'] == 'completed' and diagnostic['source_unchanged']
        and diagnostic['target_digest'] == layer['target_digest']
        and c.sha(errors_path) == diagnostic['per_vector']['sha256']
        and c.sha(args.fixed2) == diagnostic['source_sha256'][str(args.fixed2.resolve())], 'Use the completed actual BF16 source-bound e1/e2 diagnostic.')
    source = codec.load(args.fixed2)
    c.require(source['mode'] == 'fixed2' and source['geometry'] == layer['geometry']
        and c.digest(source['fixed']) == layer['fixed_digest']
        and codec.ledger(source)['total_bytes'] == layer['budget']['control_bytes'], 'Frozen source or complete fixed2 accounting changed.')
    data = torch.load(errors_path, map_location='cpu', weights_only=True)
    c.require(data['target_digest'] == layer['target_digest'] and data['source_variance'] == layer['source_variance']
        and data['e1_MSE'].dtype == data['e2_MSE'].dtype == torch.float64
        and data['e1_MSE'].shape == data['e2_MSE'].shape == (layer['vector_count'],), 'Actual per-vector source/dtype/shape changed.')
    frozen_hashes = [c.digest({'codes': stage['codes'], **stage['decoder_state']}) for stage in source['stages']]
    started = time.time()
    args.output_dir.mkdir(parents=True)
    paths = [Path(__file__), c.HERE / 'common.py', c.HERE / 'codec.py',
        args.prepared_dir / 'prepared.json', diagnostic_path, errors_path, args.fixed2]
    report = dict(status='running', pid=os.getpid(), command=[sys.executable, *sys.argv],
        started_unix=started, source_sha256={str(path.resolve()): c.sha(path) for path in paths},
        source_stage_hashes=frozen_hashes, target_digest=layer['target_digest'], cpu_threads=4, used_cuda=False,
        model_boundary='Existing frozen actual BF16 stages1/2. Hypothetical stages3/4 have exactly zero error; they are an optimistic mathematical relaxation, not trained/decoded models.')
    c.write(args.output_dir / 'report.json', report)
    try:
        variance = layer['source_variance']
        e1, e2 = data['e1_MSE'] / variance, data['e2_MSE'] / variance
        errors = torch.stack((e1, e2, torch.zeros_like(e1), torch.zeros_like(e1)), dim=1)
        budget = layer['budget']
        max_rows = layer['vector_count'] + budget['extra_code_rows']
        lengths, solution = allocate(errors, max_rows)
        n = len(lengths)
        counts = [int((lengths == stage).sum()) for stage in range(1, 5)]
        c.require(sum(counts) == n and sum(stage * count for stage, count in enumerate(counts, 1)) == solution['code_rows'], 'Length counts and actual code rows disagree.')
        packed = codec.pack_lengths(lengths)
        c.require(torch.equal(codec.unpack_lengths(packed, n), lengths), 'Ideal length-label packing changed the actual allocation.')
        packed_path = args.output_dir / 'ideal_lengths.bin'
        with packed_path.open('xb') as stream:
            stream.write(memoryview(packed.numpy()))
        primal, dual = solution['primal_sum'] / n, solution['dual_sum'] / n
        fixed2 = float(e2.mean())
        reference = dict(primal_NMSE=.07693826130372873, dual_NMSE=.07693826130372894,
            length_counts=[160799, 1289773, 89524, 0], origin='Independent Astra CPU recomputation on the same stored e1/e2; validation only, never used by the allocator.')
        agreement = counts == reference['length_counts'] and abs(primal - reference['primal_NMSE']) < 1e-12 and abs(dual - reference['dual_NMSE']) < 1e-12
        report.update(status='completed', source_stage_hashes_unchanged=frozen_hashes == [c.digest({'codes': stage['codes'], **stage['decoder_state']}) for stage in source['stages']],
            budget=dict(fixed2_actual_bytes=budget['control_bytes'], charged_decoders=4,
                decoder_bytes=4 * budget['decoder_bytes'], length_bytes=budget['count_bytes'],
                fixed_bytes=budget['fixed_bytes'], header_bytes=budget['header_bytes'], code_row_bytes=c.BITS // 8,
                max_code_rows=max_rows, actual_ideal_code_rows=solution['code_rows'], remaining_code_rows=max_rows - solution['code_rows'],
                actual_ideal_complete_bytes=budget['header_bytes'] + budget['fixed_bytes'] + 4 * budget['decoder_bytes'] + budget['count_bytes'] + solution['code_rows'] * (c.BITS // 8)),
            solution=dict(lambda_value=solution['lambda_value'], length_counts=counts,
                primal_NMSE=primal, dual_lower_bound_NMSE=dual, primal_minus_dual_NMSE=primal - dual,
                numerical_tolerance_NMSE=1e-12, bisection_iterations=solution['bisection_iterations']),
            fixed2_NMSE=fixed2, ideal_dual_excess_NMSE=dual - fixed2, ideal_dual_relative_excess=dual / fixed2 - 1,
            stopping_condition_met=dual - 1e-12 >= fixed2,
            decision='Do not train dense stages3/4: even ideal zero-error tails cannot beat fixed2 with frozen actual stages1/2 and the complete four-decoder/2bit-length budget.' if dual - 1e-12 >= fixed2 else 'Ideal lower bound does not rule out a better fixed bank; actual fitting is not executed by this diagnostic.',
            independent_reference=reference, independent_reference_matches=agreement,
            length_artifact=dict(path=str(packed_path), bytes=packed_path.stat().st_size, sha256=c.sha(packed_path), deployment_model=False),
            lambda_trace=solution['trace'],
            proof='For every lambda>=0, sum_i min_s(d_is+lambda*s)-lambda*M lower-bounds every allocation with sum lengths<=M. Replacing actual stage3/4 errors by zero can only improve that optimum. A lower bound above fixed2 therefore rules out every such fixed-bank prefix allocation; no diminishing-returns assumption or uncertified greedy claim is used.')
        c.require(report['source_stage_hashes_unchanged'] and agreement, 'Source freezing or independent bound cross-check failed.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started)
        c.write(args.output_dir / 'report.json', report)
    print({key: report[key] for key in ('status', 'solution', 'fixed2_NMSE', 'ideal_dual_relative_excess', 'stopping_condition_met', 'decision')}, flush=True)


if __name__ == '__main__':
    main()
