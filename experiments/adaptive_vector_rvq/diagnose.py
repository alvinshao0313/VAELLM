"""Read-only L0 common-threshold diagnostics from the completed fixed2 codec."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from . import codec
from . import common as c
from experiments.night_sequence_20261026.queue_wait import free_gpu, process_start


def vector_mse(target, prefix):
    errors = torch.empty(len(target), dtype=torch.float64)
    for start in range(0, len(target), 65536):
        residual = target[start:start + 65536].double() - prefix[start:start + 65536].double()
        errors[start:start + len(residual)] = residual.square().mean(-1)
    c.require(bool(torch.isfinite(errors).all()) and bool((errors >= 0).all()), 'Invalid actual per-vector residual errors.')
    return errors


def quantiles(values, variance):
    levels = torch.tensor([0., .001, .01, .05, .25, .5, .75, .95, .99, .999, 1.], dtype=torch.float64)
    actual = torch.quantile(values, levels)
    return [dict(quantile=float(q), MSE=float(value), normalized=float(value / variance))
        for q, value in zip(levels, actual)]


def threshold_table(e1, e2, variance, budget):
    n, extra = len(e1), budget['extra_code_rows']
    row_bytes = c.BITS // 8
    base = budget['header_bytes'] + budget['fixed_bytes'] + 4 * budget['decoder_bytes'] + budget['count_bytes'] + n * row_bytes
    s1 = e1.sort().values
    # A third stage is reached only if both existing cumulative errors exceed tau.
    s3 = torch.minimum(e1, e2).sort().values
    events = torch.unique(torch.cat((s1, s3)), sorted=True)

    def counts(tau):
        return n - int(torch.searchsorted(s1, tau, right=True)), n - int(torch.searchsorted(s3, tau, right=True))

    def boundary(weight):
        low, high = 0, len(events)
        while low < high:
            middle = (low + high) // 2
            n2, n3 = counts(events[middle])
            # First event whose decreasing cost falls below the requested budget.
            cost = n2 + weight * n3
            below = cost <= extra if weight == 1 else cost < extra
            if below:
                high = middle
            else:
                low = middle + 1
        return low if weight == 1 else low - 1

    def describe(label, tau):
        tau = float(tau)
        active2 = e1 > tau
        active3 = active2 & (e2 > tau)
        n2, n3 = int(active2.sum()), int(active3.sum())
        c.require(counts(torch.tensor(tau, dtype=e1.dtype)) == (n2, n3), 'Sorted strict-threshold counts disagree with actual nested masks.')
        remaining = extra - n2 - n3
        feasible = 0 <= remaining <= n3
        floor1 = float(e1[~active2].sum() / n)
        floor12 = floor1 + float(e2[active2 & ~active3].sum() / n)
        gated = floor1 + float(e2[active2].sum() / n)
        return dict(label=label, tau_MSE=tau, tau_normalized=tau / variance,
            active2=n2, active2_fraction=n2 / n, active3_proxy=n3, active3_fraction=n3 / n,
            code_rows_through_stage3=n + n2 + n3, extra_rows_used=n2 + n3,
            remaining_stage4_rows=remaining, maximum_possible_stage4_rows=n3,
            bytes_through_stage3_charging_four_decoders=base + row_bytes * (n2 + n3),
            remaining_budget_bytes=budget['control_bytes'] - base - row_bytes * (n2 + n3),
            budget_can_be_filled_by_some_nested_stage4=feasible,
            hypothetical_lengths_if_remaining_rows_used=[n - n2, n2 - n3, n3 - remaining, remaining] if feasible else None,
            stop1_MSE_floor=floor1, stop1_NMSE_floor=floor1 / variance,
            stopped1_fixed2_MSE=float(e2[~active2].sum() / n),
            foregone_stage2_NMSE_on_stopped1=float((e1[~active2] - e2[~active2]).sum() / n / variance),
            stop1_plus_stop2_NMSE_floor=floor12 / variance,
            gated_existing_stage2_whole_MSE=gated, gated_existing_stage2_whole_NMSE=gated / variance,
            stop1_floor_exceeds_fixed2_total=bool(floor1 > float(e2.mean())))

    lower, upper = boundary(1), boundary(2)
    c.require(0 <= lower < len(events) and 0 <= upper < len(events), 'No finite threshold budget boundaries in actual errors.')
    n2, n3 = counts(events[lower])
    c.require(n2 + n3 <= extra, 'Minimum threshold still overruns the stage23 budget.')
    n2, n3 = counts(events[upper])
    c.require(n2 + 2 * n3 >= extra, 'Maximum threshold cannot possibly fill the nested stage4 budget.')
    rows = [describe('minimum_tau_without_stage23_budget_overrun', events[lower]),
        describe('maximum_tau_with_enough_possible_stage4_rows', events[upper])]
    if lower:
        rows.append(describe('event_immediately_below_minimum_tau', events[lower - 1]))
    if upper + 1 < len(events):
        rows.append(describe('event_immediately_above_maximum_tau', events[upper + 1]))
    for level in (.01, .05, .1, .25, .5, .75, .9, .95, .99):
        rows.append(describe(f'e1_quantile_{level:g}', torch.quantile(e1, level)))
    for level in (.5, .9, .99, .999):
        rows.append(describe(f'e2_quantile_{level:g}', torch.quantile(e2, level)))
    original_count = budget['active_counts'][1]
    rows.append(describe('original_geometric_stage2_rank_threshold', s1[n - original_count]))
    return dict(extra_code_rows=extra, extra_code_rows_per_vector=extra / n,
        fixed2_bytes=budget['control_bytes'], adaptive_mandatory_bytes=base,
        code_row_bytes=row_bytes, charged_decoder_bytes=4 * budget['decoder_bytes'],
        count_bytes=budget['count_bytes'], budget_feasible_threshold_interval_nonempty=lower <= upper,
        strict_comparison='active2 = e1 > tau; active3_proxy = e1 > tau AND e2 > tau',
        rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--fixed2', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    c.require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Assign the currently free physical GPU0 explicitly.')
    c.require(not args.output_dir.exists(), 'Write this read-only diagnostic once to a new directory.')
    resource = free_gpu(0)
    prepared = c.check_prepared(args.prepared_dir.resolve())
    layer = prepared['layers']['0']
    data = torch.load(layer['path'], map_location='cpu', weights_only=True, mmap=True)
    bundle = codec.load(args.fixed2)
    c.require(bundle['mode'] == 'fixed2' and bundle['geometry'] == data['geometry']
        and bundle['n_vectors'] == layer['vector_count'] and c.digest(bundle['fixed']) == layer['fixed_digest'], 'Diagnostic codec and real prepared L0 source differ.')
    c.require(data['vectors'].dtype == torch.float32 and tuple(data['vectors'].shape) == (layer['vector_count'], c.DIM), 'Use the real complete fresh-HF FP32 rotated vectors.')
    c.require(all(value.dtype == torch.bfloat16 for stage in bundle['stages'] for value in stage['decoder_state'].values()), 'Use the completed BF16 deployed stage states.')
    c.require(codec.ledger(bundle)['total_bytes'] == layer['budget']['control_bytes'], 'Actual fixed2 storage differs from the predetermined complete budget.')
    source_report_path = args.fixed2.parent / 'report.json'
    source_report = c.read(source_report_path)
    c.require(source_report['layer'] == 0 and source_report['allfinite'] and source_report['freshdecode']['fixed2']['exact'], 'The completed fixed2 freshdecode evidence is required.')
    inventory_before = c.digest(codec.tensor_inventory(bundle))
    args.output_dir.mkdir(parents=True)
    started = time.time()
    paths = [Path(__file__), c.HERE / 'common.py', c.HERE / 'codec.py', c.ROOT / 'litebsq/autoencoder.py',
        c.ROOT / 'litebsq/bitpack.py', c.ROOT / 'litebsq/parallel_layers.py',
        args.prepared_dir / 'prepared.json', args.fixed2, source_report_path]
    report = dict(status='running', pid=os.getpid(), start_ticks=process_start(os.getpid()), started_unix=started,
        command=[sys.executable, *sys.argv], resource=resource,
        source_sha256={str(path.resolve()): c.sha(path) for path in paths},
        input=c.stat(layer['path']), target_digest=layer['target_digest'], geometry=data['geometry'],
        source_variance=data['source_variance'], n_vectors=layer['vector_count'], dtype='BF16 decoder and sequential BF16 prefix addition; original target FP32; error reduction FP64')
    c.write(args.output_dir / 'report.json', report)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        active = torch.arange(bundle['n_vectors'])
        prefix = torch.zeros_like(data['vectors'], dtype=torch.bfloat16)
        codec.add_stage(prefix, bundle['stages'][0], active, 'cuda:0')
        e1 = vector_mse(data['vectors'], prefix)
        codec.add_stage(prefix, bundle['stages'][1], active, 'cuda:0')
        e2 = vector_mse(data['vectors'], prefix)
        variance = data['source_variance']
        for actual, expected in ((float(e1.mean()), source_report['shared_stage1']['observations'][-1]['whole_MSE']),
            (float(e2.mean()), source_report['outcomes']['fixed2']['whole_MSE'])):
            c.require(abs(actual - expected) <= 1e-16 + 1e-10 * abs(expected), 'Actual BF16 prefix MSE disagrees with completed production observations.')
        delta = (e1 - e2).contiguous()
        errors_path = args.output_dir / 'per_vector_errors.pt'
        torch.save(dict(e1_MSE=e1, e2_MSE=e2, delta2_MSE=delta, source_variance=variance,
            target_digest=layer['target_digest'], order='Original prepared vector row index',
            definition='delta2=e1-e2; cumulative actual persisted BF16 prefixes; normalized errors divide by fixed original source_variance'), errors_path)
        report.update(global_MSE=dict(stage1=float(e1.mean()), fixed2=float(e2.mean()), delta2=float(delta.mean())),
            global_NMSE=dict(stage1=float(e1.mean() / variance), fixed2=float(e2.mean() / variance), delta2=float(delta.mean() / variance)),
            fixed2_original_coordinate_and_local_output={key: value for key, value in source_report['outcomes']['fixed2'].items()
                if key.startswith(('original_BF16_', 'local_output_')) or key in ('observation_tokens', 'observation')},
            quantiles=dict(e1=quantiles(e1, variance), e2=quantiles(e2, variance), delta2=quantiles(delta, variance)),
            stage2_gain=dict(positive_count=int((delta > 0).sum()), zero_count=int((delta == 0).sum()), negative_count=int((delta < 0).sum()),
                positive_fraction=float((delta > 0).double().mean()), negative_gain_NMSE_sum_per_vector=float(delta[delta < 0].sum() / len(delta) / variance)),
            common_threshold=threshold_table(e1, e2, variance, layer['budget']),
            per_vector=dict(path=str(errors_path), sha256=c.sha(errors_path), bytes=errors_path.stat().st_size),
            source_unchanged=inventory_before == c.digest(codec.tensor_inventory(bundle)),
            boundary='Read-only fixed2 full-population stage2 proxy. No stage3 was trained or decoded. Stage4 residual capacity is accounting only, not a guaranteed common-tau allocation. Stop1/stop2 floors hold only with these prefixes and irreversible exits frozen; no downstream inference.')
        c.require(report['source_unchanged'], 'Diagnostic modified existing frozen codec state.')
        report['status'] = 'completed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started)
        c.write(args.output_dir / 'report.json', report)
    print(dict(status=report['status'], global_NMSE=report['global_NMSE'], stage2_gain=report['stage2_gain'],
        budget_boundaries=report['common_threshold']['rows'][:2], per_vector=report['per_vector']), flush=True)


if __name__ == '__main__':
    main()
