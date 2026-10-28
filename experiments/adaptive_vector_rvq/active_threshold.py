"""One L0 active-only stage2 fit at the predeclared common residual threshold."""
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
from .training import fit_stage
from experiments.night_sequence_20261026.queue_wait import free_gpu, process_start


def stage_digest(stage):
    return c.digest({'codes': stage['codes'], **stage['decoder_state']})


def vector_errors(target, prefix):
    errors = torch.empty(len(target), dtype=torch.float64)
    for start in range(0, len(target), 65536):
        residual = target[start:start + 65536].double() - prefix[start:start + 65536].double()
        errors[start:start + len(residual)] = residual.square().mean(-1)
    c.require(bool(torch.isfinite(errors).all()), 'Nonfinite actual BF16 deployed-prefix errors.')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--fixed2', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available()
        and torch.cuda.device_count() == 1, 'Explicitly assign the currently free physical GPU0.')
    c.require(not args.output_dir.exists(), 'Use a new directory for this one fixed-threshold stage2 fit.')
    resource = free_gpu(0)
    prepared = c.check_prepared(args.prepared_dir.resolve())
    row = prepared['layers']['0']
    diagnostic_path = args.diagnostic_dir / 'report.json'
    diagnostic = c.read(diagnostic_path)
    threshold = next(item for item in diagnostic['common_threshold']['rows']
        if item['label'] == 'maximum_tau_with_enough_possible_stage4_rows')
    tau = threshold['tau_MSE']
    c.require(tau == .00011472430926207956 and threshold['tau_normalized'] == .1629552020352226,
        'Use the declared full-precision budget-derived threshold, without searching or rounding.')
    diagnostic_errors_path = Path(diagnostic['per_vector']['path'])
    c.require(diagnostic['status'] == 'completed' and diagnostic['target_digest'] == row['target_digest']
        and c.sha(args.fixed2) == diagnostic['source_sha256'][str(args.fixed2.resolve())]
        and c.sha(diagnostic_errors_path) == diagnostic['per_vector']['sha256'], 'The bound real-matrix and fixed2 source identities changed.')
    data = torch.load(row['path'], map_location='cpu', weights_only=True, mmap=True)
    source = codec.load(args.fixed2)
    control_report_path = args.fixed2.parent / 'report.json'
    control_report = c.read(control_report_path)
    c.require(source['mode'] == 'fixed2' and source['geometry'] == data['geometry']
        and c.digest(source['fixed']) == row['fixed_digest'] and control_report['freshdecode']['fixed2']['exact'], 'The shared frozen stage1 and protection/rotation source must be the completed real L0 fixed2.')
    c.require(all(value.dtype == torch.bfloat16 for value in source['stages'][0]['decoder_state'].values()), 'Reuse the actual saved BF16 shared stage1.')
    frozen_source = c.digest(codec.tensor_inventory(source))
    frozen_stage1 = stage_digest(source['stages'][0])
    started = time.time()
    args.output_dir.mkdir(parents=True)
    paths = [Path(__file__), c.HERE / 'common.py', c.HERE / 'codec.py', c.HERE / 'training.py',
        args.prepared_dir / 'prepared.json', diagnostic_path, diagnostic_errors_path, args.fixed2, control_report_path]
    report = dict(status='running', scope='L0 active-only stage2 diagnostic; not a completed four-stage model',
        pid=os.getpid(), start_ticks=process_start(os.getpid()), started_unix=started,
        command=[sys.executable, *sys.argv], resource=resource,
        source_sha256={str(path.resolve()): c.sha(path) for path in paths},
        shared_stage1_digest=frozen_stage1, target_digest=row['target_digest'],
        tau_MSE=tau, tau_normalized=threshold['tau_normalized'],
        protocol=dict(steps=6000, scheduler='linear', total_scheduler_steps=6000, warmup_steps=0,
            seed=1031, batch_size=8192, lr=.003, native_arguments=vars(c.arguments()),
            sampling='Independent seed1031 randperm epochs, including the original short tail batch',
            normalization='Computed only on active stage1 residuals and fused once into the saved decoder',
            gate='Same strict actual cumulative residual MSE > tau_MSE at every stage; no quota or learned gate'),
        model_complete=False, later_stages_trained=False)
    c.write(args.output_dir / 'report.json', report)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        target = data['vectors']
        n, variance = len(target), data['source_variance']
        prefix = torch.zeros_like(target, dtype=torch.bfloat16)
        codec.add_stage(prefix, source['stages'][0], torch.arange(n), 'cuda:0')
        e1 = vector_errors(target, prefix)
        old_errors = torch.load(diagnostic_errors_path, map_location='cpu', weights_only=True)
        c.require(torch.equal(e1, old_errors['e1_MSE']), 'Actual shared-stage1 BF16 prefix differs from the source-bound diagnostic.')
        active2_mask = e1 > tau
        active2 = torch.nonzero(active2_mask, as_tuple=False).flatten()
        c.require(active2.numel() == threshold['active2'] and active2.numel() > 0, 'The strict full-precision active2 gate changed.')
        lengths = torch.ones(n, dtype=torch.uint8)
        lengths[active2] = 2
        packed = codec.pack_lengths(lengths)
        c.require(torch.equal(codec.unpack_lengths(packed, n), lengths), 'Actual threshold-derived length labels changed during packing.')
        report['prefix_gate_contract'] = dict(passed=True, actual_shared_prefix_exact=True,
            stage1_NMSE=float(e1.mean() / variance), active2=active2.numel(), stop1=n - active2.numel(),
            stop1_NMSE_floor=float(e1[~active2_mask].sum() / n / variance), packed_length_bytes=packed.numel())
        c.write(args.output_dir / 'report.json', report)
        print(dict(event='active_stage2_start', steps=6000, active2=active2.numel(), stop1=n - active2.numel(),
            tau_MSE=tau, seed=1031, batch=8192), flush=True)
        stage2, following, fit = fit_stage(target, prefix, active2, steps=6000, seed=1031, device='cuda:0',
            output=args.output_dir / 'stage2', source_variance=variance)
        c.require(fit['initial_model_digest'] == control_report['fixed2']['initial_model_digest'], 'Active-only and control stage2 initial parameters differ.')
        c.require(stage_digest(source['stages'][0]) == frozen_stage1
            and c.digest(codec.tensor_inventory(source)) == frozen_source, 'Training changed the shared frozen source stage1, source control stage2, or fixed payload.')
        stage_path = args.output_dir / 'stage2.pt'
        torch.save(stage2, stage_path)
        length_path = args.output_dir / 'stage2_lengths.bin'
        with length_path.open('xb') as stream:
            stream.write(memoryview(packed.numpy()))
        fresh_stage = torch.load(stage_path, map_location='cpu', weights_only=True)
        fresh_lengths = codec.unpack_lengths(torch.frombuffer(bytearray(length_path.read_bytes()), dtype=torch.uint8), n)
        fresh_active = torch.nonzero(fresh_lengths >= 2, as_tuple=False).flatten()
        reconstructed = prefix.clone()
        codec.add_stage(reconstructed, fresh_stage, fresh_active, 'cuda:0')
        c.require(stage_digest(fresh_stage) == stage_digest(stage2) and torch.equal(fresh_active, active2)
            and torch.equal(reconstructed, following) and torch.equal(reconstructed[~active2_mask], prefix[~active2_mask]), 'Saved active-stage2 fields, fresh deployed prefix, or skipped-stage semantics changed.')
        e2 = vector_errors(target, reconstructed)
        active3_mask = active2_mask & (e2 > tau)
        n2, n3 = active2.numel(), int(active3_mask.sum())
        stop1 = float(e1[~active2_mask].sum() / n / variance)
        stop2 = float(e2[active2_mask & ~active3_mask].sum() / n / variance)
        floor = stop1 + stop2
        control_nmse = control_report['outcomes']['fixed2']['normalized_whole_MSE']
        excluded = floor >= control_nmse
        budget = row['budget']
        base4 = budget['header_bytes'] + budget['fixed_bytes'] + 4 * budget['decoder_bytes'] + budget['count_bytes'] + n * (c.BITS // 8)
        errors_path = args.output_dir / 'per_vector_errors.pt'
        torch.save(dict(e1_MSE=e1, e2_active_stage2_MSE=e2, delta2_MSE=e1 - e2,
            source_variance=variance, target_digest=row['target_digest'], tau_MSE=tau,
            order='Original prepared vector index; inactive stage2 rows equal the shared stage1 prefix'), errors_path)
        report.update(status='completed', training=fit, freeze=dict(passed=True, shared_stage1_unchanged=True, source_fixed2_unchanged=True),
            freshdecode=dict(passed=True, stage2_state_exact=True, complete_prefix_exact=True, inactive_stage2_skipped_exact=True),
            active_counts=dict(stage1=n, stage2=n2, potential_stage3=n3, stop1=n - n2, stop2=n2 - n3),
            errors=dict(stage2_whole_MSE=float(e2.mean()), stage2_whole_NMSE=float(e2.mean() / variance),
                fixed2_control_NMSE=control_nmse, stop1_NMSE_floor=stop1, stop2_NMSE_floor=stop2,
                stop1_plus_stop2_NMSE_floor=floor, frozen_exit_floor_excess_over_control=floor - control_nmse),
            budget=dict(control_actual_bytes=budget['control_bytes'], reserved_decoder_stages=4,
                length_bytes=budget['count_bytes'], mandatory_four_decoder_bytes=base4,
                available_extra_code_rows=budget['extra_code_rows'], stage2_code_rows=n2,
                potential_stage3_code_rows=n3, stage4_maximum_possible_rows=n3,
                remaining_rows_after_stage3=budget['extra_code_rows'] - n2 - n3,
                stage3_minimum_complete_bytes=base4 + 6 * (n2 + n3),
                conservative_stage4_complete_bytes=base4 + 6 * (n2 + 2 * n3),
                stage3_minimum_feasible=n2 + n3 <= budget['extra_code_rows'],
                all_possible_stage4_rows_feasible=n2 + 2 * n3 <= budget['extra_code_rows'],
                diagnostic_partial_payload=True, actual_deployed_model_written=False),
            mathematical_early_exclusion=excluded,
            decision='Stop this candidate: frozen stop1+stop2 contributions already exceed the complete fixed2 control error, even if future active3/4 errors were zero.' if excluded
                else 'Stage2 diagnostic completed; later stages require the root decision and real budget checks. No later stages are started by this entry.',
            artifacts=dict(stage2=dict(path=str(stage_path), sha256=c.sha(stage_path), bytes=stage_path.stat().st_size),
                lengths=dict(path=str(length_path), sha256=c.sha(length_path), bytes=length_path.stat().st_size),
                errors=dict(path=str(errors_path), sha256=c.sha(errors_path), bytes=errors_path.stat().st_size)), allfinite=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started)
        c.write(args.output_dir / 'report.json', report)
    print({key: report[key] for key in ('status', 'active_counts', 'errors', 'budget', 'mathematical_early_exclusion', 'decision')}, flush=True)


if __name__ == '__main__':
    main()
