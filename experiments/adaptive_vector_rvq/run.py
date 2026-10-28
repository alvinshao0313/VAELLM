"""Fixed full-vector L0/18/35 residual-VAE pair; the caller assigns the GPU."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
import traceback

import torch

from . import common as c
from . import codec
from .budget import select_active
from .observe import compare
from .training import fit_stage, mse


def stage_digest(stage):
    return c.digest({'codes': stage['codes'], **stage['decoder_state']})


def run_layer(layer, prepared, output, steps, device):
    row = prepared['layers'][str(layer)]
    data = torch.load(row['path'], map_location='cpu', weights_only=True, mmap=True)
    target, fixed, geometry = data['vectors'], data['fixed'], data['geometry']
    n, budget = len(target), row['budget']
    full = torch.arange(n)
    zero = torch.zeros_like(target, dtype=torch.bfloat16)
    shared, shared_prefix, shared_report = fit_stage(target, zero, full, steps=steps, seed=c.SEED,
        device=device, output=output / 'shared_stage1', source_variance=data['source_variance'])
    frozen_shared = stage_digest(shared)
    stage2_seed = c.SEED + 1000
    control2, control_prefix, control_report = fit_stage(target, shared_prefix, full, steps=3 * steps, seed=stage2_seed,
        device=device, output=output / 'fixed2', source_variance=data['source_variance'])
    c.require(stage_digest(shared) == frozen_shared, 'Control stage2 modified the shared frozen stage1.')
    control = dict(mode='fixed2', n_vectors=n, geometry=geometry, fixed=fixed, stages=[shared, control2])
    adaptive_stages, prefix, previous = [shared], shared_prefix.clone(), full
    lengths = torch.ones(n, dtype=torch.uint8)
    gates, reports, frozen_prefixes = [], [], []
    for index, count in enumerate(budget['active_counts'][1:], start=1):
        active, gate = select_active(target, prefix, previous, count, data['source_variance'])
        before = [stage_digest(stage) for stage in adaptive_stages]
        stage, following, report = fit_stage(target, prefix, active, steps=steps, seed=c.SEED + 1000 * index,
            device=device, output=output / f'adaptive_stage{index + 1}', source_variance=data['source_variance'])
        after = [stage_digest(stage) for stage in adaptive_stages]
        c.require(before == after and frozen_shared == stage_digest(shared), 'Later adaptive training changed a frozen decoder or persistent code prefix.')
        worsened = 0
        for start in range(0, active.numel(), 65536):
            ids = active[start:start + 65536]
            old = (target[ids].double() - prefix[ids].double()).square().mean(-1)
            new = (target[ids].double() - following[ids].double()).square().mean(-1)
            worsened += int((new > old).sum())
        gate.update(stage=index + 1, worsened_fraction=worsened / active.numel())
        adaptive_stages.append(stage)
        lengths[active] = index + 1
        gates.append(gate)
        reports.append(report)
        frozen_prefixes.append(dict(stage=index + 1, before=before, after=after, unchanged=True))
        prefix, previous = following, active
    c.require(control_report['initial_model_digest'] == reports[0]['initial_model_digest'], 'Paired stage2 initializations must be identical before their different active data/statistics.')
    adaptive = dict(mode='adaptive', n_vectors=n, geometry=geometry, fixed=fixed,
        stages=adaptive_stages, lengths=codec.pack_lengths(lengths))
    c.require(c.digest(control['fixed']) == c.digest(adaptive['fixed']) == row['fixed_digest'], 'Shared protection/sign payload changed.')
    details, fresh_proofs = {}, {}
    for name, bundle, expected in (('fixed2', control, control_prefix), ('adaptive', adaptive, prefix)):
        path = output / (name + '.avrvq')
        ledger = codec.save(bundle, path)
        fresh = codec.load(path)
        c.require(c.digest(codec.tensor_inventory(fresh)) == c.digest(codec.tensor_inventory(bundle)), 'Fresh codec fields differ.')
        decoded = codec.decode(fresh, device)
        c.require(torch.equal(decoded, expected), 'Fresh actual deployed prefix reconstruction changed.')
        expected_bytes = budget['control_bytes' if name == 'fixed2' else 'adaptive_bytes']
        c.require(ledger['total_bytes'] == expected_bytes, 'Actual full saved codec bytes differ from the predetermined equal-bit allocation.')
        details[name] = dict(path=str(path), ledger=ledger, whole_MSE=mse(target, decoded),
            normalized_whole_MSE=mse(target, decoded) / data['source_variance'], **compare(fresh, decoded, prepared['activations'], device))
        fresh_proofs[name] = dict(exact=True, persistent_tensor_count=len(codec.tensor_inventory(fresh)), actual_file_bytes=path.stat().st_size)
    result = dict(layer=layer, budget=budget, outcomes=details, gates=gates, shared_stage1=shared_report,
        fixed2=control_report, adaptive=reports, freshdecode=fresh_proofs,
        freeze=dict(passed=True, stage1_shared_exact=True, previous_stages=frozen_prefixes),
        allfinite=True, fixed_primary_steps=3 * steps, adaptive_steps=[steps] * 3,
        comparison_boundary='Each arm uses4*base updates including the common stage1; actual shared-stage work is7*base per matrix. Fixed2 has one linear3*base schedule; adaptive resets three linearbase schedules. Original randperm epochs keep short tail batches; actual sampled vectors are reported, so FLOPs matching is approximate.')
    c.write(output / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--layers', choices=['0', '18,35', '0,18,35'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    c.require(args.steps in (2, 2000) and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Use the fixed2-update full-shape validation or2000-update pair on one explicitly assigned GPU.')
    c.require(not args.output_dir.exists(), 'Run once into a new output directory; no automatic retry/resume.')
    prepared = c.check_prepared(args.prepared_dir.resolve())
    args.output_dir.mkdir(parents=True)
    report = dict(status='running', steps=args.steps, layers={}, started_unix=time.time(),
        budget=dict(passed=False), freshdecode=dict(passed=False), freeze=dict(passed=False), allfinite=False)
    c.write(args.output_dir / 'report.json', report)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for layer in map(int, args.layers.split(',')):
            output = args.output_dir / f'layer_{layer}'
            output.mkdir()
            report['layers'][str(layer)] = run_layer(layer, prepared, output, args.steps, 'cuda:0')
            c.write(args.output_dir / 'report.json', report)
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status='completed', budget=dict(passed=True), freshdecode=dict(passed=True), freeze=dict(passed=True), allfinite=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['finished_unix'] = time.time()
        c.write(args.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
