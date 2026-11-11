"""Exactly10K fresh K24 updates per independent M3 group; original linear10K."""
import gc
import time

from . import bindings as b
from experiments.adaptive_grouped_fit_20261029.grouped import IndependentGroupedVAE, fresh_models
from experiments.down_tail23_short_cmcc_20261104 import fit as original
from train_utils.train_args import create_optimizer
from transformers import get_scheduler


def execute(values):
    b.c.require(len(values) == 3, 'Use one consecutive group of3 actual layers.')
    args = b.short.arguments(24)
    b.torch.cuda.empty_cache()
    b.torch.cuda.reset_peak_memory_stats()
    originals = fresh_models(3, device='cuda:0', args=args)
    model = IndependentGroupedVAE(originals).to('cuda:0')
    del originals
    optimizer = create_optimizer(model.parameters(), args, .003)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=10000)
    generators = [b.torch.Generator().manual_seed(2031) for _ in values]
    orders, positions = [None] * 3, [0] * 3
    rows, updates, sampled = [], [], [0] * 3
    started = time.monotonic()
    for step in range(1, 10001):
        data, ids = original.batches(values, generators, orders, positions)
        b.c.require(tuple(data.shape) == (16384, 3, 16), 'Preserve the real approved B16384/M3/D16 batch.')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, losses = model(data.to('cuda:0'))
        b.c.require(bool(b.torch.isfinite(losses).all()), 'Nonfinite independent K24 loss; no retry or fallback.')
        losses.sum().backward()
        if step <= 2 or step % 100 == 0:
            original.gradients_before_update(model)
        before = original.layer_parameters_before_update(model) if step <= 2 else None
        optimizer.step()
        if step <= 2 or step == 10000:
            evidence = original.update_evidence(model, optimizer, step=step, before=before)
            evidence['gradient_finite_checked_before_update'] = True
            updates.append(evidence)
        scheduler.step()
        sampled = [count + len(current) for count, current in zip(sampled, ids, strict=True)]
        if step <= 2 or step % 100 == 0:
            row = dict(step=step, losses=losses.detach().float().tolist(), lr=scheduler.get_last_lr()[0], elapsed_seconds=time.monotonic() - started)
            rows.append(row)
            print(dict(event='full36_independent_M3_K24', names=[value['name'] for value in values], **row), flush=True)
    b.torch.cuda.synchronize()
    fit_seconds = time.monotonic() - started
    stages = [original.export_stage(model.single_model(index), value, device='cuda:0') for index, value in enumerate(values)]
    result = dict(names=[value['name'] for value in values], actual_updates=10000, schedule_steps=10000, warmup_steps=0,
        batch_shape=[16384, 3, 16], init_seed_per_layer=2031, shuffle_seed_per_layer=2031, sampled_vectors=sampled,
        actual_update_checks=updates, rows=rows, actual_fit_seconds=fit_seconds,
        fit_and_full_encoding_seconds=time.monotonic() - started,
        peak_allocated_bytes=b.torch.cuda.max_memory_allocated(), peak_reserved_bytes=b.torch.cuda.max_memory_reserved(),
        peak_scope='Reset immediately before this group fresh initialization; includes fit and full-bank encoding only',
        endpoint='Fixed10000; no checkpoint selection, no short warm-start, no global clip')
    del model, optimizer, scheduler
    gc.collect()
    b.torch.cuda.empty_cache()
    return stages, result
