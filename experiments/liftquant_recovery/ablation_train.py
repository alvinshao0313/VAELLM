"""Matched decoder-only versus joint code/decoder block recovery.

Only optimizer membership differs between arms. Both use the same hard packed
forward, frozen initial sensitivity scales, data order, loss and decoder schedule.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from experiments.liftquant_recovery import all_bits
from experiments.liftquant_recovery.liftquant_optimizer import (
    advance_learning_rates, build_optimizer,
)
from experiments.liftquant_recovery.proxy_coordinates import motion_budget
from experiments.liftquant_recovery.recovery_runtime import (
    block_output, clear_caches, prime_packed_cache, tree_to,
)


def _write_progress(path, record):
    if path is None:
        return
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.no_grad()
def _measure(block, hidden, target, kwargs, batch_size, ntrain, device,
             *, keep_outputs=False):
    # RecoveryLinear must stay on its training hard-decode path. In particular,
    # prime_packed_cache is forbidden until all recovery adapters are detached.
    block.eval()
    per_document, outputs = [], []
    digest = hashlib.sha256()
    for start in range(0, len(hidden), batch_size):
        prediction = block_output(block, hidden[start:start + batch_size].to(device), kwargs)
        truth = target[start:start + batch_size].to(device)
        errors = (prediction.float() - truth.float()).square().flatten(1).mean(1)
        if not torch.isfinite(errors).all():
            raise FloatingPointError('Nonfinite per-document evaluation error.')
        per_document.extend(errors.cpu().tolist())
        value = prediction.detach().cpu().contiguous()
        digest.update(value.view(torch.uint8).numpy().tobytes())
        if keep_outputs:
            outputs.append(value)
    result = dict(
        per_document_mse=per_document,
        train_mse=sum(per_document[:ntrain]) / ntrain,
        heldout_mse=(sum(per_document[ntrain:]) / (len(hidden) - ntrain)
                     if ntrain < len(hidden) else None),
        output_sha256=digest.hexdigest(),
    )
    return result, torch.cat(outputs) if keep_outputs else None


@torch.no_grad()
def _audit(runtime, initial_decoders, update_codes, gradients=False):
    result = {}
    for path, module, scores, params in runtime:
        stats = all_bits.proxy_statistics(module)
        stats['decoder_delta_max'] = max(
            (parameter - initial).abs().max().item()
            for parameter, initial in zip(params, initial_decoders[path])
        )
        if not update_codes:
            storage = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
            if not torch.equal(storage, module._recovery_runtime.initial_packed):
                raise AssertionError(f'{path}: decoder-only arm changed packed codes.')
            if stats['proxy_delta_max'] != 0 or stats['bits_changed_from_initial'] != 0:
                raise AssertionError(f'{path}: decoder-only arm changed frozen proxies.')
        if gradients:
            stats['code_grad_l2'] = (scores.grad.norm().item() if update_codes else None)
            stats['decoder_grad_l2'] = math.sqrt(sum(
                parameter.grad.float().square().sum().item() for parameter in params
            ))
        result[path] = stats
    return result


def run_arm(block, targets, hidden, target, kwargs, args, device, *,
            update_codes: bool, progress_path=None) -> dict:
    """Train one arm in place and return its native, CPU-resident block metrics.

    Required args: steps, batch_size, train_samples, code_lr, decoder_lr.
    hidden/target contain fixed-length documents, training documents first.
    Optimizer state and proxies are discarded; native weights/codes stay in block.
    """
    started = time.monotonic()
    steps, batch_size, ntrain = args.steps, args.batch_size, args.train_samples
    if steps <= 0 or batch_size <= 0 or not 0 < ntrain <= len(hidden):
        raise ValueError('Expected positive steps, batch size and valid training count.')
    if ntrain % batch_size or hidden.shape != target.shape:
        raise ValueError('Training count must be a multiple of batch size; hidden/target shapes must match.')
    if args.code_lr <= 0 or args.decoder_lr <= 0 or not targets:
        raise ValueError('Expected positive learning rates and compressed targets.')
    if len({path for path, _ in targets}) != len(targets):
        raise ValueError('Duplicate compressed target path.')
    block.to(device).eval().requires_grad_(False)
    clear_caches(block)
    kwargs = tree_to(kwargs, device)
    runtime, groups, initial_decoders, geometry = [], [], {}, {}
    owned = {id(module) for module in block.modules()}
    seen_decoder_parameters = set()
    for path, module in targets:
        if id(module) not in owned:
            raise ValueError(f'{path}: target does not belong to the supplied block.')
        scores, params = all_bits.attach(module)
        scores.requires_grad_(update_codes)
        std = scores.detach().std().item()
        code_lr = min(args.code_lr, std / 50)
        geometry[path] = dict(module._recovery_runtime.geometry, proxy_std=std,
                              code_learning_rate=code_lr,
                              actual_budget=motion_budget(
                                  code_lr, steps, module._recovery_runtime.proxy_scale / 2))
        if update_codes:
            groups.append(dict(params=[scores], lr=code_lr, role='code', target=path))
        if seen_decoder_parameters.intersection(map(id, params)):
            raise ValueError('Decoder sharing inside the block requires an explicit ownership policy.')
        seen_decoder_parameters.update(map(id, params))
        groups.append(dict(params=params, lr=args.decoder_lr, role='decoder', target=path))
        initial_decoders[path] = [parameter.detach().clone() for parameter in params]
        runtime.append((path, module, scores, params))
    unexpected = [name for name, parameter in block.named_parameters()
                  if parameter.requires_grad and id(parameter) not in seen_decoder_parameters]
    if unexpected:
        raise ValueError(f'Unexpected trainable parameters: {unexpected}')
    optimizer, schedulers = build_optimizer(groups, steps)
    initial, _ = _measure(block, hidden, target, kwargs, batch_size, ntrain, device)
    record = dict(
        status='TRAINING', arm='joint' if update_codes else 'decoder_only',
        update_codes=update_codes, config=dict(
            optimizer_steps=steps, batch_size=batch_size, train_samples=ntrain,
            heldout_samples=len(hidden) - ntrain, seqlen=hidden.shape[1],
            code_lr_cap=args.code_lr, decoder_lr=args.decoder_lr,
            order='deterministic repeated ascending training-document batches',
            objective='whole-block final-hidden-state unmasked MSE',
            optimizer='pinned LiftQuant AdamW and scheduler step()+get_lr()',
        ),
        optimizer_groups=[dict(role=g['role'], target=g['target'], initial_lr=g['lr'])
                          for g in optimizer.param_groups],
        proxy_geometry=geometry,
        measurements=[dict(step=0, **initial)],
        audits=[dict(step=0, targets=_audit(runtime, initial_decoders, update_codes))],
        steps=[], first_flip_step=None, cumulative_step_bit_flips=0,
        first_flip_by_target={path: None for path, _ in targets},
    )
    _write_progress(progress_path, record)
    n_batches = ntrain // batch_size
    measurement_steps = {s for s in (96, 192, 384, steps) if s <= steps}
    for step in range(1, steps + 1):
        start = ((step - 1) % n_batches) * batch_size
        optimizer.zero_grad(set_to_none=True)
        prediction = block_output(block, hidden[start:start + batch_size].to(device), kwargs)
        truth = target[start:start + batch_size].to(device)
        loss = (prediction.float() - truth.float()).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Step {step}: nonfinite training loss.')
        loss.backward()
        for path, module, scores, params in runtime:
            if update_codes:
                if scores.grad is None or not torch.isfinite(scores.grad).all():
                    raise FloatingPointError(f'{path}: disconnected/nonfinite proxy gradients.')
                if step == 1 and scores.grad.norm().item() == 0:
                    raise ValueError(f'{path}: zero initial proxy gradient.')
            elif scores.requires_grad or scores.grad is not None:
                raise AssertionError(f'{path}: frozen proxy received gradients.')
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError(f'{path}: disconnected/nonfinite decoder gradients.')
            if step == 1 and sum(p.grad.float().square().sum().item() for p in params) == 0:
                raise ValueError(f'{path}: zero initial decoder gradient.')
        decoder_lrs = [g['lr'] for g in optimizer.param_groups if g['role'] == 'decoder']
        code_lrs = [g['lr'] for g in optimizer.param_groups if g['role'] == 'code']
        optimizer.step()
        changed_bits = 0
        for path, module, scores, params in runtime:
            if not torch.isfinite(scores).all() or any(not torch.isfinite(p).all() for p in params):
                raise FloatingPointError(f'{path}: nonfinite updated parameter.')
            delta = all_bits.project(module)
            flips = delta['changed_packed_bits']
            if not update_codes and flips:
                raise AssertionError(f'{path}: decoder-only projection changed packed bits.')
            changed_bits += flips
            if flips and record['first_flip_by_target'][path] is None:
                record['first_flip_by_target'][path] = step
        if changed_bits and record['first_flip_step'] is None:
            record['first_flip_step'] = step
        record['cumulative_step_bit_flips'] += changed_bits
        record['steps'].append(dict(
            step=step, batch_start=start, train_loss=loss.item(),
            decoder_learning_rates=decoder_lrs, code_learning_rates=code_lrs,
            step_changed_packed_bits=changed_bits,
        ))
        advance_learning_rates(optimizer, schedulers)
        if step % 32 == 0 or step == steps:
            record['audits'].append(dict(
                step=step, targets=_audit(runtime, initial_decoders, update_codes, gradients=True)))
        if step in measurement_steps:
            measured, _ = _measure(block, hidden, target, kwargs, batch_size, ntrain, device)
            record['measurements'].append(dict(step=step, **measured))
        if step % 32 == 0 or step in measurement_steps:
            record['elapsed_seconds'] = time.monotonic() - started
            _write_progress(progress_path, record)
            print(f"{record['arm']} {step}/{steps}: train loss={loss.item():.7g}, "
                  f"step flips={changed_bits}, first flip={record['first_flip_step']}", flush=True)
        del prediction, truth, loss
    optimizer.zero_grad(set_to_none=True)
    final, reference = _measure(block, hidden, target, kwargs, batch_size, ntrain,
                                device, keep_outputs=True)
    record['final'] = final
    record['endpoint_targets'] = _audit(runtime, initial_decoders, update_codes)
    record['bits_changed_from_initial'] = sum(
        stats['bits_changed_from_initial'] for stats in record['endpoint_targets'].values())
    record['total_code_bits'] = sum(scores.numel() for _, _, scores, _ in runtime)
    for _, module, _, _ in runtime:
        all_bits.detach(module)
    # Release code Adam moments and all FP32 proxies before native dense caches.
    del optimizer, schedulers, groups, runtime, initial_decoders, scores, params
    torch.cuda.empty_cache()
    prime_packed_cache(block)
    with torch.no_grad():
        max_abs = 0.0
        for start in range(0, len(hidden), batch_size):
            native = block_output(block, hidden[start:start + batch_size].to(device), kwargs).cpu()
            expected = reference[start:start + batch_size]
            torch.testing.assert_close(native, expected, rtol=0, atol=0)
            max_abs = max(max_abs, (native.float() - expected.float()).abs().max().item())
    record['hard_to_native_max_abs'] = max_abs
    record['strict_native_equivalence_documents'] = len(hidden)
    record['max_memory_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
    clear_caches(block)
    block.cpu().requires_grad_(False)
    torch.cuda.empty_cache()
    record.update(status='PASS', elapsed_seconds=time.monotonic() - started)
    _write_progress(progress_path, record)
    return record
