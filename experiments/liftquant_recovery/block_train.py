"""Pinned LiftQuant Stage2 ALL loop with native VAE coordinate adaptation."""
import torch

from experiments.liftquant_recovery import all_bits
from experiments.liftquant_recovery.liftquant_optimizer import build_optimizer, advance_learning_rates
from experiments.liftquant_recovery.proxy_coordinates import motion_budget
from experiments.liftquant_recovery.recovery_runtime import block_output, clear_caches, tree_to, prime_packed_cache


@torch.no_grad()
def measure_split(block, hidden, target, kwargs, batch_size, ntrain, device):
    """Measure overall/train/holdout SSE in one forward pass over the documents."""
    if hidden.shape != target.shape or not 0 < ntrain <= len(hidden):
        raise ValueError('Expected matching hidden/target and a nonempty training split.')
    block.eval()
    prime_packed_cache(block)
    # Each batch reduction stays FP32, as before; FP64 accumulates its scalar SSE
    # in the same precision as the previous Python-float accumulation.
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for start in range(0, len(hidden), batch_size):
        prediction = block_output(block, hidden[start:start+batch_size].to(device), kwargs)
        error = (prediction.float()-target[start:start+batch_size].to(device).float()).square()
        batch_sse = error.sum()
        totals[0].add_(batch_sse)
        split = max(0, min(ntrain-start, len(error)))
        if split == len(error):
            totals[1].add_(batch_sse)
        elif split == 0:
            totals[2].add_(batch_sse)
        else:
            totals[1].add_(error[:split].sum())
            totals[2].add_(error[split:].sum())
    all_sse, train_sse, holdout_sse = totals.cpu().tolist()
    elements_per_document = target[0].numel()
    return dict(mse=all_sse/target.numel(),
                train_mse=train_sse/(ntrain*elements_per_document),
                holdout_mse=(holdout_sse/((len(hidden)-ntrain)*elements_per_document)
                             if ntrain < len(hidden) else None))


@torch.no_grad()
def measure(block, hidden, target, kwargs, batch_size, device):
    """Compatibility API for callers that only need the overall MSE."""
    return measure_split(block, hidden, target, kwargs, batch_size, len(hidden), device)['mse']


def _all_finite(tensors):
    """One host synchronization for a whole collection, without a dense concat."""
    return bool(torch.stack([torch.isfinite(value).all() for value in tensors]).all())


def _host_step(loss, audit):
    """Transfer scalar telemetry together; count values fit exactly in FP64."""
    entries = [(path, key, value) for path, values in audit.items() for key, value in values.items()]
    values = torch.stack([loss.detach().to(torch.float64)] +
                         [value.detach().to(torch.float64) for _, _, value in entries]).cpu().tolist()
    result = {path: {} for path in audit}
    for (path, key, tensor), value in zip(entries, values[1:]):
        result[path][key] = int(value) if not tensor.is_floating_point() else value
    return values[0], result


def train_block(block, targets, hidden, target, kwargs, args, device):
    block.to(device)
    block_entry_allocated = torch.cuda.memory_allocated(device)/2**30
    block_entry_reserved = torch.cuda.memory_reserved(device)/2**30
    groups, runtime, initial_decoders, proxy_geometry = [], [], {}, {}
    ntrain = args.nsamples-(args.holdout if args.holdout is not None else args.nsamples//32)
    if ntrain < args.batch_size or ntrain % args.batch_size:
        raise ValueError('Training samples must be a positive batch multiple.')
    audit_every = getattr(args, 'audit_every', 32)
    if audit_every <= 0:
        raise ValueError('audit_every must be positive.')
    steps = args.epochs*(ntrain//args.batch_size)
    kwargs = tree_to(kwargs, device)
    before = measure_split(block, hidden, target, kwargs, args.batch_size, ntrain, device)
    clear_caches(block)
    for path, module in targets:
        scores, decoder_params = all_bits.attach(module)
        std = scores.detach().std().item()
        lr = min(args.code_lr, std/50)
        geometry = dict(module._recovery_runtime.geometry, proxy_std=std)
        geometry['actual_budget'] = motion_budget(lr, steps, geometry['initial_margin'])
        geometry['planned_3968_step_budget'] = motion_budget(lr, 3968, geometry['initial_margin'])
        proxy_geometry[path] = geometry
        groups.append(dict(params=[scores], lr=lr, role='code', target=path))
        groups.append(dict(params=decoder_params, lr=args.decoder_lr, role='decoder', target=path))
        runtime.append((path, module, scores, decoder_params))
        initial_decoders[path] = [p.detach().clone() for p in decoder_params]
    allowed = {id(p) for _, _, _, ps in runtime for p in ps}
    unexpected = [name for name, p in block.named_parameters() if p.requires_grad and id(p) not in allowed]
    if unexpected:
        raise ValueError(f'Unexpected trainables: {unexpected}')
    optimizer, schedulers = build_optimizer(groups, steps)
    trainables = [p for _, _, scores, params in runtime for p in [scores, *params]]
    record = dict(before_mse=before['mse'], before_train_mse=before['train_mse'],
                  before_holdout_mse=before['holdout_mse'],
                  ntrain=ntrain, holdout=args.nsamples-ntrain,
                  optimizer_steps=steps, targets=[p for p, _ in targets],
                  proxy_geometry=proxy_geometry, audit_every=audit_every,
                  audit_policy='full audit on first/final/every audit_every; finite checks and projection every step',
                  steps=[])
    block.eval()
    step = 0
    for epoch in range(args.epochs):
        for start in range(0, ntrain, args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            x = hidden[start:start+args.batch_size].to(device)
            y = target[start:start+args.batch_size].to(device)
            prediction = block_output(block, x, kwargs)
            loss = (prediction.float()-y.float()).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite block loss.')
            loss.backward()
            for path, module, scores, params in runtime:
                if scores.grad is None or any(p.grad is None for p in params):
                    raise FloatingPointError(f'{path}: disconnected proxy/decoder gradients')
            if not _all_finite([p.grad for p in trainables]):
                raise FloatingPointError('Nonfinite proxy/decoder gradients.')
            audit_step = step == 0 or (step+1) % audit_every == 0 or step+1 == steps
            audit = {path: {} for path, _, _, _ in runtime}
            if audit_step:
                for path, module, scores, params in runtime:
                    audit[path]['code_grad_l2'] = scores.grad.norm()
                    audit[path]['decoder_grad_l2'] = torch.stack(
                        [p.grad.float().square().sum().to(torch.float64) for p in params]).sum().sqrt()
                if step == 0 and not bool(torch.stack([
                    value != 0 for values in audit.values() for value in values.values()]).all()):
                    raise ValueError('Zero first-step proxy/decoder gradient norm.')
            used_lrs = [group['lr'] for group in optimizer.param_groups]
            optimizer.step()
            if not _all_finite(trainables):
                raise FloatingPointError('Nonfinite post-step proxy/decoder parameters.')
            for path, module, scores, params in runtime:
                if audit_step:
                    stats = all_bits.proxy_statistics(module, device=True)
                    decoder_motion = torch.stack([(p.detach()-old).abs().max()
                                                  for p, old in zip(params, initial_decoders[path])]).max()
                    audit[path].update(stats, decoder_delta_max=decoder_motion)
                audit[path].update(all_bits.project(module, device=True))
            if step == 0 and not bool(torch.stack([
                values[key] != 0 for values in audit.values()
                for key in ('proxy_delta_max', 'decoder_delta_max')]).all()):
                raise ValueError('First optimizer step did not change proxy/decoder parameters.')
            advance_learning_rates(optimizer, schedulers)
            step += 1
            loss_value, host_audit = _host_step(loss, audit)
            record['steps'].append(dict(step=step, epoch=epoch, loss=loss_value,
                                        learning_rates=used_lrs, audit=host_audit))
            if audit_step:
                print(f'optimizer step {step}/{steps}: loss={loss_value:.7g}', flush=True)
            del prediction, loss, x, y, audit
    optimizer.zero_grad(set_to_none=True)
    block.eval()
    with torch.no_grad():
        reference = block_output(block, hidden[:args.batch_size].to(device), kwargs).cpu()
    for _, module, _, _ in runtime:
        all_bits.detach(module)
    del optimizer, schedulers, runtime, groups, trainables, scores, decoder_params, initial_decoders
    torch.cuda.empty_cache()
    prime_packed_cache(block)
    with torch.no_grad():
        native = block_output(block, hidden[:args.batch_size].to(device), kwargs).cpu()
    torch.testing.assert_close(native, reference, rtol=0, atol=0)
    record['hard_to_native_max_abs'] = (native.float()-reference.float()).abs().max().item()
    after = measure_split(block, hidden, target, kwargs, args.batch_size, ntrain, device)
    record['after_mse'] = after['mse']
    record['after_train_mse'] = after['train_mse']
    if after['holdout_mse'] is not None:
        record['holdout_mse'] = after['holdout_mse']
    # Do not reset CUDA peak counters: outer callers also report whole-run peaks.
    record['max_memory_allocated_gib'] = torch.cuda.max_memory_allocated(device)/2**30
    record['memory_scope'] = 'max_memory_allocated_gib is the process peak since the last external reset, not a block-local peak'
    record['memory'] = dict(process_peak_allocated_gib=record['max_memory_allocated_gib'],
                          process_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                          block_entry_allocated_gib=block_entry_allocated,
                          block_entry_reserved_gib=block_entry_reserved,
                          block_exit_allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                          block_exit_reserved_gib=torch.cuda.memory_reserved(device)/2**30)
    clear_caches(block)
    block.cpu()
    torch.cuda.empty_cache()
    return record, reference
