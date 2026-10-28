"""Independent native BSQ stages, frozen deployed prefixes, and fixed step budgets."""
from __future__ import annotations

import gc
import math
import time

import torch
from transformers import get_scheduler

from . import common as c
from . import codec
from litebsq.autoencoder import MultiLayerVAE
from litebsq.bitpack import pack_bool_tensor_to_uint8
from train_utils.cat_train_data import compute_stage_norm_stats
from train_utils.train_args import create_optimizer


def finalize(model, mean, scale):
    # These are the same core fusion operations used by ordinary CAT exports.
    from train_utils.cat_train_pipeline import _fuse_norm_into_decoder, _fuse_q_scale_into_decoder
    decoder = model.model.decoder.get_sub_decoder(0)
    _fuse_q_scale_into_decoder(decoder, 1. / math.sqrt(c.BITS))
    _fuse_norm_into_decoder(decoder, float(mean), float(scale))
    decoder = decoder.cpu().eval().requires_grad_(False)
    state = {name: value.detach().clone().contiguous() for name, value in decoder.state_dict().items()}
    return codec.decoder_spec(decoder), state


@torch.no_grad()
def encode(model, residual, active, mean, scale, device):
    model.eval()
    codes = torch.empty((active.numel(), c.BITS // 8), dtype=torch.uint8)
    dtype = model.model.params_dtype
    for start in range(0, active.numel(), c.CHUNK):
        ids = active[start:start + c.CHUNK]
        values = ((residual.index_select(0, ids) - mean) / scale).to(device=device, dtype=dtype).unsqueeze(1)
        _, bits = model(values, is_train=False)
        codes[start:start + ids.numel()] = pack_bool_tensor_to_uint8(bits.squeeze(1).cpu(), logical_shape=(ids.numel(), c.BITS))
    return codes


def mse(target, reconstructed, ids=None):
    total = 0.
    count = 0
    if ids is None:
        ids = torch.arange(len(target))
    for start in range(0, ids.numel(), 65536):
        idx = ids[start:start + 65536]
        error = target.index_select(0, idx).double() - reconstructed.index_select(0, idx).double()
        total += float(error.square().sum())
        count += error.numel()
    return total / count if count else 0.


def fit_stage(target, prefix, active, *, steps, seed, device, output, source_variance, dtype='bf16'):
    c.require(active.numel() > 0 and steps > 0, 'Train only a nonempty active residual population.')
    started = time.perf_counter()
    cuda = torch.device(device).type == 'cuda'
    torch.manual_seed(seed)
    model = MultiLayerVAE(c.arguments(dtype)).to(device)
    if cuda:
        # Real parameter allocation initializes the CUDA allocator before its stats are reset.
        torch.cuda.reset_peak_memory_stats(device)
    initial_digest = c.digest(model.state_dict())
    residual = (target.float() - prefix.float()).contiguous()
    selected = residual.index_select(0, active)
    mean, scale = compute_stage_norm_stats(selected.unsqueeze(1))
    mean, scale = mean.reshape(()), scale.reshape(())
    normalized = ((selected - mean) / scale).to(device=device, dtype=model.model.params_dtype).unsqueeze(1)
    optimizer = create_optimizer(model.parameters(), c.arguments(dtype), .003)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=steps)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    order, position = None, 0
    observations, losses = [], []
    trained_vectors, batch_min, checked_gradients = 0, c.BATCH, []
    # Fixed2 observations are prefixes of its original3*base-step scheduler.
    milestones = {steps // 3, 2 * steps // 3, steps} if output.name == 'fixed2' else {max(1, steps // 2), steps}
    for step in range(1, steps + 1):
        if order is None or position >= active.numel():
            order = torch.randperm(active.numel(), generator=generator)
            position = 0
        indices = order[position:position + c.BATCH]
        position += indices.numel()
        batch = normalized.index_select(0, indices.to(device))
        trained_vectors += indices.numel()
        batch_min = min(batch_min, indices.numel())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, parts = model(batch, is_train=True)
        loss = parts['loss']
        c.require(bool(torch.isfinite(loss)) and loss.requires_grad, 'Nonfinite/disconnected actual BSQ stage loss.')
        loss.backward()
        if step == 1 or step % 100 == 0 or step in milestones:
            c.require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()), 'Nonfinite or disconnected encoder/decoder gradient.')
            checked_gradients.append(step)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().float()))
        if step in milestones:
            c.require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()), 'Nonfinite saved encoder/decoder parameters.')
            codes = encode(model, residual, active, mean, scale, device)
            spec, state = finalize(model, mean, scale)
            stage = dict(decoder_spec=spec, decoder_state=state, codes=codes,
                normalization=dict(mean=float(mean), scale=float(scale), active_count=active.numel()))
            reconstructed = prefix.clone()
            codec.add_stage(reconstructed, stage, active, device)
            row = dict(step=step, total_scheduler_steps=steps, learning_rate=float(scheduler.get_last_lr()[0]),
                recent_loss=sum(losses[-100:]) / len(losses[-100:]), active_MSE=mse(target, reconstructed, active),
                whole_MSE=mse(target, reconstructed), normalized_whole_MSE=mse(target, reconstructed) / source_variance)
            observations.append(row)
            c.write(output / 'observations.json', dict(status='running' if step < steps else 'completed', initial_model_digest=initial_digest,
                seed=seed, steps=steps, active_count=active.numel(), normalization=stage['normalization'], observations=observations,
                actual_sampled_vectors=trained_vectors, min_batch_vectors=batch_min, max_batch_vectors=c.BATCH,
                gradient_finite_checked_steps=checked_gradients))
            print(dict(event='stage_observation', stage=output.name, **row), flush=True)
    c.require(c.digest(model.state_dict()) != initial_digest, 'Actual stage training did not change its native parameters.')
    elapsed = time.perf_counter() - started
    resources = dict(elapsed_seconds=elapsed, updates_per_wall_second=steps / elapsed,
        timing_scope='Complete stage including active normalization, training, saved-prefix observations and final encoding')
    if cuda:
        resources.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
    del normalized, selected, residual, model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return stage, reconstructed, dict(initial_model_digest=initial_digest, seed=seed, steps=steps,
        active_count=active.numel(), normalization=stage['normalization'], observations=observations, allfinite=True,
        actual_sampled_vectors=trained_vectors, min_batch_vectors=batch_min, max_batch_vectors=c.BATCH,
        gradient_finite_checked_steps=checked_gradients, every_update_loss_finite=True, **resources)
