"""Full-vector common residual and independent BSQ stage3; never runs on import."""
from copy import deepcopy
import math
import time

from . import bindings as b
from . import allocation
from experiments.adaptive_grouped_fit_20261029.grouped import IndependentGroupedVAE, fresh_models
from experiments.adaptive_native_model_20261029 import objective
from experiments.adaptive_vector_rvq.codec import decoder_spec
from litebsq.bitpack import pack_bool_tensor_to_uint8
from train_utils.cat_train_data import compute_stage_norm_stats
from train_utils.cat_train_pipeline import _fuse_norm_into_decoder, _fuse_q_scale_into_decoder
from train_utils.train_args import create_optimizer
from transformers import get_scheduler


@b.torch.no_grad()
def common_residual(bundle, live_weight, *, device='cpu'):
    target = objective.live_vectors(bundle, live_weight.to(device)).cpu()
    prefixes = objective.decode_prefixes(bundle, device=device)
    residual = (target - prefixes[-1].float()).contiguous()
    mean, scale = (value.reshape(()) for value in compute_stage_norm_stats(residual.unsqueeze(1)))
    normalized = ((residual - mean) / scale).bfloat16().unsqueeze(1).contiguous()
    info = dict(target_digest=b.c.digest({'target': target}), prefix_digest=b.c.digest({'prefix2': prefixes[-1]}),
        residual_digest=b.c.digest({'residual': residual}), normalized_digest=b.c.digest({'normalized': normalized}),
        mean=float(mean), scale=float(scale), shape=list(normalized.shape),
        source_variance=float(target.double().var(unbiased=False)),
        normalization='One scalar mean/unbiased std of every FP32 live-minus-sequential-BF16-compact residual coordinate per Linear',
        both_K_share_this_input=True, all_vectors_before_routing=True)
    return dict(bundle=bundle, target=target, prefixes=prefixes, normalized=normalized, mean=mean, scale=scale, info=info)


def batches(values, generators, orders, positions):
    result, selected = [], []
    for index, value in enumerate(values):
        data = value['normalized']
        if orders[index] is None or positions[index] >= len(data):
            orders[index] = b.torch.randperm(len(data), generator=generators[index])
            positions[index] = 0
        ids = orders[index][positions[index]:positions[index] + b.BATCH]
        positions[index] += len(ids)
        result.append(data.index_select(0, ids.to(data.device)))
        selected.append(ids)
    b.c.require(len({tuple(batch.shape) for batch in result}) == 1, 'Preserve genuine per-layer epoch tails without padding or resampling.')
    return b.torch.cat(result, dim=1), selected


@b.torch.no_grad()
def export_stage(single, value, *, device='cpu'):
    single = deepcopy(single).to(device).eval()
    bits = int(single.model.args.codebook_bits)
    codes = b.torch.empty((len(value['normalized']), (bits + 7) // 8), dtype=b.torch.uint8)
    for begin in range(0, len(codes), 8192):
        data = value['normalized'][begin:begin + 8192].to(device)
        _, signs = single(data, is_train=False)
        codes[begin:begin + len(data)] = pack_bool_tensor_to_uint8(signs.squeeze(1).cpu(), logical_shape=(len(data), bits))
    decoder = single.model.decoder.get_sub_decoder(0)
    _fuse_q_scale_into_decoder(decoder, 1. / math.sqrt(bits))
    _fuse_norm_into_decoder(decoder, float(value['mean']), float(value['scale']))
    decoder = decoder.cpu().bfloat16().eval().requires_grad_(False)
    state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in decoder.state_dict().items()}
    expected_bytes = {8: 6944, 24: 11040}[bits]
    b.c.require(len(state) == 6 and sum(b.c.compact(tensor) for tensor in state.values()) == expected_bytes,
        'Count all six true BF16 decoder leaves, including both biases and normalization.')
    return dict(decoder_spec=decoder_spec(decoder), codes=codes, decoder_state=state,
        normalization=dict(mean=float(value['mean']), scale=float(value['scale']), full_vector_count=len(codes),
            status='stage3 only; signed BSQ scale and residual normalization fused once'))


def full_bundle(value, stage):
    n = value['bundle']['n_vectors']
    b.c.require(len(stage['codes']) == n, 'Route only after full-vector third-stage encoding.')
    return {**value['bundle'], 'stages': [*value['bundle']['stages'], stage],
        'length_frame': allocation.frame(b.torch.full((n,), 3, dtype=b.torch.uint8))}


@b.torch.no_grad()
def errors(value, stage, h, *, device='cpu'):
    bundle = full_bundle(value, stage)
    prefixes = objective.decode_prefixes(bundle, device=device)
    b.c.require(all(b.torch.equal(actual, expected) for actual, expected in zip(prefixes[:2], value['prefixes'], strict=True)),
        'The common sequential-BF16 frozen prefix changed before scoring.')
    scores = objective.prefix_errors(value['target'], prefixes, h, bundle['geometry'], value['info']['source_variance'])
    return scores[:, 1:].contiguous(), bundle


def gradients_before_update(model):
    b.c.require(all(parameter.dtype == b.torch.bfloat16 and parameter.grad is not None
        and parameter.grad.dtype == parameter.dtype and bool(b.torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()), 'Require finite actual BF16 gradients before the AdamW update.')


@b.torch.no_grad()
def layer_parameters_before_update(model):
    return [{name: value.detach().clone() for name, value in model.single_parameters(index).items()}
        for index in range(model.count)]


@b.torch.no_grad()
def update_evidence(model, optimizer, *, step, before=None):
    leaves = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        b.c.require(parameter.dtype == b.torch.bfloat16 and bool(b.torch.isfinite(parameter).all()),
            'Actual updated parameters must remain finite BF16 without persistent FP32 masters.')
        b.c.require(state['step'].numel() == 1 and bool(b.torch.isfinite(state['step']).all())
            and float(state['step']) == step, 'Every actual AdamW leaf must have the expected update count.')
        for moment in ('exp_avg', 'exp_avg_sq'):
            value = state[moment]
            b.c.require(value.dtype == parameter.dtype and value.shape == parameter.shape
                and bool(b.torch.isfinite(value).all()), 'Preserve finite actual BF16 AdamW moment state.')
        leaves[name] = dict(parameter_dtype=str(parameter.dtype), parameter_finite=True,
            step=float(state['step']), step_dtype=str(state['step'].dtype),
            exp_avg_dtype=str(state['exp_avg'].dtype), exp_avg_sq_dtype=str(state['exp_avg_sq'].dtype),
            moments_finite=True)
    displacements = None
    if before is not None:
        b.c.require(len(before) == model.count, 'Record each independent layer before its actual update.')
        displacements = []
        for index in range(model.count):
            current = model.single_parameters(index)
            changed = [name for name, value in current.items() if not b.torch.equal(value, before[index][name])]
            b.c.require(changed, 'An independent layer received no actual BF16 parameter displacement.')
            displacements.append(dict(layer=index, changed_leaves=changed, actual_displacement=True))
    return dict(actual_step=step, optimizer_leaves=leaves, layer_displacements=displacements,
        parameter_and_moment_precision='Original BF16; scalar Adam step retains native PyTorch dtype')


def fit_group(values, bits, *, device):
    """Prepared future short card only: fixed2000 updates of the original10K linear schedule."""
    b.c.require(len(values) == 3 and str(device).startswith('cuda'), 'The future card uses three actual GPU Linear groups; CPU preparation never calls fitting.')
    args = b.arguments(bits)
    originals = fresh_models(3, device=device, args=args)
    model = IndependentGroupedVAE(originals).to(device)
    del originals
    optimizer = create_optimizer(model.parameters(), args, .003)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=10000)
    generators = [b.torch.Generator().manual_seed(2031) for _ in values]
    orders, positions = [None] * 3, [0] * 3
    rows, sampled, update_checks = [], [0] * 3, []
    started = time.monotonic()
    for step in range(1, 2001):
        data, ids = batches(values, generators, orders, positions)
        b.c.require(tuple(data.shape) == (b.BATCH, 3, b.DIM), 'Actual short fitting requires full[16384,3,16] batches.')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, losses = model(data.to(device))
        b.c.require(bool(b.torch.isfinite(losses).all()), 'Nonfinite independent third-stage BSQ loss.')
        losses.sum().backward()
        if step <= 2 or step % 100 == 0 or step == 2000:
            gradients_before_update(model)
        before = layer_parameters_before_update(model) if step <= 2 else None
        optimizer.step()
        if step <= 2 or step == 2000:
            check = update_evidence(model, optimizer, step=step, before=before)
            check['gradient_finite_checked_before_update'] = True
            update_checks.append(check)
        scheduler.step()
        sampled = [count + len(current) for count, current in zip(sampled, ids, strict=True)]
        if step <= 2 or step % 100 == 0 or step == 2000:
            row = dict(step=step, losses=losses.detach().float().tolist(), lr=scheduler.get_last_lr()[0],
                elapsed_seconds=time.monotonic() - started)
            rows.append(row)
            print(dict(event='independent_D16_tail3', K=bits, **row), flush=True)
    b.torch.cuda.synchronize()
    stages = [export_stage(model.single_model(index), value, device=device) for index, value in enumerate(values)]
    return stages, dict(actual_updates=2000, schedule_steps=10000, warmup_steps=0, init_seed_per_layer=2031,
        shuffle_seed_per_layer=2031, batch_shape=[b.BATCH, 3, b.DIM], sampled_vectors=sampled, rows=rows,
        actual_update_checks=update_checks,
        selection='Fixed last2000-update endpoint; no checkpoint quality selection',
        optimizer=dict(name='original BF16 AdamW', lr=.003, betas=[.9, .95], eps=1e-8, weight_decay=0),
        elapsed_seconds=time.monotonic() - started, peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
