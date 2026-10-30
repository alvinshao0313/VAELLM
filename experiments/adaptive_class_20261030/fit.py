"""Actual D/K/H-driven independent stage3 fitting and once-only packed routing."""
from copy import deepcopy
import gc
import math
from pathlib import Path
import time

import torch
from transformers import get_scheduler

from experiments.adaptive_grouped_fit_20261029.grouped import IndependentGroupedVAE, fresh_models
from experiments.adaptive_native_model_20261029 import allocation, objective
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.native_prefix import fixed2_frame
from experiments.adaptive_vector_rvq import common as c
from experiments.adaptive_vector_rvq.codec import decoder_spec
from litebsq.bitpack import pack_bool_tensor_to_uint8
from train_utils.cat_train_data import compute_stage_norm_stats
from train_utils.cat_train_pipeline import _fuse_norm_into_decoder, _fuse_q_scale_into_decoder
from train_utils.train_args import create_optimizer
from . import artifact, sources as s, sparse


def arguments(bundle, dtype='bf16'):
    first = bundle['stages'][0]['decoder_spec']
    args = c.arguments(dtype)
    args.codebook_dim, args.codebook_bits = int(first['out_dim']), int(first['in_dim'])
    args.base_ch = args.encoder_base_ch = args.decoder_base_ch = int(first['hidden_dim'])
    args.num_res_blocks = args.encoder_num_res_blocks = args.decoder_num_res_blocks = int(first['num_res_blocks'])
    args.decoder_type, args.norm_type, args.activation_type = (first[key] for key in ('decoder_type', 'norm_type', 'activation_type'))
    c.require(args.codebook_dim == args.codebook_bits == 64 and args.base_ch == 128
              and args.num_res_blocks == 0 and args.decoder_type == 'asymmetric', 'Use the selected actual B stage3 architecture.')
    return args


@torch.no_grad()
def make_value(gate, spec, live, energy, output, device):
    bundle, lengths, resolved = s.prefix(gate, spec)
    name = spec['name']
    target_weight = live[name + '.weight']
    vectors = objective.live_vectors(bundle, target_weight.to(device)).cpu()
    prefixes = objective.decode_prefixes(bundle, device=device)
    variance = float(vectors.double().var(unbiased=False))
    errors = objective.prefix_errors(vectors, prefixes, energy[name], resolved['geometry'], variance)
    residual = (vectors - prefixes[-1].float()).contiguous()
    mean, scale = compute_stage_norm_stats(residual.unsqueeze(1))
    mean, scale = mean.reshape(()), scale.reshape(())
    normalized = ((residual - mean) / scale).bfloat16().unsqueeze(1).to(device)
    base = CompactRVQLinear(bundle, lengths, device)
    actual = base.decode_weight().detach().cpu()
    fields, rice_info = sparse.build(bundle, target_weight, actual)
    control = sparse.Fixed2RiceLinear(bundle, fields)
    output.mkdir(parents=True)
    cap = artifact.save_module(control, output / 'fixed2_rice7.pt')
    fresh, saved = artifact.load(output / 'fixed2_rice7.pt')
    c.require(c.digest(control.state_dict()) == c.digest(fresh.state_dict()) and 'length_frame' not in saved['state'],
              'The actual control must fresh-load every payload and keep length2 implicit.')
    info = dict(name=name, live_target_digest=c.digest({'weight': target_weight}), source_variance=variance,
        actual_vector_shape=list(vectors.shape), actual_normalization=dict(mean=float(mean), scale=float(scale),
        population='Every real vector, residual from actual compact BF16 prefix'),
        fixed2_NMSE=float(errors[:, 1].mean()), stage1_NMSE=float(errors[:, 0].mean()),
        control=dict(path=str(output / 'fixed2_rice7.pt'), actual_complete_bytes=cap, sha256=c.sha(output / 'fixed2_rice7.pt'),
            tensor_bytes=sum(c.compact(value) for value in saved['state'].values()), **rice_info))
    c.write(output / 'preparation.json', info)
    del base, control, fresh, actual, residual, vectors, prefixes, saved
    gc.collect()
    torch.cuda.empty_cache()
    return dict(bundle=bundle, mean=mean, scale=scale, normalized=normalized, errors=errors,
                energy=energy[name], variance=variance, cap=cap, info=info, output=output)


@torch.no_grad()
def export_stage(single, value, device):
    normalized = value['normalized']
    single.eval()
    bits = int(single.model.args.codebook_bits)
    codes = torch.empty((len(normalized), (bits + 7) // 8), dtype=torch.uint8)
    for begin in range(0, len(normalized), 8192):
        batch = normalized[begin:begin + 8192]
        _, current = single(batch, is_train=False)
        codes[begin:begin + len(batch)] = pack_bool_tensor_to_uint8(current.squeeze(1).cpu(), logical_shape=(len(batch), bits))
    decoder = single.model.decoder.get_sub_decoder(0)
    _fuse_q_scale_into_decoder(decoder, 1. / math.sqrt(bits))
    _fuse_norm_into_decoder(decoder, float(value['mean']), float(value['scale']))
    decoder = decoder.cpu().eval().requires_grad_(False)
    return dict(decoder_spec=decoder_spec(decoder), codes=codes,
        decoder_state={name: tensor.detach().cpu().contiguous().clone() for name, tensor in decoder.state_dict().items()},
        normalization=dict(mean=float(value['mean']), scale=float(value['scale']), active_count=len(normalized)))


@torch.no_grad()
def route(value, stage, live, device):
    output = value['output']
    bundle = {**value['bundle'], 'stages': [*value['bundle']['stages'], stage]}
    bundle['length_frame'] = allocation._frame(torch.full((bundle['n_vectors'],), 3, dtype=torch.uint8))
    target = objective.live_vectors(bundle, live[value['info']['name'] + '.weight'].to(device)).cpu()
    prefixes = objective.decode_prefixes(bundle, device=device)
    errors = objective.prefix_errors(target, prefixes, value['energy'], bundle['geometry'], value['variance'])
    c.require(torch.equal(errors[:, :2], value['errors']), 'The shared raw2 target/prefix/h scores changed before routing.')
    lengths, selected, measured = allocation.iterate(errors, bundle, value['cap'], artifact.dynamic_serialize, persistent_fields={})
    data = artifact.dynamic_serialize(selected, {})
    with (output / 'dynamic3.pt').open('xb') as stream:
        stream.write(data)
    fresh, loaded = artifact.load(output / 'dynamic3.pt')
    c.require(len(data) == measured['chosen']['actual_complete_bytes'] <= value['cap']
        and torch.equal(fresh.lengths, lengths), 'Actual dynamic file exceeds the same complete control cap or changed routes.')
    for index in range(2):
        original = value['bundle']['stages'][index]
        c.require(c.digest(original['decoder_state']) == c.digest(selected['stages'][index]['decoder_state']), 'Routing changed frozen raw2 decoders.')
    info = dict(**value['info'], dense_stage3_NMSE=float(errors[:, 2].mean()),
        routed_NMSE=float(errors.gather(1, lengths.long().sub(1).unsqueeze(1)).mean()), allocation=measured,
        dynamic=dict(path=str(output / 'dynamic3.pt'), actual_complete_bytes=len(data), sha256=c.sha(output / 'dynamic3.pt'),
            length_histogram=torch.bincount(lengths.long(), minlength=4).tolist(),
            persistent_digest=c.digest(loaded['state'])), shared_prefix_decoders_exact=True,
        route_iterations=measured['chosen_round'], routing='Once after full-vector stage3, then codes/lengths frozen throughout recovery')
    c.write(output / 'report.json', info)
    return info


def fit_group(values, steps, device, *, stop_after=None):
    args = arguments(values[0]['bundle'])
    batch_size = 262144 // args.codebook_dim
    originals = fresh_models(len(values), device=device, args=args)
    model = IndependentGroupedVAE(originals).to(device)
    del originals
    optimizer = create_optimizer(model.parameters(), args, .003)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=steps)
    generators = [torch.Generator().manual_seed(2031) for _ in values]
    orders, positions = [None for _ in values], [0 for _ in values]
    rows, samples = [], [0 for _ in values]
    actual_steps = stop_after or steps
    started = time.monotonic()
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()} if stop_after else None
    for step in range(1, actual_steps + 1):
        batches = []
        for index, value in enumerate(values):
            data = value['normalized']
            if orders[index] is None or positions[index] >= len(data):
                orders[index] = torch.randperm(len(data), generator=generators[index])
                positions[index] = 0
            ids = orders[index][positions[index]:positions[index] + batch_size]
            positions[index] += len(ids)
            samples[index] += len(ids)
            batches.append(data.index_select(0, ids.to(device)))
        c.require(len({batch.shape for batch in batches}) == 1, 'Preserve each independent layer tail; no padding/resampling.')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, losses = model(torch.cat(batches, dim=1))
        c.require(torch.isfinite(losses).all().item(), 'Nonfinite actual independent BSQ stage3 losses.')
        losses.sum().backward()
        if step <= 2 or step % 100 == 0 or step == actual_steps:
            c.require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()),
                'Missing/nonfinite actual encoder/decoder gradient.')
        optimizer.step()
        scheduler.step()
        if step <= 2 or step % 100 == 0 or step == actual_steps:
            c.require(all(int(optimizer.state[parameter]['step']) == step for parameter in model.parameters()), 'Independent Adam step counters diverged.')
            row = dict(step=step, losses=losses.detach().float().tolist(), lr=scheduler.get_last_lr()[0], elapsed_seconds=time.monotonic() - started)
            rows.append(row)
            print(dict(event='independent_gate_stage3', **row), flush=True)
        del losses, batches
    torch.cuda.synchronize()
    if initial is not None:
        c.require(all(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters()), 'An actual grouped parameter did not update.')
    stages = [export_stage(model.single_model(index), value, device) for index, value in enumerate(values)]
    info = dict(actual_steps=actual_steps, schedule_max_steps=steps, batch_per_layer=batch_size,
        layers=len(values), actual_sampled_vectors_per_layer=samples, rows=rows,
        elapsed_seconds=time.monotonic() - started, updates_per_wall_second=actual_steps / (time.monotonic() - started),
        entropy='Per-layer MSE/commit/sample entropy/K-bit marginal entropy, summed losses; original raw2 M36 pooled unchanged',
        shuffle_seed_per_layer=2031, initial_seed_per_layer=2031,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    return stages, info
