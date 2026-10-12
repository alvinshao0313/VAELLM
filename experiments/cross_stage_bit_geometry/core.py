"""Finite cross-stage pairs on disjoint real RHT tiles; no optimizer or model export."""
from __future__ import annotations

import time

import torch

from experiments.exact_bit_neighbor_probe.core import flip_grouped
from experiments.gate_student_context_probe.core import ready, records
from experiments.joint_reconstruction_probe.core import require, storage_bits


COUNT, SEED = 32, 31
CANDIDATES = [(-1, -1)] + [(i, -1) for i in range(32)] + [(-1, j) for j in range(32)] + [
    (i, j) for i in range(32) for j in range(32)]


def support():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    groups = torch.randperm(384, generator=generator)[:COUNT].tolist()
    offsets = torch.randint(0, 32, (COUNT,), generator=generator).tolist()
    columns = torch.randint(0, 126, (COUNT,), generator=generator).tolist()
    return [{'output_group': g, 'row_offset': r, 'input_block': c,
             'vector': (g * 32 + r) * 126 + c}
            for g, r, c in zip(groups, offsets, columns, strict=True)]


def packed_candidate(original, bits):
    value = original.clone()
    if bits[0] >= 0:
        value = flip_grouped(value, bits[0])
    if bits[1] >= 0:
        value = flip_grouped(value, 32 + bits[1])
    return value


@torch.no_grad()
def measure(layer, target, cases, *, deadline, observer):
    ready(layer)
    require(not layer.training and all(not p.requires_grad for p in layer.parameters())
            and layer.weight_rotation.block_size == 32 and layer._parallel_stage_restore_identity,
            'Use the frozen real two-stage gate with independent block32 rotation.')
    device = next(layer.parameters()).device
    require(device.type == 'cuda' and target.dtype == torch.bfloat16
            and tuple(target.shape) == (12288, 4096) and cases == support(), 'Wrong source or fixed support.')
    initial, bits_before = records(layer), storage_bits(layer)
    vectors = torch.tensor([x['vector'] for x in cases], device=device)
    rows = torch.tensor([x['output_group'] * 32 for x in cases], device=device)[:, None] + torch.arange(32, device=device)
    keep = torch.ones(4096, device=device, dtype=torch.bool)
    keep[layer.protected_input_indices] = False
    compressed_columns = torch.tensor([x['input_block'] * 32 for x in cases], device=device)[:, None] + torch.arange(32, device=device)
    columns = keep.nonzero().flatten()[compressed_columns]
    require(len(set(x['output_group'] for x in cases)) == COUNT, 'Each case must have disjoint full output rows.')
    banks = [layer.get_stage_part_vq_storage(s, 0) for s in range(2)]
    original = torch.stack([bank[vectors, 0].clone() for bank in banks], dim=1)
    lut = torch.tensor([i.bit_count() for i in range(256)], dtype=torch.uint8, device=device)
    reference = target.to(device)[rows[:, :, None], columns[:, None, :]].double()
    baseline = layer._decode_weight(torch.bfloat16)
    require(baseline.dtype == torch.bfloat16 and torch.equal(layer._decode_weight(torch.bfloat16), baseline),
            'Identical native full-matrix decodes are not repeatable.')
    base_tiles = baseline[rows[:, :, None], columns[:, None, :]].clone()
    residual = base_tiles.double() - reference
    base_sse = residual.square().sum((1, 2))
    union = torch.zeros_like(baseline, dtype=torch.bool)
    union[rows[:, :, None], columns[:, None, :]] = True
    singles = torch.empty((64, COUNT, 32, 32), dtype=torch.bfloat16, device=device)
    metrics = torch.empty((len(CANDIDATES), COUNT, 5), dtype=torch.float64)
    xors = torch.empty((len(CANDIDATES), COUNT, 2), dtype=torch.uint8)
    measured_support = []
    start = time.perf_counter()

    def install(value):
        for stage, bank in enumerate(banks):
            bank[vectors, 0] = value[:, stage]
        ready(layer)

    def tiles(weight):
        return weight[rows[:, :, None], columns[:, None, :]]

    try:
        for index, bits in enumerate(CANDIDATES):
            require(time.time() < deadline, 'Assigned geometry deadline expired; no automatic retry.')
            proposed = packed_candidate(original, bits)
            install(proposed)
            actual = torch.stack([bank[vectors, 0] for bank in banks], dim=1)
            xor = lut[(actual ^ original).long()].sum(-1).to(torch.uint8)
            expected = torch.tensor([int(b >= 0) for b in bits], dtype=torch.uint8, device=device)
            require(torch.equal(actual, proposed) and bool((xor == expected).all()), 'Actual packed XOR differs.')
            weight = layer._decode_weight(torch.bfloat16)
            changed = tiles(weight)
            # Every candidate is decoded at the complete native matrix shape. No
            # local GEMM, additive single-bit approximation or FP32 decoder is used.
            displacement = changed.double() - base_tiles.double()
            cross = 2 * (residual * displacement).sum((1, 2))
            curvature = displacement.square().sum((1, 2))
            sse = (changed.double() - reference).square().sum((1, 2))
            delta = sse - base_sse
            tolerance = 128 * torch.finfo(torch.float64).eps * torch.maximum(base_sse, sse).clamp_min(1.)
            require(bool(torch.isfinite(sse).all()) and bool((abs(delta - cross - curvature) <= tolerance).all()),
                    'Finite-step SSE decomposition differs beyond FP64 reduction error.')
            nonadditivity = torch.zeros_like(delta)
            if 1 <= index <= 64:
                singles[index - 1] = changed
            elif index >= 65:
                additive = singles[bits[0]].double() + singles[32 + bits[1]].double() - 2 * base_tiles.double()
                nonadditivity = (displacement - additive).square().sum((1, 2))
            metrics[index] = torch.stack((delta, cross, curvature, sse, nonadditivity), dim=-1).cpu()
            xors[index] = xor.cpu()
            if index in (0, 1, 64, 65, 1088):
                require(torch.equal(weight[~union], baseline[~union])
                        and torch.equal(weight[:, layer.protected_input_indices], baseline[:, layer.protected_input_indices]),
                        'A candidate affected another RHT support or protected columns.')
                # One case alone must reproduce its tile from the simultaneous
                # disjoint-case decode, at the same full native matrix shape.
                if index:
                    isolated = original.clone(); isolated[0] = proposed[0]
                    install(isolated)
                    alone = layer._decode_weight(torch.bfloat16)
                    require(torch.equal(tiles(alone)[0], changed[0])
                            and torch.equal(tiles(alone)[1:], base_tiles[1:]), 'Independent supports interacted.')
                    del alone
                measured_support.append(index)
            del weight, changed, displacement
            if index % 64 == 0 or index == 1088:
                observer({'candidate': index, 'of': 1089, 'seconds': time.perf_counter() - start})
    finally:
        install(original)
    require(records(layer) == initial and storage_bits(layer) == bits_before
            and torch.equal(layer._decode_weight(torch.bfloat16), baseline)
            and all(p.grad is None and not p.requires_grad for p in layer.parameters()),
            'Geometry observation changed persistent state or did not restore native weights.')
    single_delta = metrics[1:65, :, 0]
    pair_delta = metrics[65:, :, 0].reshape(32, 32, COUNT)
    both_worse = (single_delta[:32, None, :] > 0) & (single_delta[None, 32:, :] > 0)
    both_nonbeneficial = (single_delta[:32, None, :] >= 0) & (single_delta[None, 32:, :] >= 0)
    hidden = both_worse & (pair_delta < 0)
    single_best, single_choice = metrics[:65, :, 0].min(0)
    pair_best, pair_choice = metrics[:, :, 0].min(0)
    summary = {'status': 'completed', 'cases': cases, 'candidate_count_per_vector': 1089,
        'single_best_delta_sum': float(single_best.sum()), 'extended_best_delta_sum': float(pair_best.sum()),
        'vectors_improved_over_single_neighborhood': int((pair_best < single_best).sum()),
        'both_singles_strictly_worse_pair_beneficial_count': int(hidden.sum()),
        'vectors_with_both_singles_strictly_worse_pair_beneficial': int(hidden.any(0).any(0).sum()),
        'both_singles_nonbeneficial_pair_beneficial_count': int((both_nonbeneficial & (pair_delta < 0)).sum()),
        'vectors_without_beneficial_single_but_with_beneficial_pair': int(((single_best == 0) & (pair_best < 0)).sum()),
        'single_best_candidate_indices': single_choice.tolist(), 'extended_best_candidate_indices': pair_choice.tolist(),
        'baseline_sse_per_tile': base_sse.cpu().tolist(), 'native_full_shape_decode': True,
        'observed_outside_support_and_isolation_candidates': measured_support,
        'persistent_13_tensors_restored_exact': True, 'native_weight_restored_exact': True,
        'decoder_frozen_bf16': True, 'persistent_bits_before_after': bits_before,
        'elapsed_seconds': time.perf_counter() - start,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    return {'metrics': metrics, 'actual_stage_xor': xors,
            'candidate_bits': torch.tensor(CANDIDATES, dtype=torch.int16)}, summary
