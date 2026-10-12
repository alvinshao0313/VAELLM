"""A fixed larger, disjoint tile population for native one/two-bit SSE search."""
from __future__ import annotations

import time

import torch

from experiments.cross_stage_bit_geometry.core import CANDIDATES, SEED, packed_candidate, support as old_support
from experiments.gate_student_context_probe.core import ready, records
from experiments.rotation_factor_probe.capture import require

COUNT = 1024
ARMS = ('original', 'single', 'pair')


def support():
    cases = old_support()
    seen = {x['output_group'] * 126 + x['input_block'] for x in cases}
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    ordered = torch.randperm(384 * 126, generator=generator).tolist()
    extra = [n for n in ordered if n not in seen][:COUNT - len(cases)]
    offsets = torch.randint(0, 32, (len(extra),), generator=generator).tolist()
    for n, row in zip(extra, offsets, strict=True):
        group, column = divmod(n, 126)
        cases.append({'output_group': group, 'row_offset': row, 'input_block': column,
                      'vector': (group * 32 + row) * 126 + column})
    require(cases[:32] == old_support() and len(cases) == COUNT
            and len({x['output_group'] * 126 + x['input_block'] for x in cases}) == COUNT
            and len({x['vector'] for x in cases}) == COUNT, 'Support must retain32 and contain1024 distinct RHT tiles.')
    return cases


def coordinates(layer, cases):
    device = next(layer.parameters()).device
    vectors = torch.tensor([x['vector'] for x in cases], device=device)
    rows = torch.tensor([x['output_group'] * 32 for x in cases], device=device)[:, None] + torch.arange(32, device=device)
    keep = torch.ones(4096, device=device, dtype=torch.bool); keep[layer.protected_input_indices] = False
    compressed = torch.tensor([x['input_block'] * 32 for x in cases], device=device)[:, None] + torch.arange(32, device=device)
    return vectors, rows, keep.nonzero().flatten()[compressed]


def install(layer, vectors, packed):
    for stage in range(2):
        layer.get_stage_part_vq_storage(stage, 0)[vectors, 0] = packed[:, stage]
    ready(layer)


def chosen_packed(original, choices):
    require(tuple(choices.shape) == (COUNT,) and choices.min() >= 0 and choices.max() < len(CANDIDATES), 'Invalid choices.')
    bits = torch.tensor(CANDIDATES, device=original.device)[choices.long()]
    result = original.clone()
    for stage in range(2):
        selected = bits[:, stage] >= 0; rows = selected.nonzero().flatten(); bit = bits[selected, stage]
        result[rows, stage, bit // 8] ^= (1 << (bit % 8)).to(torch.uint8)
    return result


@torch.no_grad()
def search(layer, target, cases, *, deadline, observer):
    ready(layer); require(cases == support() and all(not p.requires_grad for p in layer.parameters()), 'Fixed frozen source required.')
    initial = records(layer); vectors, rows, columns = coordinates(layer, cases)
    original = torch.stack([layer.get_stage_part_vq_storage(s, 0)[vectors, 0].clone() for s in range(2)], dim=1)
    weight0 = layer._decode_weight(torch.bfloat16)
    tiles0 = weight0[rows[:, :, None], columns[:, None, :]].clone()
    reference = target.to(weight0.device)[rows[:, :, None], columns[:, None, :]].double()
    base_sse = (tiles0.double() - reference).square().sum((1, 2))
    best = base_sse.clone(); choices = torch.zeros(COUNT, device=weight0.device, dtype=torch.int16)
    singles = torch.empty((64, COUNT), dtype=torch.float64, device=weight0.device)
    hidden = torch.zeros(COUNT, dtype=torch.int64, device=weight0.device)
    union = torch.zeros_like(weight0, dtype=torch.bool); union[rows[:, :, None], columns[:, None, :]] = True
    require(torch.equal(layer._decode_weight(torch.bfloat16), weight0), 'Native full-matrix baseline is not repeatable.')
    choice_rows, sse_rows, tile_rows = [choices.cpu().clone()], [base_sse.cpu()], [tiles0.cpu()]
    started = time.time(); checked = []
    try:
        for index, bits in enumerate(CANDIDATES[1:], 1):
            require(time.time() < deadline, 'Fixed search deadline expired; no automatic retry.')
            proposed = packed_candidate(original, bits); install(layer, vectors, proposed)
            weight = layer._decode_weight(torch.bfloat16)
            tiles = weight[rows[:, :, None], columns[:, None, :]]
            sse = (tiles.double() - reference).square().sum((1, 2))
            require(bool(torch.isfinite(sse).all()), 'Nonfinite native SSE.')
            if index <= 64:
                singles[index - 1] = sse - base_sse
            else:
                hidden += ((singles[bits[0]] > 0) & (singles[32 + bits[1]] > 0) & (sse < base_sse)).long()
            improved = sse < best
            best = torch.where(improved, sse, best); choices = torch.where(improved, index, choices).to(torch.int16)
            if index in (1, 64, 65, 1088):
                require(torch.equal(weight[~union], weight0[~union])
                        and torch.equal(weight[:, layer.protected_input_indices], weight0[:, layer.protected_input_indices]),
                        'A candidate changed weights outside the disjoint unprotected tile union.')
                isolated = original.clone(); isolated[0] = proposed[0]; install(layer, vectors, isolated)
                alone = layer._decode_weight(torch.bfloat16)[rows[:, :, None], columns[:, None, :]]
                require(torch.equal(alone[0], tiles[0]) and torch.equal(alone[1:], tiles0[1:]),
                        'Distinct input tiles within shared output groups interacted.')
                checked.append(index); del alone
            if index in (64, 1088):
                selected = chosen_packed(original, choices); install(layer, vectors, selected)
                endpoint = layer._decode_weight(torch.bfloat16)[rows[:, :, None], columns[:, None, :]]
                actual_sse = (endpoint.double() - reference).square().sum((1, 2))
                require(torch.equal(actual_sse, best), 'Combined disjoint selected tiles differ from individually scored candidates.')
                choice_rows.append(choices.cpu().clone()); sse_rows.append(best.cpu()); tile_rows.append(endpoint.cpu()); del endpoint
            if index % 64 == 0 or index == 1088:
                observer({'phase': 'search', 'candidate': index, 'of': 1089, 'seconds': time.time() - started})
            del weight, tiles
    finally:
        install(layer, vectors, original)
    require(records(layer) == initial and torch.equal(layer._decode_weight(torch.bfloat16), weight0), 'Original source was not restored.')
    raw = {'choices': torch.stack(choice_rows), 'search_sse_per_tile': torch.stack(sse_rows),
           'candidate_bits': torch.tensor(CANDIDATES, dtype=torch.int16)}
    return raw, torch.stack(tile_rows), {'status': 'completed', 'vectors': COUNT, 'candidates': 1089,
        'both_singles_worse_pair_beneficial_count': int(hidden.sum()), 'vectors_with_hidden_pair': int((hidden > 0).sum()),
        'vectors_extended_better': int((sse_rows[2] < sse_rows[1]).sum()),
        'vectors_single_local_minimum_escaped': int(((sse_rows[1] == sse_rows[0]) & (sse_rows[2] < sse_rows[0])).sum()),
        'sse_by_arm': {arm: float(values.sum()) for arm, values in zip(ARMS, sse_rows)},
        'support_isolation_checked_candidates': checked, 'combined_selected_sse_exact': True,
        'source13_and_original_full_weight_restored_exact': True, 'seconds': time.time() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
