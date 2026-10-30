"""Small structural proposals and trust measured on actual BF16/group8 weights."""
import math

from selected_bindings import core, torch, SUPPORT, TRUST, tiles

Geometry, project = tiles.Geometry, tiles.project


@torch.no_grad()
def install_banks(layer, packed):
    for stage in range(2):
        layer.get_stage_part_vq_storage(stage, 0).copy_(packed[:, stage:stage + 1])
    layer.clear_decoded_weight_cache()
    layer._clear_parallel_stage_decode_runtime_cache()
    layer._build_parallel_stage_decode_plan()
    actual = layer._get_parallel_stage_grouped_vq_packed(device=packed.device)
    core.require(torch.equal(actual, packed), 'Persistent banks and refreshed grouped banks differ.')


@torch.no_grad()
def native_weight(native):
    with native.native(exact_values=False):
        layer = next(iter(native.selected.values()))
        weight = layer._cached_weight.detach()
        core.require(weight.dtype == torch.bfloat16 and weight.shape == (4096, 12288), 'Use actual original group8 BF16 W.')
    return weight


def tile_scores(geometry, first, second, ids):
    values = []
    for start in range(0, len(ids), 128):
        part = ids[start:start + 128]
        a = geometry.extract(first, part).detach().to('cpu').double()
        b = geometry.extract(second, part).detach().to('cpu').double()
        values.append((a - b).square().sum((-1, -2)))
    return torch.cat(values)


def native_metrics(geometry, baseline, candidate, target, selected):
    columns = geometry.columns.to(baseline.device)
    outside = torch.ones(geometry.tiles, dtype=torch.bool)
    outside[selected.cpu()] = False
    # Comparing every actual tile also proves that disjoint actual native
    # displacements can be accumulated to choose a prefix without binary decode.
    deltas = tile_scores(geometry, candidate, baseline, torch.arange(geometry.tiles))
    core.require(bool((deltas[outside] == 0).all()), 'Native proposal moved a nonselected tile.')
    denominator = before = after = 0.
    for start in range(0, baseline.shape[0], 128):
        p = baseline[start:start + 128, columns].detach().to('cpu').double()
        b = candidate[start:start + 128, columns].detach().to('cpu').double()
        t = target[start:start + 128, columns].detach().to('cpu').double()
        denominator += float(p.square().sum())
        before += float((p - t).square().sum()); after += float((b - t).square().sum())
    protected = torch.ones(baseline.shape[1], dtype=torch.bool, device=baseline.device)
    protected[columns] = False
    core.require(torch.equal(baseline[:, protected], candidate[:, protected]) and denominator > 0
        and all(math.isfinite(v) for v in (denominator, before, after)), 'Native protection or Frobenius metric invalid.')
    displacement = float(deltas.sum())
    return {'baseline_norm_squared': denominator, 'displacement_squared': displacement,
        'relative_displacement': math.sqrt(displacement / denominator),
        'before_target_SSE': before, 'after_target_SSE': after,
        'outside_selected_actual_weight_exact': True, 'protected_actual_weight_exact': True}, deltas


@torch.no_grad()
def restricted_step(layer, native, target, deadline):
    geometry = Geometry(layer)
    core.require(target.dtype == torch.float32 and not target.requires_grad, 'PV target must be detached FP32.')
    layer._build_parallel_stage_decode_plan()
    original = layer._get_parallel_stage_grouped_vq_packed(device=target.device).clone()
    baseline = native_weight(native)
    ids = torch.arange(geometry.tiles)
    scores = tile_scores(geometry, target, baseline, ids)
    core.require(bool(torch.isfinite(scores).all()) and geometry.tiles == 48128, 'Invalid actual down tile scores.')
    selected = torch.argsort(scores, descending=True, stable=True)[:SUPPORT]
    proposal = original.clone(); rows = []
    modes = [(module, module.training) for module in layer.modules()]
    try:
        layer.eval()
        for batch in project(layer, original, target, tile_ids=selected, tile_batch=64):
            core.tick(deadline)
            proposal[batch.vector_indices] = batch.packed
            rows.append(batch.stats)
    finally:
        for module, training in modes:
            module.training = training
    install_banks(layer, proposal)
    full = native_weight(native)
    full_metrics, deltas = native_metrics(geometry, baseline, full, target, selected)
    cumulative = deltas[selected].cumsum(0)
    threshold = TRUST ** 2 * full_metrics['baseline_norm_squared']
    prefix = int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=torch.float64), right=True))
    core.require(0 <= prefix <= SUPPORT, 'Invalid trust prefix length.')
    final = original.clone()
    if prefix:
        indices = geometry.indices(selected[:prefix].to(target.device))[0]
        final[indices] = proposal[indices]
    install_banks(layer, final)
    actual = native_weight(native)
    metrics, _ = native_metrics(geometry, baseline, actual, target, selected[:prefix])
    core.require(metrics['relative_displacement'] <= TRUST, 'Actual final native weight exceeds trust radius.')
    prefix_metrics = dict(metrics)
    rejected = metrics['after_target_SSE'] > metrics['before_target_SSE']
    if rejected:
        install_banks(layer, original)
        actual = native_weight(native)
        core.require(torch.equal(actual, baseline), 'Rejected projection failed to retain actual D-only native baseline.')
        metrics, _ = native_metrics(geometry, baseline, actual, target, selected[:0])
        final = original
    xor = original ^ final
    popcount = torch.tensor([x.bit_count() for x in range(256)], dtype=torch.int64, device=xor.device)
    bits = int(popcount[xor.long()].sum())
    pairs = int((xor != 0).any(-1).any(-1).sum())
    return {'selected_tiles_in_priority_order': selected.tolist(), 'selected_tiles': SUPPORT,
        'highest_target_residual_tile_squared': float(scores[selected[0]]),
        'lowest_selected_target_residual_tile_squared': float(scores[selected[-1]]),
        'selected_actual_native_baseline_target_SSE': float(scores[selected].sum()),
        'local_baseline_target_SSE': sum(row['before_sse'] for row in rows),
        'proposal_local_batches': rows, 'proposal_native_metrics': full_metrics,
        'prefix_native_metrics_before_possible_rejection': prefix_metrics,
        'trust_prefix_tiles': prefix, 'target_distance_rejected': rejected,
        'native_metrics': metrics, 'actual_changed_bits': bits, 'actual_changed_vector_pairs': pairs,
        'effective_discrete_update': bits > 0, 'candidate_single_bit_neighbors': SUPPORT * 2 * 32 * 48,
        'native_group8_BF16_after_D_update': True,
        'trust_used_actual_native_per_tile_FP64_additivity': True,
        'longest_prefix_rule': 'Priority order; cumsum actual disjoint native squared displacement <= trust^2*||W_P||^2. No forced minimum change.',
        'D_only_kept_if_codes_rejected': True}
