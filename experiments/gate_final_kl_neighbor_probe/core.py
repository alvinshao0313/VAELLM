"""Four fixed groups of one-bit proposals scored by a frozen final-KL derivative."""
from __future__ import annotations

import logging
import math
import time

import torch

from experiments.exact_bit_neighbor_probe.core import flip_grouped, packed_decode
from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_final_kl_accept_probe.core import score
from experiments.gate_final_kl_gradient_probe.core import cached_call_guard, dot64, tensor_stats
from experiments.gate_group_accept_probe.core import BANKS, VECTORS_PER_GROUP, apply_choices
from experiments.gate_local_delta_probe.core import discrepancy
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_student_context_probe.data import SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import (
    document_batch, native_policy, probability_metrics, selected_logits, version_inventory,
)
from experiments.rotation_factor_probe.capture import require
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint

GROUPS = (0, 1, 2, 3)
SINGLE_CHECKS = tuple({'group': g, 'vector': g * VECTORS_PER_GROUP + offset, 'candidate': candidate}
                      for g, offset, candidate in zip(GROUPS, (0, 125, 126, 4031), (0, 31, 32, 63)))


@torch.no_grad()
def adjoint_check(rotation, gradient, displacement):
    """FP32 linear rotation identity, explicitly excluding native BF16 rounding."""
    require(gradient.dtype == displacement.dtype == torch.float32, 'Adjoint check must use independent FP32 transforms.')
    transformed = rotation(gradient, inverse=False)
    restored = rotation(displacement, inverse=True)
    left, right = dot64(transformed, displacement), dot64(gradient, restored)
    scale = math.sqrt(tensor_stats(gradient)['squared_l2'] * tensor_stats(displacement)['squared_l2'])
    eps = torch.finfo(torch.float32).eps
    # Two sides have 5 butterfly levels each. The two evaluation paths also
    # normalize each axis; 64*eps conservatively covers their FP32 operations.
    gamma = 64 * eps / (1 - 64 * eps)
    tolerance = gamma * scale
    require(abs(left - right) <= tolerance, 'FP32 adjoint discrepancy exceeds its norm-scaled rounding bound.')
    return {'left_rotated_dot': left, 'right_original_dot': right, 'absolute_error': abs(left - right),
            'norm_product': scale, 'error_over_norm_product': abs(left - right) / scale if scale else 0.,
            'tolerance': tolerance, 'gamma64_eps': gamma, 'dot_reduction_dtype': 'float64',
            'transform_dtype': 'float32', 'native_bf16_rounding_included': False}


def gradient_at_cache(model, gate, weight, batch, teacher, *, deadline, observer):
    """Same validated BF16-leaf/FP32-accumulation path, without its alpha probes."""
    leaf = weight.detach().requires_grad_(True); gate._cached_weight = leaf
    total = torch.zeros_like(weight, dtype=torch.float32); rows = []; started = time.time()
    handle, calls = cached_call_guard(gate, leaf)
    try:
        with torch.enable_grad():
            for index in range(32):
                require(time.time() < deadline, 'Assigned deadline expired.')
                item = document_batch(batch, index, weight.device)
                logits = model(**item, use_cache=False).logits
                selected, labels = selected_logits(logits, item)
                require(torch.equal(labels.cpu(), teacher[index]['labels']), 'Causal teacher labels changed.')
                kl, nll = probability_metrics(selected, teacher[index])
                grad, = torch.autograd.grad(kl.double().sum(), leaf)
                require(grad.dtype == torch.bfloat16 and bool(torch.isfinite(grad).all()), 'Invalid BF16 leaf gradient.')
                total.add_(grad.float())
                row = {'index': index, 'causal_tokens': len(labels), 'kl_sum': float(kl.detach().double().sum()),
                       'nll_sum': float(nll.detach().double().sum()), 'gradient': tensor_stats(grad.detach())}
                rows.append(row); observer('gradient', row)
                del item, logits, selected, labels, kl, nll, grad
            require(gate._cached_weight is leaf and calls['count'] == 32 and leaf.grad is None, 'Cache leaf was not reused for all32 documents.')
    finally:
        handle.remove(); gate._cached_weight = weight
    summary = tensor_stats(total)
    require(summary['nonzero'] > 0 and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Unexpected gradient destination or zero derivative.')
    return total, {'documents': rows, 'accumulator': summary, 'actual_cache_hits': calls['count'],
                   'seconds': time.time() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(weight.device),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved(weight.device)}


@torch.no_grad()
def enumerate_neighbors(gate, rotated_gradient, *, deadline):
    count = len(GROUPS) * VECTORS_PER_GROUP
    require(gate._parallel_stage_restore_identity and gate._parallel_stage_layout == [(0, 0), (1, 0)]
            and all(p.dtype == torch.bfloat16 and not p.requires_grad for p in gate._parallel_stage_decoder.parameters()),
            'Keep the actual finalized BF16 two-stage decoder; no promotion, scale fusion or trainable bits.')
    packed = gate._get_parallel_stage_grouped_vq_packed(device=rotated_gradient.device)[:count].clone()
    baseline = packed_decode(gate, packed)
    g = rotated_gradient.contiguous().view(-1, 32)[:count]
    best = torch.zeros(count, dtype=torch.float64, device=g.device)
    choices = torch.zeros(count, dtype=torch.uint8, device=g.device)
    chosen_outputs = baseline.clone(); singles = {}; started = time.time()
    for candidate in range(64):
        require(time.time() < deadline, 'Neighbor enumeration exceeded the assigned deadline.')
        values = packed_decode(gate, flip_grouped(packed, candidate))
        delta = values.float() - baseline.float()
        scores = (g.double() * delta.double()).sum(-1)
        require(bool(torch.isfinite(scores).all()), 'Nonfinite rotated-coordinate proxy.')
        improved = scores < best  # Original0 remains for nonnegative values; earlier candidates win ties.
        choices = torch.where(improved, candidate + 1, choices).to(torch.uint8)
        best = torch.where(improved, scores, best)
        chosen_outputs = torch.where(improved[:, None], values, chosen_outputs)
        for item in SINGLE_CHECKS:
            if item['candidate'] == candidate:
                singles[item['vector']] = {'output': values[item['vector']].clone(), 'dot': float(scores[item['vector']])}
    selected = packed.clone()
    for stage in range(2): apply_choices(selected[:, stage:stage + 1], choices, stage)
    repeated = packed_decode(gate, selected)
    require(torch.equal(repeated, chosen_outputs) and bool((best <= 0).all())
            and bool((best[choices > 0] < 0).all()), 'Chosen codes differ from the same-shape scored outputs.')
    independently_reduced = (g.double() * (repeated.float() - baseline.float()).double()).sum(-1)
    require(torch.equal(independently_reduced, best), 'Selected proxy scores do not match the actual packed outputs.')
    return choices, baseline, repeated, best, singles, {'vectors': count, 'candidates_per_vector': 64,
        'candidate_count': count * 64, 'selected_vectors': int((choices > 0).sum()),
        'candidate_order': list(range(64)), 'tie_rule': 'Original code wins nonnegative scores; earlier bit wins equal negative scores.',
        'same_shape_selected_decode_exact': True, 'selected_proxy_recomputed_exact': True,
        'seconds': time.time() - started}


@torch.no_grad()
def probe(data, batch, teacher, previous, *, deadline, observer):
    device = torch.device('cuda:0'); source = data['source_data']
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True, expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Require fresh f927.')
    model.eval().requires_grad_(False); model.config.use_cache = False; gate = model.get_submodule(TARGET)
    require(records(gate) == source['expected_gate_state'], 'Original gate identity differs.')
    torch.cuda.reset_peak_memory_stats(device)
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    policy = native_policy(model, source['model_policy']); frozen = version_inventory(model, exclude_gate=True)
    initial = records(gate); require(len(frozen) == 3410 and len(initial) == 13, 'Unexpected frozen inventory.')
    banks = [gate.get_stage_part_vq_storage(s, 0).clone() for s in range(2)]
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj) for i in range(36)]
    def prime():
        ready(gate)
        result = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True, group_size=8, compute_device=device)
        require(result['warmed'] == 36 and result['failed'] == 0, 'Native group8 cache rebuild failed.')
        return result
    w0 = gate._cached_weight.detach().clone()
    def logits0(): return model(**document_batch(batch, 0, device), use_cache=False).logits.detach().cpu()
    original_logits = logits0(); before = score(model, batch, teacher, deadline=deadline)
    require(before == previous['diagnostic']['before'] and before['causal_tokens'] == 4257, 'Identical native source/inputs must reproduce the verified baseline.')
    memory_before = {'allocated_bytes': torch.cuda.memory_allocated(device), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                     'reserved_bytes': torch.cuda.memory_reserved(device), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    torch.cuda.reset_peak_memory_stats(device)
    gradient, grad_report = gradient_at_cache(model, gate, w0, batch, teacher, deadline=deadline, observer=observer)
    require(all({k: row[k] for k in ('index', 'causal_tokens', 'kl_sum', 'nll_sum')} == before['documents'][i]
                for i, row in enumerate(grad_report['documents'])) and torch.equal(logits0(), original_logits),
            'New derivative forward or original cache restore differs from its validated native baseline.')
    keep = torch.ones(4096, dtype=torch.bool, device=device); keep[gate.protected_input_indices] = False
    compressed_gradient = gradient[:, keep].contiguous()
    rotated_gradient = gate.weight_rotation(compressed_gradient, inverse=False)
    native_z0 = gate._decode_compressed_weight(dtype=torch.bfloat16).clone()
    single_decode0 = gate._finalize_decoded_weight_from_compressed(native_z0, torch.bfloat16)
    native_single_vs_group8 = discrepancy(single_decode0, w0); del single_decode0
    choices, chunk0, chosen_chunk, best, singles, enumeration = enumerate_neighbors(gate, rotated_gradient, deadline=deadline)
    count = choices.numel(); native_blocks0 = native_z0.view(-1, 32)
    chunk_native_before = discrepancy(chunk0, native_blocks0[:count])
    delta_z = torch.zeros_like(compressed_gradient)
    delta_z.view(-1, 32)[:count] = chosen_chunk.float() - chunk0.float()
    adjoint = adjoint_check(gate.weight_rotation, compressed_gradient, delta_z)
    del delta_z, compressed_gradient, rotated_gradient
    def restore():
        for s in range(2): gate.get_stage_part_vq_storage(s, 0).copy_(banks[s])
        prime()
        require(records(gate) == initial and torch.equal(gate._cached_weight, w0) and torch.equal(logits0(), original_logits),
                'Independent proposal restore changed original codes/cache/full logits.')
    single_reports = []
    for item in SINGLE_CHECKS:
        require(time.time() < deadline, 'Single-bit path checks exceeded the deadline.')
        vector = item['vector']; stage, bit = divmod(item['candidate'], 32)
        try:
            gate.get_stage_part_vq_storage(stage, 0)[vector, 0, bit // 8].bitwise_xor_(1 << (bit % 8)); prime()
            native_z1 = gate._decode_compressed_weight(dtype=torch.bfloat16)
            native_blocks1 = native_z1.view(-1, 32)
            require(torch.equal(native_blocks0[:vector], native_blocks1[:vector])
                    and torch.equal(native_blocks0[vector + 1:], native_blocks1[vector + 1:]), 'Single bit changed another native decoded vector.')
            delta_w = gate._cached_weight.float() - w0.float()
            actual_xor = [popcounts(banks[s], gate.get_stage_part_vq_storage(s, 0)) for s in range(2)]
            require(actual_xor == [int(s == stage) for s in range(2)], 'Native single-bit XOR inventory differs.')
            lo, hi = item['group'] * 32, (item['group'] + 1) * 32
            columns = keep.nonzero().flatten()[(vector % 126) * 32:(vector % 126 + 1) * 32]
            outside = torch.ones(4096, dtype=torch.bool, device=device); outside[columns] = False
            require(not bool(torch.count_nonzero(delta_w[:lo])) and not bool(torch.count_nonzero(delta_w[hi:]))
                    and not bool(torch.count_nonzero(delta_w[lo:hi, outside])), 'Single-bit original-coordinate support differs from the actual32x32 tile.')
            row = {**item, 'stage': stage, 'bit': bit, 'rotated_proxy_dot': singles[vector]['dot'],
                'native_weight_dot': dot64(gradient, delta_w), 'weight_delta': tensor_stats(delta_w),
                'chunk_native_vector_before': discrepancy(chunk0[vector], native_blocks0[vector]),
                'chunk_native_vector_after': discrepancy(singles[vector]['output'], native_blocks1[vector]),
                'actual_stage_xor': actual_xor, 'native_rotated_other_vectors_exact': True, 'actual_original32x32_support_verified': True}
            del native_z1, native_blocks1, delta_w
        finally:
            restore()
        row['original_codes_cache_logits_restored_exact'] = True; single_reports.append(row); observer('single_bit', row)
    group_reports = []
    for group in GROUPS:
        begin = time.time(); start, end = group * VECTORS_PER_GROUP, (group + 1) * VECTORS_PER_GROUP
        require(records(gate) == initial and time.time() < deadline, 'Each fixed group must start independently at f927.')
        selected = choices[start:end]
        try:
            for stage in range(2): apply_choices(gate.get_stage_part_vq_storage(stage, 0)[start:end], selected, stage)
            cache = prime(); endpoint = gate._cached_weight
            state = records(gate)
            require(all(state[k] == value for k, value in initial.items() if k not in BANKS), 'A noncode gate tensor changed.')
            delta_w = endpoint.float() - w0.float()
            lo, hi = group * 32, (group + 1) * 32
            require(not bool(torch.count_nonzero(delta_w[:lo])) and not bool(torch.count_nonzero(delta_w[hi:]))
                    and not bool(torch.count_nonzero(delta_w[:, gate.protected_input_indices])), 'Group support/protected columns changed.')
            native_z1 = gate._decode_compressed_weight(dtype=torch.bfloat16)
            single1 = gate._finalize_decoded_weight_from_compressed(native_z1, torch.bfloat16)
            chunk_native = discrepancy(chosen_chunk[start:end], native_z1.view(-1, 32)[start:end])
            single_group = discrepancy(single1, endpoint)
            actual_dot = dot64(gradient, delta_w); proxy = float(best[start:end].sum())
            stage_xor = [popcounts(banks[s], gate.get_stage_part_vq_storage(s, 0)) for s in range(2)]
            require(stage_xor == [int(((selected > s * 32) & (selected <= (s + 1) * 32)).sum()) for s in range(2)],
                    'Actual candidate banks differ from the one-bit-per-vector proposal.')
            handle, calls = cached_call_guard(gate, endpoint)
            try: metrics = score(model, batch, teacher, deadline=deadline)
            finally: handle.remove()
            require(calls['count'] == 32 and gate._cached_weight is endpoint, 'Actual group8 endpoint was not reused.')
            row = {'group': group, 'selected_vectors': int((selected > 0).sum()),
                'stage_xor': stage_xor,
                'choice_histogram': torch.bincount(selected.long(), minlength=65).cpu().tolist(),
                'rotated_per_vector_proxy_sum': proxy, 'native_original_coordinate_weight_dot': actual_dot,
                'native_dot_minus_proxy': actual_dot - proxy, 'native_weight_delta': tensor_stats(delta_w),
                'chunk_vs_native_single_compressed_after': chunk_native, 'single_layer_vs_group8_weight_after': single_group,
                'metrics': metrics, 'observed_kl_delta_sum': metrics['kl_sum'] - before['kl_sum'],
                'observed_nll_delta_sum': metrics['nll_sum'] - before['nll_sum'], 'actual_cache_hits': calls['count'],
                'cache': cache, 'persistent_noncode_gate_exact': True}
            del endpoint, delta_w, native_z1, single1
        finally:
            restore()
        row.update(original_codes_cache_logits_restored_exact=True, seconds=time.time() - begin)
        group_reports.append(row); observer('group', row)
    final = score(model, batch, teacher, deadline=deadline)
    require(final == before and records(gate) == initial and version_inventory(model, exclude_gate=True) == frozen
            and native_policy(model, source['model_policy']) == policy
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Final frozen/restoration contract failed.')
    report = {'status': 'completed', 'source_checkpoint_id': SOURCE_ID, 'before': before, 'gradient': grad_report,
        'enumeration': enumeration, 'adjoint_fp32': adjoint, 'single_bit_checks': single_reports, 'groups': group_reports,
        'chunk_vs_native_single_compressed_before': chunk_native_before, 'single_layer_vs_group8_weight_before': native_single_vs_group8,
        'final_restored': final, 'warmup': warmup, 'native_policy': policy, 'memory_before_gradient': memory_before,
        'peak_allocated_after_warmup_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_after_warmup_bytes': torch.cuda.max_memory_reserved(device), 'frozen_complement_count': len(frozen),
        'frozen_complement_versions_and_pointers_exact': True, 'all13_final_bytes_exact': True,
        'all_model_parameters_require_grad_false_and_grad_none': True, 'full_train_restore_exact': True,
        'boundary': 'Rotated FP32-adjoint scores use packed BF16 stage sums; they are proposal proxies, not exact original-coordinate BF16 enumeration. Individual vector changes need not add after native inverse/BF16 rounding. Only four fixed independent groups, no heldout or acceptance.'}
    return choices.cpu(), report
