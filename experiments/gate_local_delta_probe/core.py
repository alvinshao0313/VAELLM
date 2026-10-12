"""Observe sixteen single-bit neighbors; never accept an update or train weights."""
from __future__ import annotations

import time
import torch
from torch.nn import functional as F

from experiments.gate_student_context_probe.core import Objective, ready, records
from experiments.joint_reconstruction_probe.core import require
from experiments.rotation_factor_probe.capture import tensor_record


def select_candidates(choices):
    require(choices.dtype == torch.uint8 and choices.shape == (1548288,)
            and int(choices.max()) <= 64, 'Require the saved original f927 weight-search choices.')
    candidates = []
    for stage in range(2):
        indices = ((choices > stage * 32) & (choices <= (stage + 1) * 32)).nonzero().flatten()
        require(len(indices) >= 8, 'Each stage needs eight genuine previously selected weight neighbors.')
        for quantile in range(8):
            rank = (2 * quantile + 1) * len(indices) // 16
            vector = int(indices[rank])
            bit = int(choices[vector]) - 1 - stage * 32
            row, block = divmod(vector, 126)
            candidates.append({'stage': stage, 'bit': bit, 'vector': vector,
                'saved_choice': int(choices[vector]), 'within_stage_winner_rank': rank,
                'within_stage_winner_count': len(indices), 'mid_quantile': quantile,
                'rotated_row': row, 'rotated_column_start': block * 32,
                'original_output_start': row // 32 * 32})
    require(len({x['vector'] for x in candidates}) == 16, 'Candidates must be distinct original vectors.')
    return candidates


def sse_delta(before, after, target):
    e = (before.float() - target.float()).double()
    # Match the existing metric's two independently rounded FP32 residuals.
    change = (after.float() - target.float()).double() - e
    return float((2 * e * change + change.square()).sum())


def discrepancy(a, b):
    delta = a.float() - b.float()
    return {'exact': bool(torch.equal(a, b)), 'max_abs': float(delta.abs().max()),
        'different_elements': int(torch.count_nonzero(delta)),
        'relative_l2': float(delta.double().square().sum().sqrt()
                             / b.double().square().sum().sqrt().clamp_min(1e-30))}


def formula(da, e, down_columns, dtype):
    da, e, d = da.to(dtype), e.to(dtype), down_columns.to(dtype)
    cross = 2 * (da * (e @ d)).sum()
    quadratic = ((da.T @ da) * (d.T @ d)).sum()
    return {'cross': float(cross), 'quadratic': float(quadratic), 'delta': float(cross + quadratic)}


def rank_summary(rows, field):
    actual = [r['native_full_sse_delta'] for r in rows]
    proxy = [r[field] for r in rows]
    sign = lambda x: (x > 0) - (x < 0)
    informative = [i for i, r in enumerate(rows) if abs(actual[i]) > r['sse_reduction_tolerance']]
    ordering = {'concordant': 0, 'discordant': 0, 'ties': 0}
    for i in range(len(rows)):
        for j in range(i):
            a, p = sign(actual[i] - actual[j]), sign(proxy[i] - proxy[j])
            ordering['ties' if not a or not p else 'concordant' if a == p else 'discordant'] += 1
    return {'field': field, 'exact_sign_agreements': sum(sign(a) == sign(p) for a, p in zip(actual, proxy)),
        'informative_native_count': len(informative),
        'informative_sign_agreements': sum(sign(actual[i]) == sign(proxy[i]) for i in informative),
        'false_improvement_count': sum(p < 0 < a for a, p in zip(actual, proxy)),
        'native_best_candidate': min(range(len(rows)), key=lambda i: actual[i]),
        'proxy_best_candidate': min(range(len(rows)), key=lambda i: proxy[i]), 'pairwise_order': ordering,
        'max_abs_delta_error': max(abs(a-p) for a, p in zip(actual, proxy))}


@torch.no_grad()
def measure(layer, archive, cache, candidates, *, deadline, observer):
    device = next(layer.parameters()).device
    require(device.type == 'cuda' and not layer.training and all(not p.requires_grad for p in layer.parameters()),
            'Use one frozen native projection on the assigned CUDA device.')
    ready(layer)
    require(layer._parallel_stage_restore_identity and layer.parallel_parts == 1
            and layer.weight_rotation.block_size == 32, 'This support derivation requires identity-layout block32 gate.')
    initial_state = records(layer)
    torch.cuda.reset_peak_memory_stats(device)
    objective = Objective(archive, cache, device).eval()
    x, up, down, target = objective.inputs, objective.up_output, objective.student_down, objective.reference
    require(x.shape == (4289, 4096), 'Use all fixed train32 tokens, not a prefix or heldout.')
    keep = torch.ones(4096, dtype=torch.bool, device=device)
    keep[layer.protected_input_indices] = False
    original_inputs = keep.nonzero().flatten()
    hf = archive['targets']['model.layers.18.mlp.gate_proj']['reference_weight'].to(device)
    rotated_target = layer.weight_rotation(hf[:, keep].float().contiguous(), inverse=False)
    rotated0 = layer._decode_compressed_weight(dtype=torch.bfloat16).clone()
    weight0 = layer._decode_weight(dtype=torch.bfloat16).clone()
    require(torch.equal(weight0, layer._finalize_decoded_weight_from_compressed(rotated0, torch.bfloat16)),
            'Native full weight and native compressed/finalize path disagree.')
    z0 = F.linear(x, weight0)
    a0 = F.silu(z0) * up
    y0 = objective.prediction(weight0)
    require(torch.equal(y0, F.linear(a0, down)), 'The explicit BF16 nodes differ from the existing Objective.')
    e = y0.float() - target.float()
    base_sse = float(e.double().square().sum())
    target_sse = float(target.double().square().sum())
    initial_repeat = discrepancy(objective.prediction(weight0), y0)
    require(initial_repeat['exact'], 'Identical native shape/input must be repeatable before comparing tiny deltas.')
    rows = []
    for index, choice in enumerate(candidates):
        require(time.time() < deadline, 'Assigned deadline expired; no automatic retry or extension.')
        started = time.perf_counter()
        g, stage, bit = choice['vector'], choice['stage'], choice['bit']
        r, c, j0 = choice['rotated_row'], choice['rotated_column_start'], choice['original_output_start']
        j = torch.arange(j0, j0 + 32, device=device)
        columns = original_inputs[c:c + 32]
        bank = layer.get_stage_part_vq_storage(stage, 0)
        saved_byte = int(bank[g, 0, bit // 8])
        bank[g, 0, bit // 8].bitwise_xor_(1 << (bit % 8))
        try:
            ready(layer)
            require(int(layer._get_parallel_stage_grouped_vq_packed(device=device)[g, stage, bit // 8])
                    == saved_byte ^ (1 << (bit % 8)), 'Grouped packed cache did not observe the genuine single-bit mutation.')
            rotated1 = layer._decode_compressed_weight(dtype=torch.bfloat16)
            weight1 = layer._decode_weight(dtype=torch.bfloat16)
            require(torch.equal(weight1, layer._finalize_decoded_weight_from_compressed(rotated1, torch.bfloat16)),
                    'Candidate native decode/finalize path differs.')
            flat0, flat1 = rotated0.view(-1, 32), rotated1.view(-1, 32)
            require(torch.equal(flat0[:g], flat1[:g]) and torch.equal(flat0[g+1:], flat1[g+1:]),
                    'One fixed-decoder bit flip changed another decoded VQ vector.')
            require(torch.equal(weight0[:j0], weight1[:j0]) and torch.equal(weight0[j0+32:], weight1[j0+32:]),
                    'Inverse rotation changed weights outside the derived output block.')
            patch0, patch1 = weight0[j], weight1[j]
            outside = torch.ones(4096, dtype=torch.bool, device=device); outside[columns] = False
            require(torch.equal(patch0[:, outside], patch1[:, outside])
                    and torch.equal(weight0[:, layer.protected_input_indices], weight1[:, layer.protected_input_indices]),
                    'Original input support or protected columns changed.')
            z1 = F.linear(x, weight1)
            a1 = F.silu(z1) * up
            require(torch.equal(a0[:, :j0], a1[:, :j0]) and torch.equal(a0[:, j0+32:], a1[:, j0+32:]),
                    'Unchanged gate rows changed BF16 activations on the same GEMM shape.')
            y1 = objective.prediction(weight1)
            require(torch.equal(y1, F.linear(a1, down)), 'Native Objective and explicit candidate nodes differ.')
            local_z = F.linear(x, patch1)
            local_a = F.silu(local_z) * up[:, j]
            local_full_a = a0.clone(); local_full_a[:, j] = local_a
            local_y = F.linear(local_full_a, down)  # Preserve the complete native down-GEMM shape/rounding.
            da_full = a1[:, j].float() - a0[:, j].float()
            da_local = local_a.float() - a0[:, j].float()
            full_formula = formula(da_full, e, down[:, j], torch.float64)
            local_formula = formula(da_local, e, down[:, j], torch.float64)
            local_formula32 = formula(da_local, e, down[:, j], torch.float32)
            native_delta = sse_delta(y0, y1, target)
            direct_delta = float((y1.float() - target.float()).double().square().sum()) - base_sse
            tolerance = 64 * torch.finfo(torch.float64).eps * max(1., base_sse, abs(base_sse + native_delta))
            require(abs(native_delta-direct_delta) <= tolerance, 'FP64 direct SSE and delta expansion differ beyond reduction error.')
            w_delta = sse_delta(patch0[:, columns], patch1[:, columns], hf[j][:, columns])
            rot_delta = sse_delta(flat0[g], flat1[g], rotated_target[r, c:c+32])
            weight_change = patch1[:, columns].float() - patch0[:, columns].float()
            row = {**choice, 'index': index, 'native_full_sse_delta': native_delta,
                'native_full_nmse_delta': native_delta / target_sse, 'direct_sse_delta': direct_delta,
                'sse_reduction_tolerance': tolerance, 'local_bf16_recompute_sse_delta': sse_delta(y0, local_y, target),
                'local_formula_sse_delta': local_formula['delta'], 'full_deltaA_formula_sse_delta': full_formula['delta'],
                'local_formula_fp32_sse_delta': local_formula32['delta'],
                'formula_components_fp64': {'full_deltaA': full_formula, 'local_deltaA': local_formula},
                'original_weight_sse_delta': w_delta, 'rotated_weight_sse_delta': rot_delta,
                'support': {'outputs': j.cpu().tolist(), 'inputs': columns.cpu().tolist(),
                    'original_changed_entries': int(torch.count_nonzero(weight_change)),
                    'original_changed_rows': int((weight_change != 0).any(1).sum()),
                    'original_changed_columns': int((weight_change != 0).any(0).sum()),
                    'outside_support_exact': True, 'protected_exact': True, 'other_rotated_vectors_exact': True},
                'local_vs_complete_gate_z': discrepancy(local_z, z1[:, j]),
                'local_vs_complete_activation': discrepancy(local_a, a1[:, j]),
                'local_vs_complete_mlp': discrepancy(local_y, y1),
                'native_output_changed_elements': int(torch.count_nonzero(y1 != y0)),
                'seconds': time.perf_counter() - started}
            require(all(torch.isfinite(torch.tensor(row[k])) for k in ('native_full_sse_delta',
                'local_bf16_recompute_sse_delta', 'local_formula_sse_delta', 'original_weight_sse_delta')),
                'Nonfinite measured candidate.')
            rows.append(row); observer(row)
        finally:
            bank[g, 0, bit // 8] = saved_byte
            ready(layer)
    require(records(layer) == initial_state, 'Observation changed persistent codes, decoder, rotation or protection.')
    restored_weight = layer._decode_weight(dtype=torch.bfloat16)
    require(torch.equal(restored_weight, weight0) and torch.equal(objective.prediction(restored_weight), y0),
            'Restoring all candidates did not restore the original native weight/output.')
    return {'status': 'completed', 'candidates': rows, 'initial_repeat': initial_repeat,
        'baseline_sse': base_sse, 'target_sse': target_sse, 'baseline_nmse': base_sse/target_sse,
        'train_input': tensor_record(x), 'target': tensor_record(target), 'persistent_state_restored_exact': True,
        'native_weight_and_output_restored_exact': True,
        'rank_comparisons': [rank_summary(rows, field) for field in ('local_formula_sse_delta',
            'full_deltaA_formula_sse_delta', 'local_bf16_recompute_sse_delta', 'local_formula_fp32_sse_delta')],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
