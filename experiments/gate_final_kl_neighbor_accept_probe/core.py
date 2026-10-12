"""Generate all fixed proposals from one original final-KL gradient; search is reused."""
from __future__ import annotations

import gc
import logging
import time
import weakref

import torch

from experiments.exact_bit_neighbor_probe.core import flip_grouped, packed_decode
from experiments.gate_final_kl_accept_probe.core import score
from experiments.gate_final_kl_neighbor_probe.core import gradient_at_cache
from experiments.gate_group_accept_probe.core import GROUPS, VECTORS_PER_GROUP, apply_choices, group_inventory
from experiments.gate_student_context_probe.core import records
from experiments.gate_student_context_probe.data import SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import native_policy, version_inventory
from experiments.rotation_factor_probe.capture import require, tensor_record
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint

CHUNK = 4 * VECTORS_PER_GROUP
TOTAL = GROUPS * VECTORS_PER_GROUP


@torch.no_grad()
def enumerate_all(gate, rotated_gradient, reference, reference_choices, *, deadline, observer):
    require(gate._parallel_stage_restore_identity and gate._parallel_stage_layout == [(0, 0), (1, 0)]
            and all(p.dtype == torch.bfloat16 and not p.requires_grad for p in gate._parallel_stage_decoder.parameters()),
            'Preserve the finalized native two-stage BF16 decoder and identity vector layout.')
    packed = gate._get_parallel_stage_grouped_vq_packed(device=rotated_gradient.device)
    require(tuple(packed.shape) == (TOTAL, 2, 4) and packed.dtype == torch.uint8
            and tuple(rotated_gradient.shape) == (12288, 4032) and TOTAL % CHUNK == 0,
            'Use all original384 groups and exactly the verified16128-row decoder shape in every chunk.')
    gradient = rotated_gradient.contiguous().view(TOTAL, 32)
    choices = torch.zeros(TOTAL, dtype=torch.uint8, device=gradient.device)
    group_scores = []; chunk_reports = []; first_check = None
    started = time.time()
    for chunk_index, start in enumerate(range(0, TOTAL, CHUNK)):
        begin = time.time(); end = start + CHUNK
        original = packed[start:end]; baseline = packed_decode(gate, original); g = gradient[start:end]
        best = torch.zeros(CHUNK, dtype=torch.float64, device=g.device)
        selected_codes = torch.zeros(CHUNK, dtype=torch.uint8, device=g.device)
        chosen_outputs = baseline.clone()
        for candidate in range(64):
            require(time.time() < deadline, 'Enumeration exceeded the assigned deadline; do not truncate proposals.')
            values = packed_decode(gate, flip_grouped(original, candidate))
            delta = values.float() - baseline.float()
            dots = (g.double() * delta.double()).sum(-1)
            require(bool(torch.isfinite(dots).all()), 'Nonfinite final-KL gradient proposal score.')
            improved = dots < best
            selected_codes = torch.where(improved, candidate + 1, selected_codes).to(torch.uint8)
            best = torch.where(improved, dots, best)
            chosen_outputs = torch.where(improved[:, None], values, chosen_outputs)
        selected = original.clone()
        for stage in range(2): apply_choices(selected[:, stage:stage + 1], selected_codes, stage)
        repeated = packed_decode(gate, selected)
        independently_reduced = (g.double() * (repeated.float() - baseline.float()).double()).sum(-1)
        require(torch.equal(repeated, chosen_outputs) and torch.equal(independently_reduced, best)
                and bool((best <= 0).all()) and bool((best[selected_codes > 0] < 0).all()),
                'Mixed selected codes must reproduce the same-shape scored outputs and strict-negative scores.')
        choices[start:end] = selected_codes
        scores = [float(best[i * VECTORS_PER_GROUP:(i + 1) * VECTORS_PER_GROUP].sum()) for i in range(4)]
        group_scores.extend(scores)
        if chunk_index == 0:
            actual_choices = selected_codes.cpu()
            require(reference_choices.shape == actual_choices.shape, 'Four-group reference geometry differs.')
            first_check = {'reference_choices': tensor_record(reference_choices), 'actual_choices': tensor_record(actual_choices),
                'choices_equal': torch.equal(actual_choices, reference_choices),
                'different_vector_choices': int((actual_choices != reference_choices).sum()),
                'proxy_group_sums': scores,
                'proxy_group_differences': [value - row['rotated_per_vector_proxy_sum'] for value, row in zip(scores, reference['groups'])],
                'boundary': 'Independent-run derivative/proxy/choice differences are observations, not automatic failures. No old choices replace the actual current derivative proposals.'}
        row = {'chunk': chunk_index, 'vector_start': start, 'vector_end': end, 'decoder_rows': CHUNK,
               'selected_vectors': int((selected_codes > 0).sum()), 'proxy_sum': float(best.sum()),
               'same_shape_selected_decode_and_score_exact': True, 'seconds': time.time() - begin}
        chunk_reports.append(row); observer('enumeration_chunk', row)
    result = choices.cpu()
    return result, {'vectors': TOTAL, 'groups': GROUPS, 'chunks': len(chunk_reports), 'chunk_vectors': CHUNK,
        'candidates_per_vector': 64, 'candidate_count': TOTAL * 64, 'selected_vectors': int((result > 0).sum()),
        'candidate_order': list(range(64)), 'tie_rule': 'Original0 wins nonnegative values; earlier bit wins equal negative values.',
        'group_proxy_sums': group_scores, 'group_inventory': group_inventory(result),
        'first_chunk_reference': first_check, 'chunk_reports': chunk_reports, 'seconds': time.time() - started}


@torch.no_grad()
def generate(data, batch, teacher, previous, reference_choices, *, deadline, observer):
    source = data['source_data']; device = torch.device('cuda:0'); started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True, expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Generate proposals only from fresh f927.')
    model.eval().requires_grad_(False); model.config.use_cache = False; gate = model.get_submodule(TARGET)
    require(records(gate) == source['expected_gate_state'], 'Initial thirteen gate tensors differ.')
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    policy = native_policy(model, source['model_policy']); frozen = version_inventory(model, exclude_gate=True); initial = records(gate)
    require(len(frozen) == 3410 and len(initial) == 13, 'Unexpected frozen native inventory.')
    weight0 = gate._cached_weight.detach().clone()
    before = score(model, batch, teacher, deadline=deadline)
    require(before['causal_tokens'] == 4257, 'Use the entire original32 training documents.')
    gradient, grad_report = gradient_at_cache(model, gate, weight0, batch, teacher, deadline=deadline, observer=observer)
    keep = torch.ones(4096, dtype=torch.bool, device=device); keep[gate.protected_input_indices] = False
    unprotected = gradient[:, keep].contiguous()
    rotated = gate.weight_rotation(unprotected, inverse=False)
    choices, enumeration = enumerate_all(gate, rotated, previous['diagnostic'], reference_choices, deadline=deadline, observer=observer)
    require(records(gate) == initial and torch.equal(gate._cached_weight, weight0)
            and version_inventory(model, exclude_gate=True) == frozen
            and native_policy(model, source['model_policy']) == policy
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Generating proposals mutated persistent state or failed to restore the cache leaf.')
    current_rows = grad_report['documents']; old_rows = previous['diagnostic']['gradient']['documents']
    comparison = {'original_forward_equal': before == previous['diagnostic']['before'],
        'gradient_accumulator_statistics_equal': grad_report['accumulator'] == previous['diagnostic']['gradient']['accumulator'],
        'gradient_documents_equal': current_rows == old_rows,
        'max_abs_gradient_forward_kl_difference': max(abs(a['kl_sum'] - b['kl_sum']) for a, b in zip(current_rows, old_rows)),
        'max_abs_gradient_vs_current_no_grad_kl_difference': max(abs(a['kl_sum'] - b['kl_sum']) for a, b in zip(current_rows, before['documents'])),
        'boundary': 'Statistics do not constitute a full saved-gradient byte comparison; actual independent-run differences remain visible.'}
    peak = torch.cuda.max_memory_allocated(device); reserved = torch.cuda.max_memory_reserved(device)
    model_ref, gate_ref = weakref.ref(model), weakref.ref(gate)
    del gradient, unprotected, rotated, keep, weight0, gate, model
    gc.collect(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    require(model_ref() is None and gate_ref() is None, 'Release the proposal student before the independent fresh search load.')
    return choices, {'status': 'completed', 'source_checkpoint_id': SOURCE_ID, 'gradient_evaluations': 1,
        'before': before, 'gradient': grad_report, 'enumeration': enumeration, 'reference_observations': comparison,
        'warmup': warmup, 'native_policy': policy, 'all13_bytes_frozen': True, 'frozen_complement_count': 3410,
        'frozen_complement_versions_and_pointers_exact': True, 'all_model_parameters_require_grad_false_and_grad_none': True,
        'model_and_gate_weakrefs_dead_before_search': True, 'allocated_after_release': torch.cuda.memory_allocated(device),
        'reserved_after_release': torch.cuda.memory_reserved(device), 'peak_allocated_bytes': peak,
        'peak_reserved_bytes': reserved, 'seconds': time.time() - started}
