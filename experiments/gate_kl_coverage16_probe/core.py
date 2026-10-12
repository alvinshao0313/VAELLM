"""Same finite BF16 neighbors and initial mixed128 KL derivative, first16 groups."""
from __future__ import annotations

import logging
import time

import torch

from experiments.exact_bit_neighbor_probe.core import flip_grouped, packed_decode
from experiments.gate_final_kl_gradient_probe.core import dot64, tensor_stats
from experiments.gate_final_kl_neighbor_probe.core import adjoint_check
from experiments.gate_kl_calibration_probe.core import gradient128, score_population
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_tail_objective_probe.core import document_batch, native_policy, version_inventory
from experiments.gate_kl_coverage16_probe.common import (
    ARM, BANKS, CHUNK, GROUPS, SOURCE_ID, TARGET, VECTORS, choice_inventory, replay_codes,
)
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint


@torch.no_grad()
def enumerate16(gate, rotated_gradient, *, deadline, observer):
    """The existing enumerate_all loop, limited to four unchanged16128-row chunks."""
    require(gate._parallel_stage_restore_identity and gate._parallel_stage_layout == [(0, 0), (1, 0)]
            and all(p.dtype == torch.bfloat16 and not p.requires_grad
                    for p in gate._parallel_stage_decoder.parameters()), 'Preserve the finalized native BF16 decoder.')
    packed = gate._get_parallel_stage_grouped_vq_packed(device=rotated_gradient.device)
    require(tuple(packed.shape) == (1548288, 2, 4) and packed.dtype == torch.uint8
            and tuple(rotated_gradient.shape) == (12288, 4032), 'Original packed/rotation geometry differs.')
    gradient = rotated_gradient.contiguous().view(-1, 32)
    choices = torch.zeros(VECTORS, dtype=torch.uint8, device=gradient.device)
    output_delta = torch.zeros_like(rotated_gradient)
    chunks = []; started = time.time()
    for start in range(0, VECTORS, CHUNK):
        tick = time.time(); end = start + CHUNK
        original = packed[start:end]; baseline = packed_decode(gate, original); g = gradient[start:end]
        best = torch.zeros(CHUNK, dtype=torch.float64, device=g.device)
        selected_codes = torch.zeros(CHUNK, dtype=torch.uint8, device=g.device)
        chosen_outputs = baseline.clone()
        for candidate in range(64):
            require(time.time() < deadline, 'Enumeration deadline expired; never truncate or retry.')
            values = packed_decode(gate, flip_grouped(original, candidate))
            dots = (g.double() * (values.float() - baseline.float()).double()).sum(-1)
            require(bool(torch.isfinite(dots).all()), 'Nonfinite final-KL proposal score.')
            improved = dots < best
            selected_codes = torch.where(improved, candidate + 1, selected_codes).to(torch.uint8)
            best = torch.where(improved, dots, best)
            chosen_outputs = torch.where(improved[:, None], values, chosen_outputs)
        selected = original.clone()
        from experiments.gate_group_accept_probe.core import apply_choices
        for stage in range(2):
            apply_choices(selected[:, stage:stage + 1], selected_codes, stage)
        repeated = packed_decode(gate, selected)
        reduced = (g.double() * (repeated.float() - baseline.float()).double()).sum(-1)
        require(torch.equal(repeated, chosen_outputs) and torch.equal(reduced, best)
                and bool((best <= 0).all()) and bool((best[selected_codes > 0] < 0).all()),
                'Selected packed outputs or proxy values differ from the same-shape enumeration.')
        choices[start:end] = selected_codes
        output_delta.view(-1, 32)[start:end] = repeated.float() - baseline.float()
        row = {'vector_start': start, 'vector_end': end, 'decoder_rows': CHUNK,
            'selected_vectors': int((selected_codes > 0).sum()), 'proxy_sum': float(best.sum()),
            'same_shape_selected_decode_and_score_exact': True, 'seconds': time.time() - tick}
        chunks.append(row); observer('enumeration_chunk', row)
    return choices, output_delta, {'vectors': VECTORS, 'groups': GROUPS, 'chunks': chunks,
        'candidates_per_vector': 64, 'candidate_count': VECTORS * 64,
        'selected_vectors': int((choices > 0).sum()), 'seconds': time.time() - started,
        'candidate_order': list(range(64)), 'tie_rule': 'Original0 wins nonnegative scores; first bit wins negative ties.'}


@torch.no_grad()
def propose(data, batches, teacher, old_choices, *, deadline, observer):
    source = data['source_data']; device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats(device)
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu',
                                                  strict=True, expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
            'Proposals must start from actual fresh f927, never55a.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    gate = model.get_submodule(TARGET)
    require(records(gate) == source['expected_gate_state'], 'Initial gate differs from the captured f927 state.')
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    policy = native_policy(model, source['model_policy']); frozen = version_inventory(model, exclude_gate=True)
    initial = records(gate)
    require(len(initial) == 13 and len(frozen) == 3410, 'Native persistent inventory differs.')
    original = {key: value.detach().clone() for key, value in gate.state_dict().items()}
    weight0 = gate._cached_weight.detach().clone()
    def first_logits():
        return model(**document_batch(batches[ARM], 0, device), use_cache=False).logits.detach().cpu()
    logits0 = first_logits()
    gradient, grad_report = gradient128(model, gate, weight0, batches[ARM], teacher[ARM], deadline=deadline,
                                        observer=lambda row: observer('gradient', row))
    keep = torch.ones(4096, dtype=torch.bool, device=device); keep[gate.protected_input_indices] = False
    unprotected = gradient[:, keep].contiguous()
    rotated = gate.weight_rotation(unprotected, inverse=False)
    choices, delta_z, enumeration = enumerate16(gate, rotated, deadline=deadline, observer=observer)
    adjoint = adjoint_check(gate.weight_rotation, unprotected, delta_z)
    del unprotected, rotated, delta_z
    chosen_state, xor = replay_codes(original, choices)
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj)
               for i in range(36)]
    def prime():
        ready(gate)
        result = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True,
                                             group_size=8, compute_device=device)
        require(result['warmed'] == 36 and result['failed'] == 0, 'Native gate group8 refresh failed.')
        return result
    try:
        for key in BANKS:
            gate.state_dict()[key].copy_(chosen_state[key])
        cache = prime(); endpoint = records(gate)
        require(endpoint == {key: tensor_record(value) for key, value in chosen_state.items()}
                and torch.equal(gate._cached_weight[512:], weight0[512:])
                and torch.equal(gate._cached_weight[:, gate.protected_input_indices],
                                weight0[:, gate.protected_input_indices]), 'Selected support or protected columns changed.')
        actual_dot = dot64(gradient, gate._cached_weight.float() - weight0.float())
        delta_stats = tensor_stats(gate._cached_weight.float() - weight0.float())
        # Report these fixed observations; no score controls choices, fitting or full8 execution.
        populations = (ARM, 'train32', 'alpaca_held32', 'fineweb_rows500000_500031',
                       'alpaca_confirmation32', 'fineweb_rows500032_500063')
        measured = {name: score_population(model, batches[name], teacher[name], deadline=deadline)
                    for name in populations}
    finally:
        gate.load_state_dict(original, strict=True); prime()
        require(records(gate) == initial and torch.equal(gate._cached_weight, weight0)
                and torch.equal(first_logits(), logits0), 'Original codes/cache/full logits did not restore.')
    require(version_inventory(model, exclude_gate=True) == frozen
            and native_policy(model, source['model_policy']) == policy
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Frozen complement or policy changed.')
    current_first = choices[:CHUNK].cpu()
    return choices.cpu(), {'source_checkpoint_id': SOURCE_ID, 'initial_state': initial,
        'applied_state': endpoint, 'stage_xor': xor, 'choice_inventory': choice_inventory(choices),
        'gradient': grad_report, 'enumeration': enumeration, 'adjoint_fp32': adjoint,
        'native_original_coordinate_weight_dot': actual_dot, 'native_weight_delta': delta_stats,
        'first_four_groups_vs_historical': {'equal': torch.equal(current_first, old_choices),
            'different_vector_choices': int((current_first != old_choices).sum()),
            'boundary': 'Independent-run floating/choice differences are recorded, never overwritten with old choices.'},
        'fixed_endpoint_observations': measured, 'warmup': warmup, 'candidate_cache': cache,
        'all13_original_bytes_restored': True, 'frozen_complement_count': len(frozen),
        'frozen_complement_versions_and_pointers_exact': True,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'boundary': 'One initial derivative, four native chunks, all sixteen groups applied together; no greedy acceptance.'}
