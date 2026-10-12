"""Fixed four-group proposals: change calibration text, never the decoder or search."""
from __future__ import annotations

import logging
import time

import torch

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_final_kl_accept_probe.core import score
from experiments.gate_final_kl_gradient_probe.core import cached_call_guard, dot64, tensor_stats
from experiments.gate_final_kl_neighbor_probe.core import adjoint_check, enumerate_neighbors
from experiments.gate_final_kl_neighbor_transfer.core import choice_inventory
from experiments.gate_group_accept_probe.core import BANKS, apply_choices
from experiments.gate_local_delta_probe.core import discrepancy
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_student_context_probe.data import SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import (
    document_batch, native_policy, probability_metrics, selected_logits, version_inventory,
)
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint

ARMS = ('alpaca128', 'alpaca64_fineweb64')
OBSERVATIONS = ('train32', 'alpaca_held32', 'fineweb_rows500000_500031',
                'alpaca_confirmation32', 'fineweb_rows500032_500063')
POPULATIONS = ARMS + OBSERVATIONS


def aggregate(rows):
    tokens = sum(row['causal_tokens'] for row in rows)
    kl, nll = (sum(row[key] for row in rows) for key in ('kl_sum', 'nll_sum'))
    return {'documents': rows, 'causal_tokens': tokens, 'kl_sum': kl, 'kl_mean': kl / tokens,
            'nll_sum': nll, 'nll_mean': nll / tokens}


@torch.no_grad()
def score_population(model, batch, teacher, *, deadline):
    """Reuse the native32 scorer, preserving final document-order FP64 reduction."""
    count = batch['input_ids'].shape[0]
    require(count in (32, 128) and len(teacher) == count
            and tuple(batch['input_ids'].shape) == (count, 256), 'Explicit complete population required.')
    rows = []
    for start in range(0, count, 32):
        part = score(model, {key: value[start:start + 32] for key, value in batch.items()},
                     teacher[start:start + 32], deadline=deadline)
        rows.extend({**row, 'index': start + row['index']} for row in part['documents'])
    require(len(rows) == count, 'Scorer omitted documents.')
    return aggregate(rows)


def gradient128(model, gate, weight, batch, teacher, *, deadline, observer):
    """Accumulate128 individual native BF16 leaf derivatives, in fixed FP32 order."""
    count = batch['input_ids'].shape[0]
    require(count == len(teacher) == 128 and tuple(batch['input_ids'].shape) == (128, 256),
            'This experiment requires128 actual documents, not a32-row helper prefix.')
    leaf = weight.detach().requires_grad_(True); gate._cached_weight = leaf
    total = torch.zeros_like(weight, dtype=torch.float32); rows = []; started = time.time(); prefix = None
    handle, calls = cached_call_guard(gate, leaf)
    try:
        with torch.enable_grad():
            for index in range(count):
                require(time.time() < deadline, 'Assigned deadline expired; do not shorten calibration.')
                item = document_batch(batch, index, weight.device)
                logits = model(**item, use_cache=False).logits
                selected, labels = selected_logits(logits, item)
                require(torch.equal(labels.cpu(), teacher[index]['labels']), 'Causal teacher labels changed.')
                kl, nll = probability_metrics(selected, teacher[index])
                grad, = torch.autograd.grad(kl.double().sum(), leaf)
                require(grad.dtype == torch.bfloat16 and bool(torch.isfinite(grad).all()), 'Invalid native leaf gradient.')
                total.add_(grad.float())
                row = {'index': index, 'causal_tokens': len(labels), 'kl_sum': float(kl.detach().double().sum()),
                       'nll_sum': float(nll.detach().double().sum()), 'gradient': tensor_stats(grad.detach())}
                rows.append(row); observer(row)
                if index == 31:
                    prefix = tensor_stats(total)
                del item, logits, selected, labels, kl, nll, grad
            require(gate._cached_weight is leaf and calls['count'] == count and leaf.grad is None,
                    'Not all128 documents used the cached leaf.')
    finally:
        handle.remove(); gate._cached_weight = weight
    summary = tensor_stats(total)
    require(summary['nonzero'] > 0 and all(not p.requires_grad and p.grad is None for p in model.parameters()),
            'A persistent parameter received a gradient or the derivative is zero.')
    return total, {'documents': rows, 'aggregate': aggregate([{k: row[k] for k in
        ('index', 'causal_tokens', 'kl_sum', 'nll_sum')} for row in rows]), 'accumulator': summary,
        'first32_accumulator': prefix, 'actual_cache_hits': calls['count'], 'seconds': time.time() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(weight.device),
        'accumulation': 'BF16 returned derivative per document, FP32 add_ in the listed order; no per-source or per-document mean.'}


def comparisons(current, baseline):
    result = {}
    for population in current:
        now, old = current[population], baseline[population]
        require(now['causal_tokens'] == old['causal_tokens'] and len(now['documents']) == len(old['documents']),
                'Paired populations changed.')
        value = {key + '_delta': now[key] - old[key] for key in ('kl_sum', 'kl_mean', 'nll_sum', 'nll_mean')}
        for metric in ('kl_sum', 'nll_sum'):
            changes = [a[metric] - b[metric] for a, b in zip(now['documents'], old['documents'], strict=True)]
            value[metric + '_documents'] = {'improved': sum(x < 0 for x in changes),
                'equal': sum(x == 0 for x in changes), 'worsened': sum(x > 0 for x in changes)}
        result[population] = value
    return result


@torch.no_grad()
def probe(data, batches, teacher, reference_choices, historical, *, deadline, observer):
    require(tuple(batches) == POPULATIONS and set(teacher) == set(POPULATIONS), 'Use the two fixed populations and five observation domains.')
    source = data['source_data']; device = torch.device('cuda:0')
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True,
                                                  expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Require fresh f927.')
    model.eval().requires_grad_(False); model.config.use_cache = False; gate = model.get_submodule(TARGET)
    require(records(gate) == source['expected_gate_state'], 'Initial gate differs from f927.')
    torch.cuda.reset_peak_memory_stats(device)
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    policy = native_policy(model, source['model_policy']); frozen = version_inventory(model, exclude_gate=True)
    initial = records(gate); banks = [gate.get_stage_part_vq_storage(stage, 0).clone() for stage in range(2)]
    require(len(initial) == 13 and len(frozen) == 3410, 'Unexpected persistent inventory.')
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj) for i in range(36)]
    weight0 = gate._cached_weight.detach().clone()
    def first_logits():
        return model(**document_batch(batches['train32'], 0, device), use_cache=False).logits.detach().cpu()
    original_logits = first_logits()
    def prime():
        ready(gate)
        result = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True, group_size=8, compute_device=device)
        require(result['warmed'] == 36 and result['failed'] == 0, 'Native group8 refresh failed.')
        return result
    def restore():
        for stage in range(2): gate.get_stage_part_vq_storage(stage, 0).copy_(banks[stage])
        prime()
        require(records(gate) == initial and torch.equal(gate._cached_weight, weight0)
                and torch.equal(first_logits(), original_logits), 'Original codes/cache/full logits were not restored.')
    def apply(choices):
        inventory = choice_inventory(choices); choices = choices.to(device)
        for stage in range(2): apply_choices(gate.get_stage_part_vq_storage(stage, 0)[:len(choices)], choices, stage)
        cache = prime(); endpoint = records(gate)
        require(all(endpoint[key] == value for key, value in initial.items() if key not in BANKS), 'A noncode local tensor changed.')
        xor = []
        for stage in range(2):
            expected = banks[stage].clone(); apply_choices(expected[:len(choices)], choices, stage)
            require(torch.equal(expected, gate.get_stage_part_vq_storage(stage, 0)), 'Complete bank choices replay failed.')
            xor.append(popcounts(banks[stage], expected)); del expected
        require(xor == [sum(row['stage_xor'][s] for row in inventory) for s in range(2)]
                and torch.equal(gate._cached_weight[128:], weight0[128:])
                and torch.equal(gate._cached_weight[:, gate.protected_input_indices], weight0[:, gate.protected_input_indices]),
                'Changed codes/support/protected columns differ from the fixed four-group proposal.')
        return {'choice_inventory': inventory, 'stage_xor': xor, 'endpoint_state': endpoint,
                'complete_bank_replay_exact': True, 'local11_noncode_bytes_exact': True, 'cache': cache}
    choices_by_arm = {}; generations = {}
    for arm in ARMS:
        require(records(gate) == initial and time.time() < deadline, 'Every initial derivative must use fresh f927 state.')
        gradient, grad_report = gradient128(model, gate, weight0, batches[arm], teacher[arm], deadline=deadline,
                                            observer=lambda row: observer('gradient', {'arm': arm, **row}))
        keep = torch.ones(4096, dtype=torch.bool, device=device); keep[gate.protected_input_indices] = False
        compressed_gradient = gradient[:, keep].contiguous()
        rotated_gradient = gate.weight_rotation(compressed_gradient, inverse=False)
        choices, chunk0, chosen, best, singles, enumeration = enumerate_neighbors(gate, rotated_gradient, deadline=deadline)
        delta_z = torch.zeros_like(compressed_gradient); delta_z.view(-1, 32)[:len(choices)] = chosen.float() - chunk0.float()
        adjoint = adjoint_check(gate.weight_rotation, compressed_gradient, delta_z)
        del delta_z, compressed_gradient, rotated_gradient, singles
        native0 = gate._decode_compressed_weight(dtype=torch.bfloat16)
        chunk_before = discrepancy(chunk0, native0.view(-1, 32)[:len(choices)]); del native0
        try:
            applied = apply(choices)
            delta_w = gate._cached_weight.float() - weight0.float()
            actual_dot = dot64(gradient, delta_w)
            native1 = gate._decode_compressed_weight(dtype=torch.bfloat16)
            chunk_after = discrepancy(chosen, native1.view(-1, 32)[:len(choices)])
            generations[arm] = {'gradient': grad_report, 'enumeration': enumeration, 'adjoint_fp32': adjoint,
                'rotated_per_vector_proxy_sum': float(best.sum()), 'native_original_coordinate_weight_dot': actual_dot,
                'native_dot_minus_proxy': actual_dot - float(best.sum()), 'native_weight_delta': tensor_stats(delta_w),
                'chunk_vs_native_compressed_before': chunk_before, 'chunk_vs_native_compressed_after': chunk_after,
                'applied_state': applied, 'choices': tensor_record(choices.cpu())}
            choices_by_arm[arm] = choices.cpu()
            del delta_w, native1
        finally:
            restore()
        del gradient, choices, chunk0, chosen, best
        observer('proposal', {'arm': arm, 'stage_xor': applied['stage_xor'], 'native_weight_dot': actual_dot})
    # Both proposals are fixed before any old/new observation-domain forward.
    arm_choices = {'f927': None, 'historical_alpaca32': reference_choices, **choices_by_arm}
    measured = {}; endpoint_records = {}
    for arm, choices in arm_choices.items():
        require(records(gate) == initial, 'Each measured arm must start from original f927.')
        try:
            if choices is not None: endpoint_records[arm] = apply(choices)
            endpoint = gate._cached_weight; handle, calls = cached_call_guard(gate, endpoint)
            try:
                measured[arm] = {}
                for population in POPULATIONS:
                    measured[arm][population] = score_population(model, batches[population], teacher[population], deadline=deadline)
                    observer('score', {'arm': arm, 'population': population, **{k: measured[arm][population][k]
                        for k in ('causal_tokens', 'kl_mean', 'nll_mean')}})
            finally:
                handle.remove()
            require(calls['count'] == sum(batch['input_ids'].shape[0] for batch in batches.values())
                    and gate._cached_weight is endpoint, 'Scoring omitted a population or replaced its cache.')
        finally:
            if choices is not None: restore()
    historic = {}
    for arm, old_key in (('f927', 'before'), ('historical_alpaca32', 'endpoint')):
        current = {old_name: measured[arm][new_name] for old_name, new_name in
                   (('train', 'train32'), ('alpaca_held32', 'alpaca_held32'), ('fineweb_rows500000_500031', 'fineweb_rows500000_500031'))}
        historic[arm] = {'scores_exact': current == historical[old_key],
                        'differences': comparisons(current, historical[old_key])}
    require(records(gate) == initial and version_inventory(model, exclude_gate=True) == frozen
            and native_policy(model, source['model_policy']) == policy and torch.equal(first_logits(), original_logits)
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Final original/frozen-state contract failed.')
    return choices_by_arm, {'status': 'completed', 'source_checkpoint_id': SOURCE_ID, 'generations': generations,
        'endpoints': measured, 'endpoint_records': endpoint_records, 'historical_common_observations': historic,
        'comparisons_vs_f927': {arm: comparisons(measured[arm], measured['f927']) for arm in arm_choices if arm != 'f927'},
        'comparisons_vs_historical32': {arm: comparisons(measured[arm], measured['historical_alpaca32']) for arm in ARMS},
        'mixed_minus_alpaca128': comparisons(measured[ARMS[1]], measured[ARMS[0]]), 'initial_state': initial,
        'all13_final_bytes_exact': True, 'frozen_complement_count': len(frozen),
        'frozen_complement_versions_and_pointers_exact': True, 'all_parameters_frozen_and_grad_none': True,
        'original_cache_and_full_logits_restored_exact': True, 'warmup': warmup, 'native_policy': policy,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'boundary': '128doc token-sum initial derivatives, not source-equal weighting. Two fixed simultaneous four-group proposals; no greedy acceptance, decoder fitting, heldout selection or framework integration.'}
