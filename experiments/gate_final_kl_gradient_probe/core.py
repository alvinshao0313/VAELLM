"""Read-only final-KL derivatives of the actual native BF16 gate cache."""
from __future__ import annotations

import logging
import time

import torch

from experiments.gate_final_kl_accept_probe.core import score
from experiments.gate_group_accept_probe.core import BANKS, VECTORS_PER_GROUP, apply_choices, group_inventory
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

ALPHAS = (0.25, 0.5, 1.0)


def tensor_stats(value):
    # Keep FP64 reduction scratch bounded; never retain/save the large gradient.
    squared, nonzero, maximum = 0.0, 0, 0.0
    for chunk in value.split(128):
        require(bool(torch.isfinite(chunk).all()), 'Nonfinite diagnostic tensor.')
        squared += float(chunk.double().square().sum())
        nonzero += int(torch.count_nonzero(chunk))
        maximum = max(maximum, float(chunk.abs().max()))
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'nonzero': nonzero,
            'squared_l2': squared, 'max_abs': maximum}


def dot64(left, right):
    require(left.shape == right.shape, 'Gradient/displacement geometry differs.')
    return sum(float((a.double() * b.double()).sum())
               for a, b in zip(left.split(128), right.split(128)))


def cached_call_guard(gate, expected):
    """Observe the real forward branch prerequisites without replacing forward."""
    calls = {'count': 0}
    def observe(module, args):
        dtype = getattr(module, '_decoder_compute_dtype', None) or args[0].dtype
        require(module._cached_weight is expected and module.cache_decoded_weight
                and not module.trainable_decode and module.temporary and not module.always_use_original
                and expected.dtype == dtype and expected.device == args[0].device,
                'The actual VAELinear forward would miss or replace the diagnostic cache.')
        calls['count'] += 1
    return gate.register_forward_pre_hook(observe), calls


@torch.no_grad()
def probe(data, batch, teacher, choices, *, deadline, observer):
    device = torch.device('cuda:0'); source = data['source_data']
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True,
                                                  expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
            'Require the original fresh f927 full model.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    gate = model.get_submodule(TARGET)
    require(records(gate) == source['expected_gate_state'], 'Initial native gate differs from f927.')
    torch.cuda.reset_peak_memory_stats(device)
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    policy = native_policy(model, source['model_policy'])
    frozen = version_inventory(model, exclude_gate=True); initial = records(gate)
    require(len(frozen) == 3410 and len(initial) == 13 and all(not p.requires_grad for p in model.parameters()),
            'Persistent parameters must remain frozen.')
    banks = [gate.get_stage_part_vq_storage(stage, 0).clone() for stage in range(2)]
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj) for i in range(36)]
    def prime():
        ready(gate)
        result = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True,
                                              group_size=8, compute_device=device)
        require(result['warmed'] == 36 and result['failed'] == 0, 'Native group8 cache rebuild failed.')
        return result
    original_weight = gate._cached_weight.detach().clone()
    require(original_weight.dtype == torch.bfloat16 and original_weight.shape == (12288, 4096), 'Unexpected native cache.')
    prime(); require(torch.equal(gate._cached_weight, original_weight), 'Category/global group8 cache differs.')
    def first_logits():
        return model(**document_batch(batch, 0, device), use_cache=False).logits.detach().cpu()
    original_logits = first_logits()
    before = score(model, batch, teacher, deadline=deadline)
    require(before['causal_tokens'] == 4257, 'Use every original train32 causal position.')
    pre_gradient_memory = {'allocated_bytes': torch.cuda.memory_allocated(device),
        'reserved_bytes': torch.cuda.memory_reserved(device), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    torch.cuda.reset_peak_memory_stats(device)
    leaf = original_weight.detach().requires_grad_(True)
    gate._cached_weight = leaf
    accumulator = torch.zeros_like(original_weight, dtype=torch.float32)
    gradient_rows = []; gradient_started = time.time(); grad_logits_difference = None
    hook, observed = cached_call_guard(gate, leaf)
    try:
        with torch.enable_grad():
            require(leaf.is_leaf and not torch.is_inference_mode_enabled(), 'Need a normal autograd leaf, not an inference tensor.')
            for index in range(32):
                require(time.time() < deadline, 'Assigned diagnostic deadline expired.')
                item = document_batch(batch, index, device)
                logits = model(**item, use_cache=False).logits
                if index == 0:
                    grad_logits_difference = discrepancy(logits.detach().cpu(), original_logits)
                selected, labels = selected_logits(logits, item)
                require(torch.equal(labels.cpu(), teacher[index]['labels']), 'Causal teacher labels changed.')
                kl, nll = probability_metrics(selected, teacher[index])
                gradient, = torch.autograd.grad(kl.double().sum(), leaf)
                require(gradient.dtype == torch.bfloat16 and gradient.shape == leaf.shape
                        and bool(torch.isfinite(gradient).all()), 'Invalid native BF16 leaf gradient.')
                accumulator.add_(gradient.float())
                row = {'index': index, 'causal_tokens': len(labels), 'kl_sum': float(kl.detach().double().sum()),
                       'nll_sum': float(nll.detach().double().sum()), 'gradient': tensor_stats(gradient.detach())}
                gradient_rows.append(row); observer('gradient', row)
                del item, logits, selected, labels, kl, nll, gradient
            require(gate._cached_weight is leaf, 'The final backward document replaced the cache leaf.')
    finally:
        hook.remove(); gate._cached_weight = original_weight
    require(observed['count'] == 32 and leaf.grad is None
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Unexpected gradient destination/cache calls.')
    gradient_stats = tensor_stats(accumulator)
    require(gradient_stats['nonzero'] > 0 and records(gate) == initial, 'Zero total gradient or persistent gate mutation.')
    require(torch.equal(first_logits(), original_logits), 'Restoring the original cache changed complete logits.')
    gradient_report = {'documents': gradient_rows, 'accumulator': gradient_stats, 'actual_cache_hits': observed['count'],
        'leaf_dtype': str(leaf.dtype), 'sum_kl': sum(r['kl_sum'] for r in gradient_rows),
        'sum_nll': sum(r['nll_sum'] for r in gradient_rows), 'forward_documents_equal_no_grad': all(
            {k: row[k] for k in ('index', 'causal_tokens', 'kl_sum', 'nll_sum')} == before['documents'][i]
            for i, row in enumerate(gradient_rows)), 'first_forward_logits_difference': grad_logits_difference,
        'original_logits_after_gradient_exact': True, 'all_persistent_parameters_require_grad_false_and_grad_none': True,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'seconds': time.time() - gradient_started}
    del leaf
    choices = choices.to(device); inventory = group_inventory(choices); trajectory = []
    for item in inventory[:8]:
        started = time.time(); group = item['group']; start = group * VECTORS_PER_GROUP; end = start + VECTORS_PER_GROUP
        require(time.time() < deadline and records(gate) == initial, 'Every group must start from original f927.')
        try:
            for stage in range(2):
                apply_choices(gate.get_stage_part_vq_storage(stage, 0)[start:end], choices[start:end], stage)
            cache_report = prime(); endpoint = gate._cached_weight.detach()
            changed = records(gate)
            require(all(changed[k] == value for k, value in initial.items() if k not in BANKS), 'A noncode gate tensor changed.')
            delta = endpoint.float() - original_weight.float()
            lo, hi = group * 32, (group + 1) * 32
            require(not bool(torch.count_nonzero(delta[:lo])) and not bool(torch.count_nonzero(delta[hi:]))
                    and not bool(torch.count_nonzero(delta[:, gate.protected_input_indices])),
                    'Actual original-coordinate displacement escaped the block32/protection support.')
            full_dot = dot64(accumulator, delta)
            points = []
            for alpha in ALPHAS:
                # alpha=1 is the actual native endpoint, not an arithmetically rebuilt approximation.
                value = endpoint if alpha == 1.0 else (original_weight.float() + alpha * delta).bfloat16()
                gate._cached_weight = value
                hook, calls = cached_call_guard(gate, value)
                begin = time.time()
                try:
                    metrics = score(model, batch, teacher, deadline=deadline)
                finally:
                    hook.remove()
                require(calls['count'] == 32 and gate._cached_weight is value, 'An interpolation/native cache was not reused.')
                actual_delta = delta if alpha == 1.0 else value.float() - original_weight.float()
                actual_dot = dot64(accumulator, actual_delta)
                observed_delta = metrics['kl_sum'] - before['kl_sum']
                points.append({'alpha': alpha, 'native_deployable_endpoint': alpha == 1.0, 'metrics': metrics,
                    'actual_cache_hits': calls['count'], 'weight_delta': tensor_stats(actual_delta),
                    'ideal_alpha_grad_dot': alpha * full_dot, 'actual_bf16_displacement_grad_dot': actual_dot,
                    'observed_kl_delta_sum': observed_delta, 'observed_kl_delta_mean': observed_delta / 4257,
                    'observed_nll_delta_sum': metrics['nll_sum'] - before['nll_sum'],
                    'remainder_vs_actual_displacement': observed_delta - actual_dot,
                    'seconds': time.time() - begin})
                del value, actual_delta
            row = {**item, 'cache': cache_report, 'native_original_coordinate_weight_delta': tensor_stats(delta),
                   'full_displacement_grad_dot': full_dot, 'points': points}
            del endpoint, delta
        finally:
            gate._cached_weight = original_weight
            for stage in range(2):
                gate.get_stage_part_vq_storage(stage, 0)[start:end].copy_(banks[stage][start:end])
            prime()
            require(records(gate) == initial and torch.equal(gate._cached_weight, original_weight)
                    and torch.equal(first_logits(), original_logits), 'Group restore changed codes/native cache/full logits.')
        row.update(seconds=time.time() - started, restored_all13_exact=True, restored_cache_exact=True,
                   restored_complete_logits_exact=True)
        trajectory.append(row); observer('group', row)
    final = score(model, batch, teacher, deadline=deadline)
    require(final == before and records(gate) == initial and version_inventory(model, exclude_gate=True) == frozen
            and native_policy(model, source['model_policy']) == policy
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Final frozen/restoration contract failed.')
    return {'status': 'completed', 'source_checkpoint_id': SOURCE_ID, 'groups': list(range(8)), 'alphas': list(ALPHAS),
        'before': before, 'gradient': gradient_report, 'trajectory': trajectory, 'final_restored': final,
        'warmup': warmup, 'native_policy': policy, 'pre_gradient_memory': pre_gradient_memory,
        'peak_allocated_after_warmup_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_after_warmup_bytes': torch.cuda.max_memory_reserved(device),
        'original_all13_exact': True, 'frozen_complement_count': len(frozen),
        'frozen_complement_versions_and_pointers_exact': True, 'final_full_train_scores_exact': True,
        'all_persistent_parameters_require_grad_false_and_grad_none': True,
        'saved_model_or_gradient': False,
        'boundary': 'BF16 autograd gradients accumulated in FP32; FP64 contractions. Remainders include finite steps, nonlinearities, forward/backward rounding and BF16 interpolation rounding, not a pure Hessian measurement.'}
