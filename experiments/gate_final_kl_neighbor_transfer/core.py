"""Measure the fixed four-group code proposal, without further selection or fitting."""
from __future__ import annotations

import logging
import time

import torch

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_final_kl_accept_probe.core import score
from experiments.gate_final_kl_gradient_probe.core import cached_call_guard
from experiments.gate_group_accept_probe.core import BANKS, VECTORS_PER_GROUP, apply_choices
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_student_context_probe.data import SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import document_batch, native_policy, version_inventory
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint

DOMAINS = ('train', 'alpaca_held32', 'fineweb_rows500000_500031')
GROUPS = (0, 1, 2, 3)


def choice_inventory(choices):
    require(choices.dtype == torch.uint8 and tuple(choices.shape) == (4 * VECTORS_PER_GROUP,)
            and int(choices.max()) <= 64, 'Use only the saved four fixed groups, no new proposal search.')
    rows = []
    for group in GROUPS:
        selected = choices[group * VECTORS_PER_GROUP:(group + 1) * VECTORS_PER_GROUP]
        rows.append({'group': group, 'selected_vectors': int((selected > 0).sum()),
            'stage_xor': [int(((selected > stage * 32) & (selected <= (stage + 1) * 32)).sum()) for stage in range(2)],
            'choice_histogram': torch.bincount(selected.long(), minlength=65).tolist()})
    return rows


@torch.no_grad()
def measure(data, batches, teacher, choices, expected_before, *, deadline, observer):
    require(tuple(batches) == DOMAINS and set(teacher) == set(DOMAINS), 'Measure exactly the three fixed populations.')
    inventory = choice_inventory(choices); source = data['source_data']; device = torch.device('cuda:0')
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True, expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Always start from fresh f927.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    gate = model.get_submodule(TARGET)
    require(records(gate) == source['expected_gate_state'], 'Fresh thirteen gate tensors differ from the original source.')
    torch.cuda.reset_peak_memory_stats(device)
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    policy = native_policy(model, source['model_policy']); frozen = version_inventory(model, exclude_gate=True)
    initial = records(gate)
    require(len(initial) == 13 and len(frozen) == 3410, 'Original native persistent inventory differs.')
    banks = [gate.get_stage_part_vq_storage(stage, 0).clone() for stage in range(2)]
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj) for i in range(36)]

    def prime():
        ready(gate)
        result = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True, group_size=8, compute_device=device)
        require(result['warmed'] == 36 and result['failed'] == 0, 'Native category group8 rebuild failed.')
        return result

    def logits0(domain):
        return model(**document_batch(batches[domain], 0, device), use_cache=False).logits.detach().cpu()

    def all_scores(arm):
        result = {}; endpoint = gate._cached_weight; handle, calls = cached_call_guard(gate, endpoint)
        try:
            for domain in DOMAINS:
                started = time.time(); result[domain] = score(model, batches[domain], teacher[domain], deadline=deadline)
                observer({'arm': arm, 'domain': domain, 'seconds': time.time() - started,
                          **{k: result[domain][k] for k in ('causal_tokens', 'kl_sum', 'kl_mean', 'nll_sum', 'nll_mean')}})
        finally:
            handle.remove()
        require(calls['count'] == 96 and gate._cached_weight is endpoint, 'All96 documents must hit the same native cached endpoint.')
        return result

    weight0 = gate._cached_weight.detach().clone()
    original_logits = {domain: logits0(domain) for domain in DOMAINS}
    before = all_scores('original_f927')
    require(before == expected_before, 'Original per-document KL/NLL must reproduce the verified same-source three-domain baseline.')
    choices = choices.to(device); count = choices.numel()
    try:
        for stage in range(2):
            apply_choices(gate.get_stage_part_vq_storage(stage, 0)[:count], choices, stage)
        cache = prime(); endpoint_records = records(gate)
        require(all(endpoint_records[key] == value for key, value in initial.items() if key not in BANKS), 'A noncode gate tensor changed.')
        actual_xor = []
        for stage in range(2):
            expected = banks[stage].clone(); apply_choices(expected[:count], choices, stage)
            require(torch.equal(expected, gate.get_stage_part_vq_storage(stage, 0)), 'Saved compact choices do not replay the complete actual bank.')
            actual_xor.append(popcounts(banks[stage], expected)); del expected
        require(actual_xor == [sum(row['stage_xor'][s] for row in inventory) for s in range(2)], 'Changed-bit inventory differs from the fixed proposals.')
        weight1 = gate._cached_weight.detach().clone()
        require(torch.equal(weight1[128:], weight0[128:])
                and torch.equal(weight1[:, gate.protected_input_indices], weight0[:, gate.protected_input_indices]),
                'Combined proposal changed rows outside0..127 or protected columns.')
        repeated_cache = prime()
        require(torch.equal(weight1, gate._cached_weight) and records(gate) == endpoint_records, 'Repeated native group8 endpoint is not exact.')
        del weight1
        endpoint = all_scores('fixed_groups0_1_2_3')
    finally:
        for stage in range(2): gate.get_stage_part_vq_storage(stage, 0).copy_(banks[stage])
        restore_cache = prime()
        require(records(gate) == initial and torch.equal(gate._cached_weight, weight0)
                and all(torch.equal(logits0(domain), original_logits[domain]) for domain in DOMAINS),
                'Restoring original codes changed persistent state, native cache or complete first-document logits.')
    require(version_inventory(model, exclude_gate=True) == frozen and native_policy(model, source['model_policy']) == policy
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Frozen complement, policy or gradient contract failed.')
    comparisons = {}
    for domain in DOMAINS:
        comparisons[domain] = {key + '_delta': endpoint[domain][key] - before[domain][key]
                               for key in ('kl_sum', 'kl_mean', 'nll_sum', 'nll_mean')}
        for metric in ('kl_sum', 'nll_sum'):
            differences = [a[metric] - b[metric] for a, b in zip(endpoint[domain]['documents'], before[domain]['documents'])]
            comparisons[domain][metric + '_documents'] = {'improved': sum(x < 0 for x in differences),
                'equal': sum(x == 0 for x in differences), 'worsened': sum(x > 0 for x in differences)}
    return {'status': 'completed', 'source_checkpoint_id': SOURCE_ID, 'groups': list(GROUPS),
        'choice_inventory': inventory, 'actual_stage_xor': actual_xor, 'before': before, 'endpoint': endpoint,
        'comparisons': comparisons, 'initial_state': initial, 'endpoint_state': endpoint_records,
        'choices_complete_bank_replay_exact': True, 'noncode_gate_bytes_exact': True,
        'frozen_complement_count': len(frozen), 'frozen_complement_versions_and_pointers_exact': True,
        'all13_final_bytes_exact': True, 'original_cache_and_three_full_logits_restored_exact': True,
        'candidate_group8_repeated_exact': True, 'all_model_parameters_require_grad_false_and_grad_none': True,
        'cache_hits_per_arm': 96, 'warmup': warmup, 'candidate_cache': cache, 'repeated_cache': repeated_cache,
        'restore_cache': restore_cache, 'native_policy': policy,
        'original_first_document_logits': {domain: tensor_record(value) for domain, value in original_logits.items()},
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
