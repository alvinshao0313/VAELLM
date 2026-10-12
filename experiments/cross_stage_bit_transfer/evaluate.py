"""Frozen three-arm native whole-model observations on two existing development domains."""
from __future__ import annotations

import logging

import torch

from experiments.cross_stage_bit_transfer.search import ARMS, chosen_packed, coordinates, install
from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_final_kl_accept_probe.core import score
from experiments.gate_final_kl_gradient_probe.core import cached_call_guard
from experiments.gate_local_delta_probe.core import discrepancy
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_student_context_probe.data import SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import document_batch, native_policy, version_inventory
from experiments.gate_tail_objective_probe.data import DOMAINS
from experiments.rotation_factor_probe.capture import require, tensor_record
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint


def compare(left, right):
    result = {}
    for domain in DOMAINS:
        result[domain] = {key + '_delta': left[domain][key] - right[domain][key]
                          for key in ('kl_sum', 'kl_mean', 'nll_sum', 'nll_mean')}
        for key in ('kl_sum', 'nll_sum'):
            delta = [a[key] - b[key] for a, b in zip(left[domain]['documents'], right[domain]['documents'], strict=True)]
            result[domain][key + '_documents'] = {'delta': delta, 'improved': sum(v < 0 for v in delta),
                                                'equal': sum(v == 0 for v in delta), 'worsened': sum(v > 0 for v in delta)}
    return result


@torch.no_grad()
def evaluate(data, batches, teacher, cases, choices, projection_tiles, target, expected_before, *, deadline, observer):
    require(tuple(batches) == DOMAINS and set(teacher) == set(DOMAINS), 'Use exactly the two prepared development domains.')
    device = torch.device('cuda:0'); source = data['source_data']
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True, expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Fresh f927 strict load failed.')
    model.eval().requires_grad_(False); model.config.use_cache = False; gate = model.get_submodule(TARGET)
    require(records(gate) == source['expected_gate_state'], 'Fresh target state differs.')
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    policy = native_policy(model, source['model_policy']); frozen = version_inventory(model, exclude_gate=True)
    initial = records(gate); require(len(initial) == 13 and len(frozen) == 3410, 'Wrong persistent state inventory.')
    vectors, rows, columns = coordinates(gate, cases)
    banks = [gate.get_stage_part_vq_storage(s, 0).clone() for s in range(2)]
    selected0 = torch.stack([b[vectors, 0] for b in banks], dim=1)
    weight0 = gate._cached_weight.detach().clone()
    reference = target.to(device)[rows[:, :, None], columns[:, None, :]].double()
    union = torch.zeros_like(weight0, dtype=torch.bool); union[rows[:, :, None], columns[:, None, :]] = True
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj) for i in range(36)]

    def prime():
        ready(gate)
        value = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True, group_size=8, compute_device=device)
        require(value['warmed'] == 36 and value['failed'] == 0, 'Native group8 category cache rebuild failed.')
        return value

    def logits0(domain):
        return model(**document_batch(batches[domain], 0, device), use_cache=False).logits.detach().cpu()

    original_logits = {domain: logits0(domain) for domain in DOMAINS}
    results, endpoint_records, geometry, stage_xor, restore_reports = {}, {}, {}, {}, []
    for arm_index, arm in enumerate(ARMS):
        try:
            if arm != 'original':
                proposed = chosen_packed(selected0, choices[arm_index].to(device)); install(gate, vectors, proposed); prime()
            state = records(gate)
            require(all(state[k] == value for k, value in initial.items() if k not in ('vq_weight', 'vq_weight_s1')),
                    'A frozen decoder/protection/rotation tensor changed.')
            xor = []
            for s in range(2):
                expected = banks[s].clone()
                expected[vectors, 0] = chosen_packed(selected0, choices[arm_index].to(device))[:, s]
                require(torch.equal(expected, gate.get_stage_part_vq_storage(s, 0)), 'Compact choices do not replay the complete bank.')
                xor.append(popcounts(banks[s], expected)); del expected
            stage_xor[arm] = xor; endpoint_records[arm] = state
            weight = gate._cached_weight
            require(torch.equal(weight[~union], weight0[~union])
                    and torch.equal(weight[:, gate.protected_input_indices], weight0[:, gate.protected_input_indices]),
                    'Group8 endpoint changed outside support or protected columns.')
            tiles = weight[rows[:, :, None], columns[:, None, :]]
            sse = (tiles.double() - reference).square().sum((1, 2))
            geometry[arm] = {'group8_sse_per_tile': sse.cpu().tolist(), 'group8_sse_sum': float(sse.sum()),
                            'projection_search_vs_group8': discrepancy(projection_tiles[arm_index].to(device), tiles)}
            handle, calls = cached_call_guard(gate, weight)
            try:
                results[arm] = {}
                for domain in DOMAINS:
                    results[arm][domain] = score(model, batches[domain], teacher[domain], deadline=deadline)
                    observer({'phase': 'full_model', 'arm': arm, 'domain': domain,
                              **{k: results[arm][domain][k] for k in ('kl_mean', 'nll_mean', 'causal_tokens')}})
            finally:
                handle.remove()
            require(calls['count'] == 64 and gate._cached_weight is weight, 'All64 documents must hit the actual cached endpoint.')
            if arm == 'original':
                require(results[arm] == expected_before, 'Original per-document scores differ from the existing verified baseline.')
        finally:
            if arm != 'original':
                for s in range(2): gate.get_stage_part_vq_storage(s, 0).copy_(banks[s])
                restored = prime()
                require(records(gate) == initial and torch.equal(gate._cached_weight, weight0)
                        and all(torch.equal(logits0(d), original_logits[d]) for d in DOMAINS), 'Native source/logits restoration failed.')
                restore_reports.append({'after_arm': arm, 'cache': restored, 'all13_state_and_weight_exact': True,
                                        'both_domains_first_full_logits_exact': True})
    require(version_inventory(model, exclude_gate=True) == frozen and native_policy(model, source['model_policy']) == policy
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Frozen complementary state/policy changed.')
    comparisons = {name: compare(results[left], results[right]) for name, left, right in (
        ('single_minus_original', 'single', 'original'), ('pair_minus_original', 'pair', 'original'), ('pair_minus_single', 'pair', 'single'))}
    return {'status': 'completed', 'arms': results, 'comparisons': comparisons, 'actual_stage_xor': stage_xor,
        'search_vs_deployment_geometry': geometry, 'endpoint_gate_state': endpoint_records,
        'frozen_complement_count': 3410, 'frozen_complement_versions_pointers_exact': True,
        'all_parameters_frozen': True, 'restore_reports': restore_reports, 'cache_hits_per_arm': 64,
        'source_first_logits': {d: tensor_record(v) for d, v in original_logits.items()},
        'native_policy': policy, 'warmup': warmup,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
