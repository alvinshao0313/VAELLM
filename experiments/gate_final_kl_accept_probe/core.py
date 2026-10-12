"""Fixed proposals accepted only by complete native-model training-document mass KL."""
from __future__ import annotations

import logging
import time

import torch

from experiments.gate_group_accept_probe.core import BANKS, GROUPS, VECTORS_PER_GROUP, apply_choices, group_inventory
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_student_context_probe.data import SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import (
    document_batch, native_policy, probability_metrics, selected_logits, version_inventory,
)
from experiments.rotation_factor_probe.capture import require
from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint


@torch.no_grad()
def score(model, batch, teacher, *, deadline):
    """FP32 native token losses, then per-document FP64 sums in original row order."""
    rows = []
    device = next(model.parameters()).device
    require(len(teacher) == 32 and tuple(batch['input_ids'].shape) == (32, 256), 'Keep all32 fixed256 documents.')
    for index in range(32):
        require(time.time() < deadline, 'Assigned deadline expired; no early-stop endpoint or retry.')
        item = document_batch(batch, index, device)
        logits = model(**item, use_cache=False).logits
        selected, labels = selected_logits(logits, item)
        require(torch.equal(labels.cpu(), teacher[index]['labels']), 'Teacher/student causal labels differ.')
        kl, nll = probability_metrics(selected, teacher[index])
        rows.append({'index': index, 'causal_tokens': len(labels),
                     'kl_sum': float(kl.double().sum()), 'nll_sum': float(nll.double().sum())})
        del item, logits, selected, labels, kl, nll
    total = sum(row['causal_tokens'] for row in rows)
    kl_sum = sum(row['kl_sum'] for row in rows)
    nll_sum = sum(row['nll_sum'] for row in rows)
    return {'documents': rows, 'causal_tokens': total, 'kl_sum': kl_sum,
            'kl_mean': kl_sum / total, 'nll_sum': nll_sum, 'nll_mean': nll_sum / total}


@torch.no_grad()
def search(data, batches, teacher, choices, *, groups, deadline, observer):
    require(groups in (8, GROUPS), 'Only the separate8-group mechanism check or full384 pass is defined.')
    source = data['source_data']
    device = torch.device('cuda:0')
    model, meta, loaded = load_v6_model_checkpoint(source['source_checkpoint'], map_location='cpu', strict=True,
                                                  expected_kind='final_model')
    require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
            'Always load fresh original f927, never the mechanism or locally accepted endpoint.')
    model.eval().requires_grad_(False); model.config.use_cache = False
    gate = model.get_submodule(TARGET)
    require(records(gate) == source['expected_gate_state'], 'Initial thirteen gate tensors differ from f927.')
    torch.cuda.reset_peak_memory_stats(device)
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    policy = native_policy(model, source['model_policy'])
    frozen = version_inventory(model, exclude_gate=True)
    require(len(frozen) == 3410 and all(not p.requires_grad for p in model.parameters()), 'Frozen complete-model inventory differs.')
    original = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
    initial_records = records(gate)
    initial_banks = [gate.get_stage_part_vq_storage(stage, 0).clone() for stage in range(2)]
    targets = [NamedVAELinearTarget(f'model.layers.{i}.mlp.gate_proj', model.model.layers[i].mlp.gate_proj) for i in range(36)]

    def prime():
        ready(gate)
        result = prime_named_vae_linear_cache(targets, dtype=torch.bfloat16, clear_existing=True,
                                              group_size=8, compute_device=device)
        require(result['warmed'] == 36 and result['failed'] == 0, 'Native category group8 cache rebuild failed.')
        return result

    initial_weight = gate._cached_weight.detach().clone()
    prime()
    require(torch.equal(initial_weight, gate._cached_weight), 'Category-only rebuild changed original global-group8 weight.')
    del initial_weight
    original_logits = model(**document_batch(batches['train'], 0, device), use_cache=False).logits.detach().cpu()
    before = {'train': score(model, batches['train'], teacher['train'], deadline=deadline)}
    inventory = group_inventory(choices)
    choices = choices.to(device)

    def mutate(group, restore=False):
        start, end = group * VECTORS_PER_GROUP, (group + 1) * VECTORS_PER_GROUP
        for stage in range(2):
            view = gate.get_stage_part_vq_storage(stage, 0)[start:end]
            if restore:
                view.copy_(initial_banks[stage][start:end])
            else:
                require(torch.equal(view, initial_banks[stage][start:end]), 'Visited groups must not overlap a new proposal.')
                apply_choices(view, choices[start:end], stage)
        return prime()

    rollback = None
    if groups == 8:
        # This roundtrip is a mechanism check, not a forced accept/reject decision.
        mutate(0)
        trial = score(model, batches['train'], teacher['train'], deadline=deadline)
        mutate(0, restore=True)
        restored = score(model, batches['train'], teacher['train'], deadline=deadline)
        require(restored == before['train'] and records(gate) == initial_records,
                'Explicit proposal/restore roundtrip changed the complete native score or persistent state.')
        rollback = {'group': 0, 'trial_kl_sum': trial['kl_sum'], 'restored_original_score_exact': True,
                    'restored_all13_tensors_exact': True, 'used_for_acceptance': False}

    accepted = torch.zeros(GROUPS, dtype=torch.bool, device=device)
    current = before['train']; trajectory = []; started = time.time()
    for item in inventory[:groups]:
        begin = time.time(); group = item['group']; keep = False; previous = current
        try:
            prime_result = mutate(group)
            trial = score(model, batches['train'], teacher['train'], deadline=deadline)
            keep = trial['kl_sum'] < previous['kl_sum']
            if keep:
                accepted[group] = True; current = trial
        finally:
            if not keep:
                mutate(group, restore=True)
        row = {**item, 'before_kl_sum': previous['kl_sum'], 'trial_kl_sum': trial['kl_sum'],
               'delta_kl_sum': trial['kl_sum'] - previous['kl_sum'], 'accepted': keep,
               'after_kl_sum': current['kl_sum'], 'trial_documents': trial['documents'],
               'cache': prime_result, 'seconds': time.time() - begin}
        trajectory.append(row); observer(row)
    effective = torch.where(accepted.repeat_interleave(VECTORS_PER_GROUP), choices, torch.zeros_like(choices))
    stage_xor = []
    for stage in range(2):
        expected = initial_banks[stage].clone(); apply_choices(expected, effective, stage)
        require(torch.equal(expected, gate.get_stage_part_vq_storage(stage, 0)), 'Final mask does not reproduce the actual bank.')
        stage_xor.append(int(((effective > stage * 32) & (effective <= (stage + 1) * 32)).sum()))
    final_records = records(gate)
    require(all(final_records[k] == value for k, value in initial_records.items() if k not in BANKS), 'A noncode gate tensor changed.')
    final_train = score(model, batches['train'], teacher['train'], deadline=deadline)
    require(final_train == current and current['kl_sum'] <= before['train']['kl_sum'], 'Repeated final score or monotonicity failed.')
    state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
    endpoint = {'train': final_train}
    # These populations are first scored only after all training decisions are fixed.
    for domain in batches:
        if domain != 'train':
            endpoint[domain] = score(model, batches[domain], teacher[domain], deadline=deadline)
    final_weight = gate._cached_weight.detach().clone()
    gate.load_state_dict(original, strict=True); prime()
    restored_logits = model(**document_batch(batches['train'], 0, device), use_cache=False).logits.detach().cpu()
    require(torch.equal(restored_logits, original_logits) and records(gate) == initial_records,
            'Original13 restore did not reproduce the original complete forward.')
    for domain in batches:
        if domain != 'train':
            before[domain] = score(model, batches[domain], teacher[domain], deadline=deadline)
    gate.load_state_dict(state, strict=True); prime()
    require(torch.equal(final_weight, gate._cached_weight) and records(gate) == final_records,
            'Strict final13 reload changed native group8 weight.')
    del final_weight
    require(version_inventory(model, exclude_gate=True) == frozen and native_policy(model, source['model_policy']) == policy,
            'Non-target persistent state or native RoPE/backend changed.')
    report = {'status': 'completed', 'groups': groups, 'order': list(range(groups)), 'accepted_groups': int(accepted.sum()),
        'stage_xor': stage_xor, 'trajectory': trajectory, 'before': before, 'endpoint': endpoint,
        'initial_state': initial_records, 'final_state': final_records, 'noncode_gate_frozen_exact': True,
        'frozen_complement_count': len(frozen), 'frozen_complement_versions_and_pointers_exact': True,
        'mask_replay_exact': True, 'final_train_score_repeated_exact': True, 'original_full_logits_restore_exact': True,
        'strict_final13_and_group8_weight_reload_exact': True, 'rollback_mechanism': rollback,
        'warmup': warmup, 'native_policy': policy, 'search_and_endpoint_seconds': time.time() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'initial_source_checkpoint_id': SOURCE_ID}
    return state, {'accepted_groups': accepted.cpu(), 'effective_choices': effective.cpu()}, report
