"""Matched fresh64 D versus D plus original signed-INT8 q; original train-only OPD."""
import gc
import math
import time
import qat_common as c
import qat_rollout as rollout
from qat_controller import QController
from qat_state import optimization_proof, state_steps
from pair_verify import sparse_consumers

torch = c.torch


def fit(model, data, targets, archive, r_targets, record, arm, phase, output, observe, deadline):
    indices = c.SHORT_GROUPS if phase == 'short2' else list(range(32)) * 2
    c.require(arm in c.ARMS and phase in ('short2', 'formal64') and (phase != 'short2' or arm == 'DQ'),
        'Use declared fresh D/DQ64 or new-path DQ short2 only.')
    initial = c.native.copy_decoders(model); modules = c.native.modules(model)
    initial_q = {name + '.protected_input_qvalues': module.protected_input_qvalues.detach().cpu().clone()
        for name, module in modules.items()}
    mutable = c.mutable(record, arm)
    frozen_versions = c.native.frozen_versions(model, mutable)
    frozen = c.state_proof(model.state_dict(), mutable)
    c.require(frozen == record['frozen_state'][arm], 'Fresh same-a1 frozen complement differs.')
    decoder = c.native.training_mode(model)
    controller = QController(model, modules, record) if arm == 'DQ' else None
    qcoords = controller.parameters if controller is not None else {}
    active = {**decoder, **qcoords}; parameters = list(active.values())
    groups = [{'params': list(decoder.values()), 'lr': 1e-6, 'name': 'D'}]
    if qcoords: groups.append({'params': list(qcoords.values()), 'lr': .03, 'name': 'q'})
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8, weight_decay=0.)
    identities = {name: id(p) for name, p in active.items()}
    c.require(not optimizer.state and len(active) == (1568 if arm == 'DQ' else 1344), 'Fresh actual Adam ownership required.')
    by_id = {case['case_id']: case for case in data['training_cases']}
    seen = {name: False for name in active}
    teacher = c.load_teacher(); teacher_before = rollout.teacher_stamp(teacher)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c.HF, use_fast=True, local_files_only=True)
    trajectory = []
    for step, index in enumerate(indices, 1):
        c.tick(deadline); started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        group = data['groups'][index]
        m = [by_id[key] for key in group['case_ids'][:32]]
        qa = [by_id[key] for key in group['case_ids'][32:]]
        c.require(len(group['case_ids']) == 40 and len(qa) == 8 and all(case['domain'] == 'mmlu' for case in m),
            'Preserve original complete M32/QA8 groups.')
        with sparse_consumers(modules, phase == 'short2' and step == 1) as native_sparse:
            generated, rollout_rows, transition = rollout.collect(model, teacher, tokenizer, qa, modules,
                active, optimizer, deadline, controller)
        c.require(transition['decoder']['initialized_leaves'] == (0 if step == 1 else len(active))
            and transition['decoder']['optimizer_steps'] == ([] if step == 1 else [step - 1]),
            'Native OPD must preserve continuous nonempty Adam state after update1.')
        observe('native_QA_rollouts_complete', {'step': step, 'group_index': index, 'rows': rollout_rows, 'transition': transition})
        with sparse_consumers(modules, phase == 'short2' and step == 1) as live_sparse:
            loss = c.objective.kl(c.objective.mmlu_distribution(model, m), torch.cat([targets[case['case_id']] for case in m])) * .5
            c.require(torch.isfinite(loss).item(), 'Nonfinite complete M32 mixed-target KL.')
            m_loss = float(loss.detach()); loss.backward(); del loss
        observe('M_backward_complete', {'step': step, 'group_index': index, 'weighted_loss': m_loss})
        qa_loss, docs = 0., []
        for original_case in qa:
            c.tick(deadline); key = original_case['case_id']; case = generated[key]
            target = rollout.target(teacher, case)
            logp = c.objective.log_distribution(c.objective.answer_logits(model, case), case)
            loss = c.objective.kl(logp, target) * (.5 / 8)
            c.require(torch.isfinite(loss).item(), 'Nonfinite complete native QA KL.')
            value = float(loss.detach()); qa_loss += value; loss.backward(); del loss, logp, target
            docs.append({'case_id': key, 'domain': case['domain'], 'input_tokens': len(case['input_ids']),
                'target_tokens': len(case['target_ids']), 'document_weight': .5 / 8, 'weighted_loss': value})
            observe('QA_backward_complete', {'step': step, **docs[-1]})
        r_loss, r_rows = 0., []
        for task in c.TASKS:
            c.tick(deadline); cases = archive['tasks'][task][index * 4:(index + 1) * 4]
            loss, stats = c.retention.task_kl(model, cases, r_targets)
            value = float(loss.detach()); r_loss += value; loss.backward(); del loss
            r_rows.append({'task': task, 'case_ids': [case['case_id'] for case in cases], 'weighted_loss': value, **stats})
            observe('R_backward_complete', {'step': step, 'group_index': index, **r_rows[-1]})
        c.require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item() for p in parameters)
            and all(p.grad is None for name, p in model.named_parameters() if name not in decoder),
            'Every actual D/q FP32 gradient must be finite; registered frozen parameters must receive none.')
        for name, p in active.items(): seen[name] |= bool(torch.count_nonzero(p.grad))
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.5, error_if_nonfinite=True))
        c.require(math.isfinite(norm) and norm > 0, 'Require finite nonzero shared gradient norm.')
        factor = 1 - (step - 1) / 500
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = (1e-6 if parameter_group['name'] == 'D' else .03) * factor
        before_update = {name: p.detach().cpu().clone() for name, p in active.items()} if phase == 'short2' else None
        state_ids = {name: id(optimizer.state[p]) for name, p in active.items() if p in optimizer.state}
        optimizer.step()
        c.require({name: id(p) for name, p in active.items()} == identities
            and all(id(optimizer.state[active[name]]) == identity for name, identity in state_ids.items()),
            'Keep original Parameter and nonempty Adam state objects throughout64 updates.')
        continuous = state_steps(active, optimizer, step)
        motion = {name: float((p.detach() - before_update[name].to(p.device)).abs().max())
            for name, p in active.items()} if before_update is not None else None
        if motion is not None:
            c.require(all(math.isfinite(value) and value > 0 for value in motion.values()), 'All D/q leaves need real FP32 updates.')
            del before_update
        c.require(c.native.frozen_versions(model, mutable) == frozen_versions, 'Frozen source identity/version changed.')
        qstats = controller.statistics(advance=True) if controller is not None else None
        torch.cuda.synchronize()
        row = {'step': step, 'group_index': index, 'arm': arm,
            'group_LRs': {group['name']: group['lr'] for group in optimizer.param_groups},
            'M_weighted_loss': m_loss, 'QA_weighted_loss': qa_loss, 'R_weighted_loss': r_loss,
            'objective': m_loss + qa_loss + r_loss, 'M_case_ids': group['case_ids'][:32],
            'QA_documents': docs, 'R_tasks': r_rows, 'rollouts': rollout_rows, 'rollout_transition': transition,
            'gradient_norm_before_shared_clip': norm, 'optimizer_state': continuous, 'q_statistics': qstats,
            'FP32_leaf_update_absmax': motion,
            'native_sparse_consumers': native_sparse if phase == 'short2' and step == 1 else None,
            'live_sparse_consumers': live_sparse if phase == 'short2' and step == 1 else None,
            'seconds': time.monotonic() - started, **rollout.memory()}
        trajectory.append(row); observe('matched_optimizer_update', row)
    extra = []
    if phase == 'short2':
        for item in record['extra_short_retention_batches']:
            c.tick(deadline); optimizer.zero_grad(set_to_none=True)
            index, task = item['group_index'], item['task']
            loss, stats = c.retention.task_kl(model, archive['tasks'][task][index * 4:(index + 1) * 4], r_targets)
            value = float(loss.detach()); loss.backward(); del loss
            c.require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in parameters),
                'Actual peak R D/q backward failed.')
            extra.append({**item, **stats, 'weighted_loss': value, 'real_forward_backward': True,
                'optimizer_updated': False, **rollout.memory()})
            observe('extra_peak_R_backward_complete', extra[-1])
    c.require(all(seen.values()) and rollout.teacher_stamp(teacher) == teacher_before,
        'Every declared D/q leaf must receive real nonzero gradients; HF teacher must remain unchanged.')
    c.require(controller is None or all(entry['calls'] > 0 for entry in controller.entries.values()),
        'All224 live protection row adapters must be consumed by the actual model.')
    del teacher, tokenizer; gc.collect(); torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True); before = optimization_proof(active, optimizer)
    endpoint = {name: p.detach().to(device='cpu', dtype=torch.bfloat16, copy=True) for name, p in decoder.items()}
    qstats = controller.statistics() if controller is not None else None
    native_commit = controller.commit_terminal(active, optimizer) if controller is not None else None
    if controller is not None:
        endpoint.update({name: value.detach().cpu().clone() for name, value in controller.native_payload().items()})
    changes = sum(int((endpoint[name] != value).sum()) for name, value in initial.items())
    c.require(changes > 0 and before['initialized_leaves'] == len(active) and before['steps'] == [len(indices)],
        'Require real D update and the declared actual update budget.')
    output.mkdir(parents=True, exist_ok=False); path = output / 'native_endpoint.pt'
    payload = {'schema': 'llama_protected_qat_native_endpoint_v1', 'source_checkpoint_id': c.SOURCE_ID,
        'steps': len(indices), 'arm': arm, 'phase': phase, 'prepared_sha256': c.sha(c.HERE / 'prepared.json'), 'state': endpoint}
    torch.save(payload, path); saved = torch.load(path, map_location='cpu', weights_only=True)
    c.require(set(saved['state']) == set(endpoint) and all(torch.equal(saved['state'][name], value) for name, value in endpoint.items())
        and before == optimization_proof(active, optimizer), 'Endpoint save/read changed actual values or optimizer.')
    c.require(c.state_proof(model.state_dict(), mutable) == frozen, 'Actual frozen complement changed.')
    del optimizer, active, parameters, decoder, qcoords, controller
    c.native.install_decoders(model, {name: endpoint[name] for name in initial})
    c.require(all(torch.equal(model.state_dict()[name].cpu(), value) for name, value in endpoint.items()), 'Native endpoint install differs.')
    with torch.no_grad():
        c.require(torch.isfinite(c.objective.answer_logits(model, by_id[data['groups'][indices[0]]['case_ids'][0]])).all().item(),
            'Committed native endpoint forward failed.')
    with torch.no_grad():
        for name, module in modules.items(): module.protected_input_qvalues.copy_(initial_q[name + '.protected_input_qvalues'].to(module.protected_input_qvalues.device))
    c.native.install_decoders(model, initial)
    c.require(all(torch.equal(model.state_dict()[name].cpu(), value) for name, value in {**initial, **initial_q}.items())
        and c.state_proof(model.state_dict(), mutable) == frozen, 'Terminal source restoration differs.')
    c.tick(deadline)
    return {'steps': len(indices), 'arm': arm, 'phase': phase, 'group_indices': indices, 'trajectory': trajectory,
        'trainable_tensors': 1568 if arm == 'DQ' else 1344,
        'trainable_parameters': 117324800 if arm == 'DQ' else 4078592, 'frozen_tensors': frozen['tensors'],
        'all_D_q_leaves_nonzero_finite_gradient': True, 'frozen_before': frozen, 'frozen_after': frozen,
        'extra_peak_R_checks': extra,
        'optimizer_before_native_copy': before, 'native_q_commit': native_commit, 'q_statistics': qstats,
        'compact_save_reload_exact': True, 'native_endpoint_installed_exact': True, 'source_restored_exact': True,
        'fresh_optimizer': True, 'full_F_teacher_unchanged': True, 'D_BF16_changed_elements': changes, 'endpoint': c.binding(path)}
