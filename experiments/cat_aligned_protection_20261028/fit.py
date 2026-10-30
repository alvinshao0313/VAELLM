"""Fresh actual aligned02 D+scale32 or D+q64 with current-student native OPD and R."""
import math
from pathlib import Path
import time
import bindings as b
import data
import rollout
from controller import ProtectionController
from snapshot import save_snapshot


def fit(model, teacher, tokenizer, recovery, retention, source_targets, record, observe, deadline):
    choice = b.ARMS[record['arm']]
    indices, r_indices = record['group_indices'], record['R_group_indices']
    b.require(record['steps'] == record['planned_steps'] == len(indices) == len(r_indices)
        and (record['phase'] == 'short2' and indices == [0, 19] and r_indices == [17, 28]
            or record['phase'] == 'formal' and indices == r_indices == list(range(32)) * (choice['steps']//32)),
        'Only fixed representative short2 or exact fresh ordered32/64 is supported.')
    modules = b.selected_modules(model, record)
    initial = b.fit_core.copy_parameters(modules)
    initial_protection = {key: model.state_dict()[key].detach().cpu().clone() for key in record['protection_parameters']}
    mutable = set(initial) | set(initial_protection)
    frozen = b.fit_core.frozen_proof(model, mutable)
    b.require(frozen == record['validated_frozen_state'] and frozen['tensors'] == 1659, 'Wrong actual source frozen complement.')
    active = b.fit_core.training_mode(model, modules)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train(); model.config.use_cache = False
    b.require(len(active) == 1512 and set(active) == set(record['decoder_parameters'])
        and sum(p.numel() for p in active.values()) == record['trainable_elements']
        and model.is_gradient_checkpointing and model.model.gradient_checkpointing
        and all(m._parallel_stage_decoder.use_checkpoint for m in modules.values()), 'Require source-derived all252 FP32D and native checkpoint levels.')
    controller = ProtectionController(model, modules, record)
    protection = controller.parameters
    all_active = {**active, **protection}
    parameters = list(all_active.values())
    optimizer = b.torch.optim.AdamW([dict(params=list(active.values()), lr=1e-6, name='decoder'),
        dict(params=list(protection.values()), lr=choice['learning_rate'], name='protection')],
        betas=(.9, .95), eps=1e-8, weight_decay=0)
    teacher_before = rollout.teacher_stamp(teacher)
    identities = {name: id(p) for name, p in all_active.items()}
    gradient_seen = {name: False for name in all_active}
    rows = []
    for step, index in enumerate(indices, 1):
        b.core.tick(deadline)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        b.torch.cuda.reset_peak_memory_stats()
        group = record['groups'][index]
        b.require(group['group_index'] == index and len(group['case_ids']) == 40
            and all(group['weights'][key] == (.5/32 if slot < 32 else .5/8)
                for slot, key in enumerate(group['case_ids'])), 'Original M32/QA8 document weights changed.')
        qa_cases = [recovery[key]['case'] for key in group['case_ids'][32:]]
        trajectories, generation_rows, transition = rollout.collect(model, teacher, tokenizer,
            qa_cases, modules, active, optimizer, record, deadline, controller)
        b.require(rollout.teacher_stamp(teacher) == teacher_before, 'Frozen full252HF teacher changed.')
        targets = {}
        for case in qa_cases:
            b.core.tick(deadline)
            targets[case['case_id']] = rollout.target(teacher, trajectories[case['case_id']])
        observe('current_student_rollouts_and_F_targets_ready', dict(step=step, group_index=index,
            generated_tokens=sum(row['generated_tokens'] for row in generation_rows), transition=transition))
        factor = 1 - (step-1)/500
        optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'] = 1e-6*factor, choice['learning_rate']*factor
        losses = dict(M=0., QA=0., R=0.)
        m_cases = [recovery[key]['case'] for key in group['case_ids'][:32]]
        b.require(all(case['domain'] == 'mmlu' for case in m_cases), 'Wrong native M case domain.')
        target = b.torch.cat([recovery[case['case_id']]['target_log_probs'] for case in m_cases])
        loss, _ = b.mmlu.weighted_loss(b.mmlu.log_distribution(model, m_cases), target)
        b.require(b.torch.isfinite(loss).item(), 'Nonfinite original once-mixed M loss.')
        losses['M'] = float(loss.detach()); loss.backward(); del loss
        observe('M_backward_complete', dict(step=step, weighted_loss=losses['M']))
        qa_rows = []
        for key in group['case_ids'][32:]:
            b.core.tick(deadline)
            case = trajectories[key]
            logits = b.base.answer_logits(model, case)
            loss = b.base.answer_kl(logits, case, targets[key])*(.5/8)
            b.require(b.torch.isfinite(loss).item(), 'Nonfinite current complete-prefix full-vocabulary QA KL.')
            value = float(loss.detach()); losses['QA'] += value; loss.backward(); del loss, logits
            row = dict(case_id=key, domain=case['domain'], input_tokens=len(case['input_ids']),
                answer_tokens=len(case['target_ids']), weighted_loss=value, teacher_target=b.core.tensor_record(targets[key]))
            qa_rows.append(row); observe('QA_backward_complete', dict(step=step, **row))
        r_group, r_rows = r_indices[step-1], []
        for task in b.TASKS:
            b.core.tick(deadline)
            cases = retention['tasks'][task][r_group*4:(r_group+1)*4]
            loss, shape = data.loss.task_kl(model, cases, source_targets)
            value = float(loss.detach()); losses['R'] += value; loss.backward(); del loss
            row = dict(task=task, group_index=r_group, case_ids=[case['case_id'] for case in cases], weighted_loss=value, batch=shape)
            r_rows.append(row); observe('retention_task_backward', dict(step=step, **row))
        b.require(all(p.grad is not None and p.grad.dtype == b.torch.float32 and b.torch.isfinite(p.grad).all().item() for p in parameters)
            and all(p.grad is None for name, p in model.named_parameters() if name not in active),
            'Require all1764 actual FP32 gradients and frozen nondecoder model parameters.')
        for name, p in all_active.items():
            gradient_seen[name] |= bool(b.torch.count_nonzero(p.grad))
        protection_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in protection.values()))
        norm = float(b.torch.nn.utils.clip_grad_norm_(parameters, 1.5, error_if_nonfinite=True))
        b.require(math.isfinite(norm) and norm > 0, 'Require finite nonzero combined gradient before shared clip.')
        optimizer.step()
        b.require({name: id(p) for name, p in all_active.items()} == identities and all(
            optimizer.state[p]['exp_avg'].dtype == optimizer.state[p]['exp_avg_sq'].dtype == b.torch.float32
            and int(optimizer.state[p]['step']) == step for p in parameters), 'Continuous FP32 masters/Adam changed.')
        b.torch.cuda.synchronize()
        row = dict(step=step, group_index=index, R_group_index=r_group, arm=record['arm'],
            lr=1e-6*factor, protection_lr=choice['learning_rate']*factor, domain_losses=losses,
            weighted_training_objective=sum(losses.values()), M_case_ids=group['case_ids'][:32],
            QA_case_ids=group['case_ids'][32:], QA_documents=qa_rows, R_tasks=r_rows, rollouts=generation_rows,
            transition=transition, gradient_norm_before_clip=norm, protection_gradient_norm_before_clip=protection_norm,
            parameter_identity_continuous=True, adam_step=step, protection_statistics=controller.statistics(advance=True),
            memory=rollout.memory(), seconds=time.monotonic()-started)
        rows.append(row); observe('aligned_protection_update', row)
        del trajectories, targets, generation_rows, qa_rows
    b.require(all(gradient_seen.values()) and b.fit_core.frozen_proof(model, mutable) == frozen,
        'Every leaf must receive real nonzero gradient and all1659 frozen fields remain exact.')
    metadata = dict(source_checkpoint_id=record['source_checkpoint_id'], profile=b.PROFILE, arm=record['arm'],
        phase=record['phase'], steps=len(indices), planned_steps=len(indices), group_indices=indices, R_group_indices=r_indices,
        decoder_elements=record['trainable_elements'], protection_elements=record['protection_elements'],
        prepared_sha256=record['prepared_sha256'], frozen_state=frozen, contract=b.CONTRACT)
    endpoint = save_snapshot(active, protection, optimizer, controller, metadata, Path(record['run_dir']) / 'protected_endpoint.pt')
    changes = {name: dict(elements=p.numel(), fp32_delta_l2=float((p.detach().cpu()-initial[name].float()).double().norm()),
        bf16_changed_elements=int((p.detach().cpu().bfloat16() != initial[name]).sum())) for name, p in active.items()}
    b.require(any(v['bf16_changed_elements'] for v in changes.values()) and rollout.teacher_stamp(teacher) == teacher_before,
        'Require real BF16 decoder change and unchanged full-F teacher.')
    statistics = controller.statistics()
    commit = controller.commit_terminal(active, optimizer)
    b.require(b.fit_core.frozen_proof(model, mutable) == frozen, 'Native terminal commit changed frozen fields.')
    for p in parameters:
        p.grad = None
    del optimizer, active, protection, parameters, all_active, controller
    return dict(steps=len(rows), planned_steps=len(rows), group_indices=indices, R_group_indices=r_indices, trajectory=rows,
        parameter_changes=changes, frozen_state_before=frozen, frozen_state_after=frozen,
        all_D_and_protection_leaves_received_nonzero_gradient=True, teacher_unchanged=True,
        trainable_tensors=1764, trainable_parameters=record['trainable_elements']+record['protection_elements'],
        decoder_tensors=1512, protection_tensors=252, protection_statistics=statistics, native_commit=commit,
        endpoint=endpoint), initial, initial_protection
