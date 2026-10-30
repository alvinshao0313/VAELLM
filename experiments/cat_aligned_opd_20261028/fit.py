"""Fixed HOT M32/QA8 updates; only QA teacher-forcing trajectories become native OPD."""
import math
import time

from bindings import core, torch, base, fit_core, selected_modules, mmlu, STEPS
import rollout


def fit(model, teacher, tokenizer, recovery, record, phase, observe, deadline):
    modules = selected_modules(model, record)
    initial = fit_core.copy_parameters(modules)
    frozen = fit_core.frozen_proof(model, set(initial))
    core.require(frozen == record['validated_frozen_state'], 'Actual source frozen tensors differ from F_A_HOT.')
    active = fit_core.training_mode(model, modules)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train(); model.config.use_cache = False
    core.require(set(active) == set(record['decoder_parameters']) and len(active) == 1512
        and sum(p.numel() for p in active.values()) == record['trainable_elements']
        and model.is_gradient_checkpointing and model.model.gradient_checkpointing
        and all(value._parallel_stage_decoder.use_checkpoint for value in modules.values()),
        'Require all original252/1512 FP32 decoder leaves and original two checkpoint levels.')
    optimizer = torch.optim.AdamW(list(active.values()), lr=1e-6, betas=(.9, .95), eps=1e-8, weight_decay=0)
    teacher_before = rollout.teacher_stamp(teacher)
    parameter_ids = {name: id(p) for name, p in active.items()}
    rows = []
    for step, index in enumerate(STEPS[phase], 1):
        core.tick(deadline)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        group = record['groups'][index]
        core.require(group['group_index'] == index and group['qa_group_index'] == index%16
            and len(group['case_ids']) == 40, 'Bound complete HOT group changed.')
        qa_cases = [recovery[key]['case'] for key in group['case_ids'][32:]]
        trajectories, generation_rows, transition = rollout.collect(model, teacher, tokenizer, qa_cases,
            modules, active, optimizer, record, deadline)
        core.require(rollout.teacher_stamp(teacher) == teacher_before,
            'Teacher device transitions changed its Parameter identity/version/dtype or original RoPE.')
        targets = {}
        teacher_started = time.monotonic()
        for case in qa_cases:
            core.tick(deadline)
            key = case['case_id']
            targets[key] = rollout.target(teacher, trajectories[key])
        teacher_seconds = time.monotonic()-teacher_started
        observe('current_rollouts_and_F_targets_ready', {'step': step, 'group_index': index,
            'documents': 8, 'generated_tokens': sum(row['generated_tokens'] for row in generation_rows),
            'teacher_forward_seconds': teacher_seconds, 'transition': transition})
        torch.cuda.reset_peak_memory_stats()
        lr = 1e-6*(1-(step-1)/500)
        optimizer.param_groups[0]['lr'] = lr
        losses = {'mmlu': 0., 'QA': 0.}
        m_cases = [recovery[key]['case'] for key in group['case_ids'][:32]]
        m_started = time.monotonic()
        target = torch.cat([recovery[case['case_id']]['target_log_probs'] for case in m_cases])
        loss, _ = mmlu.weighted_loss(mmlu.log_distribution(model, m_cases), target)
        core.require(torch.isfinite(loss).item(), 'Nonfinite unchanged hybrid M32 loss.')
        losses['mmlu'] = float(loss.detach())
        loss.backward(); del loss
        observe('M_backward_complete', {'step': step, 'group_index': index, 'execution_batch_size': 32,
            'weighted_loss': losses['mmlu'], 'seconds': time.monotonic()-m_started})
        qa_rows = []
        for key in group['case_ids'][32:]:
            core.tick(deadline)
            case = trajectories[key]
            core.require(group['weights'][key] == .5/8, 'Original QA document weight changed.')
            q_started = time.monotonic()
            logits = base.answer_logits(model, case)
            loss = base.answer_kl(logits, case, targets[key])*(.5/8)
            core.require(torch.isfinite(loss).item(), 'Nonfinite full-prefix/full-vocabulary OPD loss.')
            weighted = float(loss.detach())
            losses['QA'] += weighted
            loss.backward(); del loss, logits
            row = {'case_id': key, 'domain': case['domain'], 'input_tokens': len(case['input_ids']),
                'answer_tokens': len(case['target_ids']), 'document_weight': .5/8, 'weighted_loss': weighted,
                'teacher_target': core.tensor_record(targets[key]), 'seconds': time.monotonic()-q_started}
            qa_rows.append(row)
            observe('QA_backward_complete', {'step': step, **row})
        core.require(all(p.grad is not None and p.grad.dtype == torch.float32
            and torch.isfinite(p.grad).all().item() for p in active.values())
            and all(p.grad is None for name, p in model.named_parameters() if name not in active),
            'Require every original decoder FP32 gradient; all nondecoder parameters remain frozen.')
        norm = float(torch.nn.utils.clip_grad_norm_(list(active.values()), 1.5, error_if_nonfinite=True))
        core.require(math.isfinite(norm) and norm > 0, 'Require the original finite nonzero shared decoder gradient.')
        optimizer.step()
        core.require({name: id(p) for name, p in active.items()} == parameter_ids
            and all(optimizer.state[p]['exp_avg'].dtype == optimizer.state[p]['exp_avg_sq'].dtype == torch.float32
                and int(optimizer.state[p]['step']) == step for p in active.values()),
            'Persistent FP32 Adam state/Parameter identity or update counter changed.')
        torch.cuda.synchronize()
        row = {'step': step, 'group_index': index, 'qa_group_index': group['qa_group_index'], 'lr': lr,
            'domain_losses': losses, 'weighted_training_objective': sum(losses.values()),
            'M_case_ids': group['case_ids'][:32], 'QA_case_ids': group['case_ids'][32:],
            'QA_documents': qa_rows, 'rollouts': generation_rows, 'transition': transition,
            'teacher_forward_seconds': teacher_seconds, 'student_training_memory': rollout.memory(),
            'seconds': time.monotonic()-started, 'gradient_norm_before_clip': norm,
            'adam_step': step, 'parameter_identity_continuous': True,
            'student_FWBW': 9, 'teacher_QA_FW': 8,
            'retention_lambda': 0., 'mmlu_execution_batch_size': 32}
        rows.append(row)
        observe('nativegreedy_OPD_hotpot_update', row)
        del trajectories, targets, generation_rows, qa_rows
    changes = {name: {'elements': p.numel(),
        'fp32_delta_l2': float((p.detach().cpu()-initial[name].float()).double().norm()),
        'bf16_changed_elements': int((p.detach().cpu().bfloat16() != initial[name]).sum())}
        for name, p in active.items()}
    core.require(any(row['bf16_changed_elements'] > 0 for row in changes.values()),
        'Require the original update to survive native BF16 deployment.')
    for p in active.values():
        p.grad = None
    core.require(rollout.teacher_stamp(teacher) == teacher_before, 'Frozen F teacher changed during scoring.')
    del optimizer, active
    after = fit_core.frozen_proof(model, set(initial))
    core.require(after == frozen, 'Codes/protection/norm/head or other frozen source state changed.')
    return {'steps': len(rows), 'group_indices': STEPS[phase], 'learning_rate': 1e-6,
        'trainable_tensors': 1512, 'trainable_parameters': record['trainable_elements'], 'student_projection_count': 252,
        'trajectory': rows, 'parameter_changes': changes, 'frozen_state_before': frozen,
        'frozen_state_after': after, 'teacher_unchanged': True,
        'student_FWBW': len(rows)*9, 'teacher_QA_FW': len(rows)*8,
        'instrumentation': 'decoder/master/Adam exact checks and native generation tracing included in wall time'}, initial
