"""Same complete M32/QA8 objective, with one joint clip then stage-Muon/aux Adam."""
import math
import time

from . import bindings as b
from . import objective

motion, opt, rollout, lifecycle = b.motion, b.optimizer, b.rollout, b.lifecycle


def gradient_stamp(active):
    return {name:dict(identity=id(p.grad),data_ptr=p.grad.data_ptr(),version=p.grad._version,
        value=b.core.tensor_record(p.grad)) for name,p in active.items()}


def fit(model,teacher,tokenizer,recovery,record,phase,observe,deadline, *, arm, full_targets):
    modules = b.mature.selected_modules(model,record)
    initial = lifecycle.copy_parameters(modules)
    frozen = lifecycle.frozen_proof(model,set(initial))
    b.require(frozen == record['validated_frozen_state'], 'Actual mixed-model frozen complement differs.')
    active = lifecycle.training_mode(model,modules)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    model.config.use_cache = False
    b.require(set(active) == set(record['decoder_parameters']) and len(active) == record['trainable_tensors']
        and sum(p.numel() for p in active.values()) == record['trainable_elements']
        and model.is_gradient_checkpointing and model.model.gradient_checkpointing
        and all((all(decoder.use_checkpoint for decoder in module.decoders) if isinstance(module,b.CompactRVQLinear) else module._parallel_stage_decoder.use_checkpoint) for module in modules.values()),
        'Keep all actual FP32 D leaves and both original checkpoint levels.')
    optimizer = opt.MuonAdam(active,record['matrix_layout'])
    teacher_before = rollout.teacher_stamp(teacher)
    parameter_ids = {name:id(p) for name,p in active.items()}
    rows = []
    for step,index in enumerate(b.STEPS[phase],1):
        b.core.tick(deadline)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        b.torch.cuda.reset_peak_memory_stats()
        group = record['groups'][index]
        b.require(group['group_index'] == index and group['qa_group_index'] == index%16
            and len(group['case_ids']) == 40, 'Keep the original complete HOT group order.')
        qa_cases = [recovery[key]['case'] for key in group['case_ids'][32:]]
        trajectories,generation_rows,transition = rollout.collect(model,teacher,tokenizer,qa_cases,
            modules,active,optimizer,record,deadline)
        b.require(rollout.teacher_stamp(teacher) == teacher_before, 'Full-F teacher changed during native transitions.')
        targets = {}
        teacher_started = time.monotonic()
        for case in qa_cases:
            b.core.tick(deadline)
            targets[case['case_id']] = rollout.target(teacher,trajectories[case['case_id']])
        teacher_seconds = time.monotonic()-teacher_started
        observe('current_rollouts_and_F_targets_ready',dict(step=step,group_index=index,documents=8,
            generated_tokens=sum(row['generated_tokens'] for row in generation_rows),
            teacher_forward_seconds=teacher_seconds,transition=transition))
        b.torch.cuda.reset_peak_memory_stats()
        losses = {'mmlu':0.,'QA':0.}
        m_cases = [recovery[key]['case'] for key in group['case_ids'][:32]]
        if arm in b.TARGET_ARMS:
            loss, M_diagnostics = objective.loss(model, m_cases, full_targets, arm, diagnostics=True)
        else:
            target = b.torch.cat([recovery[case['case_id']]['target_log_probs'] for case in m_cases])
            loss, _ = b.mature.mmlu.weighted_loss(b.mature.mmlu.log_distribution(model, m_cases), target)
            M_diagnostics = dict(original_cached_conditional4_mixed_target=True,
                M_case_ids=[case['case_id'] for case in m_cases], domain_weight=.5)
        losses['mmlu'] = float(loss.detach())
        loss.backward()
        del loss
        observe('M_backward_complete',dict(step=step,group_index=index,execution_batch_size=32,weighted_loss=losses['mmlu']))
        qa_rows = []
        for key in group['case_ids'][32:]:
            b.core.tick(deadline)
            case = trajectories[key]
            b.require(group['weights'][key] == .5/8, 'Original QA document weight changed.')
            logits = b.mature.base.answer_logits(model,case)
            loss = b.mature.base.answer_kl(logits,case,targets[key])*(.5/8)
            b.require(b.torch.isfinite(loss).item(), 'Nonfinite full-prefix/full-vocabulary OPD objective.')
            weighted = float(loss.detach())
            losses['QA'] += weighted
            loss.backward()
            del loss,logits
            row = dict(case_id=key,domain=case['domain'],input_tokens=len(case['input_ids']),
                answer_tokens=len(case['target_ids']),document_weight=.5/8,weighted_loss=weighted,
                teacher_target=b.core.tensor_record(targets[key]))
            qa_rows.append(row)
            observe('QA_backward_complete',dict(step=step,**row))
        b.require(all(p.grad is not None and p.grad.dtype == b.torch.float32 and b.torch.isfinite(p.grad).all().item()
            for p in active.values()) and all(p.grad is None for name,p in model.named_parameters() if name not in active),
            'Every actual D gradient must be finite; all nondecoder leaves remain frozen.')
        stage_gradient_norms = None
        if phase == 'short2':
            stage_gradient_norms = {name:[float(stage.double().norm()) for stage in
                opt.stages(active[name].grad,layout['stages'])] for name,layout in record['matrix_layout'].items()}
            b.require(all(module._parallel_stage_layout == [(stage,0) for stage in range(module.residual_stages)]
                for module in modules.values() if not isinstance(module,b.CompactRVQLinear)), 'Actual stage ordering changed during the live path.')
        norm = float(b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True))
        b.require(math.isfinite(norm) and norm > 0, 'Require one finite nonzero joint M+QA clip.')
        calibration = None
        if step == 1:
            master_before = lifecycle.leaves(active)
            gradient_before = gradient_stamp(active)
            calibration = optimizer.calibrate()
            b.require(lifecycle.leaves(active) == master_before and gradient_stamp(active) == gradient_before,
                'Fresh kappa calibration changed a real master/gradient object or value.')
            calibration['actual_clipped_gradient_sha256'] = b.core.mapping_digest(gradient_before)
        original_rates = opt.schedule(optimizer,step)
        multiplier = b.LR_MULTIPLIERS[arm]
        for optimizer_group in optimizer.param_groups:
            optimizer_group['lr'] *= multiplier
        rates = {optimizer_group['group_name']: optimizer_group['lr'] for optimizer_group in optimizer.param_groups}
        before_update = {name:p.detach().cpu().clone() for name,p in active.items()}
        state_ids = {name:id(optimizer.state[p]) for name,p in active.items() if p in optimizer.state}
        optimizer.step()
        b.require({name:id(p) for name,p in active.items()} == parameter_ids
            and all(id(optimizer.state[active[name]]) == identity for name,identity in state_ids.items()),
            'Continuous actual Parameter/momentum/Adam state identities changed.')
        state = opt.state_steps(active,optimizer,step)
        displacement = motion.parameter_motion(active,before_update)
        del before_update
        b.torch.cuda.synchronize()
        row = dict(step=step,group_index=index,qa_group_index=group['qa_group_index'],group_LRs=rates,
            domain_losses=losses,weighted_training_objective=sum(losses.values()),
            gradient_norm_before_clip=norm,optimizer_state=state,Muon_calibration=calibration,
            Muon_fixed_kappa=optimizer.kappa,NS_wall_seconds=optimizer.muon.last_NS_seconds,
            NS_timing='Wall time of BF16 NS5 including required finite-check synchronization; prior kernels synchronized before timing.',
            parameter_displacement=displacement,stage_matrix_gradient_norms=stage_gradient_norms,
            M_case_ids=group['case_ids'][:32],QA_case_ids=group['case_ids'][32:],QA_documents=qa_rows,
            rollouts=generation_rows,transition=transition,teacher_forward_seconds=teacher_seconds,
            student_training_memory=rollout.memory(),seconds=time.monotonic()-started,
            parameter_identity_continuous=True,student_FWBW=9,teacher_QA_FW=8,
            retention_lambda=0.,mmlu_execution_batch_size=32, M_target_arm=arm, M_diagnostics=M_diagnostics,
            original_scheduled_group_LRs=original_rates, overall_LR_multiplier=multiplier)
        rows.append(row)
        observe('mixed_Muon_OPD_update',row)
        del trajectories,targets,generation_rows,qa_rows
    changes = {name:dict(elements=p.numel(),
        fp32_delta_l2=float((p.detach().cpu()-initial[name].float()).double().norm()),
        bf16_changed_elements=int((p.detach().cpu().bfloat16() != initial[name]).sum())) for name,p in active.items()}
    b.require(any(row['bf16_changed_elements'] > 0 for row in changes.values()), 'No actual update survived BF16 deployment.')
    optimizer.zero_grad(set_to_none=True)
    b.require(rollout.teacher_stamp(teacher) == teacher_before, 'Frozen full-F teacher changed.')
    after = lifecycle.frozen_proof(model,set(initial))
    b.require(after == frozen, 'Codes/protection/norm/head changed under D-only Muon.')
    return dict(steps=len(rows),group_indices=b.STEPS[phase],trainable_tensors=len(active),
        trainable_parameters=record['trainable_elements'],trajectory=rows,parameter_changes=changes,
        frozen_state_before=frozen,frozen_state_after=after,teacher_unchanged=True,
        Muon_fixed_kappa=optimizer.kappa,optimizer_final_state=opt.state_steps(active,optimizer,len(rows))),initial
