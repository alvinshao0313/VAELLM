"""Honest32 selected D/code or selected-MLP D recovery on current native trajectories."""
from pathlib import Path
import time
import selected_bindings as b
from selected_capture import WeightCapture
from selected_projection import restricted_step, install_banks
from selected_native import NativeOPD, chunks
from selected_state import optimization_proof, value_proof, frozen_stamp, save_endpoint


def fit(model, teacher, tokenizer, recovery, archive, R_targets, record, observe, deadline):
    stage, scope = record['stage'], record['scope']
    selected = {name: model.get_submodule(name) for name in scope['selected']}
    initial_D = b.fit_core.copy_parameters(selected)
    initial_codes = {key: model.state_dict()[key].detach().cpu().clone() for key in scope['code_parameters']}
    original = {key: v._cached_weight for key, v in model.named_modules() if isinstance(v, b.core.VAELinear)}
    native_chunks = chunks(model, selected, stage)
    flags = {key: v._parallel_stage_decoder.use_checkpoint for key, v in selected.items()}
    active = b.fit_core.training_mode(model, selected)
    for key, cache in original.items():
        if key not in selected:
            model.get_submodule(key)._cached_weight = cache
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    b.require(set(active) == set(scope['decoder_parameters']) and len(active) == b.STAGES[stage]['decoder_tensors']
        and sum(p.numel() for p in active.values()) == scope['trainable_elements']
        and all(p.dtype == b.torch.float32 for p in active.values()), 'Actual selected decoder scope changed.')
    mutable = set(scope['mutable'])
    frozen = b.fit_core.frozen_proof(model, mutable)
    b.require(frozen == scope['frozen'], 'Wrong actual parent frozen complement.')
    stamp = frozen_stamp(model, mutable)
    optimizer = b.torch.optim.AdamW(list(active.values()), lr=1e-6, betas=(.9, .95), eps=1e-8, weight_decay=0)
    b.require(not optimizer.state, 'Each selected stage starts a fresh optimizer.')
    native = NativeOPD(model, selected, active, optimizer, original, native_chunks, flags)
    layer = next(iter(selected.values())) if stage == 'J32' else None
    geometry = b.tiles.Geometry(layer) if layer is not None else None
    proxy = b.proxy.AccumulatingAdam((4096, 12032), 'cuda:0', b.ETA) if stage == 'J32' else None
    teacher_before = b.rollout.teacher_stamp(teacher)
    identities = {key: id(p) for key, p in active.items()}
    moment_ids, rows = None, []
    seen = {key: False for key in active}
    indices, r_indices = record['group_indices'], record['R_group_indices']
    b.require(indices == ([19, 0] if record['phase'] == 'short2' else list(range(32)))
        and r_indices == ([17, 28] if record['phase'] == 'short2' else list(range(32))), 'Only fixed genuine short2 or ordered formal32.')
    for step, (index, r_index) in enumerate(zip(indices, r_indices, strict=True), 1):
        b.core.tick(deadline)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if proxy is not None:
            proxy.optimizer.zero_grad(set_to_none=True)
        U_before = None if proxy is None else optimization_proof({'U': proxy.parameter}, proxy.optimizer)
        group = record['groups'][index]
        trajectories, generated, transition = native.collect(teacher, tokenizer,
            [recovery[key]['case'] for key in group['case_ids'][32:]], record, deadline, exact_values=record['phase'] == 'short2')
        b.require(proxy is None or optimization_proof({'U': proxy.parameter}, proxy.optimizer) == U_before,
            'Native generation changed continuous U/Adam.')
        b.require(b.rollout.teacher_stamp(teacher) == teacher_before, 'Frozen full-F teacher changed.')
        full_F = {key: b.rollout.target(teacher, case) for key, case in trajectories.items()}
        observe('selected_native_OPD_ready', dict(step=step, group_index=index, transition=transition))
        optimizer.param_groups[0]['lr'] = 1e-6*(1-(step-1)/500)
        b.torch.cuda.reset_peak_memory_stats()
        losses, qa_rows, r_rows = dict(M=0., QA=0., R=0.), [], []
        capture = WeightCapture(layer) if layer is not None else None
        try:
            if capture is not None:
                capture.begin('M32')
            cases = [recovery[key]['case'] for key in group['case_ids'][:32]]
            mixed = b.torch.cat([recovery[case['case_id']]['target_log_probs'] for case in cases])
            loss, per_case = b.mmlu.weighted_loss(b.mmlu.log_distribution(model, cases), mixed)
            if capture is not None:
                capture.close_forward()
            b.require(b.torch.isfinite(loss).item(), 'Nonfinite once-mixed original M loss.')
            losses['M'] = float(loss.detach())
            loss.backward()
            if capture is not None:
                capture.end_backward()
            del loss, per_case
            observe('M_backward', dict(step=step, weighted_loss=losses['M']))
            for key in group['case_ids'][32:]:
                b.core.tick(deadline)
                if capture is not None:
                    capture.begin('QA/' + key)
                case = trajectories[key]
                logits = b.base.answer_logits(model, case)
                loss = b.base.answer_kl(logits, case, full_F[key])*(.5/8)
                if capture is not None:
                    capture.close_forward()
                b.require(b.torch.isfinite(loss).item(), 'Nonfinite full-prefix full-vocabulary QA loss.')
                value = float(loss.detach())
                losses['QA'] += value
                loss.backward()
                if capture is not None:
                    capture.end_backward()
                del loss, logits
                qa_rows.append(dict(case_id=key, domain=case['domain'], input_tokens=len(case['input_ids']),
                    answer_tokens=len(case['target_ids']), weighted_loss=value))
                observe('QA_backward', dict(step=step, **qa_rows[-1]))
            before_R = {key: p.grad.detach().clone() for key, p in active.items()} if record['phase'] == 'short2' else None
            for task in b.TASKS:
                b.core.tick(deadline)
                if capture is not None:
                    capture.begin('R/' + task)
                cases = archive['tasks'][task][r_index*4:(r_index+1)*4]
                loss, shape = b.inputs.loss.task_kl(model, cases, R_targets)
                if capture is not None:
                    capture.close_forward()
                b.require(b.torch.isfinite(loss).item(), 'Nonfinite actual04 full-vocabulary retention loss.')
                value = float(loss.detach())
                losses['R'] += value
                loss.backward()
                if capture is not None:
                    capture.end_backward()
                del loss
                r_rows.append(dict(task=task, group_index=r_index, case_ids=[case['case_id'] for case in cases], weighted_loss=value, shape=shape))
                observe('R_backward', dict(step=step, **r_rows[-1]))
            accumulation = None if capture is None else capture.proof()
        finally:
            if capture is not None:
                capture.remove()
        b.require(all(p.grad is not None and p.grad.dtype == b.torch.float32 and b.torch.isfinite(p.grad).all().item()
            for p in active.values()) and all(p.grad is None for key, p in model.named_parameters() if key not in active),
            'Invalid actual selected-D gradient or gradient on frozen parameters.')
        for key, p in active.items():
            seen[key] |= bool(b.torch.count_nonzero(p.grad))
        incremental_R = None if before_R is None else sum(float((p.grad-before_R[key]).double().square().sum())
            for key, p in active.items())**.5
        norm = float(b.torch.nn.utils.clip_grad_norm_(list(active.values()), 1.5, error_if_nonfinite=True))
        b.require(norm > 0, 'Require a real nonzero selected-D gradient.')
        optimizer.step()
        b.torch.cuda.synchronize()
        proof = optimization_proof(active, optimizer)
        current_ids = {key: (id(optimizer.state[p]['exp_avg']), id(optimizer.state[p]['exp_avg_sq'])) for key, p in active.items()}
        moment_ids = current_ids if moment_ids is None else moment_ids
        b.require(moment_ids == current_ids and {key: id(p) for key, p in active.items()} == identities
            and all(row['Adam_step'] == step for row in proof['parameters'].values()), 'Continuous selected-D optimizer identity/step changed.')
        projection, PV_seconds = None, 0.
        if proxy is not None:
            PV_started = time.monotonic()
            target_support = proxy.step(capture.weight[:, geometry.columns.cpu()], capture.gradient[:, geometry.columns])
            target = capture.weight.to(device='cuda:0', dtype=b.torch.float32)
            target[:, geometry.columns] = target_support
            U_after_step = optimization_proof({'U': proxy.parameter}, proxy.optimizer)
            D_values = {key: value_proof(p) for key, p in active.items()}
            projection = restricted_step(layer, native, target, deadline)
            b.require(optimization_proof(active, optimizer) == proof and {key: value_proof(p) for key, p in active.items()} == D_values
                and optimization_proof({'U': proxy.parameter}, proxy.optimizer) == U_after_step,
                'Projection/rejection changed updated D/U or either continuous optimizer.')
            projection.update(U_and_Adam_preserved_including_rejection=True, updated_D_master_values_preserved=True)
            del target_support, target
            PV_seconds = time.monotonic()-PV_started
            b.require(proxy.proof()['initializations_from_actual_weight'] == 1 and proxy.proof()['delta_decay'] == 0,
                'Absolute U may only initialize once and never reanchor.')
        native.frozen_caches()
        b.require(frozen_stamp(model, mutable) == stamp, 'Frozen persistent tensor identity/version changed.')
        row = dict(step=step, group_index=index, R_group_index=r_index, domain_losses=losses, weighted_objective=sum(losses.values()),
            D_learning_rate=optimizer.param_groups[0]['lr'], D_gradient_norm_before_clip=norm, incremental_R_gradient_norm=incremental_R,
            D_Adam_steps=[step], D_moment_identities_preserved=True, transition=transition, rollouts=generated,
            shared_gW=accumulation, W_proxy=None if proxy is None else proxy.proof(), projection=projection,
            M_case_ids=group['case_ids'][:32], QA_documents=qa_rows, R_tasks=r_rows,
            PV_extra_seconds=PV_seconds, seconds=time.monotonic()-started, memory=b.rollout.memory())
        rows.append(row)
        observe('selected_update', {key: row[key] for key in ('step', 'group_index', 'domain_losses', 'D_gradient_norm_before_clip',
            'PV_extra_seconds', 'seconds', 'memory')})
        del capture, full_F, trajectories, generated, before_R, U_before
    b.require(all(seen.values()) and b.fit_core.frozen_proof(model, mutable) == frozen
        and b.rollout.teacher_stamp(teacher) == teacher_before, 'Selected update/frozen complement/full-F contract failed.')
    if record['phase'] == 'short2':
        b.require(rows[1]['transition']['initialized_Adam_leaves'] == len(active) and rows[1]['incremental_R_gradient_norm'] > 0
            and max(v['prompt_tokens'] for v in rows[1]['rollouts']) == 32561, 'Need longest real Narrative/nonempty Adam/R second update.')
    endpoint = save_endpoint(model, active, optimizer, proxy, record, Path(record['run_dir']) / 'selected_endpoint.pt')
    payload = b.torch.load(endpoint['path'], map_location='cpu', weights_only=True, mmap=True)
    with native.native(exact_values=True) as terminal_transition:
        live = model.state_dict()
        b.require(all(b.torch.equal(live[key].cpu(), value) for key, value in payload['state'].items())
            and b.fit_core.frozen_proof(model, mutable) == frozen, 'Actual terminal native view differs from compact/frozen source.')
    changes = {key: dict(elements=p.numel(), FP32_delta_l2=float((p.detach().cpu()-initial_D[key].float()).double().norm()),
        BF16_changed_elements=int((p.detach().cpu().bfloat16() != initial_D[key]).sum())) for key, p in active.items()}
    b.require(any(row['FP32_delta_l2'] > 0 for row in changes.values()), 'No actual selected-D update occurred.')
    for p in active.values():
        p.grad = None
    del native, optimizer, proxy
    with b.torch.no_grad():
        for key, p in active.items():
            p.data = initial_D[key].to(device=p.device, copy=True)
        if stage == 'J32':
            packed = b.torch.cat([initial_codes[key].to('cuda:0') for key in b.CODE_KEYS], dim=1)
            install_banks(layer, packed)
            del packed
        for key, v in selected.items():
            v.disable_trainable_decode()
            v._parallel_stage_decoder.use_checkpoint = flags[key]
        for key, cache in original.items():
            model.get_submodule(key)._cached_weight = cache
        model.eval().requires_grad_(False)
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
    b.require(b.core.state_proof(model) == record['source_initial_state']
        and all(model.get_submodule(key)._cached_weight is cache for key, cache in original.items()),
        'Complete actual parent state/252 native cache restoration failed.')
    return dict(steps=len(rows), planned_steps=record['planned_steps'], group_indices=indices, R_group_indices=r_indices,
        trainable_tensors=len(active), trainable_parameters=scope['trainable_elements'], frozen_state_before=frozen, frozen_state_after=frozen,
        trajectory=rows, endpoint=endpoint, D_parameter_changes=changes, source_restored_exact=True, source_tensors_restored=3423,
        all252_original_cache_references_restored=True, terminal_native_equals_compact=True, terminal_transition=terminal_transition,
        all_selected_leaves_received_nonzero_gradient=True, teacher_unchanged=True, student_FW_BW=16*len(rows),
        extra_whole_backward_or_VJP=0, new_observation_FW=0, exact_resume_claimed=False)
