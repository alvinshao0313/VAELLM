"""Formal32 traversal, reusing the immutable BR numerical primitives."""
import math
import time

from . import bindings as b

torch, fit = b.torch, b.fit


@torch.no_grad()
def evaluate_block(student, index, streams):
    rows = []
    for group_index in b.GROUPS:
        group = streams.group(group_index)
        fit.check_group(group, student.config.hidden_size)
        entries = []
        for batch in group.batches:
            loss, row = fit.entry_objective(student.model.layers[index], batch, group.denominator, 'cuda:0')
            entries.append(row)
            del loss
        sse = sum(row['squared_error'] for row in entries)
        energy = sum(row['teacher_squared_energy'] for row in entries)
        rows.append(dict(group_index=group_index, valid_tokens=group.valid_tokens, denominator=group.denominator,
            MSE=sse/group.denominator, teacher_mean_square=energy/group.denominator,
            normalized_squared_error=sse/energy if energy else None, entries=entries))
    return rows


def train_block(student, index, streams, active, observe):
    owners = fit.projection_parameters(student, index, active)
    block = student.model.layers[index]
    local = {name.removeprefix(f'model.layers.{index}.') for name in active}
    frozen = fit.frozen_proof(block, local)
    initial = {name: value.detach().cpu().bfloat16() for name, value in active.items()}
    master_ids = {name: (id(value), value.data_ptr()) for name, value in active.items()}
    optimizer = torch.optim.AdamW(list(active.values()), lr=1e-6, betas=(.9, .95), eps=1e-8,
                                  weight_decay=0., foreach=False)
    b.require(not optimizer.state, 'Every BR block starts with a fresh empty Adam optimizer.')
    updates = []
    try:
        for step, group_index in enumerate(b.GROUPS, 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            group = streams.group(group_index)
            fit.check_group(group, student.config.hidden_size)
            entries = []
            for batch in group.batches:
                loss, entry = fit.entry_objective(block, batch, group.denominator, 'cuda:0')
                loss.backward()
                entries.append(entry)
                del loss
            gradients = fit.gradient_check(student, active, owners)
            norm = float(torch.nn.utils.clip_grad_norm_(list(active.values()), 1.5,
                                                       error_if_nonfinite=True, foreach=False))
            b.require(math.isfinite(norm) and norm > 0., 'The nine-entry joint gradient norm must be finite and nonzero.')
            lr = 1e-6*(1-(step-1)/500)
            optimizer.param_groups[0]['lr'] = lr
            state_ids = {name: id(optimizer.state[value]) for name, value in active.items() if value in optimizer.state}
            optimizer.step()
            adam = fit.optimizer_check(optimizer, active, step)
            b.require(master_ids == {name: (id(value), value.data_ptr()) for name, value in active.items()}
                and all(id(optimizer.state[active[name]]) == previous for name, previous in state_ids.items()),
                'FP32 master storage or Adam state identity changed inside the current block.')
            motion = fit.parameter_motion(active, initial, owners)
            b.require(fit.frozen_proof(block, local) == frozen, 'A current-block nondecoder persistent field changed.')
            torch.cuda.synchronize()
            row = dict(step=step, group_index=group_index, lr=lr, valid_tokens=group.valid_tokens,
                denominator=group.denominator, entries=entries,
                MSE=sum(entry['squared_error'] for entry in entries)/group.denominator,
                gradient_norm_before_clip=norm, gradient_norm_after_clip=min(norm, 1.5),
                clip_coefficient=min(1., 1.5/(norm+1e-6)), projection_gradients=gradients,
                parameter_motion=motion, Adam=adam, seconds=time.monotonic()-started,
                memory=fit.memory(), frozen_persistent_fields_exact=True,
                master_and_Adam_state_identity_continuous=True)
            updates.append(row)
            observe('BR32_update', dict(layer=index, step=step, group_index=group_index, lr=lr,
                MSE=row['MSE'], raw_gradient_norm=norm, seconds=row['seconds'], memory=row['memory']))
    finally:
        optimizer.zero_grad(set_to_none=True)
    b.require(len(updates) == 32 and [row['group_index'] for row in updates] == list(b.GROUPS),
              'Formal BR requires all32 actual original-group optimizer updates.')
    return dict(layer=index, steps=len(updates), group_indices=list(b.GROUPS), base_lr=1e-6,
        schedule_steps=500, warmup_steps=0, optimizer='AdamW', betas=[.9, .95], eps=1e-8,
        weight_decay=0., foreach=False, joint_clip=1.5, fresh_optimizer=True,
        trainable_leaves=len(active), trainable_elements=sum(value.numel() for value in active.values()),
        active_names=sorted(active), frozen_before=frozen, frozen_after=fit.frozen_proof(block, local),
        updates=updates, no_optimizer_resume_artifact=True)
