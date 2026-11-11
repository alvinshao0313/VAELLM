"""Optimize one real Qwen3 block with the fixed nine-entry token objective."""
from collections import defaultdict
import hashlib
import json
import math
import time

from . import bindings as b
from . import runtime
from .rolling import PROJECTIONS, ReconstructionGroup, tree_to

torch = b.torch
GROUPS = (2, 10)
BASE_LR = 1e-6


def tensor_record(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=b.digest({'value': value}))


def state_records(model, exclude=()):
    excluded = set(exclude)
    state = model.state_dict()
    b.require(all(torch.is_tensor(value) for value in state.values()),
              'The BR native/compact endpoint must contain only persistent tensors.')
    return {name: tensor_record(value) for name, value in state.items() if name not in excluded}


def proof_from_records(records, exclude=()):
    excluded = set(exclude)
    selected = {name: value for name, value in records.items() if name not in excluded}
    return dict(tensors=len(selected), sha256=hashlib.sha256(
        json.dumps(selected, sort_keys=True, allow_nan=False).encode()).hexdigest())


def frozen_proof(model, exclude=()):
    return proof_from_records(state_records(model, exclude))


def memory(device='cuda:0'):
    cpu = {}
    with open('/proc/self/status', encoding='utf-8') as stream:
        for line in stream:
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM'):
                cpu[key + '_bytes'] = int(value.split()[0]) * 1024
    gpu = (dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
        peak_allocated=torch.cuda.max_memory_allocated(device), peak_reserved=torch.cuda.max_memory_reserved(device))
        if torch.cuda.is_initialized() else {})
    return dict(CPU=cpu, GPU=gpu)


def projection_parameters(student, layer_index, active):
    expected = runtime.decoder_parameters(student, layer_index)
    b.require(set(active) == set(expected) and all(active[name] is value for name, value in expected.items()),
              'Optimize every actual decoder leaf of the current seven projections only.')
    b.require({name for name, value in student.named_parameters() if value.requires_grad} == set(active)
        and all(value.requires_grad and value.dtype == torch.float32 for value in active.values()),
        'Only the current block decoder FP32 masters may require gradients.')
    prefixes = tuple(f'model.layers.{layer_index}.{name}' for name in PROJECTIONS)
    owners = {}
    for name in active:
        matches = [prefix for prefix in prefixes if name.startswith(prefix + '.')]
        b.require(len(matches) == 1, 'Decoder leaf has an ambiguous or unrelated projection owner: ' + name)
        owners[name] = matches[0]
    b.require(set(owners.values()) == set(prefixes) and len(active) == len({id(value) for value in active.values()}),
              'All seven actual projection decoders must have unique optimizer ownership.')
    return owners


def entry_objective(block, batch, denominator, device):
    b.require(type(denominator) is int and denominator > 0,
              'Use the complete group valid-token by hidden-size denominator.')
    hidden, target = batch.hidden_states, batch.target_hidden_states
    mask = batch.valid_mask
    b.require(hidden.device.type == target.device.type == mask.device.type == 'cpu'
        and hidden.shape == target.shape and hidden.dtype == target.dtype == torch.bfloat16
        and hidden.shape[:2] == mask.shape and mask.dtype == torch.bool
        and not hidden.requires_grad and not target.requires_grad,
        'Use the actual detached CPU stream entries and original valid-token masks.')
    result = block(hidden.to(device), **tree_to(batch.forward_kwargs, device))
    output = result[0] if isinstance(result, tuple) else result
    b.require(output.shape == hidden.shape and output.dtype == torch.bfloat16
        and bool(torch.isfinite(output).all()), 'The real complete block must return finite BF16 hidden states.')
    target = target.to(device)
    difference = (output.float() - target.float()).masked_fill(~mask.to(device).unsqueeze(-1), 0.)
    sse = difference.square().sum(dtype=torch.float32)
    energy = target.float().masked_fill(~mask.to(device).unsqueeze(-1), 0.).square().sum(dtype=torch.float32)
    loss = sse / denominator
    b.require(loss.dtype == torch.float32 and bool(torch.isfinite(loss)), 'Nonfinite FP32 valid-token block MSE.')
    return loss, dict(squared_error=float(sse.detach()), teacher_squared_energy=float(energy),
                     valid_tokens=int(mask.sum()), padded_tokens=mask.numel())


def check_group(group, hidden_size):
    b.require(isinstance(group, ReconstructionGroup) and len(group.batches) == 9
        and group.valid_tokens == sum(int(batch.valid_mask.sum()) for batch in group.batches)
        and group.denominator == group.valid_tokens * hidden_size,
        'Keep one complete M32 plus eight QA entries under one exact group denominator.')


@torch.no_grad()
def evaluate_layer(student, layer_index, streams, *, device='cuda:0'):
    block = student.model.layers[layer_index]
    rows = []
    for index in GROUPS:
        group = streams.group(index)
        check_group(group, student.config.hidden_size)
        entries = []
        for batch in group.batches:
            loss, row = entry_objective(block, batch, group.denominator, device)
            entries.append(row)
            del loss
        sse = sum(row['squared_error'] for row in entries)
        energy = sum(row['teacher_squared_energy'] for row in entries)
        rows.append(dict(group_index=index, valid_tokens=group.valid_tokens, denominator=group.denominator,
            MSE=sse/group.denominator, teacher_mean_square=energy/group.denominator,
            normalized_squared_error=sse/energy if energy else None, entries=entries))
    return rows


def gradient_check(student, active, owners):
    norms, leaves, zero_leaves = defaultdict(float), defaultdict(int), defaultdict(int)
    for name, parameter in active.items():
        grad = parameter.grad
        b.require(grad is not None and grad.dtype == torch.float32 and bool(torch.isfinite(grad).all()),
                  'Missing or nonfinite actual FP32 decoder gradient: ' + name)
        squared = float(grad.double().square().sum())
        norms[owners[name]] += squared
        leaves[owners[name]] += 1
        zero_leaves[owners[name]] += squared == 0.
    b.require(all(math.isfinite(value) and value > 0 for value in norms.values())
        and all(parameter.grad is None for name, parameter in student.named_parameters() if name not in active),
        'Every current projection must receive nonzero finite gradients; frozen parameters must receive none.')
    return {name: dict(norm=math.sqrt(value), leaves=leaves[name], zero_gradient_leaves=zero_leaves[name])
            for name, value in norms.items()}


@torch.no_grad()
def parameter_motion(active, initial, owners):
    rows = defaultdict(lambda: dict(delta_squared=0., initial_squared=0., bf16_changed_elements=0, elements=0))
    for name, parameter in active.items():
        now, old = parameter.detach().cpu(), initial[name]
        row = rows[owners[name]]
        row['delta_squared'] += float((now - old.float()).double().square().sum())
        row['initial_squared'] += float(old.double().square().sum())
        row['bf16_changed_elements'] += int((now.bfloat16() != old).sum())
        row['elements'] += parameter.numel()
    result = {name: dict(fp32_delta_L2=math.sqrt(row['delta_squared']),
        fp32_relative_L2=math.sqrt(row['delta_squared']/row['initial_squared']) if row['initial_squared'] else None,
        bf16_changed_elements=row['bf16_changed_elements'], elements=row['elements'],
        bf16_survival_fraction=row['bf16_changed_elements']/row['elements']) for name, row in rows.items()}
    b.require(all(row['fp32_delta_L2'] > 0 and row['bf16_changed_elements'] > 0 for row in result.values()),
              'Every actual projection update must survive the eventual BF16 decoder commit.')
    return result


def optimizer_check(optimizer, active, step):
    for name, parameter in active.items():
        state = optimizer.state[parameter]
        b.require(int(state['step']) == step
            and state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
            and state['exp_avg'].device == state['exp_avg_sq'].device == parameter.device
            and bool(torch.isfinite(state['exp_avg']).all()) and bool(torch.isfinite(state['exp_avg_sq']).all())
            and bool(torch.isfinite(parameter).all()), 'Adam state/master did not advance continuously: ' + name)
    return dict(step=step, initialized_leaves=len(active), moment_dtype='torch.float32',
                every_state_nonempty=True, foreach=False)


def fit_layer(student, layer_index, streams, active, observe, deadline, *, device='cuda:0'):
    owners = projection_parameters(student, layer_index, active)
    block = student.model.layers[layer_index]
    local_active = {name.removeprefix(f'model.layers.{layer_index}.') for name in active}
    frozen = frozen_proof(block, local_active)
    initial = {name: parameter.detach().cpu().bfloat16() for name, parameter in active.items()}
    master_ids = {name: (id(parameter), parameter.data_ptr()) for name, parameter in active.items()}
    optimizer = torch.optim.AdamW(list(active.values()), lr=BASE_LR, betas=(.9, .95), eps=1e-8,
                                  weight_decay=0., foreach=False)
    b.require(not optimizer.state, 'Each block must start with a fresh empty Adam optimizer.')
    updates = []
    try:
        for step, index in enumerate(GROUPS, 1):
            b.require(time.monotonic() < deadline, 'The bounded BR mechanism short exceeded its wall-clock budget.')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            group = streams.group(index)
            check_group(group, student.config.hidden_size)
            entries = []
            for entry_index, batch in enumerate(group.batches):
                b.require(time.monotonic() < deadline, 'The bounded BR mechanism short exceeded its wall-clock budget.')
                loss, row = entry_objective(block, batch, group.denominator, device)
                loss.backward()
                entries.append(dict(entry_index=entry_index, **row))
                del loss
                observe('BR_entry_backward', dict(layer=layer_index, step=step, group_index=index,
                    entry_index=entry_index, **row))
            gradients = gradient_check(student, active, owners)
            norm = float(torch.nn.utils.clip_grad_norm_(list(active.values()), 1.5,
                                                       error_if_nonfinite=True, foreach=False))
            b.require(math.isfinite(norm) and norm > 0., 'The single joint nine-entry gradient norm must be finite and nonzero.')
            lr = BASE_LR * (1 - (step-1)/500)
            optimizer.param_groups[0]['lr'] = lr
            state_ids = {name: id(optimizer.state[parameter]) for name, parameter in active.items() if parameter in optimizer.state}
            optimizer.step()
            adam = optimizer_check(optimizer, active, step)
            b.require(master_ids == {name: (id(parameter), parameter.data_ptr()) for name, parameter in active.items()}
                and all(id(optimizer.state[active[name]]) == value for name, value in state_ids.items()),
                'Live FP32 Parameter/storage or Adam state identity changed within one block.')
            motion = parameter_motion(active, initial, owners)
            b.require(frozen_proof(block, local_active) == frozen,
                      'Training changed a current-block code, route, protection, rotation, norm or nondecoder field.')
            torch.cuda.synchronize(device)
            row = dict(layer=layer_index, step=step, group_index=index, lr=lr,
                valid_tokens=group.valid_tokens, denominator=group.denominator,
                MSE=sum(entry['squared_error'] for entry in entries)/group.denominator,
                entries=entries, gradient_norm_before_clip=norm, gradient_norm_after_clip=min(norm, 1.5),
                clip_coefficient=min(1., 1.5/(norm+1e-6)), projection_gradients=gradients,
                parameter_motion=motion, Adam=adam, seconds=time.monotonic()-started, memory=memory(device),
                all_seven_projection_gradients_finite_nonzero=True, frozen_persistent_fields_exact=True,
                master_and_Adam_state_identity_continuous=True)
            updates.append(row)
            observe('BR_block_update', row)
    finally:
        optimizer.zero_grad(set_to_none=True)
    return dict(layer=layer_index, steps=2, group_indices=list(GROUPS), base_lr=BASE_LR,
        schedule_steps=500, warmup_steps=0, optimizer='AdamW', betas=[.9, .95], eps=1e-8,
        weight_decay=0., foreach=False, joint_clip=1.5, fresh_optimizer=True,
        trainable_leaves=len(active), trainable_elements=sum(value.numel() for value in active.values()),
        active_names=sorted(active), frozen_before=frozen, frozen_after=frozen_proof(block, local_active),
        updates=updates, no_optimizer_resume_artifact=True)
