"""Small selected-D proofs and genuine terminal2/32 native compact payloads."""
import hashlib
from pathlib import Path
import selected_bindings as b


def value_proof(value):
    digest = hashlib.sha256()
    parts = (value[start:start+128] for start in range(0, len(value), 128)) if value.ndim == 2 else (value,)
    for part in parts:
        copied = part.detach().cpu().contiguous()
        digest.update(copied.reshape(-1).view(b.torch.uint8).numpy().tobytes())
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest.hexdigest())


def reference(value):
    return dict(id=id(value), data_ptr=value.data_ptr(), version=value._version, shape=list(value.shape), dtype=str(value.dtype))


def optimization_proof(active, optimizer, *, exact_values=False):
    grouped = [p for group in optimizer.param_groups for p in group['params']]
    b.require(len(grouped) == len(active) and len({id(p) for p in grouped}) == len(active)
        and {id(p) for p in grouped} == {id(p) for p in active.values()}, 'Optimizer must cover exactly declared actual leaves.')
    rows = {}
    for name, p in active.items():
        state = optimizer.state.get(p, {})
        b.require(p.dtype == b.torch.float32 and p.requires_grad and (not state or set(state) == {'step', 'exp_avg', 'exp_avg_sq'})
            and (not state or state['exp_avg'].dtype == state['exp_avg_sq'].dtype == b.torch.float32), 'FP32 master/Adam contract changed.')
        row = dict(parameter=reference(p), gradient=None if p.grad is None else reference(p.grad), requires_grad=p.requires_grad,
            Adam={k: reference(v) for k, v in state.items()}, Adam_step=None if not state else int(state['step']))
        if exact_values:
            row.update(master_value=value_proof(p), gradient_value=None if p.grad is None else value_proof(p.grad),
                Adam_values={k: value_proof(v) for k, v in state.items()})
        rows[name] = row
    return dict(parameters=rows, groups=[dict(LR=g['lr'], parameter_ids=[id(p) for p in g['params']]) for g in optimizer.param_groups],
        exact_values=exact_values)


def frozen_stamp(model, mutable):
    return {key: (v.data_ptr(), v._version, str(v.dtype), tuple(v.shape))
        for key, v in model.state_dict(keep_vars=True).items() if key not in mutable}


def terminal_steps(phase, actual, planned):
    b.require(phase in ('short2', 'formal') and actual == planned == (2 if phase == 'short2' else 32),
        'Save only a genuine independent terminal2 or honest formal32; no128 parent contract.')


@b.torch.no_grad()
def save_endpoint(model, active, optimizer, proxy, record, path):
    path = Path(path)
    terminal_steps(record['phase'], record['steps'], record['planned_steps'])
    scope = record['scope']
    before = optimization_proof(active, optimizer, exact_values=True)
    b.require(not path.exists() and all(row['Adam_step'] == record['steps'] for row in before['parameters'].values()),
        'Only actual terminal continuous-D Adam may be compacted.')
    U_before = None if proxy is None else optimization_proof({'U': proxy.parameter}, proxy.optimizer, exact_values=True)
    b.require(proxy is None or proxy.steps == record['steps'], 'Actual continuous-U step differs from D terminal.')
    live = model.state_dict()
    state = {key: live[key].detach().to(device='cpu', dtype=b.torch.bfloat16 if key in active else b.torch.uint8, copy=True)
        for key in scope['mutable']}
    b.require(len(state) == b.STAGES[record['stage']]['mutable_tensors']
        and all(list(v.shape) == list(live[key].shape) and v.dtype == (b.torch.bfloat16 if key in active else b.torch.uint8)
            and b.torch.isfinite(v).all().item() for key, v in state.items()), 'Native compact shape/dtype/finite contract changed.')
    payload = dict(schema='CG_aligned_selected_endpoint_v1', profile=b.PROFILE, stage=record['stage'], phase=record['phase'],
        steps=record['steps'], planned_steps=record['planned_steps'], source_checkpoint_id=record['source_checkpoint_id'],
        R_source_checkpoint_id=record['R_source_checkpoint_id'], prepared_sha256=record['prepared_sha256'], contract=b.CONTRACT,
        frozen_state=scope['frozen'], state=state, inference_only=True, exact_resume_claimed=False)
    b.torch.save(payload, path)
    saved = b.torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    b.require(set(saved) == set(payload) and all(saved[k] == v for k, v in payload.items() if k != 'state')
        and set(saved['state']) == set(state) and all(b.torch.equal(saved['state'][k], v) for k, v in state.items())
        and optimization_proof(active, optimizer, exact_values=True) == before
        and (proxy is None or optimization_proof({'U': proxy.parameter}, proxy.optimizer, exact_values=True) == U_before),
        'Saving changed native bytes/live D/U/gradients/Adam.')
    return {**b.binding(path), 'steps': record['steps'], 'planned_steps': record['planned_steps'], 'native_tensors': len(state),
        'save_reload_exact': True, 'live_D_and_U_optimization_unchanged': True,
        'D_Adam_steps': [record['steps']], 'U_Adam_steps': None if proxy is None else proxy.steps,
        'FP32_master_U_gW_Adam_RNG_serialized': False}
