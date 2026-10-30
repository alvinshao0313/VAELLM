"""Exact FP32 master/gradient/Adam continuity across native OPD generation."""
import qat_common as c


def optimization_proof(active, optimizer):
    c.require(sum(len(group['params']) for group in optimizer.param_groups) == len(active)
        and {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in active.values()},
        'Each declared D/q leaf must have exactly one optimizer owner.')
    rows = {}
    for name, value in active.items():
        state = optimizer.state.get(value, {})
        rows[name] = {'identity': id(value), 'pointer': value.data_ptr(), 'version': value._version,
            'master': c.tensor_record(value), 'gradient': None if value.grad is None else c.tensor_record(value.grad),
            'optimizer_state_object': id(state) if state else None,
            'optimizer': {key: c.tensor_record(item) if isinstance(item, c.torch.Tensor) else item for key, item in state.items()}}
    return {'tensors': len(rows), 'sha256': c.digest(rows),
        'groups': [{key: value for key, value in group.items() if key != 'params'} for group in optimizer.param_groups],
        'initialized_leaves': sum(bool(optimizer.state.get(p)) for p in active.values()),
        'steps': sorted({int(state['step']) for state in optimizer.state.values() if state})}


def state_steps(active, optimizer, step):
    for name, value in active.items():
        state = optimizer.state.get(value, {})
        c.require(int(state.get('step', -1)) == step and all(state[key].dtype == c.torch.float32
            and c.torch.isfinite(state[key]).all().item() for key in ('exp_avg', 'exp_avg_sq')),
            'Invalid continuous FP32 Adam state: ' + name)
    return {'initialized_leaves': len(active), 'steps': [step], 'all_state_FP32_finite': True}
