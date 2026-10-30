"""Exact two-group live optimization proofs and genuine terminal native compact payloads."""
from pathlib import Path
import bindings as b


def optimization_proof(decoder, protection, optimizer):
    b.require(len(decoder) == 1512 and len(protection) == 252 and not set(decoder) & set(protection)
        and len(optimizer.param_groups) == 2, 'Require all1764 actual continuous optimizer leaves.')
    rows, groups = {}, []
    for group, (name, parameters) in zip(optimizer.param_groups, (('decoder', decoder), ('protection', protection)), strict=True):
        b.require(group['name'] == name and len(group['params']) == len(parameters)
            and all(a is e for a, e in zip(group['params'], parameters.values(), strict=True)), 'Continuous optimizer references changed.')
        groups.append(dict(name=name, lr=group['lr'], parameter_ids=[id(p) for p in group['params']]))
        for key, p in parameters.items():
            state = optimizer.state.get(p, {})
            b.require(p.dtype == b.torch.float32 and p.requires_grad and b.torch.isfinite(p).all().item()
                and (not state or set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                    and state['exp_avg'].dtype == state['exp_avg_sq'].dtype == b.torch.float32), 'Changed FP32 master/moments.')
            rows[key] = dict(parameter_id=id(p), data_ptr=p.data_ptr(), version=p._version, master=b.core.tensor_record(p),
                gradient=None if p.grad is None else dict(id=id(p.grad), data_ptr=p.grad.data_ptr(), value=b.core.tensor_record(p.grad)),
                state_id=None if not state else id(state), adam={k: dict(id=id(v), data_ptr=v.data_ptr(), value=b.core.tensor_record(v))
                    for k, v in state.items()})
    active = {**decoder, **protection}
    return dict(tensors=len(rows), groups=groups, sha256=b.core.mapping_digest(rows),
        initialized_leaves=sum(bool(optimizer.state.get(p, {})) for p in active.values()),
        adam_steps=sorted({int(optimizer.state[p]['step']) for p in active.values() if optimizer.state.get(p, {})}))


@b.torch.no_grad()
def save_snapshot(decoder, protection, optimizer, controller, metadata, path):
    path, steps = Path(path), metadata['steps']
    arm, phase = metadata['arm'], metadata['phase']
    b.require(not path.exists() and phase in ('short2', 'formal') and steps == metadata['planned_steps']
        and steps == (2 if phase == 'short2' else b.ARMS[arm]['steps']), 'Save only genuine terminal actual/planned2,32,64.')
    before, buffers = optimization_proof(decoder, protection, optimizer), controller.buffer_proof()
    b.require(before['initialized_leaves'] == 1764 and before['adam_steps'] == [steps], 'All1764 continuous Adam states must match actual terminal.')
    state = {key: p.detach().to(device='cpu', dtype=b.torch.bfloat16, copy=True) for key, p in decoder.items()}
    native = {key: value.to(device='cpu', copy=True) for key, value in controller.native_payload().items()}
    expected_dtype = b.torch.bfloat16 if controller.kind == 'scale' else b.torch.uint8
    b.require(len(state) == 1512 and sum(v.numel() for v in state.values()) == metadata['decoder_elements']
        and len(native) == 252 and sum(v.numel() for v in native.values()) == metadata['protection_elements']
        and all(v.dtype == b.torch.bfloat16 and v.is_contiguous() and b.torch.isfinite(v).all().item() for v in state.values())
        and all(v.dtype == expected_dtype and v.is_contiguous() for v in native.values())
        and (controller.kind != 'scale' or all(b.torch.isfinite(v).all().item() and (v > 0).all().item() for v in native.values())),
        'Compact must retain source-shaped native BF16D plus BF16scale/uint8q only.')
    payload = dict(schema='CG_aligned_protection_endpoint_v1', **metadata, state=state, protection=native)
    b.torch.save(payload, path)
    saved = b.torch.load(path, map_location='cpu', weights_only=True)
    b.require(set(saved) == set(payload) and all(saved[k] == v for k, v in metadata.items())
        and all(set(saved[k]) == set(v) and all(b.torch.equal(saved[k][key], value) for key, value in v.items())
            for k, v in (('state', state), ('protection', native))), 'Actual native compact save/reload differs.')
    after = optimization_proof(decoder, protection, optimizer)
    b.require(before == after and buffers == controller.buffer_proof(), 'Saving altered live optimization/native buffers.')
    return dict(path=str(path), sha256=b.sha256(path), bytes=path.stat().st_size, steps=steps,
        planned_steps=steps, tensors=1764, save_reload_exact=True, before=before, after=after,
        live_optimization_and_native_buffers_unchanged_exact=True, contains_FP32_master_or_optimizer=False)


@b.torch.no_grad()
def native_roundtrip(model, initial_decoder, initial_protection, training, record, source):
    payload = b.torch.load(training['endpoint']['path'], map_location='cpu', weights_only=True)
    live = model.state_dict()
    b.require(all(b.torch.equal(live[k].cpu(), v) for k, v in payload['protection'].items()), 'Native terminal buffers differ from compact.')
    deployment = b.fit_core.restore(model, payload['state'])
    live = model.state_dict()
    mutable = set(payload['state']) | set(payload['protection'])
    b.require(len(live) == 3423 and b.fit_core.frozen_proof(model, mutable) == record['validated_frozen_state']
        and all(b.torch.equal(live[k].cpu(), v) for k, v in {**payload['state'], **payload['protection']}.items())
        and all('_decode_protected_input_weight_rows' not in m.__dict__ and '_decode_weight' not in m.__dict__
            for m in model.modules() if isinstance(m, b.core.VAELinear))
        and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Actual native endpoint install/frozen complement differs.')
    for key, value in initial_protection.items():
        live[key].copy_(value)
    b.fit_core.restore(model, initial_decoder)
    b.require(b.core.state_proof(model) == source, 'Complete actual aligned02 restoration failed.')
    return dict(native_endpoint_installed_exact=True, source_restored_exact=True,
        native_state_tensors=3423, frozen_tensors=1659, deployment=deployment)
