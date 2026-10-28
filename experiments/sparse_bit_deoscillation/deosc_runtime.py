"""Native one-down score/decoder updates, common optimizer anchor and BF16 endpoints."""
from dataclasses import asdict
import random

import numpy as np

import deosc_bindings as b
from controller import BankOscillation


def frozen_stamp(model, excluded):
    return {key: (value.data_ptr(), value._version, str(value.dtype))
            for key, value in model.state_dict(keep_vars=True).items() if key not in excluded}


def cache_refs(model):
    return {name: (layer._cached_weight, tuple(getattr(layer, field) for field in
            ('_cached_sparse_residual_row_indices', '_cached_sparse_residual_col_indices', '_cached_sparse_residual_values')))
            for name, layer in model.named_modules() if isinstance(layer, b.core.VAELinear)}


def assert_caches(model, refs):
    for name, (weight, sparse) in refs.items():
        layer = model.get_submodule(name)
        b.require(layer._cached_weight is weight and all(getattr(layer, field) is expected
            for field, expected in zip(('_cached_sparse_residual_row_indices', '_cached_sparse_residual_col_indices',
                                        '_cached_sparse_residual_values'), sparse)), 'Frozen native cache changed: ' + name)


def install(model):
    from sparse_bit_tuning.config import SparseBitTuningConfig
    from sparse_bit_tuning.manager import SparseBitTuningManager
    target = model.get_submodule(b.TARGET)
    model.requires_grad_(False)
    target._deosc_native_checkpoint_flag = target._parallel_stage_decoder.use_checkpoint
    target.enable_trainable_sparse_bit_decode_graph(parallel_stage_decode=True)
    for p in target._parallel_stage_decoder.parameters():
        p.data = p.data.float()
    target._decoder_compute_dtype = b.t.bfloat16
    target._parallel_stage_decoder.use_checkpoint = True
    target.prime_sparse_residual_cache(dtype=b.t.bfloat16)
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    b.require(len(active) == 6 and all(name.startswith(b.TARGET + '._parallel_stage_decoder.')
        and p.dtype == b.t.float32 for name, p in active.items()), 'Only six real FP32 decoder leaves may train.')
    manager = SparseBitTuningManager(root_model=model, targets=[(b.TARGET, target)],
        target_devices={b.TARGET: b.t.device('cuda:0')}, training_seed=0, streaming=False,
        calibration_dtype=b.t.bfloat16, config=SparseBitTuningConfig(enabled=True, active_ratio=1.,
            optimizer='adam', bit_lr=b.BIT_LR, weight_decay=0., round_steps=512,
            proxy_coordinates='decoder_sensitivity'))
    manager.initialize_scores()
    manager.configure_schedule(total_optimizer_steps=512)
    # This is the common experimental protocol for every arm. Production early
    # round termination would confound score reanchoring with moment resets.
    manager.bit_round_steps = manager.stable_steps = b.FINAL_STEPS + 1
    b.require(sum(spec.n_active for spec in manager.bank_specs) == 147849216,
              'The full native L18-down bit geometry changed.')
    trackers = {spec.canonical_key: BankOscillation(manager.score_module.score_view(spec),
        manager.score_module.coordinate_scales[spec.canonical_key]) for spec in manager.bank_specs}
    optimizer = b.t.optim.AdamW(list(active.values()), lr=b.DECODER_LR, betas=(.9, .95), eps=1e-8, weight_decay=0)
    mutable = set(active) | {b.TARGET + '.vq_weight', b.TARGET + '.vq_weight_s1'}
    excluded = mutable | {'sparse_bit_tuning.score_chunks.0'}
    frozen = frozen_stamp(model, excluded)
    b.require(len(frozen) == 4063 and len(mutable) == 8,
              'Only six decoder fields and two code banks may change from bestabs4071.')
    refs = {name: value for name, value in cache_refs(model).items() if name != b.TARGET}
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    return target, active, manager, trackers, optimizer, frozen, excluded, refs


def update(model, active, manager, trackers, optimizer, training, record, step, *, freeze, deadline, emit):
    group = (step - 1) % 32
    model.zero_grad(set_to_none=True)
    pending = None
    def begin(identity):
        nonlocal pending
        b.require(pending is None, 'The preceding original loss was not consumed.')
        pending = identity
    def consume(loss, identity, domain, case_ids, *, task=None):
        nonlocal pending
        b.require(identity == pending and bool(b.t.isfinite(loss)), 'A finite original loss is required.')
        row = {'identity': identity, 'domain': domain, 'case_ids': case_ids,
               'loss_value': float(loss.detach())}
        if task is not None:
            row['task'] = task
        loss.backward()
        pending = None
        row['memory'] = b.memory()
        emit('loss_backward', step=step, group_index=group, identity=identity,
             domain=domain, weighted_loss=row['loss_value'])
        return row
    rows = b.previous.original_losses.objectives(model, training, record, group, deadline, emit, begin, consume)
    b.require(len(rows) == 16 and all(p.grad is not None and bool(b.t.isfinite(p.grad).all())
              for p in active.values()), 'Every complete original group must accumulate16 real backwards.')
    manager.bit_optimizer.validate_gradients()
    b.require(all(bool(b.t.isfinite(p.grad).all()) for p in manager.score_module.score_chunks),
              'The complete real score gradients must be finite.')
    norm = float(b.t.nn.utils.clip_grad_norm_(list(active.values()), 1.5, error_if_nonfinite=True))
    lr = b.DECODER_LR * (1 - (step - 1) / 500)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.step()
    telemetry = None if freeze else asdict(manager.optimizer_step())
    model.zero_grad(set_to_none=True)
    observed = {spec.canonical_key: trackers[spec.canonical_key].observe(
        manager.score_module.score_view(spec), step=step) for spec in manager.bank_specs}
    if telemetry is not None:
        b.require(not telemetry['round_ended'] and telemetry['global_bit_round'] == 0
            and sum(row['step_flip_events'] for row in observed.values()) == telemetry['step_flip_count'],
            'Every observed score crossing must equal the native packed SET event count without a round reset.')
    else:
        b.require(all(row['step_flip_events'] == 0 and row['natural_u_travel'] == 0
                      for row in observed.values()), 'Freeze-codes must preserve the actual anchor scores.')
    b.require(all(int(optimizer.state[p]['step']) == step for p in active.values()), 'Decoder Adam clock mismatch.')
    return {'step': step, 'group_index': group, 'loss_rows': rows, 'decoder_learning_rate': lr,
            'actual_bit_learning_rate': 0. if freeze else manager.bit_optimizer.lr,
            'bit_state_step': manager.bit_round_step, 'decoder_Adam_step': step,
            'decoder_gradient_norm_before_clip': norm, 'native_telemetry': telemetry,
            'observations': observed, 'domain_losses': {domain: sum(row['loss_value'] for row in rows
                if (row['domain'] if row['domain'] in ('M', 'C1-R') else 'QA') == domain)
                for domain in ('M', 'QA', 'C1-R')}, 'memory': b.memory()}


def cpu_tree(value):
    if b.t.is_tensor(value):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def rng_state():
    numpy = np.random.get_state()
    return {'python': random.getstate(), 'torch': b.t.get_rng_state(), 'cuda': b.t.cuda.get_rng_state(),
            'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:])}


def restore_rng(state):
    random.setstate(state['python'])
    b.t.set_rng_state(state['torch'])
    b.t.cuda.set_rng_state(state['cuda'])
    name, vector, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.array(vector, dtype=np.uint32), position, gaussian, cached))


def anchor(active, manager, trackers, optimizer, *, step):
    b.require(step > 0 and manager.bit_round_step == step and optimizer.state,
              'The common anchor must contain genuine nonempty optimizers.')
    return {'format': 'sparse_bit_deoscillation_anchor_v1', 'source_checkpoint_id': b.SOURCE_ID,
            'global_step': step, 'target': b.TARGET, 'decoder': cpu_tree({key: p for key, p in active.items()}),
            'decoder_optimizer': cpu_tree(optimizer.state_dict()), 'bit': manager.exact_state_dict(),
            'observation': {key: tracker.state_dict() for key, tracker in trackers.items()}, 'rng': rng_state()}


@b.t.no_grad()
def restore_anchor(saved, active, manager, trackers, optimizer):
    b.require(saved['format'] == 'sparse_bit_deoscillation_anchor_v1' and saved['source_checkpoint_id'] == b.SOURCE_ID
        and saved['target'] == b.TARGET and set(saved['decoder']) == set(active), 'Shared real optimizer anchor mismatch.')
    for key, p in active.items():
        p.copy_(saved['decoder'][key])
    optimizer.load_state_dict(saved['decoder_optimizer'])
    manager.load_exact_state_dict(saved['bit'])
    for key, tracker in trackers.items():
        tracker.load_state_dict(saved['observation'][key])
    restore_rng(saved['rng'])
    return int(saved['global_step'])


def value_proof(active, manager, optimizer):
    return {'decoder': {key: b.core.tensor_record(p.detach().cpu()) for key, p in active.items()},
            'decoder_Adam': {key: {name: b.core.tensor_record(value.detach().cpu())
                for name, value in optimizer.state[p].items() if b.t.is_tensor(value)} for key, p in active.items()},
            'bit_tensor_identity_version': [(value.data_ptr(), value._version) for value in
                list(manager.score_module.score_chunks) + list(manager.bit_optimizer.state_tensors())],
            'bit_round_step': manager.bit_round_step, 'cumulative_flip_count': manager.cumulative_flip_count}


@b.t.no_grad()
def native_weight_record(model, target, active, manager, optimizer):
    from litebsq.vae_linear_prewarm import prime_named_vae_linear_cache, _build_vae_linear_prewarm_signature
    manager.final_commit()
    before = value_proof(active, manager, optimizer)
    chunk, caches = [], {}
    masters = {key: p.data for key, p in active.items()}
    binding = target._sparse_bit_binding
    modes = {module: module.training for module in model.modules()}
    checkpoint = target._parallel_stage_decoder.use_checkpoint
    try:
        model.eval()
        for key, p in active.items():
            p.data = masters[key].bfloat16()
        target._sparse_bit_binding = None
        target.disable_trainable_decode()
        target._parallel_stage_decoder.use_checkpoint = target._deosc_native_checkpoint_flag
        # Dtype and use_checkpoint are part of the real prewarm signature. Choose
        # peers only after restoring their deployed BF16/native values.
        signature = _build_vae_linear_prewarm_signature(name=b.TARGET, base_layer=target, target_dtype=b.t.bfloat16)
        peers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, b.core.VAELinear)
            and _build_vae_linear_prewarm_signature(name=name, base_layer=layer, target_dtype=b.t.bfloat16) == signature]
        index = [name for name, _ in peers].index(b.TARGET)
        chunk = peers[index // 8 * 8:index // 8 * 8 + 8]
        caches = {name: value for name, value in cache_refs(model).items() if name in {key for key, _ in chunk}}
        b.require([name for name, _ in chunk] == [f'model.layers.{index}.mlp.down_proj' for index in range(16, 24)],
                  'The deployed L18-down native peers must remain the actual layers16..23 group8.')
        result = prime_named_vae_linear_cache(chunk, dtype=b.t.bfloat16, clear_existing=True, group_size=8)
        b.require(result['warmed'] == result['total'] == len(chunk) and result['failed'] == result['skipped'] == 0,
                  'The actual native group8 endpoint reconstruction failed.')
        for name, layer in chunk:
            if name != b.TARGET:
                b.require(b.t.equal(layer._cached_weight, caches[name][0]), 'A frozen native peer changed during readout.')
        record = b.core.tensor_record(target._cached_weight.detach().cpu())
    finally:
        for key, p in active.items():
            p.data = masters[key]
        target.enable_trainable_sparse_bit_decode_graph(parallel_stage_decode=True)
        target._parallel_stage_decoder.use_checkpoint = checkpoint
        target._sparse_bit_binding = binding
        for module, training in modes.items():
            module.training = training
        for name, layer in chunk:
            layer._cached_weight = caches[name][0]
            for field, value in zip(('_cached_sparse_residual_row_indices', '_cached_sparse_residual_col_indices',
                                      '_cached_sparse_residual_values'), caches[name][1]):
                setattr(layer, field, value)
    b.require(value_proof(active, manager, optimizer) == before, 'Native BF16 snapshot altered live masters or Adam.')
    return record, {'native_group8': [name for name, _ in chunk], 'live_master_and_Adam_preserved': True}


@b.t.no_grad()
def endpoint(path, model, target, active, manager, optimizer, *, step, arm):
    b.require(not path.exists(), 'Never overwrite an endpoint.')
    path.mkdir(parents=True)
    native, proof = native_weight_record(model, target, active, manager, optimizer)
    current = model.state_dict()
    keys = set(active) | {b.TARGET + '.vq_weight', b.TARGET + '.vq_weight_s1'}
    tensors = {key: current[key].detach().to(device='cpu',
        dtype=b.t.bfloat16 if key in active else b.t.uint8, copy=True).contiguous() for key in sorted(keys)}
    payload = {'format': 'sparse_bit_deoscillation_endpoint_v1', 'source_checkpoint_id': b.SOURCE_ID,
        'source_dir': str(b.SOURCE), 'target': b.TARGET, 'global_step': step, 'arm': arm, 'tensors': tensors}
    b.t.save(payload, path / 'payload.pt')
    loaded = b.t.load(path / 'payload.pt', map_location='cpu', weights_only=True)
    records = {key: b.core.tensor_record(value) for key, value in tensors.items()}
    b.require({key: b.core.tensor_record(value) for key, value in loaded['tensors'].items()} == records,
              'The actual eight-field endpoint failed save/reload.')
    report = {'status': 'saved', **{key: value for key, value in payload.items() if key != 'tensors'},
        'payload': b.binding(path / 'payload.pt'), 'tensor_records': records,
        'native_weight_record': native, 'native_snapshot': proof,
        'actual_bit_learning_rate': 0. if arm == 'freeze_codes' else manager.bit_optimizer.lr,
        'configured_bit_optimizer_learning_rate': manager.bit_optimizer.lr,
        'bit_optimizer_updates_enabled': arm != 'freeze_codes', 'bit_state_step': manager.bit_round_step,
        'decoder_Adam_step': step, 'source_pins': b.own_bindings(), 'frozen_source_fields': 4063,
        'score_state_is_deployment_payload': False}
    b.write(path / 'endpoint.json', report)
    return report
