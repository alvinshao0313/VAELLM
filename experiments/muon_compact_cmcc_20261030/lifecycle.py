"""Explicit native and compact training/deployment ownership and cache transitions."""
from contextlib import contextmanager

from litebsq.vae_linear import prime_named_vae_linear_cache
from train_utils.distill_precision import install_precision_runtime

from . import bindings as b
from . import optimizer as opt


def decoder_parameters(modules):
    result = {}
    for name, module in modules.items():
        prefix, decoder = ('decoders.', module.decoders) if isinstance(module, b.CompactRVQLinear) else ('_parallel_stage_decoder.', module._parallel_stage_decoder)
        result.update({name + '.' + prefix + key: value for key, value in decoder.named_parameters()})
    b.require(len(result) == len({id(value) for value in result.values()}), 'Duplicate decoder ownership.')
    return result


def copy_parameters(modules):
    return {name: value.detach().cpu().clone() for name, value in decoder_parameters(modules).items()}


def frozen_proof(model, active_names):
    entries = {name: b.core.tensor_record(value) for name, value in model.state_dict().items() if name not in active_names}
    return dict(tensors=len(entries), sha256=b.core.mapping_digest(entries))


def training_mode(model, modules):
    model.eval().requires_grad_(False)
    for module in modules.values():
        if isinstance(module, b.CompactRVQLinear):
            module.enable_decoder_training()
        else:
            module.packed_vq_decoder_linear = True
            module._parallel_stage_decoder.use_checkpoint = True
            plan = b.core.enable_vae_linear_by_execution_plan(module, mode='trainable_decoder')
            b.require(plan.use_packed, 'Native decoder lost its original packed path.')
            for parameter in module._parallel_stage_decoder.parameters():
                parameter.data = parameter.data.float()
        module.train()
    install_precision_runtime(model, b.torch.bfloat16)
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    b.require(set(active) == set(decoder_parameters(modules)) and all(p.dtype == b.torch.float32 for p in active.values()),
              'Only all actual FP32 decoder masters may train.')
    verify_live(modules)
    return active


def verify_live(modules):
    for name, module in modules.items():
        if isinstance(module, b.CompactRVQLinear):
            b.require(module._decoded_cache is None and module._cache_versions is None
                and all(decoder.use_checkpoint for decoder in module.decoders), 'Compact live cache/checkpoint mismatch: '+name)
        else:
            b.require(module.trainable_decode and not module.cache_decoded_weight and module._cached_weight is None
                and module._decoder_compute_dtype == b.torch.bfloat16 and module._parallel_stage_decoder.use_checkpoint,
                'Native live decoding contract changed: '+name)


def clear(modules):
    for module in modules.values():
        module.clear_decoded_weight_cache()


@b.torch.no_grad()
def prewarm(modules):
    native = [(name, module) for name, module in modules.items() if not isinstance(module, b.CompactRVQLinear)]
    compact = [(name, module) for name, module in modules.items() if isinstance(module, b.CompactRVQLinear)]
    warmed = prime_named_vae_linear_cache(native, dtype=b.torch.bfloat16, clear_existing=True, group_size=8)
    b.require(warmed == dict(total=len(native), warmed=len(native), skipped=0, failed=0), 'Native group8 prewarm failed.')
    for name, module in native:
        b.require(module._cached_weight is not None and module._cached_weight.dtype == b.torch.bfloat16
            and not b.torch.is_inference(module._cached_weight), 'Native BF16 normal cache missing: '+name)
    for name, module in compact:
        module.clear_decoded_weight_cache()
        module(b.torch.empty((0, module.in_features), device=module.length_frame.device, dtype=b.torch.bfloat16))
        b.require(module._decoded_cache is not None and module._decoded_cache.dtype == b.torch.bfloat16
            and not b.torch.is_inference(module._decoded_cache), 'Compact BF16 normal cache missing: '+name)
    return dict(native=warmed, native_group_size=8, compact_warmed=len(compact))


def leaves(active):
    return {name: dict(identity=id(p), data_ptr=p.data_ptr(), version=p._version, value=b.core.tensor_record(p),
        requires_grad=p.requires_grad, grad_identity=None if p.grad is None else id(p.grad)) for name,p in active.items()}


def optimizer_stamp(optimizer, active):
    return dict(identity=id(optimizer), parameter_refs=[[id(p) for p in group['params']] for group in optimizer.param_groups],
        metadata=opt.metadata(optimizer), state={name:dict(identity=id(state), values={key:dict(identity=id(value),
            data_ptr=value.data_ptr(), value=b.core.tensor_record(value)) if isinstance(value,b.torch.Tensor) else value
            for key,value in state.items()}) for name,p in active.items() if (state := optimizer.state.get(p, {}))})


@contextmanager
def rollout_mode(model, modules, active, optimizer, evidence, expected_elements):
    b.require(sum(p.numel() for p in active.values()) == expected_elements
        and all(p.dtype == b.torch.float32 and p.requires_grad and p.grad is None for p in active.values())
        and getattr(model, '_cache', None) is None, 'Rollout must start before accumulation with exact FP32 masters.')
    verify_live(modules)
    before, state_before = leaves(active), optimizer_stamp(optimizer, active)
    masters = {name:p.data for name,p in active.items()}
    modes = {child:child.training for child in model.modules()}
    cache_setting = model.config.use_cache
    initialized = len(state_before['state'])
    b.require(initialized in (0, len(active)), 'Optimizer must be fresh or completely initialized.')
    try:
        model.eval()
        model.config.use_cache = True
        with b.torch.no_grad():
            for name,p in active.items():
                p.data = masters[name].to(b.torch.bfloat16)
            for module in modules.values():
                if not isinstance(module, b.CompactRVQLinear):
                    module.disable_trainable_decode()
            evidence['prewarm'] = prewarm(modules)
        yield
    finally:
        clear(modules)
        for name,p in active.items():
            p.data = masters[name]
        for module in modules.values():
            if isinstance(module, b.CompactRVQLinear):
                module.enable_decoder_training()
            else:
                module.enable_trainable_decode(parallel_stage_decode=True)
                module._parallel_stage_decoder.use_checkpoint = True
        for child, training in modes.items():
            child.training = training
        model.config.use_cache = cache_setting
        verify_live(modules)
        b.require(getattr(model, '_cache', None) is None and leaves(active) == before
            and optimizer_stamp(optimizer, active) == state_before,
            'Native/compact generation changed FP32 master, Parameter, optimizer state, kappa or LR.')
        evidence.update(master_and_optimizer_exact=True, parameter_count=len(active), optimizer_initialized_leaves=initialized,
            Muon_momentum_initialized_leaves=sum('momentum_buffer' in value['values'] for value in state_before['state'].values()),
            auxiliary_Adam_initialized_leaves=sum('exp_avg' in value['values'] for value in state_before['state'].values()),
            Muon_group_metadata_exact=True, Muon_fixed_kappa=optimizer.kappa,
            group_LRs={group['group_name']:group['lr'] for group in optimizer.param_groups},
            master_before_sha256=b.core.mapping_digest(before), master_after_sha256=b.core.mapping_digest(leaves(active)),
            optimizer_before_sha256=b.core.mapping_digest(state_before), optimizer_after_sha256=b.core.mapping_digest(optimizer_stamp(optimizer,active)),
            decoded_and_KV_cache_cleared=True)


@b.torch.no_grad()
def deployment_mode(model, modules):
    model.eval().requires_grad_(False)
    model.gradient_checkpointing_disable()
    if hasattr(model, '_require_grads_hook'):
        model.disable_input_require_grads()
        delattr(model, '_require_grads_hook')
    for module in modules.values():
        if not isinstance(module,b.CompactRVQLinear):
            module.disable_trainable_decode()
    for parameter in decoder_parameters(modules).values():
        parameter.grad = None
        parameter.data = parameter.data.bfloat16()
    install_precision_runtime(model,b.torch.bfloat16)
    b.checkpoint.native.refresh_vae_linear_runtime_after_state_load(model)
    clear(modules)
    result = prewarm(modules)
    b.require({p.dtype for p in model.parameters()} == {b.torch.bfloat16}, 'Deployment parameters must all be BF16.')
    return result
