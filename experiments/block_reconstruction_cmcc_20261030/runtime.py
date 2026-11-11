"""Canonical full-group prewarm and block-local decoder ownership on one GPU."""
import time

from . import bindings as b
from train_utils.decoder_execution import enable_vae_linear_by_execution_plan

torch = b.torch


def modules(model):
    result = {name: module for name, module in model.named_modules()
        if isinstance(module, (b.VAELinear, b.CompactRVQLinear))}
    b.require(len(result) == 252, 'Keep all252 actual compressed projections.')
    return result


def block_modules(model, index):
    b.require(0 <= index < 36, 'The block index must be in the actual Qwen3 stack.')
    prefix = f'model.layers.{index}.'
    result = {name: module for name, module in modules(model).items() if name.startswith(prefix)}
    b.require(len(result) == 7 and {name.rsplit('.', 1)[1] for name in result} == set(b.CATEGORIES),
        'The complete Transformer block must expose its seven actual projections.')
    return result


def remember_deployment_flags(model):
    flags = {}
    for name, module in modules(model).items():
        if isinstance(module, b.VAELinear):
            flags[name] = dict(consumer='native', use_checkpoint=module._parallel_stage_decoder.use_checkpoint,
                packed_vq_decoder_linear=module.packed_vq_decoder_linear)
        else:
            flags[name] = dict(consumer='compact', use_checkpoint=[decoder.use_checkpoint for decoder in module.decoders])
    model._br_deployment_flags = flags
    return flags


def _restore_deployment(module, flags):
    b.require(all(p.dtype == torch.bfloat16 and not p.requires_grad for p in module.parameters()),
        'Commit BF16 parameters and freeze them before deployment prewarm.')
    module.eval()
    if isinstance(module, b.VAELinear):
        module.disable_trainable_decode()
        module._parallel_stage_decoder.use_checkpoint = flags['use_checkpoint']
        module.packed_vq_decoder_linear = flags['packed_vq_decoder_linear']
    else:
        for decoder, value in zip(module.decoders, flags['use_checkpoint'], strict=True):
            decoder.use_checkpoint = value
        module.clear_decoded_weight_cache()


@torch.no_grad()
def _warm_native(native, device):
    result = b.prime_named_vae_linear_cache(native, dtype=torch.bfloat16, clear_existing=True,
        group_size=8, compute_device=device)
    b.require(result == dict(total=len(native), warmed=len(native), skipped=0, failed=0),
        'The complete actual native group8 set did not prewarm.')
    for name, module in native:
        value = module._cached_weight
        b.require(value is not None and value.dtype == torch.bfloat16 and not value.requires_grad
            and not torch.is_inference(value), 'Use ordinary no_grad BF16 deployment caches: '+name)
    return result


@torch.no_grad()
def _warm_compact(module, device):
    module.clear_decoded_weight_cache()
    module(torch.empty((0, module.in_features), dtype=torch.bfloat16, device=device))
    b.require(module._decoded_cache is not None and module._decoded_cache.dtype == torch.bfloat16
        and not torch.is_inference(module._decoded_cache), 'The compact deployment cache is missing or has the wrong tensor mode.')


@torch.no_grad()
def prepare_deployment_block(model, index, device):
    started = time.monotonic()
    device = torch.device(device)
    b.require(device.type == 'cuda', 'Measure the real GPU group8 path.')
    model.eval().requires_grad_(False)
    all_modules = modules(model)
    selected = block_modules(model, index)
    flags = model._br_deployment_flags
    native = []
    for name, module in all_modules.items():
        _restore_deployment(module, flags[name])
        if isinstance(module, b.VAELinear):
            module.to(device)
            native.append((name, module))
    model.model.layers[index].to(device)
    b.require(len(native) == 216, 'The real mixed parent has216 native projections; do not substitute a partial prewarm set.')
    warmed = _warm_native(native, device)
    for name, module in native:
        if name not in selected:
            module.clear_decoded_weight_cache()
            module.cpu()
    for module in selected.values():
        if isinstance(module, b.CompactRVQLinear):
            _warm_compact(module, device)
    torch.cuda.synchronize(device)
    return dict(layer=index, native=warmed, native_group_size=8, current_compact=1,
        strategy='Full same-device native set in original order, then retain only this block cache.',
        seconds=time.monotonic()-started, allocated_bytes=torch.cuda.memory_allocated(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def decoder_parameters(model, index):
    result = {}
    for name, module in block_modules(model, index).items():
        prefix, decoder = ('decoders.', module.decoders) if isinstance(module, b.CompactRVQLinear) else (
            '_parallel_stage_decoder.', module._parallel_stage_decoder)
        result.update({name+'.'+prefix+key: p for key, p in decoder.named_parameters()})
    b.require(len(result) == len({id(p) for p in result.values()}), 'Decoder parameters have duplicate ownership.')
    return result


def enable_block_training(model, index):
    model.eval().requires_grad_(False)
    block = model.model.layers[index]
    for name, module in block_modules(model, index).items():
        b.original.validate_supported(module, name)
        if isinstance(module, b.CompactRVQLinear):
            module.enable_decoder_training()
        else:
            module.packed_vq_decoder_linear = True
            plan = enable_vae_linear_by_execution_plan(module, mode='trainable_decoder')
            b.require(plan.use_packed, 'Native block training must retain its actual packed decoder path.')
            module._parallel_stage_decoder.use_checkpoint = True
            for p in module._parallel_stage_decoder.parameters():
                p.data = p.data.float()
    b.install_precision_runtime(block, torch.bfloat16)
    block.train()
    active = decoder_parameters(model, index)
    actual = {name: p for name, p in model.named_parameters() if p.requires_grad}
    b.require(set(active) == set(actual) and all(actual[name] is p and p.dtype == torch.float32
        for name, p in active.items()), 'Only the current complete block decoder FP32 masters may train.')
    return active


@torch.no_grad()
def commit_block(model, index):
    active = decoder_parameters(model, index)
    for p in active.values():
        b.require(p.dtype == torch.float32 and torch.isfinite(p).all().item(), 'Commit finite actual FP32 decoder masters.')
        p.grad = None
        p.data = p.data.to(torch.bfloat16)
        p.requires_grad_(False)
    model.model.layers[index].eval()
    for name, module in block_modules(model, index).items():
        _restore_deployment(module, model._br_deployment_flags[name])
    return dict(layer=index, committed_parameters=len(active), deployment_dtype='torch.bfloat16')


@torch.no_grad()
def offload_block(model, index):
    for module in block_modules(model, index).values():
        module.clear_decoded_weight_cache()
    model.model.layers[index].cpu()


@torch.no_grad()
def prepare_full_deployment(model, device):
    model.eval().requires_grad_(False)
    b.require(all(p.dtype == torch.bfloat16 for p in model.parameters()), 'Every student deployment Parameter must remain BF16.')
    _move_model_preserving_rope(model, device)
    if not hasattr(model, '_br_deployment_flags'):
        remember_deployment_flags(model)
    b.install_precision_runtime(model, torch.bfloat16)
    native, compact = [], []
    for name, module in modules(model).items():
        _restore_deployment(module, model._br_deployment_flags[name])
        if isinstance(module, b.VAELinear):
            native.append((name, module))
        else:
            compact.append(module)
    warmed = _warm_native(native, device)
    for module in compact:
        _warm_compact(module, device)
    return dict(native=warmed, native_group_size=8, compact=len(compact))


def _move_model_preserving_rope(model, device):
    rope = model.model.rotary_emb
    b.require(rope.inv_freq.dtype == rope.original_inv_freq.dtype == torch.float32,
        'Keep original RoPE precision during device-only movement.')
    aliased = rope.original_inv_freq is rope.inv_freq
    original_frequency = rope.original_inv_freq
    model.to(device=device)
    rope.original_inv_freq = rope.inv_freq if aliased else original_frequency.to(device=device)
    b.require(rope.inv_freq.dtype == rope.original_inv_freq.dtype == torch.float32
        and (rope.original_inv_freq is rope.inv_freq) == aliased,
        'Device movement changed native RoPE precision or its original alias relationship.')


@torch.no_grad()
def offload_full_deployment(model):
    for module in modules(model).values():
        module.clear_decoded_weight_cache()
    _move_model_preserving_rope(model, 'cpu')
