"""All-code STE adapter with fixed decoder-sensitivity proxy coordinates.

No edits or global patches to VAELLM; forward and saved payload remain hard bits.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
from litebsq.packed_bit_linear import _packed_u8_linear_forward, resolve_parallel_linear_weight_bias
from litebsq.vae_linear import VAELinear, apply_activation
from sparse_bit_tuning.packed_ops import _compute_decoder_weight_bias_grads
from experiments.liftquant_recovery.proxy_coordinates import hard_bits, measure_scale, ste_gate


class _AllBitsLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, weight, bias, scores, activation_dtype, proxy_scale):
        out, packed, *_ = _packed_u8_linear_forward(
            packed, weight, bias, logical_in_dim=scores.shape[-1], activation_dtype=activation_dtype)
        ctx.save_for_backward(packed, weight, scores)
        ctx.forward_weight_dtype = activation_dtype
        ctx.proxy_scale = proxy_scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        packed, weight, scores = ctx.saved_tensors
        dw = db = ds = None
        if ctx.needs_input_grad[1]:
            dw, computed_db = _compute_decoder_weight_bias_grads(
                packed, grad_out, logical_in_dim=scores.shape[-1], weight_dtype=weight.dtype)
            if ctx.needs_input_grad[2]:
                db = computed_db
        elif ctx.needs_input_grad[2]:
            # Preserve the shared packed backward's FP32 reduction for bias-only use.
            db = grad_out.contiguous().sum(dim=0, dtype=torch.float32).to(dtype=weight.dtype)
        if ctx.needs_input_grad[3]:
            ds = torch.empty_like(scores, dtype=torch.float32)
            with torch.autocast('cuda', enabled=False):
                w = weight.to(ctx.forward_weight_dtype).float()
                for start in range(0, len(ds), 16384):
                    g = grad_out[start:start+16384].float().transpose(0, 1)
                    gradient = torch.bmm(g, w).transpose(0, 1) / ctx.proxy_scale
                    ds[start:start+16384] = gradient * ste_gate(scores[start:start+16384], ctx.proxy_scale)
        return None, dw, db, ds, None, None


def _decoder_tail(decoder, h):
    if decoder.decoder_type == 'linear':
        return h
    for block in decoder.blocks:
        h = block(h)
    return decoder.linear_out(apply_activation(decoder.norm_out(h), str(decoder.activation_type)))


@torch.no_grad()
def decode_sample(decoder, packed, dtype=torch.bfloat16):
    """Exact packed arithmetic used to measure existing single-bit effects."""
    linear = decoder.linear if decoder.decoder_type == 'linear' else decoder.linear_in
    weight, bias = resolve_parallel_linear_weight_bias(linear)
    with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
        h, *_ = _packed_u8_linear_forward(packed, weight, bias,
                                          logical_in_dim=decoder.in_dim, activation_dtype=dtype)
        return _decoder_tail(decoder, h)


def decode_with_proxy(decoder, packed, scores, scale, dtype):
    def decode(codes):
        linear = decoder.linear if decoder.decoder_type == 'linear' else decoder.linear_in
        weight, bias = resolve_parallel_linear_weight_bias(linear)
        h = _AllBitsLinear.apply(codes, weight, bias, scores, dtype, scale)
        return _decoder_tail(decoder, h)
    if torch.is_grad_enabled():
        return checkpoint(decode, packed, use_reentrant=False)
    return decode(packed)


class RecoveryLinear(VAELinear):
    def _decode_packed_u8_with_decoder(self, decoder, packed_vq, *, logical_shape, activation_dtype, **kwargs):
        runtime = self._recovery_runtime
        if tuple(logical_shape) != tuple(runtime.scores.shape):
            raise ValueError('Expected the original single-stage/part code geometry.')
        dtype = getattr(self, '_decoder_compute_dtype', None) or activation_dtype
        return decode_with_proxy(decoder, packed_vq, runtime.scores, runtime.proxy_scale, dtype)


def decoder_for(module):
    decoder = getattr(module, '_parallel_stage_decoder', None)
    return decoder if decoder is not None else module.get_stage_part_decoder(stage_idx=0, part_idx=0)


def attach(module):
    if type(module) is not VAELinear or module.residual_stages != 1 or module.parallel_parts != 1:
        raise ValueError('Recovery requires native VAELinear, one stage and one part.')
    if getattr(module, '_sparse_bit_binding', None) is not None:
        raise ValueError('Refusing an active sparse-bit runtime.')
    shape = tuple(module.get_stage_part_vq_spec(stage_idx=0, part_idx=0)['logical_shape'])
    if len(shape) != 3 or shape[1] != 1:
        raise ValueError(f'Unsupported code geometry: {shape}')
    packed = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
    decoder = decoder_for(module)
    if decoder.decoder_type not in ('linear', 'symmetric', 'asymmetric'):
        raise ValueError(f'Unsupported decoder: {decoder.decoder_type}')
    dtype = getattr(module, '_decoder_compute_dtype', None) or torch.bfloat16
    geometry = measure_scale(packed, shape, lambda value: decode_sample(decoder, value, dtype))
    scale = geometry['scale']
    scores = nn.Parameter(torch.empty(shape, device=packed.device, dtype=torch.float32))
    with torch.no_grad():
        for start in range(0, len(scores), 65536):
            chunk_shape = (min(65536, len(scores)-start), *shape[1:])
            bits = unpack_uint8_tensor_to_bool(packed[start:start+65536], logical_shape=chunk_shape)
            scores[start:start+65536].copy_((bits.float()-0.5)*scale)
    module._recovery_runtime = SimpleNamespace(
        scores=scores, shape=shape, proxy_scale=scale, geometry=geometry,
        initial_packed=packed.detach().clone(), old_parallel=module.parallel_stage_decode,
        old_checkpoint=decoder.use_checkpoint)
    module.enable_trainable_decode(parallel_stage_decode=False)
    module.parallel_stage_decode = module._recovery_runtime.old_parallel
    decoder.requires_grad_(True)
    module.__class__ = RecoveryLinear
    # A coordinate change must leave every packed bit unchanged before updates.
    if project(module)['changed_packed_bits'] != 0:
        raise ValueError('Proxy initialization changed the source hard codes.')
    return scores, list(decoder.parameters())


def _python_statistics(values, integer_keys=()):
    """Transfer a small statistics vector once, retaining exact integer counts.

    FP64 also preserves the previous Python-double division of FP32 extrema.
    The counts here are bounded by the resident proxy size, far below 2**53.
    """
    names = list(values)
    scalars = torch.stack([values[name].to(torch.float64) for name in names]).cpu().tolist()
    return {name: int(value) if name in integer_keys else value
            for name, value in zip(names, scalars)}


@torch.no_grad()
def project(module, *, device=False, collect_stats=True):
    """Project every code each call; statistics never synchronize inside chunks.

    device=True returns zero-dimensional device tensors. collect_stats=False
    skips only counting, not projection, cache invalidation or grouped copying.
    """
    runtime = module._recovery_runtime
    storage = module.get_stage_part_vq_storage(stage_idx=0, part_idx=0)
    byte_counts, bit_counts = [], []
    if collect_stats:
        popcount = getattr(runtime, 'popcount', None)
        if popcount is None or popcount.device != storage.device:
            popcount = torch.tensor([i.bit_count() for i in range(256)],
                                    device=storage.device, dtype=torch.int16)
            runtime.popcount = popcount
    for start in range(0, len(runtime.scores), 65536):
        scores = runtime.scores[start:start+65536]
        packed = pack_bool_tensor_to_uint8(hard_bits(scores, runtime.proxy_scale), logical_shape=tuple(scores.shape))
        original = storage[start:start+65536]
        if collect_stats:
            byte_counts.append((original != packed).sum())
            bit_counts.append(popcount[(original ^ packed).long()].sum())
        original.copy_(packed)
    module.clear_decoded_weight_cache()
    if getattr(module, '_parallel_stage_grouped_vq_packed', None) is not None:
        module._parallel_stage_grouped_vq_packed.copy_(storage)
    if not collect_stats:
        return {}
    counts = dict(changed_packed_bytes=torch.stack(byte_counts).sum(),
                  changed_packed_bits=torch.stack(bit_counts).sum())
    return counts if device else _python_statistics(counts, integer_keys=counts)


@torch.no_grad()
def proxy_statistics(module, *, device=False):
    """Reduce unchanged audit formulas on-device; copy only the final scalars."""
    runtime = module._recovery_runtime
    motions, margins, saturated_counts, changed_counts = [], [], [], []
    for start in range(0, len(runtime.scores), 65536):
        scores = runtime.scores[start:start+65536]
        bits = unpack_uint8_tensor_to_bool(runtime.initial_packed[start:start+65536], logical_shape=tuple(scores.shape))
        motions.append((scores-(bits.float()-0.5)*runtime.proxy_scale).abs().max())
        margins.append(scores.abs().min())
        saturated_counts.append((~ste_gate(scores, runtime.proxy_scale)).sum())
        changed_counts.append((hard_bits(scores, runtime.proxy_scale) != bits).sum())
    max_motion = torch.stack(motions).max()
    values = dict(proxy_delta_max=max_motion, minimum_margin=torch.stack(margins).min(),
                  saturated_count=torch.stack(saturated_counts).sum(),
                  bits_changed_from_initial=torch.stack(changed_counts).sum())
    if device:
        values['saturated_fraction'] = values.pop('saturated_count').to(torch.float64)/runtime.scores.numel()
        values['proxy_delta_over_initial_margin'] = max_motion.to(torch.float64)/(runtime.proxy_scale/2)
    else:
        values = _python_statistics(values, integer_keys=('saturated_count', 'bits_changed_from_initial'))
        # Keep exactly the original Python division for the default reporting API.
        values['saturated_fraction'] = values.pop('saturated_count')/runtime.scores.numel()
        values['proxy_delta_over_initial_margin'] = values['proxy_delta_max']/(runtime.proxy_scale/2)
    return values


def detach(module):
    project(module)
    runtime = module._recovery_runtime
    decoder_for(module).use_checkpoint = runtime.old_checkpoint
    module.__class__ = VAELinear
    del module._recovery_runtime
    module.disable_trainable_decode()
    module.requires_grad_(False)
    module.clear_decoded_weight_cache()
