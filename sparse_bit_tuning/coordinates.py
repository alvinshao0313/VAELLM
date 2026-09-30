"""Calibrate training-only coordinates without mutating model state or RNG."""
from copy import deepcopy

import torch

from litebsq.bit_sensitivity import measure_scale
from litebsq.packed_bit_linear import _packed_u8_linear_forward, resolve_parallel_linear_weight_bias
from litebsq.vae_linear import apply_activation


@torch.no_grad()
def calibrate_bank_scale(module, spec, *, compute_dtype=None):
    device = torch.device(spec.device)
    # Grouped decoder extraction can construct fresh modules and consume RNG.
    # A temporary bank copy also preserves streaming/offload residency and hooks.
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
        decoder = deepcopy(module.get_stage_part_decoder(spec.stage_idx, spec.part_idx))
        decoder = decoder.to(device=device).eval().requires_grad_(False)
        decoder.use_checkpoint = False
        packed = module.get_stage_part_vq_storage(spec.stage_idx, spec.part_idx).detach().to(device)
        dtype = getattr(module, "_decoder_compute_dtype", None) or compute_dtype
        if dtype is None:
            # Direct manager callers with mixed parameter/input dtypes must pass
            # calibration_dtype; the E2E entry derives it from AMP/backbone input.
            dtype = next(decoder.parameters()).dtype
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"Unsupported decoder calibration dtype: {dtype}.")
        if decoder.decoder_type not in {"linear", "symmetric", "asymmetric"}:
            raise ValueError(f"Unsupported decoder for bit sensitivity: {decoder.decoder_type}.")

        def decode(codes):
            first = decoder.linear if decoder.decoder_type == "linear" else decoder.linear_in
            weight, bias = resolve_parallel_linear_weight_bias(first)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                h, *_ = _packed_u8_linear_forward(
                    codes, weight, bias, logical_in_dim=decoder.in_dim, activation_dtype=dtype,
                )
                if decoder.decoder_type == "linear":
                    return h
                for block in decoder.blocks:
                    h = block(h)
                return decoder.linear_out(apply_activation(decoder.norm_out(h), str(decoder.activation_type)))

        return measure_scale(packed, spec.logical_shape, decode)["scale"]
