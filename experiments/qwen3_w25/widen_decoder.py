"""Isolated H128-to-H256 initialization for the existing asymmetric decoder.

Uniform duplication preserves LayerNorm statistics in exact arithmetic. Keeping
the original output columns followed by zeros avoids tying the two copies during
subsequent optimization. Native finite-precision parity is measured separately.
"""

from __future__ import annotations

import torch

from experiments.qwen3_w25.expand_decoder import validate_packed_r0_decoder
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear


@torch.no_grad()
def widen_packed_decoder_h128_to_h256_(module: VAELinear) -> dict:
    old = validate_packed_r0_decoder(module)
    reference = next(old.parameters())
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        new = Decoder(
            in_dim=old.in_dim, out_dim=old.out_dim, hidden_dim=256,
            num_res_blocks=0, norm_type=old.norm_type,
            activation_type=old.activation_type, decoder_type="asymmetric",
            use_checkpoint=old.use_checkpoint, num_models=old.num_models,
        )
    new.to(device=reference.device, dtype=reference.dtype)
    stages = old.num_models
    # Packed output-channel dimensions concatenate stages; duplicate within each
    # stage so stage 0 and stage 1 never exchange weights or norm parameters.
    incoming = old.linear_in.conv.weight.reshape(stages, 128, old.in_dim, 1)
    new.linear_in.conv.weight.copy_(torch.cat((incoming, incoming), dim=1).reshape_as(new.linear_in.conv.weight))
    bias = old.linear_in.conv.bias.reshape(stages, 128)
    new.linear_in.conv.bias.copy_(torch.cat((bias, bias), dim=1).reshape_as(new.linear_in.conv.bias))
    for name in ("weight", "bias"):
        value = getattr(old.norm_out, name)
        getattr(new.norm_out, name).copy_(torch.cat((value, value), dim=-1))
    outgoing = old.linear_out.conv.weight.reshape(stages, old.out_dim, 128, 1)
    expanded = new.linear_out.conv.weight.reshape(stages, old.out_dim, 256, 1)
    expanded.zero_()
    expanded[:, :, :128].copy_(outgoing)
    new.linear_out.conv.bias.copy_(old.linear_out.conv.bias)
    new._q_scale_fused = old._q_scale_fused
    new.train(old.training).requires_grad_(reference.requires_grad)
    old_count = sum(value.numel() for value in old.parameters())
    new_count = sum(value.numel() for value in new.parameters())
    module._parallel_stage_decoder = new
    module.clear_decoded_weight_cache()
    return {
        "old_hidden_dim": 128, "new_hidden_dim": 256,
        "old_decoder_type": old.decoder_type, "new_decoder_type": new.decoder_type,
        "added_parameters": new_count - old_count,
        "added_tensor_bytes": (new_count - old_count) * reference.element_size(),
        "output_initialization": "original columns followed by zeros per stage",
        "parameter_dtype": str(reference.dtype), "wholefused_supported": False,
    }
