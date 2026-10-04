"""Add an identity residual block to the existing packed R0 decoder.

This changes the native decoder graph, not its initial mathematical function.
R1 cannot use the R0 whole-decoder fused kernel; deployment parity is a separate
GPU validation, not a property asserted by this CPU-compatible transformation.
"""

from __future__ import annotations

import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.parallel_layers import ResnetBlock1D
from litebsq.vae_linear import VAELinear


def validate_packed_r0_decoder(module: VAELinear) -> Decoder:
    """Check the authorized two-stage H128 profile without changing it."""
    if not isinstance(module, VAELinear):
        raise TypeError("Expansion requires a VAELinear.")
    decoder = getattr(module, "_parallel_stage_decoder", None)
    if (not isinstance(decoder, Decoder) or not module.parallel_stage_decode
            or module.residual_stages != 2 or module.parallel_parts != 1
            or decoder.num_models != 2):
        raise ValueError("Expansion requires an existing packed two-stage, single-part decoder.")
    if (decoder.decoder_type != "symmetric" or decoder.hidden_dim != 128
            or decoder.norm_type != "layer" or decoder.activation_type != "swish"):
        raise ValueError("Expansion requires the existing H128 symmetric LayerNorm/SiLU decoder.")
    if decoder.num_res_blocks != 0 or len(decoder.blocks) != 0:
        raise ValueError("Expansion requires R0; an existing residual block must not be replaced.")
    parameters = list(decoder.parameters())
    profiles = {(p.device, p.dtype, p.requires_grad) for p in parameters}
    if len(profiles) != 1 or parameters[0].dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Decoder parameters must share device, FP32/BF16 dtype and trainability.")
    return decoder


@torch.no_grad()
def expand_packed_decoder_r0_to_r1_(module: VAELinear) -> dict:
    """Keep the original decoder/parameters and append one zero-output block."""
    decoder = validate_packed_r0_decoder(module)
    reference = next(decoder.parameters())
    # The constructor initializes a Linear randomly before we zero it. Migration
    # must not advance the caller's random stream for later training or probes.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        block = ResnetBlock1D(
            in_channels=128, out_channels=128, norm_type="layer",
            activation_type="swish", num_models=2,
        )
    block.to(device=reference.device, dtype=reference.dtype)
    block.norm1.weight.fill_(1)
    block.norm1.bias.zero_()
    block.linear1.conv.weight.zero_()
    block.linear1.conv.bias.zero_()
    block.train(decoder.training).requires_grad_(reference.requires_grad)
    decoder.blocks.append(block)
    decoder.num_res_blocks = 1
    module.clear_decoded_weight_cache()
    state = block.state_dict()
    return {
        "old_num_res_blocks": 0, "new_num_res_blocks": 1,
        "added_tensor_keys": ["_parallel_stage_decoder.blocks.0." + key for key in state],
        "added_parameters": sum(p.numel() for p in block.parameters()),
        "added_tensor_bytes": sum(t.numel() * t.element_size() for t in state.values()),
        "parameter_dtype": str(reference.dtype),
        "wholefused_supported": False,
    }
