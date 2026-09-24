"""Full-Linear output loss and memory chunking without changing BSQ entropy."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunks(fn, x: torch.Tensor, chunk_size: int, *, recompute: bool) -> torch.Tensor:
    values = []
    for chunk in x.split(chunk_size):
        values.append(checkpoint(fn, chunk, use_reentrant=False) if recompute else fn(chunk))
    return torch.cat(values, dim=0)


def full_vae_forward(vae, blocks: torch.Tensor, *, chunk_vectors: int, recompute: bool = True):
    """Exactly one production BSQ call over all vectors (including batch entropy)."""
    auto = vae.model
    if blocks.ndim != 3 or blocks.shape[1] != 1:
        raise ValueError("Expected row-major blocks [vectors, 1, 32].")
    if auto.arch_spec.norm_type == "batch":
        raise ValueError("BatchNorm would change its statistics when memory chunking.")

    def encode(x):
        with auto._autocast_context(x):
            return auto.encoder(x)

    def decode(z):
        with auto._autocast_context(z):
            return auto.decoder(z.to(auto.params_dtype))

    h = _chunks(encode, blocks, chunk_vectors, recompute=recompute)
    with auto._autocast_context(h):
        quant = auto._parse_quantizer_output(auto.quantizer(h), h.device)
    decoded = _chunks(decode, quant.z, chunk_vectors, recompute=recompute)
    auxiliary = quant.aux_loss * auto.lfq_weight * auto.num_models
    return decoded, auxiliary, quant.bit_indices


def linear_output_mse(error_weight: torch.Tensor, inputs: torch.Tensor, *, chunk_size: int,
                      recompute: bool = True) -> torch.Tensor:
    """mean((X @ (W_hat-W).T)**2), with ALL input-channel cross terms.

    Bias is unchanged, so it cancels exactly. Token chunks are summed with their
    true sizes; neither token sampling nor block-diagonal Gram approximation.
    """
    if error_weight.ndim != 2 or inputs.ndim != 2 or inputs.shape[1] != error_weight.shape[1]:
        raise ValueError("Expected error_weight[out,in] and inputs[valid_tokens,in].")
    if not len(inputs) or chunk_size < 1:
        raise ValueError("Need valid tokens and a positive chunk size.")

    def squared_sum(delta, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), delta.float()).square().sum()

    total = error_weight.new_zeros((), dtype=torch.float32)
    for chunk in inputs.split(chunk_size):
        term = (checkpoint(squared_sum, error_weight, chunk, use_reentrant=False)
                if recompute else squared_sum(error_weight, chunk))
        total = total + term
    return total / (len(inputs) * error_weight.shape[0])
