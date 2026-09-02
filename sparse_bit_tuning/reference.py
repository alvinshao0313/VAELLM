from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .sampler import AffineSamplerState


def unpack_packed_bits(packed: torch.Tensor, *, logical_in_dim: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be uint8, got {packed.dtype}.")
    if packed.ndim != 3:
        raise ValueError(f"packed must be [B,M,P], got {tuple(packed.shape)}.")
    logical_in = int(logical_in_dim)
    if logical_in < 1:
        raise ValueError(f"logical_in_dim must be >=1, got {logical_in}.")
    expected = (logical_in + 7) // 8
    if int(packed.shape[-1]) != expected:
        raise ValueError(f"packed width {packed.shape[-1]} != ceil({logical_in}/8)={expected}.")
    shifts = torch.arange(logical_in, device=packed.device, dtype=torch.long)
    bytes_ = shifts // 8
    bits = shifts % 8
    values = (packed[..., bytes_] >> bits) & 1
    return values.to(torch.bool)


def pack_bool_bits(bits: torch.Tensor) -> torch.Tensor:
    if bits.ndim != 3:
        raise ValueError(f"bits must be [B,M,IN], got {tuple(bits.shape)}.")
    logical_in = int(bits.shape[-1])
    packed_width = (logical_in + 7) // 8
    out = torch.zeros((*bits.shape[:-1], packed_width), dtype=torch.uint8, device=bits.device)
    src = bits.to(torch.uint8)
    for latent_idx in range(logical_in):
        byte_idx = latent_idx // 8
        bit_offset = latent_idx % 8
        out[..., byte_idx] |= src[..., latent_idx] << bit_offset
    return out


def read_bank_hard_bits(
    packed: torch.Tensor,
    *,
    state: AffineSamplerState,
    model_idx: int,
    logical_in_dim: int,
) -> torch.Tensor:
    dense = unpack_packed_bits(packed, logical_in_dim=logical_in_dim)
    values = []
    for logical_idx in state.active_indices():
        block_idx = int(logical_idx) // int(logical_in_dim)
        latent_idx = int(logical_idx) % int(logical_in_dim)
        values.append(dense[block_idx, int(model_idx), latent_idx])
    return torch.stack(values).to(torch.bool)


def scores_from_hard_bits(hard_bits: torch.Tensor) -> torch.Tensor:
    return torch.where(
        hard_bits.to(torch.bool),
        torch.ones_like(hard_bits, dtype=torch.float16),
        -torch.ones_like(hard_bits, dtype=torch.float16),
    )


def apply_bank_scores_reference(
    packed: torch.Tensor,
    *,
    state: AffineSamplerState,
    scores: torch.Tensor,
    model_idx: int,
    logical_in_dim: int,
) -> tuple[torch.Tensor, int]:
    if int(scores.numel()) != int(state.n_active):
        raise ValueError(f"score numel {scores.numel()} != n_active {state.n_active}.")
    dense = unpack_packed_bits(packed, logical_in_dim=logical_in_dim).clone()
    flips = 0
    score_flat = scores.reshape(-1)
    for q, logical_idx in enumerate(state.active_indices()):
        block_idx = int(logical_idx) // int(logical_in_dim)
        latent_idx = int(logical_idx) % int(logical_in_dim)
        old = bool(dense[block_idx, int(model_idx), latent_idx].item())
        new = bool(score_flat[q].item() >= 0.0)
        flips += int(old != new)
        dense[block_idx, int(model_idx), latent_idx] = new
    return pack_bool_bits(dense), int(flips)


def dense_first_linear_reference(
    packed: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    logical_in_dim: int,
    activation_dtype: torch.dtype,
) -> torch.Tensor:
    bits = unpack_packed_bits(packed, logical_in_dim=logical_in_dim).to(dtype=activation_dtype)
    return torch.einsum("bmi,mhi->bmh", bits, weight.to(dtype=activation_dtype)) + bias.to(dtype=activation_dtype)


def dense_active_score_grad_reference(
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    *,
    states: Sequence[AffineSamplerState],
    model_indices: Sequence[int],
) -> list[torch.Tensor]:
    if len(states) != len(model_indices):
        raise ValueError("states/model_indices length mismatch.")
    grads: list[torch.Tensor] = []
    grad_fp32 = grad_out.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    for state, model_idx in zip(states, model_indices):
        per_model = grad_fp32[:, int(model_idx), :]
        rows = []
        for logical_idx in state.active_indices():
            latent_idx = int(logical_idx) % int(state.n_bits // max(1, grad_out.shape[0]))
            rows.append((per_model * weight_fp32[int(model_idx), :, latent_idx]).sum())
        grads.append(torch.stack(rows).to(torch.float16))
    return grads
