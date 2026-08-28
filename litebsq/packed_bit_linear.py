"""Triton kernels for multiplying bit-packed VQ codes by decoder linear weights.

The VAE checkpoints store the binary latent code along the last dimension in
little-endian uint8 bitpacks (bit 0 is latent coordinate 0). During decoder
training those codes are frozen, so materializing a dense bool/bf16 latent is
unnecessary. This module fuses bit unpacking with the decoder's first linear
projection and recomputes the packed input in backward when forming dW.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def packed_u8_linear_available() -> bool:
    return bool(_TRITON_AVAILABLE)


if _TRITON_AVAILABLE:

    @triton.jit
    def _packed_u8_linear_fwd_kernel(
        packed_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        B,
        M,
        IN: tl.constexpr,
        H,
        packed_stride_b,
        packed_stride_m,
        packed_stride_p,
        weight_stride_m,
        weight_stride_h,
        weight_stride_i,
        bias_stride_m,
        bias_stride_h,
        out_stride_b,
        out_stride_m,
        out_stride_h,
        MM_KIND: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_b = offs_b < B
        mask_h = offs_h < H
        acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

        # K/codebook_bits is tiled rather than hard-coded. Each tile decodes
        # only the bits needed by this matmul fragment directly from uint8.
        for k_start in tl.range(0, IN, BLOCK_K):
            offs_i = k_start + tl.arange(0, BLOCK_K)
            mask_i = offs_i < IN
            byte_offsets = offs_i // 8
            bit_offsets = offs_i % 8
            packed = tl.load(
                packed_ptr
                + offs_b[:, None] * packed_stride_b
                + pid_m * packed_stride_m
                + byte_offsets[None, :] * packed_stride_p,
                mask=mask_b[:, None] & mask_i[None, :],
                other=0,
            )
            bits = (packed >> bit_offsets[None, :]) & 1

            weight = tl.load(
                weight_ptr
                + pid_m * weight_stride_m
                + offs_h[:, None] * weight_stride_h
                + offs_i[None, :] * weight_stride_i,
                mask=mask_h[:, None] & mask_i[None, :],
                other=0.0,
            )
            if MM_KIND == 1:
                bits = bits.to(tl.bfloat16)
                weight = weight.to(tl.bfloat16)
            elif MM_KIND == 2:
                bits = bits.to(tl.float16)
                weight = weight.to(tl.float16)
            else:
                bits = bits.to(tl.float32)
                weight = weight.to(tl.float32)
            acc += tl.dot(bits, tl.trans(weight), out_dtype=tl.float32)
        bias = tl.load(
            bias_ptr + pid_m * bias_stride_m + offs_h * bias_stride_h,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        acc += bias[None, :]

        tl.store(
            out_ptr
            + offs_b[:, None] * out_stride_b
            + pid_m * out_stride_m
            + offs_h[None, :] * out_stride_h,
            acc,
            mask=mask_b[:, None] & mask_h[None, :],
        )


    @triton.jit
    def _packed_u8_linear_dw_kernel(
        packed_ptr,
        grad_ptr,
        partial_ptr,
        B,
        M,
        IN: tl.constexpr,
        H,
        rows_per_split,
        packed_stride_b,
        packed_stride_m,
        packed_stride_p,
        grad_stride_b,
        grad_stride_m,
        grad_stride_h,
        partial_stride_r,
        partial_stride_m,
        partial_stride_h,
        partial_stride_i,
        MM_KIND: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        N_I_TILES: tl.constexpr,
    ):
        split_id = tl.program_id(0)
        pid_m = tl.program_id(1)
        tile_id = tl.program_id(2)
        pid_h = tile_id // N_I_TILES
        pid_i = tile_id % N_I_TILES

        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_i = pid_i * BLOCK_IN + tl.arange(0, BLOCK_IN)
        mask_h = offs_h < H
        mask_i = offs_i < IN
        byte_offsets = offs_i // 8
        bit_offsets = offs_i % 8

        split_start = split_id * rows_per_split
        split_end = tl.minimum(split_start + rows_per_split, B)
        acc = tl.zeros((BLOCK_H, BLOCK_IN), dtype=tl.float32)

        for row_start in tl.range(split_start, split_end, BLOCK_B):
            offs_b = row_start + tl.arange(0, BLOCK_B)
            mask_b = offs_b < split_end
            grad = tl.load(
                grad_ptr
                + offs_b[:, None] * grad_stride_b
                + pid_m * grad_stride_m
                + offs_h[None, :] * grad_stride_h,
                mask=mask_b[:, None] & mask_h[None, :],
                other=0.0,
            )
            packed = tl.load(
                packed_ptr
                + offs_b[:, None] * packed_stride_b
                + pid_m * packed_stride_m
                + byte_offsets[None, :] * packed_stride_p,
                mask=mask_b[:, None] & mask_i[None, :],
                other=0,
            )
            bits = (packed >> bit_offsets[None, :]) & 1

            if MM_KIND == 1:
                grad = grad.to(tl.bfloat16)
                bits = bits.to(tl.bfloat16)
            elif MM_KIND == 2:
                grad = grad.to(tl.float16)
                bits = bits.to(tl.float16)
            else:
                grad = grad.to(tl.float32)
                bits = bits.to(tl.float32)
            acc += tl.dot(tl.trans(grad), bits, out_dtype=tl.float32)

        tl.store(
            partial_ptr
            + split_id * partial_stride_r
            + pid_m * partial_stride_m
            + offs_h[:, None] * partial_stride_h
            + offs_i[None, :] * partial_stride_i,
            acc,
            mask=mask_h[:, None] & mask_i[None, :],
        )


def _resolve_mm_kind(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float16:
        return 2
    if dtype == torch.float32:
        return 0
    raise ValueError(f"packed uint8 decoder linear supports fp32/bf16/fp16 weights, got {dtype}.")


def _validate_inputs(
    packed: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    logical_in_dim: int,
) -> Tuple[int, int, int, int]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for packed uint8 decoder linear.")
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be uint8, got {packed.dtype}.")
    if packed.ndim != 3:
        raise ValueError(f"packed must have shape [B, M, ceil(IN/8)], got {tuple(packed.shape)}.")
    if weight.ndim != 3:
        raise ValueError(f"weight must have shape [M, H, IN], got {tuple(weight.shape)}.")
    if bias.ndim != 2:
        raise ValueError(f"bias must have shape [M, H], got {tuple(bias.shape)}.")
    if packed.device.type != "cuda" or weight.device.type != "cuda" or bias.device.type != "cuda":
        raise ValueError("packed uint8 decoder linear requires packed/weight/bias on CUDA.")
    if packed.device != weight.device or packed.device != bias.device:
        raise ValueError(
            f"packed/weight/bias device mismatch: {packed.device}, {weight.device}, {bias.device}."
        )
    if not weight.is_floating_point() or not bias.is_floating_point():
        raise ValueError("weight and bias must be floating tensors.")
    if weight.dtype != bias.dtype:
        raise ValueError(f"weight/bias dtype mismatch: {weight.dtype} vs {bias.dtype}.")
    _resolve_mm_kind(weight.dtype)

    B, M, packed_width = (int(v) for v in packed.shape)
    weight_m, H, weight_in = (int(v) for v in weight.shape)
    logical_in = int(logical_in_dim)
    if logical_in < 1:
        raise ValueError(f"logical_in_dim must be positive, got {logical_in}.")
    if weight_m != M or int(bias.shape[0]) != M or int(bias.shape[1]) != H:
        raise ValueError(
            f"packed/weight/bias model dimensions mismatch: packed M={M}, "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}."
        )
    if weight_in != logical_in:
        raise ValueError(f"weight input dim {weight_in} != logical_in_dim {logical_in}.")
    expected_packed = (logical_in + 7) // 8
    if packed_width != expected_packed:
        raise ValueError(
            f"packed width {packed_width} != ceil(logical_in_dim/8)={expected_packed}."
        )
    return B, M, logical_in, H


def _packed_u8_linear_forward(
    packed: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    logical_in_dim: int,
    activation_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, int, int, int, int]:
    B, M, IN, H = _validate_inputs(
        packed,
        weight,
        bias,
        logical_in_dim=int(logical_in_dim),
    )
    packed_c = packed.contiguous()
    _resolve_mm_kind(activation_dtype)
    out = torch.empty((B, M, H), device=weight.device, dtype=activation_dtype)
    block_k = 32
    block_b = 64
    block_h = 32
    mm_kind = _resolve_mm_kind(activation_dtype)
    grid = (triton.cdiv(B, block_b), M, triton.cdiv(H, block_h))
    _packed_u8_linear_fwd_kernel[grid](
        packed_c,
        weight,
        bias,
        out,
        B,
        M,
        IN,
        H,
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        bias.stride(0),
        bias.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        MM_KIND=mm_kind,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out, packed_c, B, M, IN, H


class _PackedU8Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        packed: Tensor,
        weight: Tensor,
        bias: Tensor,
        logical_in_dim: int,
        activation_dtype: torch.dtype,
    ) -> Tensor:
        out, packed_c, _B, M, IN, H = _packed_u8_linear_forward(
            packed,
            weight,
            bias,
            logical_in_dim=int(logical_in_dim),
            activation_dtype=activation_dtype,
        )
        ctx.save_for_backward(packed_c)
        ctx.logical_in_dim = int(IN)
        ctx.hidden_dim = int(H)
        ctx.num_models = int(M)
        ctx.weight_dtype = weight.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (packed,) = ctx.saved_tensors
        grad_c = grad_out.contiguous()
        B = int(packed.shape[0])
        M = int(ctx.num_models)
        IN = int(ctx.logical_in_dim)
        H = int(ctx.hidden_dim)
        block_b = 64
        block_h = 32
        block_in = 32
        n_i_tiles = triton.cdiv(IN, block_in)

        reduction_splits = min(64, max(1, triton.cdiv(B, block_b * 8)))
        rows_per_split = triton.cdiv(B, reduction_splits)
        partial = torch.empty(
            (reduction_splits, M, H, IN),
            device=grad_c.device,
            dtype=torch.float32,
        )
        mm_kind = _resolve_mm_kind(grad_c.dtype)
        grid = (reduction_splits, M, triton.cdiv(H, block_h) * n_i_tiles)
        _packed_u8_linear_dw_kernel[grid](
            packed,
            grad_c,
            partial,
            B,
            M,
            IN,
            H,
            rows_per_split,
            packed.stride(0),
            packed.stride(1),
            packed.stride(2),
            grad_c.stride(0),
            grad_c.stride(1),
            grad_c.stride(2),
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            partial.stride(3),
            MM_KIND=mm_kind,
            BLOCK_B=block_b,
            BLOCK_IN=block_in,
            BLOCK_H=block_h,
            N_I_TILES=n_i_tiles,
            num_warps=4,
        )
        grad_weight = partial.sum(dim=0, dtype=torch.float32).to(dtype=ctx.weight_dtype)
        grad_bias = grad_c.sum(dim=0, dtype=torch.float32).to(dtype=ctx.weight_dtype)
        return None, grad_weight, grad_bias, None, None


def packed_u8_linear(
    packed: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    logical_in_dim: int,
    activation_dtype: torch.dtype,
) -> Tensor:
    """Compute ``bool_unpack(packed) @ weight.T + bias`` without dense unpack."""

    if not torch.is_grad_enabled() or not (bool(weight.requires_grad) or bool(bias.requires_grad)):
        out, _packed_c, _B, _M, _IN, _H = _packed_u8_linear_forward(
            packed,
            weight,
            bias,
            logical_in_dim=int(logical_in_dim),
            activation_dtype=activation_dtype,
        )
        return out
    return _PackedU8Linear.apply(
        packed,
        weight,
        bias,
        int(logical_in_dim),
        activation_dtype,
    )


def resolve_parallel_linear_weight_bias(linear) -> Tuple[Tensor, Tensor]:
    """Return nn.Linear/ParallelLinear parameters as [M,H,IN] and [M,H] views."""

    if isinstance(linear, torch.nn.Linear):
        if linear.bias is None:
            raise ValueError("packed uint8 decoder linear currently requires a bias term.")
        return linear.weight.unsqueeze(0), linear.bias.unsqueeze(0)

    num_models = int(getattr(linear, "num_models", 0))
    in_features = int(getattr(linear, "in_features", 0))
    out_features = int(getattr(linear, "out_features", 0))
    if num_models < 1 or in_features < 1 or out_features < 1:
        raise ValueError("linear must expose positive num_models/in_features/out_features.")

    if num_models == 1:
        base = getattr(linear, "linear", None)
        if base is None or not isinstance(getattr(base, "weight", None), torch.Tensor):
            raise ValueError("single-model ParallelLinear must expose linear.weight.")
        if base.bias is None:
            raise ValueError("packed uint8 decoder linear currently requires a bias term.")
        return base.weight.unsqueeze(0), base.bias.unsqueeze(0)

    conv = getattr(linear, "conv", None)
    if conv is None or not isinstance(getattr(conv, "weight", None), torch.Tensor):
        raise ValueError("multi-model ParallelLinear must expose conv.weight.")
    if conv.bias is None:
        raise ValueError("packed uint8 decoder linear currently requires a bias term.")
    weight = conv.weight[:, :, 0].view(num_models, out_features, in_features)
    bias = conv.bias.view(num_models, out_features)
    return weight, bias


def packed_u8_parallel_linear(
    packed: Tensor,
    linear,
    *,
    logical_in_dim: Optional[int] = None,
    activation_dtype: torch.dtype,
) -> Tensor:
    """Apply a ParallelLinear directly to bit-packed binary inputs."""

    weight, bias = resolve_parallel_linear_weight_bias(linear)
    resolved_in = int(weight.shape[-1]) if logical_in_dim is None else int(logical_in_dim)
    return packed_u8_linear(
        packed,
        weight,
        bias,
        logical_in_dim=resolved_in,
        activation_dtype=activation_dtype,
    )
