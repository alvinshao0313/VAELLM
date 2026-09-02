from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor

from litebsq.packed_bit_linear import (
    _packed_u8_linear_dw_kernel,
    _packed_u8_linear_forward,
    _resolve_mm_kind,
    resolve_parallel_linear_weight_bias,
)
from .sampler import AffineSamplerState
from .triton_kernels import launch_dscore, launch_init_scores, launch_set_scores

try:
    import triton
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]


@dataclass
class PackedBitRuntimeMeta:
    states: Tuple[AffineSamplerState, ...]
    model_indices: Tuple[int, ...]
    logical_in_dim: int
    score_offsets_py: Tuple[int, ...]
    n_bits: Tensor
    n_active: Tensor
    cursor: Tensor
    stride: Tensor
    offset: Tensor
    inverse: Tensor
    remaining: Tensor
    secondary_stride: Tensor
    secondary_offset: Tensor
    secondary_inverse: Tensor
    model_idx: Tensor
    score_offset: Tensor

    @property
    def num_banks(self) -> int:
        return len(self.states)

    @property
    def max_active(self) -> int:
        return max(int(state.n_active) for state in self.states)

    @property
    def total_active(self) -> int:
        return sum(int(state.n_active) for state in self.states)

    @property
    def device(self) -> torch.device:
        return self.n_bits.device

    @classmethod
    def build(
        cls,
        *,
        states: Sequence[AffineSamplerState],
        model_indices: Sequence[int],
        score_offsets: Sequence[int] | None,
        logical_in_dim: int,
        device: torch.device,
    ) -> "PackedBitRuntimeMeta":
        states_t = tuple(states)
        model_t = tuple(int(v) for v in model_indices)
        if not states_t:
            raise ValueError("PackedBitRuntimeMeta requires at least one bank.")
        if len(states_t) != len(model_t):
            raise ValueError("states/model_indices length mismatch.")
        logical_in = int(logical_in_dim)
        if logical_in < 1:
            raise ValueError(f"logical_in_dim must be >=1, got {logical_in}.")
        if score_offsets is None:
            offsets = []
            cursor = 0
            for state in states_t:
                offsets.append(cursor)
                cursor += int(state.n_active)
            score_offsets_t = tuple(offsets)
        else:
            score_offsets_t = tuple(int(v) for v in score_offsets)
            if len(score_offsets_t) != len(states_t):
                raise ValueError("score_offsets/states length mismatch.")
        metas = [state.subset_meta() for state in states_t]
        dev = torch.device(device)

        def ints(values) -> Tensor:
            return torch.tensor(list(values), dtype=torch.int64, device=dev)

        return cls(
            states=states_t,
            model_indices=model_t,
            logical_in_dim=logical_in,
            score_offsets_py=score_offsets_t,
            n_bits=ints(state.n_bits for state in states_t),
            n_active=ints(state.n_active for state in states_t),
            cursor=ints(state.cursor for state in states_t),
            stride=ints(state.stride for state in states_t),
            offset=ints(state.offset for state in states_t),
            inverse=ints(state.inverse for state in states_t),
            remaining=ints(meta.remaining for meta in metas),
            secondary_stride=ints(meta.secondary_stride for meta in metas),
            secondary_offset=ints(meta.secondary_offset for meta in metas),
            secondary_inverse=ints(meta.secondary_inverse for meta in metas),
            model_idx=ints(model_t),
            score_offset=ints(score_offsets_t),
        )


def initialize_scores_from_packed(
    packed: Tensor,
    score_span: Tensor,
    meta: PackedBitRuntimeMeta,
) -> None:
    if packed.device != score_span.device or packed.device != meta.device:
        raise ValueError(
            f"packed/score/meta device mismatch: {packed.device}, {score_span.device}, {meta.device}."
        )
    if score_span.dtype != torch.float16:
        raise ValueError(f"score_span must be FP16, got {score_span.dtype}.")
    if int(score_span.numel()) < max(
        offset + int(state.n_active) for offset, state in zip(meta.score_offsets_py, meta.states)
    ):
        raise ValueError("score_span is too small for runtime metadata.")
    launch_init_scores(packed, score_span, meta)


def project_scores_to_packed(
    packed: Tensor,
    score_span: Tensor,
    meta: PackedBitRuntimeMeta,
    *,
    flip_counter: Tensor | None = None,
) -> Tensor:
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be uint8, got {packed.dtype}.")
    if packed.device != score_span.device or packed.device != meta.device:
        raise ValueError(
            f"packed/score/meta device mismatch: {packed.device}, {score_span.device}, {meta.device}."
        )
    return launch_set_scores(packed, score_span, meta, flip_counter=flip_counter)


def _compute_decoder_weight_bias_grads(
    packed: Tensor,
    grad_out: Tensor,
    *,
    logical_in_dim: int,
    weight_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for packed decoder gradients.")
    grad_c = grad_out.contiguous()
    B, M, H = (int(v) for v in grad_c.shape)
    IN = int(logical_in_dim)
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
    grad_weight = partial.sum(dim=0, dtype=torch.float32).to(dtype=weight_dtype)
    grad_bias = grad_c.sum(dim=0, dtype=torch.float32).to(dtype=weight_dtype)
    return grad_weight, grad_bias


class _SparseBitPackedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        packed: Tensor,
        weight: Tensor,
        bias: Tensor,
        score_span: Tensor,
        meta: PackedBitRuntimeMeta,
        logical_in_dim: int,
        activation_dtype: torch.dtype,
    ) -> Tensor:
        out, packed_c, _B, _M, IN, _H = _packed_u8_linear_forward(
            packed,
            weight,
            bias,
            logical_in_dim=int(logical_in_dim),
            activation_dtype=activation_dtype,
        )
        # save_for_backward is deliberate: the existing saved-tensor offload hooks
        # must be able to move/restore both packed codes and first-layer weights.
        ctx.save_for_backward(packed_c, weight)
        ctx.meta = meta
        ctx.logical_in_dim = int(IN)
        ctx.weight_dtype = weight.dtype
        ctx.score_dtype = score_span.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        packed, weight = ctx.saved_tensors
        meta: PackedBitRuntimeMeta = ctx.meta
        grad_c = grad_out.contiguous()
        grad_packed = None
        grad_weight = None
        grad_bias = None
        grad_score = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            computed_weight, computed_bias = _compute_decoder_weight_bias_grads(
                packed,
                grad_c,
                logical_in_dim=int(ctx.logical_in_dim),
                weight_dtype=ctx.weight_dtype,
            )
            if ctx.needs_input_grad[1]:
                grad_weight = computed_weight
            if ctx.needs_input_grad[2]:
                grad_bias = computed_bias
        if ctx.needs_input_grad[3]:
            total = max(
                offset + int(state.n_active)
                for offset, state in zip(meta.score_offsets_py, meta.states)
            )
            grad_score = torch.empty((total,), device=grad_c.device, dtype=ctx.score_dtype)
            launch_dscore(grad_c, weight, grad_score, meta)
        return grad_packed, grad_weight, grad_bias, grad_score, None, None, None


def bit_aware_packed_u8_linear(
    packed: Tensor,
    weight: Tensor,
    bias: Tensor,
    score_span: Tensor,
    meta: PackedBitRuntimeMeta,
    *,
    logical_in_dim: int,
    activation_dtype: torch.dtype,
) -> Tensor:
    if not torch.is_grad_enabled():
        raise RuntimeError("bit_aware_packed_u8_linear is training/autograd only.")
    if not bool(score_span.requires_grad):
        raise RuntimeError("Sparse Bit score span must require gradients during training.")
    return _SparseBitPackedLinear.apply(
        packed,
        weight,
        bias,
        score_span,
        meta,
        int(logical_in_dim),
        activation_dtype,
    )


def bit_aware_packed_parallel_linear(
    packed: Tensor,
    linear,
    score_span: Tensor,
    meta: PackedBitRuntimeMeta,
    *,
    logical_in_dim: int | None = None,
    activation_dtype: torch.dtype,
) -> Tensor:
    weight, bias = resolve_parallel_linear_weight_bias(linear)
    resolved_in = int(weight.shape[-1]) if logical_in_dim is None else int(logical_in_dim)
    return bit_aware_packed_u8_linear(
        packed,
        weight,
        bias,
        score_span,
        meta,
        logical_in_dim=resolved_in,
        activation_dtype=activation_dtype,
    )
