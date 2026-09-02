from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def sparse_bit_triton_available() -> bool:
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


if _TRITON_AVAILABLE:

    @triton.jit
    def _active_logical_idx(
        q,
        n_bits,
        n_active,
        cursor,
        stride,
        offset,
        remaining,
        secondary_stride,
        secondary_offset,
    ):
        is_tail = remaining < n_active
        use_primary = (~is_tail) | (q < remaining)
        primary_pos = cursor + q
        filler_q = q - remaining
        safe_cursor = tl.maximum(cursor, 1)
        filler_pos = (secondary_stride * filler_q + secondary_offset) % safe_cursor
        pos = tl.where(use_primary, primary_pos, filler_pos)
        return (stride * pos + offset) % n_bits


    @triton.jit
    def _init_scores_kernel(
        packed_ptr,
        score_ptr,
        n_bits_ptr,
        n_active_ptr,
        cursor_ptr,
        stride_ptr,
        offset_ptr,
        remaining_ptr,
        secondary_stride_ptr,
        secondary_offset_ptr,
        model_idx_ptr,
        score_offset_ptr,
        B: tl.constexpr,
        M: tl.constexpr,
        IN: tl.constexpr,
        P: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        bank = tl.program_id(0)
        pid_q = tl.program_id(1)
        q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        n_bits = tl.load(n_bits_ptr + bank).to(tl.int64)
        n_active = tl.load(n_active_ptr + bank).to(tl.int64)
        cursor = tl.load(cursor_ptr + bank).to(tl.int64)
        stride = tl.load(stride_ptr + bank).to(tl.int64)
        offset = tl.load(offset_ptr + bank).to(tl.int64)
        remaining = tl.load(remaining_ptr + bank).to(tl.int64)
        sec_stride = tl.load(secondary_stride_ptr + bank).to(tl.int64)
        sec_offset = tl.load(secondary_offset_ptr + bank).to(tl.int64)
        model_idx = tl.load(model_idx_ptr + bank).to(tl.int64)
        score_offset = tl.load(score_offset_ptr + bank).to(tl.int64)
        mask = q < n_active
        logical = _active_logical_idx(
            q.to(tl.int64),
            n_bits,
            n_active,
            cursor,
            stride,
            offset,
            remaining,
            sec_stride,
            sec_offset,
        )
        block_idx = logical // IN
        latent_idx = logical % IN
        byte_idx = latent_idx // 8
        bit_offset = latent_idx % 8
        packed_offset = (block_idx * M + model_idx) * P + byte_idx
        packed = tl.load(packed_ptr + packed_offset, mask=mask, other=0).to(tl.int32)
        bit = (packed >> bit_offset) & 1
        score = tl.where(bit != 0, 1.0, -1.0).to(tl.float16)
        tl.store(score_ptr + score_offset + q, score, mask=mask)


    @triton.jit
    def _set_scores_full_words_kernel(
        packed_ptr,
        score_ptr,
        flip_ptr,
        n_bits_ptr,
        n_active_ptr,
        cursor_ptr,
        stride_ptr,
        offset_ptr,
        remaining_ptr,
        secondary_stride_ptr,
        secondary_offset_ptr,
        model_idx_ptr,
        score_offset_ptr,
        PACKED_NUMEL,
        B: tl.constexpr,
        M: tl.constexpr,
        IN: tl.constexpr,
        P: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        bank = tl.program_id(0)
        pid_q = tl.program_id(1)
        q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        n_bits = tl.load(n_bits_ptr + bank).to(tl.int64)
        n_active = tl.load(n_active_ptr + bank).to(tl.int64)
        cursor = tl.load(cursor_ptr + bank).to(tl.int64)
        stride = tl.load(stride_ptr + bank).to(tl.int64)
        offset = tl.load(offset_ptr + bank).to(tl.int64)
        remaining = tl.load(remaining_ptr + bank).to(tl.int64)
        sec_stride = tl.load(secondary_stride_ptr + bank).to(tl.int64)
        sec_offset = tl.load(secondary_offset_ptr + bank).to(tl.int64)
        model_idx = tl.load(model_idx_ptr + bank).to(tl.int64)
        score_offset = tl.load(score_offset_ptr + bank).to(tl.int64)
        mask_q = q < n_active
        logical = _active_logical_idx(
            q.to(tl.int64),
            n_bits,
            n_active,
            cursor,
            stride,
            offset,
            remaining,
            sec_stride,
            sec_offset,
        )
        block_idx = logical // IN
        latent_idx = logical % IN
        byte_idx = latent_idx // 8
        bit_offset = latent_idx % 8
        global_byte = (block_idx * M + model_idx) * P + byte_idx
        word_idx = global_byte // 4
        byte_lane = global_byte % 4
        bit_in_word = byte_lane * 8 + bit_offset
        full_words = PACKED_NUMEL // 4
        valid = mask_q & (word_idx < full_words)
        score = tl.load(score_ptr + score_offset + q, mask=mask_q, other=-1.0)
        new_bit = score >= 0.0
        bit_mask = (1 << bit_in_word).to(tl.uint32)
        byte_ptr = packed_ptr + word_idx * 4
        word_ptr = tl.cast(byte_ptr, tl.pointer_type(tl.uint32))
        old_or = tl.atomic_or(word_ptr, bit_mask, mask=valid & new_bit)
        old_and = tl.atomic_and(word_ptr, ~bit_mask, mask=valid & (~new_bit))
        old_word = tl.where(new_bit, old_or, old_and).to(tl.uint32)
        old_bit = (old_word >> bit_in_word) & 1
        flipped = valid & (old_bit != new_bit.to(tl.uint32))
        flip_count = tl.sum(flipped.to(tl.int32), axis=0)
        tl.atomic_add(flip_ptr, flip_count)


    @triton.jit
    def _tail_bytes_kernel(
        packed_ptr,
        score_ptr,
        flip_ptr,
        n_bits_ptr,
        n_active_ptr,
        cursor_ptr,
        stride_ptr,
        offset_ptr,
        inverse_ptr,
        remaining_ptr,
        secondary_stride_ptr,
        secondary_offset_ptr,
        secondary_inverse_ptr,
        model_idx_ptr,
        score_offset_ptr,
        PACKED_NUMEL: tl.constexpr,
        B: tl.constexpr,
        M: tl.constexpr,
        IN: tl.constexpr,
        P: tl.constexpr,
        NUM_BANKS: tl.constexpr,
    ):
        tail_bytes = PACKED_NUMEL % 4
        tail_lane = tl.program_id(0)
        valid_byte = tail_lane < tail_bytes
        global_byte = (PACKED_NUMEL // 4) * 4 + tail_lane
        original = tl.load(packed_ptr + global_byte, mask=valid_byte, other=0).to(tl.uint32)
        updated = original
        local_flips = tl.zeros((), dtype=tl.int32)
        block_idx = global_byte // (M * P)
        rem = global_byte % (M * P)
        model_idx_for_byte = rem // P
        byte_idx = rem % P
        for bank in tl.static_range(0, NUM_BANKS):
            model_idx = tl.load(model_idx_ptr + bank).to(tl.int64)
            bank_match = valid_byte & (model_idx == model_idx_for_byte)
            n_bits = tl.load(n_bits_ptr + bank).to(tl.int64)
            n_active = tl.load(n_active_ptr + bank).to(tl.int64)
            cursor = tl.load(cursor_ptr + bank).to(tl.int64)
            stride = tl.load(stride_ptr + bank).to(tl.int64)
            offset = tl.load(offset_ptr + bank).to(tl.int64)
            inverse = tl.load(inverse_ptr + bank).to(tl.int64)
            remaining = tl.load(remaining_ptr + bank).to(tl.int64)
            sec_offset = tl.load(secondary_offset_ptr + bank).to(tl.int64)
            sec_inverse = tl.load(secondary_inverse_ptr + bank).to(tl.int64)
            score_offset = tl.load(score_offset_ptr + bank).to(tl.int64)
            for bit_offset in tl.static_range(0, 8):
                latent_idx = byte_idx * 8 + bit_offset
                logical = block_idx * IN + latent_idx
                logical_valid = bank_match & (latent_idx < IN) & (logical < n_bits)
                primary_delta = (logical - offset + n_bits) % n_bits
                pos = tl.where(
                    n_bits == 1,
                    0,
                    (inverse * primary_delta) % n_bits,
                )
                is_tail = remaining < n_active
                primary_active = tl.where(is_tail, pos >= cursor, (pos >= cursor) & (pos < cursor + n_active))
                primary_q = pos - cursor
                fill = n_active - remaining
                safe_cursor = tl.maximum(cursor, 1)
                secondary_delta = (pos - sec_offset + safe_cursor) % safe_cursor
                filler_t = tl.where(
                    cursor <= 1,
                    0,
                    (sec_inverse * secondary_delta) % safe_cursor,
                )
                filler_active = is_tail & (pos < cursor) & (filler_t < fill)
                active = logical_valid & (primary_active | filler_active)
                q = tl.where(primary_active, primary_q, remaining + filler_t)
                score = tl.load(score_ptr + score_offset + q, mask=active, other=-1.0)
                new_bit = score >= 0.0
                old_bit = (updated >> bit_offset) & 1
                bit_mask = tl.full((), 1 << bit_offset, tl.uint32)
                updated = tl.where(active & new_bit, updated | bit_mask, updated)
                updated = tl.where(active & (~new_bit), updated & (~bit_mask), updated)
                local_flips += (active & (old_bit != new_bit.to(tl.uint32))).to(tl.int32)
        tl.store(packed_ptr + global_byte, updated.to(tl.uint8), mask=valid_byte)
        tl.atomic_add(flip_ptr, local_flips)


    @triton.jit
    def _dscore_kernel(
        grad_ptr,
        weight_ptr,
        out_ptr,
        n_bits_ptr,
        n_active_ptr,
        cursor_ptr,
        stride_ptr,
        offset_ptr,
        remaining_ptr,
        secondary_stride_ptr,
        secondary_offset_ptr,
        model_idx_ptr,
        score_offset_ptr,
        B,
        M,
        IN: tl.constexpr,
        H,
        grad_stride_b,
        grad_stride_m,
        grad_stride_h,
        weight_stride_m,
        weight_stride_h,
        weight_stride_i,
        BLOCK_Q: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        bank = tl.program_id(0)
        pid_q = tl.program_id(1)
        q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        n_bits = tl.load(n_bits_ptr + bank).to(tl.int64)
        n_active = tl.load(n_active_ptr + bank).to(tl.int64)
        cursor = tl.load(cursor_ptr + bank).to(tl.int64)
        stride = tl.load(stride_ptr + bank).to(tl.int64)
        offset = tl.load(offset_ptr + bank).to(tl.int64)
        remaining = tl.load(remaining_ptr + bank).to(tl.int64)
        sec_stride = tl.load(secondary_stride_ptr + bank).to(tl.int64)
        sec_offset = tl.load(secondary_offset_ptr + bank).to(tl.int64)
        model_idx = tl.load(model_idx_ptr + bank).to(tl.int64)
        score_offset = tl.load(score_offset_ptr + bank).to(tl.int64)
        mask_q = q < n_active
        logical = _active_logical_idx(
            q.to(tl.int64),
            n_bits,
            n_active,
            cursor,
            stride,
            offset,
            remaining,
            sec_stride,
            sec_offset,
        )
        block_idx = logical // IN
        latent_idx = logical % IN
        acc = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        for h_start in tl.range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H
            grad = tl.load(
                grad_ptr
                + block_idx[:, None] * grad_stride_b
                + model_idx * grad_stride_m
                + offs_h[None, :] * grad_stride_h,
                mask=mask_q[:, None] & mask_h[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr
                + model_idx * weight_stride_m
                + offs_h[None, :] * weight_stride_h
                + latent_idx[:, None] * weight_stride_i,
                mask=mask_q[:, None] & mask_h[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(grad * weight, axis=1)
        tl.store(out_ptr + score_offset + q, acc.to(tl.float16), mask=mask_q)


    @triton.jit
    def _rms_by_bank_kernel(
        grad_ptr,
        rms_ptr,
        n_active_ptr,
        score_offset_ptr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        bank = tl.program_id(0)
        n_active = tl.load(n_active_ptr + bank).to(tl.int64)
        start = tl.load(score_offset_ptr + bank).to(tl.int64)
        acc = tl.zeros((), dtype=tl.float32)
        for begin in tl.range(0, n_active, BLOCK):
            offs = begin + tl.arange(0, BLOCK)
            mask = offs < n_active
            grad = tl.load(grad_ptr + start + offs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(grad * grad, axis=0)
        mean = acc / n_active.to(tl.float32)
        tl.store(rms_ptr + bank, tl.sqrt(mean + EPS))


    @triton.jit
    def _rms_sgd_update_kernel(
        score_ptr,
        grad_ptr,
        rms_ptr,
        flip_ptr,
        n_active_ptr,
        score_offset_ptr,
        LR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        bank = tl.program_id(0)
        pid = tl.program_id(1)
        n_active = tl.load(n_active_ptr + bank).to(tl.int64)
        start = tl.load(score_offset_ptr + bank).to(tl.int64)
        q = pid * BLOCK + tl.arange(0, BLOCK)
        mask = q < n_active
        score = tl.load(score_ptr + start + q, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + start + q, mask=mask, other=0.0).to(tl.float32)
        rms = tl.load(rms_ptr + bank).to(tl.float32)
        updated = tl.maximum(-1.0, tl.minimum(1.0, score - LR * grad / rms))
        updated_fp16 = updated.to(tl.float16)
        flipped = mask & ((score >= 0.0) != (updated_fp16 >= 0.0))
        tl.atomic_add(flip_ptr, tl.sum(flipped.to(tl.int32), axis=0))
        tl.store(score_ptr + start + q, updated_fp16, mask=mask)


    @triton.jit
    def _adam_update_kernel(
        score_ptr,
        grad_ptr,
        m_ptr,
        v_ptr,
        flip_ptr,
        n_active_ptr,
        score_offset_ptr,
        LR: tl.constexpr,
        BETA1: tl.constexpr,
        BETA2: tl.constexpr,
        EPS: tl.constexpr,
        BIAS1: tl.constexpr,
        BIAS2: tl.constexpr,
        WEIGHT_DECAY: tl.constexpr,
        USE_ADAMW: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        bank = tl.program_id(0)
        pid = tl.program_id(1)
        n_active = tl.load(n_active_ptr + bank).to(tl.int64)
        start = tl.load(score_offset_ptr + bank).to(tl.int64)
        q = pid * BLOCK + tl.arange(0, BLOCK)
        mask = q < n_active
        idx = start + q
        score = tl.load(score_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        m = BETA1 * m + (1.0 - BETA1) * grad
        v = BETA2 * v + (1.0 - BETA2) * grad * grad
        if USE_ADAMW:
            score = score * (1.0 - LR * WEIGHT_DECAY)
        m_hat = m / BIAS1
        v_hat = v / BIAS2
        updated = score - LR * m_hat / (tl.sqrt(v_hat) + EPS)
        updated = tl.maximum(-1.0, tl.minimum(1.0, updated))
        updated_fp16 = updated.to(tl.float16)
        flipped = mask & ((score >= 0.0) != (updated_fp16 >= 0.0))
        tl.atomic_add(flip_ptr, tl.sum(flipped.to(tl.int32), axis=0))
        tl.store(m_ptr + idx, m, mask=mask)
        tl.store(v_ptr + idx, v, mask=mask)
        tl.store(score_ptr + idx, updated_fp16, mask=mask)


def _require_triton() -> None:
    if not sparse_bit_triton_available():
        raise RuntimeError("Sparse Bit Tuning requires CUDA + Triton for the production path.")


def launch_init_scores(
    packed: torch.Tensor,
    score_span: torch.Tensor,
    meta,
) -> None:
    _require_triton()
    max_active = int(meta.max_active)
    block = 256
    grid = (int(meta.num_banks), triton.cdiv(max_active, block))
    _init_scores_kernel[grid](
        packed,
        score_span,
        meta.n_bits,
        meta.n_active,
        meta.cursor,
        meta.stride,
        meta.offset,
        meta.remaining,
        meta.secondary_stride,
        meta.secondary_offset,
        meta.model_idx,
        meta.score_offset,
        B=int(packed.shape[0]),
        M=int(packed.shape[1]),
        IN=int(meta.logical_in_dim),
        P=int(packed.shape[2]),
        BLOCK_Q=block,
        num_warps=4,
    )


def launch_set_scores(
    packed: torch.Tensor,
    score_span: torch.Tensor,
    meta,
    *,
    flip_counter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _require_triton()
    if not packed.is_contiguous():
        raise ValueError("Sparse Bit packed SET requires contiguous uint8 packed storage.")
    if flip_counter is None:
        flip_counter = torch.zeros((), device=packed.device, dtype=torch.int32)
    else:
        flip_counter.zero_()
    block = 256
    grid = (int(meta.num_banks), triton.cdiv(int(meta.max_active), block))
    _set_scores_full_words_kernel[grid](
        packed,
        score_span,
        flip_counter,
        meta.n_bits,
        meta.n_active,
        meta.cursor,
        meta.stride,
        meta.offset,
        meta.remaining,
        meta.secondary_stride,
        meta.secondary_offset,
        meta.model_idx,
        meta.score_offset,
        int(packed.numel()),
        B=int(packed.shape[0]),
        M=int(packed.shape[1]),
        IN=int(meta.logical_in_dim),
        P=int(packed.shape[2]),
        BLOCK_Q=block,
        num_warps=4,
    )
    tail = int(packed.numel()) % 4
    if tail:
        _tail_bytes_kernel[(tail,)](
            packed,
            score_span,
            flip_counter,
            meta.n_bits,
            meta.n_active,
            meta.cursor,
            meta.stride,
            meta.offset,
            meta.inverse,
            meta.remaining,
            meta.secondary_stride,
            meta.secondary_offset,
            meta.secondary_inverse,
            meta.model_idx,
            meta.score_offset,
            PACKED_NUMEL=int(packed.numel()),
            B=int(packed.shape[0]),
            M=int(packed.shape[1]),
            IN=int(meta.logical_in_dim),
            P=int(packed.shape[2]),
            NUM_BANKS=int(meta.num_banks),
            num_warps=1,
        )
    return flip_counter


def launch_dscore(
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    score_grad: torch.Tensor,
    meta,
) -> None:
    _require_triton()
    block_q = 32
    block_h = 128
    grid = (int(meta.num_banks), triton.cdiv(int(meta.max_active), block_q))
    _dscore_kernel[grid](
        grad_out,
        weight,
        score_grad,
        meta.n_bits,
        meta.n_active,
        meta.cursor,
        meta.stride,
        meta.offset,
        meta.remaining,
        meta.secondary_stride,
        meta.secondary_offset,
        meta.model_idx,
        meta.score_offset,
        int(grad_out.shape[0]),
        int(grad_out.shape[1]),
        int(meta.logical_in_dim),
        int(grad_out.shape[2]),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        BLOCK_Q=block_q,
        BLOCK_H=block_h,
        num_warps=4,
    )


def launch_rms_sgd_update(
    score: torch.Tensor,
    grad: torch.Tensor,
    meta,
    *,
    lr: float,
    eps: float = 1e-8,
    flip_counter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _require_triton()
    if flip_counter is None:
        flip_counter = torch.zeros((), device=score.device, dtype=torch.int32)
    else:
        flip_counter.zero_()
    rms = torch.empty((int(meta.num_banks),), device=score.device, dtype=torch.float32)
    _rms_by_bank_kernel[(int(meta.num_banks),)](
        grad,
        rms,
        meta.n_active,
        meta.score_offset,
        EPS=float(eps),
        BLOCK=1024,
        num_warps=8,
    )
    block = 256
    _rms_sgd_update_kernel[(int(meta.num_banks), triton.cdiv(int(meta.max_active), block))](
        score,
        grad,
        rms,
        flip_counter,
        meta.n_active,
        meta.score_offset,
        LR=float(lr),
        BLOCK=block,
        num_warps=4,
    )
    return flip_counter


def launch_adam_update(
    score: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    meta,
    *,
    lr: float,
    step: int,
    weight_decay: float = 0.0,
    adamw: bool = False,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    flip_counter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _require_triton()
    if flip_counter is None:
        flip_counter = torch.zeros((), device=score.device, dtype=torch.int32)
    else:
        flip_counter.zero_()
    t = int(step)
    if t < 1:
        raise ValueError(f"Adam step must be >=1, got {t}.")
    bias1 = 1.0 - float(beta1) ** t
    bias2 = 1.0 - float(beta2) ** t
    block = 256
    _adam_update_kernel[(int(meta.num_banks), triton.cdiv(int(meta.max_active), block))](
        score,
        grad,
        exp_avg,
        exp_avg_sq,
        flip_counter,
        meta.n_active,
        meta.score_offset,
        LR=float(lr),
        BETA1=float(beta1),
        BETA2=float(beta2),
        EPS=float(eps),
        BIAS1=float(bias1),
        BIAS2=float(bias2),
        WEIGHT_DECAY=float(weight_decay),
        USE_ADAMW=bool(adamw),
        BLOCK=block,
        num_warps=4,
    )
    return flip_counter
