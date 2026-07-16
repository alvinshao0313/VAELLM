"""Fused multi-stage symmetric decoder (LayerNorm + swish) with Triton forward.

Inference and training share the same fused forward. For bf16/fp16 inputs,
matmuls use Tensor Cores (fp32 accumulate, then cast to the input dtype like
F.linear) and LayerNorm/swish follow the same bf16/fp16 rounding as a serial
PyTorch chain. Backward uses batched GEMM across stages (one-shot).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def reference_multistage_symmetric_decode(
    x: Tensor,
    w_in: Tensor,
    b_in: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    w_out: Tensor,
    b_out: Tensor,
    *,
    eps: float = 1e-5,
) -> Tensor:
    """Pure PyTorch reference: x[B,S,In] -> y[B,Out]."""
    h = torch.einsum("bsi,shi->bsh", x, w_in) + b_in.unsqueeze(0)
    mean = h.mean(dim=-1, keepdim=True)
    var = ((h - mean) ** 2).mean(dim=-1, keepdim=True)
    h_hat = (h - mean) * torch.rsqrt(var + eps)
    h_n = h_hat * ln_w.unsqueeze(0) + ln_b.unsqueeze(0)
    h_a = _swish(h_n)
    y = torch.einsum("bsh,soh->bso", h_a, w_out) + b_out.unsqueeze(0)
    return y.sum(dim=1)


def _next_pow2(n: int) -> int:
    v = 1
    while v < int(n):
        v *= 2
    return int(v)


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_fwd_kernel(
        x_ptr,
        w_in_ptr,
        b_in_ptr,
        ln_w_ptr,
        ln_b_ptr,
        w_out_ptr,
        b_out_ptr,
        y_ptr,
        h_pre_ptr,
        h_act_ptr,
        B,
        S,
        IN,
        H,
        OUT,
        stride_x_b,
        stride_x_s,
        stride_win_s,
        stride_wout_s,
        stride_h_b,
        stride_h_s,
        eps,
        SAVE_AUX: tl.constexpr,
        # 0: fp32/TF32 matmul; 1: bf16 Tensor Core; 2: fp16 Tensor Core
        MM_KIND: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start = pid * BLOCK_M
        offs_m = start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < B

        offs_in = tl.arange(0, BLOCK_IN)
        offs_h = tl.arange(0, BLOCK_H)
        offs_out = tl.arange(0, BLOCK_OUT)
        mask_in = offs_in < IN
        mask_h = offs_h < H
        mask_out = offs_out < OUT

        acc = tl.zeros((BLOCK_M, BLOCK_OUT), dtype=tl.float32)

        for s in range(0, S):
            # Load stage weights once for this tile of rows.
            # Matmul operands stay in bf16/fp16 when MM_KIND!=0 so tl.dot hits Tensor Cores.
            w_in = tl.load(
                w_in_ptr + s * stride_win_s + offs_h[:, None] * IN + offs_in[None, :],
                mask=mask_h[:, None] & mask_in[None, :],
                other=0.0,
            )
            b_in = tl.load(b_in_ptr + s * H + offs_h, mask=mask_h, other=0.0)
            ln_w = tl.load(ln_w_ptr + s * H + offs_h, mask=mask_h, other=0.0)
            ln_b = tl.load(ln_b_ptr + s * H + offs_h, mask=mask_h, other=0.0)
            w_out = tl.load(
                w_out_ptr + s * stride_wout_s + offs_out[:, None] * H + offs_h[None, :],
                mask=mask_out[:, None] & mask_h[None, :],
                other=0.0,
            )
            b_out = tl.load(b_out_ptr + s * OUT + offs_out, mask=mask_out, other=0.0)

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x_b + s * stride_x_s + offs_in[None, :],
                mask=mask_m[:, None] & mask_in[None, :],
                other=0.0,
            )

            if MM_KIND == 0:
                w_in = w_in.to(tl.float32)
                w_out = w_out.to(tl.float32)
                x = x.to(tl.float32)
                b_in_f = b_in.to(tl.float32)
                ln_w_f = ln_w.to(tl.float32)
                ln_b_f = ln_b.to(tl.float32)
                b_out_f = b_out.to(tl.float32)
                # h = x @ W_in.T + b_in ; x[BLOCK_M, IN], W_in[H, IN] -> [BLOCK_M, H]
                h = tl.dot(x, tl.trans(w_in)) + b_in_f[None, :]

                if SAVE_AUX:
                    tl.store(
                        h_pre_ptr + offs_m[:, None] * stride_h_b + s * stride_h_s + offs_h[None, :],
                        h,
                        mask=mask_m[:, None] & mask_h[None, :],
                    )

                mean = tl.sum(h, axis=1) / H
                centered = h - mean[:, None]
                centered = tl.where(mask_h[None, :], centered, 0.0)
                var = tl.sum(centered * centered, axis=1) / H
                rstd = 1.0 / tl.sqrt(var + eps)
                h_hat = centered * rstd[:, None]
                h_n = h_hat * ln_w_f[None, :] + ln_b_f[None, :]
                sig = 1.0 / (1.0 + tl.exp(-h_n))
                h_a = h_n * sig

                if SAVE_AUX:
                    tl.store(
                        h_act_ptr + offs_m[:, None] * stride_h_b + s * stride_h_s + offs_h[None, :],
                        h_a,
                        mask=mask_m[:, None] & mask_h[None, :],
                    )

                y_s = tl.dot(h_a, tl.trans(w_out)) + b_out_f[None, :]
                acc += y_s
            else:
                # Half-precision path matching serial F.linear + bf16/fp16 LN/swish:
                # gemm fp32-acc -> cast to element dtype; LN reductions fp32->cast;
                # elementwise (including var squares) stay in element dtype.
                h = tl.dot(x, tl.trans(w_in), out_dtype=tl.float32) + b_in.to(tl.float32)[None, :]
                if MM_KIND == 1:
                    h = h.to(tl.bfloat16)
                else:
                    h = h.to(tl.float16)

                if SAVE_AUX:
                    tl.store(
                        h_pre_ptr + offs_m[:, None] * stride_h_b + s * stride_h_s + offs_h[None, :],
                        h.to(tl.float32),
                        mask=mask_m[:, None] & mask_h[None, :],
                    )

                mean = (tl.sum(h.to(tl.float32), axis=1) / H)
                if MM_KIND == 1:
                    mean = mean.to(tl.bfloat16)
                else:
                    mean = mean.to(tl.float16)
                centered = h - mean[:, None]
                # Zero masked lanes in element dtype before squaring.
                centered = tl.where(mask_h[None, :], centered, centered * 0)
                sq = centered * centered
                sq_f = tl.where(mask_h[None, :], sq.to(tl.float32), 0.0)
                var = tl.sum(sq_f, axis=1) / H
                # Match PyTorch: (var + eps) rounds to element dtype before rsqrt.
                if MM_KIND == 1:
                    var = var.to(tl.bfloat16)
                    var_eps = (var.to(tl.float32) + eps).to(tl.bfloat16)
                    rstd = (1.0 / tl.sqrt(var_eps.to(tl.float32))).to(tl.bfloat16)
                else:
                    var = var.to(tl.float16)
                    var_eps = (var.to(tl.float32) + eps).to(tl.float16)
                    rstd = (1.0 / tl.sqrt(var_eps.to(tl.float32))).to(tl.float16)
                h_hat = centered * rstd[:, None]
                h_n = h_hat * ln_w[None, :] + ln_b[None, :]
                sig = (1.0 / (1.0 + tl.exp(-h_n.to(tl.float32))))
                if MM_KIND == 1:
                    sig = sig.to(tl.bfloat16)
                else:
                    sig = sig.to(tl.float16)
                h_a = h_n * sig

                if SAVE_AUX:
                    tl.store(
                        h_act_ptr + offs_m[:, None] * stride_h_b + s * stride_h_s + offs_h[None, :],
                        h_a.to(tl.float32),
                        mask=mask_m[:, None] & mask_h[None, :],
                    )

                y_s = tl.dot(h_a, tl.trans(w_out), out_dtype=tl.float32) + b_out.to(tl.float32)[None, :]
                if MM_KIND == 1:
                    y_s = y_s.to(tl.bfloat16)
                    # Match serial stage sum in bf16: acc = acc + y_s
                    acc = (acc.to(tl.bfloat16) + y_s).to(tl.float32)
                else:
                    y_s = y_s.to(tl.float16)
                    acc = (acc.to(tl.float16) + y_s).to(tl.float32)

        tl.store(
            y_ptr + offs_m[:, None] * OUT + offs_out[None, :],
            acc,
            mask=mask_m[:, None] & mask_out[None, :],
        )

    @triton.jit
    def _fused_fwd_kernel_layered(
        x_ptr,
        w_in_ptr,
        b_in_ptr,
        ln_w_ptr,
        ln_b_ptr,
        w_out_ptr,
        b_out_ptr,
        y_ptr,
        B,
        S,
        IN,
        H,
        OUT,
        stride_x_l,
        stride_x_b,
        stride_x_s,
        stride_win_l,
        stride_win_s,
        stride_wout_l,
        stride_wout_s,
        stride_bin_l,
        stride_lnw_l,
        stride_lnb_l,
        stride_bout_l,
        stride_y_l,
        eps,
        MM_KIND: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_l = tl.program_id(1)
        start = pid_m * BLOCK_M
        offs_m = start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < B

        offs_in = tl.arange(0, BLOCK_IN)
        offs_h = tl.arange(0, BLOCK_H)
        offs_out = tl.arange(0, BLOCK_OUT)
        mask_in = offs_in < IN
        mask_h = offs_h < H
        mask_out = offs_out < OUT

        acc = tl.zeros((BLOCK_M, BLOCK_OUT), dtype=tl.float32)
        x_base = x_ptr + pid_l * stride_x_l
        win_base = w_in_ptr + pid_l * stride_win_l
        wout_base = w_out_ptr + pid_l * stride_wout_l
        bin_base = b_in_ptr + pid_l * stride_bin_l
        lnw_base = ln_w_ptr + pid_l * stride_lnw_l
        lnb_base = ln_b_ptr + pid_l * stride_lnb_l
        bout_base = b_out_ptr + pid_l * stride_bout_l
        y_base = y_ptr + pid_l * stride_y_l

        for s in range(0, S):
            w_in = tl.load(
                win_base + s * stride_win_s + offs_h[:, None] * IN + offs_in[None, :],
                mask=mask_h[:, None] & mask_in[None, :],
                other=0.0,
            )
            b_in = tl.load(bin_base + s * H + offs_h, mask=mask_h, other=0.0)
            ln_w = tl.load(lnw_base + s * H + offs_h, mask=mask_h, other=0.0)
            ln_b = tl.load(lnb_base + s * H + offs_h, mask=mask_h, other=0.0)
            w_out = tl.load(
                wout_base + s * stride_wout_s + offs_out[:, None] * H + offs_h[None, :],
                mask=mask_out[:, None] & mask_h[None, :],
                other=0.0,
            )
            b_out = tl.load(bout_base + s * OUT + offs_out, mask=mask_out, other=0.0)
            x = tl.load(
                x_base + offs_m[:, None] * stride_x_b + s * stride_x_s + offs_in[None, :],
                mask=mask_m[:, None] & mask_in[None, :],
                other=0.0,
            )

            if MM_KIND == 0:
                w_in = w_in.to(tl.float32)
                w_out = w_out.to(tl.float32)
                x = x.to(tl.float32)
                h = tl.dot(x, tl.trans(w_in)) + b_in.to(tl.float32)[None, :]
                mean = tl.sum(h, axis=1) / H
                centered = h - mean[:, None]
                centered = tl.where(mask_h[None, :], centered, 0.0)
                var = tl.sum(centered * centered, axis=1) / H
                rstd = 1.0 / tl.sqrt(var + eps)
                h_hat = centered * rstd[:, None]
                h_n = h_hat * ln_w.to(tl.float32)[None, :] + ln_b.to(tl.float32)[None, :]
                sig = 1.0 / (1.0 + tl.exp(-h_n))
                h_a = h_n * sig
                y_s = tl.dot(h_a, tl.trans(w_out)) + b_out.to(tl.float32)[None, :]
                acc += y_s
            else:
                h = tl.dot(x, tl.trans(w_in), out_dtype=tl.float32) + b_in.to(tl.float32)[None, :]
                if MM_KIND == 1:
                    h = h.to(tl.bfloat16)
                else:
                    h = h.to(tl.float16)
                mean = tl.sum(h.to(tl.float32), axis=1) / H
                if MM_KIND == 1:
                    mean = mean.to(tl.bfloat16)
                else:
                    mean = mean.to(tl.float16)
                centered = h - mean[:, None]
                centered = tl.where(mask_h[None, :], centered, centered * 0)
                sq = centered * centered
                sq_f = tl.where(mask_h[None, :], sq.to(tl.float32), 0.0)
                var = tl.sum(sq_f, axis=1) / H
                if MM_KIND == 1:
                    var = var.to(tl.bfloat16)
                    var_eps = (var.to(tl.float32) + eps).to(tl.bfloat16)
                    rstd = (1.0 / tl.sqrt(var_eps.to(tl.float32))).to(tl.bfloat16)
                else:
                    var = var.to(tl.float16)
                    var_eps = (var.to(tl.float32) + eps).to(tl.float16)
                    rstd = (1.0 / tl.sqrt(var_eps.to(tl.float32))).to(tl.float16)
                h_hat = centered * rstd[:, None]
                h_n = h_hat * ln_w[None, :] + ln_b[None, :]
                sig = 1.0 / (1.0 + tl.exp(-h_n.to(tl.float32)))
                if MM_KIND == 1:
                    sig = sig.to(tl.bfloat16)
                else:
                    sig = sig.to(tl.float16)
                h_a = h_n * sig
                y_s = tl.dot(h_a, tl.trans(w_out), out_dtype=tl.float32) + b_out.to(tl.float32)[None, :]
                if MM_KIND == 1:
                    y_s = y_s.to(tl.bfloat16)
                    acc = (acc.to(tl.bfloat16) + y_s).to(tl.float32)
                else:
                    y_s = y_s.to(tl.float16)
                    acc = (acc.to(tl.float16) + y_s).to(tl.float32)

        tl.store(
            y_base + offs_m[:, None] * OUT + offs_out[None, :],
            acc,
            mask=mask_m[:, None] & mask_out[None, :],
        )


def _fused_forward_triton(
    x: Tensor,
    w_in: Tensor,
    b_in: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    w_out: Tensor,
    b_out: Tensor,
    *,
    eps: float,
    save_aux: bool,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is required for fused multistage decode forward.")
    if x.device.type != "cuda":
        raise RuntimeError("fused multistage decode Triton forward requires CUDA.")

    B, S, IN = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    H = int(w_in.shape[1])
    OUT = int(w_out.shape[1])
    if tuple(w_in.shape) != (S, H, IN):
        raise ValueError(f"w_in shape {tuple(w_in.shape)} != {(S, H, IN)}")
    if tuple(w_out.shape) != (S, OUT, H):
        raise ValueError(f"w_out shape {tuple(w_out.shape)} != {(S, OUT, H)}")

    if x.dtype != w_in.dtype or x.dtype != w_out.dtype:
        raise ValueError(
            f"fused decode matmul dtypes must match: x={x.dtype}, w_in={w_in.dtype}, w_out={w_out.dtype}"
        )
    if x.dtype == torch.bfloat16:
        mm_kind = 1
    elif x.dtype == torch.float16:
        mm_kind = 2
    else:
        mm_kind = 0

    x_c = x.contiguous()
    w_in_c = w_in.contiguous()
    b_in_c = b_in.contiguous()
    ln_w_c = ln_w.contiguous()
    ln_b_c = ln_b.contiguous()
    w_out_c = w_out.contiguous()
    b_out_c = b_out.contiguous()

    y = torch.empty((B, OUT), device=x.device, dtype=torch.float32)
    if save_aux:
        h_pre = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        h_act = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        stride_h_b = h_pre.stride(0)
        stride_h_s = h_pre.stride(1)
    else:
        h_pre = torch.empty((0,), device=x.device, dtype=torch.float32)
        h_act = torch.empty((0,), device=x.device, dtype=torch.float32)
        stride_h_b = 0
        stride_h_s = 0

    BLOCK_IN = min(_next_pow2(IN), 64)
    BLOCK_H = min(_next_pow2(H), 256)
    BLOCK_OUT = min(_next_pow2(OUT), 64)
    if BLOCK_IN < IN or BLOCK_H < H or BLOCK_OUT < OUT:
        raise ValueError(f"fused decode dims too large: IN={IN} H={H} OUT={OUT}")

    # Tile many rows per program so stage weights are reused.
    BLOCK_M = 32 if save_aux else 128
    grid = (triton.cdiv(B, BLOCK_M),)
    _fused_fwd_kernel[grid](
        x_c,
        w_in_c,
        b_in_c,
        ln_w_c,
        ln_b_c,
        w_out_c,
        b_out_c,
        y,
        h_pre,
        h_act,
        B,
        S,
        IN,
        H,
        OUT,
        x_c.stride(0),
        x_c.stride(1),
        w_in_c.stride(0),
        w_out_c.stride(0),
        stride_h_b,
        stride_h_s,
        float(eps),
        SAVE_AUX=save_aux,
        MM_KIND=mm_kind,
        BLOCK_M=BLOCK_M,
        BLOCK_IN=BLOCK_IN,
        BLOCK_H=BLOCK_H,
        BLOCK_OUT=BLOCK_OUT,
    )
    y_out = y.to(dtype=x.dtype)
    if save_aux:
        return y_out, h_pre.to(dtype=x.dtype), h_act.to(dtype=x.dtype)
    return y_out, None, None


def _fused_backward_batched(
    dy: Tensor,
    x: Tensor,
    w_in: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    w_out: Tensor,
    h_pre: Tensor,
    h_act: Tensor,
    *,
    eps: float,
    input_dtype: Optional[torch.dtype] = None,
) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One-shot batched backward across stages. dy: [B, Out].

    For bf16/fp16 inputs, LN/swish reverse uses the same element-dtype rounding
    chain as the Triton forward (mean/var/rsqrt/sigmoid cast back to input dtype).
    Matmul gradients still accumulate in fp32 then cast to the parameter dtype.
    """
    B, S, H = h_act.shape
    OUT = int(dy.shape[-1])
    IN = int(x.shape[-1])
    dtype = input_dtype if input_dtype is not None else x.dtype
    use_half = dtype in (torch.bfloat16, torch.float16)
    dy_f = dy.float()

    if use_half:
        h_pre_e = h_pre.to(dtype=dtype)
        h_act_e = h_act.to(dtype=dtype)
        ln_w_e = ln_w.to(dtype=dtype)
        ln_b_e = ln_b.to(dtype=dtype)
        w_out_e = w_out.to(dtype=dtype)
        w_in_e = w_in.to(dtype=dtype)
        x_e = x.to(dtype=dtype)

        g_w_out = (
            torch.mm(dy_f.transpose(0, 1), h_act_e.float().reshape(B, S * H))
            .view(OUT, S, H)
            .permute(1, 0, 2)
            .contiguous()
            .to(dtype=dtype)
        )
        g_b_out = dy_f.sum(dim=0).unsqueeze(0).expand(S, -1).contiguous().to(dtype=dtype)
        dh_act = torch.einsum("bo,soh->bsh", dy_f, w_out_e.float()).to(dtype=dtype)

        # Match Triton half forward LN/swish dtype chain.
        mean = (h_pre_e.float().mean(dim=-1, keepdim=True)).to(dtype=dtype)
        centered = h_pre_e - mean
        var = ((centered * centered).float().mean(dim=-1, keepdim=True)).to(dtype=dtype)
        var_eps = (var.float() + float(eps)).to(dtype=dtype)
        rstd = (torch.rsqrt(var_eps.float())).to(dtype=dtype)
        h_hat = centered * rstd
        h_n = h_hat * ln_w_e.unsqueeze(0) + ln_b_e.unsqueeze(0)
        sig = torch.sigmoid(h_n.float()).to(dtype=dtype)
        dh_n = dh_act * (sig + h_n * sig * (1.0 - sig))

        dln_w = (dh_n * h_hat).sum(dim=0)
        dln_b = dh_n.sum(dim=0)
        dh_hat = dh_n * ln_w_e.unsqueeze(0)
        sum_dh = dh_hat.sum(dim=-1, keepdim=True)
        sum_dh_hat = (dh_hat * h_hat).sum(dim=-1, keepdim=True)
        dh_pre = (dh_hat - (sum_dh + h_hat * sum_dh_hat) / H) * rstd

        g_w_in = torch.empty((S, H, IN), device=dy.device, dtype=dtype)
        gx = torch.empty((B, S, IN), device=dy.device, dtype=dtype) if x.requires_grad else None
        dh_pre_f = dh_pre.float()
        x_f = x_e.float()
        for stage_idx in range(S):
            g_w_in[stage_idx] = torch.matmul(
                dh_pre_f[:, stage_idx, :].transpose(0, 1),
                x_f[:, stage_idx, :],
            ).to(dtype=dtype)
            if gx is not None:
                gx[:, stage_idx, :] = torch.matmul(
                    dh_pre_f[:, stage_idx, :],
                    w_in_e[stage_idx].float(),
                ).to(dtype=dtype)
        g_b_in = dh_pre_f.sum(dim=0).to(dtype=dtype)
        return gx, g_w_in, g_b_in, dln_w, dln_b, g_w_out, g_b_out

    g_w_out = (
        torch.mm(dy_f.transpose(0, 1), h_act.reshape(B, S * H))
        .view(OUT, S, H)
        .permute(1, 0, 2)
        .contiguous()
    )
    g_b_out = dy_f.sum(dim=0).unsqueeze(0).expand(S, -1).contiguous()
    dh_act = torch.einsum("bo,soh->bsh", dy_f, w_out.float())

    mean = h_pre.mean(dim=-1, keepdim=True)
    var = ((h_pre - mean) ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    h_hat = (h_pre - mean) * rstd
    h_n = h_hat * ln_w.float().unsqueeze(0) + ln_b.float().unsqueeze(0)
    sig = torch.sigmoid(h_n)
    dh_n = dh_act * (sig + h_n * sig * (1.0 - sig))

    dln_w = (dh_n * h_hat).sum(dim=0)
    dln_b = dh_n.sum(dim=0)
    dh_hat = dh_n * ln_w.float().unsqueeze(0)
    sum_dh = dh_hat.sum(dim=-1, keepdim=True)
    sum_dh_hat = (dh_hat * h_hat).sum(dim=-1, keepdim=True)
    dh_pre = (dh_hat - (sum_dh + h_hat * sum_dh_hat) / H) * rstd

    g_w_in = torch.empty((S, H, IN), device=dy.device, dtype=torch.float32)
    gx = torch.empty((B, S, IN), device=dy.device, dtype=x.dtype) if x.requires_grad else None
    x_f = x.float()
    w_in_f = w_in.float()
    for stage_idx in range(S):
        g_w_in[stage_idx] = torch.matmul(dh_pre[:, stage_idx, :].transpose(0, 1), x_f[:, stage_idx, :])
        if gx is not None:
            gx[:, stage_idx, :] = torch.matmul(dh_pre[:, stage_idx, :], w_in_f[stage_idx]).to(dtype=x.dtype)
    g_b_in = dh_pre.sum(dim=0)
    return gx, g_w_in, g_b_in, dln_w, dln_b, g_w_out, g_b_out


class FusedMultistageSymmetricDecode(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        w_in: Tensor,
        b_in: Tensor,
        ln_w: Tensor,
        ln_b: Tensor,
        w_out: Tensor,
        b_out: Tensor,
        eps: float,
    ) -> Tensor:
        need_grad = any(
            bool(getattr(t, "requires_grad", False))
            for t in (x, w_in, b_in, ln_w, ln_b, w_out, b_out)
        )
        ctx.eps = float(eps)
        ctx.need_grad = need_grad

        if x.device.type == "cuda" and _TRITON_AVAILABLE:
            if need_grad:
                y, h_pre, h_act = _fused_forward_triton(
                    x, w_in, b_in, ln_w, ln_b, w_out, b_out, eps=float(eps), save_aux=True
                )
            else:
                y, _, _ = _fused_forward_triton(
                    x, w_in, b_in, ln_w, ln_b, w_out, b_out, eps=float(eps), save_aux=False
                )
        else:
            y = reference_multistage_symmetric_decode(
                x, w_in, b_in, ln_w, ln_b, w_out, b_out, eps=float(eps)
            )
            if need_grad:
                h_pre = torch.einsum("bsi,shi->bsh", x, w_in) + b_in.unsqueeze(0)
                mean = h_pre.mean(dim=-1, keepdim=True)
                var = ((h_pre - mean) ** 2).mean(dim=-1, keepdim=True)
                h_hat = (h_pre - mean) * torch.rsqrt(var + float(eps))
                h_n = h_hat * ln_w.unsqueeze(0) + ln_b.unsqueeze(0)
                h_act = _swish(h_n)

        if need_grad:
            ctx.save_for_backward(x, w_in, b_in, ln_w, ln_b, w_out, b_out, h_pre, h_act)
        else:
            ctx.save_for_backward()
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        if not bool(getattr(ctx, "need_grad", False)):
            return (None,) * 8
        x, w_in, b_in, ln_w, ln_b, w_out, b_out, h_pre, h_act = ctx.saved_tensors
        eps = float(ctx.eps)
        gx, g_w_in, g_b_in, g_ln_w, g_ln_b, g_w_out, g_b_out = _fused_backward_batched(
            dy.contiguous(),
            x,
            w_in,
            ln_w,
            ln_b,
            w_out,
            h_pre,
            h_act,
            eps=eps,
            input_dtype=x.dtype,
        )
        if not x.requires_grad:
            gx = None
        return gx, g_w_in, g_b_in, g_ln_w, g_ln_b, g_w_out, g_b_out, None


def fused_multistage_symmetric_decode(
    x: Tensor,
    w_in: Tensor,
    b_in: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    w_out: Tensor,
    b_out: Tensor,
    *,
    eps: float = 1e-5,
) -> Tensor:
    return FusedMultistageSymmetricDecode.apply(
        x, w_in, b_in, ln_w, ln_b, w_out, b_out, float(eps)
    )


def packed_symmetric_decoder_supports_fuse(decoder: nn.Module) -> bool:
    if str(getattr(decoder, "decoder_type", "")) != "symmetric":
        return False
    if int(getattr(decoder, "num_res_blocks", -1)) != 0:
        return False
    if str(getattr(decoder, "norm_type", "")).lower() != "layer":
        return False
    if str(getattr(decoder, "activation_type", "")).lower() != "swish":
        return False
    if int(getattr(decoder, "num_models", 0)) < 1:
        return False
    return hasattr(decoder, "linear_in") and hasattr(decoder, "linear_out") and hasattr(decoder, "norm_out")


def extract_packed_symmetric_stage_weights(
    decoder: nn.Module,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not packed_symmetric_decoder_supports_fuse(decoder):
        raise ValueError("decoder does not support fused multistage symmetric decode.")
    S = int(decoder.num_models)
    H = int(decoder.hidden_dim)
    IN = int(decoder.in_dim)
    OUT = int(decoder.out_dim)

    lin_in = decoder.linear_in
    lin_out = decoder.linear_out
    if int(lin_in.num_models) == 1:
        w_in = lin_in.linear.weight.unsqueeze(0)
        b_in = lin_in.linear.bias.unsqueeze(0)
    else:
        w_in = lin_in.conv.weight[:, :, 0].view(S, H, IN)
        b_in = lin_in.conv.bias.view(S, H)

    if int(lin_out.num_models) == 1:
        w_out = lin_out.linear.weight.unsqueeze(0)
        b_out = lin_out.linear.bias.unsqueeze(0)
    else:
        w_out = lin_out.conv.weight[:, :, 0].view(S, OUT, H)
        b_out = lin_out.conv.bias.view(S, OUT)

    norm = decoder.norm_out
    if int(norm.num_models) == 1:
        ln_w = norm.norm.weight.unsqueeze(0)
        ln_b = norm.norm.bias.unsqueeze(0)
    else:
        ln_w = norm.weight
        ln_b = norm.bias
    return w_in, b_in, ln_w, ln_b, w_out, b_out


def fused_decode_packed_symmetric_decoder(
    decoder: nn.Module,
    grouped_vq: Tensor,
    *,
    eps: float = 1e-5,
) -> Tensor:
    w_in, b_in, ln_w, ln_b, w_out, b_out = extract_packed_symmetric_stage_weights(decoder)
    return fused_multistage_symmetric_decode(
        grouped_vq, w_in, b_in, ln_w, ln_b, w_out, b_out, eps=eps
    )


def _fused_forward_triton_batched_layers(
    x: Tensor,
    w_in: Tensor,
    b_in: Tensor,
    ln_w: Tensor,
    ln_b: Tensor,
    w_out: Tensor,
    b_out: Tensor,
    *,
    eps: float,
) -> Tensor:
    """x: [L,B,S,IN]; weights: [L,S,...] -> y: [L,B,OUT] via one 2D-grid Triton launch."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("triton is required for batched fused decode.")
    L, B, S, IN = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
    H = int(w_in.shape[2])
    OUT = int(w_out.shape[2])
    if tuple(w_in.shape) != (L, S, H, IN):
        raise ValueError(f"w_in shape {tuple(w_in.shape)} != {(L, S, H, IN)}")
    if tuple(w_out.shape) != (L, S, OUT, H):
        raise ValueError(f"w_out shape {tuple(w_out.shape)} != {(L, S, OUT, H)}")

    if x.dtype == torch.bfloat16:
        mm_kind = 1
    elif x.dtype == torch.float16:
        mm_kind = 2
    else:
        mm_kind = 0

    x_c = x.contiguous()
    w_in_c = w_in.contiguous()
    b_in_c = b_in.contiguous()
    ln_w_c = ln_w.contiguous()
    ln_b_c = ln_b.contiguous()
    w_out_c = w_out.contiguous()
    b_out_c = b_out.contiguous()
    y = torch.empty((L, B, OUT), device=x.device, dtype=torch.float32)

    BLOCK_IN = min(_next_pow2(IN), 64)
    BLOCK_H = min(_next_pow2(H), 256)
    BLOCK_OUT = min(_next_pow2(OUT), 64)
    if BLOCK_IN < IN or BLOCK_H < H or BLOCK_OUT < OUT:
        raise ValueError(f"fused decode dims too large: IN={IN} H={H} OUT={OUT}")
    BLOCK_M = 128
    grid = (triton.cdiv(B, BLOCK_M), L)
    _fused_fwd_kernel_layered[grid](
        x_c,
        w_in_c,
        b_in_c,
        ln_w_c,
        ln_b_c,
        w_out_c,
        b_out_c,
        y,
        B,
        S,
        IN,
        H,
        OUT,
        x_c.stride(0),
        x_c.stride(1),
        x_c.stride(2),
        w_in_c.stride(0),
        w_in_c.stride(1),
        w_out_c.stride(0),
        w_out_c.stride(1),
        b_in_c.stride(0),
        ln_w_c.stride(0),
        ln_b_c.stride(0),
        b_out_c.stride(0),
        y.stride(0),
        float(eps),
        MM_KIND=mm_kind,
        BLOCK_M=BLOCK_M,
        BLOCK_IN=BLOCK_IN,
        BLOCK_H=BLOCK_H,
        BLOCK_OUT=BLOCK_OUT,
    )
    return y.to(dtype=x.dtype)


@torch.no_grad()
def fused_decode_batched_same_shape_decoders(
    decoders: Sequence[nn.Module],
    grouped_vqs: Sequence[Tensor],
    *,
    eps: float = 1e-5,
) -> List[Tensor]:
    """Batched no-grad fuse for same-shaped packed symmetric decoders.

    Stacks layer weights once and runs fused Triton decode for each layer with
    shared launch setup. Falls back to per-layer ``fused_decode_packed_symmetric_decoder``
    when shapes diverge or CUDA/Triton is unavailable.
    """
    if not decoders:
        return []
    if len(decoders) != len(grouped_vqs):
        raise ValueError(
            f"decoders/grouped_vqs length mismatch: {len(decoders)} vs {len(grouped_vqs)}"
        )
    if len(decoders) == 1:
        return [fused_decode_packed_symmetric_decoder(decoders[0], grouped_vqs[0], eps=eps)]

    for decoder in decoders:
        if not packed_symmetric_decoder_supports_fuse(decoder):
            return [
                fused_decode_packed_symmetric_decoder(dec, vq, eps=eps)
                for dec, vq in zip(decoders, grouped_vqs)
            ]

    weight_packs = [extract_packed_symmetric_stage_weights(decoder) for decoder in decoders]
    first_vq = grouped_vqs[0]
    B, S, IN = int(first_vq.shape[0]), int(first_vq.shape[1]), int(first_vq.shape[2])
    device = first_vq.device
    dtype = first_vq.dtype
    for vq in grouped_vqs[1:]:
        if tuple(vq.shape) != (B, S, IN) or vq.device != device or vq.dtype != dtype:
            return [
                fused_decode_packed_symmetric_decoder(dec, vq, eps=eps)
                for dec, vq in zip(decoders, grouped_vqs)
            ]

    w_in = torch.stack([w[0] for w in weight_packs], dim=0)
    b_in = torch.stack([w[1] for w in weight_packs], dim=0)
    ln_w = torch.stack([w[2] for w in weight_packs], dim=0)
    ln_b = torch.stack([w[3] for w in weight_packs], dim=0)
    w_out = torch.stack([w[4] for w in weight_packs], dim=0)
    b_out = torch.stack([w[5] for w in weight_packs], dim=0)
    x = torch.stack(list(grouped_vqs), dim=0)

    if device.type != "cuda" or not _TRITON_AVAILABLE:
        ys = []
        for i in range(len(decoders)):
            ys.append(
                fused_multistage_symmetric_decode(
                    grouped_vqs[i],
                    weight_packs[i][0],
                    weight_packs[i][1],
                    weight_packs[i][2],
                    weight_packs[i][3],
                    weight_packs[i][4],
                    weight_packs[i][5],
                    eps=eps,
                )
            )
        return ys

    y = _fused_forward_triton_batched_layers(
        x, w_in, b_in, ln_w, ln_b, w_out, b_out, eps=float(eps)
    )
    return [y[i].contiguous() for i in range(int(y.shape[0]))]
