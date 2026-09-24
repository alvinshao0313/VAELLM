"""Reference and optional Triton kernels for full Linear output alignment.

The public API works with row-major 32-channel blocks.  ``decoded_blocks`` and
``target_blocks`` can be either ``[out, nblocks, 32]`` or the packed
``[out*nblocks, 1, 32]`` layout used by the VAE.  Inputs are ``[tokens, in]``.
The loss is computed after summing every input block:

    mean((sum_k X[:, k, :] @ (W_hat-W)[:, k, :].T) ** 2)

The Triton path only runs on CUDA.  CPU and unsupported dtypes use the exact
PyTorch reference path.  The custom autograd wrapper stores the output error
for a cheap backward through the decoded blocks; this is deliberate because
the decoded blocks are the VAE's differentiable output.
"""
from __future__ import annotations

from typing import Tuple

import torch

try:  # Triton is optional for importability and CPU unit tests.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only on installations without Triton
    triton = None
    tl = None


_BLOCK_SIZE = 32


def _as_blocked(value: torch.Tensor, *, out_features: int | None = None) -> torch.Tensor:
    """Return a contiguous ``[out, nblocks, 32]`` view."""
    if value.ndim == 3 and value.shape[-1] == _BLOCK_SIZE:
        if value.shape[1] == 1:
            if out_features is None:
                raise ValueError("Packed [vectors,1,32] needs out_features.")
            if value.shape[0] % out_features:
                raise ValueError("Packed vector count is not divisible by out_features.")
            return value.reshape(out_features, value.shape[0] // out_features, _BLOCK_SIZE).contiguous()
        return value.contiguous()
    if value.ndim == 2 and value.shape[1] % _BLOCK_SIZE == 0:
        return value.reshape(value.shape[0], value.shape[1] // _BLOCK_SIZE, _BLOCK_SIZE).contiguous()
    raise ValueError("Expected [out,in], [out,nblocks,32], or [vectors,1,32].")


def _validate(decoded: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if inputs.ndim != 2:
        raise ValueError("inputs must be [valid_tokens,in_features].")
    if decoded.ndim == 3 and decoded.shape[1] == 1:
        if target.ndim == 2:
            out_hint = target.shape[0]
        elif target.ndim == 3 and target.shape[1] != 1:
            out_hint = target.shape[0]
        else:
            nblocks = inputs.shape[1] // _BLOCK_SIZE
            if nblocks < 1 or decoded.shape[0] % nblocks:
                raise ValueError("Packed vector count does not match input block count.")
            out_hint = decoded.shape[0] // nblocks
        decoded_b = _as_blocked(decoded, out_features=out_hint)
    else:
        decoded_b = _as_blocked(decoded)
    target_b = _as_blocked(target, out_features=decoded_b.shape[0])
    if decoded_b.shape != target_b.shape:
        raise ValueError(f"decoded/target shape mismatch: {decoded_b.shape} vs {target_b.shape}")
    out_features, nblocks, width = decoded_b.shape
    if width != _BLOCK_SIZE or inputs.shape[1] != nblocks * _BLOCK_SIZE:
        raise ValueError("Input width must match all 32-channel blocks.")
    if inputs.shape[0] == 0:
        raise ValueError("At least one valid token is required.")
    return decoded_b, target_b, inputs.contiguous()


def output_error_reference(decoded: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Exact PyTorch reference error ``X @ (W_hat-W).T``."""
    decoded_b, target_b, inputs = _validate(decoded, target, inputs)
    x = inputs.reshape(inputs.shape[0], -1, _BLOCK_SIZE).float()
    delta = decoded_b.float() - target_b.float()
    return torch.einsum("tki,oki->to", x, delta)


if triton is not None:

    @triton.jit
    def _output_error_kernel(
        x_ptr, delta_ptr, error_ptr,
        n_tokens, n_outputs, n_inputs,
        stride_xt, stride_dt, stride_et, stride_eo,
        BLOCK_T: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_I: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_o = tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        mask_t = offs_t < n_tokens
        mask_o = offs_o < n_outputs
        acc = tl.zeros((BLOCK_T, BLOCK_O), dtype=tl.float32)
        for i in tl.range(0, n_inputs, BLOCK_I):
            offs_i = i + tl.arange(0, BLOCK_I)
            mask_i = offs_i < n_inputs
            x = tl.load(x_ptr + offs_t[:, None] * stride_xt + offs_i[None, :],
                        mask=mask_t[:, None] & mask_i[None, :], other=0.0)
            d = tl.load(delta_ptr + offs_o[:, None] * stride_dt + offs_i[None, :],
                        mask=mask_o[:, None] & mask_i[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(d), input_precision="ieee")
        tl.store(error_ptr + offs_t[:, None] * stride_et + offs_o[None, :] * stride_eo,
                 acc, mask=mask_t[:, None] & mask_o[None, :])


def output_error_triton(decoded: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute output error with a Triton tiled kernel, or exact fallback."""
    decoded_b, target_b, inputs = _validate(decoded, target, inputs)
    if triton is None or not inputs.is_cuda or decoded_b.device != inputs.device:
        return output_error_reference(decoded_b, target_b, inputs)
    delta = (decoded_b - target_b).reshape(decoded_b.shape[0], -1).contiguous()
    x = inputs.contiguous()
    # Triton tl.dot requires identical operand dtypes. Keep the reference
    # semantics (fp32 accumulation over physical weights) when teacher inputs
    # are bf16 and the original target weights are fp32.
    if x.dtype != delta.dtype:
        x = x.float()
        delta = delta.float()
    # cuBLAS GEMM is substantially faster than the generic Triton tile for
    # the wide Qwen projections while computing the same full X @ delta.T
    # objective. Keep Triton for small/irregular matrices and as a portable
    # reference for kernel checks.
    if x.shape[1] >= 1024 and x.shape[0] >= 1024 and delta.shape[0] >= 1024:
        return torch.mm(x, delta.t())
    error = torch.empty((x.shape[0], delta.shape[0]), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(x.shape[0], 64), triton.cdiv(delta.shape[0], 64))
    _output_error_kernel[grid](
        x, delta, error, x.shape[0], delta.shape[0], x.shape[1],
        x.stride(0), delta.stride(0), error.stride(0), error.stride(1),
        BLOCK_T=64, BLOCK_O=64, BLOCK_I=128,
    )
    return error


class _OutputMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decoded: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor, use_triton: bool):
        decoded_b, target_b, inputs = _validate(decoded, target, inputs)
        error = output_error_triton(decoded_b, target_b, inputs) if use_triton else output_error_reference(decoded_b, target_b, inputs)
        ctx.save_for_backward(inputs, error)
        ctx.decoded_shape = decoded.shape
        ctx.target_shape = target.shape
        ctx.block_shape = decoded_b.shape
        return error.square().mean()

    @staticmethod
    def backward(ctx, grad_output):
        inputs, error = ctx.saved_tensors
        x = inputs.reshape(inputs.shape[0], -1, _BLOCK_SIZE).float()
        # The flattened GEMM is equivalent to the einsum and uses the
        # accelerator's tuned matmul path for wide projections.
        grad_delta = torch.mm(error.t(), x.reshape(x.shape[0], -1)).reshape(ctx.block_shape)
        grad_delta = grad_delta * (2.0 / (error.shape[0] * error.shape[1])) * grad_output
        return grad_delta.reshape(ctx.block_shape).reshape(ctx.decoded_shape), None, None, None


def output_mse(
    decoded: torch.Tensor,
    target: torch.Tensor,
    inputs: torch.Tensor,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Differentiable full-Linear output MSE.

    ``use_triton`` requests the CUDA kernel; unsupported devices/dtypes fall
    back to the reference implementation without changing the result.
    """
    return _OutputMSE.apply(decoded, target, inputs, bool(use_triton and triton is not None))


__all__ = ["output_error_reference", "output_error_triton", "output_mse"]
