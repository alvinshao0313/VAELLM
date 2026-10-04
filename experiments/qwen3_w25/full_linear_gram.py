"""Isolated full-coordinate Gram diagnostic for a fixed-input Linear loss.

This module is not connected to the packed-code optimizer. Algebraic equivalence
does not imply identical floating-point gradients or hard-code trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FullLinearGram:
    """Temporary, detached raw X.T @ X and its valid-token count, not model state."""

    matrix: torch.Tensor
    token_count: int


def _check_compute_dtype(dtype: torch.dtype, device: torch.device) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Full Linear Gram requires explicit float32 or float64 computation.")
    if device.type == "cuda" and dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable CUDA matmul TF32 explicitly before using the float32 Gram diagnostic.")


def build_full_linear_gram(inputs: torch.Tensor, *, dtype: torch.dtype) -> FullLinearGram:
    """Build unnormalized G from every valid token and every original input column.

    The caller supplies the same already-selected valid tokens as the direct
    output objective. No input or protected columns are sampled or removed.
    """
    _check_compute_dtype(dtype, inputs.device)
    if inputs.ndim != 2 or not inputs.shape[0] or not inputs.shape[1]:
        raise ValueError("Expected nonempty inputs[valid_tokens, original_in_features].")
    if not inputs.is_floating_point():
        raise ValueError("Gram inputs must be floating point.")
    with torch.no_grad(), torch.autocast(device_type=inputs.device.type, enabled=False):
        x = inputs.detach().to(dtype=dtype)
        matrix = x.T @ x
    return FullLinearGram(matrix=matrix, token_count=int(inputs.shape[0]))


def full_linear_gram_mse(error_weight: torch.Tensor, gram: FullLinearGram) -> torch.Tensor:
    """Compute trace(E G E.T) / (valid_tokens * out_features) with autograd.

    E contains the entire original-coordinate weight error, including frozen
    INT8 protection errors. Autograd retains their cross terms in the trainable
    columns' gradient. It also differentiates the actual stored G without
    assuming bitwise symmetry. No PSD correction or loss clipping is applied.
    """
    matrix = gram.matrix
    _check_compute_dtype(matrix.dtype, matrix.device)
    if (error_weight.ndim != 2 or not error_weight.shape[0] or not error_weight.shape[1]
            or matrix.ndim != 2 or matrix.shape != (error_weight.shape[1], error_weight.shape[1])):
        raise ValueError("Expected error_weight[out, original_in] and a matching full square Gram.")
    if not isinstance(gram.token_count, int) or isinstance(gram.token_count, bool) or gram.token_count <= 0:
        raise ValueError("Gram token_count must be a positive integer.")
    if matrix.requires_grad or matrix.grad_fn is not None:
        raise ValueError("Gram must be detached fixed-input state.")
    if not error_weight.is_floating_point() or error_weight.device != matrix.device:
        raise ValueError("Floating-point error_weight and Gram must be on the same device.")
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        error = error_weight.to(dtype=matrix.dtype)
        return ((error @ matrix) * error).sum() / (gram.token_count * error.shape[0])
