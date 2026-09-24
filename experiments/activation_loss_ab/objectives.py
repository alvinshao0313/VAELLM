"""Input-axis, non-transposed 32-weight output-reconstruction objective.

The matrix is the uncentered second moment E[x x.T], not covariance.
One full, unsplit Linear weight is flattened row-major. Its input width
must be divisible by block_dim. No channel removal or rotation is implicit.
"""
from __future__ import annotations

import torch


def block_gram_sum(inputs: torch.Tensor, block_dim: int = 32) -> torch.Tensor:
    if inputs.ndim != 2 or inputs.shape[1] % block_dim:
        raise ValueError("Expected [tokens, inputs] with input width divisible by block_dim.")
    x = inputs.detach().float().reshape(inputs.shape[0], -1, block_dim)
    return torch.einsum("tgi,tgj->gij", x, x)


def gather_grams(grams: torch.Tensor, block_indices: torch.Tensor) -> torch.Tensor:
    if grams.ndim != 3 or grams.shape[1] != grams.shape[2] or grams.shape[0] < 1:
        raise ValueError("Expected [input_groups, block_dim, block_dim] second moments.")
    # Different output rows reuse the same input-channel groups.
    ids = block_indices.to(device=grams.device, dtype=torch.long) % grams.shape[0]
    return grams.index_select(0, ids)


def block_output_mse(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    grams: torch.Tensor,
    block_indices: torch.Tensor,
) -> torch.Tensor:
    if reconstructed.shape != target.shape or target.ndim != 3 or target.shape[1] != 1:
        raise ValueError("This experiment supports one full matrix: [blocks, 1, block_dim].")
    if target.shape[0] != block_indices.numel() or target.shape[2] != grams.shape[-1]:
        raise ValueError("Block-index/second-moment dimensions do not match the batch.")
    error = (reconstructed.float() - target.float()).squeeze(1)
    selected = gather_grams(grams, block_indices)
    quadratic = torch.bmm(error.unsqueeze(1), torch.bmm(selected, error.unsqueeze(2)))
    # /D makes the diagonal case exactly the existing elementwise AMSE,
    # rather than introducing an extra factor D against the BSQ auxiliary loss.
    return quadratic.mean() / error.shape[-1]
