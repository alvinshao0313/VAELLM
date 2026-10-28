"""Fixed row/column scales in canonical rotated [output,input] coordinates."""
from __future__ import annotations

import torch
from torch import nn


class BiaxialScale(nn.Module):
    def __init__(self, row: torch.Tensor, column: torch.Tensor):
        super().__init__()
        if (row.ndim != 1 or column.ndim != 1 or row.dtype != torch.bfloat16
                or column.dtype != torch.bfloat16 or not row.numel() or not column.numel()
                or not bool(torch.isfinite(row).all() and torch.isfinite(column).all())
                or not bool((row > 0).all() and (column > 0).all())):
            raise ValueError('Biaxial scales must be finite positive BF16 vectors.')
        self.register_buffer('row', row.detach().clone())
        self.register_buffer('column', column.detach().clone())

    @classmethod
    def from_rotated(cls, weight: torch.Tensor):
        value = weight.float()
        if value.ndim != 2 or not bool(torch.isfinite(value).all()):
            raise ValueError('Scale initialization needs a finite rotated matrix.')
        rms = value.square().mean().sqrt()
        if not bool(rms > 0):
            raise ValueError('An all-zero matrix has no scale normalization target.')
        row = value.square().mean(dim=1).sqrt() / rms
        row = torch.where(row > 0, row, torch.ones_like(row))
        # Compute the specified sequential column statistic before storage rounding.
        first = value / row[:, None]
        column = first.square().mean(dim=0).sqrt() / rms
        column = torch.where(column > 0, column, torch.ones_like(column))
        return cls(row.to(torch.bfloat16), column.to(torch.bfloat16))

    def normalize(self, weight: torch.Tensor) -> torch.Tensor:
        self._check_shape(weight)
        return weight.float() / self.row.float()[:, None] / self.column.float()[None, :]

    def restore(self, weight: torch.Tensor) -> torch.Tensor:
        self._check_shape(weight)
        return weight.float() * self.row.float()[:, None] * self.column.float()[None, :]

    def _check_shape(self, weight):
        if tuple(weight.shape) != (self.row.numel(), self.column.numel()):
            raise ValueError('Scale/rotated matrix shape mismatch.')

    def block_factors(self, indices: torch.Tensor, *, transpose: bool, dim: int) -> torch.Tensor:
        """Match native row-major vectors after the optional canonical transpose."""
        return block_scale_factors(self.row.float(), self.column.float(), indices, transpose=transpose, dim=dim)


def block_scale_factors(row_values, column_values, indices, *, transpose, dim):
    rows, columns = row_values.numel(), column_values.numel()
    flat = indices[:, None] * dim + torch.arange(dim, device=indices.device)[None, :]
    if transpose:
        row, column = flat.remainder(rows), torch.div(flat, rows, rounding_mode='floor')
    else:
        row, column = torch.div(flat, columns, rounding_mode='floor'), flat.remainder(columns)
    return (row_values[row] * column_values[column]).unsqueeze(1)


def weighted_reconstruction_loss(decoded, target, factors, normalized_std, physical_std):
    """Native BF16 targets, mapped to physical rotated coordinates before MSE."""
    weights = factors * normalized_std.reshape(1, -1, 1) / physical_std.reshape(1, -1, 1)
    return ((decoded.float() - target.float()) * weights).square().mean()
