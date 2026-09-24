"""Local weight-space RHT: compress U W V.T, restore U.T W_hat V.

The transform acts only on the unprotected matrix. No activation hooks or
network-wide change of basis are installed. Sign buffers are checkpoint state.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Optional

import torch
from torch import nn


def rotation_spec(mode: str, block_size: int, seed: int, module_name: str) -> Optional[dict]:
    if mode == "none":
        return None
    if mode != "two_sided":
        raise ValueError(f"Unknown weight_rotation={mode!r}.")
    digest = hashlib.sha256(f"{int(seed)}:{module_name}".encode()).digest()
    return {
        "type": "two_sided_hadamard",
        "version": 1,
        "block_size": int(block_size),
        "seed": int.from_bytes(digest[:8], "little") % (2**63),
    }


def validate_rotation_spec(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping):
        raise ValueError("weight_rotation must be a mapping.")
    if set(spec) != {"type", "version", "block_size", "seed"}:
        raise ValueError("Incomplete or unknown weight_rotation metadata fields.")
    if spec["type"] != "two_sided_hadamard" or spec["version"] != 1:
        raise ValueError("Unsupported weight_rotation type/version.")
    for key in ("block_size", "seed"):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < 0:
            raise ValueError(f"weight_rotation.{key} must be a nonnegative integer.")
    if spec["seed"] >= 2**63:
        raise ValueError("weight_rotation.seed must be < 2**63.")


def _hadamard_factor(n: int) -> tuple[Optional[torch.Tensor], int]:
    # Reuse the repository's QuIP# Hadamard factors, including non-power-of-two
    # dimensions. Unsupported dimensions are rejected, never padded/truncated.
    from rotation.hadamard_utils import get_hadK

    if n < 1:
        raise ValueError(f"Hadamard dimension must be positive, got {n}.")
    try:
        had, factor = get_hadK(n)
    except AssertionError as exc:
        raise ValueError(
            f"No exact Hadamard factor for dimension {n}. Use an explicit block "
            "size dividing both unprotected dimensions (e.g. 32); full mode "
            "does not pad or silently switch to block rotation."
        ) from exc
    return had, int(factor)


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester transform on the last axis; differentiable on CPU."""
    n = x.shape[-1]
    if n == 1:
        return x
    if x.is_cuda and x.dtype != torch.float64:
        from fast_hadamard_transform import hadamard_transform

        return hadamard_transform(x.contiguous(), scale=1.0 / math.sqrt(n))
    h = 1
    shape = x.shape
    while h < n:
        blocks = x.reshape(*shape[:-1], n // (2 * h), 2, h)
        a, b = blocks.unbind(-2)
        x = torch.stack((a + b, a - b), dim=-2).reshape(shape)
        h *= 2
    return x / math.sqrt(n)


def _transform_last(
    x: torch.Tensor,
    *,
    block_size: int,
    factor: int,
    had: Optional[torch.Tensor],
    inverse: bool,
) -> torch.Tensor:
    shape = x.shape
    # Output is H x in column-vector notation, including the nonsymmetric
    # non-Sylvester factor. Its inverse MUST use H.T, not H.
    z = x.reshape(-1, factor, block_size // factor)
    z = _fwht(z)
    if had is not None:
        h = had.to(device=z.device, dtype=z.dtype)
        if inverse:
            h = h.T
        z = torch.matmul(h, z) / math.sqrt(factor)
    return z.reshape(shape)


class TwoSidedHadamard(nn.Module):
    def __init__(self, out_features: int, in_features: int, spec: Mapping[str, Any]):
        super().__init__()
        validate_rotation_spec(spec)
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.block_size = int(spec["block_size"])
        self.seed = int(spec["seed"])
        self.output_block = self.block_size or self.out_features
        self.input_block = self.block_size or self.in_features
        for size, block in ((self.out_features, self.output_block), (self.in_features, self.input_block)):
            if size < 1 or block < 1 or size % block:
                raise ValueError(
                    f"weight_rotation block_size={self.block_size} must divide "
                    f"unprotected shape ({self.out_features}, {self.in_features})."
                )
        out_had, self.output_factor = _hadamard_factor(self.output_block)
        in_had, self.input_factor = _hadamard_factor(self.input_block)
        self.register_buffer("output_had", out_had, persistent=False)
        self.register_buffer("input_had", in_had, persistent=False)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.register_buffer(
            "output_signs", torch.randint(0, 2, (self.out_features,), generator=generator, dtype=torch.int8) * 2 - 1
        )
        self.register_buffer(
            "input_signs", torch.randint(0, 2, (self.in_features,), generator=generator, dtype=torch.int8) * 2 - 1
        )

    def to_spec(self) -> dict:
        return {"type": "two_sided_hadamard", "version": 1, "block_size": self.block_size, "seed": self.seed}

    def validate_state(self) -> None:
        for key, size in (("output_signs", self.out_features), ("input_signs", self.in_features)):
            signs = getattr(self, key)
            if signs.dtype != torch.int8 or tuple(signs.shape) != (size,):
                raise ValueError(f"Invalid weight_rotation.{key} shape/dtype.")
            if not bool(((signs == 1) | (signs == -1)).all()):
                raise ValueError(f"weight_rotation.{key} contains values other than +/-1.")

    def forward(self, weight: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"weight_rotation shape mismatch: {tuple(weight.shape)}.")
        original_dtype = weight.dtype
        # Do not let outer BF16 autocast round the small-factor matmuls.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            w = weight if original_dtype == torch.float64 else weight.float()
            sin = self.input_signs.to(device=w.device, dtype=w.dtype)
            sout = self.output_signs.to(device=w.device, dtype=w.dtype)
            if inverse:
                w = _transform_last(w, block_size=self.input_block, factor=self.input_factor,
                                    had=self.input_had, inverse=True) * sin
                w = _transform_last(w.T, block_size=self.output_block, factor=self.output_factor,
                                    had=self.output_had, inverse=True).T * sout[:, None]
            else:
                w = _transform_last(w * sin, block_size=self.input_block, factor=self.input_factor,
                                    had=self.input_had, inverse=False)
                w = _transform_last((w * sout[:, None]).T, block_size=self.output_block,
                                    factor=self.output_factor, had=self.output_had, inverse=False).T
        return w.to(dtype=original_dtype).contiguous()


def build_weight_rotation(
    out_features: int, in_features: int, spec: Optional[Mapping[str, Any]]
) -> Optional[TwoSidedHadamard]:
    return None if spec is None else TwoSidedHadamard(out_features, in_features, spec)
