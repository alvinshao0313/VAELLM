"""Fixed-decoder, two-stage single-bit search with block LDL error feedback.

This is a finite-neighborhood oracle, not the exact E8P quantizer from QuIP#.
Hessians and factors are temporary FP64 state. The caller owns model-specific
rotation, packed storage, native forward measurement, and checkpoint export.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


@dataclass(frozen=True)
class ProtectedObjective:
    target: torch.Tensor
    hessian: torch.Tensor
    damping: float


@dataclass(frozen=True)
class BlockLDLFactors:
    lower: torch.Tensor
    diagonal: torch.Tensor


@dataclass(frozen=True)
class BlockLDLQResult:
    codes: torch.Tensor
    reconstructed_weight: torch.Tensor
    statistics: dict


StageDecoder = Callable[[int, torch.Tensor], torch.Tensor]


def _fp64_matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or not all(value.shape) or value.dtype != torch.float64:
        raise ValueError(f"{name} must be a nonempty FP64 matrix.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values.")


def _symmetric_hessian(value: torch.Tensor) -> None:
    _fp64_matrix(value, "hessian")
    if value.shape[0] != value.shape[1]:
        raise ValueError("hessian must be square.")
    scale = float(value.diagonal().abs().max())
    if not torch.allclose(value, value.T, rtol=1e-12, atol=1e-12 * scale):
        raise ValueError("hessian must be symmetric within FP64 roundoff.")


@torch.no_grad()
def build_noncentered_hessian(
    inputs: torch.Tensor, *, chunk_size: int = 1024, device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return E[x x.T] over every supplied valid token, without centering.

    Chunks control temporary conversion memory only; the caller must filter
    padding and select the agreed training documents before this function.
    """
    if inputs.ndim != 2 or not all(inputs.shape) or not inputs.is_floating_point():
        raise ValueError("inputs must be floating point [valid_tokens, input_features].")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    device = inputs.device if device is None else torch.device(device)
    hessian = torch.zeros((inputs.shape[1], inputs.shape[1]), device=device, dtype=torch.float64)
    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(inputs), chunk_size):
            x = inputs[start:start + chunk_size].to(device=device, dtype=torch.float64)
            hessian.addmm_(x.T, x)
        hessian.div_(len(inputs))
    _symmetric_hessian(hessian)
    return hessian


@torch.no_grad()
def prepare_protected_objective(
    weight: torch.Tensor,
    quantized_protected: torch.Tensor,
    hessian: torch.Tensor,
    compressed_indices: torch.Tensor,
    protected_indices: torch.Tensor,
    *,
    damping_ratio: float = 0.01,
) -> ProtectedObjective:
    """Complete the square while preserving the fixed protection cross term.

    With E_P = W_P - Q_P and A = H_CC + lambda I, the variable part of
    tr(E H E.T) + lambda ||E_C||² is
    tr((Wtilde_C-Q_C) A (Wtilde_C-Q_C).T), up to a constant, where
    Wtilde_C = W_C + E_P H_PC A^{-1}. A solve is used, never an inverse.
    Q_P must contain the actual deployed protected values in P's given order.
    """
    _fp64_matrix(weight, "weight")
    _symmetric_hessian(hessian)
    if hessian.shape != (weight.shape[1], weight.shape[1]) or hessian.device != weight.device:
        raise ValueError("weight and hessian must share the full input axis and device.")
    if not math.isfinite(damping_ratio) or damping_ratio <= 0:
        raise ValueError("damping_ratio must be finite and positive.")
    for name, indices in (("compressed_indices", compressed_indices),
                          ("protected_indices", protected_indices)):
        if indices.ndim != 1 or indices.dtype != torch.long or indices.device != weight.device:
            raise ValueError(f"{name} must be a same-device one-dimensional int64 tensor.")
    partition = torch.cat((compressed_indices, protected_indices)).sort().values
    if not len(compressed_indices) or not torch.equal(
        partition, torch.arange(weight.shape[1], device=weight.device)
    ):
        raise ValueError("compressed/protected indices must partition the original input axis exactly.")
    if (quantized_protected.shape != (weight.shape[0], len(protected_indices))
            or quantized_protected.dtype != torch.float64 or quantized_protected.device != weight.device
            or not bool(torch.isfinite(quantized_protected).all())):
        raise ValueError("quantized_protected must be finite same-device FP64 [output_features, P].")
    with torch.autocast(device_type=weight.device.type, enabled=False):
        a = hessian.index_select(0, compressed_indices).index_select(1, compressed_indices)
        damping = damping_ratio * float(a.diagonal().mean())
        if not math.isfinite(damping) or damping <= 0:
            raise ValueError("H_CC must have a positive finite mean diagonal for relative damping.")
        a.diagonal().add_(damping)
        target = weight.index_select(1, compressed_indices)
        if len(protected_indices):
            e_p = weight.index_select(1, protected_indices) - quantized_protected
            h_pc = hessian.index_select(0, protected_indices).index_select(1, compressed_indices)
            cross = e_p @ h_pc
            chol = torch.linalg.cholesky(a)
            target = target + torch.cholesky_solve(cross.T.contiguous(), chol).T
    return ProtectedObjective(target=target, hessian=a, damping=damping)


@torch.no_grad()
def block_ldl(hessian: torch.Tensor, block_size: int = 16) -> BlockLDLFactors:
    """Factor H = L D L.T with block-unit-lower L and SPD diagonal blocks.

    Starting from Cholesky C, normalize each column block by its C_jj on
    the right: L_ij = C_ij C_jj^{-1}, D_j = C_jj C_jj.T. No pivoting,
    diagonal fallback, added jitter, or eigenvalue clipping is performed.
    """
    _symmetric_hessian(hessian)
    if type(block_size) is not int or block_size < 1 or hessian.shape[0] % block_size:
        raise ValueError("block_size must be a positive integer dividing the Hessian width.")
    with torch.autocast(device_type=hessian.device.type, enabled=False):
        chol = torch.linalg.cholesky(hessian)
        lower = torch.zeros_like(chol)
        diagonal = []
        for start in range(0, len(hessian), block_size):
            end = start + block_size
            c_jj = chol[start:end, start:end]
            diagonal.append(c_jj @ c_jj.T)
            lower[start:, start:end] = torch.linalg.solve_triangular(
                c_jj, chol[start:, start:end], upper=False, left=False)
    return BlockLDLFactors(lower=lower, diagonal=torch.stack(diagonal))


@torch.no_grad()
def two_stage_single_bit_candidates(codes: torch.Tensor, decode_stage: StageDecoder) -> torch.Tensor:
    """Jointly decode current + K stage-0 flips + K stage-1 flips.

    codes has shape [2, rows, K]. The decoder receives bool [batch, K]
    and returns floating [batch, D]. Both outputs are added in their returned
    dtype, preserving the native two-stage sum's rounding before FP64 scoring.
    Only one stage changes per candidate; there are 1+2K, not (1+K)² choices.
    """
    if codes.ndim != 3 or codes.shape[0] != 2 or not all(codes.shape) or codes.dtype != torch.bool:
        raise ValueError("codes must be nonempty bool [2, rows, K].")
    rows, bits = codes.shape[1:]
    decoded = []
    positions = torch.arange(bits, device=codes.device)
    for stage in range(2):
        neighbors = codes[stage, :, None, :].expand(rows, bits + 1, bits).clone()
        neighbors[:, positions + 1, positions] = ~neighbors[:, positions + 1, positions]
        values = decode_stage(stage, neighbors.reshape(-1, bits))
        if (values.ndim != 2 or values.shape[0] != rows * (bits + 1)
                or not values.shape[1] or not values.is_floating_point()
                or values.device != codes.device or not bool(torch.isfinite(values).all())):
            raise ValueError("decode_stage must return finite same-device floating [batch, D].")
        decoded.append(values.reshape(rows, bits + 1, -1))
    if decoded[0].shape != decoded[1].shape or decoded[0].dtype != decoded[1].dtype:
        raise ValueError("The two stage decoders must return matching shapes and dtypes.")
    return torch.cat((decoded[0] + decoded[1][:, :1],
                      decoded[0][:, :1] + decoded[1][:, 1:]), dim=1)


@torch.no_grad()
def quantize_reverse_blocks(
    weight: torch.Tensor,
    factors: BlockLDLFactors,
    codes: torch.Tensor,
    decode_stage: StageDecoder,
    *,
    feedback: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> BlockLDLQResult:
    """Visit every input block once, in reverse, without mutating inputs.

    weight is the already compensated/rotated FP64 target [out, in]. codes
    is bool [2, out, blocks, K]. On and off arms share the finite candidate
    oracle and D metric. Off removes only the feedback correction.

    The correction uses original residuals W_k-Q_k, not corrected T_k-Q_k.
    Current is the first candidate and wins exact ties. These local decisions
    do not imply monotonic full-H or held-out error; measure both separately.
    """
    _fp64_matrix(weight, "weight")
    if type(feedback) is not bool:
        raise ValueError("feedback must be a boolean.")
    lower, diagonal = factors.lower, factors.diagonal
    if (lower.shape != (weight.shape[1], weight.shape[1]) or lower.dtype != torch.float64
            or lower.device != weight.device or diagonal.ndim != 3
            or not all(diagonal.shape) or diagonal.shape[1] != diagonal.shape[2]
            or diagonal.dtype != torch.float64 or diagonal.device != weight.device
            or diagonal.shape[0] * diagonal.shape[1] != weight.shape[1]):
        raise ValueError("FP64 LDL factors must match the weight's full input axis and device.")
    blocks, block_size = diagonal.shape[:2]
    if (codes.ndim != 4 or codes.shape[:3] != (2, weight.shape[0], blocks)
            or not codes.shape[3] or codes.dtype != torch.bool or codes.device != weight.device):
        raise ValueError("codes must be same-device bool [2, output_features, blocks, K].")
    bits = codes.shape[-1]
    updated = codes.clone()
    reconstruction = torch.empty_like(weight)
    targets = weight.clone()
    rows = torch.arange(weight.shape[0], device=weight.device)
    reports = []
    with torch.autocast(device_type=weight.device.type, enabled=False):
        for block in range(blocks - 1, -1, -1):
            start, end = block * block_size, (block + 1) * block_size
            candidates = two_stage_single_bit_candidates(codes[:, :, block, :], decode_stage).double()
            if candidates.shape[2] != block_size:
                raise ValueError("Decoded vector width must equal the LDL block size.")
            errors = targets[:, None, start:end] - candidates
            scores = ((errors @ diagonal[block]) * errors).sum(dim=-1)
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("Non-finite candidate quadratic score.")
            selected = scores.argmin(dim=1)
            reconstruction[:, start:end] = candidates[rows, selected]
            stage_flips = []
            for stage in range(2):
                mask = (selected > stage * bits) & (selected <= (stage + 1) * bits)
                chosen_rows = rows[mask]
                chosen_bits = selected[mask] - stage * bits - 1
                updated[stage, chosen_rows, block, chosen_bits] = ~updated[
                    stage, chosen_rows, block, chosen_bits]
                stage_flips.append(int(mask.sum()))
            if feedback and start:
                original_error = weight[:, start:end] - reconstruction[:, start:end]
                targets[:, :start].addmm_(original_error, lower[start:end, :start])
            report = {
                "block": block, "visited_blocks": len(reports) + 1, "total_blocks": blocks,
                "stage_flips": stage_flips,
                "current_local_score_sum": float(scores[:, 0].sum()),
                "selected_local_score_sum": float(scores[rows, selected].sum()),
            }
            reports.append(report)
            if progress is not None:
                progress(dict(report))
    return BlockLDLQResult(
        codes=updated, reconstructed_weight=reconstruction,
        statistics={
            "feedback": feedback, "blocks": blocks, "block_size": block_size,
            "bits_per_stage": bits, "candidates_per_vector": 1 + 2 * bits,
            "candidate_vector_scores": weight.shape[0] * blocks * (1 + 2 * bits),
            "stage_flips": [sum(item["stage_flips"][stage] for item in reports) for stage in range(2)],
            "block_reports": reports,
        },
    )
