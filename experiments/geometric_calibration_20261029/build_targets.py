"""Original-coordinate FP32 SPACE targets from genuine deployed BF16 weights."""
from __future__ import annotations

import time

from native_pair import require
import torch


def energy(value):
    return float(value.double().square().sum())


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


@torch.no_grad()
def build_targets(pre_bf16, post_bf16, protected):
    require(pre_bf16.dtype == post_bf16.dtype == torch.bfloat16
            and pre_bf16.shape == post_bf16.shape and pre_bf16.ndim == 2,
            'Targets require the same complete deployed BF16 linear operator geometry.')
    started = time.perf_counter()
    pre, post = pre_bf16.float(), post_bf16.float()
    delta = post - pre
    kwargs = {'driver': 'gesvd'} if pre.is_cuda else {}
    # Exact complete SVD; never approximate SVD or materialize the in-by-in P.
    _, singular, vh = torch.linalg.svd(pre, full_matrices=False, **kwargs)
    cumulative = singular.square().cumsum(0)
    require(float(cumulative[-1]) > 0, 'A zero reference matrix has no rho=.5 energy threshold.')
    rank = int(torch.searchsorted(cumulative, .5 * cumulative[-1])) + 1
    directions = vh[:rank].T.contiguous()
    removed = (delta @ directions) @ directions.T
    kept = delta - removed
    delta_energy, removed_energy, kept_energy = energy(delta), energy(removed), energy(kept)
    beta = (kept_energy / delta_energy) ** .5 if delta_energy else 1.0
    space = post - removed
    shrink = pre + beta * delta
    protected = protected.to(device=pre.device, dtype=torch.long)
    require(energy(delta[:, protected]) == 0, 'The source recovery changed protected columns.')
    orthogonal_residual = ratio(energy(kept @ directions) ** .5, delta_energy ** .5)
    require(orthogonal_residual is None or orthogonal_residual < 2e-5,
            'The complete FP32 SPACE projection failed its orthogonality identity.')
    stats = dict(rho=.5, alpha=1.0, rank=rank, captured_weight_energy=float(cumulative[rank-1] / cumulative[-1]),
                 shape=list(pre.shape), delta_energy=delta_energy, calibration_move_energy=removed_energy,
                 retained_update_energy=kept_energy, beta=beta, orthogonality_relative_residual=orthogonal_residual,
                 protected_columns=protected.numel(), original_delta_on_protection_energy=0.0,
                 svd_backend='torch.linalg.svd full_matrices=False, gesvd on CUDA; FP32',
                 svd_projection_seconds=time.perf_counter()-started,
                 targets={})
    for label, target in (('space', space), ('shrink', shrink)):
        move = target - post
        rounded = target.bfloat16().float()
        move_energy = energy(move)
        stats['targets'][label] = dict(ideal_move_energy=move_energy,
            protected_unreachable_energy=energy(move[:, protected]),
            protected_unreachable_fraction_of_move=ratio(energy(move[:, protected]), move_energy),
            bf16_rounding_energy_over_move=ratio(energy(rounded-target), move_energy),
            bf16_actual_move_energy_over_ideal=ratio(energy(rounded-post), move_energy),
            bf16_protected_changed_elements=int(torch.count_nonzero((rounded-post)[:, protected])),
            ideal_retained_update_norm=energy(target-pre)**.5,
            bf16_retained_update_norm=energy(rounded-pre)**.5)
    return pre, post, {'space': space, 'shrink': shrink}, stats


@torch.no_grad()
def endpoint_metrics(candidate, target, pre, post, protected):
    candidate = candidate.float()
    move, residual = target-post, candidate-target
    denominator, residual_energy = energy(move), energy(residual)
    actual_move = candidate-post
    inner = float((actual_move.double()*move.double()).sum())
    return dict(mse=residual_energy/target.numel(),
                error_energy=residual_energy, error_energy_over_target=ratio(residual_energy, energy(target)),
                error_energy_over_calibration_move=ratio(residual_energy, denominator),
                fit_improvement_vs_G02=1-residual_energy/denominator if denominator else None,
                calibration_direction_survival=ratio(inner, denominator),
                actual_move_energy_over_ideal=ratio(energy(actual_move), denominator),
                actual_retained_update_norm=energy(candidate-pre)**.5,
                protected_error_energy=energy(residual[:, protected]),
                protected_move_energy=energy(actual_move[:, protected]),
                bf16_changed_elements_vs_G02=int(torch.count_nonzero(actual_move)))
