"""Finite-neighbor column feedback in an existing output-vector VQ layout.

This is an isolated experiment, not a VPTQ reproduction. All arms keep the
original 49 candidates per vector. Delayed16 deliberately omits feedback
inside each sixteen-column group; it is NOT ordinary lazy GPTQ updating.
"""
from __future__ import annotations

import time

import torch

from experiments.qwen3_w25.block_ldlq import two_stage_single_bit_candidates
from experiments.rotation_factor_probe.capture import require

ARMS = ('off', 'delayed16', 'per_column')


@torch.no_grad()
def scalar_ldl(hessian):
    """H=L diag(d) L.T, using the same scalar LDL as block_ldl(H, 1)."""
    require(hessian.dtype == torch.float64 and hessian.ndim == 2
            and hessian.shape[0] == hessian.shape[1], 'Require a square FP64 Hessian.')
    chol = torch.linalg.cholesky(hessian)
    diagonal = chol.diagonal().clone()
    lower = chol / diagonal.unsqueeze(0)
    return lower, diagonal.square()


@torch.no_grad()
def choices_to_codes(original, choices):
    """Bank logical index is input_column * output_groups + output_group."""
    require(original.dtype == torch.bool and original.ndim == 4 and original.shape[0] == 2
            and tuple(choices.shape) == tuple(original.shape[1:3]) and choices.dtype == torch.uint8,
            'Expected bool [2, input_columns, output_groups, K] and uint8 [input_columns, output_groups].')
    bits = original.shape[-1]
    require(int(choices.max()) <= 2 * bits, 'A choice is outside the fixed two-stage single-bit neighborhood.')
    updated = original.clone()
    for stage in range(2):
        selected = (choices > stage * bits) & (choices <= (stage + 1) * bits)
        columns, groups = selected.nonzero(as_tuple=True)
        bit = choices[selected].long() - stage * bits - 1
        updated[stage, columns, groups, bit] = ~updated[stage, columns, groups, bit]
    return updated


@torch.no_grad()
def search_columns(target, lower, diagonal, original, decode_stage, *, deadline, observer=None):
    """Visit every input column once in descending order in each of three arms.

    For E=T-Q, tr(E H E.T)=sum_j d_j ||E_j+sum_{k>j}E_k L_kj||².
    Thus the immediate target is T_j+sum_{k>j}(T_k-Q_k)L_kj. Delayed16
    includes only completed groups, not completed columns in the current group.
    Per-column uses algebraically equivalent lazy updates OUTSIDE each group;
    its feedback INSIDE each group is immediate. Off applies neither update.
    All arms share the protection-compensated T and scalar d_j candidate metric.
    """
    out_features, in_features = target.shape
    stages, columns, groups, bits = original.shape
    require(target.dtype == lower.dtype == diagonal.dtype == torch.float64
            and stages == 2 and columns == in_features and in_features % 16 == 0
            and out_features % groups == 0 and original.dtype == torch.bool
            and lower.shape == (in_features, in_features) and diagonal.shape == (in_features,)
            and bool((diagonal > 0).all()), 'Invalid rotated scalar-LDL / transposed-code geometry.')
    width = out_features // groups
    choices = {arm: torch.zeros((in_features, groups), dtype=torch.uint8, device=target.device) for arm in ARMS}
    reconstruction = {arm: torch.empty_like(target, dtype=torch.bfloat16) for arm in ARMS}
    targets = {arm: target.clone() for arm in ARMS if arm != 'off'}
    current_score = {arm: 0.0 for arm in ARMS}
    selected_score = {arm: 0.0 for arm in ARMS}
    original_reconstruction = torch.empty_like(target, dtype=torch.bfloat16)
    row_indices = torch.arange(groups, device=target.device)
    for stop in range(in_features, 0, -16):
        require(time.time() < deadline, 'Column-search hard deadline exceeded; no partial candidate is successful.')
        start = stop - 16
        # Candidate generation is independent of the arm and all feedback targets.
        raw = original[:, start:stop].reshape(2, 16 * groups, bits)
        candidates = two_stage_single_bit_candidates(raw, decode_stage).reshape(16, groups, 1 + 2 * bits, width)
        require(candidates.dtype == torch.bfloat16, 'Keep native BF16 stage sums before FP64 scoring.')
        original_reconstruction[:, start:stop] = candidates[:, :, 0].reshape(16, out_features).T
        for arm in ARMS:
            adjusted = target if arm == 'off' else targets[arm]
            for column in range(stop - 1, start - 1, -1):
                proposal = candidates[column - start]
                delta = adjusted[:, column].reshape(groups, width)[:, None] - proposal.double()
                scores = delta.square().sum(-1) * diagonal[column]
                require(bool(torch.isfinite(scores).all()), 'Nonfinite candidate metric.')
                choice = scores.argmin(-1)  # Original candidate is zero and wins exact ties.
                choices[arm][column] = choice.to(torch.uint8)
                reconstruction[arm][:, column] = proposal[row_indices, choice].reshape(out_features)
                current_score[arm] += float(scores[:, 0].sum())
                selected_score[arm] += float(scores[row_indices, choice].sum())
                if arm == 'per_column' and column > start:
                    error = target[:, column] - reconstruction[arm][:, column].double()
                    adjusted[:, start:column].addmm_(error[:, None], lower[column:column + 1, start:column])
            if arm != 'off' and start:
                error = target[:, start:stop] - reconstruction[arm][:, start:stop].double()
                adjusted[:, :start].addmm_(error, lower[start:stop, :start])
        if observer is not None and ((in_features - start) % 256 == 0 or start == 0):
            observer({'visited_columns': in_features - start, 'total_columns': in_features})
    statistics = {}
    for arm in ('original', *ARMS):
        value = original_reconstruction if arm == 'original' else reconstruction[arm]
        error = target - value.double()
        transformed = error @ lower
        statistics[arm] = {'full_damped_protection_compensated_proxy': float((transformed.square() * diagonal).sum()),
                           'rotated_target_sse': float(error.square().sum())}
        if arm != 'original':
            selected = choices[arm]
            statistics[arm].update(candidate_current_score_sum=current_score[arm],
                candidate_selected_score_sum=selected_score[arm],
                stage_xor=[int(((selected > s * bits) & (selected <= (s + 1) * bits)).sum()) for s in range(2)],
                candidate_count_per_vector=1 + 2 * bits, selections=in_features * groups)
    return choices, reconstruction, original_reconstruction, statistics
