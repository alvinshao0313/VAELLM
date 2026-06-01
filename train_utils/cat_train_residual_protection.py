import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED,
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
    encode_blocked_quantized_sparse_residual,
    sparse_residual_blocked_storage_bytes,
    sparse_residual_coo_storage_bytes,
)


RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT = frozenset(
    {"input_act_weighted_abs", "input_act_weighted_original_weight_abs"}
)
LOW_RANK_OUTLIER_MODES = frozenset({"per_vae_low_rank", "post_vae_low_rank"})


def compute_low_rank_svd_payload(
    *,
    linear_name: str,
    weight: torch.Tensor,
    rank: int,
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weight_f = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if weight_f.ndim != 2:
        raise ValueError(f"{linear_name}: low-rank SVD expects 2D weight, got shape={tuple(weight_f.shape)}")
    out_features, in_features = int(weight_f.shape[0]), int(weight_f.shape[1])
    max_rank = min(out_features, in_features)
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"{linear_name}: outlier_low_rank must be > 0, got {rank}.")
    if rank > max_rank:
        raise ValueError(
            f"{linear_name}: outlier_low_rank={rank} exceeds min(out_features,in_features)={max_rank}."
        )
    u, s, vh = torch.linalg.svd(weight_f, full_matrices=False)
    sqrt_s = torch.sqrt(s[:rank])
    low_rank_a = (u[:, :rank] * sqrt_s.view(1, rank)).to(dtype=target_dtype).contiguous()
    low_rank_b = (sqrt_s.view(rank, 1) * vh[:rank, :]).to(dtype=target_dtype).contiguous()
    return low_rank_a, low_rank_b


def subtract_low_rank_payload(
    *,
    weight: torch.Tensor,
    low_rank_a: torch.Tensor,
    low_rank_b: torch.Tensor,
) -> torch.Tensor:
    weight_f = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    patch = low_rank_a.detach().to(device="cpu", dtype=torch.float32) @ low_rank_b.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    if tuple(patch.shape) != tuple(weight_f.shape):
        raise ValueError(
            f"low-rank patch shape mismatch: patch={tuple(patch.shape)} vs weight={tuple(weight_f.shape)}"
        )
    return (weight_f - patch).contiguous()


def build_per_vae_low_rank_payloads(
    *,
    prepared_entries: Sequence[object],
    rank: int,
) -> Tuple[List[Optional[Tuple[torch.Tensor, torch.Tensor]]], List[torch.Tensor]]:
    low_rank_payloads: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    residual_weights: List[torch.Tensor] = []
    for entry in prepared_entries:
        split_weight = entry.prepared_weight.split_weight
        low_rank_a, low_rank_b = compute_low_rank_svd_payload(
            linear_name=entry.ref.name,
            weight=split_weight,
            rank=int(rank),
            target_dtype=split_weight.dtype,
        )
        residual_weight = subtract_low_rank_payload(
            weight=split_weight,
            low_rank_a=low_rank_a,
            low_rank_b=low_rank_b,
        )
        low_rank_payloads.append((low_rank_a, low_rank_b))
        residual_weights.append(residual_weight)
    return low_rank_payloads, residual_weights


def build_post_vae_low_rank_payload(
    *,
    linear_name: str,
    original_weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
    rank: int,
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_weight = (
        original_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
        - reconstructed_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    return compute_low_rank_svd_payload(
        linear_name=linear_name,
        weight=residual_weight,
        rank=int(rank),
        target_dtype=target_dtype,
    )


def _select_sparse_residual_entries(
    *,
    linear_name: str,
    original_weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
    activation_weight: Optional[torch.Tensor],
    score_mode: str,
    top_p: float,
    min_abs: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    original_weight = original_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    reconstructed_weight = reconstructed_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(original_weight.shape) != tuple(reconstructed_weight.shape):
        raise ValueError(
            f"{linear_name}: original/reconstructed weight shape mismatch: "
            f"{tuple(original_weight.shape)} vs {tuple(reconstructed_weight.shape)}"
        )
    out_features, in_features = int(original_weight.shape[0]), int(original_weight.shape[1])
    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"{linear_name}: residual_sparse top_p must satisfy 0 < top_p <= 1, got {top_p}.")
    if float(min_abs) < 0.0:
        raise ValueError(f"{linear_name}: residual_sparse min_abs must be >= 0, got {min_abs}.")

    residual = (original_weight - reconstructed_weight).contiguous()
    abs_residual = residual.abs()
    resolved_score_mode = str(score_mode).strip().lower()
    if resolved_score_mode in {"abs", "input_act_weighted_abs"}:
        score = abs_residual
    elif resolved_score_mode in {"original_weight_abs", "input_act_weighted_original_weight_abs"}:
        score = original_weight.abs()
    else:
        raise ValueError(
            f"{linear_name}: unsupported residual sparse score mode {score_mode!r}. "
            "Expected abs, input_act_weighted_abs, original_weight_abs, "
            "or input_act_weighted_original_weight_abs."
        )

    if resolved_score_mode in RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT:
        if activation_weight is None:
            raise ValueError(f"{linear_name}: {resolved_score_mode} requires activation_weight.")
        act = activation_weight.detach().to(device="cpu", dtype=torch.float32).contiguous().abs()
        if int(act.numel()) != in_features:
            raise ValueError(
                f"{linear_name}: activation_weight size mismatch for residual_sparse, "
                f"got {int(act.numel())}, expected {in_features}."
            )
        score = score * act.view(1, in_features)

    flat_score = score.view(-1)
    flat_abs_residual = abs_residual.view(-1)
    total_numel = int(flat_score.numel())
    nnz_target = max(1, int(math.ceil(float(top_p) * float(total_numel))))
    nnz_target = min(nnz_target, total_numel)
    valid_mask = (flat_score > 0) & (flat_abs_residual >= float(min_abs))
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    valid_count = int(valid_idx.numel())
    if valid_count == 0:
        return None, None, None
    k = min(nnz_target, valid_count)
    valid_scores = flat_score.index_select(0, valid_idx)
    _, top_local_idx = torch.topk(valid_scores, k=k, largest=True, sorted=False)
    top_idx = valid_idx.index_select(0, top_local_idx)
    top_idx = torch.sort(top_idx.to(dtype=torch.int64)).values.contiguous()
    flat_residual = residual.view(-1)
    values = flat_residual.index_select(0, top_idx).to(dtype=torch.float32).contiguous()
    row_idx = torch.div(top_idx, in_features, rounding_mode="floor").to(dtype=torch.int64).contiguous()
    col_idx = torch.remainder(top_idx, in_features).to(dtype=torch.int64).contiguous()
    return row_idx, col_idx, values


def build_sparse_residual_payload(
    *,
    linear_name: str,
    original_weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
    activation_weight: Optional[torch.Tensor],
    score_mode: str,
    top_p: float,
    min_abs: float,
    codec: str,
    index_bits: int,
    value_bits: int,
    block_shape: Tuple[int, int],
) -> Tuple[Optional[Dict[str, object]], int, Dict[str, int]]:
    row_idx, col_idx, values = _select_sparse_residual_entries(
        linear_name=linear_name,
        original_weight=original_weight,
        reconstructed_weight=reconstructed_weight,
        activation_weight=activation_weight,
        score_mode=score_mode,
        top_p=top_p,
        min_abs=min_abs,
    )
    if row_idx is None or col_idx is None or values is None:
        return None, 0, {"coo_bytes": 0, "codec_bytes": 0}

    nnz = int(values.numel())
    out_features = int(original_weight.shape[0])
    in_features = int(original_weight.shape[1])
    coo_bytes = sparse_residual_coo_storage_bytes(nnz)
    resolved_codec = str(codec).strip().lower()
    if resolved_codec == SPARSE_RESIDUAL_FORMAT_COO_FP16:
        if out_features > 65535 or in_features > 65535:
            raise ValueError(
                f"{linear_name}: residual_sparse codec=coo_fp16 requires out_features/in_features <= 65535 for uint16 indices, "
                f"got out_features={out_features}, in_features={in_features}."
            )
        payload = {
            "sparse_residual_format": SPARSE_RESIDUAL_FORMAT_COO_FP16,
            "sparse_residual_row_indices": row_idx.to(dtype=torch.uint16).contiguous(),
            "sparse_residual_col_indices": col_idx.to(dtype=torch.uint16).contiguous(),
            "sparse_residual_values": values.to(dtype=torch.float16).contiguous(),
        }
        return payload, nnz, {"coo_bytes": coo_bytes, "codec_bytes": coo_bytes}
    if resolved_codec != SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED:
        raise ValueError(
            f"{linear_name}: unsupported sparse residual codec {codec!r}. "
            f"Expected {SPARSE_RESIDUAL_FORMAT_COO_FP16} or {SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED}."
        )
    blocked_payload = encode_blocked_quantized_sparse_residual(
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
        out_features=out_features,
        in_features=in_features,
        block_rows=int(block_shape[0]),
        block_cols=int(block_shape[1]),
        index_bits=int(index_bits),
        value_bits=int(value_bits),
    )
    payload = {
        "sparse_residual_format": str(blocked_payload["format"]),
        "sparse_residual_index_bits": int(blocked_payload["index_bits"]),
        "sparse_residual_value_bits": int(blocked_payload["value_bits"]),
        "sparse_residual_block_rows": int(blocked_payload["block_rows"]),
        "sparse_residual_block_cols": int(blocked_payload["block_cols"]),
        "sparse_residual_active_block_ids": blocked_payload["active_block_ids"],
        "sparse_residual_block_ptr": blocked_payload["block_ptr"],
        "sparse_residual_local_indices": blocked_payload["local_indices"],
        "sparse_residual_qvalues": blocked_payload["qvalues"],
        "sparse_residual_scales": blocked_payload["scales"],
        "sparse_residual_zero_points": blocked_payload["zero_points"],
    }
    return payload, nnz, {
        "coo_bytes": coo_bytes,
        "codec_bytes": sparse_residual_blocked_storage_bytes(blocked_payload),
    }
