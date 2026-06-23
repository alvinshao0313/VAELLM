import math
from typing import Dict, Optional, Tuple

import torch

from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED,
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
    encode_blocked_quantized_sparse_residual,
    sparse_residual_blocked_storage_bytes,
    sparse_residual_coo_storage_bytes,
)


RESIDUAL_SPARSE_RANK_METRICS = frozenset(
    {
        "sparse_residual_abs",
        "sparse_residual_actmax_abs",
        "sparse_weight_abs",
        "sparse_weight_actmax_abs",
    }
)
RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX = frozenset(
    {"sparse_residual_actmax_abs", "sparse_weight_actmax_abs"}
)


def _select_sparse_residual_entries(
    *,
    linear_name: str,
    original_weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
    activation_weight: Optional[torch.Tensor],
    rank_metric: str,
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
    resolved_rank_metric = str(rank_metric).strip().lower()
    if resolved_rank_metric in {"sparse_residual_abs", "sparse_residual_actmax_abs"}:
        score = abs_residual
    elif resolved_rank_metric in {"sparse_weight_abs", "sparse_weight_actmax_abs"}:
        score = original_weight.abs()
    else:
        raise ValueError(
            f"{linear_name}: unsupported residual_sparse rank metric {rank_metric!r}. "
            "Expected sparse_residual_abs, sparse_residual_actmax_abs, "
            "sparse_weight_abs, or sparse_weight_actmax_abs."
        )

    if resolved_rank_metric in RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX:
        if activation_weight is None:
            raise ValueError(f"{linear_name}: {resolved_rank_metric} requires activation_weight.")
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
    rank_metric: str,
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
        rank_metric=rank_metric,
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
