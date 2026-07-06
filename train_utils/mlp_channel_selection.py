from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

import torch
from torch import nn

MLP_INTERMEDIATE_ALIGNED_ACTRMS = "mlp_intermediate_aligned_actrms"
MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS = "mlp_intermediate_aligned_actmean_abs"
MLP_INTERMEDIATE_ALIGNED_ACTRMS_ABS = "mlp_intermediate_aligned_actrms_abs"
MLP_ALIGNED_RANK_METRICS = (
    MLP_INTERMEDIATE_ALIGNED_ACTRMS,
    MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS,
    MLP_INTERMEDIATE_ALIGNED_ACTRMS_ABS,
)
MLP_CATEGORIES = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class MlpScoreDetail:
    score_up: torch.Tensor
    score_gate: torch.Tensor
    score_down: torch.Tensor
    score_fused: torch.Tensor
    rank_metric: str

    def to_log_dict(self, *, protected_indices: torch.Tensor) -> Dict[str, object]:
        idx_cpu = protected_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
        sample = idx_cpu[:20].tolist()
        return {
            "rank_metric": str(self.rank_metric),
            "mean_score_up": float(self.score_up.mean().item()),
            "mean_score_gate": float(self.score_gate.mean().item()),
            "mean_score_down": float(self.score_down.mean().item()),
            "score_fused_max": float(self.score_fused.max().item()),
            "score_fused_min": float(self.score_fused.min().item()),
            "score_fused_mean": float(self.score_fused.mean().item()),
            "protected_count": int(idx_cpu.numel()),
            "protected_indices_sample": sample,
        }


def normalize_mlp_aligned_rank_metric(metric: Optional[str]) -> str:
    resolved = str(metric or "").strip().lower()
    if resolved not in MLP_ALIGNED_RANK_METRICS:
        raise ValueError(
            f"Unsupported MLP aligned rank metric={metric!r}. "
            f"Expected one of: {', '.join(MLP_ALIGNED_RANK_METRICS)}."
        )
    return resolved


def is_mlp_aligned_rank_metric(metric: Optional[str]) -> bool:
    resolved = str(metric or "").strip().lower()
    return resolved in MLP_ALIGNED_RANK_METRICS


def mlp_protect_axis_for_category(category: str) -> str:
    cat = str(category).strip()
    if cat in {"gate_proj", "up_proj"}:
        return "output"
    if cat == "down_proj":
        return "input"
    raise ValueError(f"MLP aligned protect axis is undefined for category={category!r}.")


def _normalize_fuse_weights(
    fuse_weights: Sequence[float],
    *,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    if len(fuse_weights) != 3:
        raise ValueError(f"fuse_weights must have length 3, got {len(fuse_weights)}.")
    alpha_up, alpha_gate, alpha_down = (float(v) for v in fuse_weights)
    if alpha_up <= 0.0 or alpha_gate <= 0.0 or alpha_down <= 0.0:
        raise ValueError(
            "fuse_weights entries must be > 0, "
            f"got ({alpha_up}, {alpha_gate}, {alpha_down})."
        )
    denom = alpha_up + alpha_gate + alpha_down
    if denom <= eps:
        raise ValueError(f"fuse_weights sum must be > 0, got {denom}.")
    return alpha_up / denom, alpha_gate / denom, alpha_down / denom


def _validate_mlp_weight_shapes(
    *,
    W_up: torch.Tensor,
    W_gate: torch.Tensor,
    W_down: torch.Tensor,
    act_in: torch.Tensor,
    act_mid: torch.Tensor,
) -> int:
    if W_up.ndim != 2 or W_gate.ndim != 2 or W_down.ndim != 2:
        raise ValueError("MLP weights must be 2D tensors.")
    d_ffn_up = int(W_up.shape[0])
    d_ffn_gate = int(W_gate.shape[0])
    d_ffn_down = int(W_down.shape[1])
    if not (d_ffn_up == d_ffn_gate == d_ffn_down):
        raise ValueError(
            "MLP intermediate dimension mismatch: "
            f"up_out={d_ffn_up}, gate_out={d_ffn_gate}, down_in={d_ffn_down}."
        )
    d_model_up = int(W_up.shape[1])
    d_model_gate = int(W_gate.shape[1])
    d_model_down = int(W_down.shape[0])
    if not (d_model_up == d_model_gate == d_model_down):
        raise ValueError(
            "MLP hidden dimension mismatch: "
            f"up_in={d_model_up}, gate_in={d_model_gate}, down_out={d_model_down}."
        )
    if int(act_in.numel()) != d_model_up:
        raise ValueError(
            f"MLP input activation stats shape mismatch: got {tuple(act_in.shape)}, "
            f"expected ({d_model_up},)."
        )
    if int(act_mid.numel()) != d_ffn_up:
        raise ValueError(
            f"MLP intermediate activation stats shape mismatch: got {tuple(act_mid.shape)}, "
            f"expected ({d_ffn_up},)."
        )
    return d_ffn_up


def _score_up_gate_rows(
    weight: torch.Tensor,
    act_vec: torch.Tensor,
    *,
    rank_metric: str,
) -> torch.Tensor:
    act = act_vec.detach().to(device=weight.device, dtype=weight.dtype).view(1, -1)
    if rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTRMS:
        return torch.mean(torch.abs(weight) * act, dim=1)
    if rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS:
        return torch.norm(weight * act, p=2, dim=1)
    if rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTRMS_ABS:
        return (weight.pow(2) * act.clamp_min(0.0)).sum(dim=1)
    raise ValueError(f"Unsupported MLP aligned rank metric={rank_metric!r}.")


def _score_down_cols(
    weight: torch.Tensor,
    act_vec: torch.Tensor,
    *,
    rank_metric: str,
) -> torch.Tensor:
    act = act_vec.detach().to(device=weight.device, dtype=weight.dtype).view(1, -1)
    if rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTRMS:
        return torch.mean(torch.abs(weight) * act, dim=0)
    if rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS:
        return torch.norm(weight * act, p=2, dim=0)
    if rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTRMS_ABS:
        return (weight.pow(2) * act.clamp_min(0.0)).sum(dim=0)
    raise ValueError(f"Unsupported MLP aligned rank metric={rank_metric!r}.")


def _activation_vectors_for_metric(
    block_stats: Mapping[str, torch.Tensor],
    *,
    rank_metric: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTRMS:
        act_in = block_stats["a_in"]
        act_mid = block_stats["a_mid"]
    elif rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTMEAN_ABS:
        act_in = block_stats["abs_mean_in"]
        act_mid = block_stats["abs_mean_mid"]
    elif rank_metric == MLP_INTERMEDIATE_ALIGNED_ACTRMS_ABS:
        act_in = block_stats["sq_mean_in"]
        act_mid = block_stats["sq_mean_mid"]
    else:
        raise ValueError(f"Unsupported MLP aligned rank metric={rank_metric!r}.")
    if not isinstance(act_in, torch.Tensor) or not isinstance(act_mid, torch.Tensor):
        raise TypeError(f"Invalid activation stats for rank_metric={rank_metric!r}.")
    return act_in, act_mid


def compute_mlp_intermediate_scores(
    W_up: torch.Tensor,
    W_gate: torch.Tensor,
    W_down: torch.Tensor,
    block_stats: Mapping[str, torch.Tensor],
    *,
    rank_metric: str,
    fuse_weights: Sequence[float] = (1.0, 1.0, 1.0),
    eps: float = 1e-8,
) -> MlpScoreDetail:
    resolved_metric = normalize_mlp_aligned_rank_metric(rank_metric)
    W_up_f = W_up.detach().to(dtype=torch.float32).contiguous()
    W_gate_f = W_gate.detach().to(dtype=torch.float32).contiguous()
    W_down_f = W_down.detach().to(dtype=torch.float32).contiguous()
    act_in, act_mid = _activation_vectors_for_metric(block_stats, rank_metric=resolved_metric)
    act_in_f = act_in.detach().to(dtype=torch.float32).contiguous().view(-1)
    act_mid_f = act_mid.detach().to(dtype=torch.float32).contiguous().view(-1)
    _validate_mlp_weight_shapes(
        W_up=W_up_f,
        W_gate=W_gate_f,
        W_down=W_down_f,
        act_in=act_in_f,
        act_mid=act_mid_f,
    )

    score_up = _score_up_gate_rows(W_up_f, act_in_f, rank_metric=resolved_metric)
    score_gate = _score_up_gate_rows(W_gate_f, act_in_f, rank_metric=resolved_metric)
    score_down = _score_down_cols(W_down_f, act_mid_f, rank_metric=resolved_metric)

    score_up_norm = score_up / (score_up.mean() + eps)
    score_gate_norm = score_gate / (score_gate.mean() + eps)
    score_down_norm = score_down / (score_down.mean() + eps)

    alpha_up, alpha_gate, alpha_down = _normalize_fuse_weights(fuse_weights, eps=eps)
    score_fused = (
        alpha_up * score_up_norm
        + alpha_gate * score_gate_norm
        + alpha_down * score_down_norm
    ).contiguous()

    if not torch.isfinite(score_fused).all():
        raise ValueError("MLP fused channel score contains non-finite values.")

    return MlpScoreDetail(
        score_up=score_up.contiguous(),
        score_gate=score_gate.contiguous(),
        score_down=score_down.contiguous(),
        score_fused=score_fused,
        rank_metric=resolved_metric,
    )


def select_mlp_aligned_activation_weighted_channels(
    W_up: torch.Tensor,
    W_gate: torch.Tensor,
    W_down: torch.Tensor,
    block_stats: Mapping[str, torch.Tensor],
    *,
    rank_metric: str,
    protect_count: int,
    fuse_weights: Sequence[float] = (1.0, 1.0, 1.0),
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, MlpScoreDetail]:
    detail = compute_mlp_intermediate_scores(
        W_up,
        W_gate,
        W_down,
        block_stats,
        rank_metric=rank_metric,
        fuse_weights=fuse_weights,
        eps=eps,
    )
    d_ffn = int(detail.score_fused.numel())
    k = int(protect_count)
    if k < 0:
        raise ValueError(f"protect_count must be >= 0, got {k}.")
    if k == 0:
        return torch.empty(0, dtype=torch.long), detail
    if k >= d_ffn:
        raise ValueError(
            f"protect_count={k} must be < intermediate_size={d_ffn} for MLP aligned selection."
        )
    _, idx = torch.topk(detail.score_fused, k=k, largest=True, sorted=False)
    protected_indices = torch.sort(idx.to(device="cpu", dtype=torch.long)).values.contiguous()
    return protected_indices, detail


def _linear_name_for_mlp_category(layer_idx: int, category: str) -> str:
    if category not in MLP_CATEGORIES:
        raise ValueError(f"Unsupported MLP category={category!r}.")
    return f"model.layers.{int(layer_idx)}.mlp.{category}"


def build_mlp_aligned_plans_all_layers(
    *,
    model: nn.Module,
    stats_by_mlp_block: Dict[int, Dict[str, torch.Tensor]],
    protect_count: int,
    fuse_weights: Sequence[float],
    rank_metric: str,
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
    eps: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], Dict[int, Dict[str, object]]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model must expose model.model.layers for MLP aligned channel selection.")
    layers = model.model.layers
    plan_by_linear: Dict[str, torch.Tensor] = {}
    summary_by_layer: Dict[int, Dict[str, object]] = {}
    skipped = skip_layer_keys or set()
    resolved_metric = normalize_mlp_aligned_rank_metric(rank_metric)

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        if any((int(layer_idx), cat) in skipped for cat in MLP_CATEGORIES):
            continue
        mlp = layer.mlp
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
            continue
        block_stats = stats_by_mlp_block.get(int(layer_idx))
        if block_stats is None:
            raise KeyError(f"Missing MLP activation stats for layer_idx={layer_idx}.")

        protected_indices, detail = select_mlp_aligned_activation_weighted_channels(
            up.weight,
            gate.weight,
            down.weight,
            block_stats,
            rank_metric=resolved_metric,
            protect_count=int(protect_count),
            fuse_weights=fuse_weights,
            eps=eps,
        )
        for category in MLP_CATEGORIES:
            plan_by_linear[_linear_name_for_mlp_category(layer_idx, category)] = protected_indices
        summary_by_layer[int(layer_idx)] = detail.to_log_dict(protected_indices=protected_indices)

    return plan_by_linear, summary_by_layer


def write_mlp_channel_selection_summary(
    path: str,
    *,
    summary_by_layer: Dict[int, Dict[str, object]],
    protect_count: int,
    fuse_weights: Sequence[float],
    rank_metric: str,
) -> None:
    payload = {
        "rank_metric": normalize_mlp_aligned_rank_metric(rank_metric),
        "protect_count": int(protect_count),
        "fuse_weights": [float(v) for v in fuse_weights],
        "layers": {
            str(layer_idx): layer_summary
            for layer_idx, layer_summary in sorted(summary_by_layer.items(), key=lambda item: item[0])
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
