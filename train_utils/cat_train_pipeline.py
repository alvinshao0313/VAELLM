import os
import sys
import time
import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train_utils.train_args import create_optimizer
from train_utils.cat_train_args import (
    ResolvedCategoryRuntimeConfig,
    process_cat_train_args,
    resolve_category_runtime_configs,
    resolve_skip_layer_matches,
)
from train_utils.cat_after_category_distill import run_after_category_distill
from train_utils.cat_train_runtime import (
    load_model_for_cat_train as _load_model_for_cat_train,
    save_normalized_cat_train_snapshot as _save_normalized_cat_train_snapshot,
)
from train_utils.cat_train_residual_protection import (
    RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX as _RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX,
    RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMEAN as _RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMEAN,
    build_sparse_residual_payload as _build_sparse_residual_payload,
)
from litebsq.vae_args import apply_autoencoder_arch_defaults
from litebsq.misc import set_module_by_name
from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
)
from train_utils.cat_data_prep import (
    LinearPrepRef,
    build_outlier_channel_index_plan,
    compute_channel_rank_score,
    format_intra_part_sort_mode,
    gather_wa_mse_act_max_batch,
    materialize_prepared_group_data,
    prepare_group_linear_entries,
    select_outlier_channel_indices_from_scores,
)
from train_utils.activation_utils import (
    ActivationCalibrationCache,
    collect_activation_stats_for_linears,
)
from train_utils.cat_arg_overrides import validate_category_keys
from train_utils.cat_train_data import (
    apply_stage_norm as _apply_stage_norm,
    build_block_data_loaders as _build_block_data_loaders,
    compute_stage_norm_stats as _compute_stage_norm_stats,
    reshape_blocks_for_codebook_dim as _reshape_blocks_for_codebook_dim,
    restore_stage_norm as _restore_stage_norm,
)
# 联合优化代码，已关闭：不再导入 train_utils.cat_joint_decoder。
# from train_utils.cat_joint_decoder import (
#     finetune_stage_decoders_in_subgroups as _finetune_stage_decoders_in_subgroups,
# )
from train_utils.cat_train_eval import eval_after_category as _eval_after_category
from train_utils.model_checkpoint_io import (
    _build_run_output_dir,
    register_shared_protected_residual_decoder,
    save_model_checkpoint,
)
from train_utils.utils import (
    LinearRef,
    clone_namespace as _clone_namespace,
    collect_linears as _collect_linears,
    configure_deterministic_mode,
    extract_layer_idx as _extract_layer_idx,
    format_namespace as _format_namespace,
    get_logger,
    set_seed,
    split_csv as _split_csv,
)


log = get_logger("linear_by_category")


def _safe_shared_decoder_ref(value: str) -> str:
    text = str(value).strip()
    chars = [ch if (ch.isalnum() or ch == "_") else "_" for ch in text]
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "shared_protected_residual_decoder"


def _fuse_q_scale_linear(linear: nn.Linear, q_scale: float) -> None:
    with torch.no_grad():
        weight = linear.weight.data
        bias_delta = -q_scale * weight.sum(dim=1)
        weight.mul_(q_scale * 2)
        if linear.bias is not None:
            linear.bias.data.add_(bias_delta)
        else:
            linear.bias = nn.Parameter(bias_delta)


def _fuse_q_scale_into_decoder(decoder: nn.Module, q_scale: float) -> None:
    if hasattr(decoder, "_fuse_q_scale"):
        decoder._fuse_q_scale(float(q_scale))
        return

    # 回退逻辑: 没有 Decoder._fuse_q_scale 时直接融合到第一层线性。
    decoder_type = str(getattr(decoder, "decoder_type"))
    if decoder_type == "linear":
        _fuse_q_scale_linear(decoder.linear, q_scale)
    elif decoder_type in {"symmetric", "asymmetric"}:
        _fuse_q_scale_linear(decoder.linear_in, q_scale)


def _fuse_norm_into_decoder(decoder: nn.Module, mean: float, std: float) -> None:
    decoder_type = str(getattr(decoder, "decoder_type"))
    if decoder_type == "linear":
        last = decoder.linear
    elif decoder_type in {"symmetric", "asymmetric"}:
        last = decoder.linear_out
    else:
        raise ValueError(f"Unsupported decoder_type={decoder_type} for norm fusion")

    if not isinstance(last, nn.Linear):
        raise TypeError(f"Expected nn.Linear as last layer, got {type(last)}")

    with torch.no_grad():
        last.weight.mul_(std)
        if last.bias is None:
            last.bias = nn.Parameter(torch.zeros(last.out_features, device=last.weight.device, dtype=last.weight.dtype))
        last.bias.mul_(std).add_(mean)


def _compute_recon_loss(
    *,
    recon_loss_type: str,
    x_recon: torch.Tensor,
    x: torch.Tensor,
    act_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    resolved = str(recon_loss_type).strip().lower()
    if resolved == "l1":
        return torch.nn.functional.l1_loss(x_recon, x)
    if resolved == "huber":
        return torch.nn.functional.huber_loss(x_recon, x, reduction="mean", delta=1.0)
    if resolved == "relative_l1":
        return (x_recon - x).abs().sum() / (x.abs().sum() + 1e-10)
    if resolved == "top_k_mse":
        k = max(1, int(0.1 * x.shape[-1]))
        errors = (x_recon - x).pow(2)
        topk_errors, _ = torch.topk(errors, k, dim=-1)
        return topk_errors.sum()
    if resolved == "mse":
        return torch.nn.functional.mse_loss(x_recon, x)
    if resolved == "cosine":
        x_recon_flat = x_recon.view(x_recon.size(0), -1)
        x_flat = x.view(x.size(0), -1)
        return 1 - torch.nn.functional.cosine_similarity(x_recon_flat, x_flat, dim=-1).mean()
    if resolved == "w_mse":
        return ((x_recon - x).pow(2) * x.abs()).mean()
    if resolved == "w2_mse":
        return ((x_recon - x).pow(2) * x.pow(2)).mean()
    if resolved == "wa_mse":
        if act_max is None:
            raise ValueError("recon_loss_type=wa_mse requires act_max tensor.")
        if tuple(act_max.shape) != tuple(x.shape):
            raise ValueError(
                f"wa_mse shape mismatch: act_max={tuple(act_max.shape)} vs x={tuple(x.shape)}"
            )
        x_f = x.float()
        x_recon_f = x_recon.float()
        act_f = act_max.float()
        errors = (x_recon_f - x_f).pow(2)
        weights = x_f.abs() * act_f
        return (errors * weights).mean()
    return torch.zeros((), device=x.device, dtype=torch.float32)


def _split_weight_into_part_flats(
    *,
    weight: torch.Tensor,
    transpose: bool,
    parallel_rows: int,
    parallel_cols: int,
) -> torch.Tensor:
    w = weight.t().contiguous() if bool(transpose) else weight.contiguous()
    rows_per_part = int(w.shape[0]) // int(parallel_rows)
    cols_per_part = int(w.shape[1]) // int(parallel_cols)
    parts = []
    for row_idx in range(int(parallel_rows)):
        row_start = row_idx * rows_per_part
        row_end = row_start + rows_per_part
        for col_idx in range(int(parallel_cols)):
            col_start = col_idx * cols_per_part
            col_end = col_start + cols_per_part
            parts.append(w[row_start:row_end, col_start:col_end].contiguous().view(-1))
    return torch.stack(parts, dim=0)


def _restore_split_row_order_with_meta(w_split: torch.Tensor, split_meta) -> torch.Tensor:
    # 排序代码，已关闭。原 row restore 分支保留如下：
    # restore_idx = getattr(split_meta, "restore_row_indices", None)
    # if restore_idx is None:
    #     return w_split
    # if int(restore_idx.numel()) != int(w_split.shape[0]):
    #     raise ValueError(
    #         f"{split_meta.linear_name}: restore_row_indices size {int(restore_idx.numel())} != split rows {int(w_split.shape[0])}"
    #     )
    # if restore_idx.device != w_split.device:
    #     restore_idx = restore_idx.to(device=w_split.device, non_blocking=True)
    # return w_split.index_select(0, restore_idx)
    return w_split


def _restore_split_col_order_with_meta(w_split: torch.Tensor, split_meta) -> torch.Tensor:
    # 排序代码，已关闭。原 col restore 分支保留如下：
    # restore_idx = getattr(split_meta, "restore_col_indices", None)
    # if restore_idx is None:
    #     return w_split
    # if int(restore_idx.numel()) != int(w_split.shape[1]):
    #     raise ValueError(
    #         f"{split_meta.linear_name}: restore_col_indices size {int(restore_idx.numel())} != split cols {int(w_split.shape[1])}"
    #     )
    # if restore_idx.device != w_split.device:
    #     restore_idx = restore_idx.to(device=w_split.device, non_blocking=True)
    # return w_split.index_select(1, restore_idx)
    return w_split


def _restore_part_col_order_with_meta(part_matrix: torch.Tensor, split_meta, part_idx: int) -> torch.Tensor:
    # 排序代码，已关闭。原 part col restore 分支保留如下：
    # restore_all = getattr(split_meta, "part_restore_col_indices", None)
    # if restore_all is None:
    #     return part_matrix
    # if restore_all.ndim != 2:
    #     raise ValueError(
    #         f"{split_meta.linear_name}: part_restore_col_indices must be 2D, got shape={tuple(restore_all.shape)}"
    #     )
    # if part_idx < 0 or part_idx >= int(restore_all.shape[0]):
    #     raise IndexError(
    #         f"{split_meta.linear_name}: part_idx out of range for part_restore_col_indices: {part_idx} vs {int(restore_all.shape[0])}"
    #     )
    # restore_idx = restore_all[part_idx]
    # if int(restore_idx.numel()) != int(part_matrix.shape[1]):
    #     raise ValueError(
    #         f"{split_meta.linear_name}: part_restore_col_indices[{part_idx}] size {int(restore_idx.numel())} != part cols {int(part_matrix.shape[1])}"
    #     )
    # if restore_idx.device != part_matrix.device:
    #     restore_idx = restore_idx.to(device=part_matrix.device, non_blocking=True)
    # return part_matrix.index_select(1, restore_idx)
    return part_matrix


def _restore_split_weight_from_part_flats_with_meta(
    *,
    part_flats: torch.Tensor,
    split_meta,
    dtype: torch.dtype,
) -> torch.Tensor:
    parallel_rows = int(split_meta.parallel_rows)
    parallel_cols = int(split_meta.parallel_cols)
    parallel_parts = int(parallel_rows) * int(parallel_cols)
    split_rows = int(split_meta.compressed_in_features) if bool(
        split_meta.transpose) else int(split_meta.compressed_out_features)
    split_cols = int(split_meta.compressed_out_features) if bool(
        split_meta.transpose) else int(split_meta.compressed_in_features)
    part_flats = part_flats.reshape(parallel_parts, -1).contiguous()
    if parallel_parts == 1:
        w_split = part_flats[0].view(split_rows, split_cols)
        w_split = _restore_part_col_order_with_meta(w_split, split_meta, 0)
        w_split = _restore_split_row_order_with_meta(w_split, split_meta)
        w_split = _restore_split_col_order_with_meta(w_split, split_meta)
        return w_split.contiguous().to(dtype=dtype)

    rows_per_part = split_rows // parallel_rows
    cols_per_part = split_cols // parallel_cols
    expected_per_part = int(rows_per_part) * int(cols_per_part)
    if int(part_flats.shape[1]) != expected_per_part:
        raise ValueError(
            f"{split_meta.linear_name}: per-part flat width mismatch: got {int(part_flats.shape[1])}, expected {expected_per_part}."
        )
    parts = [
        _restore_part_col_order_with_meta(
            part_flats[part_idx].view(rows_per_part, cols_per_part),
            split_meta,
            part_idx,
        )
        for part_idx in range(parallel_parts)
    ]
    row_blocks = []
    for row_idx in range(parallel_rows):
        start = row_idx * parallel_cols
        end = start + parallel_cols
        row_blocks.append(torch.cat(parts[start:end], dim=1))
    w_split = torch.cat(row_blocks, dim=0)
    w_split = _restore_split_row_order_with_meta(w_split, split_meta)
    w_split = _restore_split_col_order_with_meta(w_split, split_meta)
    return w_split.contiguous().to(dtype=dtype)


def _compressed_weights_to_group_data(
    *,
    compressed_weights: Sequence[torch.Tensor],
    split_metas: Sequence[object],
    codebook_dim: int,
) -> torch.Tensor:
    if len(compressed_weights) != len(split_metas):
        raise ValueError(
            f"compressed_weights length {len(compressed_weights)} != split_metas {len(split_metas)}"
        )
    per_linear_flat = []
    for weight, split_meta in zip(compressed_weights, split_metas):
        expected_shape = (
            int(split_meta.compressed_out_features),
            int(split_meta.compressed_in_features),
        )
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"{split_meta.linear_name}: compressed weight shape mismatch, got {tuple(weight.shape)}, expected {expected_shape}"
            )
        per_linear_flat.append(
            _split_weight_into_part_flats(
                weight=weight,
                transpose=bool(split_meta.transpose),
                parallel_rows=int(split_meta.parallel_rows),
                parallel_cols=int(split_meta.parallel_cols),
            )
        )
    stacked_flat = torch.stack(per_linear_flat, dim=0).reshape(-1, per_linear_flat[0].shape[1]).contiguous()
    total_numel = int(stacked_flat.shape[1])
    if total_numel % int(codebook_dim) != 0:
        raise ValueError(
            f"flatten_len={total_numel} not divisible by codebook_dim={int(codebook_dim)}"
        )
    return stacked_flat.view(stacked_flat.shape[0], -1, int(codebook_dim)).permute(1, 0, 2).contiguous()


def _group_data_to_compressed_weights(
    *,
    stacked_data: torch.Tensor,
    split_metas: Sequence[object],
) -> List[torch.Tensor]:
    if stacked_data.ndim != 3:
        raise ValueError(f"stacked_data must be 3D [N_blocks, P, C], got shape={tuple(stacked_data.shape)}")
    if len(split_metas) == 0:
        return []
    parts_per_linear = int(split_metas[0].parallel_rows) * int(split_metas[0].parallel_cols)
    flat = stacked_data.permute(1, 0, 2).contiguous().view(int(stacked_data.shape[1]), -1)
    weights: List[torch.Tensor] = []
    for linear_idx, split_meta in enumerate(split_metas):
        start = linear_idx * parts_per_linear
        end = start + parts_per_linear
        part_flats = flat[start:end]
        w_split = _restore_split_weight_from_part_flats_with_meta(
            part_flats=part_flats,
            split_meta=split_meta,
            dtype=stacked_data.dtype,
        )
        weight = w_split.t().contiguous() if bool(split_meta.transpose) else w_split.contiguous()
        weights.append(weight)
    return weights


def _extract_linear_weight_from_group_stacked(
    *,
    group_tag: str,
    linear_name: str,
    linear_idx: int,
    stacked_data: torch.Tensor,
    split_meta: object,
    parts_per_linear: int,
    stage_idx: int,
) -> torch.Tensor:
    if stacked_data.ndim != 3:
        raise ValueError(
            f"[{group_tag}] final residual stacked_data must be 3D after base stage_idx={stage_idx}, "
            f"got shape={tuple(stacked_data.shape)} for linear={linear_name}."
        )
    if int(parts_per_linear) < 1:
        raise ValueError(f"[{group_tag}] parts_per_linear must be >= 1, got {parts_per_linear}.")
    expected_parts = int(split_meta.parallel_rows) * int(split_meta.parallel_cols)
    if expected_parts != int(parts_per_linear):
        raise ValueError(
            f"[{group_tag}] parts mismatch for linear={linear_name} after base stage_idx={stage_idx}: "
            f"split_meta={split_meta.parallel_rows}x{split_meta.parallel_cols}, "
            f"parts_per_linear={parts_per_linear}."
        )
    start_idx = int(linear_idx) * int(parts_per_linear)
    end_idx = start_idx + int(parts_per_linear)
    if end_idx > int(stacked_data.shape[1]):
        raise ValueError(
            f"[{group_tag}] final residual model/part range out of bounds for linear={linear_name} "
            f"after base stage_idx={stage_idx}: start={start_idx} end={end_idx} "
            f"stacked_shape={tuple(stacked_data.shape)}."
        )
    flat = stacked_data.permute(1, 0, 2).contiguous().view(int(stacked_data.shape[1]), -1)
    part_flats = flat[start_idx:end_idx]
    w_split = _restore_split_weight_from_part_flats_with_meta(
        part_flats=part_flats,
        split_meta=split_meta,
        dtype=stacked_data.dtype,
    )
    return w_split.t().contiguous() if bool(split_meta.transpose) else w_split.contiguous()


def _build_protected_residual_entries_from_final_residual(
    *,
    group_tag: str,
    group_refs: Sequence[LinearRef],
    target_common_split_metas: Sequence[object],
    final_common_stacked: torch.Tensor,
    parts_per_linear: int,
    outlier_protect_axis: str,
    final_stage_idx: int,
    codebook_dim: Optional[int] = None,
) -> List[Tuple[LinearRef, str, torch.Tensor, torch.Tensor]]:
    if len(group_refs) != len(target_common_split_metas):
        raise ValueError(
            f"[{group_tag}] group_data/final residual metadata mismatch after base stage_idx={final_stage_idx}: "
            f"group_refs={len(group_refs)} split_metas={len(target_common_split_metas)}."
        )
    if final_common_stacked.ndim != 3:
        raise ValueError(
            f"[{group_tag}] final residual must be 3D [N_blocks, models, codebook_dim] after "
            f"base stage_idx={final_stage_idx}, got shape={tuple(final_common_stacked.shape)}."
        )
    expected_models = int(len(group_refs)) * int(parts_per_linear)
    if int(final_common_stacked.shape[1]) != expected_models:
        raise ValueError(
            f"[{group_tag}] final residual model/part count mismatch after base stage_idx={final_stage_idx}: "
            f"stacked_models={int(final_common_stacked.shape[1])} expected={expected_models} "
            f"group_refs={len(group_refs)} parts_per_linear={parts_per_linear} "
            f"stacked_shape={tuple(final_common_stacked.shape)}."
        )

    axis_name = str(outlier_protect_axis).strip().lower()
    if axis_name not in {"input", "output"}:
        raise ValueError(f"[{group_tag}] unsupported outlier_protect_axis={outlier_protect_axis!r}.")

    protected_residual_entries: List[Tuple[LinearRef, str, torch.Tensor, torch.Tensor]] = []
    for linear_idx, (r, split_meta) in enumerate(zip(group_refs, target_common_split_metas)):
        if str(split_meta.linear_name) != str(r.name):
            raise ValueError(
                f"[{group_tag}] split metadata order mismatch after base stage_idx={final_stage_idx}: "
                f"idx={linear_idx} meta={split_meta.linear_name} ref={r.name}."
            )
        residual_weight = _extract_linear_weight_from_group_stacked(
            group_tag=group_tag,
            linear_name=r.name,
            linear_idx=int(linear_idx),
            stacked_data=final_common_stacked,
            split_meta=split_meta,
            parts_per_linear=int(parts_per_linear),
            stage_idx=int(final_stage_idx),
        ).to(device="cpu", dtype=torch.float32).contiguous()
        expected_shape = (int(r.module.out_features), int(r.module.in_features))
        if tuple(residual_weight.shape) != expected_shape:
            raise ValueError(
                f"[{group_tag}] final residual shape mismatch for linear={r.name} "
                f"after base stage_idx={final_stage_idx}: got={tuple(residual_weight.shape)} "
                f"expected={expected_shape}."
            )

        if axis_name == "output":
            protected_idx = split_meta.protected_output_indices
            axis = "output"
            axis_dim = 0
        else:
            protected_idx = split_meta.protected_input_indices
            axis = "input"
            axis_dim = 1
        if not isinstance(protected_idx, torch.Tensor) or int(protected_idx.numel()) == 0:
            continue
        idx_cpu = protected_idx.detach().to(device="cpu", dtype=torch.long).contiguous()
        min_idx = int(idx_cpu.min().item())
        max_idx = int(idx_cpu.max().item())
        axis_size = int(residual_weight.shape[axis_dim])
        if min_idx < 0 or max_idx >= axis_size:
            raise ValueError(
                f"[{group_tag}] protected residual index out of bounds for linear={r.name} "
                f"after base stage_idx={final_stage_idx}: axis={axis} axis_size={axis_size} "
                f"idx_size={int(idx_cpu.numel())} min_idx={min_idx} max_idx={max_idx} "
                f"tensor_shape={tuple(residual_weight.shape)}."
            )
        residual_slice = residual_weight.index_select(axis_dim, idx_cpu).contiguous()
        if residual_slice.ndim != 2:
            raise ValueError(
                f"[{group_tag}] protected residual slice must be 2D for linear={r.name} "
                f"after base stage_idx={final_stage_idx}: axis={axis} idx_size={int(idx_cpu.numel())} "
                f"slice_shape={tuple(residual_slice.shape)} tensor_shape={tuple(residual_weight.shape)}."
            )
        if codebook_dim is not None and int(residual_slice.numel()) % int(codebook_dim) != 0:
            raise ValueError(
                f"[{group_tag}] protected residual slice numel is not divisible by codebook_dim for "
                f"linear={r.name} after base stage_idx={final_stage_idx}: axis={axis} "
                f"idx_size={int(idx_cpu.numel())} slice_shape={tuple(residual_slice.shape)} "
                f"codebook_dim={int(codebook_dim)} tensor_shape={tuple(residual_weight.shape)}."
            )
        protected_residual_entries.append((r, axis, idx_cpu, residual_slice))
    return protected_residual_entries


def _select_channel_residual_vae_plan_from_final_residual(
    *,
    group_tag: str,
    group_refs: Sequence[LinearRef],
    target_common_split_metas: Sequence[object],
    final_common_stacked: torch.Tensor,
    parts_per_linear: int,
    outlier_protect_count: int,
    outlier_protect_axis: str,
    outlier_channel_scope: str,
    outlier_rank_metric: str,
    activation_weight_by_linear: Optional[Dict[str, torch.Tensor]],
    activation_abs_mean_by_linear: Optional[Dict[str, torch.Tensor]],
    activation_sq_mean_by_linear: Optional[Dict[str, torch.Tensor]],
    final_stage_idx: int,
    outlier_protect_min_per_layer: int = 0,
) -> Tuple[List[object], Dict[str, float]]:
    protect_count = int(outlier_protect_count)
    if protect_count < 0:
        raise ValueError(f"[{group_tag}] outlier_protect_count must be >= 0, got {protect_count}.")
    axis = str(outlier_protect_axis).strip().lower()
    if axis not in {"input", "output"}:
        raise ValueError(f"[{group_tag}] unsupported outlier_protect_axis={outlier_protect_axis!r}.")
    scope = str(outlier_channel_scope).strip().lower()
    if scope not in {"layer", "category"}:
        raise ValueError(f"[{group_tag}] unsupported outlier_channel_scope={outlier_channel_scope!r}.")
    if len(group_refs) != len(target_common_split_metas):
        raise ValueError(
            f"[{group_tag}] cannot select channel residual VAE plan: "
            f"group_refs={len(group_refs)} split_metas={len(target_common_split_metas)}."
        )

    empty_plan = {
        r.name: torch.empty(0, dtype=torch.long)
        for r in group_refs
    }
    per_linear_scores: Dict[str, torch.Tensor] = {}
    for linear_idx, (r, split_meta) in enumerate(zip(group_refs, target_common_split_metas)):
        residual_weight = _extract_linear_weight_from_group_stacked(
            group_tag=group_tag,
            linear_name=r.name,
            linear_idx=int(linear_idx),
            stacked_data=final_common_stacked,
            split_meta=split_meta,
            parts_per_linear=int(parts_per_linear),
            stage_idx=int(final_stage_idx),
        ).to(device="cpu", dtype=torch.float32).contiguous()
        expected_shape = (int(r.module.out_features), int(r.module.in_features))
        if tuple(residual_weight.shape) != expected_shape:
            raise ValueError(
                f"[{group_tag}] final residual shape mismatch before channel ranking for linear={r.name}: "
                f"got={tuple(residual_weight.shape)} expected={expected_shape} "
                f"stage_idx={final_stage_idx} axis={axis}."
            )
        act_max = None if activation_weight_by_linear is None else activation_weight_by_linear.get(r.name)
        act_mean = None if activation_abs_mean_by_linear is None else activation_abs_mean_by_linear.get(r.name)
        act_sq_mean = None if activation_sq_mean_by_linear is None else activation_sq_mean_by_linear.get(r.name)
        per_linear_scores[r.name] = compute_channel_rank_score(
            metric=outlier_rank_metric,
            weight=r.module.weight.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            residual=residual_weight,
            act_max=act_max,
            act_mean=act_mean,
            act_sq_mean=act_sq_mean,
            axis=axis,
            transpose=bool(r.transpose),
            linear_name=r.name,
            expected_in_features=int(r.module.in_features),
            expected_out_features=int(r.module.out_features),
        )

    selected_plan, selection_stats = select_outlier_channel_indices_from_scores(
        scores_by_name=per_linear_scores,
        linear_names=[r.name for r in group_refs],
        outlier_protect_count=int(protect_count),
        outlier_protect_min_per_layer=int(outlier_protect_min_per_layer),
        outlier_channel_scope=scope,
    )
    if protect_count == 0 or not per_linear_scores:
        selected_plan = empty_plan

    updated_split_metas: List[object] = []
    for r, split_meta in zip(group_refs, target_common_split_metas):
        idx = selected_plan.get(r.name, torch.empty(0, dtype=torch.long)).detach().to(
            device="cpu",
            dtype=torch.long,
        ).contiguous()
        if axis == "input":
            updated_split_metas.append(
                replace(
                    split_meta,
                    protected_input_indices=idx,
                    protected_input_weight=None,
                    protected_output_indices=None,
                    protected_output_weight=None,
                )
            )
        else:
            updated_split_metas.append(
                replace(
                    split_meta,
                    protected_input_indices=None,
                    protected_input_weight=None,
                    protected_output_indices=idx,
                    protected_output_weight=None,
                )
            )

    score_values = torch.cat([score.detach().to(device="cpu", dtype=torch.float32).reshape(-1) for score in per_linear_scores.values()])
    selected_count = sum(int(idx.numel()) for idx in selected_plan.values())
    return updated_split_metas, {
        "num_channels": float(int(score_values.numel())),
        "topk": float(int(selected_count)),
        "score_max": float(score_values.max().item()) if int(score_values.numel()) else 0.0,
        "score_mean": float(score_values.mean().item()) if int(score_values.numel()) else 0.0,
        "min_per_layer": float(selection_stats["min_per_layer"]),
        "floor_selected_count": float(selection_stats["floor_selected_count"]),
        "global_selected_count": float(selection_stats["global_selected_count"]),
        "num_zero_protected_linears": float(selection_stats["num_zero_protected_linears"]),
    }


def _convert_stage_stacked_to_common_stacked(
    *,
    stage_stacked_data: torch.Tensor,
    stage_split_metas: Sequence[object],
    common_split_metas: Sequence[object],
    codebook_dim: int,
) -> torch.Tensor:
    if len(stage_split_metas) != len(common_split_metas):
        raise ValueError(
            f"stage_split_metas length {len(stage_split_metas)} != common_split_metas {len(common_split_metas)}"
        )
    if len(stage_split_metas) == 0:
        raise ValueError("stage_split_metas cannot be empty.")
    parts_per_linear = int(stage_split_metas[0].parallel_rows) * int(stage_split_metas[0].parallel_cols)
    stage_flat = stage_stacked_data.permute(1, 0, 2).contiguous().view(int(stage_stacked_data.shape[1]), -1)
    compressed_weights: List[torch.Tensor] = []
    for linear_idx, (stage_meta, common_meta) in enumerate(zip(stage_split_metas, common_split_metas)):
        start = linear_idx * parts_per_linear
        end = start + parts_per_linear
        part_flats = stage_flat[start:end]
        restored_split = _restore_split_weight_from_part_flats_with_meta(
            part_flats=part_flats,
            split_meta=stage_meta,
            dtype=stage_stacked_data.dtype,
        )
        compressed_weight = restored_split.t().contiguous() if bool(stage_meta.transpose) else restored_split.contiguous()
        expected_shape = (
            int(common_meta.compressed_out_features),
            int(common_meta.compressed_in_features),
        )
        if tuple(compressed_weight.shape) != expected_shape:
            raise ValueError(
                f"{stage_meta.linear_name}: common compressed weight shape mismatch, got {tuple(compressed_weight.shape)}, expected {expected_shape}"
            )
        compressed_weights.append(compressed_weight)
    return _compressed_weights_to_group_data(
        compressed_weights=compressed_weights,
        split_metas=common_split_metas,
        codebook_dim=int(codebook_dim),
    )


def _resolve_train_dtype(training_args) -> torch.dtype:
    if bool(getattr(training_args, "bf16", False)):
        return torch.bfloat16
    if bool(getattr(training_args, "fp16", False)):
        return torch.float16
    return torch.float32


def _collect_current_trainable_linears(
    model: nn.Module,
    *,
    transpose_modules: Sequence[str],
    only_decoder_projections: bool,
    target_categories: Sequence[str],
) -> List[LinearRef]:
    return _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        target_categories=target_categories,
    )


def _collect_sorted_category_refs(
    model: nn.Module,
    *,
    category: str,
    transpose_modules: Sequence[str],
    only_decoder_projections: bool,
    target_categories: Sequence[str],
) -> Tuple[List[Tuple[int, LinearRef]], int]:
    refs_sorted: List[Tuple[int, LinearRef]] = []
    missing = 0
    for ref in _collect_current_trainable_linears(
        model,
        transpose_modules=transpose_modules,
        only_decoder_projections=only_decoder_projections,
        target_categories=target_categories,
    ):
        if ref.category != category:
            continue
        layer_idx = _extract_layer_idx(ref.name)
        if layer_idx is None:
            missing += 1
            continue
        refs_sorted.append((layer_idx, ref))
    refs_sorted.sort(key=lambda item: item[0])
    return refs_sorted, missing


def _build_vae_linear_from_stage_payload(
    *,
    old_module: nn.Module,
    transpose: bool,
    split_meta,
    stage_split_metas: Sequence[object],
    stage_part_bits_payload: Sequence[object],
    stage_part_decoders_payload: Sequence[object],
    stage_codebook_dims: Sequence[int],
    parallel_rows: int,
    parallel_cols: int,
    parallel_parts: int,
    bias,
    original_weight,
    always_use_original: bool,
    protect_original_weight: bool,
    sparse_residual_kwargs: Optional[Dict[str, object]] = None,
    protected_residual_kwargs: Optional[Dict[str, object]] = None,
):
    from litebsq.vae_linear import VAELinear

    residual_stages = int(len(stage_part_bits_payload))
    if residual_stages < 1:
        raise ValueError("stage_part_bits_payload cannot be empty.")
    common_kwargs = dict(
        in_features=old_module.in_features,
        out_features=old_module.out_features,
        bias=bias,
        original_weight=original_weight,
        codebook_dim=int(stage_codebook_dims[0]),
        stage_codebook_dims=list(int(v) for v in stage_codebook_dims),
        transpose=bool(transpose),
        parallel_parts=int(parallel_parts),
        parallel_rows=int(parallel_rows),
        parallel_cols=int(parallel_cols),
        compressed_in_features=int(split_meta.compressed_in_features),
        compressed_out_features=int(split_meta.compressed_out_features),
        protected_input_indices=(
            split_meta.protected_input_indices
            if split_meta.protected_input_weight is not None
            else None
        ),
        protected_input_weight=split_meta.protected_input_weight,
        protected_output_indices=(
            split_meta.protected_output_indices
            if split_meta.protected_output_weight is not None
            else None
        ),
        protected_output_weight=split_meta.protected_output_weight,
        always_use_original=bool(always_use_original),
        protect_original_weight=bool(protect_original_weight),
    )
    if sparse_residual_kwargs:
        common_kwargs.update(dict(sparse_residual_kwargs))
    if protected_residual_kwargs:
        common_kwargs.update(dict(protected_residual_kwargs))
    if residual_stages == 1:
        return VAELinear(
            vq_weight=stage_part_bits_payload[0],
            decoder=stage_part_decoders_payload[0],
            **common_kwargs,
        )
    return VAELinear(
        vq_weight=None,
        decoder=None,
        stage_vq_weights=list(stage_part_bits_payload),
        stage_decoders=list(stage_part_decoders_payload),
        **common_kwargs,
    )


def _decode_reconstructed_linear_weight(
    *,
    old_module: nn.Module,
    transpose: bool,
    split_meta,
    stage_split_metas: Sequence[object],
    stage_part_bits_payload: Sequence[object],
    stage_part_decoders_payload: Sequence[object],
    stage_codebook_dims: Sequence[int],
    parallel_rows: int,
    parallel_cols: int,
    parallel_parts: int,
) -> torch.Tensor:
    temp_linear = _build_vae_linear_from_stage_payload(
        old_module=old_module,
        transpose=transpose,
        split_meta=split_meta,
        stage_split_metas=stage_split_metas,
        stage_part_bits_payload=stage_part_bits_payload,
        stage_part_decoders_payload=stage_part_decoders_payload,
        stage_codebook_dims=stage_codebook_dims,
        parallel_rows=parallel_rows,
        parallel_cols=parallel_cols,
        parallel_parts=parallel_parts,
        bias=None,
        original_weight=None,
        always_use_original=False,
        protect_original_weight=False,
    )
    return temp_linear._decode_weight(dtype=torch.float32).detach().to(device="cpu", dtype=torch.float32)


def _train_protected_residual_vae_payload(
    *,
    linear_name: str,
    residual_slice: torch.Tensor,
    runtime_cfg: ResolvedCategoryRuntimeConfig,
    vae_args,
    training_args,
    train_device: str,
    train_dtype: torch.dtype,
    batch_size: int,
    steps: int,
    lr: float,
    log_every: int,
    deterministic: bool,
    shuffle_seed: int,
) -> Optional[Dict[str, object]]:
    from litebsq.llm_vae import MultiLayerVAE

    protected_steps = int(steps)
    if protected_steps <= 0:
        return None
    protected_lr = float(lr)
    if protected_lr <= 0.0:
        raise ValueError(f"{linear_name}: protected residual VAE lr must be > 0, got {protected_lr}.")
    recon_loss = str(runtime_cfg.recon_loss_type).strip().lower()
    codebook_dim = int(runtime_cfg.outlier_residual_vae_codebook_dim)
    stages = int(runtime_cfg.outlier_residual_vae_stages)
    residual = residual_slice.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if residual.ndim != 2:
        raise ValueError(f"{linear_name}: protected residual slice must be 2D, got shape={tuple(residual.shape)}.")
    if int(residual.numel()) == 0:
        return None
    if int(residual.numel()) % codebook_dim != 0:
        raise ValueError(
            f"{linear_name}: protected residual numel={int(residual.numel())} is not divisible by "
            f"codebook_dim={codebook_dim}."
        )

    current = residual.view(-1, 1, codebook_dim).contiguous()
    initial_rms = float(current.float().pow(2).mean().sqrt().item())
    stage_bits: List[torch.Tensor] = []
    stage_decoders: List[nn.Module] = []
    stage_codebook_dims: List[int] = []
    last_loss = None
    last_recon = None
    last_commit = None
    shared_stage_args = _clone_namespace(
        vae_args,
        parallel_layers=1,
        residual_stages=int(stages),
        codebook_bits=int(runtime_cfg.outlier_residual_vae_codebook_bits),
        codebook_dim=int(codebook_dim),
        base_ch=int(runtime_cfg.base_ch),
        num_res_blocks=int(runtime_cfg.num_res_blocks),
        norm_type=str(runtime_cfg.norm_type),
        decoder_type=str(runtime_cfg.decoder_type),
        decoder_base_ch=(
            None if runtime_cfg.decoder_base_ch is None else int(runtime_cfg.decoder_base_ch)
        ),
        decoder_num_res_blocks=(
            None if runtime_cfg.decoder_num_res_blocks is None else int(runtime_cfg.decoder_num_res_blocks)
        ),
        recon_loss_type=recon_loss,
    )
    apply_autoencoder_arch_defaults(shared_stage_args)
    use_stage_norm = bool(getattr(shared_stage_args, "normalize_weight", False))

    for stage_idx in range(stages):
        stage_tag = f"{linear_name}/protected_residual_stage{stage_idx + 1}"
        residual_data = current.detach().clone().contiguous()
        if use_stage_norm:
            stage_norm_mean, stage_norm_scale = _compute_stage_norm_stats(residual_data)
            stage_train_data = _apply_stage_norm(residual_data, mean=stage_norm_mean, scale=stage_norm_scale)
        else:
            stage_norm_mean = None
            stage_norm_scale = None
            stage_train_data = residual_data

        effective_batch_size = min(max(1, int(batch_size)), int(stage_train_data.shape[0]))
        train_loader, eval_loader = _build_block_data_loaders(
            stage_train_data,
            batch_size=int(effective_batch_size),
            shuffle_seed=int(shuffle_seed) + int(stage_idx) if bool(deterministic) else None,
        )
        vae = MultiLayerVAE(shared_stage_args).to(train_device)
        optimizer = create_optimizer(vae.parameters(), shared_stage_args, protected_lr)
        lr_scheduler = None
        lr_scheduler_name = str(getattr(shared_stage_args, "lr_scheduler", "constant")).strip().lower()
        if lr_scheduler_name != "constant":
            import transformers

            lr_scheduler = transformers.get_scheduler(
                lr_scheduler_name,
                optimizer,
                num_warmup_steps=int(getattr(shared_stage_args, "lr_warmup_steps", 0)),
                num_training_steps=int(protected_steps),
            )
        start = time.time()
        train_iter = iter(train_loader)
        vae.train()
        for step in range(int(protected_steps)):
            try:
                x_cpu, _idx = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_cpu, _idx = next(train_iter)
            x = x_cpu.to(device=train_device, dtype=train_dtype, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _x_recon, loss_dict = vae(x, is_train=True)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            last_loss = float(loss.detach().float().item())
            recon_value = loss_dict.get("train/recon_loss")
            if isinstance(recon_value, torch.Tensor):
                last_recon = float(recon_value.detach().float().item())
            commit_value = loss_dict.get("train/commitment_loss")
            if isinstance(commit_value, torch.Tensor):
                last_commit = float(commit_value.detach().float().item())
            if log_every > 0 and (step + 1) % int(log_every) == 0:
                speed = (time.time() - start) / int(log_every)
                log.info(
                    "[%s] step=%d/%d loss=%.4e speed=%.4fs/it",
                    stage_tag,
                    step + 1,
                    int(protected_steps),
                    float(loss.detach().float().item()),
                    speed,
                )
                start = time.time()

        vae.eval()
        recon_chunks: List[torch.Tensor] = []
        bit_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for x_cpu, _idx in eval_loader:
                x = x_cpu.to(device=train_device, dtype=train_dtype, non_blocking=True)
                x_recon, bit_idx = vae(x, is_train=False)
                recon_chunks.append(x_recon.detach().to(device="cpu", dtype=stage_train_data.dtype))
                bit_chunks.append(bit_idx.detach().to(device="cpu"))
        stage_recon_norm = torch.cat(recon_chunks, dim=0)
        stage_recon = (
            _restore_stage_norm(stage_recon_norm, mean=stage_norm_mean, scale=stage_norm_scale)
            if stage_norm_mean is not None and stage_norm_scale is not None
            else stage_recon_norm
        )
        current = (current - stage_recon.to(dtype=current.dtype)).contiguous()
        stage_bits.append(torch.cat(bit_chunks, dim=0).contiguous())
        stage_codebook_dims.append(int(codebook_dim))

        decoder_in_dim = int(getattr(vae.model.decoder, "in_dim"))
        use_new_quant = bool(getattr(shared_stage_args, "new_quant", False))
        quant_q_scale = (1.0 / math.sqrt(decoder_in_dim)) if use_new_quant else 1.0
        dec = vae.model.decoder.get_sub_decoder(0)
        _fuse_q_scale_into_decoder(dec, q_scale=float(quant_q_scale))
        if use_stage_norm:
            if stage_norm_mean is None or stage_norm_scale is None:
                raise RuntimeError(f"{stage_tag}: stage norm stats missing while normalize_weight=True")
            _fuse_norm_into_decoder(
                dec,
                mean=float(stage_norm_mean[0].item()),
                std=float(stage_norm_scale[0].item()),
            )
        dec.to("cpu")
        stage_decoders.append(dec)
        del vae, train_loader, eval_loader, optimizer
        if lr_scheduler is not None:
            del lr_scheduler
        torch.cuda.empty_cache()

    return {
        "stage_vq_weights": stage_bits,
        "stage_decoders": stage_decoders,
        "stage_codebook_dims": stage_codebook_dims,
        "metrics": {
            "protected_residual_rms_before": float(initial_rms),
            "protected_residual_rms_after": float(current.float().pow(2).mean().sqrt().item()),
            "residual_vae_final_loss": last_loss,
            "residual_vae_final_recon": last_recon,
            "residual_vae_final_commit": last_commit,
        },
    }


def _train_shared_protected_residual_vae_payloads(
    *,
    group_tag: str,
    residual_slices_by_name: Dict[str, torch.Tensor],
    runtime_cfg: ResolvedCategoryRuntimeConfig,
    vae_args,
    training_args,
    train_device: str,
    train_dtype: torch.dtype,
    batch_size: int,
    steps: int,
    lr: float,
    log_every: int,
    deterministic: bool,
    shuffle_seed: int,
) -> Dict[str, Dict[str, object]]:
    if not residual_slices_by_name:
        return {}
    codebook_dim = int(runtime_cfg.outlier_residual_vae_codebook_dim)
    ordered_items = list(residual_slices_by_name.items())
    block_counts: Dict[str, int] = {}
    flat_chunks: List[torch.Tensor] = []
    for linear_name, residual_slice in ordered_items:
        residual = residual_slice.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if residual.ndim != 2:
            raise ValueError(f"{linear_name}: protected residual slice must be 2D, got shape={tuple(residual.shape)}.")
        if int(residual.numel()) == 0:
            continue
        if int(residual.numel()) % codebook_dim != 0:
            raise ValueError(
                f"{linear_name}: protected residual numel={int(residual.numel())} is not divisible by "
                f"codebook_dim={codebook_dim}."
            )
        blocks = int(residual.numel()) // int(codebook_dim)
        block_counts[linear_name] = blocks
        flat_chunks.append(residual.view(blocks, codebook_dim))
    if not flat_chunks:
        return {}

    combined = torch.cat(flat_chunks, dim=0).contiguous()
    shared_payload = _train_protected_residual_vae_payload(
        linear_name=f"{group_tag}/shared_protected_residual",
        residual_slice=combined,
        runtime_cfg=runtime_cfg,
        vae_args=vae_args,
        training_args=training_args,
        train_device=train_device,
        train_dtype=train_dtype,
        batch_size=batch_size,
        steps=int(steps),
        lr=float(lr),
        log_every=log_every,
        deterministic=deterministic,
        shuffle_seed=shuffle_seed,
    )
    if shared_payload is None:
        return {}

    stage_bits_all = shared_payload["stage_vq_weights"]
    shared_stage_decoders = shared_payload["stage_decoders"]
    stage_codebook_dims = shared_payload["stage_codebook_dims"]
    refs = [
        _safe_shared_decoder_ref(f"{group_tag}.protected_residual.stage{stage_idx}")
        for stage_idx in range(len(shared_stage_decoders))
    ]

    out: Dict[str, Dict[str, object]] = {}
    offset = 0
    for linear_name, _residual_slice in ordered_items:
        blocks = int(block_counts.get(linear_name, 0))
        if blocks <= 0:
            continue
        per_linear_stage_bits = [
            stage_bits[offset: offset + blocks].contiguous()
            for stage_bits in stage_bits_all
        ]
        offset += blocks
        out[linear_name] = {
            "stage_vq_weights": per_linear_stage_bits,
            "shared_decoder_refs": list(refs),
            "shared_stage_decoders": shared_stage_decoders,
            "stage_codebook_dims": stage_codebook_dims,
            "metrics": shared_payload.get("metrics"),
        }
    return out


def train_group_vae_payload(
    *,
    model: nn.Module,
    group_refs: Sequence[LinearRef],
    group_tag: str,
    runtime_cfg: ResolvedCategoryRuntimeConfig,
    vae_args,
    training_args,
    train_device: str,
    convert_device: str,
    do_convert: bool,
    batch_size: object,
    log_every: int,
    eval_every: int,
    eval_blocks: int,
    gpu_resident_data: bool = False,
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
    activation_runtime: Optional[Dict[str, object]] = None,
    outlier_protect_mode: str = "channel",
    outlier_channel_plan: Optional[Dict[str, torch.Tensor]] = None,
    outlier_channel_scope: str = "layer",
    outlier_rank_metric: str = "sparse_residual_abs",
    outlier_residual_min_abs: float = 1e-6,
    outlier_protect_axis: str = "input",
    outlier_protect_min_per_layer: int = 0,
    outlier_residual_codec: str = SPARSE_RESIDUAL_FORMAT_COO_FP16,
    outlier_residual_index_bits: int = 8,
    outlier_residual_value_bits: int = 8,
    outlier_residual_block_shape: Tuple[int, int] = (256, 256),
    outlier_residual_vae_decoder_share_scope: str = "none",
    outlier_residual_vae_batch_multiplier: int = 1,
    outlier_residual_vae_steps: int = 0,
    outlier_residual_vae_lr: float = 0.0,
    # 排序代码，已关闭。旧参数保留如下：
    # sort_executor=None,
    # sort_prep_workers_resolved: int = 1,
    deterministic: bool = False,
    shuffle_seed: int = 0,
) -> Optional[Dict[str, Any]]:
    from litebsq.llm_vae import MultiLayerVAE

    batch_size_text = str(batch_size).strip().lower()
    batch_size_is_all = batch_size_text == "all"
    if batch_size_is_all:
        materialize_batch_size = 8192
    else:
        materialize_batch_size = int(batch_size)
        if int(materialize_batch_size) < 1:
            raise ValueError(f"[{group_tag}] batch_size must be >= 1 or 'all', got {batch_size!r}.")
    residual_vae_batch_multiplier = int(outlier_residual_vae_batch_multiplier)
    if residual_vae_batch_multiplier < 1:
        raise ValueError(
            f"[{group_tag}] outlier_residual_vae_batch_multiplier must be >= 1, "
            f"got {residual_vae_batch_multiplier}."
        )
    protected_residual_vae_batch_size = int(materialize_batch_size) * int(residual_vae_batch_multiplier)
    requested_residual_vae_steps = int(outlier_residual_vae_steps)
    requested_residual_vae_lr = float(outlier_residual_vae_lr)
    if requested_residual_vae_steps < 0:
        raise ValueError(
            f"[{group_tag}] outlier_residual_vae_steps must be >= 0, got {requested_residual_vae_steps}."
        )
    if requested_residual_vae_lr < 0.0:
        raise ValueError(
            f"[{group_tag}] outlier_residual_vae_lr must be >= 0, got {requested_residual_vae_lr}."
        )

    train_dtype = _resolve_train_dtype(training_args)

    residual_stages = int(runtime_cfg.residual_stages)
    if residual_stages < 1:
        raise ValueError(f"residual_stages must be >= 1, got {residual_stages}")
    if len(group_refs) == 0:
        raise ValueError(f"[{group_tag}] group_refs cannot be empty.")

    stage_sort_mode = runtime_cfg.intra_part_sort_mode
    stage_codebook_bits = int(runtime_cfg.codebook_bits)
    stage_codebook_dim = int(runtime_cfg.codebook_dim)
    stage_steps = int(runtime_cfg.steps)
    stage_recon_loss = str(runtime_cfg.recon_loss_type).strip().lower()
    stage_base_ch = int(runtime_cfg.base_ch)
    stage_num_res_blocks = int(runtime_cfg.num_res_blocks)
    stage_norm_type = str(runtime_cfg.norm_type).strip().lower()
    stage_decoder_type = str(runtime_cfg.decoder_type).strip().lower()
    protected_residual_vae_steps = (
        int(requested_residual_vae_steps)
        if int(requested_residual_vae_steps) > 0
        else int(stage_steps)
    )
    protected_residual_vae_lr = (
        float(requested_residual_vae_lr)
        if float(requested_residual_vae_lr) > 0.0
        else float(getattr(vae_args, "lr"))
    )
    protected_residual_vae_codebook_bits = int(runtime_cfg.outlier_residual_vae_codebook_bits)
    protected_residual_vae_codebook_dim = int(runtime_cfg.outlier_residual_vae_codebook_dim)
    outlier_protect_count = int(runtime_cfg.outlier_protect_count)
    outlier_residual_top_p = float(runtime_cfg.outlier_residual_top_p)
    resolved_outlier_mode = str(outlier_protect_mode).strip().lower()
    resolved_protected_residual_decoder_share_scope = str(outlier_residual_vae_decoder_share_scope).strip().lower()
    resolved_outlier_rank_metric = str(outlier_rank_metric).strip().lower()
    resolved_residual_min_abs = float(outlier_residual_min_abs)
    residual_sparse_enabled = resolved_outlier_mode == "residual_sparse"
    channel_protection_enabled = resolved_outlier_mode in {
        "channel", "channel_residual_vae"} and int(outlier_protect_count) > 0
    residual_sparse_needs_activation = (
        residual_sparse_enabled
        and (
            resolved_outlier_rank_metric in _RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX
            or resolved_outlier_rank_metric in _RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMEAN
        )
    )
    channel_rank_needs_actmax = (
        channel_protection_enabled
        and resolved_outlier_rank_metric in {"channel_weight_actmax_abs", "channel_residual_actmax_abs"}
    )
    channel_rank_needs_actmean = (
        channel_protection_enabled
        and resolved_outlier_rank_metric in {"channel_weight_actmean_abs", "channel_residual_actmean_abs"}
    )
    channel_rank_needs_sq_mean = (
        channel_protection_enabled
        and resolved_outlier_rank_metric == "channel_residual_actrms_abs"
    )
    if resolved_outlier_mode not in {"none", "channel", "channel_residual_vae", "residual_sparse"}:
        raise ValueError(
            f"[{group_tag}] unsupported outlier_protect_mode={outlier_protect_mode!r}. "
            "Expected none, channel, channel_residual_vae, or residual_sparse."
        )
    if resolved_protected_residual_decoder_share_scope not in {"none", "category"}:
        raise ValueError(
            f"[{group_tag}] unsupported outlier_residual_vae_decoder_share_scope="
            f"{outlier_residual_vae_decoder_share_scope!r}. Expected one of: none, category."
        )
    if resolved_outlier_mode == "channel_residual_vae" and int(outlier_protect_count) > 0:
        log.info(
            "[%s] protected residual VAE schedule: steps=%d lr=%.6g batch=%d share_scope=%s codebook_bits=%d codebook_dim=%d",
            group_tag,
            int(protected_residual_vae_steps),
            float(protected_residual_vae_lr),
            int(protected_residual_vae_batch_size),
            resolved_protected_residual_decoder_share_scope,
            int(protected_residual_vae_codebook_bits),
            int(protected_residual_vae_codebook_dim),
        )
    if residual_sparse_enabled and int(outlier_protect_count) != 0:
        raise ValueError(
            f"[{group_tag}] residual_sparse mode requires outlier_protect_count=0, got {outlier_protect_count}."
        )
    if residual_sparse_enabled and not (0.0 < outlier_residual_top_p <= 1.0):
        raise ValueError(
            f"[{group_tag}] residual_sparse mode requires 0 < outlier_residual_top_p <= 1, "
            f"got {outlier_residual_top_p}."
        )
    if resolved_residual_min_abs < 0.0:
        raise ValueError(
            f"[{group_tag}] residual_sparse mode requires outlier_residual_min_abs >= 0, "
            f"got {resolved_residual_min_abs}."
        )
    resolved_residual_codec = str(outlier_residual_codec).strip().lower()
    use_wa_mse_loss = stage_recon_loss == "wa_mse"
    row_parts, col_parts = 1, 1
    parts_per_linear = 1
    sort_mode = str(runtime_cfg.intra_part_sort_mode).strip().lower()
    # 排序代码，已关闭。原 act_spectral_cosine 动态 activation 触发条件保留如下：
    # needs_dynamic_activation = (
    #     use_wa_mse_loss
    #     or sort_mode == "act_spectral_cosine"
    #     or (resolved_outlier_mode == "channel" and int(outlier_protect_count) > 0)
    #     or residual_sparse_needs_activation
    # )
    needs_dynamic_activation = (
        use_wa_mse_loss
        or residual_sparse_needs_activation
        or channel_rank_needs_actmax
        or channel_rank_needs_actmean
        or channel_rank_needs_sq_mean
    )
    effective_activation_weight: Optional[Dict[str, torch.Tensor]] = None
    effective_activation_abs_mean: Optional[Dict[str, torch.Tensor]] = None
    effective_activation_sq_mean: Optional[Dict[str, torch.Tensor]] = None
    if needs_dynamic_activation:
        if activation_runtime is None:
            raise ValueError(
                f"[{group_tag}] dynamic activation runtime is required for wa_mse or outlier protection."
            )
        calib_device = str(activation_runtime.get("device") or train_device)
        linear_items = [(r.name, r.module) for r in group_refs]
        dynamic_act_stats, new_cache = collect_activation_stats_for_linears(
            model=model,
            linear_items=linear_items,
            model_path=str(activation_runtime["model_path"]),
            access_token=activation_runtime.get("access_token"),
            dataset=str(activation_runtime.get("dataset", "")),
            nsamples=int(activation_runtime.get("nsamples", 512)),
            seqlen=int(activation_runtime.get("seqlen", 512)),
            seed=int(activation_runtime.get("seed", 0)),
            device=calib_device,
            cache=activation_runtime.get("cache"),  # type: ignore[arg-type]
            log_every=int(activation_runtime.get("log_every", 0)),
            logger=log,
        )
        activation_runtime["cache"] = new_cache
        effective_activation_weight = {
            name: stats["max"]
            for name, stats in dynamic_act_stats.items()
            if isinstance(stats.get("max"), torch.Tensor)
        }
        effective_activation_abs_mean = {
            name: stats["abs_mean"]
            for name, stats in dynamic_act_stats.items()
            if isinstance(stats.get("abs_mean"), torch.Tensor)
        }
        effective_activation_sq_mean = {
            name: stats["sq_mean"]
            for name, stats in dynamic_act_stats.items()
            if isinstance(stats.get("sq_mean"), torch.Tensor)
        }
        log.info(
            "[%s] refreshed dynamic activation stats (linears=%d, dataset=%s, nsamples=%d, seqlen=%d).",
            group_tag,
            len(dynamic_act_stats),
            str(activation_runtime.get("dataset", "")),
            int(activation_runtime.get("nsamples", 512)),
            int(activation_runtime.get("seqlen", 512)),
        )

    prep_refs = [
        LinearPrepRef(
            name=r.name,
            weight=r.module.weight,
            in_features=int(r.module.in_features),
            out_features=int(r.module.out_features),
            transpose=bool(r.transpose),
        )
        for r in group_refs
    ]
    prepared_entries = prepare_group_linear_entries(
        group_refs=prep_refs,
        activation_weight_by_linear=effective_activation_weight,
        activation_abs_mean_by_linear=effective_activation_abs_mean,
        outlier_protect_count=(
            int(outlier_protect_count)
            if resolved_outlier_mode == "channel" and channel_protection_enabled
            else 0
        ),
        outlier_protect_axis=str(outlier_protect_axis),
        recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
        intra_part_sort_mode=stage_sort_mode,
        outlier_channel_plan=outlier_channel_plan if resolved_outlier_mode == "channel" and channel_protection_enabled else None,
        apply_outlier_channel_removal=resolved_outlier_mode == "channel",
    )
    initial_split_weights_by_linear: Optional[List[torch.Tensor]] = None
    target_common_result = materialize_prepared_group_data(
        prepared_entries=prepared_entries,
        intra_parallel=(row_parts, col_parts),
        codebook_dim=int(stage_codebook_dim),
        batch_size=int(materialize_batch_size),
        normalize_weight=False,
        recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
        train_device=train_device,
        intra_part_sort_mode="none",
        split_weights_by_linear=initial_split_weights_by_linear,
        shuffle_seed=int(shuffle_seed) if bool(deterministic) else None,
    )
    num_models = int(target_common_result.num_models)
    target_common_split_metas = target_common_result.split_metas
    if initial_split_weights_by_linear is None:
        current_residual_weights = [
            entry.prepared_weight.split_weight.detach().to(device="cpu").contiguous()
            for entry in prepared_entries
        ]
    else:
        current_residual_weights = [
            weight.detach().to(device="cpu").contiguous()
            for weight in initial_split_weights_by_linear
        ]
    use_wa_mse = bool(target_common_result.use_wa_mse)
    if len(target_common_split_metas) != len(group_refs):
        raise RuntimeError(
            f"[{group_tag}] split metadata mismatch: len(split_metas)={len(target_common_split_metas)} "
            f"vs len(group_refs)={len(group_refs)}"
        )
    if resolved_outlier_mode == "channel" and channel_protection_enabled:
        per_linear_protected = []
        for ref, meta in zip(group_refs, target_common_split_metas):
            if str(outlier_protect_axis) == "output":
                protected_idx = meta.protected_output_indices
                total_channels = int(ref.module.out_features)
            else:
                protected_idx = meta.protected_input_indices
                total_channels = int(ref.module.in_features)
            protected_count = int(protected_idx.numel()) if isinstance(protected_idx, torch.Tensor) else 0
            per_linear_protected.append(
                f"{ref.name}:{protected_count}/{total_channels}"
            )
        log.info(
            "[%s] outlier protection mode=%s axis=%s count=%d protected_channels=%s",
            group_tag,
            resolved_outlier_mode,
            str(outlier_protect_axis),
            int(outlier_protect_count),
            ",".join(per_linear_protected),
        )
    if residual_sparse_enabled:
        log.info(
            "[%s] residual sparse protection enabled: top_p=%.6f rank_metric=%s min_abs=%.6e codec=%s index_bits=%d value_bits=%d block=%dx%d",
            group_tag,
            outlier_residual_top_p,
            resolved_outlier_rank_metric,
            resolved_residual_min_abs,
            resolved_residual_codec,
            int(outlier_residual_index_bits),
            int(outlier_residual_value_bits),
            int(outlier_residual_block_shape[0]),
            int(outlier_residual_block_shape[1]),
        )
    if use_wa_mse:
        log.info("[%s] wa_mse enabled with online act_max gather.", group_tag)
    need_stage_payload = bool(do_convert or residual_stages > 1)
    all_stage_bits: List[torch.Tensor] = []
    all_stage_decoders: List[List[nn.Module]] = []
    all_stage_codebook_dims: List[int] = []
    all_stage_split_metas: List[List[object]] = []

    shared_stage_args = _clone_namespace(
        vae_args,
        parallel_layers=num_models,
        residual_stages=int(runtime_cfg.residual_stages),
        codebook_bits=int(stage_codebook_bits),
        codebook_dim=int(stage_codebook_dim),
        base_ch=int(stage_base_ch),
        num_res_blocks=int(stage_num_res_blocks),
        norm_type=str(stage_norm_type),
        decoder_type=str(stage_decoder_type),
        decoder_base_ch=(
            None if runtime_cfg.decoder_base_ch is None else int(runtime_cfg.decoder_base_ch)
        ),
        decoder_num_res_blocks=(
            None if runtime_cfg.decoder_num_res_blocks is None else int(runtime_cfg.decoder_num_res_blocks)
        ),
        recon_loss_type=str(stage_recon_loss),
    )
    apply_autoencoder_arch_defaults(shared_stage_args)
    use_stage_norm = bool(getattr(shared_stage_args, "normalize_weight", False))

    final_base_residual_stage_idx: Optional[int] = None
    for stage_idx in range(residual_stages):
        stage_tag = f"{group_tag}/stage{stage_idx + 1}"
        stage_vae_args = _clone_namespace(shared_stage_args)
        common_stage_result = materialize_prepared_group_data(
            prepared_entries=prepared_entries,
            intra_parallel=(row_parts, col_parts),
            codebook_dim=int(stage_codebook_dim),
            batch_size=int(materialize_batch_size),
            normalize_weight=False,
            recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
            train_device=train_device,
            intra_part_sort_mode="none",
            split_weights_by_linear=current_residual_weights,
            shuffle_seed=int(shuffle_seed) + int(stage_idx) if bool(deterministic) else None,
        )
        stage_prep_result = common_stage_result
        # 排序代码，已关闭。原 stage 排序 materialize 分支保留如下：
        # if sort_mode != "none":
        #     prep_start_time = time.time()
        #     stage_prep_result = materialize_prepared_group_data(
        #         prepared_entries=prepared_entries,
        #         codebook_dim=int(stage_codebook_dim),
        #         batch_size=int(materialize_batch_size),
        #         normalize_weight=False,
        #         recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
        #         train_device=train_device,
        #         intra_part_sort_mode=stage_sort_mode,
        #         sort_executor=sort_executor,
        #         split_weights_by_linear=current_residual_weights,
        #         shuffle_seed=int(shuffle_seed) + int(stage_idx) if bool(deterministic) else None,
        #     )
        #     prep_duration_sec = float(time.time() - prep_start_time)
        #     sort_task_count = int(len(group_refs))
        #     effective_sort_workers = 1
        #     sort_backend = "cpu_serial"
        #     if sort_executor is not None and sort_task_count > 1 and int(sort_prep_workers_resolved) > 1:
        #         effective_sort_workers = min(int(sort_prep_workers_resolved), sort_task_count)
        #         sort_backend = "cpu_process"
        #     log.info(
        #         "[%s] 排序预处理完成: sort_backend=%s sort_prep_workers_resolved=%d sort_task_count=%d duration_sec=%.2f",
        #         stage_tag,
        #         sort_backend,
        #         effective_sort_workers,
        #         sort_task_count,
        #         prep_duration_sec,
        #     )
        current_common_stacked = common_stage_result.stacked_data.detach().clone().contiguous()
        stage_result = stage_prep_result
        residual_data = _reshape_blocks_for_codebook_dim(
            stage_result.stacked_data.detach().clone().contiguous(),
            codebook_dim=int(stage_codebook_dim),
        )
        part_metas = stage_result.part_metas
        stage_split_metas = list(stage_result.split_metas)
        residual_rms_before = float(current_common_stacked.float().pow(2).mean().sqrt().item())

        if use_stage_norm:
            stage_norm_mean, stage_norm_scale = _compute_stage_norm_stats(residual_data)
            stage_train_data = _apply_stage_norm(
                residual_data,
                mean=stage_norm_mean,
                scale=stage_norm_scale,
            )
        else:
            stage_norm_mean = None
            stage_norm_scale = None
            stage_train_data = residual_data

        gpu_resident_enabled = bool(gpu_resident_data)
        gpu_stage_train_data: Optional[torch.Tensor] = None
        if gpu_resident_enabled:
            gpu_stage_train_data = stage_train_data.to(
                device=train_device, dtype=train_dtype, non_blocking=True).contiguous()
            log.info(
                "[%s] VAE gpu_resident_data enabled: blocks=%d batch_size=%s",
                stage_tag,
                int(stage_train_data.shape[0]),
                str(batch_size),
            )

        effective_batch_size = int(stage_train_data.shape[0]) if batch_size_is_all else int(materialize_batch_size)
        # eval 固定覆盖完整 residual stage，但始终按 batch 重构，避免 all-batch 一次前向 OOM。
        eval_batch_size = int(materialize_batch_size)
        all_batch_gpu_cache = bool(batch_size_is_all and stage_recon_loss != "wa_mse" and not gpu_resident_enabled)
        if int(effective_batch_size) < 1:
            raise RuntimeError(f"[{stage_tag}] effective VAE batch size must be >= 1, got {effective_batch_size}.")
        if batch_size_is_all:
            log.info("[%s] VAE batch_size=all(effective=%d)", stage_tag, int(effective_batch_size))
        if gpu_resident_enabled:
            train_loader = None
        elif all_batch_gpu_cache:
            log.info("[%s] VAE all-batch GPU cache enabled.", stage_tag)
            train_loader = None
        else:
            train_loader, _unused_eval_loader = _build_block_data_loaders(
                stage_train_data,
                batch_size=int(effective_batch_size),
                shuffle_seed=int(shuffle_seed) + int(stage_idx) if bool(deterministic) else None,
            )
            del _unused_eval_loader
        if gpu_resident_enabled:
            eval_loader = None
        else:
            _unused_train_loader, eval_loader = _build_block_data_loaders(
                stage_train_data,
                batch_size=int(eval_batch_size),
                shuffle_seed=int(shuffle_seed) + int(stage_idx) if bool(deterministic) else None,
            )
            del _unused_train_loader
        vae = MultiLayerVAE(stage_vae_args).to(train_device)

        # 2) 训练当前 residual stage 对应的 VAE。
        optimizer = create_optimizer(vae.parameters(), stage_vae_args, stage_vae_args.lr)
        lr_scheduler = None
        lr_scheduler_name = str(getattr(stage_vae_args, "lr_scheduler", "constant")).strip().lower()
        if lr_scheduler_name != "constant":
            import transformers

            lr_scheduler = transformers.get_scheduler(
                lr_scheduler_name,
                optimizer,
                num_warmup_steps=int(getattr(stage_vae_args, "lr_warmup_steps", 0)),
                num_training_steps=int(stage_steps),
            )

        log.info(
            "[%s] start (residual_rms=%.6e, steps=%d, blocks=%d, bits=%d, dim=%d, recon_loss=%s, base_ch=%d, num_res_blocks=%d, norm_type=%s, decoder_type=%s, stage_norm=%s)",
            stage_tag,
            residual_rms_before,
            int(stage_steps),
            int(residual_data.shape[0]),
            int(stage_codebook_bits),
            int(stage_codebook_dim),
            stage_recon_loss,
            int(stage_base_ch),
            int(stage_num_res_blocks),
            stage_norm_type,
            stage_decoder_type,
            "on" if use_stage_norm else "off",
        )
        start = time.time()
        num_stage_blocks = int(stage_train_data.shape[0])
        gpu_train_order: Optional[torch.Tensor] = None
        gpu_train_pos = 0
        gpu_train_generator = None
        if gpu_resident_enabled and bool(deterministic):
            gpu_train_generator = torch.Generator()
            gpu_train_generator.manual_seed(int(shuffle_seed) + int(stage_idx))

        def _next_gpu_train_indices() -> torch.Tensor:
            nonlocal gpu_train_order, gpu_train_pos
            if gpu_train_order is None or int(gpu_train_pos) >= num_stage_blocks:
                gpu_train_order = torch.randperm(
                    num_stage_blocks,
                    generator=gpu_train_generator,
                    dtype=torch.long,
                )
                gpu_train_pos = 0
            end = min(int(gpu_train_pos) + int(effective_batch_size), num_stage_blocks)
            batch_idx = gpu_train_order[int(gpu_train_pos):end]
            gpu_train_pos = int(end)
            return batch_idx

        def _iter_eval_tensors():
            if gpu_stage_train_data is not None:
                for start_idx in range(0, num_stage_blocks, int(eval_batch_size)):
                    yield gpu_stage_train_data[start_idx:start_idx + int(eval_batch_size)]
            else:
                for x_cpu_batch, _eval_idx_batch in eval_loader:
                    yield x_cpu_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)

        if gpu_resident_enabled:
            x_all = None
            train_iter = None
            if batch_size_is_all:
                x_all = gpu_stage_train_data
        elif all_batch_gpu_cache:
            x_all = stage_train_data.to(device=train_device, dtype=train_dtype, non_blocking=True)
            train_iter = None
        else:
            x_all = None
            train_iter = iter(train_loader)
        for step in range(int(stage_steps)):
            act_max_batch = None
            if gpu_resident_enabled:
                if batch_size_is_all:
                    x = x_all
                    if stage_recon_loss == "wa_mse":
                        block_idx_batch = torch.arange(num_stage_blocks, device=train_device, dtype=torch.long)
                        act_max_batch = gather_wa_mse_act_max_batch(
                            block_idx_batch=block_idx_batch,
                            part_metas=part_metas,
                            codebook_dim=int(stage_codebook_dim),
                            train_device=train_device,
                            target_dtype=train_dtype,
                        )
                else:
                    block_idx_batch = _next_gpu_train_indices()
                    block_idx_gpu = block_idx_batch.to(device=train_device, dtype=torch.long, non_blocking=True)
                    x = gpu_stage_train_data.index_select(0, block_idx_gpu)
                    if stage_recon_loss == "wa_mse":
                        act_max_batch = gather_wa_mse_act_max_batch(
                            block_idx_batch=block_idx_gpu,
                            part_metas=part_metas,
                            codebook_dim=int(stage_codebook_dim),
                            train_device=train_device,
                            target_dtype=train_dtype,
                        )
            elif all_batch_gpu_cache:
                x = x_all
            else:
                try:
                    x_batch, block_idx_batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x_batch, block_idx_batch = next(train_iter)

                x = x_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
                if stage_recon_loss == "wa_mse":
                    act_max_batch = gather_wa_mse_act_max_batch(
                        block_idx_batch=block_idx_batch,
                        part_metas=part_metas,
                        codebook_dim=int(stage_codebook_dim),
                        train_device=train_device,
                        target_dtype=train_dtype,
                    )
            optimizer.zero_grad(set_to_none=True)
            _, loss_dict = vae(x, is_train=True, act_max=act_max_batch)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if log_every > 0 and (step + 1) % int(log_every) == 0:
                speed = (time.time() - start) / int(log_every)
                recon = loss_dict.get("train/recon_loss")
                commit = loss_dict.get("train/commitment_loss")
                log.info(
                    "[%s] step=%d/%d loss=%.4e recon=%.4e commit=%.4e speed=%.4fs/it",
                    stage_tag,
                    step + 1,
                    stage_steps,
                    float(loss.detach().float().item()),
                    float(recon.detach().float().item()) if isinstance(recon, torch.Tensor) else float("nan"),
                    float(commit.detach().float().item()) if isinstance(commit, torch.Tensor) else float("nan"),
                    speed,
                )
                start = time.time()

            if eval_every > 0 and (step + 1) % int(eval_every) == 0:
                vae.eval()
                with torch.no_grad():
                    mse_sum = 0.0
                    mse_numel = 0
                    top_k_mse_sum = 0.0
                    top_k_mse_numel = 0
                    eval_blocks_seen = 0
                    for x_eval in _iter_eval_tensors():
                        x_recon, _ = vae(x_eval, is_train=False)
                        x_eval_f = x_eval.float()
                        x_recon_f = x_recon.float()
                        batch_numel = int(x_eval_f.numel())
                        if batch_numel > 0:
                            batch_mse = torch.nn.functional.mse_loss(x_recon_f, x_eval_f, reduction="mean")
                            mse_sum += float(batch_mse.detach().cpu().item()) * batch_numel
                            mse_numel += batch_numel
                            eval_blocks_seen += int(x_eval_f.shape[0])

                        # 对每个并行模型（P 维）独立选 top-k：
                        # x_eval/x_recon: [B, P, C] -> [P, B*C]
                        flat_eval = x_eval_f.permute(1, 0, 2).reshape(x_eval_f.shape[1], -1)
                        flat_recon = x_recon_f.permute(1, 0, 2).reshape(x_recon_f.shape[1], -1)
                        k = min(100, flat_eval.shape[1])
                        _, topk_idx = torch.topk(flat_eval.abs(), k=k, dim=1)
                        top_eval = torch.gather(flat_eval, dim=1, index=topk_idx)
                        top_recon = torch.gather(flat_recon, dim=1, index=topk_idx)
                        top_k_numel = int(top_eval.numel())
                        if top_k_numel > 0:
                            batch_top_k_mse = torch.nn.functional.mse_loss(
                                top_recon,
                                top_eval,
                                reduction="mean",
                            )
                            top_k_mse_sum += float(batch_top_k_mse.detach().cpu().item()) * top_k_numel
                            top_k_mse_numel += top_k_numel
                    mse = mse_sum / float(mse_numel) if mse_numel > 0 else 0.0
                    top_k_mse = top_k_mse_sum / float(top_k_mse_numel) if top_k_mse_numel > 0 else 0.0
                log.info(
                    "[%s] eval@step=%d full_residual_blocks=%d mse=%.6e top_k_mse(k=100)=%.6e",
                    stage_tag,
                    step + 1,
                    int(eval_blocks_seen),
                    float(mse),
                    float(top_k_mse),
                )
                vae.train()

        # 3) 对当前 stage 的 residual 生成重构，更新下一阶段 residual。
        vae.eval()
        stage_recon_chunks: List[torch.Tensor] = []
        stage_bit_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for x_in in _iter_eval_tensors():
                x_recon, bit_idx = vae(x_in, is_train=False)
                stage_recon_chunks.append(x_recon.detach().to(device="cpu", dtype=residual_data.dtype))
                if need_stage_payload:
                    stage_bit_chunks.append(bit_idx.detach().to("cpu"))

        stage_recon_full_norm = torch.cat(stage_recon_chunks, dim=0)
        if tuple(stage_recon_full_norm.shape) != tuple(stage_train_data.shape):
            raise RuntimeError(
                f"[{stage_tag}] recon shape mismatch: recon={tuple(stage_recon_full_norm.shape)} "
                f"vs stage_train_data={tuple(stage_train_data.shape)}"
            )
        if stage_norm_mean is not None and stage_norm_scale is not None:
            stage_recon_full = _restore_stage_norm(
                stage_recon_full_norm,
                mean=stage_norm_mean,
                scale=stage_norm_scale,
            )
        else:
            stage_recon_full = stage_recon_full_norm
        if tuple(stage_recon_full.shape) != tuple(residual_data.shape):
            raise RuntimeError(
                f"[{stage_tag}] denorm recon shape mismatch: recon={tuple(stage_recon_full.shape)} "
                f"vs residual={tuple(residual_data.shape)}"
            )
        stage_common_recon = _convert_stage_stacked_to_common_stacked(
            stage_stacked_data=stage_recon_full,
            stage_split_metas=stage_split_metas,
            common_split_metas=target_common_split_metas,
            codebook_dim=int(stage_codebook_dim),
        ).to(device="cpu", dtype=current_common_stacked.dtype)
        current_common_stacked = (current_common_stacked - stage_common_recon).contiguous()
        current_residual_weights = _group_data_to_compressed_weights(
            stacked_data=current_common_stacked,
            split_metas=target_common_split_metas,
        )
        final_base_residual_stage_idx = int(stage_idx)
        residual_rms_after = float(current_common_stacked.float().pow(2).mean().sqrt().item())
        log.info(
            "[%s] residual rms: before=%.6e after=%.6e",
            stage_tag,
            residual_rms_before,
            residual_rms_after,
        )
        if gpu_stage_train_data is not None:
            del gpu_stage_train_data
            gpu_stage_train_data = None

        if need_stage_payload:
            if not stage_bit_chunks:
                raise RuntimeError(f"[{stage_tag}] no bit indices collected during conversion.")
            stage_full_bits = torch.cat(stage_bit_chunks, dim=0)  # [N_blocks, P, latent_dim]
            all_stage_bits.append(stage_full_bits)
            all_stage_codebook_dims.append(int(stage_codebook_dim))
            all_stage_split_metas.append(stage_split_metas)

            decoder_in_dim = int(getattr(vae.model.decoder, "in_dim"))
            use_new_quant = bool(getattr(stage_vae_args, "new_quant", False))
            quant_q_scale = (1.0 / math.sqrt(decoder_in_dim)) if use_new_quant else 1.0

            decoders: List[nn.Module] = []
            for i in range(num_models):
                dec = vae.model.decoder.get_sub_decoder(i)
                _fuse_q_scale_into_decoder(dec, q_scale=float(quant_q_scale))
                if use_stage_norm:
                    if stage_norm_mean is None or stage_norm_scale is None:
                        raise RuntimeError(f"[{stage_tag}] stage norm stats missing while normalize_weight=True")
                    _fuse_norm_into_decoder(
                        dec,
                        mean=float(stage_norm_mean[i].item()),
                        std=float(stage_norm_scale[i].item()),
                    )
                decoders.append(dec)
            all_stage_decoders.append(decoders)

        if x_all is not None:
            del x_all
        del vae, train_loader, eval_loader, optimizer, common_stage_result, stage_result
        if lr_scheduler is not None:
            del lr_scheduler
        torch.cuda.empty_cache()

    if residual_stages > 1:
        # 联合优化代码，已关闭。原 joint decoder 调用保留如下：
        # if (
        #     len(all_stage_bits) != residual_stages
        #     or len(all_stage_decoders) != residual_stages
        #     or len(all_stage_split_metas) != residual_stages
        #     or len(all_stage_codebook_dims) != residual_stages
        # ):
        #     raise RuntimeError(
        #         f"[{group_tag}] joint fine-tune payload mismatch: bits={len(all_stage_bits)} "
        #         f"decoders={len(all_stage_decoders)} split_metas={len(all_stage_split_metas)} "
        #         f"codebook_dims={len(all_stage_codebook_dims)} residual_stages={residual_stages}"
        #     )
        # joint_steps = int(runtime_cfg.joint_decoder_steps)
        # joint_lr = float(runtime_cfg.joint_decoder_lr)
        # joint_group_size = max(1, min(int(runtime_cfg.joint_decoder_group_size), len(group_refs)))
        # joint_batch_size = runtime_cfg.joint_decoder_batch_size
        # if joint_steps > 0:
        #     log.info(
        #         "[%s/joint] start (mode=%s, steps=%d, lr=%.3e, recon_loss=%s, stages=%d, joint_group_size=%d, joint_decoder_batch_size=%s)",
        #         group_tag,
        #         "patch" if joint_batch_size is not None else "full",
        #         joint_steps,
        #         joint_lr,
        #         stage_recon_loss,
        #         residual_stages,
        #         joint_group_size,
        #         "none" if joint_batch_size is None else str(int(joint_batch_size)),
        #     )
        #     all_stage_decoders = _finetune_stage_decoders_in_subgroups(
        #         group_tag=group_tag,
        #         group_refs=group_refs,
        #         shared_stage_args=shared_stage_args,
        #         joint_steps=joint_steps,
        #         joint_lr=joint_lr,
        #         joint_group_size=joint_group_size,
        #         joint_decoder_batch_size=joint_batch_size,
        #         train_device=train_device,
        #         train_dtype=train_dtype,
        #         log_every=log_every,
        #         eval_every=eval_every,
        #         eval_blocks=eval_blocks,
        #         codebook_dim=int(stage_codebook_dim),
        #         recon_loss_type=stage_recon_loss,
        #         intra_part_sort_mode=stage_sort_mode,
        #         target_common_result=target_common_result,
        #         all_stage_bits=all_stage_bits,
        #         all_stage_decoders=all_stage_decoders,
        #         all_stage_split_metas=all_stage_split_metas,
        #         parts_per_linear=parts_per_linear,
        #         convert_stage_to_common_fn=_convert_stage_stacked_to_common_stacked,
        #         recon_loss_fn=_compute_recon_loss,
        #         logger=log,
        #         shuffle_seed=int(shuffle_seed) if bool(deterministic) else None,
        #     )
        pass

    if not do_convert:
        del current_residual_weights, target_common_result, all_stage_bits, all_stage_decoders, all_stage_codebook_dims, all_stage_split_metas
        torch.cuda.empty_cache()
        return None

    if (
        len(all_stage_bits) != residual_stages
        or len(all_stage_decoders) != residual_stages
        or len(all_stage_codebook_dims) != residual_stages
        or len(all_stage_split_metas) != residual_stages
    ):
        raise RuntimeError(
            f"[{group_tag}] stage payload mismatch: bits={len(all_stage_bits)} "
            f"decoders={len(all_stage_decoders)} codebook_dims={len(all_stage_codebook_dims)} "
            f"split_metas={len(all_stage_split_metas)} "
            f"residual_stages={residual_stages}"
        )

    protected_residual_payload_by_name: Dict[str, Dict[str, object]] = {}
    if resolved_outlier_mode == "channel_residual_vae" and int(outlier_protect_count) > 0:
        if final_base_residual_stage_idx != int(residual_stages) - 1:
            raise RuntimeError(
                f"[{group_tag}/channel_residual_vae] final residual is not from the last base VAE stage: "
                f"final_stage_idx={final_base_residual_stage_idx} expected={int(residual_stages) - 1}."
            )
        log.info(
            "[%s/channel_residual_vae] using final residual from base VAE stage output; skip base re-decode",
            group_tag,
        )
        log.info(
            "[%s/channel_residual_vae] selecting protected channels after base VAE stages using final residual",
            group_tag,
        )
        target_common_split_metas, channel_rank_summary = _select_channel_residual_vae_plan_from_final_residual(
            group_tag=group_tag,
            group_refs=group_refs,
            target_common_split_metas=target_common_split_metas,
            final_common_stacked=current_common_stacked,
            parts_per_linear=int(parts_per_linear),
            outlier_protect_count=int(outlier_protect_count),
            outlier_protect_axis=str(outlier_protect_axis),
            outlier_channel_scope=str(outlier_channel_scope),
            outlier_rank_metric=resolved_outlier_rank_metric,
            activation_weight_by_linear=effective_activation_weight,
            activation_abs_mean_by_linear=effective_activation_abs_mean,
            activation_sq_mean_by_linear=effective_activation_sq_mean,
            final_stage_idx=int(final_base_residual_stage_idx),
            outlier_protect_min_per_layer=int(outlier_protect_min_per_layer),
        )
        protected_residual_entries = _build_protected_residual_entries_from_final_residual(
            group_tag=group_tag,
            group_refs=group_refs,
            target_common_split_metas=target_common_split_metas,
            final_common_stacked=current_common_stacked,
            parts_per_linear=int(parts_per_linear),
            outlier_protect_axis=str(outlier_protect_axis),
            final_stage_idx=int(final_base_residual_stage_idx),
            codebook_dim=int(protected_residual_vae_codebook_dim),
        )
        protected_numel = sum(int(residual_slice.numel()) for _r, _axis, _idx_cpu, residual_slice in protected_residual_entries)
        if protected_numel > 0:
            protected_sq_sum = sum(
                float(residual_slice.float().pow(2).sum().item())
                for _r, _axis, _idx_cpu, residual_slice in protected_residual_entries
            )
            protected_rms = math.sqrt(protected_sq_sum / float(protected_numel))
        else:
            protected_rms = 0.0
        log.info(
            "[%s/channel_residual_vae] metric=%s axis=%s num_channels=%d topk=%d score_max=%.6e score_mean=%.6e protected_residual_rms=%.6e",
            group_tag,
            resolved_outlier_rank_metric,
            str(outlier_protect_axis),
            int(channel_rank_summary["num_channels"]),
            int(channel_rank_summary["topk"]),
            float(channel_rank_summary["score_max"]),
            float(channel_rank_summary["score_mean"]),
            float(protected_rms),
        )
        for r, axis, idx_cpu, residual_slice in protected_residual_entries:
            residual_rms = float(residual_slice.float().pow(2).mean().sqrt().item())
            log.info(
                "[%s/channel_residual_vae] axis=%s channels=%d residual_rms=%.6e",
                r.name,
                axis,
                int(idx_cpu.numel()),
                residual_rms,
            )

        shared_payload_by_name: Dict[str, Dict[str, object]] = {}
        if resolved_protected_residual_decoder_share_scope == "category":
            shared_payload_by_name = _train_shared_protected_residual_vae_payloads(
                group_tag=group_tag,
                residual_slices_by_name={
                    r.name: residual_slice
                    for r, _axis, _idx_cpu, residual_slice in protected_residual_entries
                },
                runtime_cfg=runtime_cfg,
                vae_args=vae_args,
                training_args=training_args,
                train_device=train_device,
                train_dtype=train_dtype,
                batch_size=int(protected_residual_vae_batch_size),
                steps=int(protected_residual_vae_steps),
                lr=float(protected_residual_vae_lr),
                log_every=log_every,
                deterministic=deterministic,
                shuffle_seed=int(shuffle_seed),
            )

        for r, axis, idx_cpu, residual_slice in protected_residual_entries:
            if resolved_protected_residual_decoder_share_scope == "category":
                residual_payload = shared_payload_by_name.get(r.name)
            else:
                residual_payload = _train_protected_residual_vae_payload(
                    linear_name=r.name,
                    residual_slice=residual_slice,
                    runtime_cfg=runtime_cfg,
                    vae_args=vae_args,
                    training_args=training_args,
                    train_device=train_device,
                    train_dtype=train_dtype,
                    batch_size=int(protected_residual_vae_batch_size),
                    steps=int(protected_residual_vae_steps),
                    lr=float(protected_residual_vae_lr),
                    log_every=log_every,
                    deterministic=deterministic,
                    shuffle_seed=int(shuffle_seed),
                )
            if residual_payload is None:
                continue
            protected_residual_payload_by_name[r.name] = {
                "protected_residual_axis": axis,
                "protected_residual_indices": idx_cpu,
                "protected_residual_stage_vq_weights": residual_payload["stage_vq_weights"],
                "protected_residual_stage_codebook_dims": residual_payload["stage_codebook_dims"],
            }
            if resolved_protected_residual_decoder_share_scope == "category":
                protected_residual_payload_by_name[r.name].update(
                    {
                        "protected_residual_shared_decoder_refs": residual_payload["shared_decoder_refs"],
                        "protected_residual_shared_stage_decoders": residual_payload["shared_stage_decoders"],
                    }
                )
            else:
                protected_residual_payload_by_name[r.name]["protected_residual_stage_decoders"] = residual_payload["stage_decoders"]
            log.info(
                "[%s] protected residual VAE patch for %s: axis=%s channels=%d stages=%d steps=%d decoder_share_scope=%s codebook_bits=%d codebook_dim=%d",
                group_tag,
                r.name,
                axis,
                int(idx_cpu.numel()),
                int(runtime_cfg.outlier_residual_vae_stages),
                int(protected_residual_vae_steps),
                resolved_protected_residual_decoder_share_scope,
                int(protected_residual_vae_codebook_bits),
                int(protected_residual_vae_codebook_dim),
            )

    for stage_decoders in all_stage_decoders:
        for decoder in stage_decoders:
            decoder.to("cpu")
    payload = {
        "format": "vaellm_group_vae_payload",
        "version": 1,
        "target_common_split_metas": target_common_split_metas,
        "parts_per_linear": int(parts_per_linear),
        "row_parts": int(row_parts),
        "col_parts": int(col_parts),
        "residual_stages": int(residual_stages),
        "all_stage_bits": all_stage_bits,
        "all_stage_decoders": all_stage_decoders,
        "all_stage_codebook_dims": all_stage_codebook_dims,
        "all_stage_split_metas": all_stage_split_metas,
        "resolved_outlier_mode": resolved_outlier_mode,
        "residual_sparse_enabled": bool(residual_sparse_enabled),
        "effective_activation_weight": effective_activation_weight,
        "effective_activation_abs_mean": effective_activation_abs_mean,
        "outlier_residual_top_p": float(outlier_residual_top_p),
        "resolved_outlier_rank_metric": resolved_outlier_rank_metric,
        "resolved_residual_min_abs": float(resolved_residual_min_abs),
        "resolved_residual_codec": resolved_residual_codec,
        "outlier_residual_index_bits": int(outlier_residual_index_bits),
        "outlier_residual_value_bits": int(outlier_residual_value_bits),
        "outlier_residual_block_shape": tuple(int(v) for v in outlier_residual_block_shape),
        "resolved_protected_residual_vae_codebook_bits": int(protected_residual_vae_codebook_bits),
        "resolved_protected_residual_vae_codebook_dim": int(protected_residual_vae_codebook_dim),
        "protected_residual_payload_by_name": protected_residual_payload_by_name,
    }
    del current_residual_weights, target_common_result
    torch.cuda.empty_cache()
    return payload


def apply_group_vae_payload(
    *,
    model: nn.Module,
    group_refs: Sequence[LinearRef],
    group_tag: str,
    payload: Dict[str, Any],
    convert_device: str,
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
) -> None:
    if str(payload.get("format", "")) != "vaellm_group_vae_payload":
        raise ValueError(f"[{group_tag}] invalid VAE payload format: {payload.get('format')!r}.")
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"[{group_tag}] unsupported VAE payload version: {payload.get('version')!r}.")
    target_common_split_metas = payload["target_common_split_metas"]
    parts_per_linear = int(payload["parts_per_linear"])
    row_parts = int(payload["row_parts"])
    col_parts = int(payload["col_parts"])
    residual_stages = int(payload["residual_stages"])
    all_stage_bits = payload["all_stage_bits"]
    all_stage_decoders = payload["all_stage_decoders"]
    all_stage_codebook_dims = payload["all_stage_codebook_dims"]
    all_stage_split_metas = payload["all_stage_split_metas"]
    resolved_outlier_mode = str(payload["resolved_outlier_mode"])
    residual_sparse_enabled = bool(payload["residual_sparse_enabled"])
    effective_activation_weight = payload.get("effective_activation_weight")
    effective_activation_abs_mean = payload.get("effective_activation_abs_mean")
    outlier_residual_top_p = float(payload["outlier_residual_top_p"])
    resolved_outlier_rank_metric = str(payload["resolved_outlier_rank_metric"])
    resolved_residual_min_abs = float(payload["resolved_residual_min_abs"])
    resolved_residual_codec = str(payload["resolved_residual_codec"])
    outlier_residual_index_bits = int(payload["outlier_residual_index_bits"])
    outlier_residual_value_bits = int(payload["outlier_residual_value_bits"])
    outlier_residual_block_shape = tuple(int(v) for v in payload["outlier_residual_block_shape"])
    protected_residual_payload_by_name = payload.get("protected_residual_payload_by_name") or {}

    if (
        len(all_stage_bits) != residual_stages
        or len(all_stage_decoders) != residual_stages
        or len(all_stage_codebook_dims) != residual_stages
        or len(all_stage_split_metas) != residual_stages
    ):
        raise RuntimeError(
            f"[{group_tag}] stage payload mismatch: bits={len(all_stage_bits)} "
            f"decoders={len(all_stage_decoders)} codebook_dims={len(all_stage_codebook_dims)} "
            f"split_metas={len(all_stage_split_metas)} "
            f"residual_stages={residual_stages}"
        )

    for i, r in enumerate(group_refs):
        old = r.module
        split_meta = target_common_split_metas[i]
        stage_split_metas = [all_stage_split_metas[stage_idx][i] for stage_idx in range(residual_stages)]
        if str(split_meta.linear_name) != str(r.name):
            raise RuntimeError(
                f"[{group_tag}] split metadata order mismatch at idx={i}: "
                f"meta={split_meta.linear_name}, ref={r.name}"
            )
        if int(split_meta.parallel_rows) * int(split_meta.parallel_cols) != int(parts_per_linear):
            raise RuntimeError(
                f"[{group_tag}] split parts mismatch at idx={i}: "
                f"meta={split_meta.parallel_rows}x{split_meta.parallel_cols}, expected={parts_per_linear}"
            )
        layer_idx = _extract_layer_idx(r.name)
        skip_this = bool(
            skip_layer_keys
            and layer_idx is not None
            and (int(layer_idx), str(r.category)) in skip_layer_keys
        )
        start_idx = i * parts_per_linear
        end_idx = start_idx + parts_per_linear
        stage_part_bits_payload: List[object] = []
        stage_part_decoders_payload: List[object] = []
        for stage_idx in range(residual_stages):
            stage_bits = all_stage_bits[stage_idx]
            stage_decoders = all_stage_decoders[stage_idx]
            part_bits = []
            part_decoders = []
            for model_idx in range(start_idx, end_idx):
                part_bits.append(stage_bits[:, model_idx, :].unsqueeze(1))  # [N_blocks, 1, latent_dim]
                part_decoders.append(stage_decoders[model_idx])
            if parts_per_linear > 1:
                stage_part_bits_payload.append(part_bits)
                stage_part_decoders_payload.append(part_decoders)
            else:
                stage_part_bits_payload.append(part_bits[0])
                stage_part_decoders_payload.append(part_decoders[0])

        sparse_residual_kwargs = None
        reconstructed_weight = None
        if residual_sparse_enabled:
            reconstructed_weight = _decode_reconstructed_linear_weight(
                old_module=old,
                transpose=r.transpose,
                split_meta=split_meta,
                stage_split_metas=stage_split_metas,
                stage_part_bits_payload=stage_part_bits_payload,
                stage_part_decoders_payload=stage_part_decoders_payload,
                stage_codebook_dims=all_stage_codebook_dims,
                parallel_rows=row_parts,
                parallel_cols=col_parts,
                parallel_parts=parts_per_linear,
            )
        if residual_sparse_enabled:
            activation_weight = None
            activation_mean = None
            if resolved_outlier_rank_metric in _RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX:
                if effective_activation_weight is None or r.name not in effective_activation_weight:
                    raise ValueError(
                        f"[{group_tag}] missing activation vector for residual_sparse scoring at linear '{r.name}'."
                    )
                activation_weight = effective_activation_weight[r.name]
            if resolved_outlier_rank_metric in _RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMEAN:
                if effective_activation_abs_mean is None or r.name not in effective_activation_abs_mean:
                    raise ValueError(
                        f"[{group_tag}] missing activation mean vector for residual_sparse scoring at linear '{r.name}'."
                    )
                activation_mean = effective_activation_abs_mean[r.name]
            if reconstructed_weight is None:
                raise RuntimeError(f"[{group_tag}] missing reconstructed weight for residual_sparse payload.")
            sparse_residual_kwargs, sparse_nnz, sparse_storage = _build_sparse_residual_payload(
                linear_name=r.name,
                original_weight=old.weight,
                reconstructed_weight=reconstructed_weight,
                activation_weight=activation_weight,
                activation_mean=activation_mean,
                rank_metric=resolved_outlier_rank_metric,
                top_p=outlier_residual_top_p,
                min_abs=resolved_residual_min_abs,
                codec=resolved_residual_codec,
                index_bits=outlier_residual_index_bits,
                value_bits=outlier_residual_value_bits,
                block_shape=outlier_residual_block_shape,
            )
            log.info(
                "[%s] residual sparse patch for %s: nnz=%d top_p=%.6f rank_metric=%s min_abs=%.6e codec=%s bytes(codec=%d coo=%d)",
                group_tag,
                r.name,
                sparse_nnz,
                outlier_residual_top_p,
                resolved_outlier_rank_metric,
                resolved_residual_min_abs,
                resolved_residual_codec,
                int(sparse_storage["codec_bytes"]),
                int(sparse_storage["coo_bytes"]),
            )
        protected_residual_kwargs = protected_residual_payload_by_name.get(r.name)
        if protected_residual_kwargs:
            shared_refs = protected_residual_kwargs.get("protected_residual_shared_decoder_refs")
            shared_decoders = protected_residual_kwargs.get("protected_residual_shared_stage_decoders")
            if shared_refs is not None or shared_decoders is not None:
                if not isinstance(shared_refs, (list, tuple)) or not isinstance(shared_decoders, (list, tuple)):
                    raise TypeError(f"[{group_tag}] invalid shared protected residual decoder payload for {r.name}.")
                if len(shared_refs) != len(shared_decoders):
                    raise ValueError(
                        f"[{group_tag}] shared protected residual decoder ref/object length mismatch for {r.name}: "
                        f"{len(shared_refs)} vs {len(shared_decoders)}"
                    )
                for ref, decoder in zip(shared_refs, shared_decoders):
                    register_shared_protected_residual_decoder(model, str(ref), decoder)
        new_linear = _build_vae_linear_from_stage_payload(
            old_module=old,
            transpose=r.transpose,
            split_meta=split_meta,
            stage_split_metas=stage_split_metas,
            stage_part_bits_payload=stage_part_bits_payload,
            stage_part_decoders_payload=stage_part_decoders_payload,
            stage_codebook_dims=all_stage_codebook_dims,
            parallel_rows=row_parts,
            parallel_cols=col_parts,
            parallel_parts=parts_per_linear,
            bias=old.bias,
            original_weight=old.weight,
            always_use_original=skip_this,
            protect_original_weight=skip_this,
            sparse_residual_kwargs=sparse_residual_kwargs,
            protected_residual_kwargs=protected_residual_kwargs,
        ).to(convert_device)
        new_linear.to("cpu")
        set_module_by_name(model, r.name, new_linear)

    torch.cuda.empty_cache()


def _train_group_vae_and_replace(
    *,
    model: nn.Module,
    group_refs: Sequence[LinearRef],
    group_tag: str,
    runtime_cfg: ResolvedCategoryRuntimeConfig,
    vae_args,
    training_args,
    train_device: str,
    convert_device: str,
    do_convert: bool,
    batch_size: object,
    log_every: int,
    eval_every: int,
    eval_blocks: int,
    gpu_resident_data: bool = False,
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
    activation_runtime: Optional[Dict[str, object]] = None,
    outlier_protect_mode: str = "channel",
    outlier_channel_plan: Optional[Dict[str, torch.Tensor]] = None,
    outlier_channel_scope: str = "layer",
    outlier_rank_metric: str = "sparse_residual_abs",
    outlier_residual_min_abs: float = 1e-6,
    outlier_protect_axis: str = "input",
    outlier_protect_min_per_layer: int = 0,
    outlier_residual_codec: str = SPARSE_RESIDUAL_FORMAT_COO_FP16,
    outlier_residual_index_bits: int = 8,
    outlier_residual_value_bits: int = 8,
    outlier_residual_block_shape: Tuple[int, int] = (256, 256),
    outlier_residual_vae_decoder_share_scope: str = "none",
    outlier_residual_vae_batch_multiplier: int = 1,
    outlier_residual_vae_steps: int = 0,
    outlier_residual_vae_lr: float = 0.0,
    # 排序代码，已关闭。旧参数保留如下：
    # sort_executor=None,
    # sort_prep_workers_resolved: int = 1,
    deterministic: bool = False,
    shuffle_seed: int = 0,
) -> None:
    payload = train_group_vae_payload(
        model=model,
        group_refs=group_refs,
        group_tag=group_tag,
        runtime_cfg=runtime_cfg,
        vae_args=vae_args,
        training_args=training_args,
        train_device=train_device,
        convert_device=convert_device,
        do_convert=do_convert,
        batch_size=batch_size,
        gpu_resident_data=bool(gpu_resident_data),
        log_every=log_every,
        eval_every=eval_every,
        eval_blocks=eval_blocks,
        skip_layer_keys=skip_layer_keys,
        activation_runtime=activation_runtime,
        outlier_protect_mode=outlier_protect_mode,
        outlier_channel_plan=outlier_channel_plan,
        outlier_channel_scope=outlier_channel_scope,
        outlier_rank_metric=outlier_rank_metric,
        outlier_residual_min_abs=outlier_residual_min_abs,
        outlier_protect_axis=outlier_protect_axis,
        outlier_protect_min_per_layer=outlier_protect_min_per_layer,
        outlier_residual_codec=outlier_residual_codec,
        outlier_residual_index_bits=outlier_residual_index_bits,
        outlier_residual_value_bits=outlier_residual_value_bits,
        outlier_residual_block_shape=outlier_residual_block_shape,
        outlier_residual_vae_decoder_share_scope=outlier_residual_vae_decoder_share_scope,
        outlier_residual_vae_batch_multiplier=outlier_residual_vae_batch_multiplier,
        outlier_residual_vae_steps=outlier_residual_vae_steps,
        outlier_residual_vae_lr=outlier_residual_vae_lr,
        deterministic=deterministic,
        shuffle_seed=shuffle_seed,
    )
    if not do_convert:
        return
    if payload is None:
        raise RuntimeError(f"[{group_tag}] VAE payload is missing while do_convert=True.")
    apply_group_vae_payload(
        model=model,
        group_refs=group_refs,
        group_tag=group_tag,
        payload=payload,
        convert_device=convert_device,
        skip_layer_keys=skip_layer_keys,
    )


def run_cat_train(*, cat_args, hf_args, training_args, vae_args) -> None:
    global log
    distill_after_category = str(getattr(cat_args, "distill_after_category", "none")).strip().lower()
    if bool(getattr(training_args, "distill_hif4_act", False)) and distill_after_category == "none":
        raise ValueError("--distill_hif4_act 仅在每类后蒸馏阶段生效，因此必须设置 --distill_after_category。")
    if distill_after_category != "none" and not bool(cat_args.convert):
        raise ValueError("--distill_after_category requires --convert，因为每类后蒸馏必须作用在已替换的压缩模型上。")
    configure_deterministic_mode(bool(getattr(cat_args, "deterministic", False)))
    set_seed(cat_args.seed)

    os.makedirs(cat_args.output_dir, exist_ok=True)
    run_output_dir = _build_run_output_dir(cat_args.output_dir, vae_args.model_path)
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "linear_by_category.log")
    log = get_logger("linear_by_category")
    cat_args.output_dir = run_output_dir

    log.info("Run output directory: %s", run_output_dir)
    if bool(getattr(cat_args, "deterministic", False)):
        log.info("Deterministic mode enabled: torch deterministic algorithms on, TF32 disabled.")
    log.info(
        "Args:\nscript=%s\nvae=%s\ntraining=%s",
        _format_namespace(cat_args),
        _format_namespace(vae_args),
        _format_namespace(training_args),
    )

    model = _load_model_for_cat_train(cat_args=cat_args, hf_args=hf_args, vae_args=vae_args)
    activation_runtime: Optional[Dict[str, object]] = None
    outlier_protect_axis = str(getattr(cat_args, "outlier_protect_axis", "input")).strip().lower()
    transpose_modules = _split_csv(cat_args.transpose_modules)
    target_categories = _split_csv(cat_args.target_categories)
    only_decoder_projections = not bool(cat_args.include_all_linears)
    eval_tasks_text = str(getattr(cat_args, "eval_tasks", "")).strip()
    run_task_eval = bool(eval_tasks_text)
    run_category_eval = bool(cat_args.eval_ppl) or run_task_eval
    if not run_category_eval:
        log.info("跳过类别后评估：--eval_ppl=false 且 --eval_tasks 为空。")
    elif not bool(cat_args.convert):
        log.info("未开启 --convert；类别后评估会直接针对当前模型状态执行。")
    all_linears = _collect_current_trainable_linears(
        model,
        transpose_modules=transpose_modules,
        only_decoder_projections=only_decoder_projections,
        target_categories=target_categories,
    )
    discovered_categories = [r.category for r in all_linears]
    discovered_category_set = set(discovered_categories)
    missing_target_categories = [
        category for category in target_categories if category not in discovered_category_set
    ]
    if missing_target_categories:
        raise ValueError(
            "target_categories contains categories not found in model: "
            + ",".join(missing_target_categories)
        )
    discovered_skip_keys = []
    for r in all_linears:
        li = _extract_layer_idx(r.name)
        if li is not None:
            discovered_skip_keys.append((li, r.category))
    skip_layer_keys, matched, missing = resolve_skip_layer_matches(
        getattr(cat_args, "skip_layers", ""),
        discovered_skip_keys,
    )
    if skip_layer_keys:
        if matched:
            log.info(
                "skip_layers 生效: %s",
                ",".join(f"{li}.{cat}" for li, cat in matched),
            )
        if missing:
            raise ValueError(
                "skip_layers contains unknown layer/category pairs: "
                + ",".join(f"{li}.{cat}" for li, cat in missing)
            )

    linear_group_size = int(cat_args.linear_group_size)
    if linear_group_size < 1:
        raise ValueError(f"linear_group_size must be >= 1, got {linear_group_size}")

    active_categories = list(target_categories)
    if not active_categories:
        raise ValueError("No active categories discovered for training.")

    resolved_category_cfgs = resolve_category_runtime_configs(cat_args, vae_args, active_categories)
    resolved_outlier_mode = str(getattr(cat_args, "outlier_protect_mode", "channel")).strip().lower()
    resolved_outlier_rank_metric = str(getattr(cat_args, "outlier_rank_metric", "sparse_residual_abs")).strip().lower()
    log.info("outlier_mode=%s outlier_rank_metric=%s", resolved_outlier_mode, resolved_outlier_rank_metric)
    if resolved_outlier_mode == "residual_sparse":
        nonzero_counts = {
            cat: int(cfg.outlier_protect_count)
            for cat, cfg in resolved_category_cfgs.items()
            if int(cfg.outlier_protect_count) != 0
        }
        if nonzero_counts:
            raise ValueError(
                "residual_sparse mode requires outlier_protect_count=0 for all active categories, got "
                + ",".join(f"{cat}:{count}" for cat, count in nonzero_counts.items())
            )
        invalid_top_p = {
            cat: float(cfg.outlier_residual_top_p)
            for cat, cfg in resolved_category_cfgs.items()
            if not (0.0 < float(cfg.outlier_residual_top_p) <= 1.0)
        }
        if invalid_top_p:
            raise ValueError(
                "residual_sparse mode requires 0 < outlier_residual_top_p <= 1 for all active categories, got "
                + ",".join(f"{cat}:{top_p}" for cat, top_p in invalid_top_p.items())
            )
    distill_tables = (
        (cat_args.lora_rank, "--lora_rank"),
        (cat_args.lora_alpha, "--lora_alpha"),
        (cat_args.lora_dropout, "--lora_dropout"),
        (cat_args.distill_steps, "--distill_steps"),
        (cat_args.distill_batch_size, "--distill_batch_size"),
        (cat_args.distill_nsamples, "--distill_nsamples"),
        (cat_args.distill_lr, "--distill_lr"),
        (cat_args.distill_weight_decay, "--distill_weight_decay"),
        (cat_args.distill_log_every, "--distill_log_every"),
        (cat_args.distill_temperature, "--distill_temperature"),
        (cat_args.distill_loss_alpha, "--distill_loss_alpha"),
        (cat_args.distill_loss_type, "--distill_loss_type"),
        (cat_args.distill_hidden_loss_weight, "--distill_hidden_loss_weight"),
        (cat_args.distill_pre_mlp_hidden_loss_weight, "--distill_pre_mlp_hidden_loss_weight"),
        (cat_args.lora_use_dora, "--lora_use_dora"),
    )
    for table, arg_name in distill_tables:
        validate_category_keys(table, active_categories, arg_name)

    category_codebook: Dict[str, Tuple[int, int]] = {
        cat: (
            int(resolved_category_cfgs[cat].codebook_bits),
            int(resolved_category_cfgs[cat].codebook_dim),
        )
        for cat in active_categories
    }
    category_residual_vae_codebook: Dict[str, Tuple[int, int]] = {
        cat: (
            int(resolved_category_cfgs[cat].outlier_residual_vae_codebook_bits),
            int(resolved_category_cfgs[cat].outlier_residual_vae_codebook_dim),
        )
        for cat in active_categories
    }
    category_outlier_protect_count: Dict[str, int] = {
        cat: int(resolved_category_cfgs[cat].outlier_protect_count) for cat in active_categories
    }
    category_outlier_residual_top_p: Dict[str, float] = {
        cat: float(resolved_category_cfgs[cat].outlier_residual_top_p) for cat in active_categories
    }
    category_sort_modes: Dict[str, str] = {
        cat: str(resolved_category_cfgs[cat].intra_part_sort_mode)
        for cat in active_categories
    }
    category_sort_mode_desc: Dict[str, str] = {
        cat: format_intra_part_sort_mode(category_sort_modes[cat]) for cat in active_categories
    }
    unique_sort_mode_desc = sorted(set(category_sort_mode_desc.values()))
    any_sort_enabled = any(mode != "none" for mode in category_sort_modes.values())
    if any_sort_enabled:
        raise ValueError("排序代码已关闭；cat_train 只支持 intra_part_sort_mode=none。")

    any_wa_mse = any(str(resolved_category_cfgs[cat].recon_loss_type).strip(
    ).lower() == "wa_mse" for cat in active_categories)
    any_outlier_protect = any(count > 0 for count in category_outlier_protect_count.values())
    channel_protect_needs_activation = (
        resolved_outlier_mode in {"channel", "channel_residual_vae"}
        and resolved_outlier_rank_metric in {
            "channel_weight_actmax_abs",
            "channel_weight_actmean_abs",
            "channel_residual_actmax_abs",
            "channel_residual_actmean_abs",
            "channel_residual_actrms_abs",
        }
        and any_outlier_protect
    )
    residual_sparse_needs_activation = (
        resolved_outlier_mode == "residual_sparse"
        and (
            resolved_outlier_rank_metric in _RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX
            or resolved_outlier_rank_metric in _RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMEAN
        )
    )
    sort_needs_act = False  # 排序代码，已关闭。
    if any_wa_mse or channel_protect_needs_activation or sort_needs_act or residual_sparse_needs_activation:
        activation_dataset = str(getattr(cat_args, "wa_mse_calib_dataset", "")).strip()
        if not activation_dataset:
            raise ValueError(
                "--wa_mse_calib_dataset must be set when dynamic activation calibration is enabled. "
                "Use ratio-style dataset specs such as 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
            )
        activation_cache: Optional[ActivationCalibrationCache] = None
        activation_runtime = {
            "cache": activation_cache,
            "dataset": activation_dataset,
            "nsamples": int(getattr(cat_args, "wa_mse_calib_nsamples", 512)),
            "seqlen": int(getattr(cat_args, "wa_mse_calib_seqlen", 512)),
            "seed": int(getattr(cat_args, "wa_mse_calib_seed", 0)),
            "device": str(getattr(cat_args, "wa_mse_calib_device", "")).strip() or str(cat_args.train_device),
            "log_every": int(getattr(cat_args, "wa_mse_calib_log_every", 0)),
            "model_path": str(vae_args.model_path),
            "access_token": hf_args.access_token,
        }
        enabled_features: List[str] = []
        if any_wa_mse:
            enabled_features.append("wa_mse")
        if channel_protect_needs_activation:
            enabled_features.append("outlier_protect")
        if residual_sparse_needs_activation:
            enabled_features.append("residual_sparse_score")
        log.info(
            "Dynamic activation recalibration enabled for %s: dataset=%s nsamples=%d seqlen=%d seed=%d device=%s",
            ",".join(enabled_features),
            str(activation_runtime["dataset"]),
            int(activation_runtime["nsamples"]),
            int(activation_runtime["seqlen"]),
            int(activation_runtime["seed"]),
            str(activation_runtime["device"]),
        )

    if any_outlier_protect:
        enabled_counts = ",".join(
            f"{cat}:{count}"
            for cat, count in category_outlier_protect_count.items()
            if count > 0
        )
        log.info("Outlier protection enabled: axis=%s count_by_category=%s", outlier_protect_axis, enabled_counts)
    if resolved_outlier_mode == "residual_sparse":
        unique_top_p = sorted(set(category_outlier_residual_top_p.values()))
        if len(unique_top_p) == 1:
            log.info(
                "Residual sparse protection enabled: top_p=%.6f rank_metric=%s min_abs=%.6e codec=%s index_bits=%d value_bits=%d block=%dx%d",
                unique_top_p[0],
                resolved_outlier_rank_metric,
                float(cat_args.outlier_residual_min_abs),
                cat_args.outlier_residual_codec,
                int(cat_args.outlier_residual_index_bits),
                int(cat_args.outlier_residual_value_bits),
                int(cat_args.outlier_residual_block_shape[0]),
                int(cat_args.outlier_residual_block_shape[1]),
            )
        else:
            log.info(
                "Residual sparse protection enabled: top_p_by_category={%s} rank_metric=%s min_abs=%.6e codec=%s index_bits=%d value_bits=%d block=%dx%d",
                ",".join(f"{cat}:{category_outlier_residual_top_p[cat]:.6f}" for cat in active_categories),
                resolved_outlier_rank_metric,
                float(cat_args.outlier_residual_min_abs),
                cat_args.outlier_residual_codec,
                int(cat_args.outlier_residual_index_bits),
                int(cat_args.outlier_residual_value_bits),
                int(cat_args.outlier_residual_block_shape[0]),
                int(cat_args.outlier_residual_block_shape[1]),
            )
    log.info(
        "并行配置: linear_group_size=%d, intra_part_sort_mode=%s, total_num_models=%d",
        linear_group_size,
        unique_sort_mode_desc[0] if len(
            unique_sort_mode_desc) == 1 else f"per_category{{{','.join(f'{cat}:{category_sort_mode_desc[cat]}' for cat in active_categories)}}}",
        linear_group_size,
    )
    unique_codebook = sorted(set(category_codebook.values()))
    if unique_codebook:
        if len(unique_codebook) == 1:
            cb_bits, cb_dim = unique_codebook[0]
            log.info("codebook 配置: bits=%d, dim=%d", cb_bits, cb_dim)
        else:
            per_cat_cb_desc = ",".join(
                f"{cat}:[bits={category_codebook[cat][0]},dim={category_codebook[cat][1]}]"
                for cat in active_categories
            )
            log.info("codebook 配置: per_category{%s}", per_cat_cb_desc)
    unique_residual_stages = sorted(set(int(resolved_category_cfgs[cat].residual_stages) for cat in active_categories))
    if len(unique_residual_stages) == 1:
        log.info("residual_stages 配置: %d", unique_residual_stages[0])
    else:
        per_cat_stage_desc = ",".join(
            f"{cat}:{int(resolved_category_cfgs[cat].residual_stages)}"
            for cat in active_categories
        )
        log.info("residual_stages 配置: per_category{%s}", per_cat_stage_desc)

    # 排序代码，已关闭：不再创建排序预处理 ProcessPoolExecutor。

    try:
        eval_tokenizer = None
        if run_task_eval:
            from transformers import AutoTokenizer

            log.info("加载类别后下游任务评估 tokenizer: %s", vae_args.model_path)
            eval_tokenizer = AutoTokenizer.from_pretrained(
                vae_args.model_path,
                use_fast=True,
                token=hf_args.access_token,
            )

        snapshot_path = _save_normalized_cat_train_snapshot(
            run_output_dir=run_output_dir,
            cat_args=cat_args,
            vae_args=vae_args,
            training_args=training_args,
            resolved_category_cfgs=resolved_category_cfgs,
        )
        log.info("Saved normalized parameter snapshot: %s", snapshot_path)
        lora_round_idx = 0
        any_distill_after_overrides = any(table.is_override_enabled() for table, _ in distill_tables)
        if any_distill_after_overrides:
            log.info(
                "Distill after-category overrides enabled: keys=%s",
                ",".join(
                    sorted(
                        {
                            key
                            for table, _arg_name in distill_tables
                            for key in table.by_after_category.keys()
                        }
                    )
                ),
            )
        for cat_idx, cat in enumerate(active_categories):
            refs_sorted, missing = _collect_sorted_category_refs(
                model,
                category=cat,
                transpose_modules=transpose_modules,
                only_decoder_projections=only_decoder_projections,
                target_categories=target_categories,
            )
            if missing:
                log.warning("[%s] %d modules missing layer_idx, skipped.", cat, missing)
            if not refs_sorted:
                continue

            cat_cfg = resolved_category_cfgs[cat]
            refs = [ref for _, ref in refs_sorted]
            cat_codebook_bits, cat_codebook_dim = category_codebook[cat]
            cat_residual_vae_codebook_bits, cat_residual_vae_codebook_dim = category_residual_vae_codebook[cat]
            log.info(
                "=== Category: %s (%d linears, residual_stages=%d, codebook_bits=%d, codebook_dim=%d, residual_vae_codebook_bits=%d, residual_vae_codebook_dim=%d, recon_loss=%s, sort=%s, steps=%d) ===",
                cat,
                len(refs),
                int(cat_cfg.residual_stages),
                int(cat_codebook_bits),
                int(cat_codebook_dim),
                int(cat_residual_vae_codebook_bits),
                int(cat_residual_vae_codebook_dim),
                str(cat_cfg.recon_loss_type),
                category_sort_mode_desc[cat],
                int(cat_cfg.steps),
                # 联合优化代码，已关闭。旧日志字段保留如下：
                # float(cat_cfg.joint_decoder_lr),
                # int(cat_cfg.joint_decoder_group_size),
                # "none" if cat_cfg.joint_decoder_batch_size is None else str(int(cat_cfg.joint_decoder_batch_size)),
            )
            ordered_refs = [r for _, r in refs_sorted]
            if bool(cat_args.allow_tail_group):
                planned_refs = list(ordered_refs)
            else:
                planned_count = (len(ordered_refs) // int(linear_group_size)) * int(linear_group_size)
                planned_refs = list(ordered_refs[:planned_count])
            if skip_layer_keys:
                eligible_plan_refs = []
                for ref in planned_refs:
                    layer_idx = _extract_layer_idx(ref.name)
                    if layer_idx is not None and (int(layer_idx), ref.category) in skip_layer_keys:
                        continue
                    eligible_plan_refs.append(ref)
            else:
                eligible_plan_refs = planned_refs

            outlier_channel_plan: Optional[Dict[str, torch.Tensor]] = None
            if resolved_outlier_mode == "channel" and int(cat_cfg.outlier_protect_count) > 0:
                plan_activation_weight: Optional[Dict[str, torch.Tensor]] = None
                plan_activation_abs_mean: Optional[Dict[str, torch.Tensor]] = None
                if resolved_outlier_rank_metric in {"channel_weight_actmax_abs", "channel_weight_actmean_abs"}:
                    if activation_runtime is None:
                        raise ValueError(
                            f"[{cat}] dynamic activation runtime is required for activation-weighted outlier channel scoring."
                        )
                    dynamic_act_stats, new_cache = collect_activation_stats_for_linears(
                        model=model,
                        linear_items=[(r.name, r.module) for r in eligible_plan_refs],
                        model_path=str(activation_runtime["model_path"]),
                        access_token=activation_runtime.get("access_token"),
                        dataset=str(activation_runtime.get("dataset", "")),
                        nsamples=int(activation_runtime.get("nsamples", 512)),
                        seqlen=int(activation_runtime.get("seqlen", 512)),
                        seed=int(activation_runtime.get("seed", 0)),
                        device=str(activation_runtime.get("device") or cat_args.train_device),
                        cache=activation_runtime.get("cache"),  # type: ignore[arg-type]
                        log_every=int(activation_runtime.get("log_every", 0)),
                        logger=log,
                    )
                    activation_runtime["cache"] = new_cache
                    plan_activation_weight = {
                        name: stats["max"]
                        for name, stats in dynamic_act_stats.items()
                        if isinstance(stats.get("max"), torch.Tensor)
                    }
                    plan_activation_abs_mean = {
                        name: stats["abs_mean"]
                        for name, stats in dynamic_act_stats.items()
                        if isinstance(stats.get("abs_mean"), torch.Tensor)
                    }
                plan_refs = [
                    LinearPrepRef(
                        name=r.name,
                        weight=r.module.weight,
                        in_features=int(r.module.in_features),
                        out_features=int(r.module.out_features),
                        transpose=bool(r.transpose),
                    )
                    for r in eligible_plan_refs
                ]
                outlier_channel_plan = build_outlier_channel_index_plan(
                    group_refs=plan_refs,
                    activation_weight_by_linear=plan_activation_weight,
                    activation_abs_mean_by_linear=plan_activation_abs_mean,
                    outlier_protect_count=int(cat_cfg.outlier_protect_count),
                    outlier_protect_axis=outlier_protect_axis,
                    outlier_channel_scope=str(cat_args.outlier_channel_scope),
                    outlier_rank_metric=resolved_outlier_rank_metric,
                    outlier_protect_min_per_layer=int(cat_args.outlier_protect_min_per_layer),
                )
                for ref in planned_refs:
                    outlier_channel_plan.setdefault(ref.name, torch.empty(0, dtype=torch.long))
                log.info(
                    "[%s] outlier channel plan: mode=%s scope=%s eligible_linears=%d total_channels=%d",
                    cat,
                    resolved_outlier_mode,
                    str(cat_args.outlier_channel_scope),
                    len(plan_refs),
                    sum(int(v.numel()) for v in outlier_channel_plan.values()),
                )

            for start in range(0, len(ordered_refs), linear_group_size):
                group_refs = ordered_refs[start:start + linear_group_size]
                if len(group_refs) < linear_group_size and not cat_args.allow_tail_group:
                    log.info("[%s] tail group size=%d skipped (set --allow_tail_group to include).", cat, len(group_refs))
                    break
                layer_indices = [idx for idx, _ in refs_sorted[start:start + linear_group_size]]
                group_tag = f"{cat}.L{layer_indices[0]}-{layer_indices[-1]}"
                log.info(
                    "---- Group: %s (linears=%d, num_models=%d) ----",
                    group_tag,
                    len(group_refs),
                    len(group_refs),
                )
                _train_group_vae_and_replace(
                    model=model,
                    group_refs=group_refs,
                    group_tag=group_tag,
                    runtime_cfg=cat_cfg,
                    vae_args=vae_args,
                    training_args=training_args,
                    train_device=cat_args.train_device,
                    convert_device=cat_args.convert_device,
                    do_convert=bool(cat_args.convert),
                    batch_size=cat_args.batch_size,
                    log_every=cat_args.log_every,
                    eval_every=cat_args.eval_every,
                    eval_blocks=cat_args.eval_blocks,
                    gpu_resident_data=bool(getattr(cat_args, "gpu_resident_data", False)),
                    skip_layer_keys=skip_layer_keys,
                    activation_runtime=activation_runtime,
                    outlier_protect_mode=resolved_outlier_mode,
                    outlier_channel_plan=outlier_channel_plan,
                    outlier_channel_scope=str(cat_args.outlier_channel_scope),
                    outlier_rank_metric=resolved_outlier_rank_metric,
                    outlier_residual_min_abs=cat_args.outlier_residual_min_abs,
                    outlier_protect_axis=outlier_protect_axis,
                    outlier_protect_min_per_layer=int(cat_args.outlier_protect_min_per_layer),
                    outlier_residual_codec=cat_args.outlier_residual_codec,
                    outlier_residual_index_bits=cat_args.outlier_residual_index_bits,
                    outlier_residual_value_bits=cat_args.outlier_residual_value_bits,
                    outlier_residual_block_shape=cat_args.outlier_residual_block_shape,
                    outlier_residual_vae_decoder_share_scope=cat_args.outlier_residual_vae_decoder_share_scope,
                    outlier_residual_vae_batch_multiplier=cat_args.outlier_residual_vae_batch_multiplier,
                    outlier_residual_vae_steps=cat_args.outlier_residual_vae_steps,
                    outlier_residual_vae_lr=cat_args.outlier_residual_vae_lr,
                    deterministic=bool(cat_args.deterministic),
                    shuffle_seed=int(cat_args.seed) + int(cat_idx) * 100000 + int(start),
                )
            if run_category_eval and distill_after_category == "none":
                log.info("类别训练后评估...")
                _eval_after_category(
                    model=model,
                    vae_args=vae_args,
                    ppl_limit=cat_args.ppl_limit,
                    category=cat,
                    logger=log,
                    eval_device=cat_args.train_device,
                    eval_hif4_act=cat_args.eval_hif4_act,
                    eval_ppl=cat_args.eval_ppl,
                    eval_tasks=eval_tasks_text,
                    tokenizer=eval_tokenizer,
                )

            if distill_after_category != "none":
                if run_category_eval:
                    log.info("每类后蒸馏前评估...")
                    _eval_after_category(
                        model=model,
                        vae_args=vae_args,
                        ppl_limit=cat_args.ppl_limit,
                        category=cat,
                        logger=log,
                        eval_device=cat_args.train_device,
                        eval_hif4_act=cat_args.eval_hif4_act,
                        eval_ppl=cat_args.eval_ppl,
                        eval_tasks=eval_tasks_text,
                        tokenizer=eval_tokenizer,
                    )
                distill_result = run_after_category_distill(
                    model=model,
                    category=cat,
                    cat_args=cat_args,
                    vae_args=vae_args,
                    training_args=training_args,
                    logger=log,
                    lora_round_idx=lora_round_idx,
                    transpose_modules=transpose_modules,
                    only_decoder_projections=only_decoder_projections,
                    target_categories=target_categories,
                )
                model = distill_result.model
                lora_round_idx = int(distill_result.next_lora_round_idx)

            if run_category_eval and distill_after_category != "none":
                log.info("每类后蒸馏后评估...")
                _eval_after_category(
                    model=model,
                    vae_args=vae_args,
                    ppl_limit=cat_args.ppl_limit,
                    category=cat,
                    logger=log,
                    eval_device=cat_args.train_device,
                    eval_hif4_act=cat_args.eval_hif4_act,
                    eval_ppl=cat_args.eval_ppl,
                    eval_tasks=eval_tasks_text,
                    tokenizer=eval_tokenizer,
                )

        if run_category_eval:
            log.info("所有类别训练完成后最终评估...")
            _eval_after_category(
                model=model,
                vae_args=vae_args,
                ppl_limit=cat_args.ppl_limit,
                category="none",
                logger=log,
                eval_device=cat_args.train_device,
                eval_hif4_act=cat_args.eval_hif4_act,
                eval_ppl=cat_args.eval_ppl,
                eval_tasks=eval_tasks_text,
                tokenizer=eval_tokenizer,
            )
        if cat_args.save_model:
            if not cat_args.convert:
                raise ValueError("--save_model requires --convert")
            from transformers import AutoTokenizer
            from e2e_common.post_norm_head import fuse_post_norm_head_linear
            from e2e_common.peft_proxy import iter_named_peft_vae_proxies
            from litebsq.vae_linear import clear_model_vae_linear_cache

            model_out = os.path.join(run_output_dir, "final_model")
            tok = AutoTokenizer.from_pretrained(vae_args.model_path, use_fast=True, token=hf_args.access_token)
            fused_post_norm_head = fuse_post_norm_head_linear(model)
            if fused_post_norm_head:
                log.info("Final save: fused post_norm_linear into lm_head.weight.")
            leftover_proxies = [name for name, _proxy in iter_named_peft_vae_proxies(model)]
            if leftover_proxies:
                raise RuntimeError(
                    "Final save found unexported PeftVAELinearProxy modules: "
                    + ", ".join(leftover_proxies)
                )
            cleared = clear_model_vae_linear_cache(model)
            log.info("Final save: cleared decoded cache for %d VAELinear modules.", cleared)
            save_paths = save_model_checkpoint(
                model,
                model_out,
                base_model_path=vae_args.model_path,
                tokenizer=tok,
                save_config=True,
                extra_meta={"stage": "final"},
                unload_vae_original_weights=bool(cat_args.unload_vae_original_weights_on_final_save),
            )
            log.info("Saved final model to %s", save_paths["output_dir"])
    finally:
        # 排序代码，已关闭：sort_executor 始终为 None。
        pass

    log.info("Done.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    cat_args, hf_args, training_args, vae_args = process_cat_train_args(argv)
    return run_cat_train(
        cat_args=cat_args,
        hf_args=hf_args,
        training_args=training_args,
        vae_args=vae_args,
    )


if __name__ == "__main__":
    main()
