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
from train_utils.cat_category_runtime import (
    ResolvedCategoryRuntimeConfig,
    resolve_category_runtime_configs,
)
from train_utils.cat_after_category_distill import run_after_category_distill
from train_utils.distill_teacher import DistillTeacherRuntime, resolve_distill_teacher_dtype
from train_utils.distributed_guard import distributed_guarded_main
from train_utils.cat_train_runtime import (
    build_cat_run_output_dir as _build_run_output_dir,
    build_distributed_cat_run_output_dir as _build_distributed_run_output_dir,
    load_cat_resume_distill_progress,
    load_model_for_cat_train as _load_model_for_cat_train,
    save_normalized_cat_train_snapshot as _save_normalized_cat_train_snapshot,
)
from train_utils.cat_runtime_state_v6 import (
    build_cat_cross_category_runtime_identity,
    build_cat_runtime_state,
    restore_cat_runtime_state,
    validate_cat_runtime_identity,
)
from train_utils.channel_protection import (
    AdaptiveChannelPlan,
    ChannelLinearSpec,
    category_raw_budget,
    global_raw_budget,
    group_layer_scope_inventory,
    layer_scope_group_seed_offsets,
    resolve_adaptive_channel_plan,
    validate_adaptive_channel_tail_policy,
    validate_global_channel_runtime,
    vae_group_shuffle_seed,
)
from train_utils.config.targets import (
    discover_cat_projection_inventory,
    parse_compression_categories,
    parse_skip_layers,
    parse_target_layers,
    validate_skip_layers_against_inventory,
)
from train_utils.lora_utils import resolve_distill_train_device
from litebsq.vae_args import apply_autoencoder_arch_defaults
from litebsq.misc import set_module_by_name
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
    activation_stats_to_views,
    collect_activation_stats_for_linears,
    collect_mlp_block_activation_stats,
    subset_activation_stats,
)
from train_utils.mlp_channel_selection import (
    MLP_CATEGORIES,
    build_mlp_aligned_plans_all_layers,
    is_mlp_aligned_rank_metric,
    mlp_protect_axis_for_category,
    write_mlp_channel_selection_summary,
)
from train_utils.cat_arg_overrides import resolve_after_category_value, validate_category_keys
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
from train_utils.cat_checkpoint_v6 import save_cat_v6_full_checkpoint
from train_utils.cat_step_resume_v6 import prune_completed_cat_round_roots
from train_utils.cat_inline_distributed import (
    _resolve_cat_inline_vae_wait_timeout_sec,
    broadcast_adaptive_channel_plan,
    broadcast_group_vae_payload,
    initialize_cat_payload_group,
)
from train_utils.lora_utils import (
    distill_distributed_barrier,
    distill_rank,
    distill_world_size,
    ensure_distill_process_group_initialized,
    is_distill_main_process,
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


def _load_checkpoint_tokenizer(model_path: str, access_token):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=access_token,
    )


def _validate_inline_after_category_mode(mode: str) -> None:
    resolved = str(mode).strip().lower()
    allowed = {
        "remaining_lora",
        "remaining_lora_current_decoder",
        "remaining_lora_prefix_decoder",
    }
    if resolved not in allowed:
        raise ValueError(
            "WORLD_SIZE > 1 inline cat_train requires --after_category_mode to be one of: "
            + ",".join(sorted(allowed))
        )



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
    if resolved == "mse":
        return torch.nn.functional.mse_loss(x_recon, x)
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
    if resolved == "amse":
        if act_max is None:
            raise ValueError("recon_loss_type=amse requires hessian_diag/channel_weight tensor.")
        if tuple(act_max.shape) != tuple(x.shape):
            raise ValueError(
                f"amse shape mismatch: hessian_diag={tuple(act_max.shape)} vs x={tuple(x.shape)}"
            )
        x_f = x.float()
        x_recon_f = x_recon.float()
        h_f = act_max.float()
        errors = (x_recon_f - x_f).pow(2)
        return (errors * h_f).mean()
    raise ValueError(
        f"Unsupported recon_loss_type={resolved!r}."
    )


def _compute_reconstruction_eval_metrics(
    x_eval: torch.Tensor,
    x_recon: torch.Tensor,
    *,
    top_k: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if tuple(x_eval.shape) != tuple(x_recon.shape):
        raise ValueError(
            "reconstruction eval shape mismatch: "
            f"{tuple(x_eval.shape)} vs {tuple(x_recon.shape)}"
        )
    if x_eval.ndim != 3:
        raise ValueError(
            "reconstruction eval expects [B, P, C], "
            f"got {tuple(x_eval.shape)}"
        )
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    x_eval_f = x_eval.float()
    squared_error = (x_recon.float() - x_eval_f).pow(2)
    overall_sum = squared_error.sum()
    overall_numel = int(squared_error.numel())

    flat_reference = x_eval_f.permute(1, 0, 2).reshape(
        int(x_eval_f.shape[1]),
        -1,
    )
    flat_error = squared_error.permute(1, 0, 2).reshape(
        int(squared_error.shape[1]),
        -1,
    )
    resolved_k = min(int(top_k), int(flat_reference.shape[1]))
    selected_indices = torch.topk(
        flat_reference.abs(),
        k=resolved_k,
        dim=1,
    ).indices
    selected_error = torch.gather(
        flat_error,
        dim=1,
        index=selected_indices,
    )
    selected_sum = selected_error.sum()
    selected_numel = int(selected_error.numel())
    return (
        overall_sum,
        selected_sum,
        overall_numel,
        selected_numel,
    )


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
    compression_categories: Sequence[str],
) -> List[LinearRef]:
    return _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        categories=compression_categories,
    )


def _collect_sorted_category_refs(
    model: nn.Module,
    *,
    category: str,
    transpose_modules: Sequence[str],
    only_decoder_projections: bool,
    compression_categories: Sequence[str],
) -> Tuple[List[Tuple[int, LinearRef]], int]:
    refs_sorted: List[Tuple[int, LinearRef]] = []
    missing = 0
    for ref in _collect_current_trainable_linears(
        model,
        transpose_modules=transpose_modules,
        only_decoder_projections=only_decoder_projections,
        compression_categories=compression_categories,
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


def _filter_eligible_vae_refs(
    refs_sorted: Sequence[Tuple[int, LinearRef]],
    skip_layer_keys: Set[Tuple[int, str]],
) -> List[Tuple[int, LinearRef]]:
    skipped = {
        (int(layer_idx), str(category))
        for layer_idx, category in (skip_layer_keys or set())
    }
    eligible: List[Tuple[int, LinearRef]] = []
    for layer_idx, ref in refs_sorted:
        if (int(layer_idx), str(ref.category)) in skipped:
            continue
        eligible.append((int(layer_idx), ref))
    return eligible


def _channel_spec_from_ref(
    ref: LinearRef,
    *,
    codebook_dim: int,
    intra_parallel: Tuple[int, int],
    ref_position: int,
    axis: str,
    category: str,
    scores: Optional[torch.Tensor] = None,
) -> ChannelLinearSpec:
    return ChannelLinearSpec(
        name=ref.name,
        in_features=int(ref.module.in_features),
        out_features=int(ref.module.out_features),
        codebook_dim=int(codebook_dim),
        transpose=bool(ref.transpose),
        intra_parallel=tuple(int(v) for v in intra_parallel),
        ref_position=int(ref_position),
        scores=scores,
        axis=str(axis),
        category=str(category),
    )


def _score_channel_specs(
    specs: Sequence[ChannelLinearSpec],
    refs_by_name: Dict[str, LinearRef],
    *,
    metric: str,
    axis: str,
    activation_weight_by_linear: Optional[Dict[str, torch.Tensor]],
    activation_abs_mean_by_linear: Optional[Dict[str, torch.Tensor]],
) -> List[ChannelLinearSpec]:
    scored: List[ChannelLinearSpec] = []
    for spec in specs:
        ref = refs_by_name[spec.name]
        act = None if activation_weight_by_linear is None else activation_weight_by_linear.get(ref.name)
        act_mean = None if activation_abs_mean_by_linear is None else activation_abs_mean_by_linear.get(ref.name)
        score = compute_channel_rank_score(
            metric=metric,
            weight=ref.module.weight,
            residual=None,
            linear_name=ref.name,
            act_max=act,
            act_mean=act_mean,
            act_sq_mean=None,
            axis=axis,
            transpose=bool(ref.transpose),
            expected_in_features=int(ref.module.in_features),
            expected_out_features=int(ref.module.out_features),
        )
        scored.append(replace(spec, scores=score))
    return scored


def _activation_views_for_refs(
    refs: Sequence[LinearRef],
    activation_runtime: Optional[Dict[str, Any]],
    *,
    category: str,
    rank_metric: str,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    if rank_metric not in {"channel_weight_actmax_abs", "channel_weight_actmean_abs"}:
        return None, None
    if activation_runtime is None:
        raise ValueError(
            f"[{category}] dynamic activation runtime is required for activation-weighted channel scoring."
        )
    stats_by_linear = activation_runtime.get("stats_by_linear")
    if not isinstance(stats_by_linear, dict):
        raise ValueError(
            f"[{category}] precomputed activation stats are required but missing from activation_runtime."
        )
    subset_stats = subset_activation_stats(stats_by_linear, [ref.name for ref in refs])
    weight_view, abs_mean_view, _ = activation_stats_to_views(subset_stats)
    return weight_view, abs_mean_view


def _pairs_from_name_groups(
    name_groups: Sequence[Sequence[str]],
    pair_by_name: Dict[str, Tuple[int, LinearRef]],
) -> List[List[Tuple[int, LinearRef]]]:
    grouped: List[List[Tuple[int, LinearRef]]] = []
    for names in name_groups:
        grouped.append([pair_by_name[name] for name in names])
    return grouped


def _outlier_plan_from_adaptive(
    plan: AdaptiveChannelPlan,
    names: Optional[Set[str]] = None,
) -> Dict[str, torch.Tensor]:
    selected = plan.selected_indices
    if names is not None:
        selected = {name: indices for name, indices in selected.items() if name in names}
    return {
        name: torch.tensor(indices, dtype=torch.long)
        for name, indices in selected.items()
    }


def _is_skipped_linear_ref(ref: LinearRef, skip_layer_keys: Set[Tuple[int, str]]) -> bool:
    layer_idx = _extract_layer_idx(ref.name)
    return (
        layer_idx is not None
        and (int(layer_idx), str(ref.category)) in {
            (int(skipped_layer_idx), str(skipped_category))
            for skipped_layer_idx, skipped_category in (skip_layer_keys or set())
        }
    )


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
    protected_channel_quant_format: str = "none",
):
    from litebsq.vae_linear import VAELinear

    def _has_protected_input_payload(split_meta) -> bool:
        return (
            split_meta.protected_input_weight is not None
            or split_meta.protected_input_qvalues is not None
        )

    def _has_protected_output_payload(split_meta) -> bool:
        return (
            split_meta.protected_output_weight is not None
            or split_meta.protected_output_qvalues is not None
        )

    residual_stages = int(len(stage_part_bits_payload))
    if residual_stages < 1:
        raise ValueError("stage_part_bits_payload cannot be empty.")
    common_kwargs = dict(
        in_features=old_module.in_features,
        out_features=old_module.out_features,
        bias=bias,
        original_weight=None,
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
            if _has_protected_input_payload(split_meta)
            else None
        ),
        protected_input_weight=split_meta.protected_input_weight,
        protected_input_qvalues=split_meta.protected_input_qvalues,
        protected_input_scales=split_meta.protected_input_scales,
        protected_output_indices=(
            split_meta.protected_output_indices
            if _has_protected_output_payload(split_meta)
            else None
        ),
        protected_output_weight=split_meta.protected_output_weight,
        protected_output_qvalues=split_meta.protected_output_qvalues,
        protected_output_scales=split_meta.protected_output_scales,
        protected_channel_quant_format=str(protected_channel_quant_format),
        always_use_original=False,
        protect_original_weight=False,
    )
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
    activation_runtime: Optional[Dict[str, object]] = None,
    channel_protect_mode: str = "channel",
    channel_plan: Optional[Dict[str, torch.Tensor]] = None,
    channel_scope: str = "layer",
    channel_rank_metric: str = "channel_weight_abs",
    channel_axis: str = "input",
    channel_quant: str = "none",
    channel_min_per_layer: int = 0,
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
    stage_activation_type = str(runtime_cfg.activation_type).strip().lower()
    stage_decoder_type = str(runtime_cfg.decoder_type).strip().lower()
    channel_protect_count = int(runtime_cfg.channel_protect_count)
    resolved_channel_mode = str(channel_protect_mode).strip().lower()
    resolved_channel_rank_metric = str(channel_rank_metric).strip().lower()
    channel_protection_enabled = resolved_channel_mode == "channel" and int(channel_protect_count) > 0
    channel_rank_needs_actmax = (
        channel_protection_enabled
        and resolved_channel_rank_metric == "channel_weight_actmax_abs"
    )
    channel_rank_needs_actmean = (
        channel_protection_enabled
        and resolved_channel_rank_metric == "channel_weight_actmean_abs"
    )
    if resolved_channel_mode not in {"none", "channel"}:
        raise ValueError(
            f"[{group_tag}] unsupported channel_protect_mode={channel_protect_mode!r}. "
            "Expected none or channel."
        )
    use_wa_mse_loss = stage_recon_loss == "wa_mse"
    use_channel_weight_loss = stage_recon_loss in {"wa_mse", "amse"}
    intra_parallel = tuple(getattr(runtime_cfg, "intra_parallel", (1, 1)))
    if len(intra_parallel) != 2:
        raise ValueError(f"[{group_tag}] intra_parallel must be a (row_parts, col_parts) pair, got {intra_parallel!r}.")
    row_parts, col_parts = int(intra_parallel[0]), int(intra_parallel[1])
    parts_per_linear = int(row_parts) * int(col_parts)
    sort_mode = str(runtime_cfg.intra_part_sort_mode).strip().lower()
    needs_dynamic_activation = (
        use_channel_weight_loss
        or channel_rank_needs_actmax
        or channel_rank_needs_actmean
    )
    effective_activation_weight: Optional[Dict[str, torch.Tensor]] = None
    effective_activation_abs_mean: Optional[Dict[str, torch.Tensor]] = None
    effective_activation_sq_mean: Optional[Dict[str, torch.Tensor]] = None
    if needs_dynamic_activation:
        if activation_runtime is None:
            raise ValueError(
                f"[{group_tag}] dynamic activation runtime is required for wa_mse/amse or channel protection."
            )
        stats_by_linear = activation_runtime.get("stats_by_linear")
        if not isinstance(stats_by_linear, dict):
            raise ValueError(
                f"[{group_tag}] precomputed activation stats are required but missing from activation_runtime."
            )
        subset_stats = subset_activation_stats(stats_by_linear, [r.name for r in group_refs])
        (
            effective_activation_weight,
            effective_activation_abs_mean,
            effective_activation_sq_mean,
        ) = activation_stats_to_views(subset_stats)
        log.info(
            "[%s] using precomputed activation stats (linears=%d).",
            group_tag,
            len(subset_stats),
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
    stage_recon_loss_resolved = str(stage_recon_loss).strip().lower()
    if stage_recon_loss_resolved == "wa_mse":
        prep_channel_weight_by_linear = effective_activation_weight
    elif stage_recon_loss_resolved == "amse":
        prep_channel_weight_by_linear = effective_activation_sq_mean
    else:
        prep_channel_weight_by_linear = effective_activation_weight
    prepared_entries = prepare_group_linear_entries(
        group_refs=prep_refs,
        activation_weight_by_linear=prep_channel_weight_by_linear,
        activation_abs_mean_by_linear=effective_activation_abs_mean,
        channel_protect_count=(
            int(channel_protect_count)
            if resolved_channel_mode == "channel" and channel_protection_enabled
            else 0
        ),
        channel_axis=str(channel_axis),
        channel_quant=str(channel_quant),
        recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
        intra_part_sort_mode=stage_sort_mode,
        channel_plan=channel_plan if resolved_channel_mode == "channel" and channel_protection_enabled else None,
        apply_outlier_channel_removal=resolved_channel_mode == "channel",
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
    if resolved_channel_mode == "channel" and channel_protection_enabled:
        per_linear_protected = []
        for ref, meta in zip(group_refs, target_common_split_metas):
            if str(channel_axis) == "output":
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
            "[%s] channel protection mode=%s axis=%s count=%d protected_channels=%s",
            group_tag,
            resolved_channel_mode,
            str(channel_axis),
            int(channel_protect_count),
            ",".join(per_linear_protected),
        )
    if use_wa_mse:
        log.info("[%s] %s enabled with online channel weight gather.", group_tag, stage_recon_loss)
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
        activation_type=str(stage_activation_type),
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
        all_batch_gpu_cache = bool(
            batch_size_is_all
            and stage_recon_loss not in {"wa_mse", "amse"}
            and not gpu_resident_enabled
        )
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
            "[%s] start (residual_rms=%.6e, steps=%d, blocks=%d, bits=%d, dim=%d, recon_loss=%s, base_ch=%d, num_res_blocks=%d, norm_type=%s, activation_type=%s, decoder_type=%s, stage_norm=%s)",
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
            stage_activation_type,
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
                    if stage_recon_loss in {"wa_mse", "amse"}:
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
                    if stage_recon_loss in {"wa_mse", "amse"}:
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
                if stage_recon_loss in {"wa_mse", "amse"}:
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
                        (
                            batch_overall_sum,
                            batch_selected_sum,
                            batch_overall_numel,
                            batch_selected_numel,
                        ) = _compute_reconstruction_eval_metrics(
                            x_eval,
                            x_recon,
                            top_k=100,
                        )
                        if batch_overall_numel > 0:
                            mse_sum += float(batch_overall_sum.detach().cpu().item())
                            mse_numel += batch_overall_numel
                            eval_blocks_seen += int(x_eval.shape[0])
                        if batch_selected_numel > 0:
                            top_k_mse_sum += float(batch_selected_sum.detach().cpu().item())
                            top_k_mse_numel += batch_selected_numel
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

        # 训完立刻释放 Adam 状态，降低随后全量重构的显存尖峰（不影响重构数值）。
        del optimizer
        optimizer = None
        if lr_scheduler is not None:
            del lr_scheduler
            lr_scheduler = None
        # all_batch_gpu_cache 路径下 x_all 仅服务训练取 batch；重构走 eval_loader。
        # gpu_resident + batch_size=all 时 x_all 与 gpu_stage_train_data 是同一引用，不能在这里删。
        if x_all is not None and x_all is not gpu_stage_train_data:
            del x_all
            x_all = None
        torch.cuda.empty_cache()

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
            x_all = None
        del vae, train_loader, eval_loader, common_stage_result, stage_result
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
        "protected_channel_quant_format": str(channel_quant),
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
    protected_channel_quant_format = str(payload.get("protected_channel_quant_format", "none"))

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
            protected_channel_quant_format=protected_channel_quant_format,
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
    activation_runtime: Optional[Dict[str, object]] = None,
    channel_protect_mode: str = "channel",
    channel_plan: Optional[Dict[str, torch.Tensor]] = None,
    channel_scope: str = "layer",
    channel_rank_metric: str = "channel_weight_abs",
    channel_axis: str = "input",
    channel_quant: str = "none",
    channel_min_per_layer: int = 0,
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
        activation_runtime=activation_runtime,
        channel_protect_mode=channel_protect_mode,
        channel_plan=channel_plan,
        channel_scope=channel_scope,
        channel_rank_metric=channel_rank_metric,
        channel_axis=channel_axis,
        channel_quant=channel_quant,
        channel_min_per_layer=channel_min_per_layer,
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
    )


def _train_group_vae_and_replace_inline_distributed(
    *,
    model: nn.Module,
    group_refs: Sequence[LinearRef],
    group_tag: str,
    inline_distributed: bool,
    **vae_group_kwargs,
) -> int:
    """Run VAE on rank 0, then apply its broadcast payload on every rank."""
    if not inline_distributed:
        _train_group_vae_and_replace(
            model=model,
            group_refs=group_refs,
            group_tag=group_tag,
            **vae_group_kwargs,
        )
        return len(group_refs) if bool(vae_group_kwargs["do_convert"]) else 0

    rank = distill_rank()
    if rank == 0:
        log.info("Cat inline VAE train start: group=%s rank=0", group_tag)
        vae_started_at = time.perf_counter()
        payload = train_group_vae_payload(
            model=model,
            group_refs=group_refs,
            group_tag=group_tag,
            **vae_group_kwargs,
        )
        vae_elapsed_sec = time.perf_counter() - vae_started_at
        log.info(
            "Cat inline VAE train complete: group=%s rank=0 elapsed_sec=%.1f elapsed_min=%.2f",
            group_tag,
            vae_elapsed_sec,
            vae_elapsed_sec / 60.0,
        )
    else:
        payload = None
        log.info("Cat inline VAE wait: group=%s rank=%d (no VAE training)", group_tag, rank)

    if not bool(vae_group_kwargs["do_convert"]):
        raise RuntimeError("Inline distributed CAT requires --convert.")
    if rank == 0:
        timeout_sec = _resolve_cat_inline_vae_wait_timeout_sec()
        log.info(
            "Cat inline VAE payload sync start: group=%s timeout_sec=%d",
            group_tag,
            timeout_sec,
        )
    sync_started_at = time.perf_counter()
    payload = broadcast_group_vae_payload(payload, src=0)
    sync_elapsed_sec = time.perf_counter() - sync_started_at
    if rank == 0:
        log.info(
            "Cat inline VAE payload sync complete: group=%s elapsed_sec=%.1f",
            group_tag,
            sync_elapsed_sec,
        )
    apply_group_vae_payload(
        model=model,
        group_refs=group_refs,
        group_tag=group_tag,
        payload=payload,
        convert_device=(
            str(vae_group_kwargs["convert_device"])
            if rank == 0
            else "cpu"
        ),
    )
    distill_distributed_barrier()
    return len(group_refs)


def run_cat_train(*, cat_args, hf_args, training_args, vae_args) -> None:
    global log
    after_category_mode = str(getattr(cat_args, "after_category_mode", "none")).strip().lower()
    world_size = distill_world_size()
    inline_distributed = world_size > 1
    if inline_distributed:
        _validate_inline_after_category_mode(after_category_mode)
        ensure_distill_process_group_initialized()
        initialize_cat_payload_group()
    if bool(getattr(training_args, "distill_hif4_act", False)) and after_category_mode == "none":
        raise ValueError("--distill_hif4_act 仅在每类后蒸馏阶段生效，因此必须设置 --after_category_mode。")
    if after_category_mode != "none" and not bool(cat_args.convert):
        raise ValueError("--after_category_mode requires --convert，因为每类后蒸馏必须作用在已替换的压缩模型上。")
    configure_deterministic_mode(bool(getattr(cat_args, "deterministic", False)))
    set_seed(cat_args.seed)

    os.makedirs(cat_args.output_dir, exist_ok=True)
    run_output_dir = (
        _build_distributed_run_output_dir(cat_args.output_dir, vae_args.model_path)
        if inline_distributed
        else _build_run_output_dir(cat_args.output_dir, vae_args.model_path)
    )
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "linear_by_category.log")
    log = get_logger("linear_by_category")
    cat_args.output_dir = run_output_dir

    log.info("Run output directory: %s", run_output_dir)
    if inline_distributed and is_distill_main_process():
        log.info(
            "Cat inline distributed mode: world_size=%d, VAE rank=0, distill=%s",
            world_size,
            after_category_mode,
        )
    if bool(getattr(cat_args, "distill_independent_categories", False)):
        log.warning(
            "--distill_independent_categories=true 仅对 cat checkpoint distill 生效；"
            "当前 inline cat_train 路径忽略该开关，仍按前缀累积压缩状态蒸馏。"
        )
    if bool(getattr(cat_args, "deterministic", False)):
        log.info("Deterministic mode enabled: torch deterministic algorithms on, TF32 disabled.")
    log.info(
        "Args:\nscript=%s\nvae=%s\ntraining=%s",
        _format_namespace(cat_args),
        _format_namespace(vae_args),
        _format_namespace(training_args),
    )

    model = _load_model_for_cat_train(cat_args=cat_args, hf_args=hf_args, vae_args=vae_args)
    teacher_runtime = None
    if after_category_mode != "none":
        teacher_runtime = DistillTeacherRuntime(
            model_path=str(vae_args.model_path),
            access_token=hf_args.access_token,
            forward_device=resolve_distill_train_device(cat_args.train_device),
            dtype=resolve_distill_teacher_dtype(training_args, model),
            model_offload=str(getattr(training_args, "distill_teacher_model_offload", "none")),
            logger=log,
        )
    activation_runtime: Optional[Dict[str, object]] = None
    channel_axis = str(getattr(cat_args, "channel_axis", "input")).strip().lower()
    channel_quant = str(getattr(cat_args, "channel_quant", "none")).strip().lower()
    transpose_modules = _split_csv(cat_args.transpose_modules)
    compression_categories = _split_csv(cat_args.compression_categories)
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
        compression_categories=compression_categories,
    )
    compression_category_values = parse_compression_categories(
        getattr(cat_args, "compression_categories", cat_args.compression_categories)
    )
    target_layers = parse_target_layers(getattr(cat_args, "target_layers", "all"))
    skip_layers = parse_skip_layers(getattr(cat_args, "skip_layers", ""))
    inventory = discover_cat_projection_inventory(
        model,
        compression_categories=compression_category_values,
    )
    discovered_category_set = {str(category) for _layer_idx, category in inventory}
    missing_compression_categories = [
        category for category in compression_categories if category not in discovered_category_set
    ]
    if missing_compression_categories:
        raise ValueError(
            "compression_categories contains categories not found in canonical CAT projection inventory: "
            + ",".join(missing_compression_categories)
        )
    validate_skip_layers_against_inventory(
        skip_layers,
        target_layers=target_layers,
        compression_categories=compression_category_values,
        inventory=inventory,
    )
    skip_layer_keys = set(skip_layers)
    if skip_layer_keys:
        log.info(
            "skip_layers 生效: %s",
            ",".join(f"{li}.{cat}" for li, cat in sorted(skip_layer_keys)),
        )
    eligible_all_linears = [
        ref for ref in all_linears
        if not _is_skipped_linear_ref(ref, skip_layer_keys)
    ]

    linear_group_size = int(cat_args.linear_group_size)
    if linear_group_size < 1:
        raise ValueError(f"linear_group_size must be >= 1, got {linear_group_size}")

    active_categories = list(compression_categories)
    if not active_categories:
        raise ValueError("No active categories discovered for training.")

    resolved_category_cfgs = resolve_category_runtime_configs(cat_args, vae_args, active_categories)
    cat_runtime_identity = build_cat_cross_category_runtime_identity(
        cat_args=cat_args,
        vae_args=vae_args,
        resolved_category_cfgs=resolved_category_cfgs,
        compression_categories=compression_category_values,
        target_layers=target_layers,
        skip_layers=skip_layers,
        transpose_modules=transpose_modules,
    )
    restored_global_adaptive_plan: Optional[AdaptiveChannelPlan] = None
    resume_runtime_payload = getattr(cat_args, "_v6_cat_runtime_state_payload", None)
    if getattr(cat_args, "resume_from_checkpoint", None):
        if resume_runtime_payload is None:
            raise FileNotFoundError(
                "Exact CAT v6 resume requires cat_runtime_state.pt on the referenced category boundary/round base."
            )
        activation_runtime, restored_global_adaptive_plan, saved_runtime_identity = restore_cat_runtime_state(
            resume_runtime_payload,
            access_token=getattr(hf_args, "access_token", None),
        )
        validate_cat_runtime_identity(saved_runtime_identity, cat_runtime_identity)
        log.info(
            "Restored exact cross-category CAT runtime state: activation=%s global_plan=%s",
            bool(activation_runtime is not None),
            bool(restored_global_adaptive_plan is not None),
        )
    resolved_channel_mode = str(getattr(cat_args, "channel_protect_mode", "channel")).strip().lower()
    if resolved_channel_mode not in {"none", "channel"}:
        raise ValueError(
            f"unsupported channel_protect_mode={resolved_channel_mode!r}. Expected none or channel."
        )
    resolved_channel_rank_metric = str(getattr(cat_args, "channel_rank_metric", "channel_weight_abs")).strip().lower()
    resolved_channel_mlp_rank_metric = str(getattr(cat_args, "channel_mlp_rank_metric", "none")).strip().lower()
    log.info(
        "channel_mode=%s channel_rank_metric=%s channel_mlp_rank_metric=%s",
        resolved_channel_mode,
        resolved_channel_rank_metric,
        resolved_channel_mlp_rank_metric,
    )
    category_codebook: Dict[str, Tuple[int, int]] = {
        cat: (
            int(resolved_category_cfgs[cat].codebook_bits),
            int(resolved_category_cfgs[cat].codebook_dim),
        )
        for cat in active_categories
    }
    category_channel_protect_count: Dict[str, int] = {
        cat: int(resolved_category_cfgs[cat].channel_protect_count) for cat in active_categories
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
    any_amse = any(str(resolved_category_cfgs[cat].recon_loss_type).strip(
    ).lower() == "amse" for cat in active_categories)
    any_channel_weight_loss = any_wa_mse or any_amse
    any_channel_protect = any(count > 0 for count in category_channel_protect_count.values())
    channel_protect_needs_activation = (
        resolved_channel_mode == "channel"
        and resolved_channel_rank_metric in {
            "channel_weight_actmax_abs",
            "channel_weight_actmean_abs",
        }
        and any_channel_protect
    )
    mlp_channel_protect_needs_activation = (
        is_mlp_aligned_rank_metric(resolved_channel_mlp_rank_metric)
        and resolved_channel_mode == "channel"
        and any(
            int(category_channel_protect_count.get(cat, 0)) > 0
            for cat in MLP_CATEGORIES
            if cat in active_categories
        )
    )
    sort_needs_act = False  # 排序代码，已关闭。
    if (
        any_channel_weight_loss
        or channel_protect_needs_activation
        or mlp_channel_protect_needs_activation
        or sort_needs_act
    ) and activation_runtime is None and (not inline_distributed or is_distill_main_process()):
        activation_dataset = str(cat_args.activation_calib_dataset).strip()
        if not activation_dataset:
            raise ValueError(
                "--activation_calib_dataset must be set when dynamic activation calibration is enabled. "
                "Use ratio-style dataset specs such as 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
            )
        activation_cache: Optional[ActivationCalibrationCache] = None
        activation_runtime = {
            "cache": activation_cache,
            "dataset": activation_dataset,
            "nsamples": int(cat_args.activation_calib_nsamples),
            "seqlen": int(cat_args.activation_calib_seqlen),
            "seed": int(cat_args.activation_calib_seed),
            "device": str(cat_args.activation_calib_device).strip() or str(cat_args.train_device),
            "log_every": int(cat_args.activation_calib_log_every),
            "model_path": str(vae_args.model_path),
            "access_token": hf_args.access_token,
        }
        enabled_features: List[str] = []
        if any_wa_mse:
            enabled_features.append("wa_mse")
        if any_amse:
            enabled_features.append("amse")
        if channel_protect_needs_activation:
            enabled_features.append("channel_protection")
        if mlp_channel_protect_needs_activation:
            enabled_features.append("mlp_channel_protection")
        log.info(
            "Dynamic activation calibration enabled for %s: dataset=%s nsamples=%d seqlen=%d seed=%d device=%s",
            ",".join(enabled_features),
            str(activation_runtime["dataset"]),
            int(activation_runtime["nsamples"]),
            int(activation_runtime["seqlen"]),
            int(activation_runtime["seed"]),
            str(activation_runtime["device"]),
        )
        linear_items = [(r.name, r.module) for r in eligible_all_linears]
        stats_by_linear, new_cache = collect_activation_stats_for_linears(
            model=model,
            linear_items=linear_items,
            model_path=str(activation_runtime["model_path"]),
            access_token=activation_runtime.get("access_token"),
            dataset=str(activation_runtime.get("dataset", "")),
            nsamples=int(activation_runtime["nsamples"]),
            seqlen=int(activation_runtime["seqlen"]),
            seed=int(activation_runtime["seed"]),
            device=str(activation_runtime["device"]),
            cache=activation_runtime.get("cache"),  # type: ignore[arg-type]
            log_every=int(activation_runtime["log_every"]),
            logger=log,
        )
        activation_runtime["cache"] = new_cache
        activation_runtime["stats_by_linear"] = stats_by_linear
        log.info(
            "Prefilled activation stats for %d linears (one-shot, dataset=%s nsamples=%d seqlen=%d seed=%d device=%s).",
            len(stats_by_linear),
            str(activation_runtime["dataset"]),
            int(activation_runtime["nsamples"]),
            int(activation_runtime["seqlen"]),
            int(activation_runtime["seed"]),
            str(activation_runtime["device"]),
        )

    mlp_channel_plan_by_linear: Optional[Dict[str, torch.Tensor]] = (
        None
        if activation_runtime is None
        else activation_runtime.get("mlp_channel_plan_by_linear")
    )
    if (
        is_mlp_aligned_rank_metric(resolved_channel_mlp_rank_metric)
        and resolved_channel_mode == "channel"
        and mlp_channel_plan_by_linear is None
        and (not inline_distributed or is_distill_main_process())
    ):
        mlp_protect_count = int(category_channel_protect_count.get("gate_proj", 0))
        if mlp_protect_count > 0:
            if activation_runtime is None:
                raise ValueError(
                    "MLP aligned channel selection requires activation_runtime, "
                    "but dynamic activation calibration was not initialized."
                )
            mlp_layer_indices = sorted(
                {
                    int(layer_idx)
                    for ref in eligible_all_linears
                    if ref.category in MLP_CATEGORIES
                    for layer_idx in [_extract_layer_idx(ref.name)]
                    if layer_idx is not None
                }
            )
            stats_by_mlp_block, mlp_cache = collect_mlp_block_activation_stats(
                model=model,
                layer_indices=mlp_layer_indices,
                model_path=str(activation_runtime["model_path"]),
                access_token=activation_runtime.get("access_token"),
                dataset=str(activation_runtime.get("dataset", "")),
                nsamples=int(activation_runtime["nsamples"]),
                seqlen=int(activation_runtime["seqlen"]),
                seed=int(activation_runtime["seed"]),
                device=str(activation_runtime["device"]),
                cache=activation_runtime.get("cache"),  # type: ignore[arg-type]
                skip_layer_keys=skip_layer_keys,
                log_every=int(activation_runtime["log_every"]),
                logger=log,
            )
            activation_runtime["cache"] = mlp_cache
            activation_runtime["stats_by_mlp_block"] = stats_by_mlp_block
            mlp_channel_plan_by_linear, mlp_summary_by_layer = build_mlp_aligned_plans_all_layers(
                model=model,
                stats_by_mlp_block=stats_by_mlp_block,
                protect_count=int(mlp_protect_count),
                fuse_weights=cat_args.channel_mlp_fuse_weights,
                rank_metric=resolved_channel_mlp_rank_metric,
                skip_layer_keys=skip_layer_keys,
            )
            activation_runtime["mlp_channel_plan_by_linear"] = mlp_channel_plan_by_linear
            summary_path = os.path.join(run_output_dir, "mlp_channel_selection_summary.json")
            write_mlp_channel_selection_summary(
                summary_path,
                summary_by_layer=mlp_summary_by_layer,
                protect_count=int(mlp_protect_count),
                fuse_weights=cat_args.channel_mlp_fuse_weights,
                rank_metric=resolved_channel_mlp_rank_metric,
            )
            log.info(
                "MLP aligned channel plan ready: layers=%d linears=%d protect_count=%d summary=%s",
                len(mlp_summary_by_layer),
                len(mlp_channel_plan_by_linear),
                int(mlp_protect_count),
                summary_path,
            )

    if any_channel_protect:
        enabled_counts = ",".join(
            f"{cat}:{count}"
            for cat, count in category_channel_protect_count.items()
            if count > 0
        )
        log.info("Channel protection enabled: axis=%s count_by_category=%s", channel_axis, enabled_counts)
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

        if not inline_distributed or is_distill_main_process():
            snapshot_path = _save_normalized_cat_train_snapshot(
                run_output_dir=run_output_dir,
                cat_args=cat_args,
                vae_args=vae_args,
                training_args=training_args,
                resolved_category_cfgs=resolved_category_cfgs,
            )
            log.info("Saved normalized parameter snapshot: %s", snapshot_path)
        if inline_distributed:
            distill_distributed_barrier()
        resume_progress = load_cat_resume_distill_progress(
            getattr(cat_args, "resume_from_checkpoint", None)
        )
        completed_categories: List[str] = list(resume_progress.completed_categories)
        completed_category_set = set(completed_categories)
        lora_round_idx = int(resume_progress.lora_round_idx)
        resume_active_category = (
            None if resume_progress.active_category is None else str(resume_progress.active_category)
        )
        resume_step_checkpoint = resume_progress.training_step_checkpoint
        distill_stage_history: List[dict] = [
            dict(item) for item in resume_progress.distill_stage_history
        ]
        if completed_categories or distill_stage_history:
            log.info(
                "Inline CAT resume progress: completed_categories=%s lora_round_idx=%d history_entries=%d",
                ",".join(completed_categories) if completed_categories else "(none)",
                int(lora_round_idx),
                int(len(distill_stage_history)),
            )
        global_adaptive_plan: Optional[AdaptiveChannelPlan] = restored_global_adaptive_plan
        global_raw_budget_value = (
            0 if global_adaptive_plan is None else int(global_adaptive_plan.raw_budget)
        )
        plan_is_main = (not inline_distributed) or is_distill_main_process()
        plan_world_size = int(world_size) if inline_distributed else 1
        if (
            resolved_channel_mode == "channel"
            and str(cat_args.channel_scope).strip().lower() == "global"
        ):
            validate_global_channel_runtime(
                channel_scope="global",
                channel_protect_mode=str(resolved_channel_mode),
                channel_axis=str(channel_axis),
            )
            if global_adaptive_plan is not None:
                validate_adaptive_channel_tail_policy(
                    scope="global",
                    budget=int(global_raw_budget_value),
                    allow_tail_group=bool(cat_args.allow_tail_group),
                )
                if plan_is_main:
                    log.info(
                        "Restored global channel plan: raw_budget=%d used_channels=%d",
                        int(global_adaptive_plan.raw_budget),
                        int(global_adaptive_plan.used_channels),
                    )
            else:
                ratio = float(getattr(cat_args, "channel_protect_count_ratio", 0.0) or 0.0)
                all_specs: List[ChannelLinearSpec] = []
                all_refs: List[LinearRef] = []
                ref_pos = 0
                for cat in active_categories:
                    if str(cat) in completed_category_set:
                        continue
                    refs_sorted, _missing = _collect_sorted_category_refs(
                        model,
                        category=cat,
                        transpose_modules=transpose_modules,
                        only_decoder_projections=only_decoder_projections,
                        compression_categories=compression_categories,
                    )
                    pairs = _filter_eligible_vae_refs(refs_sorted, skip_layer_keys)
                    if target_layers != "all":
                        allowed_layers = {int(idx) for idx in target_layers}
                        pairs = [
                            (layer_idx, ref)
                            for layer_idx, ref in pairs
                            if int(layer_idx) in allowed_layers
                        ]
                    cat_cfg = resolved_category_cfgs[cat]
                    intra_parallel = tuple(getattr(cat_cfg, "intra_parallel", (1, 1)))
                    for _layer_idx, ref in pairs:
                        all_specs.append(
                            _channel_spec_from_ref(
                                ref,
                                codebook_dim=int(cat_cfg.codebook_dim),
                                intra_parallel=intra_parallel,
                                ref_position=int(ref_pos),
                                axis=str(channel_axis),
                                category=str(cat),
                            )
                        )
                        all_refs.append(ref)
                        ref_pos += 1
                global_raw_budget_value = global_raw_budget(all_specs, ratio)
                validate_adaptive_channel_tail_policy(
                    scope="global",
                    budget=int(global_raw_budget_value),
                    allow_tail_group=bool(cat_args.allow_tail_group),
                )
                if int(global_raw_budget_value) > 0:
                    refs_by_name = {ref.name: ref for ref in all_refs}

                    def _global_activation_views(_specs: Sequence[ChannelLinearSpec]):
                        return _activation_views_for_refs(
                            all_refs,
                            activation_runtime,
                            category="global",
                            rank_metric=resolved_channel_rank_metric,
                        )

                    def _global_score_specs(specs, act_weight, act_mean):
                        return _score_channel_specs(
                            specs,
                            refs_by_name,
                            metric=resolved_channel_rank_metric,
                            axis=str(channel_axis),
                            activation_weight_by_linear=act_weight,
                            activation_abs_mean_by_linear=act_mean,
                        )

                    global_adaptive_plan = resolve_adaptive_channel_plan(
                        all_specs,
                        raw_budget=int(global_raw_budget_value),
                        min_per_layer=int(cat_args.channel_min_per_layer),
                        linear_group_size=int(linear_group_size),
                        metric=str(resolved_channel_rank_metric),
                        axis=str(channel_axis),
                        scope="global",
                        group_by_category=True,
                        is_main=bool(plan_is_main),
                        world_size=int(plan_world_size),
                        broadcast_fn=broadcast_adaptive_channel_plan,
                        activation_view_fn=_global_activation_views,
                        score_fn=_global_score_specs,
                        run_output_dir=str(run_output_dir) if plan_is_main else None,
                    )
                    if plan_is_main:
                        log.info(
                            "global channel plan: eligible_linears=%d raw_budget=%d used_channels=%d",
                            len(all_specs),
                            int(global_adaptive_plan.raw_budget),
                            int(global_adaptive_plan.used_channels),
                        )

        for cat_idx, cat in enumerate(active_categories):
            if str(cat) in completed_category_set:
                log.info(
                    "[%s] resume progress: category already completed; skip VAE compression and after-category recovery.",
                    str(cat),
                )
                continue
            resuming_active_recovery = bool(
                resume_step_checkpoint is not None
                and resume_active_category is not None
                and str(cat) == resume_active_category
            )
            if resume_step_checkpoint is not None and not resuming_active_recovery:
                raise RuntimeError(
                    "CAT training-step resume reached a non-completed category before its active recovery round: "
                    f"active_category={resume_active_category!r}, current={cat!r}, "
                    f"completed={completed_categories}."
                )
            if resuming_active_recovery and after_category_mode == "none":
                raise ValueError(
                    "CAT training-step resume requires an active after-category recovery mode; "
                    "after_category_mode=none has no Trainer step state to resume."
                )

            current_category_target_names: Optional[Tuple[str, ...]] = None
            resume_newly_compressed_target_count = 0
            if resuming_active_recovery:
                resume_source = getattr(cat_args, "_v6_resume_source", None)
                if resume_source is None or getattr(resume_source, "model_checkpoint_kind", None) != "round_base":
                    raise RuntimeError("CAT training-step resume is missing its resolved v6 round_base source.")
                round_base_meta = dict(resume_source.model_checkpoint_meta)
                compressed_names = set(str(name) for name in round_base_meta.get("compressed_targets") or ())
                allowed_resume_layers = (
                    None if target_layers == "all" else {int(v) for v in target_layers}
                )
                current_category_target_names = tuple(
                    str(name)
                    for (layer_idx, category), name in inventory.items()
                    if str(category) == str(cat)
                    and (allowed_resume_layers is None or int(layer_idx) in allowed_resume_layers)
                    and (int(layer_idx), str(category)) not in skip_layer_keys
                    and str(name) in compressed_names
                )
                resume_newly_compressed_target_count = len(current_category_target_names)
                if not current_category_target_names:
                    raise RuntimeError(
                        "CAT training-step round_base contains no compressed targets for its active category: "
                        f"category={cat!r}."
                    )
                log.info(
                    "[%s] exact step resume: reuse round_base compressed state for %d targets; "
                    "skip VAE compression and resume after-category Trainer from %s.",
                    str(cat),
                    int(resume_newly_compressed_target_count),
                    str(resume_step_checkpoint),
                )
                refs_sorted, missing = [], 0
            else:
                refs_sorted, missing = _collect_sorted_category_refs(
                    model,
                    category=cat,
                    transpose_modules=transpose_modules,
                    only_decoder_projections=only_decoder_projections,
                    compression_categories=compression_categories,
                )
            if missing:
                log.warning("[%s] %d modules missing layer_idx, skipped.", cat, missing)
            if not refs_sorted and not resuming_active_recovery:
                continue

            cat_cfg = resolved_category_cfgs[cat]
            eligible_vae_pairs = (
                [] if resuming_active_recovery else _filter_eligible_vae_refs(refs_sorted, skip_layer_keys)
            )
            if target_layers != "all":
                allowed_layers = {int(idx) for idx in target_layers}
                eligible_vae_pairs = [
                    (layer_idx, ref)
                    for layer_idx, ref in eligible_vae_pairs
                    if int(layer_idx) in allowed_layers
                ]
            refs = [ref for _, ref in eligible_vae_pairs]
            cat_codebook_bits, cat_codebook_dim = category_codebook[cat]
            log.info(
                "=== Category: %s (%d eligible linears, %d discovered linears, residual_stages=%d, codebook_bits=%d, codebook_dim=%d, recon_loss=%s, sort=%s, steps=%d) ===",
                cat,
                len(refs),
                len(refs_sorted),
                int(cat_cfg.residual_stages),
                int(cat_codebook_bits),
                int(cat_codebook_dim),
                str(cat_cfg.recon_loss_type),
                category_sort_mode_desc[cat],
                int(cat_cfg.steps),
                # 联合优化代码，已关闭。旧日志字段保留如下：
                # float(cat_cfg.joint_decoder_lr),
                # int(cat_cfg.joint_decoder_group_size),
                # "none" if cat_cfg.joint_decoder_batch_size is None else str(int(cat_cfg.joint_decoder_batch_size)),
            )
            if not eligible_vae_pairs and not resuming_active_recovery:
                log.info("[%s] no eligible VAE refs after skip filtering; skipping VAE groups.", cat)

            category_protect_axis = channel_axis
            if (
                is_mlp_aligned_rank_metric(resolved_channel_mlp_rank_metric)
                and cat in MLP_CATEGORIES
            ):
                category_protect_axis = mlp_protect_axis_for_category(cat)

            channel_scope = str(cat_args.channel_scope).strip().lower()
            intra_parallel = tuple(getattr(cat_cfg, "intra_parallel", (1, 1)))
            adaptive_budget = 0
            if (
                not resuming_active_recovery
                and resolved_channel_mode == "channel"
                and channel_scope == "category"
            ):
                category_specs = [
                    _channel_spec_from_ref(
                        ref,
                        codebook_dim=int(cat_cfg.codebook_dim),
                        intra_parallel=intra_parallel,
                        ref_position=int(pos),
                        axis=str(category_protect_axis),
                        category=str(cat),
                    )
                    for pos, (_layer_idx, ref) in enumerate(eligible_vae_pairs)
                ]
                adaptive_budget = category_raw_budget(category_specs, int(cat_cfg.channel_protect_count))
            elif (
                not resuming_active_recovery
                and resolved_channel_mode == "channel"
                and channel_scope == "global"
            ):
                adaptive_budget = int(global_raw_budget_value)
            validate_adaptive_channel_tail_policy(
                scope=channel_scope,
                budget=int(adaptive_budget),
                allow_tail_group=bool(cat_args.allow_tail_group),
            )

            pair_by_name = {ref.name: (int(layer_idx), ref) for layer_idx, ref in eligible_vae_pairs}
            channel_plan: Optional[Dict[str, torch.Tensor]] = None
            group_seed_offsets: List[int]
            if (
                not resuming_active_recovery
                and resolved_channel_mode == "channel"
                and channel_scope in {"category", "global"}
                and int(adaptive_budget) > 0
            ):
                if channel_scope == "global":
                    if global_adaptive_plan is None:
                        raise RuntimeError("global channel plan was not built before category grouping.")
                    name_groups = list(global_adaptive_plan.groups_by_category.get(str(cat), []))
                    group_seed_offsets = list(
                        global_adaptive_plan.group_seed_offsets_by_category.get(str(cat), [])
                    )
                    channel_plan = _outlier_plan_from_adaptive(
                        global_adaptive_plan,
                        names=set(pair_by_name),
                    )
                    used_channels = sum(
                        int(global_adaptive_plan.counts.get(name, 0)) for name in pair_by_name
                    )
                else:
                    refs_for_plan = [ref for _layer_idx, ref in eligible_vae_pairs]
                    refs_by_name = {ref.name: ref for ref in refs_for_plan}

                    def _category_activation_views(_specs: Sequence[ChannelLinearSpec]):
                        return _activation_views_for_refs(
                            refs_for_plan,
                            activation_runtime,
                            category=str(cat),
                            rank_metric=resolved_channel_rank_metric,
                        )

                    def _category_score_specs(specs, act_weight, act_mean):
                        return _score_channel_specs(
                            specs,
                            refs_by_name,
                            metric=resolved_channel_rank_metric,
                            axis=str(category_protect_axis),
                            activation_weight_by_linear=act_weight,
                            activation_abs_mean_by_linear=act_mean,
                        )

                    category_plan = resolve_adaptive_channel_plan(
                        category_specs,
                        raw_budget=int(adaptive_budget),
                        min_per_layer=int(cat_args.channel_min_per_layer),
                        linear_group_size=int(linear_group_size),
                        metric=str(resolved_channel_rank_metric),
                        axis=str(category_protect_axis),
                        scope="category",
                        category=str(cat),
                        is_main=bool(plan_is_main),
                        world_size=int(plan_world_size),
                        broadcast_fn=broadcast_adaptive_channel_plan,
                        activation_view_fn=_category_activation_views,
                        score_fn=_category_score_specs,
                        run_output_dir=str(run_output_dir) if plan_is_main else None,
                    )
                    name_groups = list(category_plan.groups)
                    group_seed_offsets = list(category_plan.group_seed_offsets)
                    channel_plan = _outlier_plan_from_adaptive(category_plan)
                    used_channels = int(category_plan.used_channels)
                vae_groups = _pairs_from_name_groups(name_groups, pair_by_name)
                if len(group_seed_offsets) != len(vae_groups):
                    raise RuntimeError(
                        f"[{cat}] adaptive group seed offsets ({len(group_seed_offsets)}) "
                        f"do not match vae_groups ({len(vae_groups)})."
                    )
                if plan_is_main:
                    log.info(
                        "[%s] adaptive channel plan: scope=%s eligible_linears=%d used_channels=%d groups=%d",
                        cat,
                        channel_scope,
                        len(pair_by_name),
                        int(used_channels),
                        len(vae_groups),
                    )
            else:
                eligible_names = [ref.name for _layer_idx, ref in eligible_vae_pairs]
                name_groups = group_layer_scope_inventory(
                    eligible_names,
                    linear_group_size=int(linear_group_size),
                    allow_tail_group=bool(cat_args.allow_tail_group),
                )
                group_seed_offsets = layer_scope_group_seed_offsets(
                    eligible_names,
                    linear_group_size=int(linear_group_size),
                    allow_tail_group=bool(cat_args.allow_tail_group),
                )
                vae_groups = _pairs_from_name_groups(name_groups, pair_by_name)
                planned_refs = [ref for group in vae_groups for _layer_idx, ref in group]
                if (
                    resolved_channel_mode == "channel"
                    and int(cat_cfg.channel_protect_count) > 0
                    and (not inline_distributed or is_distill_main_process())
                ):
                    if (
                        is_mlp_aligned_rank_metric(resolved_channel_mlp_rank_metric)
                        and cat in MLP_CATEGORIES
                        and mlp_channel_plan_by_linear is not None
                    ):
                        channel_plan = {}
                        for ref in planned_refs:
                            if ref.name not in mlp_channel_plan_by_linear:
                                raise KeyError(f"[{cat}] missing MLP aligned channel plan for eligible linear {ref.name}.")
                            channel_plan[ref.name] = mlp_channel_plan_by_linear[ref.name].detach().to(
                                device="cpu",
                                dtype=torch.long,
                            ).contiguous()
                        log.info(
                            "[%s] MLP aligned channel plan: eligible_linears=%d total_channels=%d axis=%s",
                            cat,
                            len(planned_refs),
                            sum(int(v.numel()) for v in channel_plan.values()),
                            category_protect_axis,
                        )
                    else:
                        plan_activation_weight, plan_activation_abs_mean = _activation_views_for_refs(
                            planned_refs,
                            activation_runtime,
                            category=str(cat),
                            rank_metric=resolved_channel_rank_metric,
                        )
                        plan_refs = [
                            LinearPrepRef(
                                name=r.name,
                                weight=r.module.weight,
                                in_features=int(r.module.in_features),
                                out_features=int(r.module.out_features),
                                transpose=bool(r.transpose),
                            )
                            for r in planned_refs
                        ]
                        channel_plan = build_outlier_channel_index_plan(
                            group_refs=plan_refs,
                            activation_weight_by_linear=plan_activation_weight,
                            activation_abs_mean_by_linear=plan_activation_abs_mean,
                            channel_protect_count=int(cat_cfg.channel_protect_count),
                            channel_axis=category_protect_axis,
                            channel_scope="layer",
                            channel_rank_metric=resolved_channel_rank_metric,
                            channel_min_per_layer=int(cat_args.channel_min_per_layer),
                        )
                        log.info(
                            "[%s] channel plan: mode=%s scope=layer eligible_linears=%d total_channels=%d",
                            cat,
                            resolved_channel_mode,
                            len(plan_refs),
                            sum(int(v.numel()) for v in channel_plan.values()),
                        )

            newly_compressed_target_count = int(resume_newly_compressed_target_count)
            planned_eligible_pairs = [pair for group in vae_groups for pair in group]
            if not bool(cat_args.allow_tail_group) and len(eligible_vae_pairs) != len(planned_eligible_pairs):
                log.info(
                    "[%s] tail eligible group size=%d skipped (set --allow_tail_group to include).",
                    cat,
                    len(eligible_vae_pairs) - len(planned_eligible_pairs),
                )
            for group_idx, group_pairs in enumerate(vae_groups):
                group_refs = [ref for _, ref in group_pairs]
                layer_indices = [idx for idx, _ in group_pairs]
                group_tag = f"{cat}.L{layer_indices[0]}-{layer_indices[-1]}"
                log.info(
                    "---- Group: %s (linears=%d, num_models=%d) ----",
                    group_tag,
                    len(group_refs),
                    len(group_refs),
                )
                newly_compressed_target_count += _train_group_vae_and_replace_inline_distributed(
                    model=model,
                    group_refs=group_refs,
                    group_tag=group_tag,
                    inline_distributed=inline_distributed,
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
                    activation_runtime=activation_runtime,
                    channel_protect_mode=resolved_channel_mode,
                    channel_plan=channel_plan,
                    channel_scope=str(cat_args.channel_scope),
                    channel_rank_metric=resolved_channel_rank_metric,
                    channel_axis=category_protect_axis,
                    channel_quant=channel_quant,
                    channel_min_per_layer=int(cat_args.channel_min_per_layer),
                    deterministic=bool(cat_args.deterministic),
                    shuffle_seed=vae_group_shuffle_seed(
                        int(cat_args.seed),
                        int(cat_idx),
                        int(group_seed_offsets[group_idx]),
                    ),
                )
            if run_category_eval and after_category_mode == "none":
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
                    run_output_dir=run_output_dir,
                )

            distill_result = None
            run_this_category_eval = False
            category_steps = (
                int(cat_args.resolve_after_category_config(cat).opt.steps)
                if after_category_mode != "none"
                else 0
            )
            v6_step_checkpoint = None
            save_strategy = str(getattr(training_args, "save_strategy", "steps") or "steps").strip().lower()
            needs_v6_step_runtime = bool(
                after_category_mode != "none"
                and category_steps > 0
                and (resuming_active_recovery or save_strategy != "no")
            )
            if needs_v6_step_runtime:
                from train_utils.cat_step_resume_v6 import resolve_cat_round_root

                if resuming_active_recovery:
                    resume_source = getattr(cat_args, "_v6_resume_source", None)
                    if resume_source is None:
                        raise RuntimeError("CAT step resume lost its resolved resume source.")
                    round_base_dir = os.path.abspath(str(resume_source.model_checkpoint_dir))
                    round_base_meta = dict(resume_source.model_checkpoint_meta)
                    trainer_output_dir = os.path.dirname(os.path.abspath(str(resume_step_checkpoint)))
                else:
                    round_root = resolve_cat_round_root(
                        run_output_dir,
                        category=str(cat),
                        round_idx=int(lora_round_idx),
                    )
                    round_base_dir = os.path.join(round_root, "round_base")
                    trainer_output_dir = os.path.join(round_root, "trainer_state")
                    is_round_main = (not inline_distributed) or is_distill_main_process()
                    round_tokenizer = None
                    if is_round_main:
                        round_tokenizer = _load_checkpoint_tokenizer(
                            vae_args.model_path,
                            hf_args.access_token,
                        )
                    round_save = save_cat_v6_full_checkpoint(
                        model,
                        round_base_dir,
                        checkpoint_kind="round_base",
                        category=str(cat),
                        completed_categories=completed_categories,
                        compression_categories=compression_category_values,
                        cat_args=cat_args,
                        vae_args=vae_args,
                        training_args=training_args,
                        tokenizer=round_tokenizer,
                        base_model_path=vae_args.model_path,
                        distill_stage_meta=None,
                        distill_stage_history=distill_stage_history,
                        round_idx=lora_round_idx,
                        cat_runtime_state=(
                            build_cat_runtime_state(
                                activation_runtime=activation_runtime,
                                global_adaptive_plan=global_adaptive_plan,
                                runtime_identity=cat_runtime_identity,
                            )
                            if is_round_main
                            else None
                        ),
                        is_main_process=is_round_main,
                        distributed_barrier=(distill_distributed_barrier if inline_distributed else None),
                    )
                    round_base_meta = dict(round_save["meta_payload"])
                    log.info(
                        "[%s] saved CAT v6 round_base checkpoint_id=%s path=%s",
                        str(cat),
                        str(round_base_meta["checkpoint_id"]),
                        str(round_base_dir),
                    )
                v6_step_checkpoint = {
                    "round_base_dir": str(round_base_dir),
                    "round_base_checkpoint_id": str(round_base_meta["checkpoint_id"]),
                    "round_base_meta": round_base_meta,
                    "active_category": str(cat),
                    "trainer_output_dir": str(trainer_output_dir),
                    "resume_from_checkpoint": (
                        str(resume_step_checkpoint) if resuming_active_recovery else None
                    ),
                    "base_model_path": str(vae_args.model_path),
                    "distill_stage_history": [dict(item) for item in distill_stage_history],
                    "round_idx": int(lora_round_idx),
                }

            if after_category_mode != "none":
                run_this_category_eval = bool(run_category_eval) and category_steps > 0
                if run_category_eval and not run_this_category_eval:
                    log.info(
                        "类别 %s distill_steps=%d，跳过该类别评估。",
                        str(cat),
                        category_steps,
                    )
                if run_this_category_eval and not resuming_active_recovery:
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
                        run_output_dir=run_output_dir,
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
                    compression_categories=compression_categories,
                    teacher_runtime=teacher_runtime,
                    newly_compressed_target_count=newly_compressed_target_count,
                    current_category_target_names=current_category_target_names,
                    v6_step_checkpoint=v6_step_checkpoint,
                    online_cat=True,
                )
                model = distill_result.model
                lora_round_idx = int(distill_result.next_lora_round_idx)
                if distill_result.distill_meta is not None:
                    distill_stage_history.append(dict(distill_result.distill_meta))
                if inline_distributed:
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    log.info("Cat inline distill complete: model moved to CPU on rank=%d", distill_rank())
                    distill_distributed_barrier()
                if run_this_category_eval:
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
                        run_output_dir=run_output_dir,
                    )

            if str(cat) in completed_category_set:
                raise RuntimeError(f"CAT category {cat!r} was completed twice in one run.")
            completed_categories.append(str(cat))
            completed_category_set.add(str(cat))
            if resuming_active_recovery:
                resume_step_checkpoint = None
                resume_active_category = None

            if bool(cat_args.save_model):
                if not bool(cat_args.convert):
                    raise ValueError("--save_model requires --convert")
                from transformers import AutoTokenizer
                from e2e_common.full_lora import iter_named_full_compressed_peft_proxies
                from litebsq.vae_linear import clear_model_vae_linear_cache

                if inline_distributed:
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    distill_distributed_barrier()
                leftover_proxies = [name for name, _proxy in iter_named_full_compressed_peft_proxies(model)]
                if leftover_proxies:
                    raise RuntimeError(
                        "CAT category-boundary save found unexported PEFT proxy modules: "
                        + ", ".join(leftover_proxies)
                    )
                cleared = clear_model_vae_linear_cache(model)
                def _save_category_boundary():
                    after_dir = os.path.join(run_output_dir, f"after_{cat}")
                    tok = AutoTokenizer.from_pretrained(
                        vae_args.model_path, use_fast=True, token=hf_args.access_token
                    )
                    return save_cat_v6_full_checkpoint(
                        model,
                        after_dir,
                        checkpoint_kind="category_boundary",
                        category=str(cat),
                        completed_categories=completed_categories,
                        compression_categories=compression_category_values,
                        cat_args=cat_args,
                        vae_args=vae_args,
                        training_args=training_args,
                        tokenizer=tok,
                        base_model_path=vae_args.model_path,
                        distill_stage_meta=(None if distill_result is None else distill_result.distill_meta),
                        distill_stage_history=distill_stage_history,
                        round_idx=lora_round_idx,
                        cat_runtime_state=build_cat_runtime_state(
                            activation_runtime=activation_runtime,
                            global_adaptive_plan=global_adaptive_plan,
                            runtime_identity=cat_runtime_identity,
                        ),
                        is_main_process=True,
                        distributed_barrier=None,
                    )
                save_paths = (
                    distributed_guarded_main(_save_category_boundary, barrier=True)
                    if inline_distributed
                    else _save_category_boundary()
                )
                if not inline_distributed or is_distill_main_process():
                    log.info(
                        "Category-boundary save [%s]: cleared decoded cache for %d VAELinear modules.",
                        cat,
                        cleared,
                    )
                    log.info("Saved v6 category-boundary model to %s", save_paths["output_dir"])

                def _prune_completed_rounds():
                    return prune_completed_cat_round_roots(
                        run_output_dir,
                        save_total_limit=getattr(training_args, "save_total_limit", None),
                    )
                removed_round_roots = (
                    distributed_guarded_main(_prune_completed_rounds, barrier=True)
                    if inline_distributed
                    else _prune_completed_rounds()
                )
                if not inline_distributed or is_distill_main_process():
                    if removed_round_roots:
                        log.info(
                            "CAT round retention removed %d completed round root(s): %s",
                            len(removed_round_roots),
                            list(removed_round_roots),
                        )

            if inline_distributed:
                model.to("cpu")
                torch.cuda.empty_cache()
                distill_distributed_barrier()

        from e2e_common.post_norm_head import fuse_post_norm_head_linear

        fused_post_norm_head = fuse_post_norm_head_linear(model)
        if fused_post_norm_head and (not inline_distributed or is_distill_main_process()):
            log.info("Finalized post_norm_linear into lm_head.weight before final evaluation/save.")
        if inline_distributed:
            distill_distributed_barrier()

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
                run_output_dir=run_output_dir,
            )
        if cat_args.save_candidate_artifact:
            if not cat_args.convert:
                raise ValueError("--save_candidate_artifact requires --convert")
            if not inline_distributed or is_distill_main_process():
                from mix_bit.candidate_artifact import save_candidate_artifact_from_model

                save_paths = save_candidate_artifact_from_model(
                    model=model,
                    trial_spec_path=str(cat_args.candidate_artifact_spec),
                    output_dir=str(cat_args.candidate_artifact_output_dir),
                    source_run_dir=str(run_output_dir),
                )
                log.info("Saved candidate artifact to %s", save_paths["output_dir"])
            if inline_distributed:
                distill_distributed_barrier()
        elif cat_args.save_model:
            if not cat_args.convert:
                raise ValueError("--save_model requires --convert")
            if tuple(completed_categories) != tuple(str(v) for v in active_categories):
                raise RuntimeError(
                    "Final CAT save requires every active category to be completed: "
                    f"completed={completed_categories}, active={active_categories}."
                )
            from transformers import AutoTokenizer
            from e2e_common.full_lora import iter_named_full_compressed_peft_proxies
            from litebsq.vae_linear import clear_model_vae_linear_cache

            leftover_proxies = [name for name, _proxy in iter_named_full_compressed_peft_proxies(model)]
            if leftover_proxies:
                raise RuntimeError(
                    "Final save found unexported PEFT proxy modules: "
                    + ", ".join(leftover_proxies)
                )
            cleared = clear_model_vae_linear_cache(model)
            if not inline_distributed or is_distill_main_process():
                log.info("Final save: cleared decoded cache for %d VAELinear modules.", cleared)
            def _save_final_model():
                model_out = os.path.join(run_output_dir, "final_model")
                tok = AutoTokenizer.from_pretrained(
                    vae_args.model_path, use_fast=True, token=hf_args.access_token
                )
                return save_cat_v6_full_checkpoint(
                    model,
                    model_out,
                    checkpoint_kind="final_model",
                    category=None,
                    completed_categories=completed_categories,
                    compression_categories=compression_category_values,
                    cat_args=cat_args,
                    vae_args=vae_args,
                    training_args=training_args,
                    tokenizer=tok,
                    base_model_path=vae_args.model_path,
                    distill_stage_meta=None,
                    distill_stage_history=distill_stage_history,
                    round_idx=lora_round_idx,
                    cat_runtime_state=build_cat_runtime_state(
                        activation_runtime=activation_runtime,
                        global_adaptive_plan=global_adaptive_plan,
                        runtime_identity=cat_runtime_identity,
                    ),
                    is_main_process=True,
                    distributed_barrier=None,
                )
            save_paths = (
                distributed_guarded_main(_save_final_model, barrier=True)
                if inline_distributed
                else _save_final_model()
            )
            if not inline_distributed or is_distill_main_process():
                log.info("Saved v6 final model to %s", save_paths["output_dir"])
    finally:
        # 排序代码，已关闭：sort_executor 始终为 None。
        pass

    log.info("Done.")
