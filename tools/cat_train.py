import os
import sys
import time
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
from train_utils.cat_train_runtime import (
    init_sort_prep_worker as _init_sort_prep_worker,
    load_model_for_cat_train as _load_model_for_cat_train,
    resolve_sort_prep_workers as _resolve_sort_prep_workers,
    save_normalized_cat_train_snapshot as _save_normalized_cat_train_snapshot,
)
from train_utils.cat_train_residual_protection import (
    LOW_RANK_OUTLIER_MODES as _LOW_RANK_OUTLIER_MODES,
    RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT as _RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT,
    build_per_vae_low_rank_payloads as _build_per_vae_low_rank_payloads,
    build_post_vae_low_rank_payload as _build_post_vae_low_rank_payload,
    build_sparse_residual_payload as _build_sparse_residual_payload,
)
from litebsq.vae_args import apply_autoencoder_arch_defaults
from litebsq.misc import set_module_by_name
from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
)
from train_utils.cat_data_prep import (
    LinearPrepRef,
    format_intra_part_sort_mode,
    gather_wa_mse_act_max_batch,
    materialize_prepared_group_data,
    prepare_group_weight_data,
    prepare_group_linear_entries,
)
from train_utils.activation_utils import (
    ActivationCalibrationCache,
    collect_act_max_for_linears,
)
from train_utils.cat_arg_overrides import validate_category_keys
from train_utils.cat_train_data import (
    apply_stage_norm as _apply_stage_norm,
    build_block_data_loaders as _build_block_data_loaders,
    compute_stage_norm_stats as _compute_stage_norm_stats,
    reshape_blocks_for_codebook_dim as _reshape_blocks_for_codebook_dim,
    restore_stage_norm as _restore_stage_norm,
)
from train_utils.cat_joint_decoder import (
    finetune_stage_decoders_in_subgroups as _finetune_stage_decoders_in_subgroups,
)
from train_utils.cat_train_eval import eval_after_category as _eval_after_category
from train_utils.model_checkpoint_io import (
    _build_run_output_dir,
    save_model_checkpoint,
)
from train_utils.utils import (
    LinearRef,
    clone_namespace as _clone_namespace,
    collect_linears as _collect_linears,
    configure_deterministic_mode,
    extract_layer_idx as _extract_layer_idx,
    format_intra_parallel_desc as _format_intra_parallel_desc,
    format_namespace as _format_namespace,
    get_logger,
    resolve_category_order as _resolve_category_order,
    set_seed,
    split_csv as _split_csv,
)


log = get_logger("linear_by_category")


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
    restore_idx = getattr(split_meta, "restore_row_indices", None)
    if restore_idx is None:
        return w_split
    if int(restore_idx.numel()) != int(w_split.shape[0]):
        raise ValueError(
            f"{split_meta.linear_name}: restore_row_indices size {int(restore_idx.numel())} != split rows {int(w_split.shape[0])}"
        )
    if restore_idx.device != w_split.device:
        restore_idx = restore_idx.to(device=w_split.device, non_blocking=True)
    return w_split.index_select(0, restore_idx)


def _restore_split_col_order_with_meta(w_split: torch.Tensor, split_meta) -> torch.Tensor:
    restore_idx = getattr(split_meta, "restore_col_indices", None)
    if restore_idx is None:
        return w_split
    if int(restore_idx.numel()) != int(w_split.shape[1]):
        raise ValueError(
            f"{split_meta.linear_name}: restore_col_indices size {int(restore_idx.numel())} != split cols {int(w_split.shape[1])}"
        )
    if restore_idx.device != w_split.device:
        restore_idx = restore_idx.to(device=w_split.device, non_blocking=True)
    return w_split.index_select(1, restore_idx)


def _restore_part_col_order_with_meta(part_matrix: torch.Tensor, split_meta, part_idx: int) -> torch.Tensor:
    restore_all = getattr(split_meta, "part_restore_col_indices", None)
    if restore_all is None:
        return part_matrix
    if restore_all.ndim != 2:
        raise ValueError(
            f"{split_meta.linear_name}: part_restore_col_indices must be 2D, got shape={tuple(restore_all.shape)}"
        )
    if part_idx < 0 or part_idx >= int(restore_all.shape[0]):
        raise IndexError(
            f"{split_meta.linear_name}: part_idx out of range for part_restore_col_indices: {part_idx} vs {int(restore_all.shape[0])}"
        )
    restore_idx = restore_all[part_idx]
    if int(restore_idx.numel()) != int(part_matrix.shape[1]):
        raise ValueError(
            f"{split_meta.linear_name}: part_restore_col_indices[{part_idx}] size {int(restore_idx.numel())} != part cols {int(part_matrix.shape[1])}"
        )
    if restore_idx.device != part_matrix.device:
        restore_idx = restore_idx.to(device=part_matrix.device, non_blocking=True)
    return part_matrix.index_select(1, restore_idx)


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
    projection_suffixes: Sequence[str],
) -> List[LinearRef]:
    return _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )


def _collect_sorted_category_refs(
    model: nn.Module,
    *,
    category: str,
    transpose_modules: Sequence[str],
    only_decoder_projections: bool,
    projection_suffixes: Sequence[str],
) -> Tuple[List[Tuple[int, LinearRef]], int]:
    refs_sorted: List[Tuple[int, LinearRef]] = []
    missing = 0
    for ref in _collect_current_trainable_linears(
        model,
        transpose_modules=transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
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
    low_rank_a: Optional[torch.Tensor] = None,
    low_rank_b: Optional[torch.Tensor] = None,
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
        restore_row_indices=split_meta.restore_row_indices,
        restore_col_indices=split_meta.restore_col_indices,
        part_restore_col_indices=split_meta.part_restore_col_indices,
        stage_restore_row_indices=[
            None if getattr(meta, "restore_row_indices", None) is None else meta.restore_row_indices
            for meta in stage_split_metas
        ],
        stage_restore_col_indices=[
            None if getattr(meta, "restore_col_indices", None) is None else meta.restore_col_indices
            for meta in stage_split_metas
        ],
        stage_part_restore_col_indices=[
            None if getattr(meta, "part_restore_col_indices", None) is None else meta.part_restore_col_indices
            for meta in stage_split_metas
        ],
        compressed_in_features=int(split_meta.compressed_in_features),
        compressed_out_features=int(split_meta.compressed_out_features),
        protected_input_indices=split_meta.protected_input_indices,
        protected_input_weight=split_meta.protected_input_weight,
        protected_output_indices=split_meta.protected_output_indices,
        protected_output_weight=split_meta.protected_output_weight,
        low_rank_a=low_rank_a,
        low_rank_b=low_rank_b,
        always_use_original=bool(always_use_original),
        protect_original_weight=bool(protect_original_weight),
    )
    if sparse_residual_kwargs:
        common_kwargs.update(dict(sparse_residual_kwargs))
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
        low_rank_a=None,
        low_rank_b=None,
    )
    return temp_linear._decode_weight(dtype=torch.float32).detach().to(device="cpu", dtype=torch.float32)


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
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
    activation_runtime: Optional[Dict[str, object]] = None,
    outlier_protect_mode: str = "channel",
    outlier_residual_score: str = "abs",
    outlier_residual_min_abs: float = 1e-6,
    outlier_protect_axis: str = "input",
    outlier_residual_codec: str = SPARSE_RESIDUAL_FORMAT_COO_FP16,
    outlier_residual_index_bits: int = 8,
    outlier_residual_value_bits: int = 8,
    outlier_residual_block_shape: Tuple[int, int] = (256, 256),
    sort_executor=None,
    sort_prep_workers_resolved: int = 1,
    deterministic: bool = False,
    shuffle_seed: int = 0,
) -> None:
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
    stage_decoder_type = str(runtime_cfg.decoder_type).strip().lower()
    outlier_protect_count = int(runtime_cfg.outlier_protect_count)
    outlier_low_rank = int(runtime_cfg.outlier_low_rank)
    outlier_residual_top_p = float(runtime_cfg.outlier_residual_top_p)
    resolved_outlier_mode = str(outlier_protect_mode).strip().lower()
    resolved_residual_score = str(outlier_residual_score).strip().lower()
    resolved_residual_min_abs = float(outlier_residual_min_abs)
    residual_sparse_enabled = resolved_outlier_mode == "residual_sparse"
    low_rank_enabled = resolved_outlier_mode in _LOW_RANK_OUTLIER_MODES
    residual_sparse_needs_activation = (
        residual_sparse_enabled and resolved_residual_score in _RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT
    )
    if resolved_outlier_mode not in {"none", "channel", "residual_sparse", "per_vae_low_rank", "post_vae_low_rank"}:
        raise ValueError(
            f"[{group_tag}] unsupported outlier_protect_mode={outlier_protect_mode!r}. "
            "Expected none, channel, residual_sparse, per_vae_low_rank, or post_vae_low_rank."
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
    if low_rank_enabled:
        if int(outlier_protect_count) != 0:
            raise ValueError(
                f"[{group_tag}] {resolved_outlier_mode} mode requires outlier_protect_count=0, got {outlier_protect_count}."
            )
        if float(outlier_residual_top_p) != 0.0:
            raise ValueError(
                f"[{group_tag}] {resolved_outlier_mode} mode requires outlier_residual_top_p=0, "
                f"got {outlier_residual_top_p}."
            )
        if int(outlier_low_rank) <= 0:
            raise ValueError(
                f"[{group_tag}] {resolved_outlier_mode} mode requires outlier_low_rank > 0, got {outlier_low_rank}."
            )
    resolved_residual_codec = str(outlier_residual_codec).strip().lower()
    use_wa_mse_loss = stage_recon_loss == "wa_mse"
    row_parts, col_parts = int(runtime_cfg.intra_parallel[0]), int(runtime_cfg.intra_parallel[1])
    parts_per_linear = int(row_parts) * int(col_parts)
    sort_mode = str(runtime_cfg.intra_part_sort_mode).strip().lower()
    needs_dynamic_activation = (
        use_wa_mse_loss
        or sort_mode == "act_spectral_cosine"
        or (resolved_outlier_mode == "channel" and int(outlier_protect_count) > 0)
        or residual_sparse_needs_activation
    )
    effective_activation_weight: Optional[Dict[str, torch.Tensor]] = None
    if needs_dynamic_activation:
        if activation_runtime is None:
            raise ValueError(
                f"[{group_tag}] dynamic activation runtime is required for wa_mse, act_spectral_cosine, or outlier protection."
            )
        calib_device = str(activation_runtime.get("device") or train_device)
        linear_items = [(r.name, r.module) for r in group_refs]
        dynamic_act_max, new_cache = collect_act_max_for_linears(
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
        effective_activation_weight = dynamic_act_max
        log.info(
            "[%s] refreshed dynamic activation stats (linears=%d, dataset=%s, nsamples=%d, seqlen=%d).",
            group_tag,
            len(dynamic_act_max),
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
        outlier_protect_count=int(outlier_protect_count) if resolved_outlier_mode == "channel" else 0,
        outlier_protect_axis=str(outlier_protect_axis),
        recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
        intra_part_sort_mode=stage_sort_mode,
    )
    low_rank_payloads: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None for _ in prepared_entries]
    initial_split_weights_by_linear: Optional[List[torch.Tensor]] = None
    if resolved_outlier_mode == "per_vae_low_rank":
        low_rank_payloads, initial_split_weights_by_linear = _build_per_vae_low_rank_payloads(
            prepared_entries=prepared_entries,
            rank=int(outlier_low_rank),
        )
        log.info(
            "[%s] per-vae low-rank protection enabled: rank=%d linears=%d",
            group_tag,
            int(outlier_low_rank),
            len(prepared_entries),
        )
    target_common_result = materialize_prepared_group_data(
        prepared_entries=prepared_entries,
        intra_parallel=(row_parts, col_parts),
        codebook_dim=int(stage_codebook_dim),
        batch_size=int(materialize_batch_size),
        normalize_weight=False,
        recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
        train_device=train_device,
        intra_part_sort_mode="none",
        sort_executor=None,
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
    if resolved_outlier_mode == "channel" and int(outlier_protect_count) > 0:
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
            "[%s] outlier protection axis=%s count=%d protected_channels=%s",
            group_tag,
            str(outlier_protect_axis),
            int(outlier_protect_count),
            ",".join(per_linear_protected),
        )
    if residual_sparse_enabled:
        log.info(
            "[%s] residual sparse protection enabled: top_p=%.6f score=%s min_abs=%.6e codec=%s index_bits=%d value_bits=%d block=%dx%d",
            group_tag,
            outlier_residual_top_p,
            resolved_residual_score,
            resolved_residual_min_abs,
            resolved_residual_codec,
            int(outlier_residual_index_bits),
            int(outlier_residual_value_bits),
            int(outlier_residual_block_shape[0]),
            int(outlier_residual_block_shape[1]),
        )
    if resolved_outlier_mode == "post_vae_low_rank":
        log.info(
            "[%s] post-vae low-rank protection enabled: rank=%d",
            group_tag,
            int(outlier_low_rank),
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
            sort_executor=None,
            split_weights_by_linear=current_residual_weights,
            shuffle_seed=int(shuffle_seed) + int(stage_idx) if bool(deterministic) else None,
        )
        stage_prep_result = common_stage_result
        if sort_mode != "none":
            prep_start_time = time.time()
            stage_prep_result = materialize_prepared_group_data(
                prepared_entries=prepared_entries,
                intra_parallel=(row_parts, col_parts),
                codebook_dim=int(stage_codebook_dim),
                batch_size=int(materialize_batch_size),
                normalize_weight=False,
                recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
                train_device=train_device,
                intra_part_sort_mode=stage_sort_mode,
                sort_executor=sort_executor,
                split_weights_by_linear=current_residual_weights,
                shuffle_seed=int(shuffle_seed) + int(stage_idx) if bool(deterministic) else None,
            )
            prep_duration_sec = float(time.time() - prep_start_time)
            sort_task_count = int(len(group_refs))
            effective_sort_workers = 1
            sort_backend = "cpu_serial"
            if sort_executor is not None and sort_task_count > 1 and int(sort_prep_workers_resolved) > 1:
                effective_sort_workers = min(int(sort_prep_workers_resolved), sort_task_count)
                sort_backend = "cpu_process"
            log.info(
                "[%s] 排序预处理完成: sort_backend=%s sort_prep_workers_resolved=%d sort_task_count=%d duration_sec=%.2f",
                stage_tag,
                sort_backend,
                effective_sort_workers,
                sort_task_count,
                prep_duration_sec,
            )
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

        effective_batch_size = int(stage_train_data.shape[0]) if batch_size_is_all else int(materialize_batch_size)
        eval_batch_size = int(stage_train_data.shape[0]) if batch_size_is_all else int(materialize_batch_size)
        all_batch_gpu_cache = bool(batch_size_is_all and stage_recon_loss != "wa_mse")
        if int(effective_batch_size) < 1:
            raise RuntimeError(f"[{stage_tag}] effective VAE batch size must be >= 1, got {effective_batch_size}.")
        if batch_size_is_all:
            log.info("[%s] VAE batch_size=all(effective=%d)", stage_tag, int(effective_batch_size))
        if all_batch_gpu_cache:
            log.info("[%s] VAE all-batch GPU cache enabled.", stage_tag)
            train_loader = None
        else:
            train_loader, _unused_eval_loader = _build_block_data_loaders(
                stage_train_data,
                batch_size=int(effective_batch_size),
                shuffle_seed=int(shuffle_seed) + int(stage_idx) if bool(deterministic) else None,
            )
            del _unused_eval_loader
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
        lr_scheduler_name = str(getattr(stage_vae_args, "lr_scheduler", "none"))
        if lr_scheduler_name != "none":
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
        if all_batch_gpu_cache:
            x_all = stage_train_data.to(device=train_device, dtype=train_dtype, non_blocking=True)
            train_iter = None
        else:
            x_all = None
            train_iter = iter(train_loader)
        for step in range(int(stage_steps)):
            act_max_batch = None
            if all_batch_gpu_cache:
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
                    mse_acc = []
                    top_k_mse_acc = []
                    total = 0
                    for x_eval_batch, _eval_idx_batch in eval_loader:
                        if total >= int(eval_blocks):
                            break
                        x_eval_batch = x_eval_batch[: max(0, int(eval_blocks) - total)]
                        total += x_eval_batch.shape[0]
                        x_eval = x_eval_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
                        x_recon, _ = vae(x_eval, is_train=False)
                        x_eval_f = x_eval.float()
                        x_recon_f = x_recon.float()
                        mse_acc.append(torch.nn.functional.mse_loss(x_recon_f, x_eval_f))

                        # 对每个并行模型（P 维）独立选 top-k：
                        # x_eval/x_recon: [B, P, C] -> [P, B*C]
                        flat_eval = x_eval_f.permute(1, 0, 2).reshape(x_eval_f.shape[1], -1)
                        flat_recon = x_recon_f.permute(1, 0, 2).reshape(x_recon_f.shape[1], -1)
                        k = min(100, flat_eval.shape[1])
                        _, topk_idx = torch.topk(flat_eval.abs(), k=k, dim=1)
                        top_eval = torch.gather(flat_eval, dim=1, index=topk_idx)
                        top_recon = torch.gather(flat_recon, dim=1, index=topk_idx)
                        top_k_mse_acc.append(torch.nn.functional.mse_loss(top_recon, top_eval))
                    mse = torch.stack(mse_acc).mean() if mse_acc else torch.tensor(0.0)
                    top_k_mse = torch.stack(top_k_mse_acc).mean() if top_k_mse_acc else torch.tensor(0.0)
                log.info(
                    "[%s] eval@step=%d mse=%.6e top_k_mse(k=100)=%.6e",
                    stage_tag,
                    step + 1,
                    float(mse.detach().cpu().item()),
                    float(top_k_mse.detach().cpu().item()),
                )
                vae.train()

        # 3) 对当前 stage 的 residual 生成重构，更新下一阶段 residual。
        vae.eval()
        stage_recon_chunks: List[torch.Tensor] = []
        stage_bit_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for x_in_batch, _eval_idx_batch in eval_loader:
                x_in = x_in_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
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
        residual_rms_after = float(current_common_stacked.float().pow(2).mean().sqrt().item())
        log.info(
            "[%s] residual rms: before=%.6e after=%.6e",
            stage_tag,
            residual_rms_before,
            residual_rms_after,
        )

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
        if (
            len(all_stage_bits) != residual_stages
            or len(all_stage_decoders) != residual_stages
            or len(all_stage_split_metas) != residual_stages
            or len(all_stage_codebook_dims) != residual_stages
        ):
            raise RuntimeError(
                f"[{group_tag}] joint fine-tune payload mismatch: bits={len(all_stage_bits)} "
                f"decoders={len(all_stage_decoders)} split_metas={len(all_stage_split_metas)} "
                f"codebook_dims={len(all_stage_codebook_dims)} residual_stages={residual_stages}"
            )
        joint_steps = int(runtime_cfg.joint_decoder_steps)
        joint_lr = float(runtime_cfg.joint_decoder_lr)
        joint_group_size = max(1, min(int(runtime_cfg.joint_decoder_group_size), len(group_refs)))
        joint_batch_size = runtime_cfg.joint_decoder_batch_size
        if joint_steps > 0:
            log.info(
                "[%s/joint] start (mode=%s, steps=%d, lr=%.3e, recon_loss=%s, stages=%d, joint_group_size=%d, joint_decoder_batch_size=%s)",
                group_tag,
                "patch" if joint_batch_size is not None else "full",
                joint_steps,
                joint_lr,
                stage_recon_loss,
                residual_stages,
                joint_group_size,
                "none" if joint_batch_size is None else str(int(joint_batch_size)),
            )
            all_stage_decoders = _finetune_stage_decoders_in_subgroups(
                group_tag=group_tag,
                group_refs=group_refs,
                shared_stage_args=shared_stage_args,
                joint_steps=joint_steps,
                joint_lr=joint_lr,
                joint_group_size=joint_group_size,
                joint_decoder_batch_size=joint_batch_size,
                train_device=train_device,
                train_dtype=train_dtype,
                log_every=log_every,
                eval_every=eval_every,
                eval_blocks=eval_blocks,
                codebook_dim=int(stage_codebook_dim),
                recon_loss_type=stage_recon_loss,
                intra_part_sort_mode=stage_sort_mode,
                target_common_result=target_common_result,
                all_stage_bits=all_stage_bits,
                all_stage_decoders=all_stage_decoders,
                all_stage_split_metas=all_stage_split_metas,
                parts_per_linear=parts_per_linear,
                convert_stage_to_common_fn=_convert_stage_stacked_to_common_stacked,
                recon_loss_fn=_compute_recon_loss,
                logger=log,
                shuffle_seed=int(shuffle_seed) if bool(deterministic) else None,
            )

    if not do_convert:
        del current_residual_weights, target_common_result, all_stage_bits, all_stage_decoders, all_stage_codebook_dims, all_stage_split_metas
        torch.cuda.empty_cache()
        return

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
        low_rank_a = None
        low_rank_b = None
        reconstructed_weight = None
        if residual_sparse_enabled or resolved_outlier_mode == "post_vae_low_rank":
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
        if resolved_outlier_mode == "per_vae_low_rank":
            payload = low_rank_payloads[i]
            if payload is None:
                raise RuntimeError(f"[{group_tag}] missing per-vae low-rank payload for {r.name}.")
            low_rank_a, low_rank_b = payload
        elif resolved_outlier_mode == "post_vae_low_rank":
            if reconstructed_weight is None:
                raise RuntimeError(f"[{group_tag}] missing reconstructed weight for post-vae low-rank payload.")
            low_rank_a, low_rank_b = _build_post_vae_low_rank_payload(
                linear_name=r.name,
                original_weight=old.weight,
                reconstructed_weight=reconstructed_weight,
                rank=int(outlier_low_rank),
                target_dtype=old.weight.dtype,
            )
            log.info(
                "[%s] post-vae low-rank patch for %s: rank=%d",
                group_tag,
                r.name,
                int(outlier_low_rank),
            )
        if residual_sparse_enabled:
            activation_weight = None
            if resolved_residual_score in _RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT:
                if effective_activation_weight is None or r.name not in effective_activation_weight:
                    raise ValueError(
                        f"[{group_tag}] missing activation vector for residual_sparse scoring at linear '{r.name}'."
                    )
                activation_weight = effective_activation_weight[r.name]
            if reconstructed_weight is None:
                raise RuntimeError(f"[{group_tag}] missing reconstructed weight for residual_sparse payload.")
            sparse_residual_kwargs, sparse_nnz, sparse_storage = _build_sparse_residual_payload(
                linear_name=r.name,
                original_weight=old.weight,
                reconstructed_weight=reconstructed_weight,
                activation_weight=activation_weight,
                score_mode=resolved_residual_score,
                top_p=outlier_residual_top_p,
                min_abs=resolved_residual_min_abs,
                codec=resolved_residual_codec,
                index_bits=outlier_residual_index_bits,
                value_bits=outlier_residual_value_bits,
                block_shape=outlier_residual_block_shape,
            )
            log.info(
                "[%s] residual sparse patch for %s: nnz=%d top_p=%.6f score=%s min_abs=%.6e codec=%s bytes(codec=%d coo=%d)",
                group_tag,
                r.name,
                sparse_nnz,
                outlier_residual_top_p,
                resolved_residual_score,
                resolved_residual_min_abs,
                resolved_residual_codec,
                int(sparse_storage["codec_bytes"]),
                int(sparse_storage["coo_bytes"]),
            )
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
            low_rank_a=low_rank_a,
            low_rank_b=low_rank_b,
        ).to(convert_device)
        new_linear.to("cpu")
        set_module_by_name(model, r.name, new_linear)

    del current_residual_weights, target_common_result, all_stage_bits, all_stage_decoders, all_stage_codebook_dims, all_stage_split_metas
    torch.cuda.empty_cache()


def main(argv: Optional[Sequence[str]] = None) -> None:
    global log
    cat_args, hf_args, training_args, vae_args = process_cat_train_args(argv)
    if bool(getattr(training_args, "lora_hif4_act", False)) and not bool(cat_args.lora_after_category):
        raise ValueError("--lora_hif4_act 仅在 LoRA 阶段生效，因此必须同时开启 --lora_after_category。")
    if bool(cat_args.lora_after_category) and not bool(cat_args.convert):
        raise ValueError("--lora_after_category requires --convert，因为 LoRA 补偿必须作用在已替换的压缩模型上。")
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
    projection_suffixes = _split_csv(cat_args.projection_suffixes)
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
        projection_suffixes=projection_suffixes,
    )
    discovered_categories = [r.category for r in all_linears]
    category_order = _resolve_category_order(cat_args.category_order, discovered_categories)
    discovered_category_set = set(discovered_categories)
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

    active_categories = [c for c in category_order if c in discovered_category_set]
    if not active_categories:
        raise ValueError("No active categories discovered for training.")

    resolved_category_cfgs = resolve_category_runtime_configs(cat_args, vae_args, active_categories)
    resolved_outlier_mode = str(getattr(cat_args, "outlier_protect_mode", "channel")).strip().lower()
    resolved_residual_score = str(getattr(cat_args, "outlier_residual_score", "abs")).strip().lower()
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
    lora_tables = (
        (cat_args.lora_rank, "--lora_rank"),
        (cat_args.lora_alpha, "--lora_alpha"),
        (cat_args.lora_dropout, "--lora_dropout"),
        (cat_args.lora_steps, "--lora_steps"),
        (cat_args.lora_batch_size, "--lora_batch_size"),
        (cat_args.lora_nsamples, "--lora_nsamples"),
        (cat_args.lora_lr, "--lora_lr"),
        (cat_args.lora_weight_decay, "--lora_weight_decay"),
        (cat_args.lora_log_every, "--lora_log_every"),
        (cat_args.lora_temperature, "--lora_temperature"),
        (cat_args.lora_loss_alpha, "--lora_loss_alpha"),
        (cat_args.lora_loss_type, "--lora_loss_type"),
        (cat_args.lora_use_dora, "--lora_use_dora"),
    )
    for table, arg_name in lora_tables:
        validate_category_keys(table, active_categories, arg_name)

    category_intra_parallel: Dict[str, Tuple[int, int]] = {
        cat: tuple(resolved_category_cfgs[cat].intra_parallel) for cat in active_categories
    }
    category_codebook: Dict[str, Tuple[int, int]] = {
        cat: (
            int(resolved_category_cfgs[cat].codebook_bits),
            int(resolved_category_cfgs[cat].codebook_dim),
        )
        for cat in active_categories
    }
    category_outlier_protect_count: Dict[str, int] = {
        cat: int(resolved_category_cfgs[cat].outlier_protect_count) for cat in active_categories
    }
    category_outlier_residual_top_p: Dict[str, float] = {
        cat: float(resolved_category_cfgs[cat].outlier_residual_top_p) for cat in active_categories
    }
    category_outlier_low_rank: Dict[str, int] = {
        cat: int(resolved_category_cfgs[cat].outlier_low_rank) for cat in active_categories
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

    any_wa_mse = any(str(resolved_category_cfgs[cat].recon_loss_type).strip(
    ).lower() == "wa_mse" for cat in active_categories)
    any_outlier_protect = any(count > 0 for count in category_outlier_protect_count.values())
    residual_sparse_needs_activation = (
        resolved_outlier_mode == "residual_sparse"
        and resolved_residual_score in _RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT
    )
    sort_needs_act = any(mode == "act_spectral_cosine" for mode in category_sort_modes.values())
    if any_wa_mse or any_outlier_protect or sort_needs_act or residual_sparse_needs_activation:
        activation_dataset = str(getattr(cat_args, "wa_mse_calib_dataset", "")).strip()
        if not activation_dataset:
            raise ValueError(
                "--wa_mse_calib_dataset must be set when dynamic activation calibration is enabled. "
                "Use ratio-style dataset specs such as 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
            )
        activation_runtime = {
            "cache": None,  # type: Optional[ActivationCalibrationCache]
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
        if any_outlier_protect:
            enabled_features.append("outlier_protect")
        if sort_needs_act:
            enabled_features.append("act_spectral_cosine_sort")
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
                "Residual sparse protection enabled: top_p=%.6f score=%s min_abs=%.6e codec=%s index_bits=%d value_bits=%d block=%dx%d",
                unique_top_p[0],
                resolved_residual_score,
                float(cat_args.outlier_residual_min_abs),
                cat_args.outlier_residual_codec,
                int(cat_args.outlier_residual_index_bits),
                int(cat_args.outlier_residual_value_bits),
                int(cat_args.outlier_residual_block_shape[0]),
                int(cat_args.outlier_residual_block_shape[1]),
            )
        else:
            log.info(
                "Residual sparse protection enabled: top_p_by_category={%s} score=%s min_abs=%.6e codec=%s index_bits=%d value_bits=%d block=%dx%d",
                ",".join(f"{cat}:{category_outlier_residual_top_p[cat]:.6f}" for cat in active_categories),
                resolved_residual_score,
                float(cat_args.outlier_residual_min_abs),
                cat_args.outlier_residual_codec,
                int(cat_args.outlier_residual_index_bits),
                int(cat_args.outlier_residual_value_bits),
                int(cat_args.outlier_residual_block_shape[0]),
                int(cat_args.outlier_residual_block_shape[1]),
            )
    if resolved_outlier_mode in _LOW_RANK_OUTLIER_MODES:
        unique_low_rank = sorted(set(category_outlier_low_rank.values()))
        if len(unique_low_rank) == 1:
            log.info(
                "Low-rank protection enabled: mode=%s rank=%d",
                resolved_outlier_mode,
                int(unique_low_rank[0]),
            )
        else:
            log.info(
                "Low-rank protection enabled: mode=%s rank_by_category={%s}",
                resolved_outlier_mode,
                ",".join(f"{cat}:{category_outlier_low_rank[cat]}" for cat in active_categories),
            )

    unique_parallel = sorted(set(category_intra_parallel.values()))
    if unique_parallel:
        if len(unique_parallel) == 1:
            intra_row_parts, intra_col_parts = unique_parallel[0]
            intra_parts_per_linear = int(intra_row_parts) * int(intra_col_parts)
            intra_parallel_desc = _format_intra_parallel_desc(intra_row_parts, intra_col_parts)
            log.info(
                "并行配置: linear_group_size=%d, intra_parallel=%s (rows=%d, cols=%d), intra_part_sort_mode=%s, total_num_models=%d",
                linear_group_size,
                intra_parallel_desc,
                intra_row_parts,
                intra_col_parts,
                unique_sort_mode_desc[0] if len(
                    unique_sort_mode_desc) == 1 else f"per_category{{{','.join(f'{cat}:{category_sort_mode_desc[cat]}' for cat in active_categories)}}}",
                linear_group_size * intra_parts_per_linear,
            )
        else:
            per_cat_desc = ",".join(
                f"{cat}:{_format_intra_parallel_desc(*category_intra_parallel[cat])}"
                for cat in active_categories
            )
            models_per_group_values = sorted(
                linear_group_size * int(rp) * int(cp)
                for rp, cp in unique_parallel
            )
            log.info(
                "并行配置: linear_group_size=%d, intra_parallel=per_category{%s}, intra_part_sort_mode=per_category{%s}, total_num_models_per_group=[%d,%d]",
                linear_group_size,
                per_cat_desc,
                ",".join(f"{cat}:{category_sort_mode_desc[cat]}" for cat in active_categories),
                models_per_group_values[0],
                models_per_group_values[-1],
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

    sort_prep_workers_resolved = 1
    sort_executor = None
    if any_sort_enabled:
        sort_prep_workers_resolved = _resolve_sort_prep_workers(
            int(cat_args.sort_prep_workers),
            linear_group_size=int(linear_group_size),
        )
        if sort_prep_workers_resolved > 1:
            sort_executor = ProcessPoolExecutor(
                max_workers=int(sort_prep_workers_resolved),
                mp_context=mp.get_context("spawn"),
                initializer=_init_sort_prep_worker,
            )
            log.info(
                "排序预处理并行已启用: sort_backend=cpu_process sort_prep_workers_resolved=%d requested=%d",
                int(sort_prep_workers_resolved),
                int(cat_args.sort_prep_workers),
            )
        else:
            log.info(
                "排序预处理使用串行: sort_backend=cpu_serial sort_prep_workers_resolved=1 requested=%d",
                int(cat_args.sort_prep_workers),
            )

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
        any_lora_after_overrides = any(table.is_override_enabled() for table, _ in lora_tables)
        if any_lora_after_overrides:
            log.info(
                "LoRA after-category overrides enabled: keys=%s",
                ",".join(
                    sorted(
                        {
                            key
                            for table, _arg_name in lora_tables
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
                projection_suffixes=projection_suffixes,
            )
            if missing:
                log.warning("[%s] %d modules missing layer_idx, skipped.", cat, missing)
            if not refs_sorted:
                continue

            cat_cfg = resolved_category_cfgs[cat]
            refs = [ref for _, ref in refs_sorted]
            cat_row_parts, cat_col_parts = category_intra_parallel[cat]
            cat_codebook_bits, cat_codebook_dim = category_codebook[cat]
            cat_parts_per_linear = int(cat_row_parts) * int(cat_col_parts)
            cat_intra_parallel_desc = _format_intra_parallel_desc(cat_row_parts, cat_col_parts)
            log.info(
                "=== Category: %s (%d linears, residual_stages=%d, intra_parallel=%s rows=%d cols=%d, codebook_bits=%d, codebook_dim=%d, recon_loss=%s, sort=%s, steps=%d, joint_lr=%.3e, joint_group=%d, joint_batch=%s) ===",
                cat,
                len(refs),
                int(cat_cfg.residual_stages),
                cat_intra_parallel_desc,
                cat_row_parts,
                cat_col_parts,
                int(cat_codebook_bits),
                int(cat_codebook_dim),
                str(cat_cfg.recon_loss_type),
                category_sort_mode_desc[cat],
                int(cat_cfg.steps),
                float(cat_cfg.joint_decoder_lr),
                int(cat_cfg.joint_decoder_group_size),
                "none" if cat_cfg.joint_decoder_batch_size is None else str(int(cat_cfg.joint_decoder_batch_size)),
            )
            ordered_refs = [r for _, r in refs_sorted]

            for start in range(0, len(ordered_refs), linear_group_size):
                group_refs = ordered_refs[start:start + linear_group_size]
                if len(group_refs) < linear_group_size and not cat_args.allow_tail_group:
                    log.info("[%s] tail group size=%d skipped (set --allow_tail_group to include).", cat, len(group_refs))
                    break
                layer_indices = [idx for idx, _ in refs_sorted[start:start + linear_group_size]]
                group_tag = f"{cat}.L{layer_indices[0]}-{layer_indices[-1]}"
                log.info(
                    "---- Group: %s (linears=%d, intra_parallel=%s, num_models=%d) ----",
                    group_tag,
                    len(group_refs),
                    cat_intra_parallel_desc,
                    len(group_refs) * cat_parts_per_linear,
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
                    skip_layer_keys=skip_layer_keys,
                    activation_runtime=activation_runtime,
                    outlier_protect_mode=resolved_outlier_mode,
                    outlier_residual_score=resolved_residual_score,
                    outlier_residual_min_abs=cat_args.outlier_residual_min_abs,
                    outlier_protect_axis=outlier_protect_axis,
                    outlier_residual_codec=cat_args.outlier_residual_codec,
                    outlier_residual_index_bits=cat_args.outlier_residual_index_bits,
                    outlier_residual_value_bits=cat_args.outlier_residual_value_bits,
                    outlier_residual_block_shape=cat_args.outlier_residual_block_shape,
                    sort_executor=sort_executor,
                    sort_prep_workers_resolved=int(sort_prep_workers_resolved),
                    deterministic=bool(cat_args.deterministic),
                    shuffle_seed=int(cat_args.seed) + int(cat_idx) * 100000 + int(start),
                )
            if run_category_eval and not bool(cat_args.lora_after_category):
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

            if cat_args.lora_after_category:
                from train_utils.lora_utils import lora_finetune_remaining_categories
                if run_category_eval:
                    log.info("LoRA 微调前评估...")
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
                current_remaining_linears = _collect_current_trainable_linears(
                    model,
                    transpose_modules=transpose_modules,
                    only_decoder_projections=only_decoder_projections,
                    projection_suffixes=projection_suffixes,
                )
                remaining_categories = list(dict.fromkeys(r.category for r in current_remaining_linears))
                model = lora_finetune_remaining_categories(
                    model=model,
                    remaining_categories=remaining_categories,
                    target_names=[r.name for r in current_remaining_linears],
                    cat_args=cat_args,
                    vae_args=vae_args,
                    training_args=training_args,
                    logger=log,
                    lora_round_idx=lora_round_idx,
                    after_category=cat,
                )
                lora_round_idx += 1

            if run_category_eval and bool(cat_args.lora_after_category):
                log.info("LoRA 微调后评估...")
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
            from litebsq.vae_linear import clear_model_vae_linear_cache

            model_out = os.path.join(run_output_dir, "final_model")
            tok = AutoTokenizer.from_pretrained(vae_args.model_path, use_fast=True, token=hf_args.access_token)
            fused_post_norm_head = fuse_post_norm_head_linear(model)
            if fused_post_norm_head:
                log.info("Final save: fused post_norm_linear into lm_head.weight.")
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
        if sort_executor is not None:
            sort_executor.shutdown(wait=True, cancel_futures=False)
            log.info("排序预处理进程池已关闭。")

    log.info("Done.")


if __name__ == "__main__":
    main()
