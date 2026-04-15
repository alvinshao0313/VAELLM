import os
import sys
import time
import math
import json
import argparse
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

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
from litebsq.vae_args import apply_autoencoder_arch_defaults
from litebsq.misc import set_module_by_name
from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED,
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
    encode_blocked_quantized_sparse_residual,
    sparse_residual_blocked_storage_bytes,
    sparse_residual_coo_storage_bytes,
)
from train_utils.cat_data_prep import (
    LinearPrepRef,
    format_intra_part_sort_mode,
    gather_wa_mse_act_max_batch,
    prepare_group_weight_data,
)
from train_utils.activation_utils import (
    ActivationCalibrationCache,
    collect_act_max_for_linears,
)
from train_utils.cat_arg_overrides import validate_category_keys
from train_utils.hif4_act import applied_hif4_act
from train_utils.model_checkpoint_io import (
    META_FILENAME,
    _build_run_output_dir,
    load_model_checkpoint,
    resolve_checkpoint_dir,
    save_model_checkpoint,
)
from train_utils.utils import (
    LinearRef,
    clone_namespace as _clone_namespace,
    collect_linears as _collect_linears,
    extract_layer_idx as _extract_layer_idx,
    format_intra_parallel_desc as _format_intra_parallel_desc,
    format_namespace as _format_namespace,
    get_logger,
    resolve_category_order as _resolve_category_order,
    set_seed,
    split_csv as _split_csv,
)


log = get_logger("linear_by_category")


_RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT = frozenset(
    {"input_act_weighted_abs", "input_act_weighted_original_weight_abs"}
)


def _to_jsonable(value):
    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return value.to_jsonable()
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        return _to_jsonable(value.value)
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, argparse.Namespace):
        return {k: _to_jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _resolve_rot_block_size(codebook_dim_value) -> int:
    if hasattr(codebook_dim_value, "has_default"):
        if not bool(getattr(codebook_dim_value, "has_default", False)):
            raise ValueError("--rot_llm requires --codebook_dim to provide a default value.")
        return int(getattr(codebook_dim_value, "default"))
    return int(codebook_dim_value)


def _save_normalized_cat_train_snapshot(
    *,
    run_output_dir: str,
    cat_args,
    vae_args,
    training_args,
    resolved_category_cfgs: Dict[str, ResolvedCategoryRuntimeConfig],
) -> str:
    snapshot_path = os.path.join(run_output_dir, "normalized_cat_train_args.json")
    payload = {
        "cat_args": _to_jsonable(cat_args),
        "vae_args": _to_jsonable(vae_args),
        "training_args": _to_jsonable(training_args),
        "resolved_category_runtime": {
            category: _to_jsonable(cfg)
            for category, cfg in resolved_category_cfgs.items()
        },
    }
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return snapshot_path


def _load_model_for_cat_train(*, cat_args, hf_args, vae_args) -> nn.Module:
    if getattr(cat_args, "resume_from_checkpoint", None):
        if bool(getattr(cat_args, "rot_llm", False)):
            raise ValueError(
                "--resume_from_checkpoint cannot be combined with --rot_llm because the checkpoint already contains model weights to resume from.")

        checkpoint_dir = resolve_checkpoint_dir(str(cat_args.resume_from_checkpoint))
        meta_path = os.path.join(checkpoint_dir, META_FILENAME)
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        base_model_path = meta.get("base_model_path")
        if base_model_path is None:
            base_model_path = getattr(vae_args, "model_path", None)
        if not base_model_path:
            raise ValueError(
                f"Cannot determine base model path for resumed checkpoint: {checkpoint_dir}. "
                "Please save checkpoints with base_model_path metadata or pass --model_path."
            )

        log.info("Resuming from checkpoint: %s", checkpoint_dir)
        log.info("Resume base model path: %s", str(base_model_path))
        model, load_meta, load_result = load_model_checkpoint(
            checkpoint_dir,
            access_token=hf_args.access_token,
            base_model_path=str(base_model_path),
            map_location="cpu",
            strict=True,
        )
        vae_args.model_path = str(load_meta.get("base_model_path") or base_model_path)
        log.info(
            "Checkpoint loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s",
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
            str(load_meta.get("converted_module_count")),
        )
        return model

    log.info("Loading model: %s", vae_args.model_path)
    from rotation.model_utils import get_model

    model = get_model(vae_args.model_path, hf_args.access_token)
    if bool(getattr(cat_args, "rot_llm", False)):
        from rotation.model_rotation import prepare_model

        rot_block_size = _resolve_rot_block_size(getattr(vae_args, "codebook_dim", 32))
        log.info("Applying offline LLM rotation fusion before VAE compression.")
        log.info("Rotation block size resolved from --codebook_dim default: %d", rot_block_size)
        model = prepare_model(model, rot_block_size=rot_block_size)
    return model


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


def _build_block_data_loaders(
    stacked_data: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    block_indices = torch.arange(stacked_data.shape[0], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(stacked_data, block_indices)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, eval_loader


def _reshape_blocks_for_codebook_dim(
    stacked_data: torch.Tensor,
    *,
    codebook_dim: int,
) -> torch.Tensor:
    target_dim = int(codebook_dim)
    if target_dim < 1:
        raise ValueError(f"codebook_dim must be >=1, got {target_dim}")
    if int(stacked_data.shape[-1]) == target_dim:
        return stacked_data
    num_models = int(stacked_data.shape[1])
    flat = stacked_data.permute(1, 0, 2).contiguous().view(num_models, -1)
    total_numel = int(flat.shape[1])
    if total_numel % target_dim != 0:
        raise ValueError(
            f"Cannot reshape residual blocks: total_numel_per_model={total_numel} not divisible by codebook_dim={target_dim}"
        )
    return flat.view(num_models, -1, target_dim).permute(1, 0, 2).contiguous()


def _compute_stage_norm_stats(
    stage_data: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if stage_data.ndim != 3:
        raise ValueError(f"stage_data must be 3D [N_blocks, P, C], got shape={tuple(stage_data.shape)}")
    num_models = int(stage_data.shape[1])
    flat = stage_data.permute(1, 0, 2).contiguous().view(num_models, -1)
    mean = flat.mean(dim=1, keepdim=True)
    scale = flat.std(dim=1, keepdim=True).clamp_min(float(eps))
    return mean, scale


def _apply_stage_norm(
    stage_data: torch.Tensor,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    num_blocks = int(stage_data.shape[0])
    num_models = int(stage_data.shape[1])
    codebook_dim = int(stage_data.shape[2])
    flat = stage_data.permute(1, 0, 2).contiguous().view(num_models, -1)
    norm_flat = (flat - mean) / scale
    return norm_flat.view(num_models, num_blocks, codebook_dim).permute(1, 0, 2).contiguous()


def _restore_stage_norm(
    stage_data_norm: torch.Tensor,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    num_blocks = int(stage_data_norm.shape[0])
    num_models = int(stage_data_norm.shape[1])
    codebook_dim = int(stage_data_norm.shape[2])
    flat = stage_data_norm.permute(1, 0, 2).contiguous().view(num_models, -1)
    raw_flat = flat * scale + mean
    return raw_flat.view(num_models, num_blocks, codebook_dim).permute(1, 0, 2).contiguous()


def _eval_ppl_after_category(
    model: nn.Module,
    vae_args,
    ppl_limit: int,
    category: str,
    eval_device: str = "cuda",
    eval_hif4_act: bool = False,
) -> None:
    from train_utils.eval_utils import calculate_ppl

    log.info("开始类别 %s 的 PPL 评估...", category)
    model.eval()
    model.to(eval_device)
    with applied_hif4_act(
        model,
        enabled=bool(eval_hif4_act),
        logger=log,
        log_prefix=f"[ppl:{category}] ",
    ):
        with torch.no_grad():
            ppl_args = _clone_namespace(vae_args, limit=int(ppl_limit))
            ppl_result = calculate_ppl(model, ppl_args)
    model.to("cpu")
    torch.cuda.empty_cache()
    log.info("类别 %s 训练后 PPL: %.2f", category, float(ppl_result.get("wiki_ppl", float("nan"))))


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
        compressed_in_features=int(split_meta.compressed_in_features),
        compressed_out_features=int(split_meta.compressed_out_features),
        protected_input_indices=split_meta.protected_input_indices,
        protected_input_weight=split_meta.protected_input_weight,
        protected_output_indices=split_meta.protected_output_indices,
        protected_output_weight=split_meta.protected_output_weight,
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

    if resolved_score_mode in _RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT:
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


def _build_sparse_residual_payload(
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
    batch_size: int,
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
) -> None:
    from litebsq.llm_vae import MultiLayerVAE

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
    outlier_residual_top_p = float(runtime_cfg.outlier_residual_top_p)
    resolved_outlier_mode = str(outlier_protect_mode).strip().lower()
    resolved_residual_score = str(outlier_residual_score).strip().lower()
    resolved_residual_min_abs = float(outlier_residual_min_abs)
    residual_sparse_enabled = resolved_outlier_mode == "residual_sparse"
    residual_sparse_needs_activation = (
        residual_sparse_enabled and resolved_residual_score in _RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT
    )
    if resolved_outlier_mode not in {"channel", "residual_sparse"}:
        raise ValueError(
            f"[{group_tag}] unsupported outlier_protect_mode={outlier_protect_mode!r}. "
            "Expected channel or residual_sparse."
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
            dataset=str(activation_runtime.get("dataset", "wikitext2")),
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
            str(activation_runtime.get("dataset", "wikitext2")),
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
    prep_result = prepare_group_weight_data(
        group_refs=prep_refs,
        intra_parallel=(row_parts, col_parts),
        codebook_dim=int(stage_codebook_dim),
        batch_size=int(batch_size),
        # 多阶残差独立 norm：这里保持原始域，后续在每个 stage 内单独做标准化。
        normalize_weight=False,
        recon_loss_type="wa_mse" if use_wa_mse_loss else stage_recon_loss,
        activation_weight_by_linear=effective_activation_weight,
        train_device=train_device,
        intra_part_sort_mode=stage_sort_mode,
        outlier_protect_count=int(outlier_protect_count) if resolved_outlier_mode == "channel" else 0,
        outlier_protect_axis=str(outlier_protect_axis),
    )
    num_models = int(prep_result.num_models)
    stacked_data = prep_result.stacked_data
    use_wa_mse = bool(prep_result.use_wa_mse)
    part_metas = prep_result.part_metas
    split_metas = prep_result.split_metas
    if resolved_outlier_mode == "channel" and int(outlier_protect_count) > 0:
        per_linear_protected = []
        for ref, meta in zip(group_refs, split_metas):
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
    if len(split_metas) != len(group_refs):
        raise RuntimeError(
            f"[{group_tag}] split metadata mismatch: len(split_metas)={len(split_metas)} "
            f"vs len(group_refs)={len(group_refs)}"
        )
    if use_wa_mse:
        log.info("[%s] wa_mse enabled with online act_max gather.", group_tag)

    residual_data = _reshape_blocks_for_codebook_dim(
        stacked_data.detach().clone().contiguous(),
        codebook_dim=int(stage_codebook_dim),
    )
    all_stage_bits: List[torch.Tensor] = []
    all_stage_decoders: List[List[nn.Module]] = []
    all_stage_codebook_dims: List[int] = []

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

        train_loader, eval_loader = _build_block_data_loaders(stage_train_data, batch_size=int(batch_size))
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

        residual_rms_before = float(residual_data.float().pow(2).mean().sqrt().item())
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
        train_iter = iter(train_loader)
        for step in range(int(stage_steps)):
            try:
                x_batch, block_idx_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_batch, block_idx_batch = next(train_iter)

            x = x_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
            act_max_batch = None
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
                    "[%s] step=%d/%d loss=%.6f recon=%.6f commit=%.6f speed=%.4fs/it",
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
                if do_convert:
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
        residual_data = (residual_data - stage_recon_full).contiguous()
        residual_rms_after = float(residual_data.float().pow(2).mean().sqrt().item())
        log.info(
            "[%s] residual rms: before=%.6e after=%.6e",
            stage_tag,
            residual_rms_before,
            residual_rms_after,
        )

        if do_convert:
            if not stage_bit_chunks:
                raise RuntimeError(f"[{stage_tag}] no bit indices collected during conversion.")
            stage_full_bits = torch.cat(stage_bit_chunks, dim=0)  # [N_blocks, P, latent_dim]
            all_stage_bits.append(stage_full_bits)
            all_stage_codebook_dims.append(int(stage_codebook_dim))

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

        del vae, train_loader, eval_loader, optimizer
        if lr_scheduler is not None:
            del lr_scheduler
        torch.cuda.empty_cache()

    # # 保存分组 VAE，便于复现实验和离线分析。
    # group_dir = os.path.join(output_dir, "vae_by_category", group_tag.replace("/", "_"))
    # os.makedirs(group_dir, exist_ok=True)
    # torch.save(vae.state_dict(), os.path.join(group_dir, "vae_state.pt"))

    if not do_convert:
        del stacked_data, residual_data
        torch.cuda.empty_cache()
        return

    if (
        len(all_stage_bits) != residual_stages
        or len(all_stage_decoders) != residual_stages
        or len(all_stage_codebook_dims) != residual_stages
    ):
        raise RuntimeError(
            f"[{group_tag}] stage payload mismatch: bits={len(all_stage_bits)} "
            f"decoders={len(all_stage_decoders)} codebook_dims={len(all_stage_codebook_dims)} "
            f"residual_stages={residual_stages}"
        )

    for i, r in enumerate(group_refs):
        old = r.module
        split_meta = split_metas[i]
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
        if residual_sparse_enabled:
            activation_weight = None
            if resolved_residual_score in _RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT:
                if effective_activation_weight is None or r.name not in effective_activation_weight:
                    raise ValueError(
                        f"[{group_tag}] missing activation vector for residual_sparse scoring at linear '{r.name}'."
                    )
                activation_weight = effective_activation_weight[r.name]
            reconstructed_weight = _decode_reconstructed_linear_weight(
                old_module=old,
                transpose=r.transpose,
                split_meta=split_meta,
                stage_part_bits_payload=stage_part_bits_payload,
                stage_part_decoders_payload=stage_part_decoders_payload,
                stage_codebook_dims=all_stage_codebook_dims,
                parallel_rows=row_parts,
                parallel_cols=col_parts,
                parallel_parts=parts_per_linear,
            )
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
        ).to(convert_device)
        new_linear.to("cpu")
        set_module_by_name(model, r.name, new_linear)

    del stacked_data, residual_data, all_stage_bits, all_stage_decoders, all_stage_codebook_dims
    torch.cuda.empty_cache()


def main(argv: Optional[Sequence[str]] = None) -> None:
    global log
    cat_args, hf_args, training_args, vae_args = process_cat_train_args(argv)
    if bool(getattr(training_args, "lora_hif4_act", False)) and not bool(cat_args.lora_after_category):
        raise ValueError("--lora_hif4_act 仅在 LoRA 阶段生效，因此必须同时开启 --lora_after_category。")
    if bool(cat_args.lora_after_category) and not bool(cat_args.convert):
        raise ValueError("--lora_after_category requires --convert，因为 LoRA 补偿必须作用在已替换的压缩模型上。")
    set_seed(cat_args.seed)

    os.makedirs(cat_args.output_dir, exist_ok=True)
    run_output_dir = _build_run_output_dir(cat_args.output_dir, vae_args.model_path)
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "linear_by_category.log")
    log = get_logger("linear_by_category")
    cat_args.output_dir = run_output_dir

    log.info("Run output directory: %s", run_output_dir)
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
    run_ppl_eval = bool(cat_args.convert)
    if not run_ppl_eval:
        log.info("跳过 PPL 评估：--convert=false 时模型权重不会被替换。")
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
    category_sort_modes: Dict[str, str] = {
        cat: str(resolved_category_cfgs[cat].intra_part_sort_mode)
        for cat in active_categories
    }
    category_sort_mode_desc: Dict[str, str] = {
        cat: format_intra_part_sort_mode(category_sort_modes[cat]) for cat in active_categories
    }
    unique_sort_mode_desc = sorted(set(category_sort_mode_desc.values()))

    any_wa_mse = any(str(resolved_category_cfgs[cat].recon_loss_type).strip(
    ).lower() == "wa_mse" for cat in active_categories)
    any_outlier_protect = any(count > 0 for count in category_outlier_protect_count.values())
    residual_sparse_needs_activation = (
        resolved_outlier_mode == "residual_sparse"
        and resolved_residual_score in _RESIDUAL_SPARSE_SCORE_MODES_NEED_ACT
    )
    sort_needs_act = any(mode == "act_spectral_cosine" for mode in category_sort_modes.values())
    if any_wa_mse or any_outlier_protect or sort_needs_act or residual_sparse_needs_activation:
        activation_runtime = {
            "cache": None,  # type: Optional[ActivationCalibrationCache]
            "dataset": str(getattr(cat_args, "wa_mse_calib_dataset", "wikitext2")),
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
    for cat in active_categories:
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
            "=== Category: %s (%d linears, residual_stages=%d, intra_parallel=%s rows=%d cols=%d, codebook_bits=%d, codebook_dim=%d, recon_loss=%s, sort=%s, steps=%d) ===",
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
            )
            # _eval_ppl_after_category(
            #     model=model,
            #     vae_args=vae_args,
            #     ppl_limit=cat_args.ppl_limit,
            #     category=cat,
            #     eval_device=cat_args.train_device,
            # )

        if cat_args.lora_after_category:
            from train_utils.lora_utils import lora_finetune_remaining_categories
            if run_ppl_eval:
                log.info("LoRA 微调前评估...")
                _eval_ppl_after_category(
                    model=model,
                    vae_args=vae_args,
                    ppl_limit=cat_args.ppl_limit,
                    category=cat,
                    eval_device=cat_args.train_device,
                    eval_hif4_act=cat_args.eval_hif4_act,
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

        if run_ppl_eval:
            _eval_ppl_after_category(
                model=model,
                vae_args=vae_args,
                ppl_limit=cat_args.ppl_limit,
                category=cat,
                eval_device=cat_args.train_device,
                eval_hif4_act=cat_args.eval_hif4_act,
            )
        # cat_dir_name = _safe_path_token(cat)
        # cat_model_dir = os.path.join(run_output_dir, cat_dir_name)
        # save_paths = save_model_checkpoint(
        #     model,
        #     cat_model_dir,
        #     base_model_path=vae_args.model_path,
        #     tokenizer=None,
        #     save_config=True,
        #     extra_meta={
        #         "stage": "after_category",
        #         "category": cat,
        #         "category_index": int(cat_idx),
        #         "lora_after_category": bool(cat_args.lora_after_category),
        #     },
        # )
        # log.info("Saved category checkpoint (%s): %s", cat, save_paths["output_dir"])

    if run_ppl_eval:
        _eval_ppl_after_category(
            model=model,
            vae_args=vae_args,
            ppl_limit=cat_args.ppl_limit,
            category="none",
            eval_device=cat_args.train_device,
            eval_hif4_act=cat_args.eval_hif4_act,
        )
    if cat_args.save_model:
        if not cat_args.convert:
            raise ValueError("--save_model requires --convert")
        from transformers import AutoTokenizer
        from litebsq.vae_linear import clear_model_vae_linear_cache

        model_out = os.path.join(run_output_dir, "final_model")
        tok = AutoTokenizer.from_pretrained(vae_args.model_path, use_fast=True, token=hf_args.access_token)
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

    log.info("Done.")


if __name__ == "__main__":
    main()
