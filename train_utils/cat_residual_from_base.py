import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from litebsq.misc import set_module_by_name
from litebsq.sparse_residual import (
    SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED,
    SPARSE_RESIDUAL_FORMAT_COO_FP16,
    get_default_block_shape_for_index_bits,
)
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import (
    NamedVAELinearTarget,
    clear_model_vae_linear_cache,
    prime_named_vae_linear_cache,
)
from train_utils.activation_utils import collect_activation_stats_for_linears
from train_utils.cat_data_prep import compute_channel_rank_score, select_outlier_channel_indices_from_scores
from train_utils.cat_train_eval import eval_after_category as _eval_after_category
from train_utils.cat_train_runtime import normalize_cat_runtime_vae_original_state
from train_utils.cat_train_runtime import build_cat_run_output_dir as _build_run_output_dir
from train_utils.base_reference import (
    clone_frozen_linear_from_reference,
    get_reference_module,
    load_frozen_base_reference_model,
)
from train_utils.cat_train_pipeline import (
    _apply_stage_norm,
    _build_block_data_loaders,
    _compute_stage_norm_stats,
    _fuse_norm_into_decoder,
    _fuse_q_scale_into_decoder,
    _resolve_train_dtype,
    _restore_stage_norm,
)
from train_utils.cat_train_residual_protection import (
    RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX,
    RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMEAN,
    build_sparse_residual_payload,
)
from train_utils.checkpoint_v6 import (
    META_FILENAME,
    resolve_v6_checkpoint_dir,
    save_v6_full_checkpoint,
)
from train_utils.shared_protected_residual import (
    get_shared_protected_residual_decoder_registry,
    register_shared_protected_residual_decoder,
)
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.train_args import _parse_bool_like, create_optimizer
from train_utils.utils import (
    clone_namespace as _clone_namespace,
    configure_deterministic_mode,
    get_logger,
    set_seed,
    split_csv,
)
from litebsq.vae_args import apply_autoencoder_arch_defaults


@dataclass(frozen=True)
class _ResidualVAERuntimeConfig:
    category: str
    residual_stages: int
    steps: int
    intra_part_sort_mode: str
    codebook_bits: int
    codebook_dim: int
    outlier_protect_count: int
    outlier_residual_top_p: float
    outlier_residual_vae_stages: int
    outlier_residual_vae_codebook_bits: int
    outlier_residual_vae_codebook_dim: int
    recon_loss_type: str
    base_ch: int
    num_res_blocks: int
    decoder_base_ch: Optional[int]
    decoder_num_res_blocks: Optional[int]
    norm_type: str
    activation_type: str
    decoder_type: str


def _safe_shared_decoder_ref(value: str) -> str:
    text = str(value).strip()
    chars = [ch if (ch.isalnum() or ch == "_") else "_" for ch in text]
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "shared_protected_residual_decoder"


_SPARSE_METRICS = {
    "sparse_residual_abs",
    "sparse_residual_actmax_abs",
    "sparse_residual_actmean_abs",
    "sparse_weight_abs",
    "sparse_weight_actmax_abs",
    "sparse_weight_actmean_abs",
}
_CHANNEL_METRICS = {
    "channel_weight_abs",
    "channel_weight_actmax_abs",
    "channel_weight_actmean_abs",
    "channel_residual_abs",
    "channel_residual_actmax_abs",
    "channel_residual_actmean_abs",
    "channel_residual_actrms_abs",
}
_ACTMAX_METRICS = {
    "sparse_residual_actmax_abs",
    "sparse_weight_actmax_abs",
    "channel_weight_actmax_abs",
    "channel_residual_actmax_abs",
}
_ACTMEAN_METRICS = {
    "sparse_residual_actmean_abs",
    "sparse_weight_actmean_abs",
    "channel_weight_actmean_abs",
    "channel_residual_actmean_abs",
}
_ACTRMS_METRICS = {"channel_residual_actrms_abs"}


def _metric_requires_activation(metric: Optional[str]) -> bool:
    resolved = "" if metric is None else str(metric).strip().lower()
    return resolved in _ACTMAX_METRICS or resolved in _ACTMEAN_METRICS or resolved in _ACTRMS_METRICS


@dataclass(frozen=True)
class _ResidualTarget:
    name: str
    category: str
    module: VAELinear
    transpose: bool


@dataclass(frozen=True)
class _RuntimeVAETarget:
    name: str
    category: str
    module: nn.Module
    base_layer: VAELinear


@dataclass
class _ResidualFromBaseResidency:
    stashed_vae_modules: Dict[str, nn.Module] = field(default_factory=dict)
    reference_dense_linears: Dict[str, nn.Linear] = field(default_factory=dict)


def _str_to_bool(value: object) -> bool:
    return _parse_bool_like(value, arg_name="bool")


def _parse_block_shape(raw: str) -> Tuple[int, int]:
    text = str(raw).strip().lower().replace("x", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--outlier_residual_block_shape must be ROWS,COLS.")
    rows, cols = int(parts[0]), int(parts[1])
    if rows < 1 or cols < 1:
        raise argparse.ArgumentTypeError("--outlier_residual_block_shape values must be >= 1.")
    return int(rows), int(cols)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add residual/outlier protection to an already trained base VAE checkpoint.",
        allow_abbrev=False,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_vae_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_categories", required=True)
    parser.add_argument("--transpose_modules", default="q_proj,v_proj,o_proj,down_proj")
    parser.add_argument("--include_all_linears", action="store_true")
    parser.add_argument(
        "--outlier_protect_mode",
        required=True,
        choices=("none", "residual_sparse", "channel_residual_vae"),
    )
    parser.add_argument("--outlier_rank_metric", default=None)
    parser.add_argument("--outlier_protect_axis", default=None, choices=("input", "output"))
    parser.add_argument("--outlier_channel_scope", default=None, choices=("layer", "category"))
    parser.add_argument("--outlier_protect_count", type=int, default=None)
    parser.add_argument("--outlier_protect_min_per_layer", type=int, default=0)
    parser.add_argument("--sparse_residual_ratio", type=float, default=None)
    parser.add_argument(
        "--outlier_residual_vae_decoder_share_scope",
        default=None,
        choices=("none", "category"),
    )
    parser.add_argument("--outlier_residual_vae_batch_multiplier", type=int, default=None)
    parser.add_argument("--outlier_residual_vae_steps", type=int, default=None)
    parser.add_argument("--outlier_residual_vae_lr", type=float, default=None)
    parser.add_argument("--outlier_residual_vae_stages", type=int, default=1)
    parser.add_argument("--outlier_residual_vae_codebook_bits", type=int, default=0)
    parser.add_argument("--outlier_residual_vae_codebook_dim", type=int, default=0)
    parser.add_argument("--base_batch_size", type=int, default=8192)
    parser.add_argument("--activation_calib_dataset", type=str, default="")
    parser.add_argument("--activation_calib_nsamples", type=int, default=512)
    parser.add_argument("--activation_calib_seqlen", type=int, default=512)
    parser.add_argument("--activation_calib_seed", type=int, default=0)
    parser.add_argument("--activation_calib_device", type=str, default="")
    parser.add_argument("--activation_calib_log_every", type=int, default=0)
    parser.add_argument("--replace_existing_residual_protection", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--access_token", default=None)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--deterministic", type=_str_to_bool, default=False)
    parser.add_argument("--train_device", default="cuda")
    parser.add_argument("--convert_device", default="cpu")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_ppl", type=_str_to_bool, default=False)
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument("--ppl_limit", type=int, default=-1)
    parser.add_argument("--eval_hif4_act", type=_str_to_bool, default=False)
    parser.add_argument("--eval_before_residual", type=_str_to_bool, default=True)
    parser.add_argument("--eval_after_residual", type=_str_to_bool, default=True)
    parser.add_argument("--bf16", type=_str_to_bool, default=True)
    parser.add_argument("--fp16", type=_str_to_bool, default=False)

    parser.add_argument("--codebook_bits", type=int, default=32)
    parser.add_argument("--codebook_dim", type=int, default=0)
    parser.add_argument("--base_ch", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--decoder_base_ch", type=int, default=128)
    parser.add_argument("--decoder_num_res_blocks", type=int, default=1)
    parser.add_argument("--norm_type", default="layer", choices=("group", "batch", "layer", "rms", "no"))
    parser.add_argument("--activation_type", default="swish", choices=("swish", "relu", "none", "sigmoid", "gelu", "hard_swish"))
    parser.add_argument("--decoder_type", default="symmetric", choices=("linear", "symmetric", "asymmetric"))
    parser.add_argument("--recon_loss_type", default="mse")
    parser.add_argument("--quantizer_type", default="BSQ")
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--inv_temperature", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adamw", choices=("adam", "adamw", "sgd", "rmsprop"))
    parser.add_argument("--lr_scheduler", default="linear", choices=("constant", "linear", "cosine"))
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--lfq_weight", type=float, default=2.5)
    parser.add_argument("--commitment_loss_weight", type=float, default=0.25)
    parser.add_argument("--entropy_loss_weight", type=float, default=0.01)
    parser.add_argument("--normalize_weight", action="store_true")
    parser.add_argument("--new_quant", action="store_true")
    parser.add_argument("--vae_decoder_checkpoint", type=_str_to_bool, default=True)

    parser.add_argument("--outlier_residual_min_abs", type=float, default=0.0)
    parser.add_argument(
        "--outlier_residual_codec",
        default=SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED,
        choices=(SPARSE_RESIDUAL_FORMAT_COO_FP16, SPARSE_RESIDUAL_FORMAT_BLOCKED_QUANTIZED),
    )
    parser.add_argument("--outlier_residual_index_bits", type=int, default=8)
    parser.add_argument("--outlier_residual_value_bits", type=int, default=8)
    parser.add_argument("--outlier_residual_block_shape", type=_parse_block_shape, default=None)
    return parser


def _provided_options(argv: Optional[Sequence[str]]) -> set:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    out = set()
    for item in argv:
        if not str(item).startswith("--"):
            continue
        out.add(str(item).split("=", 1)[0])
    return out


def validate_residual_from_base_args(args: argparse.Namespace, *, provided: set) -> None:
    mode = str(args.outlier_protect_mode).strip().lower()
    metric = None if args.outlier_rank_metric is None else str(args.outlier_rank_metric).strip().lower()
    if bool(args.bf16) and bool(args.fp16):
        raise ValueError("--bf16 and --fp16 are mutually exclusive.")
    if args.base_batch_size < 1:
        raise ValueError(f"--base_batch_size must be >= 1, got {args.base_batch_size}.")
    if args.outlier_residual_vae_stages < 1:
        raise ValueError(
            f"--outlier_residual_vae_stages must be >= 1, got {args.outlier_residual_vae_stages}."
        )
    if int(getattr(args, "outlier_residual_vae_codebook_bits", 0)) < 0:
        raise ValueError(
            "--outlier_residual_vae_codebook_bits must be >= 0, "
            f"got {getattr(args, 'outlier_residual_vae_codebook_bits', 0)}."
        )
    if int(getattr(args, "outlier_residual_vae_codebook_dim", 0)) < 0:
        raise ValueError(
            "--outlier_residual_vae_codebook_dim must be >= 0, "
            f"got {getattr(args, 'outlier_residual_vae_codebook_dim', 0)}."
        )
    if args.output_dir and os.path.abspath(str(args.output_dir)) == os.path.abspath(str(args.base_vae_checkpoint)):
        raise ValueError("--output_dir must not equal --base_vae_checkpoint.")
    if _metric_requires_activation(metric):
        if not str(args.activation_calib_dataset).strip():
            raise ValueError(
                f"{metric} requires online activation stats; set --activation_calib_dataset."
            )
        if int(args.activation_calib_nsamples) <= 0:
            raise ValueError(
                f"{metric} requires --activation_calib_nsamples > 0, got {args.activation_calib_nsamples}."
            )
        if int(args.activation_calib_seqlen) <= 0:
            raise ValueError(
                f"{metric} requires --activation_calib_seqlen > 0, got {args.activation_calib_seqlen}."
            )

    channel_args = {
        "--outlier_protect_axis",
        "--outlier_channel_scope",
        "--outlier_protect_count",
        "--outlier_protect_min_per_layer",
        "--outlier_residual_vae_steps",
        "--outlier_residual_vae_lr",
        "--outlier_residual_vae_codebook_bits",
        "--outlier_residual_vae_codebook_dim",
        "--outlier_residual_vae_batch_multiplier",
        "--outlier_residual_vae_decoder_share_scope",
    }
    sparse_args = {
        "--sparse_residual_ratio",
        "--outlier_residual_min_abs",
        "--outlier_residual_codec",
        "--outlier_residual_index_bits",
        "--outlier_residual_value_bits",
        "--outlier_residual_block_shape",
    }

    if mode == "none":
        disallowed = (channel_args | sparse_args | {"--outlier_rank_metric"}) & provided
        if disallowed:
            raise ValueError("--outlier_protect_mode=none does not allow: " + ",".join(sorted(disallowed)))
        return

    if metric is None:
        raise ValueError(f"--outlier_protect_mode={mode} requires --outlier_rank_metric.")

    if mode == "residual_sparse":
        if metric not in _SPARSE_METRICS:
            raise ValueError(
                "--outlier_protect_mode=residual_sparse requires a sparse_* --outlier_rank_metric, "
                f"got {metric!r}."
            )
        if args.sparse_residual_ratio is None or not (0.0 < float(args.sparse_residual_ratio) <= 1.0):
            raise ValueError("--sparse_residual_ratio must satisfy 0 < ratio <= 1 in residual_sparse mode.")
        disallowed = channel_args & provided
        if disallowed:
            raise ValueError(
                "--outlier_protect_mode=residual_sparse does not allow: " + ",".join(sorted(disallowed))
            )
        return

    if mode == "channel_residual_vae":
        if metric not in _CHANNEL_METRICS:
            raise ValueError(
                "--outlier_protect_mode=channel_residual_vae requires a channel_* --outlier_rank_metric, "
                f"got {metric!r}."
            )
        if args.outlier_protect_axis not in {"input", "output"}:
            raise ValueError("--outlier_protect_axis input/output is required in channel_residual_vae mode.")
        if args.outlier_channel_scope not in {"layer", "category"}:
            raise ValueError("--outlier_channel_scope layer/category is required in channel_residual_vae mode.")
        if args.outlier_protect_count is None or int(args.outlier_protect_count) <= 0:
            raise ValueError("--outlier_protect_count must be > 0 in channel_residual_vae mode.")
        if int(args.outlier_protect_min_per_layer) < 0:
            raise ValueError("--outlier_protect_min_per_layer must be >= 0.")
        if int(args.outlier_protect_min_per_layer) > int(args.outlier_protect_count):
            raise ValueError("--outlier_protect_min_per_layer must be <= --outlier_protect_count.")
        if args.outlier_residual_vae_steps is None or int(args.outlier_residual_vae_steps) <= 0:
            raise ValueError("--outlier_residual_vae_steps must be > 0 in channel_residual_vae mode.")
        if args.outlier_residual_vae_lr is None or float(args.outlier_residual_vae_lr) <= 0.0:
            raise ValueError("--outlier_residual_vae_lr must be > 0 in channel_residual_vae mode.")
        if (
            args.outlier_residual_vae_batch_multiplier is None
            or int(args.outlier_residual_vae_batch_multiplier) < 1
        ):
            raise ValueError("--outlier_residual_vae_batch_multiplier must be >= 1.")
        if args.outlier_residual_vae_decoder_share_scope not in {"none", "category"}:
            raise ValueError(
                "--outlier_residual_vae_decoder_share_scope none/category is required in channel_residual_vae mode."
            )
        disallowed = sparse_args & provided
        if disallowed:
            raise ValueError(
                "--outlier_protect_mode=channel_residual_vae does not allow: "
                + ",".join(sorted(disallowed))
            )
        return

    raise ValueError(f"Unsupported --outlier_protect_mode={args.outlier_protect_mode!r}.")


def _is_decoder_layer_projection(name: str, target_categories: Sequence[str]) -> bool:
    in_decoder_layers = (
        ".model.layers." in name
        or name.startswith("model.layers.")
        or ".model.decoder.layers." in name
        or name.startswith("model.decoder.layers.")
    )
    if not in_decoder_layers:
        return False
    return any(name.endswith(f".{category}") or name.endswith(category) for category in target_categories)


def _collect_residual_targets(
    model: nn.Module,
    *,
    target_categories: Sequence[str],
    transpose_modules: Sequence[str],
    include_all_linears: bool,
) -> Dict[str, List[_ResidualTarget]]:
    target_set = {str(v) for v in target_categories}
    transpose_set = {str(v) for v in transpose_modules}
    out: Dict[str, List[_ResidualTarget]] = {category: [] for category in target_categories}
    for name, module in model.named_modules():
        if not isinstance(module, VAELinear):
            continue
        category = name.rsplit(".", 1)[-1]
        if category not in target_set:
            continue
        if not include_all_linears and not _is_decoder_layer_projection(name, target_categories):
            continue
        out.setdefault(category, []).append(
            _ResidualTarget(
                name=str(name),
                category=str(category),
                module=module,
                transpose=category in transpose_set,
            )
        )
    for targets in out.values():
        targets.sort(key=lambda item: item.name)
    return out


def _collect_online_activation_stats(
    *,
    model: nn.Module,
    targets_by_category: Dict[str, List[_ResidualTarget]],
    target_categories: Sequence[str],
    args: argparse.Namespace,
    logger,
) -> Dict[str, Dict[str, torch.Tensor]]:
    metric = None if args.outlier_rank_metric is None else str(args.outlier_rank_metric).strip().lower()
    if not _metric_requires_activation(metric):
        logger.info(
            "[activation_stats] skipped because outlier_rank_metric=%s does not require activation stats",
            str(args.outlier_rank_metric),
        )
        return {}

    linear_items: List[Tuple[str, nn.Module]] = []
    for category in target_categories:
        for target in targets_by_category.get(category, []):
            linear_items.append((target.name, target.module))
    if not linear_items:
        raise ValueError("[activation_stats] no target linears found for activation stats collection.")

    dataset = str(args.activation_calib_dataset).strip()
    device = str(args.activation_calib_device).strip() or str(args.train_device)
    logger.info(
        "[activation_stats] collecting activation stats for residual-from-base "
        "dataset=%s nsamples=%d seqlen=%d seed=%d target_linears=%d",
        dataset,
        int(args.activation_calib_nsamples),
        int(args.activation_calib_seqlen),
        int(args.activation_calib_seed),
        int(len(linear_items)),
    )
    stats_by_linear, _cache = collect_activation_stats_for_linears(
        model=model,
        linear_items=linear_items,
        model_path=str(args.model_path),
        access_token=args.access_token,
        dataset=dataset,
        nsamples=int(args.activation_calib_nsamples),
        seqlen=int(args.activation_calib_seqlen),
        seed=int(args.activation_calib_seed),
        device=device,
        cache=None,
        log_every=int(args.activation_calib_log_every),
        logger=logger,
    )

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    has_max = False
    has_abs_mean = False
    has_sq_mean = False
    has_rms = False
    for name, raw_stats in stats_by_linear.items():
        fields: Dict[str, torch.Tensor] = {}
        max_tensor = raw_stats.get("max")
        if isinstance(max_tensor, torch.Tensor):
            fields["max"] = max_tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            has_max = True
        abs_mean_tensor = raw_stats.get("abs_mean")
        if isinstance(abs_mean_tensor, torch.Tensor):
            fields["abs_mean"] = abs_mean_tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            has_abs_mean = True
        sq_mean_tensor = raw_stats.get("sq_mean")
        if isinstance(sq_mean_tensor, torch.Tensor):
            fields["sq_mean"] = sq_mean_tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            has_sq_mean = True
        rms_tensor = raw_stats.get("rms")
        if isinstance(rms_tensor, torch.Tensor):
            fields["rms"] = rms_tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            has_rms = True
        out[str(name)] = fields

    if metric in _ACTMAX_METRICS and not has_max:
        raise ValueError(f"{metric} requires activation max stats from collect_activation_stats_for_linears.")
    if metric in _ACTMEAN_METRICS and not has_abs_mean:
        raise ValueError(f"{metric} requires activation mean stats from collect_activation_stats_for_linears.")
    if metric in _ACTRMS_METRICS and not (has_sq_mean or has_rms):
        raise ValueError(
            "channel_residual_actrms_abs requires activation second-moment stats from "
            "collect_activation_stats_for_linears, but they were not collected."
        )
    if metric in _ACTRMS_METRICS and not has_sq_mean:
        for fields in out.values():
            rms = fields.get("rms")
            if isinstance(rms, torch.Tensor):
                fields["sq_mean"] = rms.pow(2).contiguous()
        has_sq_mean = any(isinstance(fields.get("sq_mean"), torch.Tensor) for fields in out.values())
        if not has_sq_mean:
            raise ValueError(
                "channel_residual_actrms_abs requires activation second-moment stats from "
                "collect_activation_stats_for_linears, but they were not collected."
            )

    logger.info(
        "[activation_stats] collected stats for %d linears: has_max=%s has_abs_mean=%s has_sq_mean=%s has_rms=%s",
        int(len(out)),
        str(bool(has_max)),
        str(bool(has_abs_mean)),
        str(bool(has_sq_mean)),
        str(bool(has_rms)),
    )
    return out


def _iter_runtime_vae_targets(model: nn.Module) -> List[_RuntimeVAETarget]:
    targets: List[_RuntimeVAETarget] = []
    skip_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if isinstance(module, VAELinear):
            base_layer = module
        else:
            continue
        targets.append(
            _RuntimeVAETarget(
                name=str(name),
                category=str(name).rsplit(".", 1)[-1],
                module=module,
                base_layer=base_layer,
            )
        )
    return targets


def _get_or_create_residual_reference_linear(
    *,
    reference_model: nn.Module,
    name: str,
    residency: _ResidualFromBaseResidency,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Linear:
    existing = residency.reference_dense_linears.get(name)
    if existing is not None:
        existing.to(device=device, dtype=dtype)
        existing.requires_grad_(False)
        existing.eval()
        return existing
    clone = clone_frozen_linear_from_reference(
        reference_model,
        name,
        device=device,
        dtype=dtype,
    )
    residency.reference_dense_linears[name] = clone
    return clone


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    return get_reference_module(model, name)


def _apply_residual_from_base_residency(
    *,
    model: nn.Module,
    reference_model: nn.Module,
    residency: _ResidualFromBaseResidency,
    active_categories: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    logger,
) -> List[NamedVAELinearTarget]:
    active_set = {str(category) for category in active_categories}
    prewarm_targets: List[NamedVAELinearTarget] = []
    inactive_reference = 0
    active_vae = 0
    managed_names = {
        target.name: target.category
        for target in _iter_runtime_vae_targets(model)
    }
    managed_names.update({name: str(name).rsplit(".", 1)[-1] for name in residency.stashed_vae_modules})
    managed_names.update({name: str(name).rsplit(".", 1)[-1] for name in residency.reference_dense_linears})

    for name, category in sorted(managed_names.items()):
        module = _get_module_by_name(model, name)
        if str(category) in active_set:
            if isinstance(module, nn.Linear):
                if name not in residency.stashed_vae_modules:
                    raise RuntimeError(f"{name}: active residual-from-base category missing stashed VAELinear.")
                vae_module = residency.stashed_vae_modules.pop(name)
                set_module_by_name(model, name, vae_module)
                module = vae_module
            if not isinstance(module, VAELinear):
                raise TypeError(f"{name}: expected active VAELinear, got {type(module)}.")
            base_layer = module
            base_layer.clear_decoded_weight_cache()
            module.to(device)
            prewarm_targets.append(NamedVAELinearTarget(name=name, base_layer=base_layer))
            active_vae += 1
        else:
            if isinstance(module, VAELinear):
                base_layer = module
                base_layer.clear_decoded_weight_cache()
                module.to("cpu")
                residency.stashed_vae_modules[name] = module
            elif isinstance(module, nn.Linear):
                reference = _get_or_create_residual_reference_linear(
                    reference_model=reference_model,
                    name=name,
                    residency=residency,
                    device=device,
                    dtype=dtype,
                )
                if module is not reference:
                    raise TypeError(f"{name}: live nn.Linear is not the managed reference clone.")
                inactive_reference += 1
                continue
            else:
                raise TypeError(f"{name}: expected inactive VAELinear or reference nn.Linear, got {type(module)}.")
            reference = _get_or_create_residual_reference_linear(
                reference_model=reference_model,
                name=name,
                residency=residency,
                device=device,
                dtype=dtype,
            )
            set_module_by_name(model, name, reference)
            inactive_reference += 1
    logger.info(
        "residual-from-base active categories=%s active_vae=%d inactive_reference_linear=%d stashed_vae=%d reference_clone_cache_size=%d",
        ",".join(str(category) for category in active_categories),
        int(active_vae),
        int(inactive_reference),
        int(len(residency.stashed_vae_modules)),
        int(len(residency.reference_dense_linears)),
    )
    return prewarm_targets


def _restore_all_residual_from_base_vae(
    *,
    model: nn.Module,
    residency: _ResidualFromBaseResidency,
    logger,
) -> None:
    restored = 0
    for name, module in list(residency.stashed_vae_modules.items()):
        base_layer = module
        base_layer.clear_decoded_weight_cache()
        set_module_by_name(model, name, module)
        del residency.stashed_vae_modules[name]
        restored += 1
    for reference in residency.reference_dense_linears.values():
        reference.to("cpu")
    cleared = clear_model_vae_linear_cache(model)
    logger.info(
        "residual-from-base save state: restored all VAELinear targets; restored=%d cleared_caches=%d",
        int(restored),
        int(cleared),
    )


def _prewarm_active_residual_categories(
    *,
    category: str,
    model: nn.Module,
    reference_model: nn.Module,
    residency: _ResidualFromBaseResidency,
    active_categories: Sequence[str],
    args: argparse.Namespace,
    logger,
    stage: str,
) -> Dict[str, int]:
    prewarm_targets = _apply_residual_from_base_residency(
        model=model,
        reference_model=reference_model,
        residency=residency,
        active_categories=active_categories,
        device=torch.device(str(args.train_device)),
        dtype=_resolve_train_dtype(args),
        logger=logger,
    )
    stats = prime_named_vae_linear_cache(
        prewarm_targets,
        clear_existing=True,
        group_size=8,
        compute_device=str(args.train_device),
        logger=logger,
    )
    logger.info(
        "[%s] %s active_categories=%s prewarm total=%d warmed=%d skipped=%d failed=%d",
        category,
        stage,
        ",".join(str(item) for item in active_categories),
        int(stats.get("total", 0)),
        int(stats.get("warmed", 0)),
        int(stats.get("skipped", 0)),
        int(stats.get("failed", 0)),
    )
    return stats


def _maybe_eval_residual_stage(
    *,
    model: nn.Module,
    vae_args: argparse.Namespace,
    args: argparse.Namespace,
    category: str,
    stage: str,
    active_categories: Sequence[str],
    eval_tasks_text: str,
    eval_tokenizer: Optional[object],
    run_any_eval: bool,
    logger,
    eval_results: Dict[str, Dict[str, object]],
) -> None:
    if not run_any_eval:
        return
    enabled = (
        bool(args.eval_before_residual)
        if stage == "before_residual"
        else bool(args.eval_after_residual)
    )
    if not enabled:
        return

    key = f"{category}/{stage}"
    if stage == "before_residual":
        logger.info(
            "[eval] category=%s stage=before_residual active_categories=%s "
            "mode=cumulative_active_vae_before_current_residual",
            category,
            ",".join(str(item) for item in active_categories),
        )
        logger.info(
            "[%s] residual VAE 前评估：base VAE prewarmed, residual protection not attached",
            category,
        )
    else:
        logger.info(
            "[eval] category=%s stage=after_residual active_categories=%s "
            "mode=cumulative_active_vae_after_current_residual",
            category,
            ",".join(str(item) for item in active_categories),
        )
        logger.info(
            "[%s] residual VAE 后评估：residual protection attached and prewarmed",
            category,
        )

    _eval_after_category(
        model=model,
        vae_args=vae_args,
        ppl_limit=int(args.ppl_limit),
        category=key,
        logger=logger,
        eval_device=str(args.train_device),
        eval_hif4_act=bool(args.eval_hif4_act),
        eval_ppl=bool(args.eval_ppl),
        eval_tasks=eval_tasks_text,
        tokenizer=eval_tokenizer,
        run_output_dir=str(args.output_dir),
    )
    eval_results[key] = {
        "ran": True,
        "eval_ppl": bool(args.eval_ppl),
        "eval_tasks": eval_tasks_text,
        "active_categories": [str(item) for item in active_categories],
    }


def _has_existing_residual_protection(module: VAELinear) -> bool:
    return bool(module.has_sparse_residual() or module.has_protected_residual_vae())


def _clear_sparse_residual_payload(module: VAELinear) -> None:
    module.sparse_residual_format = SPARSE_RESIDUAL_FORMAT_COO_FP16
    module.sparse_residual_index_bits = None
    module.sparse_residual_value_bits = None
    module.sparse_residual_block_rows = None
    module.sparse_residual_block_cols = None
    for name in (
        "sparse_residual_row_indices",
        "sparse_residual_col_indices",
        "sparse_residual_values",
        "sparse_residual_active_block_ids",
        "sparse_residual_block_ptr",
        "sparse_residual_local_indices",
        "sparse_residual_qvalues",
        "sparse_residual_scales",
        "sparse_residual_zero_points",
    ):
        setattr(module, name, None)
    module.clear_sparse_residual_cache()


def _delete_module_attr(module: nn.Module, name: str) -> None:
    if name in module._modules:
        del module._modules[name]
    elif hasattr(module, name):
        delattr(module, name)


def _clear_protected_residual_payload(module: VAELinear) -> None:
    old_stages = int(getattr(module, "protected_residual_stages", 0) or 0)
    module.protected_residual_axis = None
    module.protected_residual_stages = 0
    module.protected_residual_stage_codebook_dims = []
    module._protected_residual_stage_vq_storage_specs = []
    module.protected_residual_shared_decoder_refs = None
    module.__dict__["_protected_residual_shared_stage_decoders"] = None
    module.protected_residual_indices = None
    for stage_idx in range(old_stages):
        if stage_idx == 0:
            if "protected_residual_vq_weight" in module._buffers:
                module._buffers["protected_residual_vq_weight"] = None
            _delete_module_attr(module, "protected_residual_decoder")
        else:
            buffer_name = f"protected_residual_vq_weight_s{stage_idx}"
            if buffer_name in module._buffers:
                module._buffers[buffer_name] = None
            decoder_name = f"protected_residual_decoder_s{stage_idx}"
            _delete_module_attr(module, decoder_name)
    packed_decoder = getattr(module, "_protected_residual_parallel_decoder", None)
    if packed_decoder is not None:
        _delete_module_attr(module, "_protected_residual_parallel_decoder")
    module.protected_residual_parallel_stage_decode = False
    module._protected_residual_parallel_layout = []
    module._clear_protected_residual_parallel_plan()


def _clear_existing_residual_protection(module: VAELinear) -> None:
    _clear_sparse_residual_payload(module)
    _clear_protected_residual_payload(module)
    module.clear_decoded_weight_cache()


def _decode_base_reconstruction(target: _ResidualTarget) -> torch.Tensor:
    reconstructed = target.module._decode_weight(
        dtype=torch.float32,
        include_protected_residual=False,
        include_low_rank=False,
        include_sparse_residual=False,
    ).detach().to(device="cpu", dtype=torch.float32).contiguous()
    expected = (int(target.module.out_features), int(target.module.in_features))
    if tuple(reconstructed.shape) != expected:
        raise ValueError(
            f"{target.name}: base VAE reconstruction shape mismatch, got {tuple(reconstructed.shape)}, "
            f"expected {expected}."
        )
    return reconstructed


def _reference_weight(reference_model: nn.Module, target: _ResidualTarget) -> torch.Tensor:
    reference_module = get_reference_module(reference_model, target.name)
    if not isinstance(reference_module, nn.Linear):
        raise TypeError(f"{target.name}: reference module is not nn.Linear, got {type(reference_module)}.")
    original = reference_module.weight
    expected = (int(target.module.out_features), int(target.module.in_features))
    if tuple(original.shape) != expected:
        raise ValueError(
            f"{target.name}: reference weight shape mismatch, got {tuple(original.shape)}, expected {expected}."
        )
    return original.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _select_channel_plan(
    *,
    reference_model: nn.Module,
    targets: Sequence[_ResidualTarget],
    residual_by_name: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    act_stats: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    axis = str(args.outlier_protect_axis)
    metric = str(args.outlier_rank_metric)
    scores_by_name: Dict[str, torch.Tensor] = {}
    for target in targets:
        stats = act_stats.get(target.name, {})
        scores_by_name[target.name] = compute_channel_rank_score(
            metric=metric,
            weight=_reference_weight(reference_model, target),
            residual=residual_by_name[target.name],
            act_max=stats.get("max"),
            act_mean=stats.get("abs_mean"),
            act_sq_mean=stats.get("sq_mean"),
            axis=axis,
            transpose=bool(target.transpose),
            linear_name=target.name,
            expected_in_features=int(target.module.in_features),
            expected_out_features=int(target.module.out_features),
        )

    plan, selection_stats = select_outlier_channel_indices_from_scores(
        scores_by_name=scores_by_name,
        linear_names=[target.name for target in targets],
        channel_protect_count=int(args.outlier_protect_count),
        channel_min_per_layer=int(args.outlier_protect_min_per_layer),
        channel_scope=str(args.outlier_channel_scope),
    )

    score_values = torch.cat([score.reshape(-1).to(dtype=torch.float32) for score in scores_by_name.values()])
    return plan, {
        "num_channels": float(int(score_values.numel())),
        "topk": float(sum(int(idx.numel()) for idx in plan.values())),
        "score_max": float(score_values.max().item()) if int(score_values.numel()) else 0.0,
        "score_mean": float(score_values.mean().item()) if int(score_values.numel()) else 0.0,
        "min_per_layer": float(selection_stats["min_per_layer"]),
        "floor_selected_count": float(selection_stats["floor_selected_count"]),
        "global_selected_count": float(selection_stats["global_selected_count"]),
        "num_zero_protected_linears": float(selection_stats["num_zero_protected_linears"]),
    }


def _build_runtime_cfg(category: str, args: argparse.Namespace, *, inferred_codebook_dim: int) -> _ResidualVAERuntimeConfig:
    codebook_dim = int(args.codebook_dim) if int(args.codebook_dim) > 0 else int(inferred_codebook_dim)
    codebook_bits = int(args.codebook_bits)
    residual_codebook_bits = (
        int(args.outlier_residual_vae_codebook_bits)
        if int(args.outlier_residual_vae_codebook_bits) > 0
        else int(codebook_bits)
    )
    residual_codebook_dim = (
        int(args.outlier_residual_vae_codebook_dim)
        if int(args.outlier_residual_vae_codebook_dim) > 0
        else int(codebook_dim)
    )
    return _ResidualVAERuntimeConfig(
        category=str(category),
        residual_stages=1,
        steps=int(args.outlier_residual_vae_steps or 1),
        intra_part_sort_mode="none",
        codebook_bits=int(codebook_bits),
        codebook_dim=int(codebook_dim),
        outlier_protect_count=int(args.outlier_protect_count or 0),
        outlier_residual_top_p=0.0,
        outlier_residual_vae_stages=int(args.outlier_residual_vae_stages),
        outlier_residual_vae_codebook_bits=int(residual_codebook_bits),
        outlier_residual_vae_codebook_dim=int(residual_codebook_dim),
        recon_loss_type=str(args.recon_loss_type).strip().lower(),
        base_ch=int(args.base_ch),
        num_res_blocks=int(args.num_res_blocks),
        decoder_base_ch=int(args.decoder_base_ch) if args.decoder_base_ch is not None else None,
        decoder_num_res_blocks=(
            int(args.decoder_num_res_blocks) if args.decoder_num_res_blocks is not None else None
        ),
        norm_type=str(args.norm_type).strip().lower(),
        activation_type=str(args.activation_type).strip().lower(),
        decoder_type=str(args.decoder_type).strip().lower(),
    )


def _install_protected_residual_payload(
    *,
    model: nn.Module,
    target: _ResidualTarget,
    payload: Dict[str, object],
) -> None:
    module = target.module
    shared_refs = payload.get("shared_decoder_refs")
    shared_decoders = payload.get("shared_stage_decoders")
    if shared_refs is not None or shared_decoders is not None:
        if not isinstance(shared_refs, (list, tuple)) or not isinstance(shared_decoders, (list, tuple)):
            raise TypeError(f"{target.name}: invalid shared protected residual payload.")
        if len(shared_refs) != len(shared_decoders):
            raise ValueError(
                f"{target.name}: shared protected residual decoder ref/object mismatch: "
                f"{len(shared_refs)} vs {len(shared_decoders)}."
            )
        for ref, decoder in zip(shared_refs, shared_decoders):
            register_shared_protected_residual_decoder(model, str(ref), decoder)
    module._init_protected_residual_payload(
        axis=payload["axis"],
        indices=payload["indices"],
        stage_vq_weights=payload["stage_vq_weights"],
        stage_vq_storage_specs=None,
        stage_decoders=payload.get("stage_decoders"),
        shared_decoder_refs=shared_refs,
        shared_stage_decoders=shared_decoders,
        stage_codebook_dims=payload["stage_codebook_dims"],
    )
    module.clear_decoded_weight_cache()


def _install_sparse_residual_payload(module: VAELinear, payload: Optional[Dict[str, object]]) -> None:
    _clear_sparse_residual_payload(module)
    if payload is None:
        return
    for key, value in payload.items():
        setattr(module, key, value)
    module.clear_decoded_weight_cache()


def _train_protected_residual_vae_payload(
    *,
    linear_name: str,
    residual_slice: torch.Tensor,
    runtime_cfg: _ResidualVAERuntimeConfig,
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
        activation_type=str(runtime_cfg.activation_type),
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

        # 训完立刻释放 Adam 状态，降低随后全量重构的显存尖峰。
        del optimizer
        optimizer = None
        if lr_scheduler is not None:
            del lr_scheduler
            lr_scheduler = None
        torch.cuda.empty_cache()

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
        del vae, train_loader, eval_loader
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
    runtime_cfg: _ResidualVAERuntimeConfig,
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




def _process_channel_residual_vae_category(
    *,
    model: nn.Module,
    reference_model: nn.Module,
    category: str,
    targets: Sequence[_ResidualTarget],
    residual_by_name: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    vae_args: argparse.Namespace,
    training_args: argparse.Namespace,
    act_stats: Dict[str, Dict[str, torch.Tensor]],
    logger,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    runtime_cfg = _build_runtime_cfg(category, args, inferred_codebook_dim=int(targets[0].module.codebook_dim))
    train_dtype = _resolve_train_dtype(training_args)
    batch_size = int(args.base_batch_size) * int(args.outlier_residual_vae_batch_multiplier)
    logger.info(
        "[%s/channel_residual_vae] base_codebook_bits=%d base_codebook_dim=%d residual_vae_codebook_bits=%d residual_vae_codebook_dim=%d",
        category,
        int(runtime_cfg.codebook_bits),
        int(runtime_cfg.codebook_dim),
        int(runtime_cfg.outlier_residual_vae_codebook_bits),
        int(runtime_cfg.outlier_residual_vae_codebook_dim),
    )
    plan, score_summary = _select_channel_plan(
        reference_model=reference_model,
        targets=targets,
        residual_by_name=residual_by_name,
        args=args,
        act_stats=act_stats,
    )
    protected_entries = []
    for target in targets:
        idx = plan[target.name].detach().to(device="cpu", dtype=torch.long).contiguous()
        if int(idx.numel()) == 0:
            continue
        residual = residual_by_name[target.name]
        axis_dim = 1 if str(args.outlier_protect_axis) == "input" else 0
        axis_size = int(residual.shape[axis_dim])
        if int(idx.min().item()) < 0 or int(idx.max().item()) >= axis_size:
            raise ValueError(
                f"[{category}] {target.name}: protected index out of bounds: "
                f"axis={args.outlier_protect_axis} axis_size={axis_size} idx_size={int(idx.numel())} "
                f"min={int(idx.min().item())} max={int(idx.max().item())} tensor_shape={tuple(residual.shape)}."
            )
        residual_slice = residual.index_select(axis_dim, idx).contiguous()
        if int(residual_slice.numel()) % int(runtime_cfg.outlier_residual_vae_codebook_dim) != 0:
            raise ValueError(
                f"[{category}] {target.name}: protected residual slice shape={tuple(residual_slice.shape)} "
                f"is not divisible by residual_vae_codebook_dim="
                f"{int(runtime_cfg.outlier_residual_vae_codebook_dim)}."
            )
        protected_entries.append((target, idx, residual_slice))
        logger.info(
            "[%s/channel_residual_vae] axis=%s channels=%d residual_rms=%.6e",
            target.name,
            str(args.outlier_protect_axis),
            int(idx.numel()),
            float(residual_slice.float().pow(2).mean().sqrt().item()),
        )

    payload_metrics = []
    if str(args.outlier_residual_vae_decoder_share_scope) == "category":
        shared_by_name = _train_shared_protected_residual_vae_payloads(
            group_tag=category,
            residual_slices_by_name={target.name: residual_slice for target, _idx, residual_slice in protected_entries},
            runtime_cfg=runtime_cfg,
            vae_args=vae_args,
            training_args=training_args,
            train_device=str(args.train_device),
            train_dtype=train_dtype,
            batch_size=batch_size,
            steps=int(args.outlier_residual_vae_steps),
            lr=float(args.outlier_residual_vae_lr),
            log_every=int(args.log_every),
            deterministic=bool(args.deterministic),
            shuffle_seed=int(args.seed),
        )
    else:
        shared_by_name = {}

    for target, idx, residual_slice in protected_entries:
        if str(args.outlier_residual_vae_decoder_share_scope) == "category":
            payload = shared_by_name.get(target.name)
        else:
            payload = _train_protected_residual_vae_payload(
                linear_name=target.name,
                residual_slice=residual_slice,
                runtime_cfg=runtime_cfg,
                vae_args=vae_args,
                training_args=training_args,
                train_device=str(args.train_device),
                train_dtype=train_dtype,
                batch_size=batch_size,
                steps=int(args.outlier_residual_vae_steps),
                lr=float(args.outlier_residual_vae_lr),
                log_every=int(args.log_every),
                deterministic=bool(args.deterministic),
                shuffle_seed=int(args.seed),
            )
        if payload is None:
            continue
        install_payload = {
            "axis": str(args.outlier_protect_axis),
            "indices": idx,
            "stage_vq_weights": payload["stage_vq_weights"],
            "stage_codebook_dims": payload["stage_codebook_dims"],
        }
        if str(args.outlier_residual_vae_decoder_share_scope) == "category":
            install_payload["shared_decoder_refs"] = payload["shared_decoder_refs"]
            install_payload["shared_stage_decoders"] = payload["shared_stage_decoders"]
        else:
            install_payload["stage_decoders"] = payload["stage_decoders"]
        _install_protected_residual_payload(model=model, target=target, payload=install_payload)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            payload_metrics.append(metrics)

    protected_numel = sum(int(residual_slice.numel()) for _target, _idx, residual_slice in protected_entries)
    protected_before = 0.0
    if protected_numel > 0:
        protected_sq = sum(float(residual_slice.float().pow(2).sum().item()) for _target, _idx, residual_slice in protected_entries)
        protected_before = float((protected_sq / float(protected_numel)) ** 0.5)
    final_losses = [m.get("residual_vae_final_loss") for m in payload_metrics if m.get("residual_vae_final_loss") is not None]
    final_recons = [m.get("residual_vae_final_recon") for m in payload_metrics if m.get("residual_vae_final_recon") is not None]
    final_commits = [m.get("residual_vae_final_commit") for m in payload_metrics if m.get("residual_vae_final_commit") is not None]
    final_after = [m.get("protected_residual_rms_after") for m in payload_metrics if m.get("protected_residual_rms_after") is not None]

    summary = {
        "category": category,
        "num_linears": len(targets),
        "mode": "channel_residual_vae",
        "axis": str(args.outlier_protect_axis),
        "scope": str(args.outlier_channel_scope),
        "protect_count": int(args.outlier_protect_count),
        "protect_min_per_layer": int(args.outlier_protect_min_per_layer),
        "actual_protected_channels_per_layer": {
            target.name: int(plan[target.name].numel()) for target in targets
        },
        "residual_vae_share_scope": str(args.outlier_residual_vae_decoder_share_scope),
        "residual_vae_steps": int(args.outlier_residual_vae_steps),
        "residual_vae_lr": float(args.outlier_residual_vae_lr),
        "residual_vae_batch_size": int(batch_size),
        "base_codebook_bits": int(runtime_cfg.codebook_bits),
        "base_codebook_dim": int(runtime_cfg.codebook_dim),
        "residual_vae_codebook_bits": int(runtime_cfg.outlier_residual_vae_codebook_bits),
        "residual_vae_codebook_dim": int(runtime_cfg.outlier_residual_vae_codebook_dim),
    }
    metrics = {
        "num_protected_channels": int(sum(int(plan[target.name].numel()) for target in targets)),
        "residual_vae_codebook_bits": int(runtime_cfg.outlier_residual_vae_codebook_bits),
        "residual_vae_codebook_dim": int(runtime_cfg.outlier_residual_vae_codebook_dim),
        "protected_residual_rms_before": protected_before,
        "protected_residual_rms_after": (
            float(sum(float(v) for v in final_after) / len(final_after)) if final_after else None
        ),
        "residual_vae_final_loss": (
            float(sum(float(v) for v in final_losses) / len(final_losses)) if final_losses else None
        ),
        "residual_vae_final_recon": (
            float(sum(float(v) for v in final_recons) / len(final_recons)) if final_recons else None
        ),
        "residual_vae_final_commit": (
            float(sum(float(v) for v in final_commits) / len(final_commits)) if final_commits else None
        ),
        "score_summary": score_summary,
    }
    return summary, metrics


def _process_sparse_category(
    *,
    reference_model: nn.Module,
    category: str,
    targets: Sequence[_ResidualTarget],
    reconstructed_by_name: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    act_stats: Dict[str, Dict[str, torch.Tensor]],
    logger,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total_nnz = 0
    total_codec_bytes = 0
    total_coo_bytes = 0
    for target in targets:
        activation_weight = None
        activation_mean = None
        if str(args.outlier_rank_metric) in RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMAX:
            activation_weight = act_stats.get(target.name, {}).get("max")
            if activation_weight is None:
                raise ValueError(f"{target.name}: {args.outlier_rank_metric} requires act max stats.")
        if str(args.outlier_rank_metric) in RESIDUAL_SPARSE_RANK_METRICS_NEED_ACTMEAN:
            activation_mean = act_stats.get(target.name, {}).get("abs_mean")
            if activation_mean is None:
                raise ValueError(f"{target.name}: {args.outlier_rank_metric} requires act mean stats.")
        block_shape = args.outlier_residual_block_shape
        if block_shape is None:
            block_shape = get_default_block_shape_for_index_bits(int(args.outlier_residual_index_bits))
        payload, nnz, storage = build_sparse_residual_payload(
            linear_name=target.name,
            target_weight=_reference_weight(reference_model, target),
            reconstructed_weight=reconstructed_by_name[target.name],
            activation_weight=activation_weight,
            activation_mean=activation_mean,
            rank_metric=str(args.outlier_rank_metric),
            top_p=float(args.sparse_residual_ratio),
            min_abs=float(args.outlier_residual_min_abs),
            codec=str(args.outlier_residual_codec),
            index_bits=int(args.outlier_residual_index_bits),
            value_bits=int(args.outlier_residual_value_bits),
            block_shape=tuple(int(v) for v in block_shape),
        )
        _install_sparse_residual_payload(target.module, payload)
        total_nnz += int(nnz)
        total_codec_bytes += int(storage["codec_bytes"])
        total_coo_bytes += int(storage["coo_bytes"])
        logger.info(
            "[%s/residual_sparse] nnz=%d ratio=%.6f metric=%s codec=%s bytes(codec=%d coo=%d)",
            target.name,
            int(nnz),
            float(args.sparse_residual_ratio),
            str(args.outlier_rank_metric),
            str(args.outlier_residual_codec),
            int(storage["codec_bytes"]),
            int(storage["coo_bytes"]),
        )
    summary = {
        "category": category,
        "num_linears": len(targets),
        "mode": "residual_sparse",
        "sparse_residual_ratio": float(args.sparse_residual_ratio),
        "rank_metric": str(args.outlier_rank_metric),
        "sparse_nnz": int(total_nnz),
        "sparse_codec_bytes": int(total_codec_bytes),
        "sparse_coo_bytes": int(total_coo_bytes),
    }
    return summary, dict(summary)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _prepare_residual_from_base_output_dir(args: argparse.Namespace) -> Tuple[str, str]:
    root_output_dir = str(args.output_dir)
    if bool(args.overwrite):
        os.makedirs(root_output_dir, exist_ok=True)
        return root_output_dir, root_output_dir

    os.makedirs(root_output_dir, exist_ok=True)
    run_output_dir = _build_run_output_dir(root_output_dir, str(args.model_path))
    args.output_dir = run_output_dir
    return root_output_dir, run_output_dir


def _prepare_residual_from_base_vae_args(args: argparse.Namespace) -> argparse.Namespace:
    vae_args = argparse.Namespace(**vars(args))
    if bool(args.bf16):
        vae_weight_dtype = "bf16"
    elif bool(args.fp16):
        vae_weight_dtype = "fp16"
    else:
        vae_weight_dtype = "fp32"
    vae_args.vae_weight_dtype = vae_weight_dtype
    vae_args.vae_autocast_dtype = vae_weight_dtype
    return vae_args


def run_residual_from_base(args: argparse.Namespace) -> None:
    configure_deterministic_mode(bool(args.deterministic))
    set_seed(int(args.seed))

    root_output_dir, run_output_dir = _prepare_residual_from_base_output_dir(args)
    os.environ["LOG_FILE"] = os.path.join(args.output_dir, "residual_from_base.log")
    logger = get_logger("residual_from_base")

    target_categories = split_csv(args.target_categories)
    if not target_categories:
        raise ValueError("--target_categories cannot be empty.")
    transpose_modules = split_csv(args.transpose_modules)

    logger.info("model_path=%s", str(args.model_path))
    logger.info("base_vae_checkpoint=%s", str(args.base_vae_checkpoint))
    logger.info("output_dir=%s", str(args.output_dir))
    logger.info("root_output_dir=%s", str(root_output_dir))
    logger.info("run_output_dir=%s", str(run_output_dir))
    logger.info("target_categories=%s", ",".join(target_categories))
    logger.info("outlier_protect_mode=%s", str(args.outlier_protect_mode))
    logger.info("outlier_rank_metric=%s", str(args.outlier_rank_metric))
    eval_tasks_text = str(getattr(args, "eval_tasks", "")).strip()
    run_task_eval = bool(eval_tasks_text)
    run_any_eval = bool(args.eval_ppl) or run_task_eval
    if not run_any_eval:
        logger.info("跳过 residual-from-base 评估：--eval_ppl=false 且 --eval_tasks 为空。")

    checkpoint_dir = resolve_v6_checkpoint_dir(str(args.base_vae_checkpoint))
    planned_checkpoint_out = os.path.join(args.output_dir, "checkpoint")
    if os.path.abspath(planned_checkpoint_out) == os.path.abspath(checkpoint_dir):
        raise ValueError(
            "--output_dir/checkpoint resolves to the same directory as --base_vae_checkpoint; "
            "choose a separate output directory."
        )
    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    with open(meta_path, "r", encoding="utf-8") as handle:
        base_meta = json.load(handle)

    model, load_meta, load_result = load_v6_model_checkpoint(
        checkpoint_dir,
        access_token=args.access_token,
        base_model_path=str(args.model_path),
        map_location="cpu",
        strict=True,
    )
    stripped = normalize_cat_runtime_vae_original_state(model)
    reference_model = load_frozen_base_reference_model(
        str(args.model_path),
        access_token=args.access_token,
        device="cpu",
        dtype=_resolve_train_dtype(args),
    )
    logger.info(
        "Loaded base checkpoint: dir=%s missing_keys=%d unexpected_keys=%d converted_modules=%s stripped_original_weight=%d",
        checkpoint_dir,
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
        str(load_meta.get("converted_module_count")),
        int(stripped),
    )

    targets_by_category = _collect_residual_targets(
        model,
        target_categories=target_categories,
        transpose_modules=transpose_modules,
        include_all_linears=bool(args.include_all_linears),
    )
    missing = [category for category in target_categories if not targets_by_category.get(category)]
    if missing:
        raise ValueError("target_categories contains categories without VAELinear in checkpoint: " + ",".join(missing))

    existing = [
        target.name
        for category in target_categories
        for target in targets_by_category[category]
        if _has_existing_residual_protection(target.module)
    ]
    if existing and not bool(args.replace_existing_residual_protection):
        raise ValueError(
            "Base checkpoint already contains residual protection payload for: "
            + ",".join(existing[:20])
            + ("..." if len(existing) > 20 else "")
            + ". Pass --replace_existing_residual_protection to rebuild it."
        )
    if existing:
        registry = get_shared_protected_residual_decoder_registry(model)
        registry.clear()
        for category in target_categories:
            for target in targets_by_category[category]:
                _clear_existing_residual_protection(target.module)

    eval_tokenizer = None
    if run_task_eval:
        from transformers import AutoTokenizer

        logger.info("加载 residual-from-base 下游任务评估 tokenizer: %s", str(args.model_path))
        eval_tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_path),
            use_fast=True,
            token=args.access_token,
        )

    vae_args = _prepare_residual_from_base_vae_args(args)
    training_args = SimpleNamespace(bf16=bool(args.bf16), fp16=bool(args.fp16))
    logger.info(
        "residual VAE dtype: train_dtype=%s vae_weight_dtype=%s vae_autocast_dtype=%s",
        str(_resolve_train_dtype(training_args)).replace("torch.", ""),
        str(vae_args.vae_weight_dtype),
        str(vae_args.vae_autocast_dtype),
    )
    run_start = time.time()
    payload_summary: List[Dict[str, Any]] = []
    category_metrics: Dict[str, Dict[str, Any]] = {}
    eval_results: Dict[str, Dict[str, object]] = {}
    active_categories: List[str] = []
    residency = _ResidualFromBaseResidency()
    needs_activation_stats = _metric_requires_activation(args.outlier_rank_metric)

    for category in target_categories:
        active_categories.append(category)
        targets = targets_by_category[category]
        _prewarm_active_residual_categories(
            category=category,
            model=model,
            reference_model=reference_model,
            residency=residency,
            active_categories=active_categories,
            args=args,
            logger=logger,
            stage="before_residual",
        )
        if run_any_eval and bool(args.eval_before_residual):
            _maybe_eval_residual_stage(
                model=model,
                vae_args=vae_args,
                args=args,
                category=category,
                stage="before_residual",
                active_categories=active_categories,
                eval_tasks_text=eval_tasks_text,
                eval_tokenizer=eval_tokenizer,
                run_any_eval=run_any_eval,
                logger=logger,
                eval_results=eval_results,
            )
        if needs_activation_stats:
            act_stats = _collect_online_activation_stats(
                model=model,
                targets_by_category={category: targets},
                target_categories=[category],
                args=args,
                logger=logger,
            )
        else:
            act_stats = {}

        reconstructed_by_name: Dict[str, torch.Tensor] = {}
        residual_by_name: Dict[str, torch.Tensor] = {}
        residual_numel = 0
        residual_sq_sum = 0.0
        for target in targets:
            reconstructed = _decode_base_reconstruction(target)
            original = _reference_weight(reference_model, target)
            residual = (original - reconstructed).contiguous()
            reconstructed_by_name[target.name] = reconstructed
            residual_by_name[target.name] = residual
            residual_numel += int(residual.numel())
            residual_sq_sum += float(residual.pow(2).sum().item())
        residual_rms = float((residual_sq_sum / float(residual_numel)) ** 0.5) if residual_numel else 0.0
        logger.info(
            "[%s] category=%s num_linears=%d final_residual_rms=%.6e metric=%s",
            category,
            category,
            int(len(targets)),
            float(residual_rms),
            str(args.outlier_rank_metric),
        )

        if str(args.outlier_protect_mode) == "channel_residual_vae":
            logger.info(
                "[%s/channel_residual_vae] using final residual decoded from base VAE checkpoint",
                category,
            )
            summary, metrics = _process_channel_residual_vae_category(
                model=model,
                reference_model=reference_model,
                category=category,
                targets=targets,
                residual_by_name=residual_by_name,
                args=args,
                vae_args=vae_args,
                training_args=training_args,
                act_stats=act_stats,
                logger=logger,
            )
        elif str(args.outlier_protect_mode) == "residual_sparse":
            summary, metrics = _process_sparse_category(
                reference_model=reference_model,
                category=category,
                targets=targets,
                reconstructed_by_name=reconstructed_by_name,
                args=args,
                act_stats=act_stats,
                logger=logger,
            )
        else:
            summary = {
                "category": category,
                "num_linears": len(targets),
                "mode": "none",
            }
            metrics = {}
        if str(args.outlier_protect_mode) != "none" and run_any_eval and bool(args.eval_after_residual):
            _prewarm_active_residual_categories(
                category=category,
                model=model,
                reference_model=reference_model,
                residency=residency,
                active_categories=active_categories,
                args=args,
                logger=logger,
                stage="after_residual",
            )
            _maybe_eval_residual_stage(
                model=model,
                vae_args=vae_args,
                args=args,
                category=category,
                stage="after_residual",
                active_categories=active_categories,
                eval_tasks_text=eval_tasks_text,
                eval_tokenizer=eval_tokenizer,
                run_any_eval=run_any_eval,
                logger=logger,
                eval_results=eval_results,
            )
        summary["final_residual_rms"] = float(residual_rms)
        payload_summary.append(summary)
        category_metrics[category] = metrics
        del reconstructed_by_name, residual_by_name
        torch.cuda.empty_cache()

    _restore_all_residual_from_base_vae(model=model, residency=residency, logger=logger)
    compressed_targets = tuple(
        sorted(name for name, module in model.named_modules() if isinstance(module, VAELinear))
    )
    save_paths = save_v6_full_checkpoint(
        model,
        planned_checkpoint_out,
        checkpoint_kind="final_model",
        compressed_targets=compressed_targets,
        pending_dense_targets=tuple(load_meta.get("pending_dense_targets") or ()),
        skip_targets=tuple(load_meta.get("skip_targets") or ()),
        legacy_original_only_sources=tuple(load_meta.get("legacy_original_only_sources") or ()),
        train_mode="none",
        lora_config=None,
        completed_categories=tuple(load_meta.get("completed_categories") or ()),
        compression_categories=tuple(load_meta.get("compression_categories") or ()),
        target_layers=load_meta.get("target_layers"),
        target_modules=tuple(load_meta.get("target_modules") or ()),
        base_model_path=str(args.model_path),
        tokenizer=None,
        save_config=True,
        extra_meta={
            "stage": "residual_from_base",
            "source_base_vae_checkpoint": os.path.abspath(checkpoint_dir),
            "source_base_checkpoint_created_at_utc": base_meta.get("created_at_utc"),
        },
    )

    train_time_sec = float(time.time() - run_start)
    config_payload = _jsonable(vars(args))
    metrics_payload = {
        "outlier_protect_mode": str(args.outlier_protect_mode),
        "outlier_rank_metric": args.outlier_rank_metric,
        "target_categories": target_categories,
        "train_time_sec": train_time_sec,
        "categories": category_metrics,
        "eval_results": eval_results,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(_jsonable(metrics_payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "payload_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload_summary), handle, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "completed.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "completed": True,
                "checkpoint_dir": save_paths["output_dir"],
                "train_time_sec": train_time_sec,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    logger.info("Saved residual-from-base checkpoint to %s", save_paths["output_dir"])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    provided = _provided_options(argv)
    args = parser.parse_args(argv)
    validate_residual_from_base_args(args, provided=provided)
    run_residual_from_base(args)
