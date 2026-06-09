import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn

from e2e_common.checkpoint_io import load_e2e_model_checkpoint, save_e2e_model_checkpoint
from train_utils.block_vae_lora_args import BlockVaeLoraArgs, format_skip_layers
from train_utils.model_checkpoint_io import META_FILENAME, resolve_checkpoint_dir


BLOCK_LAYER_CHECKPOINT_STAGE = "block_vae_lora_layer"
BLOCK_FINAL_CHECKPOINT_STAGE = "block_vae_lora_final"
BLOCK_CHECKPOINTS_DIRNAME = "block_checkpoints"


@dataclass(frozen=True)
class BlockResumeState:
    checkpoint_dir: str
    completed_block_layer_idx: int
    completed_block_layers: Tuple[int, ...]
    next_block_layer_idx: int


def _to_jsonable(value):
    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return value.to_jsonable()
    if is_dataclass(value):
        return {key: _to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, argparse.Namespace):
        return {key: _to_jsonable(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _read_checkpoint_meta(path: str) -> Tuple[str, Dict[str, Any]]:
    checkpoint_dir = resolve_checkpoint_dir(str(path))
    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise TypeError(f"Checkpoint meta must be a dict, got {type(meta)}.")
    return checkpoint_dir, meta


def _extra_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    extra = meta.get("extra_meta", {})
    if not isinstance(extra, dict):
        raise TypeError("checkpoint_meta.extra_meta must be a dict for block resume.")
    return extra


def _validate_resume_stage(extra: Dict[str, Any], checkpoint_dir: str) -> None:
    stage = str(extra.get("stage", "")).strip()
    if stage == BLOCK_FINAL_CHECKPOINT_STAGE:
        raise ValueError(f"Final block checkpoint cannot be used for resume: {checkpoint_dir}")
    if stage != BLOCK_LAYER_CHECKPOINT_STAGE:
        raise ValueError(
            f"Checkpoint is not a block layer checkpoint: {checkpoint_dir}. "
            f"Expected extra_meta.stage={BLOCK_LAYER_CHECKPOINT_STAGE!r}, got {stage!r}."
        )


def _int_tuple(value: object, *, field_name: str) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value)}.")
    return tuple(int(item) for item in value)


def _str_list(value: object, *, field_name: str) -> List[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value)}.")
    return [str(item) for item in value]


def _checkpoint_block_vae_categories(extra: Dict[str, Any]) -> Tuple[str, ...]:
    value = extra.get("block_vae_categories")
    if value is None:
        block_args = extra.get("block_distill", {})
        if isinstance(block_args, dict):
            value = block_args.get("block_vae_categories")
    if not isinstance(value, list):
        raise TypeError("checkpoint extra_meta.block_vae_categories must be a list.")
    return tuple(str(item) for item in value)


def load_block_resume_state(
    path: str,
    *,
    current_args: BlockVaeLoraArgs,
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
) -> BlockResumeState:
    checkpoint_dir, meta = _read_checkpoint_meta(path)
    extra = _extra_meta(meta)
    _validate_resume_stage(extra, checkpoint_dir)

    expected_selected = tuple(sorted(int(layer_idx) for layer_idx in selected_layers))
    actual_selected = _int_tuple(extra.get("selected_block_layers"), field_name="extra_meta.selected_block_layers")
    if actual_selected != expected_selected:
        raise ValueError(
            "Resume checkpoint selected_block_layers do not match current --block_layers: "
            f"checkpoint={list(actual_selected)} current={list(expected_selected)}"
        )

    expected_skip = format_skip_layers(sorted(skip_layer_keys))
    actual_skip = _str_list(extra.get("skip_layers"), field_name="extra_meta.skip_layers")
    if actual_skip != expected_skip:
        raise ValueError(
            "Resume checkpoint skip_layers do not match current --skip_layers: "
            f"checkpoint={actual_skip} current={expected_skip}"
        )

    actual_train_mode = str(extra.get("block_distill_train_mode", "")).strip().lower()
    expected_train_mode = str(current_args.block_distill_train_mode).strip().lower()
    if actual_train_mode != expected_train_mode:
        raise ValueError(
            "Resume checkpoint block_distill_train_mode does not match current argument: "
            f"checkpoint={actual_train_mode!r} current={expected_train_mode!r}"
        )
    actual_categories = _checkpoint_block_vae_categories(extra)
    expected_categories = tuple(str(category) for category in current_args.block_vae_categories)
    if actual_categories != expected_categories:
        raise ValueError(
            "Resume checkpoint block_vae_categories do not match current argument: "
            f"checkpoint={list(actual_categories)} current={list(expected_categories)}"
        )
    actual_pipeline_mode = extra.get("block_vae_pipeline_mode")
    if actual_pipeline_mode is not None:
        actual_pipeline_mode = str(actual_pipeline_mode).strip().lower()
        expected_pipeline_mode = str(current_args.block_vae_pipeline_mode).strip().lower()
        compatible_pipeline = (
            expected_pipeline_mode == "distill"
            and actual_pipeline_mode in {"distill", "pretrain_distill"}
        )
        if actual_pipeline_mode != expected_pipeline_mode and not compatible_pipeline:
            raise ValueError(
                "Resume checkpoint block_vae_pipeline_mode does not match current argument: "
                f"checkpoint={actual_pipeline_mode!r} current={expected_pipeline_mode!r}"
            )

    completed_layer_idx = int(extra.get("completed_block_layer_idx"))
    completed_layers = _int_tuple(extra.get("completed_block_layers"), field_name="extra_meta.completed_block_layers")
    next_layer_idx = int(extra.get("next_block_layer_idx", completed_layer_idx + 1))
    if next_layer_idx != completed_layer_idx + 1:
        raise ValueError(
            "Resume checkpoint next_block_layer_idx must equal completed_block_layer_idx + 1: "
            f"completed={completed_layer_idx} next={next_layer_idx}"
        )
    if completed_layer_idx not in completed_layers:
        raise ValueError(
            "Resume checkpoint completed_block_layers must contain completed_block_layer_idx: "
            f"completed={completed_layer_idx} layers={list(completed_layers)}"
        )
    if any(int(layer_idx) not in expected_selected for layer_idx in completed_layers):
        raise ValueError(
            "Resume checkpoint completed_block_layers contains layers outside current --block_layers: "
            f"completed={list(completed_layers)} selected={list(expected_selected)}"
        )

    return BlockResumeState(
        checkpoint_dir=checkpoint_dir,
        completed_block_layer_idx=int(completed_layer_idx),
        completed_block_layers=tuple(int(layer_idx) for layer_idx in completed_layers),
        next_block_layer_idx=int(next_layer_idx),
    )


def load_block_resume_model(
    path: str,
    *,
    access_token: Optional[str],
    proxy_group_size: int,
    proxy_compute_device: object,
    logger=None,
):
    checkpoint_dir, meta = _read_checkpoint_meta(path)
    _validate_resume_stage(_extra_meta(meta), checkpoint_dir)
    model, load_meta, load_result = load_e2e_model_checkpoint(
        checkpoint_dir,
        access_token=access_token,
        map_location="cpu",
        strict=True,
        materialize_proxy_decoded_linears=True,
        proxy_group_size=int(proxy_group_size),
        proxy_compute_device=proxy_compute_device,
        proxy_logger=logger,
    )
    return model, checkpoint_dir, load_meta, load_result


def build_block_checkpoint_meta(
    *,
    args: BlockVaeLoraArgs,
    completed_block_layer_idx: int,
    completed_block_layers: Sequence[int],
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
    target_module_count: int,
    block_vae_cache_manifest_hash: str = "",
) -> Dict[str, Any]:
    completed_layers = sorted(int(layer_idx) for layer_idx in completed_block_layers)
    return {
        "stage": BLOCK_LAYER_CHECKPOINT_STAGE,
        "block_distill": _to_jsonable(args),
        "block_distill_train_mode": str(args.block_distill_train_mode),
        "block_vae_pipeline_mode": str(args.block_vae_pipeline_mode),
        "block_vae_categories": [str(category) for category in args.block_vae_categories],
        "block_vae_pretrain_manifest_hash": str(block_vae_cache_manifest_hash),
        "block_vae_cache_manifest_hash": str(block_vae_cache_manifest_hash),
        "completed_block_layer_idx": int(completed_block_layer_idx),
        "completed_block_layers": completed_layers,
        "next_block_layer_idx": int(completed_block_layer_idx) + 1,
        "selected_block_layers": sorted(int(layer_idx) for layer_idx in selected_layers),
        "skip_layers": format_skip_layers(sorted(skip_layer_keys)),
        "target_module_count": int(target_module_count),
    }


def save_block_layer_checkpoint(
    *,
    model: nn.Module,
    run_output_dir: str,
    args: BlockVaeLoraArgs,
    completed_block_layer_idx: int,
    completed_block_layers: Sequence[int],
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
    target_module_count: int,
    block_vae_cache_manifest_hash: str = "",
) -> Dict[str, str]:
    checkpoint_dir = os.path.join(
        str(run_output_dir),
        BLOCK_CHECKPOINTS_DIRNAME,
        f"block_{int(completed_block_layer_idx):04d}",
    )
    extra_meta = build_block_checkpoint_meta(
        args=args,
        completed_block_layer_idx=int(completed_block_layer_idx),
        completed_block_layers=completed_block_layers,
        selected_layers=selected_layers,
        skip_layer_keys=skip_layer_keys,
        target_module_count=int(target_module_count),
        block_vae_cache_manifest_hash=str(block_vae_cache_manifest_hash),
    )
    return save_e2e_model_checkpoint(
        model,
        checkpoint_dir,
        base_model_path=args.model_path,
        tokenizer=None,
        save_config=True,
        extra_meta=extra_meta,
        unload_vae_original_weights=False,
        compact_unload_vae_original_weights=False,
    )


def prune_block_layer_checkpoints(*, run_output_dir: str, keep_last: int) -> List[str]:
    keep = int(keep_last)
    if keep < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}.")
    checkpoints_root = os.path.join(str(run_output_dir), BLOCK_CHECKPOINTS_DIRNAME)
    if keep == 0 or not os.path.isdir(checkpoints_root):
        return []

    candidates: List[Tuple[int, str]] = []
    for child in os.listdir(checkpoints_root):
        child_dir = os.path.join(checkpoints_root, child)
        if not os.path.isdir(child_dir):
            continue
        meta_path = os.path.join(child_dir, META_FILENAME)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        extra = _extra_meta(meta)
        if str(extra.get("stage", "")) != BLOCK_LAYER_CHECKPOINT_STAGE:
            continue
        candidates.append((int(extra.get("completed_block_layer_idx")), child_dir))

    candidates.sort(key=lambda item: item[0], reverse=True)
    removed: List[str] = []
    for _layer_idx, checkpoint_dir in candidates[keep:]:
        shutil.rmtree(checkpoint_dir)
        removed.append(checkpoint_dir)
    return removed
