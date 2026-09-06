"""v6 checkpoint I/O and progress contract for post-hoc CAT checkpoint distillation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Mapping, Optional, Sequence, Tuple

from torch import nn

from train_utils.cat_after_category_common import (
    resolve_canonical_after_category_mode,
    resolve_cat_after_category_stage,
)
from train_utils.checkpoint_v6 import (
    load_v6_cat_runtime_state,
    load_v6_full_checkpoint_into_model,
    load_v6_meta,
    resolve_v6_checkpoint_dir,
    save_v6_full_checkpoint,
    validate_v6_meta,
)


CURRENT_CHECKPOINT_DISTILL_MODES = frozenset(
    {"current_lora", "current_decoder", "current_lora_decoder"}
)
REMAINING_CHECKPOINT_DISTILL_MODES = frozenset(
    {"remaining_lora", "remaining_lora_current_decoder", "remaining_lora_prefix_decoder"}
)
_STABLE_INPUT_KINDS = frozenset({"category_boundary", "final_model"})


@dataclass(frozen=True)
class CheckpointDistillV6Source:
    requested_checkpoint_dir: str
    checkpoint_dir: str
    checkpoint_kind: str
    meta: Dict[str, object]
    cat_runtime_state: Optional[dict]


@dataclass(frozen=True)
class CheckpointDistillProgress:
    completed_categories: Tuple[str, ...]
    stage_history: Tuple[Dict[str, object], ...]
    lora_round_idx: int


def resolve_checkpoint_distill_mode(cat_args) -> str:
    mode = str(resolve_canonical_after_category_mode(cat_args))
    if mode in REMAINING_CHECKPOINT_DISTILL_MODES:
        raise ValueError(
            "CAT checkpoint-distill does not support remaining_* modes because its input checkpoint "
            "does not have online-CAT future dense nn.Linear semantics. "
            f"Got after_category_mode={mode!r}."
        )
    if mode not in CURRENT_CHECKPOINT_DISTILL_MODES:
        raise ValueError(
            "CAT checkpoint-distill only supports current_lora, current_decoder, "
            f"or current_lora_decoder; got {mode!r}."
        )
    return mode


def load_checkpoint_distill_v6_source(
    checkpoint_path: str,
    *,
    hf_args,
    vae_args,
    logger,
) -> tuple[nn.Module, CheckpointDistillV6Source]:
    requested = os.path.abspath(str(checkpoint_path))
    resolved = resolve_v6_checkpoint_dir(requested)
    meta = validate_v6_meta(load_v6_meta(resolved))
    kind = str(meta.get("checkpoint_kind"))
    if kind not in _STABLE_INPUT_KINDS:
        raise ValueError(
            "CAT checkpoint-distill requires a stable v6 category_boundary or final_model input; "
            f"got checkpoint_kind={kind!r}. training_step/round_base are recovery intermediates and are not accepted."
        )

    base_model_path = meta.get("base_model_path") or getattr(vae_args, "model_path", None)
    if not base_model_path:
        raise ValueError(
            f"Cannot determine base model path for v6 checkpoint-distill source: {resolved}."
        )

    logger.info("Loading v6 checkpoint for CAT checkpoint-distill: %s", resolved)
    logger.info("Checkpoint-distill base model path: %s", str(base_model_path))
    from rotation.model_utils import get_model

    model = get_model(str(base_model_path), hf_args.access_token)
    model, load_meta, load_result = load_v6_full_checkpoint_into_model(
        model,
        resolved,
        expected_kind=kind,
        map_location="cpu",
        strict=True,
    )
    vae_args.model_path = str(load_meta.get("base_model_path") or base_model_path)
    runtime_state = load_v6_cat_runtime_state(resolved, required=False)
    source = CheckpointDistillV6Source(
        requested_checkpoint_dir=requested,
        checkpoint_dir=resolved,
        checkpoint_kind=kind,
        meta=dict(load_meta),
        cat_runtime_state=(None if runtime_state is None else dict(runtime_state)),
    )
    logger.info(
        "v6 checkpoint-distill source loaded. checkpoint_id=%s kind=%s converted_module_count=%s "
        "missing_keys=%d unexpected_keys=%d",
        str(load_meta.get("checkpoint_id")),
        kind,
        str(load_meta.get("converted_module_count")),
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
    )
    return model, source


def _validate_category_list(raw, *, field_name: str) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"{field_name} must be a list/tuple, got {type(raw)}.")
    values = []
    seen = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings.")
        value = str(item)
        if value in seen:
            raise ValueError(f"{field_name} contains duplicate category {value!r}.")
        seen.add(value)
        values.append(value)
    return tuple(values)


def load_checkpoint_distill_progress(source: CheckpointDistillV6Source) -> CheckpointDistillProgress:
    extra = source.meta.get("extra_meta")
    extra = dict(extra) if isinstance(extra, Mapping) else {}
    completed = _validate_category_list(
        extra.get("checkpoint_distill_completed_categories"),
        field_name="checkpoint_distill_completed_categories",
    )
    raw_history = extra.get("checkpoint_distill_stage_history")
    if raw_history is None:
        history: Tuple[Dict[str, object], ...] = ()
    else:
        if not isinstance(raw_history, (list, tuple)):
            raise TypeError("checkpoint_distill_stage_history must be a list/tuple.")
        rows = []
        for idx, item in enumerate(raw_history):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"checkpoint_distill_stage_history[{idx}] must be a mapping, got {type(item)}."
                )
            rows.append(dict(item))
        history = tuple(rows)

    trained_categories = []
    lora_round_idx = 0
    for idx, item in enumerate(history):
        did_train = item.get("did_train")
        if not isinstance(did_train, bool):
            raise TypeError(
                f"checkpoint_distill_stage_history[{idx}].did_train must be bool, got {type(did_train)}."
            )
        if did_train:
            lora_round_idx += 1
            category = item.get("category")
            if isinstance(category, str) and category.strip():
                trained_categories.append(str(category))
    if completed and tuple(trained_categories) != completed:
        raise ValueError(
            "checkpoint-distill progress mismatch: completed categories must exactly equal the did_train=true history order. "
            f"completed={completed}, trained_history={tuple(trained_categories)}."
        )
    if not completed and trained_categories:
        completed = tuple(trained_categories)
    return CheckpointDistillProgress(
        completed_categories=completed,
        stage_history=history,
        lora_round_idx=int(lora_round_idx),
    )


def _preserved_online_cat_extra_meta(source_meta: Mapping[str, object]) -> dict:
    extra = source_meta.get("extra_meta")
    if not isinstance(extra, Mapping):
        return {}
    preserved = {}
    for key in ("distill_stage", "distill_stage_history", "implicit_tail_skip_targets"):
        if key in extra:
            value = extra[key]
            if isinstance(value, list):
                preserved[key] = [dict(item) if isinstance(item, Mapping) else item for item in value]
            elif isinstance(value, Mapping):
                preserved[key] = dict(value)
            else:
                preserved[key] = value
    return preserved


def save_checkpoint_distill_v6_model(
    model: nn.Module,
    output_dir: str,
    *,
    checkpoint_kind: str,
    category: Optional[str],
    mode: str,
    source: CheckpointDistillV6Source,
    checkpoint_distill_completed_categories: Sequence[str],
    checkpoint_distill_stage_history: Sequence[Mapping[str, object]],
    cat_args,
    training_args,
    vae_args,
    tokenizer,
    round_idx: int,
    logger,
) -> dict:
    canonical_mode = str(mode)
    if canonical_mode not in CURRENT_CHECKPOINT_DISTILL_MODES:
        raise ValueError(f"checkpoint-distill save requires current_* mode, got {canonical_mode!r}.")
    if checkpoint_kind not in {"category_boundary", "final_model"}:
        raise ValueError(f"Unsupported checkpoint-distill output kind: {checkpoint_kind!r}.")
    if checkpoint_kind == "category_boundary" and not isinstance(category, str):
        raise ValueError("checkpoint-distill category_boundary save requires category.")
    if checkpoint_kind == "final_model" and category is not None:
        raise ValueError("checkpoint-distill final_model save requires category=None.")

    source_meta = source.meta
    root_completed = tuple(str(v) for v in source_meta.get("completed_categories") or ())
    compression_categories = tuple(str(v) for v in source_meta.get("compression_categories") or ())
    config_category = str(category) if category is not None else (
        str(checkpoint_distill_completed_categories[-1])
        if checkpoint_distill_completed_categories
        else None
    )

    recovery_lora_audit = None
    resolved_learning_rates = None
    norm_train_mode = "none"
    lm_head_train_mode = "none"
    if config_category is not None:
        stage = resolve_cat_after_category_stage(
            cat_args,
            training_args,
            category=config_category,
            round_idx=int(round_idx),
        )
        cfg = stage.config
        if str(stage.mode) != canonical_mode:
            raise ValueError(
                f"checkpoint-distill save mode mismatch: stage={stage.mode!r}, expected={canonical_mode!r}."
            )
        recovery_lora_audit = {
            "rank": int(cfg.lora.rank),
            "alpha": float(cfg.lora.alpha),
            "dropout": float(cfg.lora.dropout),
        }
        norm_train_mode = str(cfg.aux.norm_train_mode)
        lm_head_train_mode = str(cfg.aux.lm_head_train_mode)
        resolved_learning_rates = {
            "learning_rate": float(cfg.opt.learning_rate),
            "decoder_lr": float(cfg.opt.resolved_decoder_lr()),
            "norm_lr": None if cfg.aux.norm_lr is None else float(cfg.aux.norm_lr),
            "lm_head_lr": None if cfg.aux.lm_head_lr is None else float(cfg.aux.lm_head_lr),
        }

    extra_meta = _preserved_online_cat_extra_meta(source_meta)
    extra_meta.update(
        {
            "stage": (
                "checkpoint_after_category_mode"
                if checkpoint_kind == "category_boundary"
                else "checkpoint_distill_final"
            ),
            "category": None if category is None else str(category),
            "checkpoint_distill_completed_categories": [
                str(v) for v in checkpoint_distill_completed_categories
            ],
            "checkpoint_distill_stage_history": [
                dict(item) for item in checkpoint_distill_stage_history
            ],
            "checkpoint_distill_source_checkpoint_id": str(source_meta.get("checkpoint_id")),
        }
    )

    result = save_v6_full_checkpoint(
        model,
        output_dir,
        checkpoint_kind=checkpoint_kind,
        compressed_targets=tuple(str(v) for v in source_meta.get("compressed_targets") or ()),
        pending_dense_targets=tuple(str(v) for v in source_meta.get("pending_dense_targets") or ()),
        skip_targets=tuple(str(v) for v in source_meta.get("skip_targets") or ()),
        legacy_original_only_sources=tuple(
            str(v) for v in source_meta.get("legacy_original_only_sources") or ()
        ),
        train_mode="none",
        after_category_mode=canonical_mode,
        norm_train_mode=norm_train_mode,
        lm_head_train_mode=lm_head_train_mode,
        lora_config=None,
        resolved_learning_rates=resolved_learning_rates,
        completed_categories=root_completed,
        compression_categories=compression_categories,
        target_layers=source_meta.get("target_layers"),
        target_modules=tuple(str(v) for v in source_meta.get("target_modules") or ()),
        finalized_status={
            "lora_finalized": True,
            "decoder_finalized": True,
            "aux_finalized": True,
            "stable_category_boundary": checkpoint_kind == "category_boundary",
            "inference_ready": checkpoint_kind == "final_model",
        },
        runtime_audit={
            "runtime": "train_utils.cat_checkpoint_distill",
            "source_checkpoint_id": str(source_meta.get("checkpoint_id")),
            "checkpoint_distill_mode": canonical_mode,
            "recovery_lora_config": recovery_lora_audit,
        },
        base_model_path=str(source_meta.get("base_model_path") or vae_args.model_path),
        tokenizer=tokenizer,
        save_config=True,
        cat_runtime_state=source.cat_runtime_state,
        extra_meta=extra_meta,
        is_main_process=True,
        distributed_barrier=None,
    )
    logger.info(
        "Saved checkpoint-distill v6 model: kind=%s checkpoint_id=%s path=%s",
        checkpoint_kind,
        str(result.get("checkpoint_id")),
        str(result.get("output_dir")),
    )
    return result


__all__ = [
    "CURRENT_CHECKPOINT_DISTILL_MODES",
    "REMAINING_CHECKPOINT_DISTILL_MODES",
    "CheckpointDistillProgress",
    "CheckpointDistillV6Source",
    "load_checkpoint_distill_progress",
    "load_checkpoint_distill_v6_source",
    "resolve_checkpoint_distill_mode",
    "save_checkpoint_distill_v6_model",
]
