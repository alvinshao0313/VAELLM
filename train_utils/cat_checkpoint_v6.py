"""Online CAT v6 category-boundary/final checkpoint helpers.

This module owns CAT target-inventory classification for stable full checkpoints.
It intentionally depends only on the v6 checkpoint API; legacy checkpoint I/O is
not a fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from torch import nn

from litebsq.vae_linear import VAELinear
from train_utils.cat_after_category_common import (
    get_or_build_cat_projection_name_inventory,
    resolve_canonical_after_category_mode,
    resolve_cat_after_category_stage,
)
from train_utils.checkpoint_v6 import iter_named_vae_linears, save_v6_full_checkpoint
from train_utils.config.targets import parse_skip_layers, parse_target_layers


@dataclass(frozen=True)
class CatV6TargetInventory:
    compressed_targets: Tuple[str, ...]
    pending_dense_targets: Tuple[str, ...]
    skip_targets: Tuple[str, ...]
    implicit_tail_skip_targets: Tuple[str, ...]
    target_layers_meta: Optional[Tuple[int, ...]]


def _coerce_target_layers(raw) -> object:
    if raw == "all":
        return "all"
    if isinstance(raw, tuple):
        return tuple(int(v) for v in raw)
    if isinstance(raw, list):
        return tuple(int(v) for v in raw)
    return parse_target_layers(raw)


def _coerce_skip_layers(raw) -> frozenset[tuple[int, str]]:
    if raw is None or raw == "":
        return frozenset()
    if isinstance(raw, (set, frozenset)):
        return frozenset((int(layer), str(category)) for layer, category in raw)
    return parse_skip_layers(raw)


def _validate_completed_prefix(
    completed_categories: Sequence[str],
    compression_categories: Sequence[str],
) -> Tuple[str, ...]:
    completed = tuple(str(v) for v in completed_categories)
    categories = tuple(str(v) for v in compression_categories)
    if len(set(completed)) != len(completed):
        raise ValueError(f"CAT completed_categories contains duplicates: {completed}")
    if len(completed) > len(categories) or completed != categories[: len(completed)]:
        raise ValueError(
            "CAT completed_categories must be an exact prefix of compression_categories: "
            f"completed={completed}, compression_categories={categories}."
        )
    return completed


def build_cat_v6_target_inventory(
    model: nn.Module,
    *,
    vae_args,
    compression_categories: Sequence[str],
    completed_categories: Sequence[str],
    target_layers,
    skip_layers,
    active_category: Optional[str] = None,
) -> CatV6TargetInventory:
    """Classify stable CAT topology into v6 compressed/pending/skip inventories."""

    categories = tuple(str(v) for v in compression_categories)
    completed = _validate_completed_prefix(completed_categories, categories)
    completed_set = set(completed)
    active = None if active_category is None else str(active_category)
    if active is not None:
        if len(completed) >= len(categories) or active != categories[len(completed)]:
            raise ValueError(
                "CAT round_base active_category must be the first category after completed prefix: "
                f"completed={completed}, active={active!r}, categories={categories}."
            )
    compressed_category_set = set(completed_set)
    if active is not None:
        compressed_category_set.add(active)
    resolved_layers = _coerce_target_layers(target_layers)
    allowed_layers = None if resolved_layers == "all" else {int(v) for v in resolved_layers}
    explicit_skip = _coerce_skip_layers(skip_layers)

    inventory = get_or_build_cat_projection_name_inventory(
        model,
        vae_args=vae_args,
        compression_categories=categories,
    )
    modules: Dict[str, nn.Module] = dict(model.named_modules())
    inventory_names = set(inventory.values())
    actual_vae_names = {name for name, _module in iter_named_vae_linears(model)}
    outside_inventory = sorted(actual_vae_names - inventory_names)
    if outside_inventory:
        raise ValueError(
            "Online CAT v6 checkpoint found VAELinear outside canonical CAT projection inventory: "
            + ", ".join(outside_inventory)
        )

    compressed = []
    pending = []
    skip = []
    implicit_tail_skip = []
    for (layer_idx, category), name in inventory.items():
        key = (int(layer_idx), str(category))
        module = modules.get(str(name))
        if module is None:
            raise ValueError(f"CAT inventory target {name!r} is missing from live model.")
        in_target_layer = allowed_layers is None or int(layer_idx) in allowed_layers
        is_explicit_skip = key in explicit_skip

        if isinstance(module, VAELinear):
            if not in_target_layer:
                raise ValueError(
                    f"CAT non-target layer unexpectedly became VAELinear: key={key} name={name!r}."
                )
            if is_explicit_skip:
                raise ValueError(f"Explicit CAT skip target unexpectedly became VAELinear: key={key} name={name!r}.")
            if str(category) not in compressed_category_set:
                raise ValueError(
                    "Stable CAT checkpoint cannot contain a compressed category outside the completed prefix/active round: "
                    f"key={key} name={name!r} completed={completed} active={active!r}."
                )
            if bool(getattr(module, "always_use_original", False)):
                raise ValueError(f"CAT compressed target is original-only instead of v6 compressed state: {name!r}.")
            compressed.append(str(name))
            continue

        if not isinstance(module, nn.Linear):
            if in_target_layer or is_explicit_skip:
                raise TypeError(
                    f"CAT stable target {name!r} must be VAELinear or ordinary nn.Linear, got {type(module)}."
                )
            continue

        if is_explicit_skip:
            skip.append(str(name))
            continue
        if not in_target_layer:
            continue
        if str(category) in compressed_category_set:
            # With allow_tail_group=false, the final incomplete group is deliberately
            # never compressed. Once that category has reached a stable round base or
            # category boundary it is a permanent original-weight target.
            skip.append(str(name))
            implicit_tail_skip.append(str(name))
        else:
            pending.append(str(name))

    compressed_tuple = tuple(compressed)
    if set(compressed_tuple) != actual_vae_names:
        raise RuntimeError(
            "CAT v6 compressed inventory is not exact live VAELinear truth: "
            f"metadata_only={sorted(set(compressed_tuple) - actual_vae_names)}, "
            f"live_only={sorted(actual_vae_names - set(compressed_tuple))}."
        )

    target_layers_meta = None if resolved_layers == "all" else tuple(int(v) for v in resolved_layers)
    return CatV6TargetInventory(
        compressed_targets=compressed_tuple,
        pending_dense_targets=tuple(pending),
        skip_targets=tuple(skip),
        implicit_tail_skip_targets=tuple(implicit_tail_skip),
        target_layers_meta=target_layers_meta,
    )


def save_cat_v6_full_checkpoint(
    model: nn.Module,
    output_dir: str,
    *,
    checkpoint_kind: str,
    category: Optional[str],
    completed_categories: Sequence[str],
    compression_categories: Sequence[str],
    cat_args,
    vae_args,
    training_args,
    tokenizer=None,
    base_model_path: Optional[str] = None,
    distill_stage_meta: Optional[dict] = None,
    distill_stage_history: Sequence[dict] = (),
    round_idx: int = 0,
    cat_runtime_state: Optional[dict] = None,
    is_main_process: bool = True,
    distributed_barrier=None,
):
    if checkpoint_kind not in {"round_base", "category_boundary", "final_model"}:
        raise ValueError(
            "save_cat_v6_full_checkpoint only owns CAT round_base/category_boundary/final_model saves, "
            f"got {checkpoint_kind!r}."
        )
    if checkpoint_kind in {"round_base", "category_boundary"} and category is None:
        raise ValueError(f"CAT {checkpoint_kind} save requires category.")
    if checkpoint_kind == "final_model" and category is not None:
        raise ValueError("CAT final_model save requires category=None.")
    inventory = build_cat_v6_target_inventory(
        model,
        vae_args=vae_args,
        compression_categories=compression_categories,
        completed_categories=completed_categories,
        target_layers=getattr(cat_args, "target_layers", "all"),
        skip_layers=getattr(cat_args, "skip_layers", ""),
        active_category=(str(category) if checkpoint_kind == "round_base" else None),
    )

    after_mode = resolve_canonical_after_category_mode(cat_args)
    lora_config = None
    resolved_learning_rates = None
    norm_train_mode = "none"
    lm_head_train_mode = "none"
    config_category = str(category) if category is not None else (
        str(completed_categories[-1]) if completed_categories else None
    )
    if after_mode != "none" and config_category is not None:
        stage = resolve_cat_after_category_stage(
            cat_args,
            training_args,
            category=config_category,
            round_idx=int(round_idx),
        )
        cfg = stage.config
        lora_config = {
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

    stage_name = {
        "round_base": "round_base",
        "category_boundary": "after_category",
        "final_model": "final",
    }[checkpoint_kind]
    extra_meta = {
        "stage": stage_name,
        "category": None if category is None else str(category),
        "active_category": str(category) if checkpoint_kind == "round_base" else None,
        "distill_stage": None if distill_stage_meta is None else dict(distill_stage_meta),
        "distill_stage_history": [dict(item) for item in distill_stage_history],
        "implicit_tail_skip_targets": list(inventory.implicit_tail_skip_targets),
        "recovery_lora_config": lora_config,
    }
    return save_v6_full_checkpoint(
        model,
        output_dir,
        checkpoint_kind=checkpoint_kind,
        compressed_targets=inventory.compressed_targets,
        pending_dense_targets=inventory.pending_dense_targets,
        skip_targets=inventory.skip_targets,
        train_mode="none",
        after_category_mode=str(after_mode),
        norm_train_mode=norm_train_mode,
        lm_head_train_mode=lm_head_train_mode,
        lora_config=None,
        resolved_learning_rates=resolved_learning_rates,
        completed_categories=completed_categories,
        compression_categories=compression_categories,
        target_layers=inventory.target_layers_meta,
        target_modules=tuple(str(v) for v in compression_categories),
        finalized_status={
            "lora_finalized": True,
            "decoder_finalized": True,
            "aux_finalized": True,
            "round_base_ready": checkpoint_kind == "round_base",
            "stable_category_boundary": checkpoint_kind == "category_boundary",
            "inference_ready": checkpoint_kind == "final_model",
        },
        base_model_path=base_model_path,
        tokenizer=tokenizer,
        save_config=True,
        cat_runtime_state=cat_runtime_state,
        extra_meta=extra_meta,
        is_main_process=bool(is_main_process),
        distributed_barrier=distributed_barrier,
    )


__all__ = [
    "CatV6TargetInventory",
    "build_cat_v6_target_inventory",
    "save_cat_v6_full_checkpoint",
]
