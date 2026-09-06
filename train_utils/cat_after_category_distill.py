"""Canonical CAT after-category dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from torch import nn

from train_utils.cat_after_category_common import (
    get_or_build_cat_projection_name_inventory,
    resolve_canonical_after_category_mode,
    resolve_cat_after_category_stage,
    run_canonical_current_decoder,
    run_canonical_current_lora,
    run_canonical_current_lora_decoder,
    run_canonical_remaining_lora,
    run_canonical_remaining_lora_current_decoder,
    run_canonical_remaining_lora_prefix_decoder,
    select_compressed_decoder_targets_from_inventory,
)
from train_utils.distill_teacher import resolve_distill_teacher_required


@dataclass(frozen=True)
class AfterCategoryDistillResult:
    model: nn.Module
    next_lora_round_idx: int
    trained_target_count: int = 0
    did_train: bool = False
    distill_meta: Optional[Dict[str, object]] = None


def _build_distill_stage_meta(
    *,
    mode: str,
    category: str,
    did_train: bool,
    newly_compressed_target_count: int,
    remaining_lora_target_count: int,
    decoder_target_count: int,
    cfg,
    training_args,
    resolved_distill_lr: Optional[float] = None,
    resolved_decoder_lr: Optional[float] = None,
) -> Dict[str, object]:
    decoder_count = int(decoder_target_count)
    distill_lr = float(cfg.lr if resolved_distill_lr is None else resolved_distill_lr)
    decoder_lr = None
    decoder_weight_decay = None
    if decoder_count > 0:
        decoder_lr = float(distill_lr if resolved_decoder_lr is None else resolved_decoder_lr)
        decoder_weight_decay = 0.0
    return {
        "mode": str(mode),
        "category": str(category),
        "did_train": bool(did_train),
        "newly_compressed_target_count": int(newly_compressed_target_count),
        "remaining_lora_target_count": int(remaining_lora_target_count),
        "decoder_target_count": decoder_count,
        "resolved_distill_lr": distill_lr,
        "resolved_decoder_lr": decoder_lr,
        "resolved_distill_weight_decay": float(cfg.weight_decay),
        "decoder_weight_decay": decoder_weight_decay,
        "teacher_required": resolve_distill_teacher_required(
            loss_type=cfg.loss_type,
            hidden_loss_weight=cfg.hidden_loss_weight,
            pre_mlp_hidden_loss_weight=cfg.pre_mlp_hidden_loss_weight,
        ),
        "distill_teacher_model_offload": str(
            getattr(training_args, "distill_teacher_model_offload", "none")
        ).strip().lower(),
    }


def run_after_category_distill(
    *,
    model: nn.Module,
    category: str,
    cat_args,
    vae_args,
    training_args,
    logger,
    lora_round_idx: int,
    transpose_modules: Sequence[str],
    only_decoder_projections: bool,
    compression_categories: Sequence[str],
    teacher_runtime=None,
    newly_compressed_target_count: int = 0,
    current_category_target_names: Optional[Sequence[str]] = None,
    v6_step_checkpoint: Optional[dict] = None,
    online_cat: bool = False,
    checkpoint_distill: bool = False,
) -> AfterCategoryDistillResult:
    del transpose_modules, only_decoder_projections
    mode = resolve_canonical_after_category_mode(cat_args)
    if mode == "none":
        return AfterCategoryDistillResult(model=model, next_lora_round_idx=int(lora_round_idx))
    if bool(online_cat) and bool(checkpoint_distill):
        raise ValueError("run_after_category_distill source cannot be both online_cat and checkpoint_distill.")
    if not bool(online_cat) and not bool(checkpoint_distill):
        raise ValueError("canonical CAT caller must declare online_cat or checkpoint_distill.")

    stage = resolve_cat_after_category_stage(
        cat_args,
        training_args,
        category=str(category),
        round_idx=int(lora_round_idx),
    )
    current_runners = {
        "current_decoder": run_canonical_current_decoder,
        "current_lora": run_canonical_current_lora,
        "current_lora_decoder": run_canonical_current_lora_decoder,
    }
    if mode in current_runners:
        if checkpoint_distill and current_category_target_names is None:
            raise ValueError("checkpoint_distill current mode requires exact current_category_target_names.")
        if current_category_target_names is None:
            inventory = get_or_build_cat_projection_name_inventory(
                model,
                vae_args=vae_args,
                compression_categories=compression_categories,
            )
            current_names = tuple(
                name
                for name, _module in select_compressed_decoder_targets_from_inventory(
                    model,
                    inventory=inventory,
                    decoder_categories=(str(category),),
                    target_layers=getattr(cat_args, "target_layers", "all"),
                    skip_layers=getattr(cat_args, "skip_layers", ""),
                )
            )
        else:
            current_names = tuple(str(name) for name in current_category_target_names)
        result = current_runners[mode](
            model=model,
            category=str(category),
            current_target_names=current_names,
            newly_compressed_target_count=int(newly_compressed_target_count),
            stage=stage,
            vae_args=vae_args,
            logger=logger,
            teacher_runtime=teacher_runtime,
            v6_step_checkpoint=v6_step_checkpoint,
        )
    else:
        remaining_runners = {
            "remaining_lora": run_canonical_remaining_lora,
            "remaining_lora_current_decoder": run_canonical_remaining_lora_current_decoder,
            "remaining_lora_prefix_decoder": run_canonical_remaining_lora_prefix_decoder,
        }
        if checkpoint_distill:
            raise ValueError("checkpoint_distill only supports current-family after-category modes.")
        try:
            runner = remaining_runners[mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported canonical after-category mode: {mode!r}.") from exc
        result = runner(
            model=model,
            category=str(category),
            compression_categories=tuple(str(value) for value in compression_categories),
            newly_compressed_target_count=int(newly_compressed_target_count),
            stage=stage,
            vae_args=vae_args,
            logger=logger,
            teacher_runtime=teacher_runtime,
            target_layers=getattr(cat_args, "target_layers", "all"),
            skip_layers=getattr(cat_args, "skip_layers", ""),
            v6_step_checkpoint=v6_step_checkpoint,
        )
    return AfterCategoryDistillResult(
        model=result.model,
        next_lora_round_idx=int(lora_round_idx) + (1 if result.did_train else 0),
        trained_target_count=int(newly_compressed_target_count),
        did_train=bool(result.did_train),
        distill_meta=dict(result.distill_meta),
    )
