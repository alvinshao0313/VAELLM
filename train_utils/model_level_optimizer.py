"""Common model-level continuous optimizer grouping for CAT/E2E.

Consumes Task 6 ``ModelLevelTrainableSelection`` inventories. Does not classify
parameters by name heuristics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from torch import nn

from train_utils.model_level_trainables import (
    ModelLevelTrainableSelection,
    assert_disjoint_component_inventories,
)

logger = logging.getLogger(__name__)

GROUP_LORA = "lora"
GROUP_DECODER = "decoder"
GROUP_NORM = "norm"
GROUP_LM_HEAD = "lm_head"


@dataclass(frozen=True)
class ModelLevelOptimizerLRConfig:
    """Resolved LR/WD contract for the four inventory components."""

    learning_rate: float
    weight_decay: float
    decoder_lr: Optional[float] = None
    norm_lr: Optional[float] = None
    lm_head_lr: Optional[float] = None

    def resolved_decoder_lr(self) -> float:
        if self.decoder_lr is None:
            return float(self.learning_rate)
        return float(self.decoder_lr)

    def resolved_norm_lr(self) -> float:
        if self.norm_lr is None:
            return float(self.learning_rate)
        return float(self.norm_lr)

    def resolved_lm_head_lr(self) -> float:
        if self.lm_head_lr is None:
            return float(self.learning_rate)
        return float(self.lm_head_lr)


def _params_list(inventory: Dict[str, nn.Parameter]) -> List[nn.Parameter]:
    return list(inventory.values())


def _assert_all_requires_grad(inventory: Dict[str, nn.Parameter], *, inventory_name: str) -> None:
    for key, param in inventory.items():
        if not bool(param.requires_grad):
            raise RuntimeError(
                f"Frozen parameter must not enter optimizer inventory {inventory_name}: key={key!r}."
            )


def build_model_level_param_groups(
    selection: ModelLevelTrainableSelection,
    *,
    lr_config: ModelLevelOptimizerLRConfig,
    model: Optional[nn.Module] = None,
) -> List[Dict[str, object]]:
    """Build HF-compatible param groups from labeled inventories.

    Mapping (exact):
    - lora_parameters     -> lr=learning_rate, wd=weight_decay
    - decoder_parameters  -> lr=decoder_lr or learning_rate, wd=0
    - norm_parameters      -> lr=norm_lr or learning_rate, wd=0
    - lm_head_parameters   -> lr=lm_head_lr or learning_rate, wd=0
    """
    assert_disjoint_component_inventories(
        decoder_parameters=selection.decoder_parameters,
        lora_parameters=selection.lora_parameters,
        norm_parameters=selection.norm_parameters,
        lm_head_parameters=selection.lm_head_parameters,
    )

    for inv_name, inv in (
        ("decoder_parameters", selection.decoder_parameters),
        ("lora_parameters", selection.lora_parameters),
        ("norm_parameters", selection.norm_parameters),
        ("lm_head_parameters", selection.lm_head_parameters),
    ):
        _assert_all_requires_grad(inv, inventory_name=inv_name)

    main_lr = float(lr_config.learning_rate)
    main_wd = float(lr_config.weight_decay)
    decoder_lr = float(lr_config.resolved_decoder_lr())
    norm_lr = float(lr_config.resolved_norm_lr())
    lm_head_lr = float(lr_config.resolved_lm_head_lr())

    groups: List[Dict[str, object]] = []
    seen_ids: Dict[int, str] = {}

    def _append(group_name: str, inventory: Dict[str, nn.Parameter], *, lr: float, weight_decay: float) -> None:
        params = _params_list(inventory)
        if not params:
            return
        for param in params:
            pid = id(param)
            if pid in seen_ids:
                raise RuntimeError(
                    "Parameter id appears in multiple optimizer groups: "
                    f"{seen_ids[pid]} and {group_name}."
                )
            seen_ids[pid] = group_name
        groups.append(
            {
                "group_name": group_name,
                "params": params,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            }
        )

    _append(GROUP_LORA, selection.lora_parameters, lr=main_lr, weight_decay=main_wd)
    _append(GROUP_DECODER, selection.decoder_parameters, lr=decoder_lr, weight_decay=0.0)
    _append(GROUP_NORM, selection.norm_parameters, lr=norm_lr, weight_decay=0.0)
    _append(GROUP_LM_HEAD, selection.lm_head_parameters, lr=lm_head_lr, weight_decay=0.0)

    if not groups:
        raise RuntimeError(
            "Model-level optimizer requires at least one non-empty trainable inventory "
            "(decoder/lora/norm/lm_head)."
        )

    if model is not None:
        expected_ids: Set[int] = set()
        for _name, param in model.named_parameters():
            if bool(param.requires_grad):
                expected_ids.add(id(param))
        grouped_ids = set(seen_ids.keys())
        missing = expected_ids - grouped_ids
        extra = grouped_ids - expected_ids
        if missing:
            raise RuntimeError(
                "Trainable parameters missing from model-level optimizer inventories: "
                f"count={len(missing)}."
            )
        if extra:
            raise RuntimeError(
                "Optimizer inventories contain parameters that are not requires_grad=True on model: "
                f"count={len(extra)}."
            )
        if len(grouped_ids) != len(expected_ids):
            raise RuntimeError("Optimizer inventory coverage mismatch.")

    return groups


def create_model_level_optimizer(
    trainer,
    *,
    selection: Optional[ModelLevelTrainableSelection] = None,
    lr_config: Optional[ModelLevelOptimizerLRConfig] = None,
):
    """Create continuous optimizer from trainer + labeled inventories."""
    if trainer.optimizer is not None:
        return trainer.optimizer

    resolved_selection = selection
    if resolved_selection is None:
        resolved_selection = getattr(trainer, "model_level_trainable_selection", None)
    if resolved_selection is None:
        raise RuntimeError(
            "create_model_level_optimizer requires ModelLevelTrainableSelection "
            "(trainer.model_level_trainable_selection or explicit selection=)."
        )

    if lr_config is None:
        attached = getattr(trainer, "model_level_optimizer_lr_config", None)
        if attached is not None:
            lr_config = attached
        else:
            decoder_lr = getattr(trainer, "decoder_lr", None)
            if decoder_lr is None:
                decoder_lr = getattr(trainer, "distill_decoder_lr", None)
            lr_config = ModelLevelOptimizerLRConfig(
                learning_rate=float(trainer.args.learning_rate),
                weight_decay=float(trainer.args.weight_decay),
                decoder_lr=None if decoder_lr is None else float(decoder_lr),
                norm_lr=getattr(trainer, "norm_lr", None),
                lm_head_lr=getattr(trainer, "lm_head_lr", None),
            )

    opt_model = getattr(trainer, "model_wrapped", None) or trainer.model
    optimizer_grouped_parameters = build_model_level_param_groups(
        resolved_selection,
        lr_config=lr_config,
        model=opt_model,
    )

    if trainer.optimizer_cls_and_kwargs is not None:
        optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
    else:
        optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(trainer.args, opt_model)
    optimizer_kwargs = dict(optimizer_kwargs)
    for key in ("params", "model", "optimizer_dict"):
        if key in optimizer_kwargs:
            optimizer_kwargs.pop(key)

    trainer.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                manager.register_module_override(module, "weight", {"optim_bits": 32})
        logger.info("Adam8bit embedding fp32 override: skipped=%sM params", skipped / 2**20)

    return trainer.optimizer


def selection_from_component_parameters(
    *,
    decoder_parameters: Optional[Dict[str, nn.Parameter]] = None,
    lora_parameters: Optional[Dict[str, nn.Parameter]] = None,
    norm_parameters: Optional[Dict[str, nn.Parameter]] = None,
    lm_head_parameters: Optional[Dict[str, nn.Parameter]] = None,
) -> ModelLevelTrainableSelection:
    """Assemble a selection from already-labeled component maps (no name guessing)."""
    selection = ModelLevelTrainableSelection(
        decoder_parameters=dict(decoder_parameters or {}),
        lora_parameters=dict(lora_parameters or {}),
        norm_parameters=dict(norm_parameters or {}),
        lm_head_parameters=dict(lm_head_parameters or {}),
    )
    assert_disjoint_component_inventories(
        decoder_parameters=selection.decoder_parameters,
        lora_parameters=selection.lora_parameters,
        norm_parameters=selection.norm_parameters,
        lm_head_parameters=selection.lm_head_parameters,
    )
    return selection


def attach_model_level_optimizer_contract(
    trainer,
    *,
    selection: ModelLevelTrainableSelection,
    lr_config: ModelLevelOptimizerLRConfig,
) -> None:
    """Attach inventories + LR contract for create_model_level_optimizer."""
    trainer.model_level_trainable_selection = selection
    trainer.model_level_optimizer_lr_config = lr_config
    # Keep legacy attrs in sync for Sparse Bit / logging paths that still read them.
    trainer.decoder_param_ids = frozenset(id(p) for p in selection.decoder_parameters.values())
    trainer.decoder_lr = float(lr_config.resolved_decoder_lr()) if selection.decoder_parameters else None
    trainer.norm_lr = None if lr_config.norm_lr is None else float(lr_config.norm_lr)
    trainer.lm_head_lr = None if lr_config.lm_head_lr is None else float(lr_config.lm_head_lr)


__all__ = [
    "GROUP_DECODER",
    "GROUP_LM_HEAD",
    "GROUP_LORA",
    "GROUP_NORM",
    "ModelLevelOptimizerLRConfig",
    "attach_model_level_optimizer_contract",
    "build_model_level_param_groups",
    "create_model_level_optimizer",
    "selection_from_component_parameters",
]
