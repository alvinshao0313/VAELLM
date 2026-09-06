"""E2E v6 mutable-state and exact-resume contract helpers.

This module contains no Trainer lifecycle hooks.  It defines the model/runtime
state that an optimizer-step checkpoint owns, so save and resume share one truth.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import (
    build_mutable_state_manifest,
    validate_mutable_state_manifest,
)
from train_utils.distill_data import FORMATTING_VERSION, tokenizer_identity
from train_utils.model_level_trainables import ModelLevelTrainableSelection


_COMPONENT_BY_INVENTORY = {
    "lora_parameters": "lora",
    "decoder_parameters": "decoder",
    "norm_parameters": "norm",
    "lm_head_parameters": "lm_head",
}


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return str(value)


def _named_parameter_by_id(model: nn.Module) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, param in model.named_parameters():
        out.setdefault(id(param), str(name))
    return out


def _named_buffer_by_id(model: nn.Module) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, buf in model.named_buffers():
        out.setdefault(id(buf), str(name))
    return out


def _iter_decoder_modules(vae: VAELinear):
    packed = getattr(vae, "_parallel_stage_decoder", None)
    if isinstance(packed, nn.Module):
        yield packed
        return
    seen: set[int] = set()
    for stage_idx in range(int(getattr(vae, "residual_stages", 1))):
        for part_idx in range(int(getattr(vae, "parallel_parts", 1))):
            decoder = vae.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
            if id(decoder) in seen:
                continue
            seen.add(id(decoder))
            yield decoder


def collect_e2e_mutable_state(
    model: nn.Module,
    *,
    selection: Optional[ModelLevelTrainableSelection],
    selected_vae_modules: Sequence[Tuple[str, VAELinear]],
) -> Tuple[Dict[str, Tensor], Dict[str, str], list[dict]]:
    """Collect the exact continuous mutable model state, excluding Sparse-Bit scores."""
    param_name_by_id = _named_parameter_by_id(model)
    buffer_name_by_id = _named_buffer_by_id(model)
    named_tensors: Dict[str, Tensor] = {}
    component_classes: Dict[str, str] = {}
    selected_param_ids: set[int] = set()

    for inventory_name, component_class in _COMPONENT_BY_INVENTORY.items():
        inventory = {} if selection is None else getattr(selection, inventory_name)
        for logical_name, param in inventory.items():
            pid = id(param)
            if pid in selected_param_ids:
                raise RuntimeError(f"duplicate Parameter id across mutable inventories: {logical_name}")
            selected_param_ids.add(pid)
            actual_name = param_name_by_id.get(pid)
            if actual_name is None:
                raise RuntimeError(
                    f"mutable {component_class} parameter {logical_name!r} is not reachable from model.named_parameters()."
                )
            if actual_name.startswith("sparse_bit_tuning."):
                raise RuntimeError("Sparse Bit score parameter leaked into continuous mutable state.")
            named_tensors[actual_name] = param
            component_classes[actual_name] = component_class

    # Any continuous requires_grad parameter must be represented exactly once.
    for actual_name, param in model.named_parameters():
        if not bool(param.requires_grad):
            continue
        if str(actual_name).startswith("sparse_bit_tuning."):
            continue
        if id(param) in selected_param_ids:
            continue
        named_tensors[str(actual_name)] = param
        component_classes[str(actual_name)] = "other_trainable"
        selected_param_ids.add(id(param))

    # Decoder buffers can mutate during training (e.g. BatchNorm running stats).
    decoder_buffer_ids: set[int] = set()
    for _module_name, vae in selected_vae_modules:
        for decoder in _iter_decoder_modules(vae):
            for _rel_name, buf in decoder.named_buffers(recurse=True):
                decoder_buffer_ids.add(id(buf))
    for buffer_id in sorted(decoder_buffer_ids):
        actual_name = buffer_name_by_id.get(buffer_id)
        if actual_name is None:
            raise RuntimeError("trainable decoder buffer is not reachable from model.named_buffers().")
        if actual_name in named_tensors and id(named_tensors[actual_name]) != buffer_id:
            raise RuntimeError(f"mutable state name collision at decoder buffer {actual_name!r}.")
        buffer_obj = dict(model.named_buffers())[actual_name]
        named_tensors[actual_name] = buffer_obj
        component_classes[actual_name] = "decoder"

    manifest = build_mutable_state_manifest(
        named_tensors,
        component_classes=component_classes,
    )
    return named_tensors, component_classes, manifest


@torch.no_grad()
def restore_e2e_mutable_state(
    model: nn.Module,
    *,
    selection: Optional[ModelLevelTrainableSelection],
    selected_vae_modules: Sequence[Tuple[str, VAELinear]],
    checkpoint_state: Mapping[str, Tensor],
    checkpoint_manifest: Sequence[Mapping[str, object]],
) -> None:
    current, component_classes, _manifest = collect_e2e_mutable_state(
        model,
        selection=selection,
        selected_vae_modules=selected_vae_modules,
    )
    validate_mutable_state_manifest(
        checkpoint_manifest,
        current,
        component_classes=component_classes,
    )
    if set(checkpoint_state) != set(current):
        raise ValueError(
            "training_model_state key mismatch after topology rebuild: "
            f"missing={sorted(set(current) - set(checkpoint_state))} "
            f"extra={sorted(set(checkpoint_state) - set(current))}"
        )
    # Validate checkpoint tensor shape/dtype through the same manifest before writing.
    validate_mutable_state_manifest(checkpoint_manifest, checkpoint_state)
    for name, target in current.items():
        source = checkpoint_state[name]
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def build_e2e_immutable_resume_contract(
    *,
    cfg,
    training_args,
    tokenizer,
    input_checkpoint_id: str,
    resolved_target_layers: Sequence[int],
    resolved_target_modules: Sequence[str],
    teacher_identity: Optional[Mapping[str, object]],
) -> dict:
    data = _jsonable(cfg.data)
    loss = _jsonable(cfg.loss)
    opt = _jsonable(cfg.opt)
    # logging/save cadence can change without altering optimizer/data/RNG math.
    if isinstance(opt, dict):
        opt.pop("logging_steps", None)
    runtime = _jsonable(cfg.runtime)
    tokenizer_name, tokenizer_revision = tokenizer_identity(tokenizer)
    eval_after_save = bool(getattr(cfg.runtime.evaluation, "eval_after_save", False))
    eval_strategy = str(getattr(training_args, "eval_strategy", "no") or "no").lower()
    evaluation_execution = {
        "eval_after_save": eval_after_save,
        "eval_strategy": eval_strategy,
        "eval_on_start": bool(getattr(training_args, "eval_on_start", False)),
    }
    if eval_after_save:
        evaluation_execution.update(
            {
                "save_strategy": str(getattr(training_args, "save_strategy", "no") or "no").lower(),
                "save_steps": _jsonable(getattr(training_args, "save_steps", None)),
            }
        )
    if eval_strategy != "no" or bool(getattr(training_args, "eval_on_start", False)):
        evaluation_execution.update(
            {
                "eval_steps": _jsonable(getattr(training_args, "eval_steps", None)),
                "eval_delay": _jsonable(getattr(training_args, "eval_delay", None)),
            }
        )
    return {
        "version": 1,
        "input_checkpoint_id": str(input_checkpoint_id),
        "train_mode": str(cfg.train_mode),
        "target_layers": [int(v) for v in resolved_target_layers],
        "target_modules": [str(v) for v in resolved_target_modules],
        "data": data,
        "loss": loss,
        "optimization": opt,
        "lora": _jsonable(cfg.lora),
        "aux": _jsonable(cfg.aux),
        "runtime": runtime,
        "tokenizer": {
            "identity": str(tokenizer_name),
            "revision": str(tokenizer_revision),
            "formatting_version": FORMATTING_VERSION,
        },
        "dataloader": {
            "num_workers": int(getattr(training_args, "dataloader_num_workers", 0) or 0),
            "drop_last": bool(getattr(training_args, "dataloader_drop_last", False)),
        },
        "distributed": {
            "world_size": int(getattr(training_args, "world_size", 1) or 1),
            "parallel_mode": str(cfg.runtime.parallel_mode),
            "layer_device_map": str(cfg.runtime.layer_device_map),
        },
        "precision": {
            "bf16": bool(getattr(training_args, "bf16", False)),
            "fp16": bool(getattr(training_args, "fp16", False)),
            "tf32": getattr(training_args, "tf32", None),
            "gradient_checkpointing": bool(getattr(training_args, "gradient_checkpointing", False)),
            "gradient_checkpointing_kwargs": _jsonable(
                getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
            ),
        },
        "evaluation_execution": evaluation_execution,
        "sparse_bit": {
            "active_ratio": float(cfg.bit_active_ratio),
            "optimizer": str(cfg.bit_optimizer),
            "bit_lr": str(cfg.bit_lr),
            "weight_decay": float(cfg.bit_weight_decay),
            "round_steps": str(cfg.bit_round_steps),
        },
        "teacher_identity": None if teacher_identity is None else _jsonable(dict(teacher_identity)),
    }


def validate_e2e_immutable_resume_contract(saved: Mapping[str, object], current: Mapping[str, object]) -> None:
    if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
        raise TypeError("immutable resume contracts must be mappings.")
    if _jsonable(dict(saved)) != _jsonable(dict(current)):
        raise ValueError(
            "E2E exact-resume immutable contract mismatch. "
            "Dataset/loss/optimizer/runtime/target/topology settings must match the saved checkpoint."
        )


__all__ = [
    "collect_e2e_mutable_state",
    "restore_e2e_mutable_state",
    "build_e2e_immutable_resume_contract",
    "validate_e2e_immutable_resume_contract",
]
