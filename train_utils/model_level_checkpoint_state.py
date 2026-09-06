"""Shared mutable model-state collection for exact model-level Trainer resume."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import build_mutable_state_manifest, validate_mutable_state_manifest
from train_utils.model_level_trainables import ModelLevelTrainableSelection


_COMPONENT_BY_INVENTORY = {
    "lora_parameters": "lora",
    "decoder_parameters": "decoder",
    "norm_parameters": "norm",
    "lm_head_parameters": "lm_head",
}


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


def collect_model_level_mutable_state(
    model: nn.Module,
    *,
    selection: Optional[ModelLevelTrainableSelection],
    selected_vae_modules: Sequence[Tuple[str, VAELinear]],
):
    """Collect all continuous mutable model state, excluding Sparse-Bit namespaces."""
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

    decoder_buffer_ids: set[int] = set()
    for _module_name, vae in selected_vae_modules:
        for decoder in _iter_decoder_modules(vae):
            for _rel_name, buf in decoder.named_buffers(recurse=True):
                decoder_buffer_ids.add(id(buf))
    named_buffers = dict(model.named_buffers())
    for buffer_id in sorted(decoder_buffer_ids):
        actual_name = buffer_name_by_id.get(buffer_id)
        if actual_name is None:
            raise RuntimeError("trainable decoder buffer is not reachable from model.named_buffers().")
        if actual_name in named_tensors and id(named_tensors[actual_name]) != buffer_id:
            raise RuntimeError(f"mutable state name collision at decoder buffer {actual_name!r}.")
        buffer_obj = named_buffers[actual_name]
        named_tensors[actual_name] = buffer_obj
        component_classes[actual_name] = "decoder"

    manifest = build_mutable_state_manifest(
        named_tensors,
        component_classes=component_classes,
    )
    return named_tensors, component_classes, manifest


@torch.no_grad()
def restore_model_level_mutable_state(
    model: nn.Module,
    *,
    selection: Optional[ModelLevelTrainableSelection],
    selected_vae_modules: Sequence[Tuple[str, VAELinear]],
    checkpoint_state: Mapping[str, Tensor],
    checkpoint_manifest: Sequence[Mapping[str, object]],
) -> None:
    current, component_classes, _manifest = collect_model_level_mutable_state(
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
    validate_mutable_state_manifest(checkpoint_manifest, checkpoint_state)
    for name, target in current.items():
        source = checkpoint_state[name]
        target.copy_(source.to(device=target.device, dtype=target.dtype))


__all__ = [
    "collect_model_level_mutable_state",
    "restore_model_level_mutable_state",
]
