"""Native decoder/remaining-LoRA scope and complete component-owned restoration."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from e2e_common.full_lora import collect_exact_peft_lora_config, iter_named_peft_lora_layers
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import build_mutable_state_manifest
from train_utils.config.configs import AuxTrainableConfig
from train_utils.distill_precision import configure_distill_precision
from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state, restore_model_level_mutable_state
from train_utils.model_level_trainables import ModelLevelTrainableSelection, build_model_level_trainable_selection

from .settings import RecoverySettings
from .protection import Protection


@contextmanager
def isolated_initialization(seed: int):
    """Initialize one stage without inheriting or consuming a previous discarded tail."""
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Use the shared semantic uint32 seed without another local remapping.")
    host, numpy_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(host)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _names(names: Sequence[str], label: str) -> tuple[str, ...]:
    names = tuple(names)
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError(f"{label} must contain unique actual module names.")
    return names


@dataclass
class NativeSelection:
    selection: ModelLevelTrainableSelection
    decoder_targets: tuple[tuple[str, VAELinear], ...]
    dense_targets: tuple[str, ...]
    settings: RecoverySettings
    protection: Protection
    execution_targets: tuple

    @property
    def model(self):
        return self.selection.peft_model


def select_native_scope(model: nn.Module, *, settings: RecoverySettings,
                        prefix_targets: Sequence[str], current_targets: Sequence[str],
                        future_dense_targets: Sequence[str], precision,
                        lora_init_seed: int,
                        aux: AuxTrainableConfig | None = None,
                        fp32_components: Sequence[str] = ("decoder", "lora")) -> NativeSelection:
    """The model adapter supplies actual names; no layer count or suffix inference."""
    prefix = _names(prefix_targets, "prefix_targets")
    current = _names(current_targets, "current_targets")
    future = _names(future_dense_targets, "future_dense_targets")
    if not current or not set(current) <= set(prefix) or set(prefix) & set(future):
        raise ValueError("Current owners must be a nonempty prefix subset, disjoint from future dense owners.")
    for name in prefix:
        if not isinstance(model.get_submodule(name), VAELinear):
            raise TypeError(f"Native-only recovery requires a real VAELinear owner: {name}")
    for name in future:
        if type(model.get_submodule(name)) is not nn.Linear:
            raise TypeError(f"Future LoRA requires an ordinary dense Linear owner: {name}")
    if any(module.__class__.__name__ in ("CompactRVQLinear", "Fixed2RiceLinear") for module in model.modules()):
        raise NotImplementedError("Mixed compact recovery is implementation_blocked in this native-only protocol adapter.")
    names = (prefix if settings.decoder_scope == "prefix" else current) if "D" in settings.groups else ()
    p_names = (prefix if settings.protection_scope == "prefix" else current) if "P" in settings.groups else ()
    p_names = tuple(name for name in p_names if model.get_submodule(name).protected_input_indices is not None
        and model.get_submodule(name).protected_input_indices.numel())
    dense = future if "L" in settings.groups else ()
    targets = tuple((name, model.get_submodule(name)) for name in names)
    protection_modules = {name: model.get_submodule(name) for name in p_names}
    execution = tuple((name, model.get_submodule(name)) for name in prefix if name in set(names) | set(p_names))
    with isolated_initialization(lora_init_seed):
        selected = build_model_level_trainable_selection(model, aux=aux or AuxTrainableConfig(),
            compressed_modules=(), dense_target_modules=dense, decoder_modules=targets,
            rank=settings.lora_rank, alpha=settings.lora_alpha, dropout=settings.lora_dropout,
            rank_explicit=True, fp32_components=fp32_components, train_decoder=bool(targets),
            train_lora=bool(dense), freeze=True)
    protection = Protection(protection_modules)
    configure_distill_precision(selected, components=fp32_components, training_args=precision)
    actual_lora = tuple(name.removeprefix("base_model.model.") for name, _layer in iter_named_peft_lora_layers(selected.peft_model))
    if set(actual_lora) != set(dense):
        raise ValueError("Actual LoRA owner inventory differs from the selected future dense inventory.")
    if dense:
        actual = collect_exact_peft_lora_config(selected.peft_model, default_rank=settings.lora_rank,
            alpha=settings.lora_alpha, dropout=settings.lora_dropout)
        if (actual["rank"], actual["alpha"], actual["dropout"]) != (settings.lora_rank, settings.lora_alpha, settings.lora_dropout):
            raise ValueError("Actual PEFT rank/alpha/dropout differs from the fixed recovery recipe.")
    if any(parameter.dtype != torch.float32 for parameter in (*selected.decoder_parameters.values(), *selected.lora_parameters.values())):
        raise ValueError("Persist the actual FP32 decoder and LoRA masters.")
    return NativeSelection(selected, targets, tuple(dense), settings, protection, execution)


def capture_mutable(selected: NativeSelection):
    values, components, manifest = collect_model_level_mutable_state(selected.model,
        selection=selected.selection, selected_vae_modules=selected.decoder_targets)
    return ({name: value.detach().cpu().contiguous().clone() for name, value in values.items()},
        dict(components), manifest)


@torch.no_grad()
def restore_mutable(selected: NativeSelection, state: Mapping[str, torch.Tensor], manifest: Sequence[Mapping]) -> dict:
    restore_model_level_mutable_state(selected.model, selection=selected.selection,
        selected_vae_modules=selected.decoder_targets, checkpoint_state=state, checkpoint_manifest=manifest)
    actual, components, actual_manifest = capture_mutable(selected)
    if actual_manifest != list(manifest) or set(actual) != set(state):
        raise ValueError("The actual component-owned mutable inventory differs after reconstruction.")
    for name, value in actual.items():
        if value.dtype != state[name].dtype or not torch.equal(value, state[name].cpu()):
            raise ValueError(f"Mutable restoration differs from the saved master: {name}")
    return dict(restored_tensors=len(actual), component_counts={component: sum(value == component for value in components.values())
        for component in sorted(set(components.values()))}, manifest=build_mutable_state_manifest(actual, component_classes=components))
