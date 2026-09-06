"""Migration-only reader for pre-v6 VAELLM model checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
from torch import nn

from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from train_utils import _legacy_checkpoint_impl as _legacy_impl


@dataclass(frozen=True)
class LegacyCheckpointInspection:
    checkpoint_dir: str
    meta: Dict[str, Any]
    base_model_path: str


_REJECTED_TRUE_KEYS = (
    "lora_use_dora",
    "use_dora",
    "use_rslora",
    "lora_use_rslora",
    "channel_residual_vae",
    "residual_sparse",
)
_REJECTED_TEXT = ("compressed_subspace", "adalora", "block_vae_lora_layer")


def _walk(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def inspect_legacy_checkpoint(source: str) -> LegacyCheckpointInspection:
    checkpoint_dir = _legacy_impl.resolve_checkpoint_dir(source)
    meta_path = os.path.join(checkpoint_dir, _legacy_impl.META_FILENAME)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise TypeError("legacy checkpoint metadata must be a JSON object")
    if str(meta.get("format")) == "vaellm_model_checkpoint_v6":
        raise ValueError("source is already a v6 checkpoint; migration is not required")

    for key, value in _walk(meta):
        lowered_key = key.lower()
        lowered_value = str(value).lower() if isinstance(value, str) else ""
        if lowered_key in _REJECTED_TRUE_KEYS and (
            value is True or lowered_value in {"true", "channel_residual_vae", "residual_sparse"}
        ):
            raise ValueError(f"legacy checkpoint uses unsupported migration topology: {key}={value!r}")
        if any(token in lowered_key or token in lowered_value for token in _REJECTED_TEXT):
            if "block_vae_lora_final" not in lowered_value:
                raise ValueError(f"legacy checkpoint uses unsupported migration topology: {key}={value!r}")
        if lowered_key == "outlier_protect_mode" and lowered_value in {
            "channel_residual_vae",
            "residual_sparse",
        }:
            raise ValueError(f"legacy residual checkpoint migration is unsupported: {value}")

    base_model_path = str(meta.get("base_model_path") or "").strip()
    if not base_model_path:
        raise ValueError("legacy checkpoint metadata is missing base_model_path")
    return LegacyCheckpointInspection(
        checkpoint_dir=os.path.abspath(checkpoint_dir),
        meta=dict(meta),
        base_model_path=base_model_path,
    )


def load_legacy_checkpoint_for_migration(
    source: str,
    *,
    access_token: str | None = None,
) -> tuple[nn.Module, LegacyCheckpointInspection]:
    inspection = inspect_legacy_checkpoint(source)
    model, _meta, _load_result = _legacy_impl.load_model_checkpoint(
        inspection.checkpoint_dir,
        access_token=access_token,
        base_model_path=inspection.base_model_path,
        map_location="cpu",
        strict=True,
        preserve_original_weights_from_base=False,
    )
    return model, inspection


@torch.no_grad()
def normalize_legacy_model_for_v6(model: nn.Module) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Convert original-only VAELinear modules to frozen ordinary Linear modules."""
    original_only = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, VAELinear) or not bool(getattr(module, "always_use_original", False)):
            continue
        original = getattr(module, "original_weight", None)
        if not isinstance(original, torch.Tensor):
            raise ValueError(
                f"legacy original-only target {name!r} is missing original_weight and cannot be migrated"
            )
        linear = nn.Linear(
            int(module.in_features),
            int(module.out_features),
            bias=getattr(module, "bias", None) is not None,
            device=original.device,
            dtype=original.dtype,
        )
        linear.weight.copy_(original)
        if linear.bias is not None:
            linear.bias.copy_(module.bias.detach().to(linear.bias))
        linear.requires_grad_(False)
        set_module_by_name(model, name, linear)
        original_only.append(str(name))

    compressed = tuple(
        sorted(name for name, module in model.named_modules() if isinstance(module, VAELinear))
    )
    return compressed, tuple(sorted(original_only))


__all__ = [
    "LegacyCheckpointInspection",
    "inspect_legacy_checkpoint",
    "load_legacy_checkpoint_for_migration",
    "normalize_legacy_model_for_v6",
]
