"""Shared protected-residual decoder registry.

This topology helper is intentionally independent of checkpoint format so online
CAT and v6 I/O can share one registry truth without importing legacy checkpoint
I/O.
"""

from __future__ import annotations

from torch import nn


SHARED_PROTECTED_RESIDUAL_DECODER_REGISTRY_ATTR = "_vaellm_shared_protected_residual_decoders"


def validate_shared_protected_residual_decoder_ref(ref: str) -> str:
    text = str(ref).strip()
    if not text:
        raise ValueError("shared protected residual decoder ref cannot be empty.")
    if "." in text:
        raise ValueError(f"shared protected residual decoder ref cannot contain '.': {text!r}")
    return text


def ensure_shared_protected_residual_decoder_registry(model: nn.Module) -> nn.ModuleDict:
    registry = getattr(model, SHARED_PROTECTED_RESIDUAL_DECODER_REGISTRY_ATTR, None)
    if registry is None:
        registry = nn.ModuleDict()
        setattr(model, SHARED_PROTECTED_RESIDUAL_DECODER_REGISTRY_ATTR, registry)
    if not isinstance(registry, nn.ModuleDict):
        raise TypeError(
            f"{SHARED_PROTECTED_RESIDUAL_DECODER_REGISTRY_ATTR} must be nn.ModuleDict, got {type(registry)}"
        )
    return registry


def get_shared_protected_residual_decoder_registry(model: nn.Module) -> nn.ModuleDict:
    return ensure_shared_protected_residual_decoder_registry(model)


def register_shared_protected_residual_decoder(model: nn.Module, ref: str, decoder: nn.Module) -> None:
    key = validate_shared_protected_residual_decoder_ref(ref)
    if not isinstance(decoder, nn.Module):
        raise TypeError(f"shared protected residual decoder must be nn.Module, got {type(decoder)}")
    registry = ensure_shared_protected_residual_decoder_registry(model)
    existing = registry[key] if key in registry else None
    if existing is not None and existing is not decoder:
        raise ValueError(f"shared protected residual decoder ref already exists with a different module: {key}")
    registry[key] = decoder


__all__ = [
    "SHARED_PROTECTED_RESIDUAL_DECODER_REGISTRY_ATTR",
    "ensure_shared_protected_residual_decoder_registry",
    "get_shared_protected_residual_decoder_registry",
    "register_shared_protected_residual_decoder",
    "validate_shared_protected_residual_decoder_ref",
]
