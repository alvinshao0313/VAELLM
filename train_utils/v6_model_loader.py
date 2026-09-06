"""Canonical convenience loader for independently loadable v6 model checkpoints."""

from __future__ import annotations

from typing import Optional

from rotation.model_utils import get_model
from train_utils.checkpoint_v6 import (
    FULL_MODEL_KINDS,
    load_v6_full_checkpoint_into_model,
    load_v6_meta,
    resolve_v6_checkpoint_dir,
    validate_v6_meta,
)


def load_v6_model_checkpoint(
    checkpoint_path: str,
    *,
    access_token: Optional[str] = None,
    base_model_path: Optional[str] = None,
    map_location: str = "cpu",
    strict: bool = True,
    expected_kind: Optional[str] = None,
):
    checkpoint_dir = resolve_v6_checkpoint_dir(checkpoint_path)
    meta = validate_v6_meta(load_v6_meta(checkpoint_dir), expected_kind=expected_kind)
    kind = str(meta["checkpoint_kind"])
    if kind not in FULL_MODEL_KINDS:
        raise ValueError(
            f"Expected an independently loadable v6 checkpoint, got checkpoint_kind={kind!r}."
        )
    resolved_base = str(base_model_path or meta.get("base_model_path") or "").strip()
    if not resolved_base:
        raise ValueError(f"v6 checkpoint {checkpoint_dir} does not define base_model_path")
    model = get_model(resolved_base, access_token)
    return load_v6_full_checkpoint_into_model(
        model,
        checkpoint_dir,
        expected_kind=kind,
        map_location=map_location,
        strict=strict,
    )


__all__ = ["load_v6_model_checkpoint"]
