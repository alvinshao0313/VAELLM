"""Optimizer-step exact Sparse Bit checkpoint sidecar.

This path is intentionally separate from ``checkpoint.py``.  The legacy
packed+coverage sidecar restores a finalized/round-boundary state and resets live
round state; this module preserves the complete live optimizer-step state.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict

import torch

from .checkpoint import SIDE_CAR_DIR

EXACT_STATE_FILE = "exact_state.pt"
EXACT_FORMAT = "sparse_bit_tuning_exact_sidecar"
EXACT_VERSION = 1


def exact_state_path(checkpoint_dir: str) -> str:
    return os.path.join(str(checkpoint_dir), SIDE_CAR_DIR, EXACT_STATE_FILE)


def exact_sidecar_complete(checkpoint_dir: str) -> bool:
    return os.path.isfile(exact_state_path(checkpoint_dir))


def _atomic_torch_save(payload: Any, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def save_exact_sidecar(checkpoint_dir: str, manager) -> str:
    if manager is None or not callable(getattr(manager, "exact_state_dict", None)):
        raise TypeError("save_exact_sidecar requires a SparseBitTuningManager-like object.")
    state = manager.exact_state_dict()
    payload: Dict[str, Any] = {
        "format": EXACT_FORMAT,
        "version": EXACT_VERSION,
        "state": state,
    }
    path = exact_state_path(checkpoint_dir)
    _atomic_torch_save(payload, path)
    return path


def load_exact_sidecar(checkpoint_dir: str) -> dict:
    path = exact_state_path(checkpoint_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sparse Bit exact resume requires sidecar: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Sparse Bit exact sidecar must contain a dict, got {type(payload)}.")
    if str(payload.get("format")) != EXACT_FORMAT or int(payload.get("version", -1)) != EXACT_VERSION:
        raise ValueError(
            "unsupported Sparse Bit exact sidecar format/version: "
            f"{payload.get('format')!r}/{payload.get('version')!r}."
        )
    state = payload.get("state")
    if not isinstance(state, dict):
        raise TypeError("Sparse Bit exact sidecar 'state' must be a dict.")
    return state


def restore_exact_sidecar(checkpoint_dir: str, manager) -> None:
    if manager is None or not callable(getattr(manager, "load_exact_state_dict", None)):
        raise TypeError("restore_exact_sidecar requires a SparseBitTuningManager-like object.")
    manager.load_exact_state_dict(load_exact_sidecar(checkpoint_dir))


__all__ = [
    "EXACT_STATE_FILE",
    "EXACT_FORMAT",
    "EXACT_VERSION",
    "exact_state_path",
    "exact_sidecar_complete",
    "save_exact_sidecar",
    "load_exact_sidecar",
    "restore_exact_sidecar",
]
