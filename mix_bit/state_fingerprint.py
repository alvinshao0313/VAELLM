"""Streaming SHA256 fingerprints for model state dicts.

Replaces the previous "clone the full state dict to CPU and compare tensors" approach
with a bounded-memory streaming fingerprint: each tensor is hashed in fixed-size CPU
chunks without ever materializing a full CPU copy of the state dict. The manifest only
stores strings and integers, so it can be written to disk and compared cheaply on reload.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from mix_bit.model_adapter import get_model_adapter
from mix_bit.schema import ResolvedRunConfig
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model


STATE_FINGERPRINT_KIND = "mix_bit_state_fingerprint_v1"
STATE_FINGERPRINT_CHUNK_BYTES = 16 * 1024 * 1024

STATE_FINGERPRINT_FILENAME = "state_fingerprint.json"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def fingerprint_tensor(
    tensor: torch.Tensor,
    *,
    chunk_bytes: int = STATE_FINGERPRINT_CHUNK_BYTES,
) -> dict[str, object]:
    """Return dtype/shape/numel/SHA metadata using bounded CPU chunks.

    The tensor must be a strided contiguous tensor; non-contiguous views and other
    layouts are rejected so we never silently materialize a contiguous copy. The full
    tensor is never cloned: only bounded CPU chunks are transferred per iteration.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"fingerprint_tensor requires a torch.Tensor, got {type(tensor)}")
    if tensor.layout != torch.strided:
        raise ValueError(
            f"Cannot fingerprint tensor with layout={tensor.layout!r}; only strided supported"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            "Cannot fingerprint non-contiguous tensor; pass a contiguous tensor to avoid "
            "materializing a full contiguous copy"
        )

    dtype_name = _dtype_name(tensor.dtype)
    shape = [int(s) for s in tensor.shape]
    numel = int(tensor.numel())

    hasher = hashlib.sha256()
    header = {"dtype": dtype_name, "shape": shape, "numel": numel}
    hasher.update(_canonical_json_bytes(header))

    bytes_per_element = max(1, int(tensor.element_size()))
    chunk_numel = max(1, int(chunk_bytes) // bytes_per_element)

    detached = tensor.detach()
    # view(-1) on a contiguous tensor is a view, not a copy.
    flat = detached.view(-1)

    start = 0
    while start < numel:
        end = min(numel, start + chunk_numel)
        chunk = flat[start:end]
        chunk_cpu = chunk.cpu()
        try:
            # Reinterpret raw bytes as uint8 so bfloat16 (no native numpy support)
            # is handled without casting the data.
            byte_view = chunk_cpu.view(torch.uint8)
            data_bytes = byte_view.numpy().tobytes(order="C")
        finally:
            del chunk
            del chunk_cpu
        hasher.update(data_bytes)
        del data_bytes
        start = end

    return {
        "dtype": dtype_name,
        "shape": shape,
        "numel": numel,
        "sha256": hasher.hexdigest(),
    }


def fingerprint_model_state(
    model: nn.Module,
    *,
    chunk_bytes: int = STATE_FINGERPRINT_CHUNK_BYTES,
) -> dict[str, object]:
    """Fingerprint every state_dict tensor without retaining tensor copies.

    Iterates ``model.state_dict()`` once, fingerprinting each tensor in bounded CPU
    chunks. The returned manifest only contains strings and integers; no tensor is
    retained after its fingerprint is computed.
    """
    state_dict = model.state_dict()
    entries: dict[str, dict[str, object]] = {}
    keys = list(state_dict.keys())
    for key in keys:
        tensor = state_dict[key]
        entries[key] = fingerprint_tensor(tensor, chunk_bytes=chunk_bytes)
        del tensor
    # Drop the state_dict reference explicitly so nothing retains the tensors.
    del state_dict
    return {
        "kind": STATE_FINGERPRINT_KIND,
        "chunk_bytes": int(chunk_bytes),
        "key_count": len(entries),
        "entries": entries,
    }


def compare_state_fingerprints(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> None:
    """Raise ValueError on kind/key/dtype/shape/numel/hash mismatch."""
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise ValueError(
            "State fingerprints must be mappings; "
            f"expected={type(expected).__name__} actual={type(actual).__name__}"
        )
    if expected.get("kind") != STATE_FINGERPRINT_KIND:
        raise ValueError(
            f"Expected fingerprint kind={STATE_FINGERPRINT_KIND!r}, "
            f"got expected.kind={expected.get('kind')!r}"
        )
    if actual.get("kind") != STATE_FINGERPRINT_KIND:
        raise ValueError(
            f"Expected fingerprint kind={STATE_FINGERPRINT_KIND!r}, "
            f"got actual.kind={actual.get('kind')!r}"
        )

    expected_entries = expected.get("entries")
    actual_entries = actual.get("entries")
    if not isinstance(expected_entries, Mapping) or not isinstance(actual_entries, Mapping):
        raise ValueError("State fingerprint entries must be mappings")

    expected_keys = set(expected_entries.keys())
    actual_keys = set(actual_entries.keys())
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"State fingerprint key set mismatch; missing={missing[:20]} unexpected={extra[:20]}"
        )

    if int(expected.get("key_count", -1)) != len(expected_entries):
        raise ValueError("Expected fingerprint key_count does not match entries length")
    if int(actual.get("key_count", -1)) != len(actual_entries):
        raise ValueError("Actual fingerprint key_count does not match entries length")
    if int(expected.get("key_count", -1)) != int(actual.get("key_count", -1)):
        raise ValueError(
            f"State fingerprint key_count mismatch: "
            f"expected={expected.get('key_count')} actual={actual.get('key_count')}"
        )

    for key in sorted(expected_keys):
        exp_entry = expected_entries[key]
        act_entry = actual_entries[key]
        if not isinstance(exp_entry, Mapping) or not isinstance(act_entry, Mapping):
            raise ValueError(f"State fingerprint entry {key!r} must be a mapping")
        for field in ("dtype", "shape", "numel", "sha256"):
            exp_val = exp_entry.get(field)
            act_val = act_entry.get(field)
            if exp_val != act_val:
                raise ValueError(
                    f"State fingerprint mismatch for {key!r}.{field}: "
                    f"expected={exp_val!r} actual={act_val!r}"
                )


def write_state_fingerprint_manifest(
    path: str | Path,
    payload: Mapping[str, object],
) -> str:
    """Write canonical JSON atomically and return the absolute path."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, manifest_path)
    return str(manifest_path.resolve())


def verify_saved_checkpoint_state(
    *,
    resolved: ResolvedRunConfig,
    output_dir: str | Path,
    expected_fingerprint: Mapping[str, object],
    expected_converted_module_names: Sequence[str],
) -> None:
    """Strict-load one saved checkpoint, compare fingerprints and reject original weights.

    Loads the base model via the profile adapter, applies the saved checkpoint with
    ``strict=True``, fingerprints the reloaded state, compares against
    ``expected_fingerprint``, and verifies every expected converted module is a
    ``VAELinear`` with ``original_weight is None``.
    """
    from litebsq.vae_linear import VAELinear

    profile = resolved.config.model_profile
    adapter = get_model_adapter(profile.adapter)
    reloaded = adapter.load_model(profile)
    try:
        load_v6_full_checkpoint_into_model(
            reloaded,
            str(output_dir),
            map_location="cpu",
            strict=True,
        )
        reload_fingerprint = fingerprint_model_state(reloaded)
        compare_state_fingerprints(expected_fingerprint, reload_fingerprint)
        for name in expected_converted_module_names:
            module = _get_module_by_name(reloaded, str(name))
            if not isinstance(module, VAELinear):
                raise TypeError(
                    f"{name}: expected VAELinear after reload, got {type(module).__name__}"
                )
            if module.original_weight is not None:
                raise ValueError(f"{name}: reloaded original_weight must be None")
    finally:
        del reloaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    module: nn.Module = model
    for part in str(name).split("."):
        module = getattr(module, part)
    return module
