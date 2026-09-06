"""CPU payload transport primitives for inline CAT distributed training."""

import io
import os
from datetime import timedelta
from typing import Any, Dict, Optional

import torch

from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool
from train_utils.lora_utils import (
    distill_rank,
    distill_world_size,
    ensure_distill_process_group_initialized,
)


_TRANSPORT_BOOL_MARKER = "__cat_inline_transport_bool__"
_PAYLOAD_GROUP = None


def _resolve_cat_inline_vae_wait_timeout_sec() -> int:
    raw = os.environ.get("CAT_INLINE_VAE_WAIT_TIMEOUT_SEC", "7200")
    try:
        timeout_sec = int(raw)
    except ValueError as exc:
        raise ValueError(
            "CAT_INLINE_VAE_WAIT_TIMEOUT_SEC must be a positive integer number of seconds, "
            f"got {raw!r}."
        ) from exc
    if timeout_sec <= 0:
        raise ValueError(
            "CAT_INLINE_VAE_WAIT_TIMEOUT_SEC must be > 0, "
            f"got {timeout_sec}."
        )
    return timeout_sec


def initialize_cat_payload_group():
    """Create and cache the world-wide Gloo group used only for CPU payload bytes."""
    global _PAYLOAD_GROUP
    if distill_world_size() <= 1:
        return None
    ensure_distill_process_group_initialized()
    if _PAYLOAD_GROUP is None:
        timeout_sec = _resolve_cat_inline_vae_wait_timeout_sec()
        _PAYLOAD_GROUP = torch.distributed.new_group(
            backend="gloo",
            timeout=timedelta(seconds=timeout_sec),
        )
    return _PAYLOAD_GROUP


def _pack_bool_tensors_for_transport(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.dtype != torch.bool:
            return obj
        logical_shape = tuple(int(v) for v in obj.shape)
        if not logical_shape:
            raise ValueError("CAT inline transport does not support scalar bool tensors.")
        return {
            _TRANSPORT_BOOL_MARKER: True,
            "logical_shape": logical_shape,
            "data": pack_bool_tensor_to_uint8(obj.to(device="cpu"), logical_shape=logical_shape),
        }
    if isinstance(obj, dict):
        return {key: _pack_bool_tensors_for_transport(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_pack_bool_tensors_for_transport(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_pack_bool_tensors_for_transport(value) for value in obj)
    return obj


def _unpack_bool_tensors_from_transport(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(_TRANSPORT_BOOL_MARKER) is True:
        if set(obj) != {_TRANSPORT_BOOL_MARKER, "logical_shape", "data"}:
            raise ValueError("Invalid CAT inline packed bool transport marker.")
        return unpack_uint8_tensor_to_bool(
            obj["data"].to(device="cpu"), logical_shape=obj["logical_shape"]
        )
    if isinstance(obj, dict):
        return {key: _unpack_bool_tensors_from_transport(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_unpack_bool_tensors_from_transport(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_unpack_bool_tensors_from_transport(value) for value in obj)
    return obj


def _broadcast_cpu_dict(
    payload: Optional[Dict[str, Any]],
    *,
    src: int,
    expected_format: str,
    expected_version: int,
    label: str,
) -> Dict[str, Any]:
    world_size = distill_world_size()
    if world_size <= 1:
        if not isinstance(payload, dict):
            raise TypeError(f"Single-rank {label} payload must be a dict.")
        return payload

    group = initialize_cat_payload_group()
    rank = distill_rank()
    if rank == int(src):
        if not isinstance(payload, dict):
            raise TypeError(f"Source {label} payload must be a dict.")
        buffer = io.BytesIO()
        torch.save(_pack_bool_tensors_for_transport(payload), buffer)
        raw = buffer.getvalue()
        length = torch.tensor([len(raw)], dtype=torch.int64, device="cpu")
    else:
        if payload is not None:
            raise TypeError(f"Non-source {label} payload must be None.")
        raw = b""
        length = torch.empty(1, dtype=torch.int64, device="cpu")

    torch.distributed.broadcast(length, src=int(src), group=group)
    byte_count = int(length.item())
    if byte_count < 1:
        raise RuntimeError(f"Invalid {label} payload byte length: {byte_count}.")
    if rank == int(src):
        byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    else:
        byte_tensor = torch.empty(byte_count, dtype=torch.uint8, device="cpu")
    torch.distributed.broadcast(byte_tensor, src=int(src), group=group)
    received = torch.load(
        io.BytesIO(byte_tensor.numpy().tobytes()), map_location="cpu", weights_only=False
    )
    result = _unpack_bool_tensors_from_transport(received)
    if not isinstance(result, dict):
        raise TypeError(f"Received {label} payload must be a dict, got {type(result)}.")
    if result.get("format") != expected_format or int(result.get("version", 0)) != expected_version:
        raise ValueError(f"Received invalid {label} payload format/version.")
    if rank == int(src):
        serialized_mb = byte_count / (1024.0 * 1024.0)
        print(
            f"{label} broadcast: world_size={world_size} serialized_mb={serialized_mb:.2f}",
            flush=True,
        )
    return result


def broadcast_group_vae_payload(payload: Optional[Dict[str, Any]], *, src: int = 0) -> Dict[str, Any]:
    """Broadcast a group VAE payload with bool tensors bit-packed for transport only."""
    return _broadcast_cpu_dict(
        payload,
        src=int(src),
        expected_format="vaellm_group_vae_payload",
        expected_version=1,
        label="CAT VAE payload",
    )


def broadcast_adaptive_channel_plan(payload: Optional[Dict[str, Any]], *, src: int = 0) -> Dict[str, Any]:
    """Broadcast a CPU-serializable adaptive channel plan. Single-rank returns locally."""
    return _broadcast_cpu_dict(
        payload,
        src=int(src),
        expected_format="vaellm_adaptive_channel_plan",
        expected_version=1,
        label="CAT adaptive channel plan",
    )
