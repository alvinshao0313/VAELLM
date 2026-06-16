import json
import os
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from e2e_common.checkpoint_io import load_e2e_model_checkpoint
from train_utils.model_checkpoint_io import META_FILENAME, resolve_checkpoint_dir


def load_checkpoint_meta(student_checkpoint_dir: str) -> Tuple[str, Dict[str, object]]:
    resolved_dir = resolve_checkpoint_dir(student_checkpoint_dir)
    meta_path = os.path.join(resolved_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing checkpoint meta: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise TypeError(f"Checkpoint meta must be a dict, got {type(meta)}")
    return resolved_dir, meta


def checkpoint_has_adapters(meta: Dict[str, object]) -> bool:
    adapter_count = meta.get("adapter_module_count")
    if adapter_count is not None and int(adapter_count) > 0:
        return True
    adapter_modules = meta.get("adapter_modules", [])
    return isinstance(adapter_modules, list) and len(adapter_modules) > 0


def reject_checkpoint_with_adapters(meta: Dict[str, object]) -> None:
    if checkpoint_has_adapters(meta):
        raise ValueError("compressed e2e finetuning requires a compact checkpoint without adapter modules.")


def resolve_base_model_path(meta: Dict[str, object], teacher_model_path: Optional[str] = None) -> str:
    explicit_path = None if teacher_model_path is None else str(teacher_model_path).strip()
    if explicit_path:
        return explicit_path
    base_model_path = meta.get("base_model_path")
    if base_model_path:
        return str(base_model_path)
    raise ValueError("Cannot resolve base model path from checkpoint meta or --teacher_model_path.")


def _read_local_rank_from_env() -> Optional[int]:
    raw = os.environ.get("LOCAL_RANK")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid LOCAL_RANK value '{raw}'. Expected an integer.") from exc


def get_decode_device_diagnostics(requested: Optional[str]) -> Dict[str, object]:
    normalized = "auto" if requested is None else str(requested).strip().lower()
    local_rank = _read_local_rank_from_env()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_available = bool(torch.cuda.is_available())
    visible_cuda_count = int(torch.cuda.device_count()) if cuda_available else 0

    if normalized == "auto":
        if not cuda_available:
            resolved_device = "cpu"
        elif visible_cuda_count == 1:
            resolved_device = "cuda:0"
        else:
            if local_rank is None:
                raise ValueError(
                    "decode_device=auto requires LOCAL_RANK when the current process sees multiple CUDA devices."
                )
            if local_rank < 0 or local_rank >= visible_cuda_count:
                raise ValueError(
                    f"LOCAL_RANK={local_rank} is out of range for visible CUDA device count={visible_cuda_count}."
                )
            resolved_device = f"cuda:{local_rank}"
    elif normalized == "cpu":
        resolved_device = "cpu"
    elif normalized == "cuda":
        if not cuda_available:
            raise RuntimeError("decode_device=cuda requested, but CUDA is unavailable.")
        resolved_device = "cuda:0"
    elif normalized.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(f"decode_device={normalized} requested, but CUDA is unavailable.")
        try:
            device_idx = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid decode device '{requested}'.") from exc
        if device_idx < 0 or device_idx >= visible_cuda_count:
            raise ValueError(
                f"decode_device={normalized} is out of range for visible CUDA device count={visible_cuda_count}."
            )
        resolved_device = f"cuda:{device_idx}"
    else:
        raise ValueError(f"Invalid decode device '{requested}'.")

    return {
        "requested_device": normalized,
        "resolved_device": resolved_device,
        "local_rank": local_rank,
        "cuda_visible_devices": None if cuda_visible_devices is None else str(cuda_visible_devices),
        "visible_cuda_count": visible_cuda_count,
    }


def resolve_decode_device(requested: Optional[str]) -> str:
    diagnostics = get_decode_device_diagnostics(requested)
    resolved_device = diagnostics["resolved_device"]
    if not isinstance(resolved_device, str):
        raise RuntimeError(f"Resolved decode device must be str, got {type(resolved_device)}")
    return resolved_device


def load_compressed_student_checkpoint(
    student_checkpoint_dir: str,
    *,
    access_token: Optional[str] = None,
    base_model_path: Optional[str] = None,
    logger=None,
) -> Tuple[nn.Module, Dict[str, object], str]:
    resolved_dir, meta = load_checkpoint_meta(student_checkpoint_dir)
    reject_checkpoint_with_adapters(meta)
    model, loaded_meta, _load_result = load_e2e_model_checkpoint(
        resolved_dir,
        access_token=access_token,
        base_model_path=base_model_path,
        map_location="cpu",
        strict=True,
        materialize_proxy_decoded_linears=False,
        proxy_logger=logger,
    )
    return model, loaded_meta, resolved_dir
