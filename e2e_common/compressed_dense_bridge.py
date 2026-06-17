import gc
import json
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from e2e_common.checkpoint_io import load_e2e_model_checkpoint
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import NamedVAELinearDecodeTarget, decode_named_vae_linear_weights
from train_utils.model_checkpoint_io import META_FILENAME, resolve_checkpoint_dir


def _resolve_reference_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


def _iter_named_vae_linears(model: nn.Module) -> Iterable[Tuple[str, VAELinear]]:
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            yield str(name), module


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
        raise ValueError("Compressed dense bridge only accepts compact checkpoints without adapter modules.")


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

    resolved_device: Optional[str] = None
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


@torch.no_grad()
def materialize_vae_linears_to_dense(
    model: nn.Module,
    *,
    group_size: int = 8,
    compute_device: Optional[object] = "cpu",
    logger=None,
) -> int:
    vae_refs = list(_iter_named_vae_linears(model))
    if not vae_refs:
        return 0

    decode_targets = [
        NamedVAELinearDecodeTarget(
            name=name,
            base_layer=module,
            target_dtype=_resolve_reference_dtype(module),
        )
        for name, module in vae_refs
    ]
    if logger is not None:
        logger.info(
            "Start dense rebuild from VAELinear: total=%d group_size=%d compute_device=%s",
            len(decode_targets),
            int(group_size),
            str(compute_device),
        )
    decoded_results = decode_named_vae_linear_weights(
        decode_targets,
        group_size=int(group_size),
        compute_device=compute_device,
        logger=logger,
        respect_cache_policy=False,
    )
    decoded_by_name = {item.name: item for item in decoded_results}
    if len(decoded_by_name) != len(vae_refs):
        raise RuntimeError(
            f"Dense rebuild decode count mismatch: decoded={len(decoded_by_name)} expected={len(vae_refs)}."
        )

    converted = 0
    for name, old_module in vae_refs:
        decoded = decoded_by_name[name]
        dense_linear = nn.Linear(
            int(old_module.in_features),
            int(old_module.out_features),
            bias=old_module.bias is not None,
            device=decoded.decoded_weight.device,
            dtype=decoded.decoded_weight.dtype,
        )
        dense_linear.weight.copy_(decoded.decoded_weight)
        if dense_linear.bias is not None and old_module.bias is not None:
            dense_linear.bias.copy_(
                old_module.bias.detach().to(device=dense_linear.bias.device, dtype=dense_linear.bias.dtype)
            )
        dense_linear.train(old_module.training)
        set_module_by_name(model, name, dense_linear)
        converted += 1

    gc.collect()
    if logger is not None:
        logger.info("Finished dense rebuild from VAELinear: converted=%d", converted)
    return converted


def build_dense_model_from_checkpoint(
    student_checkpoint_dir: str,
    *,
    access_token: Optional[str] = None,
    base_model_path: Optional[str] = None,
    logger=None,
    decode_group_size: int = 8,
    decode_device: str = "auto",
) -> Tuple[nn.Module, Dict[str, object], str]:
    resolved_decode_device = resolve_decode_device(decode_device)
    if logger is not None:
        logger.info(
            "Dense rebuild decode config: requested_device=%s resolved_device=%s group_size=%d",
            str(decode_device),
            resolved_decode_device,
            int(decode_group_size),
        )
    model, meta, resolved_dir = load_compressed_student_checkpoint(
        student_checkpoint_dir,
        access_token=access_token,
        base_model_path=base_model_path,
        logger=logger,
    )
    converted = materialize_vae_linears_to_dense(
        model,
        group_size=int(decode_group_size),
        compute_device=resolved_decode_device,
        logger=logger,
    )
    if logger is not None:
        logger.info("Dense student is ready: source=%s converted_modules=%d", resolved_dir, converted)
    return model, meta, resolved_dir
