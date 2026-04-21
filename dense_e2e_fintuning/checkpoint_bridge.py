import gc
import json
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from dense_e2e_fintuning.trainables import inject_dense_peft_adapters, resolve_target_layer_ids
from e2e_common.checkpoint_io import load_e2e_model_checkpoint
from e2e_common.post_norm_head import ensure_post_norm_head_linear
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import NamedVAELinearDecodeTarget, decode_named_vae_linear_weights
from rotation.model_utils import get_layers
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
        raise ValueError("dense_e2e_fintuning 首版只接受不带 adapter 的压缩 checkpoint。")


def resolve_base_model_path(meta: Dict[str, object], teacher_model_path: Optional[str] = None) -> str:
    explicit_path = None if teacher_model_path is None else str(teacher_model_path).strip()
    if explicit_path:
        return explicit_path
    base_model_path = meta.get("base_model_path")
    if base_model_path:
        return str(base_model_path)
    raise ValueError("Cannot resolve base model path from checkpoint meta or --teacher_model_path.")


def resolve_decode_device(requested: Optional[str]) -> str:
    normalized = "auto" if requested is None else str(requested).strip().lower()
    if normalized == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("decode_device=cuda requested, but CUDA is unavailable.")
        return "cuda:0"
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"decode_device={normalized} requested, but CUDA is unavailable.")
        try:
            device_idx = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid decode device '{requested}'.") from exc
        if device_idx < 0 or device_idx >= torch.cuda.device_count():
            raise ValueError(
                f"decode_device={normalized} is out of range for visible CUDA device count={torch.cuda.device_count()}."
            )
        return f"cuda:{device_idx}"
    raise ValueError(f"Invalid decode device '{requested}'.")


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


def rebuild_dense_peft_model_for_export(
    student_checkpoint_dir: str,
    *,
    access_token: Optional[str],
    args,
    training_args,
    state_dict: Dict[str, torch.Tensor],
    decode_group_size: int = 8,
    decode_device: str = "auto",
    logger=None,
) -> Tuple[nn.Module, Dict[str, object], object]:
    dense_model, meta, _resolved_dir = build_dense_model_from_checkpoint(
        student_checkpoint_dir,
        access_token=access_token,
        logger=logger,
        decode_group_size=int(decode_group_size),
        decode_device=decode_device,
    )
    if hasattr(dense_model, "config"):
        dense_model.config.use_cache = False
    if hasattr(dense_model, "enable_input_require_grads"):
        dense_model.enable_input_require_grads()
    if bool(getattr(args, "use_post_norm_head_linear", False)):
        ensure_post_norm_head_linear(dense_model)

    layers = list(get_layers(dense_model))
    decoder_layer_ids = resolve_target_layer_ids(getattr(args, "decoder_layer_ids", None), len(layers))
    peft_model, selection = inject_dense_peft_adapters(
        dense_model,
        args=args,
        decoder_layer_ids=decoder_layer_ids,
        total_step=int(training_args.max_steps),
    )
    load_result = peft_model.load_state_dict(state_dict, strict=True)
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        raise RuntimeError(
            f"Failed to rebuild dense export model from state_dict: "
            f"missing={getattr(load_result, 'missing_keys', [])} "
            f"unexpected={getattr(load_result, 'unexpected_keys', [])}"
        )
    return peft_model, meta, selection
