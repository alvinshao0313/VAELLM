"""Serializable cross-category CAT runtime state for exact v6 resume."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from train_utils.activation_utils import ActivationCalibrationCache
from train_utils.channel_protection import (
    AdaptiveChannelPlan,
    deserialize_adaptive_channel_plan,
    serialize_adaptive_channel_plan,
)


FORMAT = "vaellm_cat_runtime_state_v6"
VERSION = 1

_VAE_IDENTITY_FIELDS = (
    "quantizer_type",
    "gamma0",
    "gamma",
    "zeta",
    "inv_temperature",
    "l1_weight",
    "lfq_weight",
    "commitment_loss_weight",
    "entropy_loss_weight",
    "lr",
    "weight_decay",
    "optimizer",
    "beta1",
    "beta2",
    "lr_scheduler",
    "lr_warmup_steps",
    "normalize_weight",
    "new_quant",
    "vae_weight_dtype",
    "vae_autocast_dtype",
    "vae_decoder_checkpoint",
)

_CAT_IDENTITY_FIELDS = (
    "seed",
    "deterministic",
    "batch_size",
    "gpu_resident_data",
    "linear_group_size",
    "allow_tail_group",
    "include_all_linears",
    "channel_protect_mode",
    "channel_scope",
    "channel_rank_metric",
    "channel_mlp_rank_metric",
    "channel_mlp_fuse_weights",
    "channel_axis",
    "channel_quant",
    "channel_protect_count_ratio",
    "channel_min_per_layer",
    "activation_calib_dataset",
    "activation_calib_nsamples",
    "activation_calib_seqlen",
    "activation_calib_seed",
    "activation_calib_device",
)


def _cpu_nested(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_nested(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported CAT runtime-state value type: {type(value)!r}.")


def _identity_value(value):
    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return _identity_value(value.to_jsonable())
    if is_dataclass(value):
        return _identity_value(asdict(value))
    if hasattr(value, "__dict__"):
        return _identity_value(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_identity_value(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported CAT runtime identity value type: {type(value)!r}.")


def build_cat_cross_category_runtime_identity(
    *,
    cat_args,
    vae_args,
    resolved_category_cfgs: Mapping[str, object],
    compression_categories: Sequence[str],
    target_layers,
    skip_layers,
    transpose_modules: Sequence[str],
) -> Dict[str, object]:
    category_cfgs = {
        str(category): _identity_value(cfg)
        for category, cfg in resolved_category_cfgs.items()
    }
    vae_shared = {
        field: _identity_value(getattr(vae_args, field))
        for field in _VAE_IDENTITY_FIELDS
        if hasattr(vae_args, field)
    }
    cat_shared = {
        field: _identity_value(getattr(cat_args, field))
        for field in _CAT_IDENTITY_FIELDS
        if hasattr(cat_args, field)
    }
    return {
        "format": "vaellm_cat_cross_category_identity_v1",
        "compression_categories": [str(value) for value in compression_categories],
        "target_layers": _identity_value(target_layers),
        "skip_layers": [
            [int(layer_idx), str(category)]
            for layer_idx, category in sorted(
                (int(layer_idx), str(category)) for layer_idx, category in skip_layers
            )
        ],
        "transpose_modules": [str(value) for value in transpose_modules],
        "vae_shared": vae_shared,
        "cat_shared": cat_shared,
        "resolved_category_runtime": category_cfgs,
    }


def _serialize_cache(cache: Optional[ActivationCalibrationCache]):
    if cache is None:
        return None
    if not isinstance(cache, ActivationCalibrationCache):
        raise TypeError(f"activation cache must be ActivationCalibrationCache, got {type(cache)}.")
    return {
        "dataset": str(cache.dataset),
        "model_path": str(cache.model_path),
        "nsamples": int(cache.nsamples),
        "seqlen": int(cache.seqlen),
        "seed": int(cache.seed),
        "input_ids": [_cpu_nested(tensor) for tensor in cache.input_ids],
    }


def _deserialize_cache(payload) -> Optional[ActivationCalibrationCache]:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError(f"activation cache payload must be mapping, got {type(payload)}.")
    input_ids = payload.get("input_ids")
    if not isinstance(input_ids, (list, tuple)):
        raise TypeError("activation cache input_ids must be a list/tuple.")
    tensors = []
    for idx, tensor in enumerate(input_ids):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"activation cache input_ids[{idx}] must be Tensor, got {type(tensor)}.")
        tensors.append(tensor.detach().to(device="cpu", dtype=torch.long).contiguous())
    return ActivationCalibrationCache(
        dataset=str(payload["dataset"]),
        model_path=str(payload["model_path"]),
        nsamples=int(payload["nsamples"]),
        seqlen=int(payload["seqlen"]),
        seed=int(payload["seed"]),
        input_ids=tensors,
    )


def serialize_activation_runtime(runtime: Optional[Mapping[str, object]]):
    if runtime is None:
        return None
    if not isinstance(runtime, Mapping):
        raise TypeError(f"activation_runtime must be mapping, got {type(runtime)}.")
    payload: Dict[str, object] = {}
    for key in ("dataset", "nsamples", "seqlen", "seed", "device", "log_every", "model_path"):
        if key in runtime:
            payload[key] = _cpu_nested(runtime.get(key))
    payload["cache"] = _serialize_cache(runtime.get("cache"))
    for key in ("stats_by_linear", "stats_by_mlp_block", "mlp_channel_plan_by_linear"):
        if key in runtime:
            payload[key] = _cpu_nested(runtime.get(key))
    return payload


def deserialize_activation_runtime(payload, *, access_token=None):
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError(f"serialized activation runtime must be mapping, got {type(payload)}.")
    runtime = dict(payload)
    runtime["cache"] = _deserialize_cache(payload.get("cache"))
    runtime["access_token"] = access_token
    return runtime


def build_cat_runtime_state(
    *,
    activation_runtime: Optional[Mapping[str, object]],
    global_adaptive_plan: Optional[AdaptiveChannelPlan],
    runtime_identity: Mapping[str, object],
) -> Dict[str, object]:
    if not isinstance(runtime_identity, Mapping):
        raise TypeError("runtime_identity must be a mapping.")
    return {
        "format": FORMAT,
        "version": VERSION,
        "runtime_identity": _cpu_nested(dict(runtime_identity)),
        "activation_runtime": serialize_activation_runtime(activation_runtime),
        "global_adaptive_plan": (
            None if global_adaptive_plan is None else serialize_adaptive_channel_plan(global_adaptive_plan)
        ),
    }


def restore_cat_runtime_state(
    payload: Mapping[str, object],
    *,
    access_token=None,
) -> Tuple[Optional[dict], Optional[AdaptiveChannelPlan], dict]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"CAT runtime state must be mapping, got {type(payload)}.")
    if payload.get("format") != FORMAT or int(payload.get("version", 0)) != VERSION:
        raise ValueError("Invalid CAT runtime state format/version.")
    identity = payload.get("runtime_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("CAT runtime state requires runtime_identity mapping.")
    activation_runtime = deserialize_activation_runtime(
        payload.get("activation_runtime"),
        access_token=access_token,
    )
    raw_plan = payload.get("global_adaptive_plan")
    global_plan = None
    if raw_plan is not None:
        if not isinstance(raw_plan, dict):
            raise TypeError("global_adaptive_plan payload must be dict when present.")
        global_plan = deserialize_adaptive_channel_plan(raw_plan)
    return activation_runtime, global_plan, dict(identity)


def validate_cat_runtime_identity(saved: Mapping[str, object], current: Mapping[str, object]) -> None:
    if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
        raise TypeError("CAT runtime identities must be mappings.")
    if _cpu_nested(dict(saved)) != _cpu_nested(dict(current)):
        raise ValueError(
            "CAT cross-category runtime identity mismatch. Activation/channel planning settings "
            "must match the checkpoint for exact resume."
        )


__all__ = [
    "build_cat_cross_category_runtime_identity",
    "build_cat_runtime_state",
    "restore_cat_runtime_state",
    "serialize_activation_runtime",
    "deserialize_activation_runtime",
    "validate_cat_runtime_identity",
]
