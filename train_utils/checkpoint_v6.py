"""VAELLM v6 model checkpoint I/O."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from torch import Tensor, nn

from litebsq.autoencoder import Decoder
from litebsq.bitpack import validate_bitpack_u8_spec
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from train_utils.shared_protected_residual import (
    SHARED_PROTECTED_RESIDUAL_DECODER_REGISTRY_ATTR,
    ensure_shared_protected_residual_decoder_registry,
    validate_shared_protected_residual_decoder_ref,
)
from train_utils.distributed_guard import distributed_guarded_main


class _LMHeadWithPostNormLinear(nn.Module):
    """Local copy of e2e_common.post_norm_head.LMHeadWithPostNormLinear.

    Avoids importing ``e2e_common`` package ``__init__`` (which pulls legacy checkpoint I/O).
    """

    def __init__(self, lm_head: nn.Module):
        if not isinstance(lm_head, nn.Linear):
            raise TypeError(f"LMHeadWithPostNormLinear expects nn.Linear lm_head, got {type(lm_head)}")
        super().__init__()
        hidden_size = int(lm_head.in_features)
        if int(lm_head.out_features) <= 0:
            raise ValueError(f"Invalid lm_head out_features={lm_head.out_features}")
        self.post_norm_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.post_norm_linear.weight.copy_(
                torch.eye(hidden_size, dtype=self.post_norm_linear.weight.dtype)
            )
        self.lm_head = lm_head

    @property
    def weight(self):
        return self.lm_head.weight

    @property
    def bias(self):
        return self.lm_head.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.post_norm_linear(hidden_states))


def _is_post_norm_head_linear(module: Optional[nn.Module]) -> bool:
    if module is None or not isinstance(module, nn.Module):
        return False
    if isinstance(module, _LMHeadWithPostNormLinear):
        return True
    # Duck-type the e2e_common class without importing it.
    return (
        type(module).__name__ == "LMHeadWithPostNormLinear"
        and hasattr(module, "post_norm_linear")
        and hasattr(module, "lm_head")
    )


def has_post_norm_head_linear(model: nn.Module) -> bool:
    return _is_post_norm_head_linear(getattr(model, "lm_head", None))


def ensure_post_norm_head_linear(model: nn.Module) -> bool:
    lm_head = getattr(model, "lm_head", None)
    if _is_post_norm_head_linear(lm_head):
        return False
    if not isinstance(lm_head, nn.Linear):
        raise TypeError(f"Model lm_head must be nn.Linear to attach post-norm linear, got {type(lm_head)}")
    wrapped = _LMHeadWithPostNormLinear(lm_head)
    wrapped.train(lm_head.training)
    wrapped.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
    model.lm_head = wrapped
    return True


FORMAT_V6 = "vaellm_model_checkpoint_v6"
SCHEMA_VERSION = 6
CHECKPOINT_KINDS = ("training_step", "round_base", "category_boundary", "final_model")
FULL_MODEL_KINDS = ("round_base", "category_boundary", "final_model")
STATE_DICT_FILENAME = "pytorch_model.bin"
META_FILENAME = "checkpoint_meta.json"
TRAINING_MODEL_STATE_FILENAME = "training_model_state.pt"
CAT_RUNTIME_STATE_FILENAME = "cat_runtime_state.pt"
MUTABLE_STATE_MANIFEST_KEY = "mutable_state_manifest"
MUTABLE_COMPONENT_CLASSES = frozenset({"lora", "decoder", "norm", "lm_head", "other_trainable"})

_TMP_DIR_MARKER = ".tmp-"
_FORBIDDEN_EXTRA_META_RESUME_KEYS = frozenset({"resume_contract", "immutable_resume_contract"})
_REQUIRED_CONVERTED_MODULE_FIELDS = (
    "name",
    "in_features",
    "out_features",
    "codebook_dim",
    "transpose",
    "parallel_parts",
    "residual_stages",
    "has_bias",
    "vq_weights",
    "decoders",
)
_SPARSE_RESIDUAL_TENSOR_ATTRS = (
    "sparse_residual_row_indices",
    "sparse_residual_col_indices",
    "sparse_residual_values",
    "sparse_residual_active_block_ids",
    "sparse_residual_block_ptr",
    "sparse_residual_local_indices",
    "sparse_residual_qvalues",
    "sparse_residual_scales",
    "sparse_residual_zero_points",
)
def build_checkpoint_id() -> str:
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dtype_to_name(dtype: torch.dtype) -> str:
    text = str(dtype)
    if text.startswith("torch."):
        return text[len("torch.") :]
    return text


def _name_to_dtype(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype name: {name}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid torch dtype entry: {name}")
    return dtype


def _tensor_spec(tensor: Optional[Tensor]) -> Optional[Dict[str, Any]]:
    if not isinstance(tensor, Tensor):
        return None
    return {"shape": list(tensor.shape), "dtype": _dtype_to_name(tensor.dtype)}


def _as_str_list(values: Optional[Sequence[str]], *, field_name: str) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence of strings, got str")
    out: List[str] = []
    for item in values:
        text = str(item).strip()
        if not text:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        out.append(text)
    return out


def validate_target_inventories(
    compressed: Sequence[str],
    pending: Sequence[str],
    skip: Sequence[str],
) -> None:
    """Hard-error if compressed / pending_dense / skip inventories overlap."""
    compressed_set = set(_as_str_list(compressed, field_name="compressed_targets"))
    pending_set = set(_as_str_list(pending, field_name="pending_dense_targets"))
    skip_set = set(_as_str_list(skip, field_name="skip_targets"))

    overlap_cp = compressed_set & pending_set
    if overlap_cp:
        raise ValueError(f"compressed_targets and pending_dense_targets overlap: {sorted(overlap_cp)}")
    overlap_cs = compressed_set & skip_set
    if overlap_cs:
        raise ValueError(f"compressed_targets and skip_targets overlap: {sorted(overlap_cs)}")
    overlap_ps = pending_set & skip_set
    if overlap_ps:
        raise ValueError(f"pending_dense_targets and skip_targets overlap: {sorted(overlap_ps)}")


def iter_named_vae_linears(model: nn.Module) -> List[Tuple[str, VAELinear]]:
    """Return all ``(logical_name, VAELinear)`` pairs in module order."""
    out: List[Tuple[str, VAELinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            out.append((str(name), module))
    return out


def _is_peft_proxy_module(module: nn.Module) -> bool:
    if type(module).__name__ == "FullCompressedPeftProxy":
        return True
    return False


def _reject_live_training_adapter_topology(model: nn.Module) -> None:
    if isinstance(model, PeftModel):
        raise ValueError("stable full v6 checkpoint cannot save a live PeftModel wrapper.")
    offenders = [
        name or "<root>"
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer) or _is_peft_proxy_module(module)
    ]
    if offenders:
        raise ValueError(
            "stable full v6 checkpoint requires finalized LoRA topology; "
            f"live adapter/proxy modules remain: {offenders}."
        )


def _require_ordinary_linear_target(model: nn.Module, name: str, *, inventory_name: str) -> nn.Linear:
    try:
        module = _get_module_by_name(model, name)
    except AttributeError as exc:
        raise ValueError(
            f"{inventory_name} entry {name!r} cannot be resolved on model: {exc}"
        ) from exc
    if isinstance(module, VAELinear):
        raise ValueError(
            f"{inventory_name} entry {name!r} resolves to VAELinear; expected ordinary nn.Linear"
        )
    if _is_peft_proxy_module(module):
        raise ValueError(
            f"{inventory_name} entry {name!r} resolves to PEFT proxy {type(module).__name__}; "
            "expected ordinary nn.Linear"
        )
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"{inventory_name} entry {name!r} must resolve to nn.Linear (or Linear subclass), "
            f"got {type(module)}"
        )
    return module


def validate_model_target_inventories(
    model: nn.Module,
    compressed: Sequence[str],
    pending: Sequence[str],
    skip: Sequence[str],
    legacy_original_only_sources: Sequence[str] = (),
) -> None:
    """Validate target inventories against live model topology."""
    compressed_list = _as_str_list(compressed, field_name="compressed_targets")
    pending_list = _as_str_list(pending, field_name="pending_dense_targets")
    skip_list = _as_str_list(skip, field_name="skip_targets")
    legacy_list = _as_str_list(
        legacy_original_only_sources, field_name="legacy_original_only_sources"
    )
    validate_target_inventories(compressed_list, pending_list, skip_list)

    actual_names = {name for name, _ in iter_named_vae_linears(model)}
    compressed_set = set(compressed_list)
    if actual_names != compressed_set:
        missing = sorted(compressed_set - actual_names)
        extra = sorted(actual_names - compressed_set)
        raise ValueError(
            "compressed_targets must exactly match live VAELinear names; "
            f"missing={missing}, extra={extra}"
        )

    for name in pending_list:
        _require_ordinary_linear_target(model, name, inventory_name="pending_dense_targets")
    for name in skip_list:
        _require_ordinary_linear_target(model, name, inventory_name="skip_targets")

    skip_set = set(skip_list)
    legacy_set = set(legacy_list)
    if not legacy_set.issubset(skip_set):
        raise ValueError(
            "legacy_original_only_sources must be a subset of skip_targets; "
            f"offenders={sorted(legacy_set - skip_set)}"
        )


def _as_int_list(values: Optional[Sequence[Any]], *, field_name: str) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be null or a sequence of ints, got {type(values)}")
    out: List[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{field_name} entries must be ints, got {type(item)}")
        out.append(int(item))
    return out


def _validate_optional_lora_config(lora_config: Any) -> None:
    if lora_config is None:
        return
    if not isinstance(lora_config, dict):
        raise TypeError(f"lora_config must be a dict when present, got {type(lora_config)}")
    missing = sorted({"rank", "alpha", "dropout", "rank_pattern", "target_modules"} - set(lora_config))
    if missing:
        raise ValueError(f"training-step lora_config missing exact topology fields: {missing}.")
    if "rank" in lora_config and (
        isinstance(lora_config["rank"], bool) or not isinstance(lora_config["rank"], int)
    ):
        raise TypeError(f"lora_config.rank must be int when present, got {type(lora_config['rank'])}")
    if int(lora_config["rank"]) < 1:
        raise ValueError("lora_config.rank must be >= 1.")
    if "alpha" in lora_config and not isinstance(lora_config["alpha"], (int, float)):
        raise TypeError(f"lora_config.alpha must be a number when present, got {type(lora_config['alpha'])}")
    if "dropout" in lora_config and not isinstance(lora_config["dropout"], (int, float)):
        raise TypeError(
            f"lora_config.dropout must be a number when present, got {type(lora_config['dropout'])}"
        )
    rank_pattern = lora_config.get("rank_pattern")
    if rank_pattern is not None:
        if not isinstance(rank_pattern, dict):
            raise TypeError("lora_config.rank_pattern must be a dict when present.")
        for name, rank in rank_pattern.items():
            if not isinstance(name, str) or not name:
                raise TypeError("lora_config.rank_pattern keys must be non-empty strings.")
            if isinstance(rank, bool) or not isinstance(rank, int) or int(rank) < 1:
                raise ValueError(
                    f"lora_config.rank_pattern[{name!r}] must be a positive int, got {rank!r}."
                )
    target_modules = lora_config.get("target_modules")
    if target_modules is not None:
        normalized_targets = _as_str_list(target_modules, field_name="lora_config.target_modules")
        if len(normalized_targets) != len(set(normalized_targets)):
            raise ValueError("lora_config.target_modules must not contain duplicates.")
        if not set(rank_pattern or {}).issubset(set(normalized_targets)):
            raise ValueError("lora_config.rank_pattern keys must be a subset of target_modules.")


def _validate_optional_dict_field(payload: Mapping[str, Any], field_name: str) -> None:
    value = payload.get(field_name)
    if value is not None and not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict when present, got {type(value)}")


def _validate_optional_tensor_spec_field(
    raw: Mapping[str, Any],
    field_name: str,
    *,
    expected_ndim: int,
) -> None:
    value = raw.get(field_name)
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} tensor spec must be a dict when present, got {type(value)}")
    shape = value.get("shape")
    dtype = value.get("dtype")
    if not isinstance(shape, (list, tuple)) or len(shape) != int(expected_ndim):
        raise ValueError(f"{field_name} tensor spec shape must be {expected_ndim}D, got {shape!r}")
    if any(isinstance(v, bool) or not isinstance(v, int) or int(v) < 0 for v in shape):
        raise ValueError(f"{field_name} tensor spec shape must contain non-negative ints, got {shape!r}")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError(f"{field_name} tensor spec requires non-empty dtype")
    _name_to_dtype(str(dtype))


def _validate_extended_converted_module_spec(
    raw: Mapping[str, Any],
    *,
    module_name: str,
    shared_refs: set[str],
) -> None:
    tensor_ndims = {
        "protected_residual_indices": 1,
        "sparse_residual_row_indices": 1,
        "sparse_residual_col_indices": 1,
        "sparse_residual_values": 1,
        "sparse_residual_active_block_ids": 1,
        "sparse_residual_block_ptr": 1,
        "sparse_residual_local_indices": 1,
        "sparse_residual_qvalues": 1,
        "sparse_residual_scales": 1,
        "sparse_residual_zero_points": 1,
    }
    for field_name, ndim in tensor_ndims.items():
        _validate_optional_tensor_spec_field(raw, field_name, expected_ndim=ndim)

    stages = int(raw.get("protected_residual_stages") or 0)
    if stages < 0:
        raise ValueError(f"[{module_name}] protected_residual_stages must be >= 0")
    refs = raw.get("protected_residual_shared_decoder_refs")
    stage_vq = raw.get("protected_residual_stage_vq_weights")
    stage_decoders = raw.get("protected_residual_stage_decoders")
    stage_dims = raw.get("protected_residual_stage_codebook_dims")
    if stages == 0:
        if refs not in (None, []):
            raise ValueError(f"[{module_name}] shared protected residual refs require protected_residual_stages > 0")
        if stage_vq not in (None, []):
            raise ValueError(f"[{module_name}] protected residual VQ specs require protected_residual_stages > 0")
        if stage_decoders not in (None, []):
            raise ValueError(f"[{module_name}] protected residual decoder specs require protected_residual_stages > 0")
        return

    if not isinstance(stage_vq, (list, tuple)) or len(stage_vq) != stages:
        raise ValueError(
            f"[{module_name}] protected_residual_stage_vq_weights length must equal protected_residual_stages={stages}"
        )
    for idx, spec in enumerate(stage_vq):
        if not isinstance(spec, Mapping):
            raise TypeError(f"[{module_name}] protected residual VQ spec {idx} must be dict")
        validate_bitpack_u8_spec(dict(spec), arg_name=f"[{module_name}] protected_residual_stage_vq_weights[{idx}]")
    if stage_dims is not None:
        if not isinstance(stage_dims, (list, tuple)) or len(stage_dims) != stages:
            raise ValueError(
                f"[{module_name}] protected_residual_stage_codebook_dims length must equal {stages}"
            )
        if any(isinstance(v, bool) or not isinstance(v, int) or int(v) <= 0 for v in stage_dims):
            raise ValueError(f"[{module_name}] protected residual codebook dims must be positive ints")

    if refs is not None:
        if not isinstance(refs, (list, tuple)) or len(refs) != stages:
            raise ValueError(
                f"[{module_name}] protected_residual_shared_decoder_refs length must equal {stages}"
            )
        normalized_refs = [validate_shared_protected_residual_decoder_ref(ref) for ref in refs]
        missing = sorted(set(normalized_refs) - set(shared_refs))
        if missing:
            raise ValueError(f"[{module_name}] unknown shared protected residual decoder refs: {missing}")
        if stage_decoders not in (None, []):
            raise ValueError(
                f"[{module_name}] shared protected residual refs and inline decoder specs are mutually exclusive"
            )
    else:
        if not isinstance(stage_decoders, (list, tuple)) or len(stage_decoders) != stages:
            raise ValueError(
                f"[{module_name}] protected_residual_stage_decoders length must equal {stages}"
            )
        for idx, decoder_spec in enumerate(stage_decoders):
            if not isinstance(decoder_spec, Mapping):
                raise TypeError(f"[{module_name}] protected residual decoder spec {idx} must be dict")


def validate_full_converted_modules_meta(meta: Mapping[str, Any]) -> None:
    """Schema-validate full-checkpoint converted_modules and related fields before mutation."""
    converted = meta.get("converted_modules")
    if converted is None:
        converted = []
    if not isinstance(converted, list):
        raise TypeError(f"converted_modules must be a list, got {type(converted)}")
    count = meta.get("converted_module_count", len(converted))
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError(f"converted_module_count must be int, got {type(count)}")
    if int(count) != len(converted):
        raise ValueError(
            f"converted_module_count={count} != len(converted_modules)={len(converted)}"
        )

    shared_specs = _validate_shared_protected_residual_decoder_specs(
        meta.get("shared_protected_residual_decoders") or []
    )
    shared_refs = {str(item["ref"]) for item in shared_specs}

    seen_names: set[str] = set()
    for raw in converted:
        if not isinstance(raw, Mapping):
            raise TypeError(f"converted_modules entries must be dicts, got {type(raw)}")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("converted_modules entries require a non-empty string name")
        if name in seen_names:
            raise ValueError(f"duplicate converted_modules name: {name!r}")
        seen_names.add(name)
        for field_name in _REQUIRED_CONVERTED_MODULE_FIELDS:
            if field_name not in raw:
                raise ValueError(f"[{name}] missing required converted module field {field_name!r}")
        _validate_extended_converted_module_spec(
            raw,
            module_name=name,
            shared_refs=shared_refs,
        )

    _validate_optional_lora_config(meta.get("lora_config"))
    _validate_optional_dict_field(meta, "resolved_learning_rates")
    _validate_optional_dict_field(meta, "finalized_status")
    _validate_optional_dict_field(meta, "runtime_audit")
    if "post_norm_head_linear" in meta and not isinstance(meta.get("post_norm_head_linear"), bool):
        raise TypeError(
            f"post_norm_head_linear must be bool when present, got {type(meta.get('post_norm_head_linear'))}"
        )


def _normalize_component_class(component_class: str) -> str:
    text = str(component_class).strip()
    if not text:
        raise ValueError("component_class must be a non-empty string")
    lowered = text.lower()
    if "sparse" in lowered or "score" in lowered or "sparse_bit" in lowered:
        raise ValueError(
            f"component_class {text!r} is forbidden (sparse_bit/score are not mutable checkpoint classes)"
        )
    if text not in MUTABLE_COMPONENT_CLASSES:
        raise ValueError(
            f"Invalid component_class={text!r}; expected one of {sorted(MUTABLE_COMPONENT_CLASSES)}"
        )
    return text


def _tensor_storage_identity(tensor: Tensor) -> Tuple[Any, ...]:
    """Identity key that catches shared Parameter/Tensor storage aliases."""
    if tensor.numel() == 0:
        return ("empty", id(tensor), tuple(tensor.shape), _dtype_to_name(tensor.dtype))
    try:
        storage = tensor.untyped_storage()
        return (
            "storage",
            int(storage.data_ptr()),
            int(storage.size()),
            int(tensor.storage_offset()),
            tuple(tensor.size()),
            tuple(tensor.stride()),
            _dtype_to_name(tensor.dtype),
        )
    except (RuntimeError, AttributeError, TypeError):
        return ("id", id(tensor))


@dataclass
class V6CheckpointMeta:
    format: str = FORMAT_V6
    schema_version: int = SCHEMA_VERSION
    checkpoint_kind: str = "final_model"
    checkpoint_id: str = field(default_factory=build_checkpoint_id)
    created_at_utc: str = field(default_factory=_utc_now_iso)
    base_model_path: Optional[str] = None
    state_dict_file: Optional[str] = STATE_DICT_FILENAME
    train_mode: str = "none"
    after_category_mode: Optional[str] = None
    norm_train_mode: str = "none"
    lm_head_train_mode: str = "none"
    lora_config: Optional[Dict[str, Any]] = None
    resolved_learning_rates: Optional[Dict[str, Any]] = None
    compressed_targets: List[str] = field(default_factory=list)
    pending_dense_targets: List[str] = field(default_factory=list)
    skip_targets: List[str] = field(default_factory=list)
    legacy_original_only_sources: List[str] = field(default_factory=list)
    completed_categories: List[str] = field(default_factory=list)
    compression_categories: List[str] = field(default_factory=list)
    target_layers: Optional[List[int]] = None
    target_modules: List[str] = field(default_factory=list)
    immutable_resume_contract: Optional[Dict[str, Any]] = None
    finalized_status: Optional[Dict[str, Any]] = None
    runtime_audit: Optional[Dict[str, Any]] = None
    converted_modules: List[Dict[str, Any]] = field(default_factory=list)
    converted_module_count: int = 0
    shared_protected_residual_decoders: List[Dict[str, Any]] = field(default_factory=list)
    post_norm_head_linear: bool = False
    round_base_ref: Optional[str] = None
    round_base_checkpoint_id: Optional[str] = None
    mutable_state_manifest: Optional[List[Dict[str, Any]]] = None
    hf_artifact_refs: Optional[Dict[str, Any]] = None
    extra_meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Resume-critical schema fields stay present even when empty/null.
        if payload.get(MUTABLE_STATE_MANIFEST_KEY) is None:
            payload.pop(MUTABLE_STATE_MANIFEST_KEY, None)
        if payload.get("round_base_ref") is None:
            payload.pop("round_base_ref", None)
        if payload.get("round_base_checkpoint_id") is None:
            payload.pop("round_base_checkpoint_id", None)
        if payload.get("hf_artifact_refs") is None:
            payload.pop("hf_artifact_refs", None)
        if payload.get("extra_meta") is None:
            payload.pop("extra_meta", None)
        if self.checkpoint_kind == "training_step":
            payload.pop("state_dict_file", None)
        return payload

    def validate(self, *, expected_kind: Optional[str] = None) -> Dict[str, Any]:
        return validate_v6_meta(self.to_dict(), expected_kind=expected_kind)


def validate_v6_meta(meta: Mapping[str, Any], *, expected_kind: Optional[str] = None) -> Dict[str, Any]:
    """Validate v6 format/schema/kind. Never compares legacy numeric versions."""
    if not isinstance(meta, Mapping):
        raise TypeError(f"v6 meta must be a mapping, got {type(meta)}")
    payload = dict(meta)

    fmt = payload.get("format")
    if fmt != FORMAT_V6:
        raise ValueError(
            f"Unsupported checkpoint format={fmt!r}; expected {FORMAT_V6!r}. "
            "Legacy formats must be migrated before use with the v6 loader."
        )

    schema = payload.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema!r}; expected {SCHEMA_VERSION}. "
            "Do not compare against legacy numeric `version` fields."
        )

    kind = payload.get("checkpoint_kind")
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(
            f"Invalid checkpoint_kind={kind!r}; expected one of {CHECKPOINT_KINDS}."
        )
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"Expected checkpoint_kind={expected_kind!r}, got {kind!r}.")

    checkpoint_id = payload.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("checkpoint_id must be a non-empty string.")

    compressed = _as_str_list(payload.get("compressed_targets") or [], field_name="compressed_targets")
    pending = _as_str_list(payload.get("pending_dense_targets") or [], field_name="pending_dense_targets")
    skip = _as_str_list(payload.get("skip_targets") or [], field_name="skip_targets")
    validate_target_inventories(compressed, pending, skip)
    payload["compressed_targets"] = compressed
    payload["pending_dense_targets"] = pending
    payload["skip_targets"] = skip

    legacy_sources = _as_str_list(
        payload.get("legacy_original_only_sources") or [],
        field_name="legacy_original_only_sources",
    )
    if not set(legacy_sources).issubset(set(skip)):
        raise ValueError(
            "legacy_original_only_sources must be a subset of skip_targets; "
            f"offenders={sorted(set(legacy_sources) - set(skip))}"
        )
    payload["legacy_original_only_sources"] = legacy_sources

    compression_categories = _as_str_list(
        payload.get("compression_categories") or [],
        field_name="compression_categories",
    )
    payload["compression_categories"] = compression_categories

    target_layers = _as_int_list(payload.get("target_layers"), field_name="target_layers")
    payload["target_layers"] = target_layers

    target_modules = _as_str_list(
        payload.get("target_modules") or [],
        field_name="target_modules",
    )
    payload["target_modules"] = target_modules

    immutable_resume_contract = payload.get("immutable_resume_contract")
    if immutable_resume_contract is not None and not isinstance(immutable_resume_contract, dict):
        raise TypeError(
            f"immutable_resume_contract must be null or dict, got {type(immutable_resume_contract)}"
        )
    payload["immutable_resume_contract"] = immutable_resume_contract

    extra_meta = payload.get("extra_meta")
    if extra_meta is not None:
        if not isinstance(extra_meta, dict):
            raise TypeError(f"extra_meta must be a dict when present, got {type(extra_meta)}")
        forbidden = sorted(_FORBIDDEN_EXTRA_META_RESUME_KEYS & set(extra_meta.keys()))
        if forbidden:
            raise ValueError(
                "extra_meta must not contain resume-contract keys "
                f"{forbidden}; use top-level immutable_resume_contract instead."
            )

    if kind in FULL_MODEL_KINDS:
        if payload.get("lora_config") is not None:
            raise ValueError("stable full v6 checkpoint requires lora_config=null.")
        state_file = payload.get("state_dict_file", STATE_DICT_FILENAME)
        if state_file != STATE_DICT_FILENAME:
            raise ValueError(
                f"Full v6 checkpoints must use state_dict_file={STATE_DICT_FILENAME!r}, got {state_file!r}."
            )
        payload["state_dict_file"] = STATE_DICT_FILENAME
        validate_full_converted_modules_meta(payload)
    elif kind == "training_step":
        if not payload.get("round_base_ref"):
            raise ValueError("training_step meta requires round_base_ref.")
        if not payload.get("round_base_checkpoint_id"):
            raise ValueError("training_step meta requires round_base_checkpoint_id.")
        if MUTABLE_STATE_MANIFEST_KEY not in payload:
            raise ValueError(f"training_step meta requires {MUTABLE_STATE_MANIFEST_KEY}.")
        _validate_optional_lora_config(payload.get("lora_config"))
        _validate_optional_dict_field(payload, "resolved_learning_rates")
        _validate_optional_dict_field(payload, "finalized_status")
        _validate_optional_dict_field(payload, "runtime_audit")

    return payload


def _path_looks_like_tmp(path: str) -> bool:
    name = os.path.basename(os.path.abspath(path))
    return _TMP_DIR_MARKER in name and name.startswith(".")


def _call_barrier(distributed_barrier: Optional[Callable[[], None]]) -> None:
    if distributed_barrier is not None:
        distributed_barrier()


def _atomic_torch_save(obj: Any, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _atomic_write_json(payload: Mapping[str, Any], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _make_temp_sibling_dir(output_dir: str) -> str:
    abs_out = os.path.abspath(output_dir)
    parent = os.path.dirname(abs_out) or "."
    basename = os.path.basename(abs_out.rstrip(os.sep))
    if not basename:
        raise ValueError(f"Invalid output_dir: {output_dir!r}")
    os.makedirs(parent, exist_ok=True)
    tmp_dir = os.path.join(parent, f".{basename}{_TMP_DIR_MARKER}{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=False)
    return tmp_dir


def _publish_temp_dir(tmp_dir: str, output_dir: str) -> None:
    abs_out = os.path.abspath(output_dir)
    parent = os.path.dirname(abs_out) or "."
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(abs_out):
        raise FileExistsError(f"Checkpoint destination already exists: {abs_out}")
    os.replace(tmp_dir, abs_out)


def _validate_full_checkpoint_dir_contents(checkpoint_dir: str) -> None:
    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    state_path = os.path.join(checkpoint_dir, STATE_DICT_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Incomplete checkpoint: missing {META_FILENAME} under {checkpoint_dir}")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Incomplete checkpoint: missing {STATE_DICT_FILENAME} under {checkpoint_dir}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    validate_v6_meta(meta)
    kind = meta.get("checkpoint_kind")
    if kind not in FULL_MODEL_KINDS:
        raise ValueError(f"Directory is not a full v6 model checkpoint (kind={kind!r}): {checkpoint_dir}")
    extra_meta = meta.get("extra_meta")
    if isinstance(extra_meta, dict):
        runtime_file = extra_meta.get("cat_runtime_state_file")
        if runtime_file is not None:
            if str(runtime_file) != CAT_RUNTIME_STATE_FILENAME:
                raise ValueError(
                    f"Unsupported CAT runtime state filename in v6 metadata: {runtime_file!r}."
                )
            runtime_path = os.path.join(checkpoint_dir, CAT_RUNTIME_STATE_FILENAME)
            if not os.path.isfile(runtime_path):
                raise FileNotFoundError(
                    f"Incomplete checkpoint: missing {CAT_RUNTIME_STATE_FILENAME} under {checkpoint_dir}"
                )


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    module: nn.Module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _cpu_clone_nested(value):
    if isinstance(value, Tensor):
        return value.detach().to("cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_nested(item) for item in value)
    return value


def _collect_sparse_residual_specs(module: VAELinear) -> Dict[str, Any]:
    return {
        "sparse_residual_format": str(getattr(module, "sparse_residual_format", "coo_fp16")),
        "sparse_residual_index_bits": getattr(module, "sparse_residual_index_bits", None),
        "sparse_residual_value_bits": getattr(module, "sparse_residual_value_bits", None),
        "sparse_residual_block_rows": getattr(module, "sparse_residual_block_rows", None),
        "sparse_residual_block_cols": getattr(module, "sparse_residual_block_cols", None),
        "sparse_residual_row_indices": _tensor_spec(getattr(module, "sparse_residual_row_indices", None)),
        "sparse_residual_col_indices": _tensor_spec(getattr(module, "sparse_residual_col_indices", None)),
        "sparse_residual_values": _tensor_spec(getattr(module, "sparse_residual_values", None)),
        "sparse_residual_active_block_ids": _tensor_spec(getattr(module, "sparse_residual_active_block_ids", None)),
        "sparse_residual_block_ptr": _tensor_spec(getattr(module, "sparse_residual_block_ptr", None)),
        "sparse_residual_local_indices": _tensor_spec(getattr(module, "sparse_residual_local_indices", None)),
        "sparse_residual_qvalues": _tensor_spec(getattr(module, "sparse_residual_qvalues", None)),
        "sparse_residual_scales": _tensor_spec(getattr(module, "sparse_residual_scales", None)),
        "sparse_residual_zero_points": _tensor_spec(getattr(module, "sparse_residual_zero_points", None)),
    }


def _prepare_blocked_sparse_placeholder_for_rebuild(
    *,
    module_name: str,
    index_bits: Any,
    value_bits: Any,
    active_block_ids: Optional[Tensor],
    block_ptr: Optional[Tensor],
    local_indices: Optional[Tensor],
    qvalues: Optional[Tensor],
) -> None:
    if active_block_ids is None or block_ptr is None or local_indices is None or qvalues is None:
        return
    index_bits = int(index_bits)
    value_bits = int(value_bits)
    local_len = int(local_indices.numel())
    qvalues_len = int(qvalues.numel())
    if index_bits == 8:
        if local_len % 2 != 0:
            raise ValueError(
                f"[{module_name}] sparse_residual_local_indices length {local_len} is invalid for index_bits=8."
            )
        nnz = local_len // 2
    elif index_bits == 4:
        nnz = local_len
    else:
        raise ValueError(f"[{module_name}] unsupported sparse_residual_index_bits={index_bits}.")
    if value_bits == 8:
        expected_qvalues_len = nnz
    elif value_bits == 4:
        expected_qvalues_len = (nnz + 1) // 2
    else:
        raise ValueError(f"[{module_name}] unsupported sparse_residual_value_bits={value_bits}.")
    if qvalues_len != expected_qvalues_len:
        raise ValueError(
            f"[{module_name}] sparse_residual_qvalues length mismatch: got {qvalues_len}, expected {expected_qvalues_len}."
        )
    active_block_count = int(active_block_ids.numel())
    if int(block_ptr.numel()) != active_block_count + 1:
        raise ValueError(
            f"[{module_name}] sparse_residual_block_ptr length mismatch: got {int(block_ptr.numel())}, "
            f"expected {active_block_count + 1}."
        )
    if active_block_count == 0 and nnz != 0:
        raise ValueError(
            f"[{module_name}] sparse residual has nnz={nnz} but sparse_residual_active_block_ids is empty."
        )
    block_ptr.zero_()
    if int(block_ptr.numel()) > 1:
        block_ptr[1:] = int(nnz)


def _reject_always_use_original_on_save(model: nn.Module) -> None:
    offenders: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, VAELinear) and bool(getattr(module, "always_use_original", False)):
            offenders.append(name or "<root>")
    if offenders:
        raise ValueError(
            "v6 must not save VAELinear(always_use_original=True) as compressed. "
            f"Offending modules: {offenders}. "
            "Legacy original-only modules belong in skip_targets / migration (Task 14)."
        )


def _decoder_to_spec(decoder: Decoder) -> Dict[str, Any]:
    if not isinstance(decoder, Decoder):
        raise TypeError(f"Expected Decoder, got {type(decoder)}")
    if decoder.decoder_type not in {"linear", "symmetric", "asymmetric"}:
        raise ValueError(f"Unsupported decoder_type: {decoder.decoder_type}")
    first_param = next(decoder.parameters(), None)
    param_dtype = _dtype_to_name(first_param.dtype) if first_param is not None else "float32"
    return {
        "in_dim": int(decoder.in_dim),
        "out_dim": int(decoder.out_dim),
        "hidden_dim": int(getattr(decoder, "hidden_dim")),
        "num_res_blocks": int(getattr(decoder, "num_res_blocks")),
        "norm_type": str(getattr(decoder, "norm_type")),
        "activation_type": str(getattr(decoder, "activation_type", "swish")),
        "decoder_type": str(decoder.decoder_type),
        "use_checkpoint": bool(decoder.use_checkpoint),
        "param_dtype": param_dtype,
    }


def _build_decoder_from_spec(spec: Mapping[str, Any]) -> Decoder:
    decoder = Decoder(
        in_dim=int(spec["in_dim"]),
        out_dim=int(spec["out_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        num_res_blocks=int(spec["num_res_blocks"]),
        norm_type=str(spec["norm_type"]),
        activation_type=str(spec.get("activation_type", "swish")),
        decoder_type=str(spec["decoder_type"]),
        use_checkpoint=bool(spec["use_checkpoint"]),
        num_models=1,
    )
    param_dtype = spec.get("param_dtype")
    if param_dtype:
        decoder = decoder.to(dtype=_name_to_dtype(str(param_dtype)))
    return decoder


def _collect_shared_protected_residual_decoder_specs(model: nn.Module) -> List[Dict[str, Any]]:
    registry = getattr(model, SHARED_PROTECTED_RESIDUAL_DECODER_REGISTRY_ATTR, None)
    if registry is None:
        return []
    if not isinstance(registry, nn.ModuleDict):
        raise TypeError(
            f"{SHARED_PROTECTED_RESIDUAL_DECODER_REGISTRY_ATTR} must be nn.ModuleDict, got {type(registry)}"
        )
    return [
        {
            "ref": validate_shared_protected_residual_decoder_ref(ref),
            "decoder": _decoder_to_spec(decoder),
        }
        for ref, decoder in registry.items()
    ]


def _validate_shared_protected_residual_decoder_specs(raw_specs: Any) -> List[Dict[str, Any]]:
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        raise TypeError(
            f"shared_protected_residual_decoders must be a list, got {type(raw_specs)}"
        )
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in raw_specs:
        if not isinstance(item, Mapping):
            raise TypeError(
                f"shared_protected_residual_decoders entries must be dicts, got {type(item)}"
            )
        ref = validate_shared_protected_residual_decoder_ref(str(item.get("ref", "")))
        if ref in seen:
            raise ValueError(f"duplicate shared protected residual decoder ref: {ref}")
        seen.add(ref)
        decoder_spec = item.get("decoder")
        if not isinstance(decoder_spec, Mapping):
            raise ValueError(f"shared protected residual decoder {ref!r} is missing decoder spec.")
        required = {
            "in_dim",
            "out_dim",
            "hidden_dim",
            "num_res_blocks",
            "norm_type",
            "decoder_type",
            "use_checkpoint",
        }
        missing = sorted(required - set(decoder_spec))
        if missing:
            raise ValueError(f"shared protected residual decoder {ref!r} missing fields: {missing}")
        out.append({"ref": ref, "decoder": dict(decoder_spec)})
    return out


def _rebuild_shared_protected_residual_decoders(
    model: nn.Module,
    specs: Sequence[Mapping[str, Any]],
) -> Dict[str, nn.Module]:
    registry = ensure_shared_protected_residual_decoder_registry(model)
    registry.clear()
    for item in specs:
        ref = validate_shared_protected_residual_decoder_ref(str(item.get("ref", "")))
        decoder_spec = item.get("decoder")
        if not isinstance(decoder_spec, Mapping):
            raise ValueError(f"shared protected residual decoder {ref!r} is missing decoder spec.")
        registry[ref] = _build_decoder_from_spec(decoder_spec)
    return dict(registry.items())


def _collect_protected_residual_specs(module: VAELinear, *, module_name: str) -> Dict[str, Any]:
    stages = int(getattr(module, "protected_residual_stages", 0) or 0)
    payload: Dict[str, Any] = {
        "protected_residual_axis": getattr(module, "protected_residual_axis", None),
        "protected_residual_indices": _tensor_spec(getattr(module, "protected_residual_indices", None)),
        "protected_residual_stages": stages,
        "protected_residual_stage_codebook_dims": None,
        "protected_residual_parallel_stage_decode": bool(
            getattr(module, "_protected_residual_parallel_decoder", None) is not None
        ),
        "protected_residual_stage_vq_weights": None,
        "protected_residual_stage_decoders": None,
        "protected_residual_shared_decoder_refs": None,
    }
    if stages <= 0:
        return payload

    payload["protected_residual_stage_vq_weights"] = [
        validate_bitpack_u8_spec(
            module.get_protected_residual_stage_vq_spec(stage_idx),
            arg_name=f"[{module_name}] protected_residual_vq[{stage_idx}]",
        )
        for stage_idx in range(stages)
    ]
    payload["protected_residual_stage_codebook_dims"] = [
        int(v) for v in getattr(module, "protected_residual_stage_codebook_dims", [])
    ]
    if len(payload["protected_residual_stage_codebook_dims"]) != stages:
        raise ValueError(
            f"[{module_name}] protected_residual_stage_codebook_dims length "
            f"{len(payload['protected_residual_stage_codebook_dims'])} != protected_residual_stages {stages}"
        )

    raw_refs = getattr(module, "protected_residual_shared_decoder_refs", None)
    if raw_refs is not None:
        refs = [validate_shared_protected_residual_decoder_ref(ref) for ref in raw_refs]
        if len(refs) != stages:
            raise ValueError(
                f"[{module_name}] protected_residual_shared_decoder_refs length {len(refs)} != {stages}"
            )
        shared_objects = getattr(module, "_protected_residual_shared_stage_decoders", None)
        if shared_objects is None or len(shared_objects) != stages:
            raise ValueError(f"[{module_name}] missing shared protected residual decoder objects.")
        payload["protected_residual_shared_decoder_refs"] = refs
    else:
        payload["protected_residual_stage_decoders"] = [
            _decoder_to_spec(module.get_protected_residual_stage_decoder(stage_idx))
            for stage_idx in range(stages)
        ]
    return payload


def _vq_storage_spec_from_module(module: VAELinear, *, stage_idx: int, part_idx: int) -> Dict[str, Any]:
    spec = module.get_stage_part_vq_spec(stage_idx=stage_idx, part_idx=part_idx)
    return validate_bitpack_u8_spec(spec, arg_name=f"module_vq_spec[{stage_idx}][{part_idx}]")


def _validate_packed_vq_spec(spec: Mapping[str, Any], *, module_name: str, field_name: str) -> Dict[str, Any]:
    try:
        return validate_bitpack_u8_spec(dict(spec), arg_name=f"[{module_name}] {field_name}")
    except Exception as exc:
        raise ValueError(
            f"[{module_name}] only packed uint8 VQ checkpoint is supported for field {field_name}."
        ) from exc


def _make_vq_placeholders(vq_specs: Sequence[Dict[str, Any]], device: torch.device) -> List[Tensor]:
    tensors: List[Tensor] = []
    for spec in vq_specs:
        normalized = validate_bitpack_u8_spec(spec)
        shape = tuple(int(x) for x in normalized["shape"])
        dtype = _name_to_dtype(str(normalized["dtype"]))
        tensors.append(torch.zeros(shape, dtype=dtype, device=device))
    return tensors


def _build_unique_index_placeholder(shape: Sequence[int], *, dtype: torch.dtype, device) -> Tensor:
    if len(shape) != 1:
        raise ValueError(f"Index placeholder shape must be 1D, got {tuple(shape)}")
    return torch.arange(int(shape[0]), dtype=dtype, device=device)


def _ensure_bias_param(
    old_module: nn.Module,
    out_features: int,
    has_bias: bool,
) -> Optional[nn.Parameter]:
    if not has_bias:
        return None
    old_bias = getattr(old_module, "bias", None)
    if old_bias is not None:
        return nn.Parameter(torch.zeros_like(old_bias.detach()))
    old_weight = getattr(old_module, "weight", None)
    if old_weight is not None:
        return nn.Parameter(torch.zeros(out_features, dtype=old_weight.dtype, device=old_weight.device))
    return nn.Parameter(torch.zeros(out_features, dtype=torch.float32))


def _collect_vae_linear_specs(model: nn.Module) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, VAELinear):
            continue

        parallel_parts = int(module.parallel_parts)
        residual_stages = int(getattr(module, "residual_stages", 1))
        if residual_stages < 1:
            residual_stages = 1
        stage_codebook_dims = [int(v) for v in getattr(module, "stage_codebook_dims", [int(module.codebook_dim)])]
        if len(stage_codebook_dims) == 1 and residual_stages > 1:
            stage_codebook_dims = stage_codebook_dims * residual_stages
        if len(stage_codebook_dims) != residual_stages:
            raise ValueError(
                f"[{name}] stage_codebook_dims length {len(stage_codebook_dims)} != residual_stages {residual_stages}"
            )

        stage_vq_specs: List[Any] = []
        stage_decoder_specs: List[Any] = []
        for stage_idx in range(residual_stages):
            stage_vq_parts = []
            stage_decoder_parts = []
            for part_idx in range(parallel_parts):
                stage_vq_parts.append(_vq_storage_spec_from_module(module, stage_idx=stage_idx, part_idx=part_idx))
                stage_decoder_parts.append(_decoder_to_spec(module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)))
            if parallel_parts == 1:
                stage_vq_specs.append(stage_vq_parts[0])
                stage_decoder_specs.append(stage_decoder_parts[0])
            else:
                stage_vq_specs.append(stage_vq_parts)
                stage_decoder_specs.append(stage_decoder_parts)

        if parallel_parts == 1:
            vq_specs = [stage_vq_specs[0]]
            decoder_specs = [stage_decoder_specs[0]]
        else:
            vq_specs = list(stage_vq_specs[0])
            decoder_specs = list(stage_decoder_specs[0])

        spec: Dict[str, Any] = {
            "name": name,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "compressed_in_features": int(getattr(module, "compressed_in_features", module.in_features)),
            "compressed_out_features": int(getattr(module, "compressed_out_features", module.out_features)),
            "codebook_dim": int(module.codebook_dim),
            "transpose": bool(module.transpose),
            "parallel_parts": parallel_parts,
            "parallel_rows": int(getattr(module, "parallel_rows", parallel_parts)),
            "parallel_cols": int(getattr(module, "parallel_cols", 1)),
            "residual_stages": residual_stages,
            "stage_codebook_dims": stage_codebook_dims,
            "parallel_stage_decode": bool(getattr(module, "_parallel_stage_decoder", None) is not None),
            "has_bias": bool(module.bias is not None),
            "has_original_weight": bool(module.original_weight is not None),
            "always_use_original": bool(getattr(module, "always_use_original", False)),
            "protect_original_weight": bool(getattr(module, "protect_original_weight", False)),
            "vq_weights": vq_specs,
            "decoders": decoder_specs,
            "stage_vq_weights": stage_vq_specs if residual_stages > 1 else None,
            "stage_decoders": stage_decoder_specs if residual_stages > 1 else None,
            "protected_input_indices": _tensor_spec(getattr(module, "protected_input_indices", None)),
            "protected_input_weight": _tensor_spec(getattr(module, "protected_input_weight", None)),
            "protected_input_qvalues": _tensor_spec(getattr(module, "protected_input_qvalues", None)),
            "protected_input_scales": _tensor_spec(getattr(module, "protected_input_scales", None)),
            "protected_output_indices": _tensor_spec(getattr(module, "protected_output_indices", None)),
            "protected_output_weight": _tensor_spec(getattr(module, "protected_output_weight", None)),
            "protected_output_qvalues": _tensor_spec(getattr(module, "protected_output_qvalues", None)),
            "protected_output_scales": _tensor_spec(getattr(module, "protected_output_scales", None)),
            "protected_channel_quant_format": str(getattr(module, "protected_channel_quant_format", "none")),
            "low_rank_a": _tensor_spec(getattr(module, "low_rank_a", None)),
            "low_rank_b": _tensor_spec(getattr(module, "low_rank_b", None)),
            **_collect_sparse_residual_specs(module),
            **_collect_protected_residual_specs(module, module_name=name or "<root>"),
        }
        specs.append(spec)
    return specs


def _placeholder_from_tensor_spec(
    spec: Any,
    *,
    module_name: str,
    field_name: str,
    expected_ndim: int,
    default_dtype: str,
    device: torch.device,
    unique_index: bool = False,
) -> Optional[Tensor]:
    if not isinstance(spec, dict):
        return None
    shape = tuple(int(v) for v in spec.get("shape", []))
    if len(shape) != expected_ndim:
        raise ValueError(f"[{module_name}] {field_name} shape must be {expected_ndim}D, got {shape}")
    dtype = _name_to_dtype(str(spec.get("dtype", default_dtype)))
    if unique_index:
        return _build_unique_index_placeholder(shape, dtype=dtype, device=device)
    return torch.zeros(shape, dtype=dtype, device=device)


def _build_extended_vae_rebuild_kwargs(
    spec: Mapping[str, Any],
    *,
    module_name: str,
    device: torch.device,
    shared_protected_residual_decoders: Mapping[str, nn.Module],
) -> Dict[str, Any]:
    sparse_defaults = {
        "sparse_residual_row_indices": (1, "uint16"),
        "sparse_residual_col_indices": (1, "uint16"),
        "sparse_residual_values": (1, "float16"),
        "sparse_residual_active_block_ids": (1, "uint16"),
        "sparse_residual_block_ptr": (1, "int32"),
        "sparse_residual_local_indices": (1, "uint8"),
        "sparse_residual_qvalues": (1, "uint8"),
        "sparse_residual_scales": (1, "float16"),
        "sparse_residual_zero_points": (1, "float16"),
    }
    sparse_payload: Dict[str, Optional[Tensor]] = {}
    for field_name, (ndim, default_dtype) in sparse_defaults.items():
        sparse_payload[field_name] = _placeholder_from_tensor_spec(
            spec.get(field_name),
            module_name=module_name,
            field_name=field_name,
            expected_ndim=ndim,
            default_dtype=default_dtype,
            device=device,
        )
    if str(spec.get("sparse_residual_format", "")).strip().lower() == "blocked_quantized":
        _prepare_blocked_sparse_placeholder_for_rebuild(
            module_name=module_name,
            index_bits=spec.get("sparse_residual_index_bits"),
            value_bits=spec.get("sparse_residual_value_bits"),
            active_block_ids=sparse_payload["sparse_residual_active_block_ids"],
            block_ptr=sparse_payload["sparse_residual_block_ptr"],
            local_indices=sparse_payload["sparse_residual_local_indices"],
            qvalues=sparse_payload["sparse_residual_qvalues"],
        )

    protected_indices = _placeholder_from_tensor_spec(
        spec.get("protected_residual_indices"),
        module_name=module_name,
        field_name="protected_residual_indices",
        expected_ndim=1,
        default_dtype="int64",
        device=device,
        unique_index=True,
    )
    stages = int(spec.get("protected_residual_stages") or 0)
    stage_vq_payload = None
    stage_vq_storage_specs = None
    stage_decoders = None
    shared_refs = None
    shared_stage_decoders = None
    stage_codebook_dims = None
    if stages > 0:
        raw_vq_specs = spec.get("protected_residual_stage_vq_weights")
        if not isinstance(raw_vq_specs, (list, tuple)) or len(raw_vq_specs) != stages:
            raise ValueError(
                f"[{module_name}] invalid protected_residual_stage_vq_weights for stages={stages}"
            )
        normalized_vq_specs = [
            _validate_packed_vq_spec(
                one,
                module_name=module_name,
                field_name=f"protected_residual_stage_vq_weights[{idx}]",
            )
            for idx, one in enumerate(raw_vq_specs)
        ]
        stage_vq_payload = _make_vq_placeholders(normalized_vq_specs, device=device)
        stage_vq_storage_specs = [[one] for one in normalized_vq_specs]
        raw_dims = spec.get("protected_residual_stage_codebook_dims")
        stage_codebook_dims = [int(v) for v in raw_dims] if raw_dims is not None else None
        raw_refs = spec.get("protected_residual_shared_decoder_refs")
        if raw_refs is not None:
            shared_refs = [validate_shared_protected_residual_decoder_ref(ref) for ref in raw_refs]
            shared_stage_decoders = []
            for ref in shared_refs:
                decoder = shared_protected_residual_decoders.get(ref)
                if decoder is None:
                    raise ValueError(
                        f"[{module_name}] unknown shared protected residual decoder ref: {ref}"
                    )
                shared_stage_decoders.append(decoder)
        else:
            raw_decoder_specs = spec.get("protected_residual_stage_decoders")
            if not isinstance(raw_decoder_specs, (list, tuple)) or len(raw_decoder_specs) != stages:
                raise ValueError(
                    f"[{module_name}] invalid protected_residual_stage_decoders for stages={stages}"
                )
            stage_decoders = [_build_decoder_from_spec(one) for one in raw_decoder_specs]

    return {
        "sparse_residual_format": str(spec.get("sparse_residual_format", "coo_fp16")),
        "sparse_residual_index_bits": spec.get("sparse_residual_index_bits"),
        "sparse_residual_value_bits": spec.get("sparse_residual_value_bits"),
        "sparse_residual_block_rows": spec.get("sparse_residual_block_rows"),
        "sparse_residual_block_cols": spec.get("sparse_residual_block_cols"),
        **sparse_payload,
        "protected_residual_axis": spec.get("protected_residual_axis"),
        "protected_residual_indices": protected_indices,
        "protected_residual_stage_vq_weights": stage_vq_payload,
        "protected_residual_stage_vq_storage_specs": stage_vq_storage_specs,
        "protected_residual_stage_decoders": stage_decoders,
        "protected_residual_shared_decoder_refs": shared_refs,
        "protected_residual_shared_stage_decoders": shared_stage_decoders,
        "protected_residual_stage_codebook_dims": stage_codebook_dims,
    }


def _rebuild_converted_modules(
    model: nn.Module,
    converted_modules: Sequence[Mapping[str, Any]],
    *,
    shared_protected_residual_decoders: Optional[Mapping[str, nn.Module]] = None,
) -> None:
    shared_protected_residual_decoders = shared_protected_residual_decoders or {}
    for raw_spec in converted_modules:
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"converted_modules entries must be dicts, got {type(raw_spec)}")
        spec = dict(raw_spec)
        name = str(spec["name"])
        old_module = _get_module_by_name(model, name)
        weight = getattr(old_module, "weight", None)
        device = weight.device if weight is not None else torch.device("cpu")

        parallel_parts = int(spec["parallel_parts"])
        residual_stages = int(spec.get("residual_stages", 1))
        if residual_stages < 1:
            residual_stages = 1
        stage_codebook_dims_raw = spec.get("stage_codebook_dims")
        if isinstance(stage_codebook_dims_raw, (list, tuple)) and len(stage_codebook_dims_raw) > 0:
            stage_codebook_dims = [int(v) for v in stage_codebook_dims_raw]
            if len(stage_codebook_dims) == 1 and residual_stages > 1:
                stage_codebook_dims = stage_codebook_dims * residual_stages
            if len(stage_codebook_dims) != residual_stages:
                raise ValueError(
                    f"[{name}] stage_codebook_dims length {len(stage_codebook_dims)} != residual_stages {residual_stages}"
                )
        else:
            stage_codebook_dims = [int(spec["codebook_dim"]) for _ in range(residual_stages)]

        stage_vq_payload = None
        stage_vq_storage_specs = None
        stage_decoder_payload = None
        vq_payload = None
        vq_storage_specs = None
        decoder_payload = None

        if residual_stages > 1:
            stage_vq_specs = spec.get("stage_vq_weights")
            stage_decoder_specs = spec.get("stage_decoders")
            if not isinstance(stage_vq_specs, (list, tuple)):
                raise ValueError(f"[{name}] missing/invalid stage_vq_weights for residual_stages={residual_stages}")
            if not isinstance(stage_decoder_specs, (list, tuple)):
                raise ValueError(f"[{name}] missing/invalid stage_decoders for residual_stages={residual_stages}")
            if len(stage_vq_specs) != residual_stages or len(stage_decoder_specs) != residual_stages:
                raise ValueError(f"[{name}] stage_* length mismatch vs residual_stages={residual_stages}")

            stage_vq_payload = []
            stage_vq_storage_specs = []
            stage_decoder_payload = []
            for stage_idx in range(residual_stages):
                stage_vq_spec = stage_vq_specs[stage_idx]
                stage_decoder_spec = stage_decoder_specs[stage_idx]
                if parallel_parts == 1:
                    if not isinstance(stage_vq_spec, dict) or not isinstance(stage_decoder_spec, dict):
                        raise ValueError(f"[{name}] stage payloads must be dicts for single-part mode.")
                    normalized = _validate_packed_vq_spec(
                        stage_vq_spec, module_name=name, field_name=f"stage_vq_weights[{stage_idx}]"
                    )
                    stage_vq_payload.append(_make_vq_placeholders([normalized], device=device)[0])
                    stage_vq_storage_specs.append([normalized])
                    stage_decoder_payload.append(_build_decoder_from_spec(stage_decoder_spec))
                else:
                    if not isinstance(stage_vq_spec, (list, tuple)) or not isinstance(stage_decoder_spec, (list, tuple)):
                        raise ValueError(f"[{name}] stage payloads must be lists for parallel_parts={parallel_parts}.")
                    if len(stage_vq_spec) != parallel_parts or len(stage_decoder_spec) != parallel_parts:
                        raise ValueError(f"[{name}] stage part count mismatch for stage {stage_idx}.")
                    normalized_list = [
                        _validate_packed_vq_spec(
                            one, module_name=name, field_name=f"stage_vq_weights[{stage_idx}][{part_idx}]"
                        )
                        for part_idx, one in enumerate(stage_vq_spec)
                    ]
                    stage_vq_payload.append(_make_vq_placeholders(normalized_list, device=device))
                    stage_vq_storage_specs.append(normalized_list)
                    stage_decoder_payload.append([_build_decoder_from_spec(s) for s in stage_decoder_spec])
        else:
            vq_storage_specs = [
                _validate_packed_vq_spec(one, module_name=name, field_name=f"vq_weights[{idx}]")
                for idx, one in enumerate(spec["vq_weights"])
            ]
            vq_placeholders = _make_vq_placeholders(vq_storage_specs, device=device)
            decoders = [_build_decoder_from_spec(s) for s in spec["decoders"]]
            if len(vq_placeholders) != parallel_parts or len(decoders) != parallel_parts:
                raise ValueError(f"[{name}] vq/decoder count != parallel_parts={parallel_parts}")
            if parallel_parts == 1:
                vq_payload = vq_placeholders[0]
                vq_storage_specs = vq_storage_specs[0]
                decoder_payload = decoders[0]
            else:
                vq_payload = vq_placeholders
                decoder_payload = decoders

        protected_idx_payload = _placeholder_from_tensor_spec(
            spec.get("protected_input_indices"),
            module_name=name,
            field_name="protected_input_indices",
            expected_ndim=1,
            default_dtype="int64",
            device=device,
            unique_index=True,
        )
        protected_weight_payload = _placeholder_from_tensor_spec(
            spec.get("protected_input_weight"),
            module_name=name,
            field_name="protected_input_weight",
            expected_ndim=2,
            default_dtype="float32",
            device=device,
        )
        protected_out_idx_payload = _placeholder_from_tensor_spec(
            spec.get("protected_output_indices"),
            module_name=name,
            field_name="protected_output_indices",
            expected_ndim=1,
            default_dtype="int64",
            device=device,
            unique_index=True,
        )
        protected_out_weight_payload = _placeholder_from_tensor_spec(
            spec.get("protected_output_weight"),
            module_name=name,
            field_name="protected_output_weight",
            expected_ndim=2,
            default_dtype="float32",
            device=device,
        )
        protected_input_qvalues_payload = _placeholder_from_tensor_spec(
            spec.get("protected_input_qvalues"),
            module_name=name,
            field_name="protected_input_qvalues",
            expected_ndim=2,
            default_dtype="uint8",
            device=device,
        )
        protected_input_scales_payload = _placeholder_from_tensor_spec(
            spec.get("protected_input_scales"),
            module_name=name,
            field_name="protected_input_scales",
            expected_ndim=1,
            default_dtype="bfloat16",
            device=device,
        )
        protected_output_qvalues_payload = _placeholder_from_tensor_spec(
            spec.get("protected_output_qvalues"),
            module_name=name,
            field_name="protected_output_qvalues",
            expected_ndim=2,
            default_dtype="uint8",
            device=device,
        )
        protected_output_scales_payload = _placeholder_from_tensor_spec(
            spec.get("protected_output_scales"),
            module_name=name,
            field_name="protected_output_scales",
            expected_ndim=1,
            default_dtype="bfloat16",
            device=device,
        )
        low_rank_a_payload = _placeholder_from_tensor_spec(
            spec.get("low_rank_a"),
            module_name=name,
            field_name="low_rank_a",
            expected_ndim=2,
            default_dtype="float32",
            device=device,
        )
        low_rank_b_payload = _placeholder_from_tensor_spec(
            spec.get("low_rank_b"),
            module_name=name,
            field_name="low_rank_b",
            expected_ndim=2,
            default_dtype="float32",
            device=device,
        )
        if (low_rank_a_payload is None) != (low_rank_b_payload is None):
            raise ValueError(f"[{name}] low_rank_a and low_rank_b must be provided together.")

        keep_original_weight = bool(spec.get("has_original_weight", False))
        extended_kwargs = _build_extended_vae_rebuild_kwargs(
            spec,
            module_name=name,
            device=device,
            shared_protected_residual_decoders=shared_protected_residual_decoders,
        )
        new_module = VAELinear(
            in_features=int(spec["in_features"]),
            out_features=int(spec["out_features"]),
            bias=_ensure_bias_param(
                old_module=old_module,
                out_features=int(spec["out_features"]),
                has_bias=bool(spec["has_bias"]),
            ),
            original_weight=getattr(old_module, "weight", None) if keep_original_weight else None,
            vq_weight=vq_payload,
            vq_storage_specs=vq_storage_specs,
            decoder=decoder_payload,
            stage_vq_weights=stage_vq_payload,
            stage_vq_storage_specs=stage_vq_storage_specs,
            stage_decoders=stage_decoder_payload,
            codebook_dim=int(spec["codebook_dim"]),
            stage_codebook_dims=stage_codebook_dims,
            transpose=bool(spec["transpose"]),
            parallel_parts=parallel_parts,
            parallel_rows=int(spec.get("parallel_rows", parallel_parts)),
            parallel_cols=int(spec.get("parallel_cols", 1)),
            compressed_in_features=int(spec.get("compressed_in_features", spec["in_features"])),
            compressed_out_features=int(spec.get("compressed_out_features", spec["out_features"])),
            protected_input_indices=protected_idx_payload,
            protected_input_weight=protected_weight_payload,
            protected_input_qvalues=protected_input_qvalues_payload,
            protected_input_scales=protected_input_scales_payload,
            protected_output_indices=protected_out_idx_payload,
            protected_output_weight=protected_out_weight_payload,
            protected_output_qvalues=protected_output_qvalues_payload,
            protected_output_scales=protected_output_scales_payload,
            protected_channel_quant_format=str(spec.get("protected_channel_quant_format", "none")),
            low_rank_a=low_rank_a_payload,
            low_rank_b=low_rank_b_payload,
            **extended_kwargs,
            always_use_original=bool(spec.get("always_use_original", False)),
            protect_original_weight=bool(spec.get("protect_original_weight", False)),
        )
        if bool(spec.get("parallel_stage_decode", False)):
            new_module.pack_parallel_stage_decoder_(trainable=False)
        if bool(spec.get("protected_residual_parallel_stage_decode", False)):
            packed = new_module.pack_protected_residual_stage_decoder_(trainable=False)
            if not packed and getattr(new_module, "_protected_residual_shared_stage_decoders", None) is None:
                raise RuntimeError(
                    f"[{name}] failed to restore protected_residual_parallel_stage_decode."
                )
        set_module_by_name(model, name, new_module)


def refresh_vae_linear_runtime_after_state_load(model: nn.Module) -> None:
    """Rebuild parallel decode plans and clear decoded weight caches."""
    for module in model.modules():
        if not isinstance(module, VAELinear):
            continue
        if getattr(module, "_parallel_stage_decoder", None) is not None:
            module._build_parallel_stage_decode_plan()
        if getattr(module, "_protected_residual_parallel_decoder", None) is not None:
            module._build_protected_residual_parallel_decode_plan()
        module.clear_decoded_weight_cache()


def resolve_v6_checkpoint_dir(path: str) -> str:
    """Resolve a full v6 checkpoint directory; reject incomplete or tmp dirs."""
    abs_path = os.path.abspath(path)
    if _path_looks_like_tmp(abs_path):
        raise ValueError(f"Refusing incomplete temp checkpoint path: {abs_path}")

    if os.path.isfile(abs_path):
        if os.path.basename(abs_path) == META_FILENAME:
            abs_path = os.path.dirname(abs_path)
        else:
            raise FileNotFoundError(f"Expected {META_FILENAME} file or checkpoint directory, got: {abs_path}")

    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Path does not exist: {abs_path}")
    if _path_looks_like_tmp(abs_path):
        raise ValueError(f"Refusing incomplete temp checkpoint path: {abs_path}")

    meta_path = os.path.join(abs_path, META_FILENAME)
    state_path = os.path.join(abs_path, STATE_DICT_FILENAME)
    if not os.path.isfile(meta_path):
        final_dir = os.path.join(abs_path, "final_model")
        final_meta = os.path.join(final_dir, META_FILENAME)
        final_state = os.path.join(final_dir, STATE_DICT_FILENAME)
        if os.path.isfile(final_meta) and os.path.isfile(final_state):
            abs_path = final_dir
            meta_path = final_meta
            state_path = final_state
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {META_FILENAME} under {abs_path}")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Missing {STATE_DICT_FILENAME} under {abs_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    validate_v6_meta(meta)
    kind = meta.get("checkpoint_kind")
    if kind not in FULL_MODEL_KINDS:
        raise ValueError(
            f"resolve_v6_checkpoint_dir expects a full-model checkpoint, got kind={kind!r} at {abs_path}"
        )
    return abs_path


def load_v6_meta(checkpoint_dir: str) -> Dict[str, Any]:
    abs_dir = os.path.abspath(checkpoint_dir)
    if _path_looks_like_tmp(abs_dir):
        raise ValueError(f"Refusing incomplete temp checkpoint path: {abs_dir}")
    meta_path = os.path.join(abs_dir, META_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {META_FILENAME} under {abs_dir}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return validate_v6_meta(meta)


def save_v6_full_checkpoint(
    model: nn.Module,
    output_dir: str,
    *,
    checkpoint_kind: str,
    compressed_targets: Sequence[str],
    pending_dense_targets: Sequence[str] = (),
    skip_targets: Sequence[str] = (),
    legacy_original_only_sources: Sequence[str] = (),
    train_mode: str = "none",
    after_category_mode: Optional[str] = None,
    norm_train_mode: str = "none",
    lm_head_train_mode: str = "none",
    lora_config: Optional[dict] = None,
    resolved_learning_rates: Optional[dict] = None,
    completed_categories: Optional[Sequence[str]] = None,
    compression_categories: Optional[Sequence[str]] = None,
    target_layers: Optional[Sequence[int]] = None,
    target_modules: Optional[Sequence[str]] = None,
    immutable_resume_contract: Optional[dict] = None,
    finalized_status: Optional[dict] = None,
    runtime_audit: Optional[dict] = None,
    base_model_path: Optional[str] = None,
    tokenizer=None,
    save_config: bool = True,
    cat_runtime_state: Optional[Mapping[str, Any]] = None,
    extra_meta: Optional[dict] = None,
    is_main_process: bool = True,
    distributed_barrier: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Atomically save a full v6 model checkpoint (round_base / category_boundary / final_model)."""
    if checkpoint_kind not in FULL_MODEL_KINDS:
        raise ValueError(f"checkpoint_kind must be one of {FULL_MODEL_KINDS}, got {checkpoint_kind!r}")
    if lora_config is not None:
        raise ValueError(
            "stable full v6 checkpoints require lora_config=None; finalized low-rank "
            "topology is defined by each VAELinear payload shape."
        )
    _reject_live_training_adapter_topology(model)
    if (
        distributed_barrier is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return distributed_guarded_main(
            lambda: save_v6_full_checkpoint(
                model,
                output_dir,
                checkpoint_kind=checkpoint_kind,
                compressed_targets=compressed_targets,
                pending_dense_targets=pending_dense_targets,
                skip_targets=skip_targets,
                legacy_original_only_sources=legacy_original_only_sources,
                train_mode=train_mode,
                after_category_mode=after_category_mode,
                norm_train_mode=norm_train_mode,
                lm_head_train_mode=lm_head_train_mode,
                lora_config=None,
                resolved_learning_rates=resolved_learning_rates,
                completed_categories=completed_categories,
                compression_categories=compression_categories,
                target_layers=target_layers,
                target_modules=target_modules,
                immutable_resume_contract=immutable_resume_contract,
                finalized_status=finalized_status,
                runtime_audit=runtime_audit,
                base_model_path=base_model_path,
                tokenizer=tokenizer,
                save_config=save_config,
                cat_runtime_state=cat_runtime_state,
                extra_meta=extra_meta,
                is_main_process=True,
                distributed_barrier=None,
            ),
            barrier=True,
        )

    compressed = _as_str_list(compressed_targets, field_name="compressed_targets")
    pending = _as_str_list(pending_dense_targets, field_name="pending_dense_targets")
    skip = _as_str_list(skip_targets, field_name="skip_targets")
    legacy_sources = _as_str_list(legacy_original_only_sources, field_name="legacy_original_only_sources")
    validate_model_target_inventories(
        model,
        compressed,
        pending,
        skip,
        legacy_original_only_sources=legacy_sources,
    )

    # round_base must not invent category completion advancement.
    if completed_categories is None:
        completed: List[str] = []
    else:
        completed = _as_str_list(completed_categories, field_name="completed_categories")

    resolved_compression_categories = _as_str_list(
        compression_categories or [], field_name="compression_categories"
    )
    resolved_target_layers = _as_int_list(target_layers, field_name="target_layers")
    resolved_target_modules = _as_str_list(target_modules or [], field_name="target_modules")
    resolved_resume_contract = (
        dict(immutable_resume_contract) if immutable_resume_contract is not None else None
    )

    abs_out = os.path.abspath(output_dir)
    result: Dict[str, Any] = {
        "checkpoint_id": None,
        "output_dir": abs_out,
        "state_dict": None,
        "meta": None,
        "meta_payload": None,
    }

    if is_main_process:
        checkpoint_id = build_checkpoint_id()
        _reject_always_use_original_on_save(model)
        if os.path.exists(abs_out):
            raise FileExistsError(f"Checkpoint destination already exists: {abs_out}")

        tmp_dir = _make_temp_sibling_dir(abs_out)
        try:
            # Save live topology as-is (packed or serial). Do not force-pack.
            state_dict = model.state_dict()
            vae_specs = _collect_vae_linear_specs(model)
            shared_protected_residual_specs = _collect_shared_protected_residual_decoder_specs(model)
            state_path = os.path.join(tmp_dir, STATE_DICT_FILENAME)
            torch.save(state_dict, state_path)

            if save_config and getattr(model, "config", None) is not None:
                model.config.save_pretrained(tmp_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmp_dir)

            resolved_extra_meta = dict(extra_meta) if extra_meta is not None else {}
            if cat_runtime_state is not None:
                existing_runtime_file = resolved_extra_meta.get("cat_runtime_state_file")
                if existing_runtime_file not in (None, CAT_RUNTIME_STATE_FILENAME):
                    raise ValueError(
                        "extra_meta.cat_runtime_state_file conflicts with canonical CAT runtime sidecar filename."
                    )
                runtime_state_path = os.path.join(tmp_dir, CAT_RUNTIME_STATE_FILENAME)
                torch.save(_cpu_clone_nested(dict(cat_runtime_state)), runtime_state_path)
                resolved_extra_meta["cat_runtime_state_file"] = CAT_RUNTIME_STATE_FILENAME

            resolved_base = base_model_path
            if resolved_base is None and getattr(model, "config", None) is not None:
                resolved_base = getattr(model.config, "_name_or_path", None)

            meta_obj = V6CheckpointMeta(
                checkpoint_kind=checkpoint_kind,
                checkpoint_id=checkpoint_id,
                base_model_path=resolved_base,
                state_dict_file=STATE_DICT_FILENAME,
                train_mode=str(train_mode),
                after_category_mode=after_category_mode,
                norm_train_mode=str(norm_train_mode),
                lm_head_train_mode=str(lm_head_train_mode),
                lora_config=None,
                resolved_learning_rates=dict(resolved_learning_rates) if resolved_learning_rates is not None else None,
                compressed_targets=compressed,
                pending_dense_targets=pending,
                skip_targets=skip,
                legacy_original_only_sources=legacy_sources,
                completed_categories=completed,
                compression_categories=resolved_compression_categories,
                target_layers=resolved_target_layers,
                target_modules=resolved_target_modules,
                immutable_resume_contract=resolved_resume_contract,
                finalized_status=dict(finalized_status) if finalized_status is not None else None,
                runtime_audit=dict(runtime_audit) if runtime_audit is not None else None,
                converted_modules=vae_specs,
                converted_module_count=len(vae_specs),
                shared_protected_residual_decoders=shared_protected_residual_specs,
                post_norm_head_linear=bool(has_post_norm_head_linear(model)),
                extra_meta=resolved_extra_meta or None,
            )
            meta_payload = meta_obj.validate(expected_kind=checkpoint_kind)
            meta_path = os.path.join(tmp_dir, META_FILENAME)
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")

            _validate_full_checkpoint_dir_contents(tmp_dir)
            _publish_temp_dir(tmp_dir, abs_out)
            tmp_dir = ""  # published; do not delete in finally

            result["checkpoint_id"] = checkpoint_id
            result["state_dict"] = os.path.join(abs_out, STATE_DICT_FILENAME)
            result["meta"] = os.path.join(abs_out, META_FILENAME)
            result["meta_payload"] = meta_payload
            result["output_dir"] = abs_out
        finally:
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    _call_barrier(distributed_barrier)

    if not is_main_process:
        if not os.path.isdir(abs_out):
            raise FileNotFoundError(
                f"Non-main rank expected published full checkpoint at {abs_out}, but directory is missing"
            )
        meta_payload = load_v6_meta(abs_out)
        validate_v6_meta(meta_payload, expected_kind=checkpoint_kind)
        result["checkpoint_id"] = str(meta_payload["checkpoint_id"])
        result["meta"] = os.path.join(abs_out, META_FILENAME)
        result["state_dict"] = os.path.join(abs_out, STATE_DICT_FILENAME)
        result["meta_payload"] = meta_payload
        result["output_dir"] = abs_out

    if not result.get("checkpoint_id"):
        raise RuntimeError("Failed to resolve canonical checkpoint_id after full save")
    return result


def load_v6_full_checkpoint_into_model(
    model: nn.Module,
    checkpoint_dir: str,
    *,
    expected_kind: Optional[str] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], Any]:
    resolved = resolve_v6_checkpoint_dir(checkpoint_dir)
    meta = load_v6_meta(resolved)
    meta = validate_v6_meta(meta, expected_kind=expected_kind)
    validate_full_converted_modules_meta(meta)

    shared_specs = _validate_shared_protected_residual_decoder_specs(
        meta.get("shared_protected_residual_decoders") or []
    )
    shared_protected_residual_decoders = _rebuild_shared_protected_residual_decoders(
        model,
        shared_specs,
    )
    converted_modules = meta.get("converted_modules") or []
    if converted_modules:
        _rebuild_converted_modules(
            model,
            converted_modules,
            shared_protected_residual_decoders=shared_protected_residual_decoders,
        )
    if bool(meta.get("post_norm_head_linear", False)):
        ensure_post_norm_head_linear(model)

    validate_model_target_inventories(
        model,
        meta.get("compressed_targets") or [],
        meta.get("pending_dense_targets") or [],
        meta.get("skip_targets") or [],
        legacy_original_only_sources=meta.get("legacy_original_only_sources") or [],
    )

    state_dict_file = str(meta.get("state_dict_file", STATE_DICT_FILENAME))
    state_dict_path = os.path.join(resolved, state_dict_file)
    state_dict = _torch_load(state_dict_path, map_location=map_location)
    load_result = model.load_state_dict(state_dict, strict=strict)
    refresh_vae_linear_runtime_after_state_load(model)
    model.eval()
    return model, meta, load_result


def load_v6_cat_runtime_state(
    checkpoint_dir: str,
    *,
    required: bool = False,
    map_location: str = "cpu",
):
    resolved = resolve_v6_checkpoint_dir(checkpoint_dir)
    meta = load_v6_meta(resolved)
    validate_v6_meta(meta)
    extra_meta = meta.get("extra_meta")
    runtime_file = extra_meta.get("cat_runtime_state_file") if isinstance(extra_meta, dict) else None
    if runtime_file is None:
        if required:
            raise FileNotFoundError(
                f"v6 checkpoint {resolved} does not register {CAT_RUNTIME_STATE_FILENAME}."
            )
        return None
    if str(runtime_file) != CAT_RUNTIME_STATE_FILENAME:
        raise ValueError(f"Unsupported CAT runtime state filename: {runtime_file!r}.")
    runtime_path = os.path.join(resolved, CAT_RUNTIME_STATE_FILENAME)
    if not os.path.isfile(runtime_path):
        raise FileNotFoundError(f"Missing {CAT_RUNTIME_STATE_FILENAME} under {resolved}")
    payload = _torch_load(runtime_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"{CAT_RUNTIME_STATE_FILENAME} must contain a dict, got {type(payload)}.")
    return payload


def build_mutable_state_manifest(
    named_tensors: Mapping[str, Tensor],
    *,
    component_classes: Mapping[str, str],
) -> List[Dict[str, Any]]:
    if not isinstance(named_tensors, Mapping):
        raise TypeError(f"named_tensors must be a mapping, got {type(named_tensors)}")
    if not isinstance(component_classes, Mapping):
        raise TypeError(f"component_classes must be a mapping, got {type(component_classes)}")

    tensor_names = sorted(str(name) for name in named_tensors.keys())
    class_names = sorted(str(name) for name in component_classes.keys())
    if tensor_names != class_names:
        missing = sorted(set(tensor_names) - set(class_names))
        extra = sorted(set(class_names) - set(tensor_names))
        raise ValueError(
            "named_tensors and component_classes must have exact same keys; "
            f"missing_classes={missing}, extra_classes={extra}"
        )
    if len(tensor_names) != len(set(tensor_names)):
        raise ValueError("named_tensors contains duplicate names")

    seen_ids: Dict[int, str] = {}
    seen_storage: Dict[Tuple[Any, ...], str] = {}
    manifest: List[Dict[str, Any]] = []
    for name in tensor_names:
        tensor = named_tensors[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"mutable state entry {name!r} must be a Tensor, got {type(tensor)}")
        component_class = _normalize_component_class(component_classes[name])

        tid = id(tensor)
        if tid in seen_ids:
            raise ValueError(
                f"duplicate tensor identity (same object id) for names {seen_ids[tid]!r} and {name!r}"
            )
        seen_ids[tid] = name

        storage_key = _tensor_storage_identity(tensor)
        if storage_key in seen_storage and seen_storage[storage_key] != name:
            raise ValueError(
                f"duplicate tensor identity/storage for names {seen_storage[storage_key]!r} and {name!r}"
            )
        seen_storage[storage_key] = name

        manifest.append(
            {
                "name": str(name),
                "shape": list(tensor.shape),
                "dtype": _dtype_to_name(tensor.dtype),
                "component_class": component_class,
            }
        )
    return manifest


def build_uniform_mutable_state_manifest(
    named_tensors: Mapping[str, Tensor],
    *,
    component_class: str,
) -> List[Dict[str, Any]]:
    return build_mutable_state_manifest(
        named_tensors,
        component_classes={str(k): component_class for k in named_tensors},
    )


def validate_mutable_state_manifest(
    manifest: Sequence[Mapping[str, Any]],
    named_tensors: Mapping[str, Tensor],
    *,
    component_classes: Optional[Mapping[str, str]] = None,
) -> None:
    if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes)):
        raise TypeError("manifest must be a sequence of dicts")
    if not isinstance(named_tensors, Mapping):
        raise TypeError(f"named_tensors must be a mapping, got {type(named_tensors)}")

    seen_names: set[str] = set()
    expected: Dict[str, Mapping[str, Any]] = {}
    for item in manifest:
        if not isinstance(item, Mapping):
            raise TypeError(f"manifest entries must be dicts, got {type(item)}")
        if "name" not in item:
            raise ValueError("mutable_state_manifest entry missing name")
        name = str(item["name"])
        if name in seen_names:
            raise ValueError(f"duplicate name in mutable_state_manifest: {name!r}")
        seen_names.add(name)
        if "component_class" not in item:
            raise ValueError(f"mutable_state_manifest entry {name!r} missing component_class")
        component_class = _normalize_component_class(str(item["component_class"]))
        if "shape" not in item or "dtype" not in item:
            raise ValueError(f"mutable_state_manifest entry {name!r} missing shape/dtype")
        expected[name] = {
            "name": name,
            "shape": list(item["shape"]),
            "dtype": str(item["dtype"]),
            "component_class": component_class,
        }

    actual_names = {str(k) for k in named_tensors.keys()}
    expected_names = set(expected.keys())
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ValueError(
            f"mutable_state_manifest mismatch: missing={missing}, extra={extra}"
        )

    if component_classes is not None:
        class_keys = {str(k) for k in component_classes.keys()}
        if class_keys != expected_names:
            raise ValueError(
                "component_classes keys must match manifest names; "
                f"missing={sorted(expected_names - class_keys)}, "
                f"extra={sorted(class_keys - expected_names)}"
            )

    for name, tensor in named_tensors.items():
        entry = expected[str(name)]
        shape = tuple(int(x) for x in entry["shape"])
        dtype_name = str(entry["dtype"])
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"mutable state shape mismatch for {name!r}: got {tuple(tensor.shape)}, expected {shape}"
            )
        if _dtype_to_name(tensor.dtype) != dtype_name:
            raise ValueError(
                f"mutable state dtype mismatch for {name!r}: got {_dtype_to_name(tensor.dtype)}, expected {dtype_name}"
            )
        if component_classes is not None:
            expected_class = _normalize_component_class(component_classes[str(name)])
            if expected_class != entry["component_class"]:
                raise ValueError(
                    f"mutable state component_class mismatch for {name!r}: "
                    f"got {entry['component_class']!r}, expected {expected_class!r}"
                )

    # Reject aliased tensor objects / shared storage under multiple logical names.
    seen_ids: Dict[int, str] = {}
    seen_storage: Dict[Tuple[Any, ...], str] = {}
    for name, tensor in named_tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"mutable state entry {name!r} must be a Tensor, got {type(tensor)}")
        tid = id(tensor)
        if tid in seen_ids:
            raise ValueError(
                f"duplicate tensor identity (same object id) for names {seen_ids[tid]!r} and {name!r}"
            )
        seen_ids[tid] = str(name)
        storage_key = _tensor_storage_identity(tensor)
        if storage_key in seen_storage and seen_storage[storage_key] != str(name):
            raise ValueError(
                f"duplicate tensor identity/storage for names {seen_storage[storage_key]!r} and {name!r}"
            )
        seen_storage[storage_key] = str(name)


def _has_valid_v6_training_step_meta(checkpoint_dir: str) -> bool:
    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        validate_v6_meta(meta, expected_kind="training_step")
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _invalidate_existing_meta_marker(meta_path: str) -> None:
    if not os.path.isfile(meta_path):
        return
    invalidated = f"{meta_path}.invalidated-{uuid.uuid4().hex}"
    os.replace(meta_path, invalidated)


def save_v6_training_step_payload(
    checkpoint_dir: str,
    *,
    round_base_ref: str,
    round_base_checkpoint_id: str,
    mutable_state: Mapping[str, Tensor],
    mutable_state_manifest: Sequence[dict],
    train_mode: str,
    compressed_targets: Sequence[str] = (),
    pending_dense_targets: Sequence[str] = (),
    skip_targets: Sequence[str] = (),
    legacy_original_only_sources: Sequence[str] = (),
    after_category_mode: Optional[str] = None,
    norm_train_mode: str = "none",
    lm_head_train_mode: str = "none",
    lora_config: Optional[dict] = None,
    resolved_learning_rates: Optional[dict] = None,
    completed_categories: Optional[Sequence[str]] = None,
    compression_categories: Optional[Sequence[str]] = None,
    target_layers: Optional[Sequence[int]] = None,
    target_modules: Optional[Sequence[str]] = None,
    immutable_resume_contract: Optional[dict] = None,
    finalized_status: Optional[dict] = None,
    runtime_audit: Optional[dict] = None,
    base_model_path: Optional[str] = None,
    hf_artifact_refs: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
    allow_overwrite: bool = False,
    is_main_process: bool = True,
    distributed_barrier: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Write training_step custom payload (no full pytorch_model.bin).

    Invalidate old meta first (when overwriting), write ``training_model_state.pt``
    via atomic replace, then write ``checkpoint_meta.json`` last as the commit marker.
    """
    compressed = _as_str_list(compressed_targets, field_name="compressed_targets")
    pending = _as_str_list(pending_dense_targets, field_name="pending_dense_targets")
    skip = _as_str_list(skip_targets, field_name="skip_targets")
    validate_target_inventories(compressed, pending, skip)
    validate_mutable_state_manifest(mutable_state_manifest, mutable_state)

    if (
        distributed_barrier is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return distributed_guarded_main(
            lambda: save_v6_training_step_payload(
                checkpoint_dir,
                round_base_ref=round_base_ref,
                round_base_checkpoint_id=round_base_checkpoint_id,
                mutable_state=mutable_state,
                mutable_state_manifest=mutable_state_manifest,
                train_mode=train_mode,
                compressed_targets=compressed_targets,
                pending_dense_targets=pending_dense_targets,
                skip_targets=skip_targets,
                legacy_original_only_sources=legacy_original_only_sources,
                after_category_mode=after_category_mode,
                norm_train_mode=norm_train_mode,
                lm_head_train_mode=lm_head_train_mode,
                lora_config=lora_config,
                resolved_learning_rates=resolved_learning_rates,
                completed_categories=completed_categories,
                compression_categories=compression_categories,
                target_layers=target_layers,
                target_modules=target_modules,
                immutable_resume_contract=immutable_resume_contract,
                finalized_status=finalized_status,
                runtime_audit=runtime_audit,
                base_model_path=base_model_path,
                hf_artifact_refs=hf_artifact_refs,
                extra_meta=extra_meta,
                allow_overwrite=allow_overwrite,
                is_main_process=True,
                distributed_barrier=None,
            ),
            barrier=True,
        )

    if not str(round_base_ref).strip():
        raise ValueError("round_base_ref must be a non-empty string")
    if not str(round_base_checkpoint_id).strip():
        raise ValueError("round_base_checkpoint_id must be a non-empty string")

    abs_dir = os.path.abspath(checkpoint_dir)
    result: Dict[str, Any] = {
        "checkpoint_id": None,
        "output_dir": abs_dir,
        "training_model_state": None,
        "meta": None,
        "meta_payload": None,
    }

    if is_main_process:
        os.makedirs(abs_dir, exist_ok=True)
        meta_path = os.path.join(abs_dir, META_FILENAME)
        state_path = os.path.join(abs_dir, TRAINING_MODEL_STATE_FILENAME)
        has_valid_meta = _has_valid_v6_training_step_meta(abs_dir)
        has_custom_payload = os.path.isfile(state_path)
        if (has_valid_meta or has_custom_payload) and not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing training-step payload under {abs_dir}. "
                "Pass allow_overwrite=True for explicit transactional overwrite."
            )

        checkpoint_id = build_checkpoint_id()

        # Invalidate old commit marker BEFORE replacing state so loaders cannot
        # observe stale meta + new state.
        if os.path.isfile(meta_path):
            _invalidate_existing_meta_marker(meta_path)

        cpu_state = {
            name: tensor.detach().to("cpu").contiguous() if isinstance(tensor, Tensor) else tensor
            for name, tensor in mutable_state.items()
        }
        _atomic_torch_save(cpu_state, state_path)

        completed = (
            []
            if completed_categories is None
            else _as_str_list(completed_categories, field_name="completed_categories")
        )
        meta_obj = V6CheckpointMeta(
            checkpoint_kind="training_step",
            checkpoint_id=checkpoint_id,
            base_model_path=base_model_path,
            state_dict_file=None,
            train_mode=str(train_mode),
            after_category_mode=after_category_mode,
            norm_train_mode=str(norm_train_mode),
            lm_head_train_mode=str(lm_head_train_mode),
            lora_config=dict(lora_config) if lora_config is not None else None,
            resolved_learning_rates=dict(resolved_learning_rates) if resolved_learning_rates is not None else None,
            compressed_targets=compressed,
            pending_dense_targets=pending,
            skip_targets=skip,
            legacy_original_only_sources=_as_str_list(
                legacy_original_only_sources, field_name="legacy_original_only_sources"
            ),
            completed_categories=completed,
            compression_categories=_as_str_list(
                compression_categories or [], field_name="compression_categories"
            ),
            target_layers=_as_int_list(target_layers, field_name="target_layers"),
            target_modules=_as_str_list(target_modules or [], field_name="target_modules"),
            immutable_resume_contract=(
                dict(immutable_resume_contract) if immutable_resume_contract is not None else None
            ),
            finalized_status=dict(finalized_status) if finalized_status is not None else None,
            runtime_audit=dict(runtime_audit) if runtime_audit is not None else None,
            converted_modules=[],
            converted_module_count=0,
            post_norm_head_linear=False,
            round_base_ref=str(round_base_ref),
            round_base_checkpoint_id=str(round_base_checkpoint_id),
            mutable_state_manifest=[dict(item) for item in mutable_state_manifest],
            hf_artifact_refs=dict(hf_artifact_refs) if hf_artifact_refs is not None else None,
            extra_meta=dict(extra_meta) if extra_meta is not None else None,
        )
        meta_payload = meta_obj.validate(expected_kind="training_step")
        _atomic_write_json(meta_payload, meta_path)

        result["checkpoint_id"] = checkpoint_id
        result["training_model_state"] = state_path
        result["meta"] = meta_path
        result["meta_payload"] = meta_payload

    _call_barrier(distributed_barrier)

    if not is_main_process:
        if not os.path.isdir(abs_dir):
            raise FileNotFoundError(
                f"Non-main rank expected published training-step checkpoint at {abs_dir}, "
                "but directory is missing"
            )
        meta_payload = load_v6_training_step_meta(abs_dir)
        result["checkpoint_id"] = str(meta_payload["checkpoint_id"])
        result["meta"] = os.path.join(abs_dir, META_FILENAME)
        result["training_model_state"] = os.path.join(abs_dir, TRAINING_MODEL_STATE_FILENAME)
        result["meta_payload"] = meta_payload
        result["output_dir"] = abs_dir

    if not result.get("checkpoint_id"):
        raise RuntimeError("Failed to resolve canonical checkpoint_id after training-step save")
    return result


def load_v6_training_step_meta(checkpoint_dir: str) -> Dict[str, Any]:
    abs_dir = os.path.abspath(checkpoint_dir)
    if _path_looks_like_tmp(abs_dir):
        raise ValueError(f"Refusing incomplete temp checkpoint path: {abs_dir}")
    meta_path = os.path.join(abs_dir, META_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Missing {META_FILENAME} under {abs_dir} "
            f"(training-step meta is required even if {TRAINING_MODEL_STATE_FILENAME} exists)"
        )
    state_path = os.path.join(abs_dir, TRAINING_MODEL_STATE_FILENAME)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Missing {TRAINING_MODEL_STATE_FILENAME} under {abs_dir}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return validate_v6_meta(meta, expected_kind="training_step")


def validate_training_step_round_base(
    meta: Mapping[str, Any],
    round_base_meta: Mapping[str, Any],
) -> None:
    step_meta = validate_v6_meta(meta, expected_kind="training_step")
    base_meta = validate_v6_meta(round_base_meta)
    if base_meta.get("checkpoint_kind") not in FULL_MODEL_KINDS:
        raise ValueError(
            f"round_base_meta checkpoint_kind must be a full-model kind, got {base_meta.get('checkpoint_kind')!r}"
        )
    expected_id = str(step_meta["round_base_checkpoint_id"])
    actual_id = str(base_meta["checkpoint_id"])
    if expected_id != actual_id:
        raise ValueError(
            f"training_step round_base_checkpoint_id mismatch: expected {expected_id!r}, got {actual_id!r}"
        )


def resolve_training_step_round_base_ref(
    step_dir: str,
    meta: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Resolve training-step ``round_base_ref`` and validate checkpoint_id match."""
    step_meta = validate_v6_meta(meta, expected_kind="training_step")
    ref = step_meta.get("round_base_ref")
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("training_step meta requires non-empty round_base_ref")
    if os.path.isabs(ref):
        candidate = ref
    else:
        candidate = os.path.normpath(os.path.join(os.path.abspath(step_dir), ref))
    resolved = resolve_v6_checkpoint_dir(candidate)
    base_meta = load_v6_meta(resolved)
    validate_training_step_round_base(step_meta, base_meta)
    return os.path.abspath(resolved), base_meta


def load_v6_training_model_state(
    checkpoint_dir: str,
    *,
    map_location: str = "cpu",
) -> Tuple[Dict[str, Tensor], List[Dict[str, Any]]]:
    meta = load_v6_training_step_meta(checkpoint_dir)
    state_path = os.path.join(os.path.abspath(checkpoint_dir), TRAINING_MODEL_STATE_FILENAME)
    state_dict = _torch_load(state_path, map_location=map_location)
    if not isinstance(state_dict, dict):
        raise TypeError(f"{TRAINING_MODEL_STATE_FILENAME} must contain a dict, got {type(state_dict)}")
    manifest = meta.get(MUTABLE_STATE_MANIFEST_KEY) or []
    validate_mutable_state_manifest(manifest, state_dict)
    return state_dict, list(manifest)


__all__ = [
    "FORMAT_V6",
    "SCHEMA_VERSION",
    "CHECKPOINT_KINDS",
    "FULL_MODEL_KINDS",
    "STATE_DICT_FILENAME",
    "META_FILENAME",
    "TRAINING_MODEL_STATE_FILENAME",
    "CAT_RUNTIME_STATE_FILENAME",
    "MUTABLE_STATE_MANIFEST_KEY",
    "MUTABLE_COMPONENT_CLASSES",
    "V6CheckpointMeta",
    "validate_v6_meta",
    "validate_full_converted_modules_meta",
    "validate_target_inventories",
    "validate_model_target_inventories",
    "iter_named_vae_linears",
    "build_checkpoint_id",
    "save_v6_full_checkpoint",
    "load_v6_meta",
    "resolve_v6_checkpoint_dir",
    "load_v6_full_checkpoint_into_model",
    "load_v6_cat_runtime_state",
    "refresh_vae_linear_runtime_after_state_load",
    "build_mutable_state_manifest",
    "build_uniform_mutable_state_manifest",
    "validate_mutable_state_manifest",
    "save_v6_training_step_payload",
    "load_v6_training_step_meta",
    "validate_training_step_round_base",
    "resolve_training_step_round_base_ref",
    "load_v6_training_model_state",
]
