import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from e2e_fintuning.lora import (
    LoRAEmbedding,
    LoRALinear,
    LoRAVAELinear,
    ensure_lora_embedding,
    ensure_lora_linear,
    ensure_lora_vae_linear,
    iter_named_vae_module_refs,
)
from e2e_fintuning.peft_proxy import (
    PeftVAELinearProxy,
    collect_peft_vae_proxy_adapter_specs,
    ensure_peft_vae_linear_proxy,
    ensure_peft_vae_proxy_lora,
    iter_named_peft_vae_proxies,
    strip_proxy_dense_base_from_state_dict,
)
from rotation.common import separate_embeddings_and_lm_head
from rotation.model_utils import get_model
from train_utils.model_checkpoint_io import (
    META_FILENAME,
    STATE_DICT_FILENAME,
    _decoder_to_spec,
    _dtype_to_name,
    _get_module_by_name,
    _materialize_missing_bias_params_from_state_dict,
    _rebuild_converted_modules,
    _remap_legacy_parallel_linear_state_dict_keys,
    _torch_load_state_dict,
    unload_vae_original_linear_weights,
)
from train_utils.utils import extract_layer_idx


_E2E_FINETUNE_MODE = "vae_lora"


def _tensor_spec(tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if not isinstance(tensor, torch.Tensor):
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_to_name(tensor.dtype),
    }


def _embedding_and_lm_head_are_tied(model: nn.Module) -> bool:
    embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(embedding, nn.Embedding) or not isinstance(lm_head, nn.Linear):
        return False
    return embedding.weight.data_ptr() == lm_head.weight.data_ptr()


def _checkpoint_has_extra_lora(adapter_modules: Sequence[Dict[str, Any]]) -> bool:
    for spec in adapter_modules:
        if str(spec.get("adapter_type")) in {"linear_lora", "embedding_lora"}:
            return True
    return False


def _collect_single_vae_linear_spec(name: str, module) -> Dict[str, Any]:
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
            weight = module.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=part_idx)
            stage_vq_parts.append(
                {
                    "shape": list(weight.shape),
                    "dtype": _dtype_to_name(weight.dtype),
                }
            )
            decoder = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
            stage_decoder_parts.append(_decoder_to_spec(decoder))
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

    return {
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
        "has_bias": bool(module.bias is not None),
        "has_original_weight": bool(module.original_weight is not None),
        "always_use_original": bool(getattr(module, "always_use_original", False)),
        "protect_original_weight": bool(getattr(module, "protect_original_weight", False)),
        "vq_weights": vq_specs,
        "decoders": decoder_specs,
        "stage_vq_weights": stage_vq_specs if residual_stages > 1 else None,
        "stage_decoders": stage_decoder_specs if residual_stages > 1 else None,
        "restore_row_indices": _tensor_spec(getattr(module, "restore_row_indices", None)),
        "restore_col_indices": _tensor_spec(getattr(module, "restore_col_indices", None)),
        "protected_input_indices": _tensor_spec(getattr(module, "protected_input_indices", None)),
        "protected_input_weight": _tensor_spec(getattr(module, "protected_input_weight", None)),
        "protected_output_indices": _tensor_spec(getattr(module, "protected_output_indices", None)),
        "protected_output_weight": _tensor_spec(getattr(module, "protected_output_weight", None)),
    }


def _collect_e2e_module_specs(model: nn.Module):
    converted_modules: List[Dict[str, Any]] = []
    adapter_modules: List[Dict[str, Any]] = []
    for ref in iter_named_vae_module_refs(model):
        converted_modules.append(_collect_single_vae_linear_spec(ref.name, ref.base_layer))
        if isinstance(ref.adapter, LoRAVAELinear):
            adapter_modules.append(
                {
                    "name": ref.name,
                    "adapter_type": "vae_lora",
                    "base_type": "VAELinear",
                    "r": int(ref.adapter.rank),
                    "alpha": float(ref.adapter.lora_alpha),
                    "dropout": float(ref.adapter.lora_dropout_p),
                    "target_layer": extract_layer_idx(ref.name),
                    "train_mode_at_save": str(getattr(model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)),
                }
            )
    adapter_modules.extend(
        collect_peft_vae_proxy_adapter_specs(
            model,
            train_mode=str(getattr(model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)),
        )
    )
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapter_modules.append(
                {
                    "name": name,
                    "adapter_type": "linear_lora",
                    "base_type": "Linear",
                    "r": int(module.rank),
                    "alpha": float(module.lora_alpha),
                    "dropout": float(module.lora_dropout_p),
                    "train_mode_at_save": str(getattr(model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)),
                }
            )
        elif isinstance(module, LoRAEmbedding):
            adapter_modules.append(
                {
                    "name": name,
                    "adapter_type": "embedding_lora",
                    "base_type": "Embedding",
                    "r": int(module.rank),
                    "alpha": float(module.lora_alpha),
                    "dropout": float(module.lora_dropout_p),
                    "train_mode_at_save": str(getattr(model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)),
                }
            )
    return converted_modules, adapter_modules


def _build_compact_e2e_checkpoint_payload(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]]]:
    compact_state_dict = dict(state_dict)
    converted_modules, adapter_modules = _collect_e2e_module_specs(model)
    compact_converted_modules: List[Dict[str, Any]] = []
    for spec in converted_modules:
        spec_copy = dict(spec)
        keep_original = bool(spec_copy.get("always_use_original", False)) or bool(
            spec_copy.get("protect_original_weight", False)
        )
        if bool(spec_copy.get("has_original_weight", False)) and not keep_original:
            spec_copy["has_original_weight"] = False
            module_name = str(spec_copy["name"])
            compact_state_dict.pop(f"{module_name}.original_weight", None)
            compact_state_dict.pop(f"{module_name}.base_layer.original_weight", None)
        compact_converted_modules.append(spec_copy)
    strip_proxy_dense_base_from_state_dict(model, compact_state_dict)
    return compact_state_dict, compact_converted_modules, adapter_modules


def save_e2e_model_checkpoint(
    model: nn.Module,
    output_dir: str,
    *,
    base_model_path: Optional[str] = None,
    tokenizer=None,
    save_config: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
    unload_vae_original_weights: bool = False,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    compact_unload_vae_original_weights: bool = False,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    if state_dict is None and unload_vae_original_weights:
        unload_vae_original_linear_weights(model)

    if state_dict is None:
        state_dict = model.state_dict()

    if compact_unload_vae_original_weights:
        state_dict, converted_modules, adapter_modules = _build_compact_e2e_checkpoint_payload(model, state_dict)
    else:
        converted_modules, adapter_modules = _collect_e2e_module_specs(model)

    state_path = os.path.join(output_dir, STATE_DICT_FILENAME)
    torch.save(state_dict, state_path)

    if save_config and getattr(model, "config", None) is not None:
        model.config.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    if base_model_path is None and getattr(model, "config", None) is not None:
        base_model_path = getattr(model.config, "_name_or_path", None)

    meta: Dict[str, Any] = {
        "format": "vaellm_state_dict_with_meta",
        "version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_path": base_model_path,
        "state_dict_file": STATE_DICT_FILENAME,
        "converted_module_count": len(converted_modules),
        "converted_modules": converted_modules,
        "adapter_module_count": len(adapter_modules),
        "adapter_modules": adapter_modules,
    }
    if extra_meta:
        meta["extra_meta"] = extra_meta

    meta_path = os.path.join(output_dir, META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    return {
        "state_dict": state_path,
        "meta": meta_path,
        "output_dir": output_dir,
    }


def _rebuild_adapter_modules(model: nn.Module, adapter_modules: Sequence[Dict[str, Any]]) -> None:
    proxy_specs = [spec for spec in adapter_modules if str(spec.get("adapter_type")) == "peft_proxy_lora"]
    if proxy_specs:
        first = proxy_specs[0]
        requested_rank = int(first["r"])
        requested_alpha = float(first["alpha"])
        requested_dropout = float(first.get("dropout", 0.0))
        requested_rslora = bool(first.get("use_rslora", False))
        for spec in proxy_specs:
            name = str(spec["name"])
            module = _get_module_by_name(model, name)
            ensure_peft_vae_linear_proxy(model, name, module)
            if int(spec["r"]) != requested_rank:
                raise ValueError("All peft_proxy_lora modules must share the same rank.")
            if float(spec["alpha"]) != requested_alpha:
                raise ValueError("All peft_proxy_lora modules must share the same alpha.")
            if float(spec.get("dropout", 0.0)) != requested_dropout:
                raise ValueError("All peft_proxy_lora modules must share the same dropout.")
            if bool(spec.get("use_rslora", False)) != requested_rslora:
                raise ValueError("All peft_proxy_lora modules must share the same use_rslora value.")
        ensure_peft_vae_proxy_lora(
            model,
            rank=requested_rank,
            alpha=requested_alpha,
            dropout=requested_dropout,
            use_rslora=requested_rslora,
        )

    for spec in adapter_modules:
        adapter_type = str(spec.get("adapter_type"))
        if adapter_type == "peft_proxy_lora":
            continue
        name = str(spec["name"])
        module = _get_module_by_name(model, name)
        if adapter_type == "vae_lora":
            ensure_lora_vae_linear(
                model,
                name,
                module,
                rank=int(spec["r"]),
                alpha=float(spec["alpha"]),
                dropout=float(spec.get("dropout", 0.0)),
            )
            continue
        if adapter_type == "linear_lora":
            ensure_lora_linear(
                model,
                name,
                module,
                rank=int(spec["r"]),
                alpha=float(spec["alpha"]),
                dropout=float(spec.get("dropout", 0.0)),
            )
            continue
        if adapter_type == "embedding_lora":
            ensure_lora_embedding(
                model,
                name,
                module,
                rank=int(spec["r"]),
                alpha=float(spec["alpha"]),
                dropout=float(spec.get("dropout", 0.0)),
            )
            continue
        raise ValueError(f"Unsupported adapter_type: {spec.get('adapter_type')}")


def _materialize_missing_proxy_dense_base_from_model(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    current_state = model.state_dict()
    for name, _proxy in iter_named_peft_vae_proxies(model):
        for suffix in ("weight", "bias"):
            key = f"{name}.per_decoded_linear.base_layer.{suffix}"
            if key in current_state and key not in state_dict:
                state_dict[key] = current_state[key]


def load_e2e_checkpoint_into_model(
    model: nn.Module,
    model_dir: str,
    *,
    map_location: str = "cpu",
    strict: bool = True,
):
    meta_path = os.path.join(model_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    converted_modules = meta.get("converted_modules", [])
    if converted_modules:
        _rebuild_converted_modules(model, converted_modules)

    adapter_modules = meta.get("adapter_modules", [])
    if adapter_modules and _checkpoint_has_extra_lora(adapter_modules) and _embedding_and_lm_head_are_tied(model):
        separate_embeddings_and_lm_head(model)
    if adapter_modules:
        _rebuild_adapter_modules(model, adapter_modules)

    state_dict_file = str(meta.get("state_dict_file", STATE_DICT_FILENAME))
    state_dict_path = os.path.join(model_dir, state_dict_file)
    state_dict = _torch_load_state_dict(state_dict_path, map_location=map_location)
    model_state_keys = tuple(model.state_dict().keys())
    state_dict, _remap_count = _remap_legacy_parallel_linear_state_dict_keys(state_dict, model_state_keys)
    _materialize_missing_bias_params_from_state_dict(model, state_dict)
    _materialize_missing_proxy_dense_base_from_model(model, state_dict)

    load_result = model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, meta, load_result


def load_e2e_model_checkpoint(
    model_dir: str,
    *,
    access_token: Optional[str] = None,
    base_model_path: Optional[str] = None,
    map_location: str = "cpu",
    strict: bool = True,
):
    meta_path = os.path.join(model_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    base_path = base_model_path or meta.get("base_model_path")
    if not base_path:
        raise ValueError("base_model_path is required (not found in meta and not provided).")

    model = get_model(base_path, access_token)
    return load_e2e_checkpoint_into_model(
        model=model,
        model_dir=model_dir,
        map_location=map_location,
        strict=strict,
    )
