import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from e2e_fintuning.lora import ensure_lora_vae_linear, iter_named_vae_module_refs
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
        if ref.adapter is None:
            continue
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
    return converted_modules, adapter_modules


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
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    if state_dict is None and unload_vae_original_weights:
        unload_vae_original_linear_weights(model)

    if state_dict is None:
        state_dict = model.state_dict()

    state_path = os.path.join(output_dir, STATE_DICT_FILENAME)
    torch.save(state_dict, state_path)

    if save_config and getattr(model, "config", None) is not None:
        model.config.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    if base_model_path is None and getattr(model, "config", None) is not None:
        base_model_path = getattr(model.config, "_name_or_path", None)

    converted_modules, adapter_modules = _collect_e2e_module_specs(model)
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
    for spec in adapter_modules:
        if str(spec.get("adapter_type")) != "vae_lora":
            raise ValueError(f"Unsupported adapter_type: {spec.get('adapter_type')}")
        name = str(spec["name"])
        module = _get_module_by_name(model, name)
        ensure_lora_vae_linear(
            model,
            name,
            module,
            rank=int(spec["r"]),
            alpha=float(spec["alpha"]),
            dropout=float(spec.get("dropout", 0.0)),
        )


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
    if adapter_modules:
        _rebuild_adapter_modules(model, adapter_modules)

    state_dict_file = str(meta.get("state_dict_file", STATE_DICT_FILENAME))
    state_dict_path = os.path.join(model_dir, state_dict_file)
    state_dict = _torch_load_state_dict(state_dict_path, map_location=map_location)
    model_state_keys = tuple(model.state_dict().keys())
    state_dict, _remap_count = _remap_legacy_parallel_linear_state_dict_keys(state_dict, model_state_keys)
    _materialize_missing_bias_params_from_state_dict(model, state_dict)

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
