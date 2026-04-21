import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from e2e_common.post_norm_head import ensure_post_norm_head_linear
from e2e_common.peft_proxy import (
    collect_peft_vae_proxy_adapter_specs,
    ensure_peft_vae_linear_proxy,
    ensure_peft_vae_proxy_adapter,
    inject_peft_proxy_adalora_runtime_state_dict,
    iter_named_peft_vae_proxies,
    materialize_peft_proxy_decoded_linears,
    pop_peft_proxy_adalora_runtime_state_dict,
    restore_peft_proxy_adalora_runtime_state_dict,
    strip_proxy_dense_base_from_state_dict,
)
from e2e_common.proxy_trainables import iter_named_vae_module_refs
from rotation.model_utils import get_model
from train_utils.model_checkpoint_io import (
    META_FILENAME,
    STATE_DICT_FILENAME,
    _collect_sparse_residual_specs,
    _decoder_to_spec,
    _dtype_to_name,
    _get_module_by_name,
    _materialize_missing_bias_params_from_state_dict,
    _remap_legacy_parallel_linear_state_dict_keys,
    _rebuild_converted_modules,
    _torch_load_state_dict,
    unload_vae_original_linear_weights,
)


_E2E_FINETUNE_MODE = "vae_lora"


def _tensor_spec(tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if not isinstance(tensor, torch.Tensor):
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_to_name(tensor.dtype),
    }


def _reject_removed_extra_lora_checkpoint(meta: Dict[str, Any]) -> None:
    extra_meta = meta.get("extra_meta", {}) if isinstance(meta.get("extra_meta"), dict) else {}
    if bool(extra_meta.get("lora_embedding", False)) or bool(extra_meta.get("lora_lm_head", False)):
        raise ValueError("embedding/head LoRA checkpoint is no longer supported in e2e_common.")
    adapter_modules = meta.get("adapter_modules", [])
    if not isinstance(adapter_modules, list):
        return
    for spec in adapter_modules:
        adapter_type = str(spec.get("adapter_type"))
        if adapter_type not in {"peft_proxy_lora", "peft_proxy_adalora"}:
            raise ValueError(f"Unsupported adapter_type for e2e_common: {adapter_type}")


def _remap_legacy_decoder_keys_if_needed(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    logger=None,
) -> Dict[str, torch.Tensor]:
    model_state_keys = tuple(model.state_dict().keys())
    remapped_state_dict, remap_count = _remap_legacy_parallel_linear_state_dict_keys(
        state_dict,
        model_state_keys,
    )
    if int(remap_count) > 0 and logger is not None:
        logger.info(
            "Detected legacy cat checkpoint decoder key layout. Applied automatic key remap count=%d.",
            int(remap_count),
        )
    return remapped_state_dict


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
    stage_restore_row_specs = None
    stage_restore_col_specs = None
    stage_part_restore_col_specs = None
    if residual_stages > 1:
        stage_restore_row_specs = [
            _tensor_spec(module.get_stage_restore_row_indices(stage_idx))
            for stage_idx in range(residual_stages)
        ]
        stage_restore_col_specs = [
            _tensor_spec(module.get_stage_restore_col_indices(stage_idx))
            for stage_idx in range(residual_stages)
        ]
        stage_part_restore_col_specs = [
            _tensor_spec(module.get_stage_part_restore_col_indices(stage_idx))
            for stage_idx in range(residual_stages)
        ]

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
        "part_restore_col_indices": _tensor_spec(getattr(module, "part_restore_col_indices", None)),
        "stage_restore_row_indices": stage_restore_row_specs,
        "stage_restore_col_indices": stage_restore_col_specs,
        "stage_part_restore_col_indices": stage_part_restore_col_specs,
        "protected_input_indices": _tensor_spec(getattr(module, "protected_input_indices", None)),
        "protected_input_weight": _tensor_spec(getattr(module, "protected_input_weight", None)),
        "protected_output_indices": _tensor_spec(getattr(module, "protected_output_indices", None)),
        "protected_output_weight": _tensor_spec(getattr(module, "protected_output_weight", None)),
        **_collect_sparse_residual_specs(module),
    }


def _collect_e2e_module_specs(model: nn.Module):
    converted_modules: List[Dict[str, Any]] = []
    for ref in iter_named_vae_module_refs(model):
        converted_modules.append(_collect_single_vae_linear_spec(ref.name, ref.base_layer))
    adapter_modules = collect_peft_vae_proxy_adapter_specs(
        model,
        train_mode=str(getattr(model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)),
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
    strip_proxy_dense_base_from_state_dict(
        model,
        compact_state_dict,
        keep_bias=bool(getattr(model, "_e2e_vae_lora_tune_bias", False)),
    )
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
    inject_peft_proxy_adalora_runtime_state_dict(model, state_dict)

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
        "version": 5,
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


def _rebuild_proxy_adapter_modules(
    model: nn.Module,
    adapter_modules: Sequence[Dict[str, Any]],
    *,
    extra_meta: Optional[Dict[str, Any]],
) -> None:
    lora_specs = [spec for spec in adapter_modules if str(spec.get("adapter_type")) == "peft_proxy_lora"]
    adalora_specs = [spec for spec in adapter_modules if str(spec.get("adapter_type")) == "peft_proxy_adalora"]
    if lora_specs and adalora_specs:
        raise ValueError("Mixed peft_proxy_lora and peft_proxy_adalora checkpoint is not supported.")

    proxy_specs = lora_specs or adalora_specs
    if not proxy_specs:
        return

    for spec in proxy_specs:
        name = str(spec["name"])
        module = _get_module_by_name(model, name)
        ensure_peft_vae_linear_proxy(model, name, module)

    if lora_specs:
        first = lora_specs[0]
        requested_rank = int(first["r"])
        requested_alpha = float(first["alpha"])
        requested_dropout = float(first.get("dropout", 0.0))
        requested_bias_mode = str(first.get("bias", "none"))
        requested_rslora = bool(first.get("use_rslora", False))
        requested_dora = bool(first.get("use_dora", False))
        if requested_rslora and requested_dora:
            raise ValueError("Checkpoint cannot enable both rsLoRA and DoRA at the same time.")
        for spec in lora_specs:
            if int(spec["r"]) != requested_rank:
                raise ValueError("All peft_proxy_lora modules must share the same rank.")
            if float(spec["alpha"]) != requested_alpha:
                raise ValueError("All peft_proxy_lora modules must share the same alpha.")
            if float(spec.get("dropout", 0.0)) != requested_dropout:
                raise ValueError("All peft_proxy_lora modules must share the same dropout.")
            if str(spec.get("bias", requested_bias_mode)) != requested_bias_mode:
                raise ValueError("All peft_proxy_lora modules must share the same bias mode.")
            if bool(spec.get("use_rslora", False)) != requested_rslora:
                raise ValueError("All peft_proxy_lora modules must share the same use_rslora value.")
            if bool(spec.get("use_dora", False)) != requested_dora:
                raise ValueError("All peft_proxy_lora modules must share the same use_dora value.")
        variant = "dora" if requested_dora else ("rslora" if requested_rslora else "plain")
        ensure_peft_vae_proxy_adapter(
            model,
            variant=variant,
            rank=requested_rank,
            alpha=requested_alpha,
            dropout=requested_dropout,
            bias_mode=requested_bias_mode,
            init_mode="zero",
            materialize_before_inject=False,
        )
        return

    first = adalora_specs[0]
    requested_alpha = float(first["alpha"])
    requested_dropout = float(first.get("dropout", 0.0))
    requested_bias_mode = str(first.get("bias", "none"))
    requested_target_r = int(first.get("target_r", extra_meta.get("vae_adalora_target_r")))
    requested_init_r = int(first.get("init_r", first["r"]))
    requested_tinit = int(first.get("tinit", extra_meta.get("vae_adalora_tinit", 0)))
    requested_tfinal = int(first.get("tfinal", extra_meta.get("vae_adalora_tfinal", 0)))
    requested_delta_t = int(first.get("delta_t", extra_meta.get("vae_adalora_delta_t", 1)))
    requested_beta1 = float(first.get("beta1", extra_meta.get("vae_adalora_beta1", 0.85)))
    requested_beta2 = float(first.get("beta2", extra_meta.get("vae_adalora_beta2", 0.85)))
    requested_orth = float(first.get("orth_reg_weight", extra_meta.get("vae_adalora_orth_reg_weight", 0.5)))
    requested_total_step = first.get("total_step", extra_meta.get("vae_adalora_total_step"))
    requested_total_step = None if requested_total_step is None else int(requested_total_step)
    for spec in adalora_specs:
        if int(spec["r"]) != requested_init_r:
            raise ValueError("All peft_proxy_adalora modules must share the same init_r.")
        if float(spec["alpha"]) != requested_alpha:
            raise ValueError("All peft_proxy_adalora modules must share the same alpha.")
        if float(spec.get("dropout", 0.0)) != requested_dropout:
            raise ValueError("All peft_proxy_adalora modules must share the same dropout.")
        if str(spec.get("bias", requested_bias_mode)) != requested_bias_mode:
            raise ValueError("All peft_proxy_adalora modules must share the same bias mode.")
        if int(spec.get("target_r", requested_target_r)) != requested_target_r:
            raise ValueError("All peft_proxy_adalora modules must share the same target_r.")
        if int(spec.get("init_r", requested_init_r)) != requested_init_r:
            raise ValueError("All peft_proxy_adalora modules must share the same init_r.")
        if int(spec.get("tinit", requested_tinit)) != requested_tinit:
            raise ValueError("All peft_proxy_adalora modules must share the same tinit.")
        if int(spec.get("tfinal", requested_tfinal)) != requested_tfinal:
            raise ValueError("All peft_proxy_adalora modules must share the same tfinal.")
        if int(spec.get("delta_t", requested_delta_t)) != requested_delta_t:
            raise ValueError("All peft_proxy_adalora modules must share the same delta_t.")
        if float(spec.get("beta1", requested_beta1)) != requested_beta1:
            raise ValueError("All peft_proxy_adalora modules must share the same beta1.")
        if float(spec.get("beta2", requested_beta2)) != requested_beta2:
            raise ValueError("All peft_proxy_adalora modules must share the same beta2.")
        if float(spec.get("orth_reg_weight", requested_orth)) != requested_orth:
            raise ValueError("All peft_proxy_adalora modules must share the same orth_reg_weight.")
    ensure_peft_vae_proxy_adapter(
        model,
        variant="adalora",
        rank=requested_init_r,
        alpha=requested_alpha,
        dropout=requested_dropout,
        bias_mode=requested_bias_mode,
        init_mode="zero",
        total_step=requested_total_step,
        adalora_target_r=requested_target_r,
        adalora_init_r=requested_init_r,
        adalora_tinit=requested_tinit,
        adalora_tfinal=requested_tfinal,
        adalora_delta_t=requested_delta_t,
        adalora_beta1=requested_beta1,
        adalora_beta2=requested_beta2,
        adalora_orth_reg_weight=requested_orth,
        materialize_before_inject=False,
    )


def _rebuild_adapter_modules(
    model: nn.Module,
    adapter_modules: Sequence[Dict[str, Any]],
    *,
    extra_meta: Optional[Dict[str, Any]],
) -> None:
    _rebuild_proxy_adapter_modules(model, adapter_modules, extra_meta=extra_meta or {})
    for spec in adapter_modules:
        adapter_type = str(spec.get("adapter_type"))
        if adapter_type in {"peft_proxy_lora", "peft_proxy_adalora"}:
            continue
        raise ValueError(f"Unsupported adapter_type for e2e_common: {spec.get('adapter_type')}")


def _infer_lora_tune_bias_from_adapter_specs(adapter_modules: Sequence[Dict[str, Any]]) -> bool:
    for spec in adapter_modules:
        if str(spec.get("adapter_type")) not in {"peft_proxy_lora", "peft_proxy_adalora"}:
            continue
        if str(spec.get("bias", "none")).strip().lower() == "lora_only":
            return True
    return False


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
    materialize_proxy_decoded_linears: bool = True,
    proxy_group_size: int = 8,
    proxy_compute_device: Optional[object] = None,
    proxy_logger=None,
):
    meta_path = os.path.join(model_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    _reject_removed_extra_lora_checkpoint(meta)

    converted_modules = meta.get("converted_modules", [])
    if converted_modules:
        _rebuild_converted_modules(model, converted_modules)

    adapter_modules = meta.get("adapter_modules", [])
    extra_meta = meta.get("extra_meta", {}) if isinstance(meta.get("extra_meta"), dict) else {}
    if bool(extra_meta.get("use_post_norm_head_linear", False)):
        ensure_post_norm_head_linear(model)
    setattr(
        model,
        "_e2e_vae_lora_tune_bias",
        bool(extra_meta.get("vae_lora_tune_bias", _infer_lora_tune_bias_from_adapter_specs(adapter_modules))),
    )
    if adapter_modules:
        _rebuild_adapter_modules(model, adapter_modules, extra_meta=extra_meta)

    state_dict_file = str(meta.get("state_dict_file", STATE_DICT_FILENAME))
    state_dict_path = os.path.join(model_dir, state_dict_file)
    state_dict = _torch_load_state_dict(state_dict_path, map_location=map_location)
    state_dict = _remap_legacy_decoder_keys_if_needed(model, state_dict, logger=proxy_logger)
    if any(str(key).startswith("lm_head.post_norm_linear.") for key in state_dict.keys()):
        ensure_post_norm_head_linear(model)
    adalora_runtime_state = pop_peft_proxy_adalora_runtime_state_dict(state_dict)
    _materialize_missing_bias_params_from_state_dict(model, state_dict)
    _materialize_missing_proxy_dense_base_from_model(model, state_dict)

    load_result = model.load_state_dict(state_dict, strict=strict)
    if materialize_proxy_decoded_linears:
        materialize_peft_proxy_decoded_linears(
            model,
            group_size=int(proxy_group_size),
            compute_device=map_location if proxy_compute_device is None else proxy_compute_device,
            logger=proxy_logger,
        )
    restore_peft_proxy_adalora_runtime_state_dict(model, adalora_runtime_state)
    model.eval()
    return model, meta, load_result


def load_e2e_model_checkpoint(
    model_dir: str,
    *,
    access_token: Optional[str] = None,
    base_model_path: Optional[str] = None,
    map_location: str = "cpu",
    strict: bool = True,
    materialize_proxy_decoded_linears: bool = True,
    proxy_group_size: int = 8,
    proxy_compute_device: Optional[object] = None,
    proxy_logger=None,
):
    meta_path = os.path.join(model_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    _reject_removed_extra_lora_checkpoint(meta)

    base_path = base_model_path or meta.get("base_model_path")
    if not base_path:
        raise ValueError("base_model_path is required (not found in meta and not provided).")

    model = get_model(base_path, access_token)
    return load_e2e_checkpoint_into_model(
        model=model,
        model_dir=model_dir,
        map_location=map_location,
        strict=strict,
        materialize_proxy_decoded_linears=materialize_proxy_decoded_linears,
        proxy_group_size=int(proxy_group_size),
        proxy_compute_device=proxy_compute_device,
        proxy_logger=proxy_logger,
    )
