from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Sequence

import torch

ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_SAFE_WEIGHTS_FILENAME = "adapter_model.safetensors"
ADAPTER_BIN_WEIGHTS_FILENAME = "adapter_model.bin"


def read_json_dict(path: str, *, label: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a dict, got {type(payload)}.")
    return payload


def resolve_adapter_weights_path(adapter_dir: str) -> str:
    safe_path = os.path.join(str(adapter_dir), ADAPTER_SAFE_WEIGHTS_FILENAME)
    bin_path = os.path.join(str(adapter_dir), ADAPTER_BIN_WEIGHTS_FILENAME)
    if os.path.exists(safe_path):
        return safe_path
    if os.path.exists(bin_path):
        return bin_path
    raise FileNotFoundError(
        f"Missing adapter weights under {adapter_dir}: expected "
        f"{ADAPTER_SAFE_WEIGHTS_FILENAME} or {ADAPTER_BIN_WEIGHTS_FILENAME}."
    )


def read_adapter_weight_keys(adapter_dir: str) -> List[str]:
    weights_path = resolve_adapter_weights_path(adapter_dir)
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import safe_open

        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            return sorted(str(key) for key in handle.keys())

    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Adapter weights must be a state_dict dict, got {type(state_dict)}: {weights_path}")
    return sorted(str(key) for key in state_dict.keys())


def adapter_key_matches_module(key: str, module_name: str) -> bool:
    module_name = str(module_name).strip()
    if not module_name:
        return False
    return key == module_name or key.startswith(f"{module_name}.") or f".{module_name}." in key


def adapter_has_module_key(adapter_keys: Sequence[str], module_name: str) -> bool:
    return any(adapter_key_matches_module(str(key), module_name) for key in adapter_keys)


def validate_adapter_modules_to_save(adapter_dir: str, adapter_keys: Sequence[str]) -> Dict[str, Any]:
    adapter_config = read_json_dict(
        os.path.join(str(adapter_dir), ADAPTER_CONFIG_FILENAME),
        label=ADAPTER_CONFIG_FILENAME,
    )
    modules_to_save = adapter_config.get("modules_to_save")
    if modules_to_save is None:
        return adapter_config
    if not isinstance(modules_to_save, list):
        raise TypeError(f"adapter_config.modules_to_save must be a list or null, got {type(modules_to_save)}.")

    missing = [
        str(module_name)
        for module_name in modules_to_save
        if not adapter_has_module_key(adapter_keys, str(module_name))
    ]
    if missing:
        raise ValueError(
            "Adapter config declares modules_to_save but adapter weights do not contain them: "
            f"{missing}."
        )
    return adapter_config


def adapter_has_post_norm_head_linear(adapter_keys: Sequence[str]) -> bool:
    return adapter_has_module_key(adapter_keys, "lm_head.post_norm_linear")


def detach_tied_lm_head_weight_if_needed(model: torch.nn.Module, logger: logging.Logger) -> bool:
    lm_head = getattr(model, "lm_head", None)
    base_lm_head = getattr(lm_head, "lm_head", lm_head)
    model_body = getattr(model, "model", None)
    embed_tokens = getattr(model_body, "embed_tokens", None)
    if base_lm_head is None or embed_tokens is None:
        return False
    lm_weight = getattr(base_lm_head, "weight", None)
    embed_weight = getattr(embed_tokens, "weight", None)
    if lm_weight is None or embed_weight is None:
        return False
    if lm_weight.data_ptr() != embed_weight.data_ptr():
        return False

    base_lm_head.weight = torch.nn.Parameter(
        lm_weight.detach().clone(),
        requires_grad=bool(getattr(lm_weight, "requires_grad", False)),
    )
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
    logger.info("Detached tied lm_head.weight from embed_tokens.weight before post_norm_linear fusion.")
    return True


def build_peft_model_for_adapter_load(model: torch.nn.Module, adapter_dir: str):
    from peft import PeftConfig, PeftModel
    from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    peft_config.inference_mode = True
    peft_model_cls = MODEL_TYPE_TO_PEFT_MODEL_MAPPING.get(peft_config.task_type, PeftModel)
    return peft_model_cls(model, peft_config, adapter_name="default")


def assert_adapter_load_result_clean(load_result: object) -> None:
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []) or [])
    if unexpected_keys:
        raise RuntimeError(f"Adapter load produced unexpected keys: {unexpected_keys}")
