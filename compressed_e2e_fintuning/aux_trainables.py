from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import nn

from e2e_common.post_norm_head import resolve_post_norm_linear
from rotation.model_utils import get_model_type, get_pre_head_layernorm

AUX_CHECKPOINT_FILE = "e2e_aux_trainables.pt"
AUX_FORMAT = "compressed_e2e_aux_trainables"
AUX_VERSION = 1


@dataclass
class AuxiliaryTrainableSelection:
    parameters: Dict[str, nn.Parameter]
    bias_modules: list[str]
    final_norm_modules: list[str]
    post_norm_head_modules: list[str]


def _base_model(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        base = getter()
        if isinstance(base, nn.Module):
            return base
    return model


def _find_module_name(model: nn.Module, target: nn.Module, fallback: str) -> str:
    for name, module in model.named_modules():
        if module is target:
            return str(name)
    return str(fallback)


def _register_parameter(mapping: Dict[str, nn.Parameter], key: str, param: nn.Parameter) -> None:
    if key in mapping:
        raise RuntimeError(f"duplicate auxiliary trainable key: {key}")
    param.requires_grad_(True)
    mapping[str(key)] = param


def enable_compressed_lora_auxiliary_trainables(
    model: nn.Module,
    *,
    selected_vae_modules: Sequence[Tuple[str, Any]],
    sparse_bit_tuning: bool,
    tune_final_norm: bool,
    use_post_norm_head_linear: bool,
) -> AuxiliaryTrainableSelection:
    """Enable non-adapter continuous params after PEFT/proxy construction."""
    del selected_vae_modules, sparse_bit_tuning

    mapping: Dict[str, nn.Parameter] = {}
    bias_modules: list[str] = []
    final_norm_modules: list[str] = []
    post_norm_head_modules: list[str] = []

    root = _base_model(model)
    if bool(tune_final_norm):
        model_type = get_model_type(root)
        final_norm = get_pre_head_layernorm(root, model_type)
        final_norm_name = _find_module_name(root, final_norm, "model.norm")
        for rel_name, param in final_norm.named_parameters(recurse=True):
            _register_parameter(mapping, f"final_norm::{rel_name}", param)
        final_norm_modules.append(final_norm_name)

    if bool(use_post_norm_head_linear):
        post_norm = resolve_post_norm_linear(root)
        if post_norm is None:
            raise ValueError(
                "--use_post_norm_head_linear=true but model.lm_head is not LMHeadWithPostNormLinear."
            )
        post_name = _find_module_name(root, post_norm, "lm_head.post_norm_linear")
        for rel_name, param in post_norm.named_parameters(recurse=True):
            _register_parameter(mapping, f"post_norm_head::{rel_name}", param)
        post_norm_head_modules.append(post_name)

    return AuxiliaryTrainableSelection(
        parameters=mapping,
        bias_modules=sorted(set(bias_modules)),
        final_norm_modules=sorted(set(final_norm_modules)),
        post_norm_head_modules=sorted(set(post_norm_head_modules)),
    )


def snapshot_auxiliary_trainables(parameters: Mapping[str, nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {
        str(key): param.detach().to(device="cpu").clone().contiguous()
        for key, param in parameters.items()
    }


def save_auxiliary_sidecar(checkpoint_dir: str, payload: Mapping[str, torch.Tensor]) -> None:
    import os

    torch.save(
        {
            "format": AUX_FORMAT,
            "version": AUX_VERSION,
            "parameters": {str(k): v.detach().to("cpu").contiguous() for k, v in payload.items()},
        },
        os.path.join(str(checkpoint_dir), AUX_CHECKPOINT_FILE),
    )


def load_auxiliary_sidecar(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    import os

    path = os.path.join(str(checkpoint_dir), AUX_CHECKPOINT_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing compressed E2E auxiliary sidecar: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or str(payload.get("format")) != AUX_FORMAT or int(
        payload.get("version", -1)
    ) != AUX_VERSION:
        raise ValueError("Unsupported compressed E2E auxiliary sidecar format/version.")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("Auxiliary sidecar parameters must be a dict.")
    return {str(k): v.detach().to("cpu").contiguous() for k, v in parameters.items()}


@torch.no_grad()
def restore_auxiliary_trainables(
    parameters: Mapping[str, nn.Parameter],
    payload: Mapping[str, torch.Tensor],
) -> None:
    expected = set(parameters)
    provided = {str(key) for key in payload}
    if expected != provided:
        raise ValueError(
            f"Auxiliary checkpoint key mismatch: missing={sorted(expected - provided)} "
            f"extra={sorted(provided - expected)}"
        )
    for key, param in parameters.items():
        value = payload[str(key)]
        if tuple(value.shape) != tuple(param.shape):
            raise ValueError(f"{key}: auxiliary checkpoint shape {tuple(value.shape)} != {tuple(param.shape)}.")
        param.copy_(value.to(device=param.device, dtype=param.dtype))


@torch.no_grad()
def apply_auxiliary_payload_to_compressed_model(
    model: nn.Module,
    payload: Mapping[str, torch.Tensor],
) -> int:
    root = _base_model(model)
    final_norm = None
    post_norm = None
    written = 0
    for key, value in payload.items():
        text = str(key)
        if text.startswith("vae_bias::"):
            raise ValueError(
                f"vae_bias auxiliary payload is no longer supported: {text}. "
                "Re-export without VAE bias trainables."
            )
        elif text.startswith("final_norm::"):
            if final_norm is None:
                final_norm = get_pre_head_layernorm(root, get_model_type(root))
            rel_name = text.split("::", 1)[1]
            params = dict(final_norm.named_parameters(recurse=True))
            if rel_name not in params:
                raise RuntimeError(f"final norm auxiliary parameter missing on export model: {rel_name}")
            target = params[rel_name]
        elif text.startswith("post_norm_head::"):
            if post_norm is None:
                post_norm = resolve_post_norm_linear(root)
            if post_norm is None:
                raise RuntimeError("post-norm head auxiliary target missing on export model.")
            rel_name = text.split("::", 1)[1]
            params = dict(post_norm.named_parameters(recurse=True))
            if rel_name not in params:
                raise RuntimeError(f"post-norm head auxiliary parameter missing on export model: {rel_name}")
            target = params[rel_name]
        else:
            raise ValueError(f"Unknown auxiliary payload key: {text}")
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(f"{text}: export shape {tuple(target.shape)} != payload {tuple(value.shape)}.")
        target.copy_(value.to(device=target.device, dtype=target.dtype))
        written += 1
    return int(written)
