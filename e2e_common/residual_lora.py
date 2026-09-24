"""Optional low-rank skip adapters for Llama/Qwen3 in Transformers 4.51.

The default additive mode uses ``residual = x + scale * B(A(x))``;
replace mode uses only ``residual = scale * B(A(x))``.
All state belongs to the layer; the forward has no hooks or activation stashes,
so gradient-checkpoint recomputation executes the same stateless computation.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

_SITE_NAMES = ("attention", "mlp")


class ResidualLoraSite(nn.Module):
    """An identity plus zero-initialized update, or a pure low-rank replacement."""

    def __init__(self, hidden_size: int, *, rank: int, alpha: float, dropout: float, mode: str = "additive") -> None:
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in ("additive", "replace"):
            raise ValueError(f"residual LoRA mode must be additive or replace, got {mode!r}.")
        rank, hidden_size = int(rank), int(hidden_size)
        if not 1 <= rank <= min(8, hidden_size):
            raise ValueError(f"residual LoRA rank must be in [1, min(8, hidden_size)], got {rank}.")
        alpha, dropout = float(alpha), float(dropout)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("residual LoRA alpha must be finite and > 0.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("residual LoRA dropout must be in [0, 1).")
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)
        if self.mode == "additive":
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            nn.init.orthogonal_(self.lora_A.weight)
            with torch.no_grad():
                self.lora_B.weight.copy_(self.lora_A.weight.T / self.scaling)

    @property
    def dropout_p(self) -> float:
        return float(self.dropout.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_B(self.lora_A(self.dropout(x))) * self.scaling).to(dtype=x.dtype)
        return x + delta if self.mode == "additive" else delta


def _residual_decoder_forward(
    self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
    output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs,
):
    # Keep the attention/MLP inputs and the original decoder return contract.
    residual = self._residual_lora_attention(hidden_states)
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
        past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
        cache_position=cache_position, position_embeddings=position_embeddings, **kwargs,
    )
    hidden_states = residual + hidden_states
    residual = self._residual_lora_mlp(hidden_states)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    return outputs


def _unwrap_model(model):
    root = model
    for _ in range(8):
        getter = getattr(root, "get_base_model", None)
        if not callable(getter):
            break
        candidate = getter()
        if candidate is root:
            break
        root = candidate
    return root


def _decoder_layers(model):
    return tuple(getattr(getattr(_unwrap_model(model), "model", None), "layers", ()))


def _actual_layer(layer):
    current = layer
    for _ in range(8):
        if all(hasattr(current, name) for name in ("input_layernorm", "self_attn", "mlp")):
            return current
        nested = getattr(current, "layer", None)
        if not isinstance(nested, nn.Module) or nested is current:
            break
        current = nested
    return current


def _family(model):
    model_type = getattr(getattr(_unwrap_model(model), "config", None), "model_type", "")
    if model_type not in ("llama", "qwen3"):
        raise ValueError(f"Residual LoRA supports only Llama and Qwen3; got {model_type!r}.")
    return model_type


def _site_config(site):
    return {"rank": site.rank, "alpha": site.alpha, "dropout": site.dropout_p}


def _installed_layers(model):
    result = []
    for index, candidate in enumerate(_decoder_layers(model)):
        layer = _actual_layer(candidate)
        installed = [isinstance(getattr(layer, f"_residual_lora_{name}", None), ResidualLoraSite)
                     for name in _SITE_NAMES]
        if any(installed) and not all(installed):
            raise ValueError(f"Incomplete residual LoRA topology on layer {index}.")
        if all(installed):
            result.append((index, layer))
    return tuple(result)


def install_residual_lora(model: nn.Module, *, rank: int, alpha: float, dropout: float = 0.0,
                          mode: str = "additive",
                          layer_indices: Optional[Sequence[int]] = None,
                          site_configs: Optional[Mapping[str, Mapping[str, object]]] = None) -> nn.Module:
    """Install in-place, preserving existing matching parameters across CAT stages."""
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    mode = str(mode).strip().lower()
    if mode not in ("additive", "replace"):
        raise ValueError(f"residual LoRA mode must be additive or replace, got {mode!r}.")
    family = _family(model)
    expected_type = {"llama": LlamaDecoderLayer, "qwen3": Qwen3DecoderLayer}[family]
    layers = _decoder_layers(model)
    if not layers:
        raise ValueError("Residual LoRA requires a non-empty model.layers decoder stack.")
    selected = tuple(range(len(layers))) if layer_indices is None else tuple(int(i) for i in layer_indices)
    if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= len(layers) for i in selected):
        raise ValueError(f"Invalid residual LoRA layer_indices: {selected}.")
    configs = {name: {"rank": int(rank), "alpha": float(alpha), "dropout": float(dropout)} for name in _SITE_NAMES}
    for name, overrides in (site_configs or {}).items():
        if name not in configs or set(overrides) - {"rank", "alpha", "dropout"}:
            raise ValueError(f"Invalid residual LoRA site config {name!r}.")
        configs[name].update(overrides)
    # Validate every target before modifying any existing layer.
    for index in selected:
        layer = _actual_layer(layers[index])
        if type(layer) is not expected_type:
            raise ValueError(f"Unsupported residual decoder class {type(layer).__name__}.")
        for name in _SITE_NAMES:
            site = getattr(layer, f"_residual_lora_{name}", None)
            if site is not None and (not isinstance(site, ResidualLoraSite) or _site_config(site) != configs[name] or site.mode != mode):
                raise ValueError(f"Residual LoRA config conflicts with existing layer {index}/{name}.")
        if not hasattr(layer, "_residual_lora_original_forward") and "forward" in layer.__dict__:
            raise ValueError("Cannot replace an already customized decoder forward with residual LoRA.")
    for index in selected:
        layer = _actual_layer(layers[index])
        for name, norm in (("attention", layer.input_layernorm), ("mlp", layer.post_attention_layernorm)):
            attr = f"_residual_lora_{name}"
            if not hasattr(layer, attr):
                site = ResidualLoraSite(norm.weight.numel(), **configs[name], mode=mode)
                site.to(device=norm.weight.device, dtype=norm.weight.dtype)
                site.train(layer.training)
                setattr(layer, attr, site)
        if not hasattr(layer, "_residual_lora_original_forward"):
            layer._residual_lora_original_forward = layer.forward
        layer.forward = MethodType(_residual_decoder_forward, layer)
    return model


def enable_residual_lora(model: nn.Module, enabled: bool = True) -> int:
    parameters = collect_residual_lora_parameters(model)
    for parameter in parameters.values():
        parameter.requires_grad_(enabled)
    return len(parameters)


def collect_residual_lora_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    return {f"model.layers.{index}._residual_lora_{site}.{name}": param
            for index, layer in _installed_layers(model) for site in _SITE_NAMES
            for name, param in getattr(layer, f"_residual_lora_{site}").named_parameters()}


def get_residual_lora_topology(model: nn.Module) -> Optional[Dict[str, object]]:
    installed = _installed_layers(model)
    if not installed:
        return None
    configs = {name: _site_config(getattr(installed[0][1], f"_residual_lora_{name}")) for name in _SITE_NAMES}
    mode = installed[0][1]._residual_lora_attention.mode
    for _index, layer in installed:
        if any(_site_config(getattr(layer, f"_residual_lora_{name}")) != configs[name]
               or getattr(layer, f"_residual_lora_{name}").mode != mode for name in _SITE_NAMES):
            raise ValueError("Residual LoRA topology differs between decoder layers.")
    return {"version": 1, "mode": mode, "family": _family(model),
            "layer_indices": [index for index, _layer in installed], "sites": configs}


def iter_residual_lora_parameters(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    return tuple(collect_residual_lora_parameters(model).items())


def remove_residual_lora(model: nn.Module, *, layer_indices: Optional[Sequence[int]] = None) -> int:
    wanted = None if layer_indices is None else set(layer_indices)
    removed = 0
    for index, layer in _installed_layers(model):
        if wanted is not None and index not in wanted:
            continue
        layer.forward = layer._residual_lora_original_forward
        delattr(layer, "_residual_lora_original_forward")
        for name in _SITE_NAMES:
            delattr(layer, f"_residual_lora_{name}")
        removed += 1
    return removed
