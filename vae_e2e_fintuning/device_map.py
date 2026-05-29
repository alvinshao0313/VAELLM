from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from rotation.model_utils import get_embeddings, get_layers, get_lm_head, get_model_type, get_pre_head_layernorm


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def _register_input_move_hook(module: nn.Module):
    def _hook(mod, args, kwargs):
        device = _module_device(mod)
        return _move_to_device(args, device), _move_to_device(kwargs, device)

    return module.register_forward_pre_hook(_hook, with_kwargs=True)


def _visible_devices() -> List[torch.device]:
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    return [torch.device(f"cuda:{idx}") for idx in range(int(torch.cuda.device_count()))]


def _split_layers_evenly(num_layers: int, devices: List[torch.device]) -> Dict[int, torch.device]:
    if int(num_layers) < 1:
        return {}
    out: Dict[int, torch.device] = {}
    for layer_idx in range(int(num_layers)):
        device_idx = min((layer_idx * len(devices)) // int(num_layers), len(devices) - 1)
        out[layer_idx] = devices[device_idx]
    return out


def _parse_device_token(token: str) -> torch.device:
    text = str(token).strip().lower()
    if text == "cpu":
        return torch.device("cpu")
    if text.isdigit():
        return torch.device(f"cuda:{int(text)}")
    if text == "cuda":
        return torch.device("cuda:0")
    if text.startswith("cuda:"):
        return torch.device(text)
    raise ValueError(f"Invalid device token in --layer_device_map: {token!r}")


def _parse_layer_range(token: str) -> Iterable[int]:
    text = str(token).strip()
    if "-" in text:
        begin_text, end_text = [part.strip() for part in text.split("-", 1)]
        begin = int(begin_text)
        end = int(end_text)
        if begin < 0 or end < begin:
            raise ValueError(f"Invalid layer range: {token!r}")
        return range(begin, end + 1)
    idx = int(text)
    if idx < 0:
        raise ValueError(f"Invalid layer index: {token!r}")
    return [idx]


def resolve_layer_device_map(spec: str, num_layers: int) -> Dict[int, torch.device]:
    normalized = str(spec or "auto").strip().lower()
    devices = _visible_devices()
    if normalized in {"", "auto"}:
        return _split_layers_evenly(int(num_layers), devices)
    if normalized in {"cpu", "cuda"} or normalized.startswith("cuda:") or normalized.isdigit():
        device = _parse_device_token(normalized)
        return {layer_idx: device for layer_idx in range(int(num_layers))}

    resolved: Dict[int, torch.device] = {}
    for item in normalized.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                "--layer_device_map entries must use device=range, for example cuda:0=0-15,cuda:1=16-31."
            )
        device_text, range_text = [part.strip() for part in entry.split("=", 1)]
        device = _parse_device_token(device_text)
        for layer_idx in _parse_layer_range(range_text):
            if int(layer_idx) >= int(num_layers):
                raise ValueError(f"Layer index {int(layer_idx)} is out of range for num_layers={int(num_layers)}.")
            if int(layer_idx) in resolved:
                raise ValueError(f"Layer {int(layer_idx)} appears more than once in --layer_device_map.")
            resolved[int(layer_idx)] = device
    missing = [idx for idx in range(int(num_layers)) if idx not in resolved]
    if missing:
        raise ValueError(f"--layer_device_map is missing layers: {missing}")
    return resolved


def apply_layer_device_map(model: nn.Module, *, layer_device_map: Dict[int, torch.device]) -> Tuple[List[object], Dict[str, str]]:
    layers = list(get_layers(model))
    if len(layers) != len(layer_device_map):
        raise ValueError(f"layer_device_map size {len(layer_device_map)} != model layers {len(layers)}.")

    first_device = layer_device_map[0]
    last_device = layer_device_map[len(layers) - 1]
    model_type = get_model_type(model)
    for module in get_embeddings(model, model_type):
        module.to(first_device)
    for layer_idx, layer in enumerate(layers):
        layer.to(layer_device_map[int(layer_idx)])
    get_pre_head_layernorm(model, model_type).to(last_device)
    get_lm_head(model, model_type).to(last_device)

    handles = []
    for module in get_embeddings(model, model_type):
        handles.append(_register_input_move_hook(module))
    for layer in layers:
        handles.append(_register_input_move_hook(layer))
    handles.append(_register_input_move_hook(get_pre_head_layernorm(model, model_type)))
    handles.append(_register_input_move_hook(get_lm_head(model, model_type)))

    hf_device_map = {f"model.layers.{idx}": str(device) for idx, device in layer_device_map.items()}
    hf_device_map["embed_tokens"] = str(first_device)
    hf_device_map["norm"] = str(last_device)
    hf_device_map["lm_head"] = str(last_device)
    setattr(model, "hf_device_map", hf_device_map)
    setattr(model, "is_parallelizable", True)
    setattr(model, "model_parallel", True)
    return handles, hf_device_map


def apply_boundary_device_map(model: nn.Module, *, layer_device_map: Dict[int, torch.device]) -> Tuple[List[object], Dict[str, str]]:
    layers = list(get_layers(model))
    if len(layers) != len(layer_device_map):
        raise ValueError(f"layer_device_map size {len(layer_device_map)} != model layers {len(layers)}.")

    first_device = layer_device_map[0]
    last_device = layer_device_map[len(layers) - 1]
    model_type = get_model_type(model)
    for module in get_embeddings(model, model_type):
        module.to(first_device)
    get_pre_head_layernorm(model, model_type).to(last_device)
    get_lm_head(model, model_type).to(last_device)

    handles = []
    for module in get_embeddings(model, model_type):
        handles.append(_register_input_move_hook(module))
    handles.append(_register_input_move_hook(get_pre_head_layernorm(model, model_type)))
    handles.append(_register_input_move_hook(get_lm_head(model, model_type)))

    hf_device_map = {
        "embed_tokens": str(first_device),
        "norm": str(last_device),
        "lm_head": str(last_device),
    }
    return handles, hf_device_map
