from typing import Optional, Union

import torch
from torch import nn

import rotation.model_utils as model_utils


def load_frozen_base_reference_model(
    model_path: str,
    *,
    access_token: Optional[str],
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    model = model_utils.get_model(model_path, access_token)
    if dtype is not None:
        model.to(dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    if hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = False
    model.to(device)
    return model


def get_reference_module(model: nn.Module, module_name: str) -> nn.Module:
    current: object = model
    for part in str(module_name).split("."):
        if not part:
            raise ValueError(f"Invalid empty path segment in reference module path: {module_name}")
        if part.isdigit():
            try:
                current = current[int(part)]  # type: ignore[index]
                continue
            except (TypeError, IndexError, KeyError):
                pass
        if isinstance(current, nn.Module) and part in current._modules:
            current = current._modules[part]
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        raise ValueError(f"Reference module path not found: {module_name}")
    if not isinstance(current, nn.Module):
        raise ValueError(f"Reference module path does not resolve to nn.Module: {module_name}")
    return current


def clone_frozen_linear_from_reference(
    reference_model: nn.Module,
    module_name: str,
    *,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> nn.Linear:
    source = get_reference_module(reference_model, module_name)
    if not isinstance(source, nn.Linear):
        raise ValueError(f"Reference module is not nn.Linear: {module_name}")

    target_dtype = dtype if dtype is not None else source.weight.dtype
    cloned = nn.Linear(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        device=device,
        dtype=target_dtype,
    )
    cloned.weight = nn.Parameter(
        source.weight.detach().clone().to(device=device, dtype=target_dtype),
        requires_grad=False,
    )
    if source.bias is not None:
        cloned.bias = nn.Parameter(
            source.bias.detach().clone().to(device=device, dtype=target_dtype),
            requires_grad=False,
        )
    cloned.eval()
    return cloned
