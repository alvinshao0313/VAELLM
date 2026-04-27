from dataclasses import dataclass
from typing import List

import torch
from torch import nn

try:
    from peft.tuners.adalora.layer import SVDLinear as PeftAdaLoraLinear
    from peft.tuners.lora.layer import Linear as PeftLoraLinear
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError("未安装 peft。请先安装：pip install peft") from exc


_SMOOTH_PARAM_NAME = "lora_smooth_log"
_SMOOTH_TARGET_NAME_ATTR = "_raw_lora_smooth_target_name"
_SMOOTH_HOOK_ATTR = "_raw_lora_smooth_hook_handle"
_PEFT_MODEL_PREFIX = "base_model.model."


@dataclass(frozen=True)
class LoraSmoothFusionSpec:
    target_name: str
    smooth: torch.Tensor


@dataclass(frozen=True)
class LoraSmoothInfo:
    module_names: List[str]
    module_count: int
    parameter_count: int


def _is_supported_peft_linear(module: nn.Module) -> bool:
    return isinstance(module, (PeftLoraLinear, PeftAdaLoraLinear))


def _strip_peft_model_prefix(name: str) -> str:
    text = str(name)
    if text.startswith(_PEFT_MODEL_PREFIX):
        return text[len(_PEFT_MODEL_PREFIX):]
    return text


def _make_smooth_pre_hook():
    def hook(module, args, kwargs):
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            return None

        smooth_log = getattr(module, _SMOOTH_PARAM_NAME)
        smooth = torch.exp(smooth_log).to(device=x.device, dtype=x.dtype)
        if int(x.shape[-1]) != int(smooth.numel()):
            raise RuntimeError(
                f"LoRA smooth input width mismatch: input={int(x.shape[-1])}, smooth={int(smooth.numel())}."
            )
        view_shape = [1] * int(x.dim())
        view_shape[-1] = int(smooth.numel())
        return (x / smooth.view(*view_shape),) + tuple(args[1:]), kwargs

    return hook


def apply_lora_smooth(model: nn.Module) -> LoraSmoothInfo:
    module_names: List[str] = []
    parameter_count = 0
    for name, module in model.named_modules():
        if not _is_supported_peft_linear(module):
            continue

        base_layer = module.get_base_layer()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRA smooth expects nn.Linear base layer at '{name}', got {type(base_layer)}.")

        in_features = int(base_layer.in_features)
        existing = getattr(module, _SMOOTH_PARAM_NAME, None)
        if existing is None:
            smooth_log = nn.Parameter(
                torch.zeros(
                    (in_features,),
                    dtype=torch.float32,
                    device=base_layer.weight.device,
                )
            )
            module.register_parameter(_SMOOTH_PARAM_NAME, smooth_log)
        elif not isinstance(existing, nn.Parameter):
            raise TypeError(f"Existing {_SMOOTH_PARAM_NAME} at '{name}' is not an nn.Parameter.")
        elif int(existing.numel()) != in_features:
            raise ValueError(
                f"Existing {_SMOOTH_PARAM_NAME} at '{name}' has {int(existing.numel())} values, "
                f"expected {in_features}."
            )

        getattr(module, _SMOOTH_PARAM_NAME).requires_grad_(True)
        setattr(module, _SMOOTH_TARGET_NAME_ATTR, _strip_peft_model_prefix(name))
        old_handle = getattr(module, _SMOOTH_HOOK_ATTR, None)
        if old_handle is not None:
            old_handle.remove()
        setattr(module, _SMOOTH_HOOK_ATTR, module.register_forward_pre_hook(_make_smooth_pre_hook(), with_kwargs=True))
        module_names.append(str(name))
        parameter_count += in_features

    if not module_names:
        raise RuntimeError("启用 lora_smooth 失败：未找到 PEFT LoRA Linear 模块。")
    return LoraSmoothInfo(
        module_names=module_names,
        module_count=len(module_names),
        parameter_count=int(parameter_count),
    )


@torch.no_grad()
def collect_lora_smooth_fusion_specs(model: nn.Module) -> List[LoraSmoothFusionSpec]:
    specs: List[LoraSmoothFusionSpec] = []
    for name, module in model.named_modules():
        if not _is_supported_peft_linear(module):
            continue
        smooth_log = getattr(module, _SMOOTH_PARAM_NAME, None)
        if smooth_log is None:
            continue
        if not isinstance(smooth_log, torch.Tensor):
            raise TypeError(f"{_SMOOTH_PARAM_NAME} at '{name}' is not a tensor.")

        smooth = torch.exp(smooth_log.detach()).to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(smooth).all():
            raise ValueError(f"Non-finite LoRA smooth values found at '{name}'.")
        target_name = getattr(module, _SMOOTH_TARGET_NAME_ATTR, _strip_peft_model_prefix(name))
        specs.append(LoraSmoothFusionSpec(target_name=str(target_name), smooth=smooth))
    return specs


@torch.no_grad()
def fuse_lora_smooth_into_linear_weights(model: nn.Module, specs: List[LoraSmoothFusionSpec]) -> int:
    if not specs:
        return 0

    module_map = dict(model.named_modules())
    fused = 0
    for spec in specs:
        module = module_map.get(spec.target_name)
        if module is None:
            raise ValueError(f"Cannot fuse LoRA smooth: merged model has no module '{spec.target_name}'.")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Cannot fuse LoRA smooth into '{spec.target_name}': expected nn.Linear, got {type(module)}.")
        if int(module.weight.shape[1]) != int(spec.smooth.numel()):
            raise ValueError(
                f"Cannot fuse LoRA smooth into '{spec.target_name}': weight input width={int(module.weight.shape[1])}, "
                f"smooth={int(spec.smooth.numel())}."
            )

        weight_fp32 = module.weight.detach().to(dtype=torch.float32)
        smooth = spec.smooth.to(device=module.weight.device, dtype=torch.float32).view(1, -1)
        module.weight.copy_((weight_fp32 / smooth).to(dtype=module.weight.dtype))
        fused += 1
    return fused
