import math
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from peft import LoraConfig
from peft.mapping import inject_adapter_in_model
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from torch import nn

from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear


_DEFAULT_ADAPTER_NAME = "default"


def _resolve_reference_param(module: nn.Module) -> Optional[nn.Parameter]:
    for param in module.parameters():
        if param.is_floating_point():
            return param
    return None


def _resolve_proxy_dtype(base_layer: VAELinear) -> torch.dtype:
    ref_param = _resolve_reference_param(base_layer)
    if ref_param is not None and ref_param.is_floating_point():
        return ref_param.dtype
    return torch.float32


def _dropout_p(module: nn.Module) -> float:
    if isinstance(module, nn.Dropout):
        return float(module.p)
    return 0.0


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    if not module_name:
        return model
    current = model
    for token in str(module_name).split("."):
        if not hasattr(current, token):
            raise ValueError(f"Failed to resolve module '{module_name}': missing '{token}'.")
        current = getattr(current, token)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Resolved object at '{module_name}' is not an nn.Module: {type(current)}")
    return current


def is_peft_lora_linear(module: nn.Module) -> bool:
    return isinstance(module, PeftLoraLinear)


class PeftVAELinearProxy(nn.Module):
    def __init__(self, base_layer: VAELinear):
        if not isinstance(base_layer, VAELinear):
            raise TypeError(f"PeftVAELinearProxy expects VAELinear base_layer, got {type(base_layer)}")
        super().__init__()
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.temporary = bool(getattr(base_layer, "temporary", True))
        self.per_decoded_linear = self._build_decoded_linear()

        self.base_layer.cache_decoded_weight = False
        self.base_layer.clear_decoded_weight_cache()
        setattr(self.base_layer, "_skip_global_cache_prewarm", True)

    @torch.no_grad()
    def _build_decoded_linear(self) -> nn.Linear:
        previous_temporary = bool(getattr(self.base_layer, "temporary", True))
        self.base_layer.set_temporary(True)
        try:
            decoded_weight = self.base_layer._decode_weight(dtype=_resolve_proxy_dtype(self.base_layer)).detach()
        finally:
            self.base_layer.set_temporary(previous_temporary)
            self.base_layer.clear_decoded_weight_cache()

        bias = self.base_layer.bias
        decoded_bias = None if bias is None else bias.detach().to(device=decoded_weight.device, dtype=decoded_weight.dtype)
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=decoded_bias is not None,
            device=decoded_weight.device,
            dtype=decoded_weight.dtype,
        )
        linear.weight.requires_grad = False
        linear.weight.copy_(decoded_weight)
        if decoded_bias is not None:
            linear.bias.requires_grad = False
            linear.bias.copy_(decoded_bias)
        return linear

    def set_temporary(self, temporary: bool = True) -> None:
        self.base_layer.set_temporary(temporary)
        if bool(getattr(self.base_layer, "always_use_original", False)):
            self.temporary = False
        else:
            self.temporary = bool(temporary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_original = bool(getattr(self.base_layer, "always_use_original", False)) or not bool(self.temporary)
        if use_original:
            return self.base_layer(x)
        return self.per_decoded_linear(x)


def ensure_peft_vae_linear_proxy(
    model: nn.Module,
    module_name: str,
    module: nn.Module,
) -> PeftVAELinearProxy:
    if isinstance(module, PeftVAELinearProxy):
        return module
    if not isinstance(module, VAELinear):
        raise TypeError(f"Expected VAELinear or PeftVAELinearProxy at '{module_name}', got {type(module)}")
    proxy = PeftVAELinearProxy(module)
    proxy.train(module.training)
    set_module_by_name(model, module_name, proxy)
    return proxy


def iter_named_peft_vae_proxies(model: nn.Module) -> Iterator[Tuple[str, PeftVAELinearProxy]]:
    skip_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if not isinstance(module, PeftVAELinearProxy):
            continue
        skip_prefixes.append(f"{name}.base_layer")
        skip_prefixes.append(f"{name}.per_decoded_linear")
        yield name, module


def _get_default_adapter_name(module: PeftLoraLinear) -> str:
    if _DEFAULT_ADAPTER_NAME in module.lora_A:
        return _DEFAULT_ADAPTER_NAME
    if len(module.lora_A) != 1:
        raise ValueError("Only single-adapter plain PEFT LoRA is supported for VAELinear proxy export.")
    return next(iter(module.lora_A.keys()))


def _resolve_use_rslora(module: PeftLoraLinear, adapter_name: str) -> bool:
    rank = int(module.r[adapter_name])
    alpha = float(module.lora_alpha[adapter_name])
    scaling = float(module.scaling[adapter_name])
    standard = float(alpha) / float(rank)
    rslora = float(alpha) / math.sqrt(float(rank))
    if math.isclose(scaling, standard, rel_tol=1e-6, abs_tol=1e-6):
        return False
    if math.isclose(scaling, rslora, rel_tol=1e-6, abs_tol=1e-6):
        return True
    raise ValueError(
        f"Unsupported PEFT LoRA scaling for VAELinear proxy export: scaling={scaling}, "
        f"alpha={alpha}, rank={rank}."
    )


def _validate_existing_peft_proxy_linear(
    module_name: str,
    peft_linear: PeftLoraLinear,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    use_rslora: bool,
) -> None:
    adapter_name = _get_default_adapter_name(peft_linear)
    if bool(peft_linear.use_dora.get(adapter_name, False)):
        raise ValueError(f"VAELinear proxy at '{module_name}' has DoRA enabled, but当前实现只支持 plain LoRA。")
    actual_rank = int(peft_linear.r[adapter_name])
    actual_alpha = float(peft_linear.lora_alpha[adapter_name])
    actual_dropout = _dropout_p(peft_linear.lora_dropout[adapter_name])
    actual_rslora = _resolve_use_rslora(peft_linear, adapter_name)
    if actual_rank != int(rank) or actual_alpha != float(alpha) or actual_dropout != float(dropout):
        raise ValueError(
            f"Existing PEFT proxy LoRA at '{module_name}' has config "
            f"(rank={actual_rank}, alpha={actual_alpha}, dropout={actual_dropout}) "
            f"but requested (rank={rank}, alpha={alpha}, dropout={dropout})."
        )
    if actual_rslora != bool(use_rslora):
        raise ValueError(
            f"Existing PEFT proxy LoRA at '{module_name}' has use_rslora={actual_rslora}, "
            f"but requested {bool(use_rslora)}."
        )


def _validate_plain_peft_proxy_linear_for_residual_init(
    module_name: str,
    peft_linear: PeftLoraLinear,
) -> str:
    adapter_name = _get_default_adapter_name(peft_linear)
    if bool(peft_linear.merged):
        raise ValueError(f"PEFT proxy LoRA at '{module_name}' is already merged, cannot run residual_svd init.")
    if bool(peft_linear.use_dora.get(adapter_name, False)):
        raise ValueError(f"PEFT proxy LoRA at '{module_name}' has DoRA enabled, residual_svd init only supports plain LoRA.")
    if bool(_resolve_use_rslora(peft_linear, adapter_name)):
        raise ValueError(
            f"PEFT proxy LoRA at '{module_name}' uses rsLoRA scaling, residual_svd init only supports alpha/r scaling."
        )
    if bool(getattr(peft_linear, "fan_in_fan_out", False)):
        raise ValueError(
            f"PEFT proxy LoRA at '{module_name}' has fan_in_fan_out=True, residual_svd init only supports plain nn.Linear."
        )
    return adapter_name


def _enable_peft_proxy_adapters(model: nn.Module) -> None:
    for _name, proxy in iter_named_peft_vae_proxies(model):
        peft_linear = proxy.per_decoded_linear
        if not is_peft_lora_linear(peft_linear):
            continue
        adapter_name = _get_default_adapter_name(peft_linear)
        peft_linear.enable_adapters(True)
        peft_linear.set_adapter(adapter_name)


def ensure_peft_vae_proxy_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    use_rslora: bool = False,
) -> int:
    proxy_refs = list(iter_named_peft_vae_proxies(model))
    if not proxy_refs:
        return 0

    injected_count = 0
    for module_name, proxy in proxy_refs:
        per_decoded_linear = proxy.per_decoded_linear
        if is_peft_lora_linear(per_decoded_linear):
            _validate_existing_peft_proxy_linear(
                module_name,
                per_decoded_linear,
                rank=int(rank),
                alpha=float(alpha),
                dropout=float(dropout),
                use_rslora=bool(use_rslora),
            )
            injected_count += 1
            continue
        if not isinstance(per_decoded_linear, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear or PEFT Linear under '{module_name}.per_decoded_linear', "
                f"got {type(per_decoded_linear)}"
            )

    if injected_count not in {0, len(proxy_refs)}:
        raise ValueError("Detected partially injected PEFT VAELinear proxies. Refusing to continue.")

    if injected_count == 0:
        inject_adapter_in_model(
            LoraConfig(
                task_type=None,
                r=int(rank),
                lora_alpha=float(alpha),
                lora_dropout=float(dropout),
                target_modules=["per_decoded_linear"],
                bias="none",
                inference_mode=False,
                use_rslora=bool(use_rslora),
            ),
            model,
        )
        for module_name, proxy in proxy_refs:
            if not is_peft_lora_linear(proxy.per_decoded_linear):
                raise RuntimeError(f"Failed to inject PEFT LoRA into '{module_name}.per_decoded_linear'.")

    _enable_peft_proxy_adapters(model)
    return len(proxy_refs)


@torch.no_grad()
def initialize_peft_linear_from_residual_svd(
    peft_linear: PeftLoraLinear,
    residual: torch.Tensor,
    *,
    module_name: str,
) -> None:
    adapter_name = _validate_plain_peft_proxy_linear_for_residual_init(module_name, peft_linear)
    base_layer = peft_linear.base_layer
    if not isinstance(base_layer, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear base layer under '{module_name}', got {type(base_layer)}"
        )

    expected_shape = tuple(base_layer.weight.shape)
    if tuple(residual.shape) != expected_shape:
        raise ValueError(
            f"Residual shape mismatch at '{module_name}': got {tuple(residual.shape)}, expected {expected_shape}."
        )
    if not residual.is_floating_point():
        raise TypeError(f"Residual at '{module_name}' must be floating point, got {residual.dtype}.")

    weight_A = peft_linear.lora_A[adapter_name].weight
    weight_B = peft_linear.lora_B[adapter_name].weight
    scaling = float(peft_linear.scaling[adapter_name])
    if scaling <= 0.0:
        raise ValueError(f"PEFT proxy LoRA at '{module_name}' has invalid scaling={scaling}.")

    scaled_residual = residual.detach().to(device=weight_B.device, dtype=torch.float32) / scaling
    u, s, vh = torch.linalg.svd(scaled_residual, full_matrices=False)
    rank = int(weight_A.shape[0])
    k = min(rank, int(s.shape[0]))

    weight_A.zero_()
    weight_B.zero_()
    if k == 0:
        return

    sqrt_s = torch.sqrt(s[:k])
    b_factor = u[:, :k] * sqrt_s.unsqueeze(0)
    a_factor = sqrt_s.unsqueeze(1) * vh[:k, :]
    weight_B[:, :k].copy_(b_factor.to(device=weight_B.device, dtype=weight_B.dtype))
    weight_A[:k, :].copy_(a_factor.to(device=weight_A.device, dtype=weight_A.dtype))


@torch.no_grad()
def initialize_peft_vae_proxy_lora_from_teacher_residual(
    model: nn.Module,
    teacher_model: nn.Module,
) -> int:
    if teacher_model is None:
        raise ValueError("teacher_model is required for residual_svd init.")

    initialized = 0
    for name, proxy in iter_named_peft_vae_proxies(model):
        peft_linear = proxy.per_decoded_linear
        if not is_peft_lora_linear(peft_linear):
            raise ValueError(f"Expected PEFT LoRA linear under '{name}.per_decoded_linear' before residual_svd init.")
        if not isinstance(peft_linear.base_layer, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear base layer under '{name}.per_decoded_linear', got {type(peft_linear.base_layer)}"
            )

        teacher_module = _get_module_by_name(teacher_model, name)
        teacher_weight = getattr(teacher_module, "weight", None)
        if not isinstance(teacher_weight, torch.Tensor):
            raise ValueError(f"Teacher module '{name}' is missing weight, cannot run residual_svd init.")
        if not teacher_weight.is_floating_point():
            raise TypeError(f"Teacher module '{name}' weight must be floating point, got {teacher_weight.dtype}.")

        decoded_weight = peft_linear.base_layer.weight
        if tuple(teacher_weight.shape) != tuple(decoded_weight.shape):
            raise ValueError(
                f"Teacher/student weight shape mismatch at '{name}': "
                f"teacher={tuple(teacher_weight.shape)} student={tuple(decoded_weight.shape)}."
            )

        residual = teacher_weight.detach().to(device=decoded_weight.device, dtype=torch.float32) - decoded_weight.detach().to(
            device=decoded_weight.device,
            dtype=torch.float32,
        )
        initialize_peft_linear_from_residual_svd(
            peft_linear,
            residual,
            module_name=name,
        )
        initialized += 1
    return initialized


def collect_peft_vae_proxy_adapter_specs(
    model: nn.Module,
    *,
    train_mode: str,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for name, proxy in iter_named_peft_vae_proxies(model):
        peft_linear = proxy.per_decoded_linear
        if not is_peft_lora_linear(peft_linear):
            continue
        adapter_name = _get_default_adapter_name(peft_linear)
        if bool(peft_linear.use_dora.get(adapter_name, False)):
            raise ValueError(f"当前 compact checkpoint 保存不支持 DoRA: {name}")
        specs.append(
            {
                "name": name,
                "adapter_type": "peft_proxy_lora",
                "base_type": "PeftVAELinearProxy",
                "r": int(peft_linear.r[adapter_name]),
                "alpha": float(peft_linear.lora_alpha[adapter_name]),
                "dropout": float(_dropout_p(peft_linear.lora_dropout[adapter_name])),
                "use_rslora": bool(_resolve_use_rslora(peft_linear, adapter_name)),
                "train_mode_at_save": str(train_mode),
            }
        )
    return specs


def strip_proxy_dense_base_from_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> int:
    removed = 0
    for name, _proxy in iter_named_peft_vae_proxies(model):
        for suffix in ("weight", "bias"):
            key = f"{name}.per_decoded_linear.base_layer.{suffix}"
            if key in state_dict:
                state_dict.pop(key)
                removed += 1
    return removed


@torch.no_grad()
def refresh_peft_proxy_decoded_linears(model: nn.Module) -> int:
    refreshed = 0
    for name, proxy in iter_named_peft_vae_proxies(model):
        decoded_linear = proxy.per_decoded_linear
        if is_peft_lora_linear(decoded_linear):
            decoded_linear = decoded_linear.base_layer
        if not isinstance(decoded_linear, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear under '{name}.per_decoded_linear', got {type(decoded_linear)}"
            )

        target_device = decoded_linear.weight.device
        target_dtype = decoded_linear.weight.dtype
        decoded_weight = proxy.base_layer._decode_weight(dtype=target_dtype).to(
            device=target_device,
            dtype=target_dtype,
        )
        decoded_linear.weight.copy_(decoded_weight)

        base_bias = proxy.base_layer.bias
        if decoded_linear.bias is None:
            if base_bias is not None:
                raise ValueError(
                    f"Decoded linear under '{name}.per_decoded_linear' is missing bias while base_layer has bias."
                )
        else:
            if base_bias is None:
                decoded_linear.bias.zero_()
            else:
                decoded_linear.bias.copy_(
                    base_bias.detach().to(
                        device=decoded_linear.bias.device,
                        dtype=decoded_linear.bias.dtype,
                    )
                )
        refreshed += 1
    return refreshed


def convert_peft_vae_proxy_modules_to_lora(model: nn.Module) -> int:
    from e2e_fintuning.lora import LoRAVAELinear

    converted = 0
    proxy_refs = list(iter_named_peft_vae_proxies(model))
    for name, proxy in proxy_refs:
        peft_linear = proxy.per_decoded_linear
        if not is_peft_lora_linear(peft_linear):
            raise ValueError(f"Expected PEFT LoRA linear under '{name}.per_decoded_linear' before final export.")
        adapter_name = _get_default_adapter_name(peft_linear)
        if bool(peft_linear.use_dora.get(adapter_name, False)):
            raise ValueError("当前最终导出只支持 plain PEFT LoRA，不支持 DoRA。")
        if bool(_resolve_use_rslora(peft_linear, adapter_name)):
            raise ValueError("当前最终导出只支持 alpha/r 的 plain LoRA，不支持 rsLoRA。")

        wrapper = LoRAVAELinear(
            base_layer=proxy.base_layer,
            rank=int(peft_linear.r[adapter_name]),
            alpha=float(peft_linear.lora_alpha[adapter_name]),
            dropout=float(_dropout_p(peft_linear.lora_dropout[adapter_name])),
        )
        wrapper.train(proxy.training)
        wrapper.set_temporary(proxy.temporary)
        with torch.no_grad():
            wrapper.lora_A.weight.copy_(
                peft_linear.lora_A[adapter_name].weight.to(
                    device=wrapper.lora_A.weight.device,
                    dtype=wrapper.lora_A.weight.dtype,
                )
            )
            wrapper.lora_B.weight.copy_(
                peft_linear.lora_B[adapter_name].weight.to(
                    device=wrapper.lora_B.weight.device,
                    dtype=wrapper.lora_B.weight.dtype,
                )
            )
        wrapper.base_layer.cache_decoded_weight = True
        setattr(wrapper.base_layer, "_skip_global_cache_prewarm", False)
        set_module_by_name(model, name, wrapper)
        converted += 1
    return converted
