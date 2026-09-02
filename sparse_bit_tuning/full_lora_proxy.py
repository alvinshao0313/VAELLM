from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from e2e_common.compressed_subspace_lora import PeftZeroLinearCarrier
from e2e_common.peft_proxy import _adapter_uses_dora, _get_default_adapter_name, is_peft_adalora_linear, is_peft_lora_linear
from litebsq.low_rank_scope import LOW_RANK_SCOPE_FULL
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear


class FullCompressedPeftProxy(nn.Module):
    CARRIER_NAME = "full_adapter_linear"

    def __init__(self, base_layer: VAELinear):
        if not isinstance(base_layer, VAELinear):
            raise TypeError(f"Expected VAELinear, got {type(base_layer)}.")
        super().__init__()
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.temporary = bool(getattr(base_layer, "temporary", True))
        device, dtype = _resolve_float_device_dtype(base_layer)
        self.full_adapter_linear = PeftZeroLinearCarrier(
            self.in_features,
            self.out_features,
            device=device,
            dtype=dtype,
        )
        self.train(base_layer.training)

    def set_temporary(self, temporary: bool = True) -> None:
        self.base_layer.set_temporary(temporary)
        self.temporary = bool(getattr(self.base_layer, "temporary", True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_original = bool(getattr(self.base_layer, "always_use_original", False)) or not bool(self.temporary)
        if use_original:
            return self.base_layer(x)
        base_out = self.base_layer(x)
        carrier = self.full_adapter_linear
        ref_weight = carrier.base_layer.weight if is_peft_lora_linear(carrier) else carrier.weight
        if x.device != ref_weight.device:
            raise RuntimeError(f"full LoRA carrier/input device mismatch: input={x.device}, carrier={ref_weight.device}.")
        delta = carrier(x.to(dtype=ref_weight.dtype)).to(dtype=base_out.dtype)
        return base_out + delta


def _resolve_float_device_dtype(base_layer: VAELinear) -> tuple[torch.device, torch.dtype]:
    for param in base_layer.parameters():
        if param.is_floating_point():
            return param.device, param.dtype
    for buffer in base_layer.buffers():
        if buffer.is_floating_point():
            return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def _proxy_root(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        root = get_base_model()
        if isinstance(root, nn.Module):
            return root
    return model


def iter_named_full_compressed_peft_proxies(model: nn.Module) -> Iterator[Tuple[str, FullCompressedPeftProxy]]:
    root = _proxy_root(model)
    skip_prefixes: List[str] = []
    for name, module in root.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if not isinstance(module, FullCompressedPeftProxy):
            continue
        skip_prefixes.extend((f"{name}.base_layer", f"{name}.{FullCompressedPeftProxy.CARRIER_NAME}"))
        yield str(name), module


def wrap_full_compressed_peft_proxies(
    model: nn.Module,
    targets: Sequence[Tuple[str, VAELinear]],
) -> List[str]:
    names = [str(name) for name, _ in targets]
    if len(names) != len(set(names)):
        raise ValueError("full proxy targets contain duplicate module names.")
    if list(iter_named_full_compressed_peft_proxies(model)):
        raise RuntimeError("Refusing duplicate FullCompressedPeftProxy wrapping.")
    wrapped = []
    for name, base_layer in targets:
        if base_layer.has_low_rank_residual():
            raise RuntimeError(f"{name}: clear low_rank_a/b before full PEFT proxy wrapping.")
        proxy = FullCompressedPeftProxy(base_layer)
        set_module_by_name(model, str(name), proxy)
        wrapped.append(str(name))
    return wrapped


def build_full_compressed_peft_model(
    model: nn.Module,
    *,
    selected_modules: Sequence[Tuple[str, VAELinear]],
    initial_low_rank_payloads: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    rank: int,
    alpha: float,
    dropout: float,
) -> nn.Module:
    names = [str(name) for name, _ in selected_modules]
    if initial_low_rank_payloads is not None and set(initial_low_rank_payloads) != set(names):
        raise RuntimeError("Full proxy low-rank payload keys must exactly match targets.")
    if initial_low_rank_payloads is not None:
        for name, module in selected_modules:
            module._validate_low_rank_payload_tensors(
                initial_low_rank_payloads[name][0],
                initial_low_rank_payloads[name][1],
                scope=LOW_RANK_SCOPE_FULL,
            )
            module.register_parameter("low_rank_a", None)
            module.register_parameter("low_rank_b", None)
            module.clear_decoded_weight_cache()
    wrap_full_compressed_peft_proxies(model, selected_modules)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type=None,
            inference_mode=False,
            r=int(rank),
            target_modules=[FullCompressedPeftProxy.CARRIER_NAME],
            lora_alpha=float(alpha),
            lora_dropout=float(dropout),
            bias="none",
            init_lora_weights=True,
        ),
    )
    refs = list(iter_named_full_compressed_peft_proxies(peft_model))
    if len(refs) != len(selected_modules):
        raise RuntimeError(f"Full proxy PEFT count mismatch: {len(refs)} != {len(selected_modules)}.")
    if initial_low_rank_payloads is not None:
        initialize_full_proxy_lora_from_low_rank(
            peft_model,
            initial_low_rank_payloads,
            module_names=names,
        )
    return peft_model


def _select_refs(model: nn.Module, module_names: Sequence[str]) -> List[Tuple[str, FullCompressedPeftProxy]]:
    refs = dict(iter_named_full_compressed_peft_proxies(model))
    missing = [str(name) for name in module_names if str(name) not in refs]
    if missing:
        raise RuntimeError(f"Missing FullCompressedPeftProxy modules: {missing}")
    return [(str(name), refs[str(name)]) for name in module_names]


@torch.no_grad()
def initialize_full_proxy_lora_from_low_rank(
    model: nn.Module,
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    *,
    module_names: Sequence[str],
) -> int:
    initialized = 0
    for module_name, proxy in _select_refs(model, module_names):
        carrier = getattr(proxy, FullCompressedPeftProxy.CARRIER_NAME)
        if is_peft_adalora_linear(carrier) or not is_peft_lora_linear(carrier):
            raise TypeError(f"{module_name}: expected plain PEFT LoRA carrier, got {type(carrier)}.")
        adapter = _get_default_adapter_name(carrier)
        if _adapter_uses_dora(carrier, adapter):
            raise ValueError("DoRA is not supported by FullCompressedPeftProxy.")
        low_rank_a, low_rank_b = payloads[module_name]
        proxy.base_layer._validate_low_rank_payload_tensors(low_rank_a, low_rank_b, scope=LOW_RANK_SCOPE_FULL)
        lora_a = carrier.lora_A[adapter].weight
        lora_b = carrier.lora_B[adapter].weight
        scaling = float(carrier.scaling[adapter])
        if tuple(lora_a.shape) != tuple(low_rank_b.shape) or tuple(lora_b.shape) != tuple(low_rank_a.shape):
            raise RuntimeError(f"{module_name}: full LoRA payload shape mismatch.")
        lora_a.copy_(low_rank_b.to(device=lora_a.device, dtype=lora_a.dtype))
        lora_b.copy_((low_rank_a.to(device=lora_b.device, dtype=torch.float32) / scaling).to(lora_b.dtype))
        initialized += 1
    return int(initialized)


@torch.no_grad()
def extract_full_proxy_low_rank_payloads(
    model: nn.Module,
    *,
    module_names: Sequence[str],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for module_name, proxy in _select_refs(model, module_names):
        carrier = getattr(proxy, FullCompressedPeftProxy.CARRIER_NAME)
        if is_peft_adalora_linear(carrier) or not is_peft_lora_linear(carrier):
            raise TypeError(f"{module_name}: expected plain PEFT LoRA carrier, got {type(carrier)}.")
        adapter = _get_default_adapter_name(carrier)
        if _adapter_uses_dora(carrier, adapter):
            raise ValueError("DoRA is not supported by FullCompressedPeftProxy.")
        lora_a = carrier.lora_A[adapter].weight
        lora_b = carrier.lora_B[adapter].weight
        scaling = float(carrier.scaling[adapter])
        low_rank_b = lora_a.detach().to("cpu").clone().contiguous()
        low_rank_a = (
            lora_b.detach().to(device="cpu", dtype=torch.float32)
            .mul(float(scaling))
            .to(dtype=lora_b.dtype)
            .clone()
            .contiguous()
        )
        proxy.base_layer._validate_low_rank_payload_tensors(low_rank_a, low_rank_b, scope=LOW_RANK_SCOPE_FULL)
        payloads[module_name] = (low_rank_a, low_rank_b)
    return payloads
