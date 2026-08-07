"""Compressed-subspace PEFT LoRA helpers.

Subspace proxy only owns coordinate mapping; LoRA math is owned by PEFT.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from peft import LoraConfig
from peft.mapping import inject_adapter_in_model
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from torch import nn

from e2e_common.peft_proxy import (
    _adapter_uses_dora,
    _get_default_adapter_name,
    is_peft_adalora_linear,
    is_peft_lora_linear,
)
from litebsq.low_rank_scope import (
    LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
    normalize_low_rank_scope,
)
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear


def _resolve_proxy_device_dtype(base_layer: VAELinear) -> tuple[torch.device, torch.dtype]:
    for param in base_layer.parameters():
        if param.is_floating_point():
            return param.device, param.dtype
    for buffer in base_layer.buffers():
        if buffer.is_floating_point():
            return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


class PeftZeroLinearCarrier(nn.Linear):
    """nn.Linear-compatible PEFT target with O(1) frozen base storage and identically-zero forward."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        requested_in = int(in_features)
        requested_out = int(out_features)
        if requested_in <= 0 or requested_out <= 0:
            raise ValueError(
                f"carrier feature dimensions must be positive, got "
                f"in={requested_in} out={requested_out}."
            )

        # 正常初始化一个真正的 1x1 nn.Linear，保证 PyTorch Module/Linear 内部状态标准；
        # 只占 1 个 scalar storage，不创建 [requested_out, requested_in] 权重。
        super().__init__(
            1,
            1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.in_features = requested_in
        self.out_features = requested_out
        with torch.no_grad():
            self.weight.zero_()
        self.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[-1]) != int(self.in_features):
            raise RuntimeError(
                f"carrier input dim {int(x.shape[-1])} != in_features {self.in_features}."
            )
        return x.new_zeros((*x.shape[:-1], self.out_features))


def _build_compressed_indices(
    *,
    total_features: int,
    compressed_features: int,
    protected_indices: Optional[torch.Tensor],
    axis_name: str,
) -> Optional[torch.Tensor]:
    total_features = int(total_features)
    compressed_features = int(compressed_features)
    if total_features <= 0 or compressed_features <= 0:
        raise ValueError(
            f"{axis_name}: feature dimensions must be positive, got "
            f"total={total_features} compressed={compressed_features}."
        )

    if protected_indices is None or int(protected_indices.numel()) == 0:
        if compressed_features != total_features:
            raise ValueError(
                f"{axis_name}: no protected indices but compressed_features="
                f"{compressed_features} != total_features={total_features}."
            )
        return None

    protected = protected_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if int(torch.unique(protected).numel()) != int(protected.numel()):
        raise ValueError(f"{axis_name}: protected indices contain duplicates.")
    if int(protected.min().item()) < 0 or int(protected.max().item()) >= total_features:
        raise ValueError(
            f"{axis_name}: protected indices out of range for total_features={total_features}."
        )

    keep_mask = torch.ones(total_features, dtype=torch.bool, device="cpu")
    keep_mask[protected] = False
    compressed = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    if int(compressed.numel()) != compressed_features:
        raise ValueError(
            f"{axis_name}: actual non-protected count={int(compressed.numel())} "
            f"!= compressed_features={compressed_features}; "
            f"total={total_features} protected={int(protected.numel())}."
        )
    return compressed.contiguous()


class CompressedSubspacePeftProxy(nn.Module):
    CARRIER_NAME = "compressed_subspace_adapter_linear"

    def __init__(self, base_layer: VAELinear):
        if not isinstance(base_layer, VAELinear):
            raise TypeError(f"Expected VAELinear, got {type(base_layer)}.")
        super().__init__()
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.compressed_in_features = int(base_layer.compressed_in_features)
        self.compressed_out_features = int(base_layer.compressed_out_features)
        self.temporary = bool(getattr(base_layer, "temporary", True))

        device, dtype = _resolve_proxy_device_dtype(base_layer)
        self.compressed_subspace_adapter_linear = PeftZeroLinearCarrier(
            self.compressed_in_features,
            self.compressed_out_features,
            device=device,
            dtype=dtype,
        )
        compressed_input_indices = _build_compressed_indices(
            total_features=self.in_features,
            compressed_features=self.compressed_in_features,
            protected_indices=base_layer.protected_input_indices,
            axis_name="input",
        )
        compressed_output_indices = _build_compressed_indices(
            total_features=self.out_features,
            compressed_features=self.compressed_out_features,
            protected_indices=base_layer.protected_output_indices,
            axis_name="output",
        )
        self.register_buffer(
            "compressed_input_indices",
            compressed_input_indices,
            persistent=False,
        )
        self.register_buffer(
            "compressed_output_indices",
            compressed_output_indices,
            persistent=False,
        )
        self.train(base_layer.training)

    def set_temporary(self, temporary: bool = True) -> None:
        self.base_layer.set_temporary(temporary)
        if bool(getattr(self.base_layer, "always_use_original", False)):
            self.temporary = False
        else:
            self.temporary = bool(temporary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_original = (
            bool(getattr(self.base_layer, "always_use_original", False))
            or not bool(self.temporary)
        )
        if use_original:
            return self.base_layer(x)

        base_out = self.base_layer(x)
        x_sub = x
        if self.compressed_input_indices is not None:
            x_sub = x.index_select(-1, self.compressed_input_indices)

        carrier = self.compressed_subspace_adapter_linear
        ref_weight = carrier.base_layer.weight if is_peft_lora_linear(carrier) else carrier.weight
        if x_sub.device != ref_weight.device:
            raise RuntimeError(
                f"subspace carrier/input device mismatch: input={x_sub.device}, carrier={ref_weight.device}."
            )
        delta_sub = carrier(x_sub.to(dtype=ref_weight.dtype)).to(dtype=base_out.dtype)

        if self.compressed_output_indices is None:
            delta = delta_sub
        else:
            delta = base_out.new_zeros(base_out.shape)
            delta = delta.index_copy(-1, self.compressed_output_indices, delta_sub)
        return base_out + delta


def _subspace_proxy_root(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if not isinstance(base_model, nn.Module):
            raise TypeError(f"get_base_model() returned non-module: {type(base_model)}")
        return base_model
    return model


def iter_named_compressed_subspace_peft_proxies(
    model: nn.Module,
) -> Iterator[Tuple[str, CompressedSubspacePeftProxy]]:
    root = _subspace_proxy_root(model)
    skip_prefixes: List[str] = []
    for name, module in root.named_modules():
        if any(name == p or name.startswith(f"{p}.") for p in skip_prefixes):
            continue
        if not isinstance(module, CompressedSubspacePeftProxy):
            continue
        skip_prefixes.extend(
            (
                f"{name}.base_layer",
                f"{name}.{CompressedSubspacePeftProxy.CARRIER_NAME}",
            )
        )
        yield str(name), module


def _select_subspace_proxy_refs(
    model: nn.Module,
    module_names: Sequence[str],
) -> List[Tuple[str, CompressedSubspacePeftProxy]]:
    requested = [str(name) for name in module_names]
    if len(requested) != len(set(requested)):
        raise ValueError("module_names contains duplicates.")
    refs = dict(iter_named_compressed_subspace_peft_proxies(model))
    missing = [name for name in requested if name not in refs]
    if missing:
        raise RuntimeError(f"Missing CompressedSubspacePeftProxy modules: {missing}")
    return [(name, refs[name]) for name in requested]


def _assert_carrier_ready_for_injection(
    module_name: str,
    proxy: CompressedSubspacePeftProxy,
) -> PeftZeroLinearCarrier:
    carrier = getattr(proxy, CompressedSubspacePeftProxy.CARRIER_NAME, None)
    if is_peft_lora_linear(carrier) or is_peft_adalora_linear(carrier):
        raise RuntimeError(
            f"{module_name}: compressed subspace carrier already has a PEFT adapter; "
            "refusing duplicate injection."
        )
    if not isinstance(carrier, PeftZeroLinearCarrier):
        raise TypeError(
            f"{module_name}: expected PeftZeroLinearCarrier before injection, got {type(carrier)}."
        )
    if int(carrier.weight.numel()) != 1:
        raise RuntimeError(
            f"{module_name}: carrier sentinel weight must have numel==1, got {int(carrier.weight.numel())}."
        )
    if not torch.equal(carrier.weight.detach().cpu().reshape(-1), torch.zeros(1, dtype=carrier.weight.dtype)):
        raise RuntimeError(f"{module_name}: carrier sentinel weight must be identically zero.")
    if bool(carrier.weight.requires_grad):
        raise RuntimeError(f"{module_name}: carrier sentinel weight must have requires_grad=False.")
    return carrier


def _assert_carrier_injected(
    module_name: str,
    proxy: CompressedSubspacePeftProxy,
) -> PeftLoraLinear:
    carrier = getattr(proxy, CompressedSubspacePeftProxy.CARRIER_NAME, None)
    if not is_peft_lora_linear(carrier):
        raise TypeError(
            f"{module_name}: expected PEFT plain LoRA Linear after injection, got {type(carrier)}."
        )
    if is_peft_adalora_linear(carrier):
        raise TypeError(f"{module_name}: AdaLoRA carrier is not supported for subspace LoRA.")
    base_layer = carrier.base_layer
    if not isinstance(base_layer, PeftZeroLinearCarrier):
        raise TypeError(
            f"{module_name}: PEFT base_layer must remain PeftZeroLinearCarrier, got {type(base_layer)}."
        )
    if int(base_layer.weight.numel()) != 1:
        raise RuntimeError(
            f"{module_name}: injected carrier sentinel numel must stay 1, got {int(base_layer.weight.numel())}."
        )
    if not torch.equal(
        base_layer.weight.detach().cpu().reshape(-1),
        torch.zeros(1, dtype=base_layer.weight.dtype),
    ):
        raise RuntimeError(f"{module_name}: injected carrier sentinel weight must stay zero.")
    if bool(base_layer.weight.requires_grad):
        raise RuntimeError(f"{module_name}: injected carrier sentinel must keep requires_grad=False.")

    adapter_name = _get_default_adapter_name(carrier)
    if _adapter_uses_dora(carrier, adapter_name):
        raise ValueError(f"{module_name}: DoRA is not supported for compressed subspace LoRA.")
    lora_a = carrier.lora_A[adapter_name].weight
    lora_b = carrier.lora_B[adapter_name].weight
    if int(lora_a.shape[1]) != int(proxy.compressed_in_features):
        raise RuntimeError(
            f"{module_name}: lora_A input dim {int(lora_a.shape[1])} != "
            f"compressed_in_features {int(proxy.compressed_in_features)}."
        )
    if int(lora_b.shape[0]) != int(proxy.compressed_out_features):
        raise RuntimeError(
            f"{module_name}: lora_B output dim {int(lora_b.shape[0])} != "
            f"compressed_out_features {int(proxy.compressed_out_features)}."
        )
    return carrier


def inject_compressed_subspace_peft_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    proxy_refs = list(iter_named_compressed_subspace_peft_proxies(model))
    if not proxy_refs:
        return 0
    for module_name, proxy in proxy_refs:
        _assert_carrier_ready_for_injection(module_name, proxy)

    inject_adapter_in_model(
        LoraConfig(
            task_type=None,
            r=int(rank),
            lora_alpha=float(alpha),
            lora_dropout=float(dropout),
            target_modules=[CompressedSubspacePeftProxy.CARRIER_NAME],
            bias="none",
            inference_mode=False,
            init_lora_weights=True,
        ),
        model,
    )

    injected = 0
    for module_name, proxy in proxy_refs:
        _assert_carrier_injected(module_name, proxy)
        injected += 1
    if int(injected) != int(len(proxy_refs)):
        raise RuntimeError(
            f"Partial subspace PEFT injection: injected={injected} expected={len(proxy_refs)}."
        )
    return int(injected)


def wrap_vae_linears_with_compressed_subspace_peft_proxy(
    model: nn.Module,
    targets: Sequence[Tuple[str, VAELinear]],
) -> List[str]:
    existing = list(iter_named_compressed_subspace_peft_proxies(model))
    if existing:
        raise RuntimeError(
            "Refusing to wrap while existing CompressedSubspacePeftProxy modules are present: "
            f"{[name for name, _ in existing]}"
        )
    names = [str(name) for name, _ in targets]
    if len(names) != len(set(names)):
        raise ValueError("wrap targets contain duplicate module names.")

    wrapped: List[str] = []
    for module_name, base_layer in targets:
        if not isinstance(base_layer, VAELinear):
            raise TypeError(f"{module_name}: expected bare VAELinear, got {type(base_layer)}.")
        if base_layer.has_low_rank_residual():
            raise RuntimeError(
                f"{module_name}: VAELinear still has low_rank_a/b; clear before subspace wrap "
                "to avoid double-counting."
            )
        proxy = CompressedSubspacePeftProxy(base_layer)
        proxy.train(base_layer.training)
        set_module_by_name(model, module_name, proxy)
        wrapped.append(str(module_name))
    return wrapped


@torch.no_grad()
def initialize_subspace_peft_lora_from_low_rank(
    model: nn.Module,
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    *,
    module_names: Sequence[str],
) -> int:
    proxy_refs = _select_subspace_proxy_refs(model, module_names)
    requested = {str(name) for name in module_names}
    payload_keys = {str(name) for name in payloads}
    if payload_keys != requested:
        raise RuntimeError(
            f"payload keys must exactly match module_names: "
            f"payload={sorted(payload_keys)} targets={sorted(requested)}"
        )

    initialized = 0
    for module_name, proxy in proxy_refs:
        carrier = getattr(proxy, CompressedSubspacePeftProxy.CARRIER_NAME, None)
        if is_peft_adalora_linear(carrier):
            raise ValueError(f"{module_name}: AdaLoRA init from low_rank_a/b is not supported.")
        if not is_peft_lora_linear(carrier):
            raise TypeError(f"{module_name}: expected PEFT LoRA Linear, got {type(carrier)}.")
        adapter_name = _get_default_adapter_name(carrier)
        if _adapter_uses_dora(carrier, adapter_name):
            raise ValueError(f"{module_name}: DoRA init from low_rank_a/b is not supported.")

        lora_a = carrier.lora_A[adapter_name].weight
        lora_b = carrier.lora_B[adapter_name].weight
        scaling = float(carrier.scaling[adapter_name])
        if scaling == 0.0:
            raise ValueError(f"{module_name}: LoRA scaling is 0; cannot restore from low_rank_a/b.")

        low_rank_a, low_rank_b = payloads[str(module_name)]
        proxy.base_layer._validate_low_rank_payload_tensors(
            low_rank_a,
            low_rank_b,
            scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        if tuple(lora_a.shape) != tuple(low_rank_b.shape):
            raise RuntimeError(
                f"{module_name}: lora_A shape {tuple(lora_a.shape)} != low_rank_b {tuple(low_rank_b.shape)}."
            )
        if tuple(lora_b.shape) != tuple(low_rank_a.shape):
            raise RuntimeError(
                f"{module_name}: lora_B shape {tuple(lora_b.shape)} != low_rank_a {tuple(low_rank_a.shape)}."
            )
        lora_a.data.copy_(low_rank_b.to(device=lora_a.device, dtype=lora_a.dtype))
        restored_b = low_rank_a.to(device=lora_b.device, dtype=torch.float32) / float(scaling)
        lora_b.data.copy_(restored_b.to(dtype=lora_b.dtype))
        initialized += 1

    if int(initialized) != int(len(module_names)):
        raise RuntimeError(
            f"LoRA init from low_rank mismatch: initialized={initialized} expected={len(module_names)}."
        )
    return int(initialized)


@torch.no_grad()
def extract_subspace_peft_low_rank_payloads(
    model: nn.Module,
    *,
    module_names: Sequence[str],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    proxy_refs = _select_subspace_proxy_refs(model, module_names)
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for module_name, proxy in proxy_refs:
        carrier = getattr(proxy, CompressedSubspacePeftProxy.CARRIER_NAME, None)
        if is_peft_adalora_linear(carrier):
            raise ValueError(f"{module_name}: AdaLoRA export to VAELinear low_rank_a/b is not supported.")
        if not is_peft_lora_linear(carrier):
            raise TypeError(f"{module_name}: expected PEFT LoRA Linear, got {type(carrier)}.")
        adapter_name = _get_default_adapter_name(carrier)
        if _adapter_uses_dora(carrier, adapter_name):
            raise ValueError(f"{module_name}: DoRA export to VAELinear low_rank_a/b is not supported.")

        lora_a = carrier.lora_A[adapter_name].weight
        lora_b = carrier.lora_B[adapter_name].weight
        scaling = float(carrier.scaling[adapter_name])
        low_rank_b = lora_a.detach().to("cpu").clone().contiguous()
        low_rank_a = (
            (lora_b.detach().to(device="cpu", dtype=torch.float32) * float(scaling))
            .to(dtype=lora_b.dtype)
            .clone()
            .contiguous()
        )
        proxy.base_layer._validate_low_rank_payload_tensors(
            low_rank_a,
            low_rank_b,
            scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        payloads[str(module_name)] = (low_rank_a, low_rank_b)

    requested = {str(name) for name in module_names}
    if set(payloads) != requested:
        raise RuntimeError(
            f"extracted payload keys must exactly match module_names: "
            f"payload={sorted(payloads)} targets={sorted(requested)}"
        )
    return payloads


def _resolve_base_float_device_dtype(base_layer: VAELinear) -> tuple[torch.device, torch.dtype]:
    return _resolve_proxy_device_dtype(base_layer)


@torch.no_grad()
def export_compressed_subspace_peft_lora_to_vae_low_rank(
    model: nn.Module,
    *,
    module_names: Sequence[str],
    allow_overwrite: bool,
) -> int:
    root = _subspace_proxy_root(model)
    proxy_refs = _select_subspace_proxy_refs(model, module_names)
    payloads = extract_subspace_peft_low_rank_payloads(
        model,
        module_names=module_names,
    )

    exported = 0
    for module_name, proxy in proxy_refs:
        base_layer = proxy.base_layer
        low_rank_a, low_rank_b = payloads[str(module_name)]
        existing_a = getattr(base_layer, "low_rank_a", None)
        existing_b = getattr(base_layer, "low_rank_b", None)
        if (existing_a is not None or existing_b is not None) and not bool(allow_overwrite):
            raise ValueError(
                f"{module_name}: VAELinear already has low_rank_a/b; refusing to overwrite."
            )

        base_layer._validate_low_rank_payload_tensors(
            low_rank_a,
            low_rank_b,
            scope=LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        )
        device, dtype = _resolve_base_float_device_dtype(base_layer)
        a_param = nn.Parameter(
            low_rank_a.detach().to(device=device, dtype=dtype).contiguous(),
            requires_grad=False,
        )
        b_param = nn.Parameter(
            low_rank_b.detach().to(device=device, dtype=dtype).contiguous(),
            requires_grad=False,
        )
        base_layer.low_rank_scope = normalize_low_rank_scope(LOW_RANK_SCOPE_COMPRESSED_SUBSPACE)
        base_layer.register_parameter("low_rank_a", a_param)
        base_layer.register_parameter("low_rank_b", b_param)
        base_layer.clear_decoded_weight_cache()
        base_layer.train(proxy.training)
        set_module_by_name(root, module_name, base_layer)
        exported += 1

    remaining = {
        name
        for name, _ in iter_named_compressed_subspace_peft_proxies(model)
        if name in {str(n) for n in module_names}
    }
    if remaining:
        raise RuntimeError(
            f"CompressedSubspacePeftProxy modules still present after export: {sorted(remaining)}"
        )
    return int(exported)


def unwrap_compressed_subspace_peft_proxies(
    model: nn.Module,
    *,
    module_names: Optional[Sequence[str]] = None,
) -> int:
    root = _subspace_proxy_root(model)
    if module_names is None:
        proxy_refs = list(iter_named_compressed_subspace_peft_proxies(model))
    else:
        proxy_refs = _select_subspace_proxy_refs(model, module_names)

    restored = 0
    for module_name, proxy in proxy_refs:
        base_layer = proxy.base_layer
        base_layer.clear_decoded_weight_cache()
        base_layer.train(proxy.training)
        set_module_by_name(root, module_name, base_layer)
        restored += 1
    return int(restored)
