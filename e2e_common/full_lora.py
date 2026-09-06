"""Shared full-space compressed LoRA carrier/proxy for CAT and E2E.

Plain full LoRA on VAELinear keeps the compressed module intact and applies PEFT
delta through an O(1) zero-base carrier. Do not materialize a huge dense base as
the only path.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from torch import nn

from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear


def is_peft_lora_linear(module: nn.Module) -> bool:
    return isinstance(module, PeftLoraLinear)


def _get_default_adapter_name(module: nn.Module) -> str:
    active = getattr(module, "active_adapter", None)
    if isinstance(active, str) and active:
        return active
    active_adapters = getattr(module, "active_adapters", None)
    if isinstance(active_adapters, (list, tuple)) and len(active_adapters) == 1:
        return str(active_adapters[0])
    if "default" in getattr(module, "lora_A", {}):
        return "default"
    raise RuntimeError("Expected exactly one active plain LoRA adapter.")


def _resolve_float_device_dtype(base_layer: VAELinear) -> tuple[torch.device, torch.dtype]:
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

        # Standard 1x1 nn.Linear for valid Module/Linear state; O(1) storage only.
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
    rank_explicit: bool = False,
    include_lm_head: bool = False,
    dense_target_modules: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Wrap selected VAELinears and inject one PEFT adapter.

    ``include_lm_head`` and ``dense_target_modules`` share the same adapter
    (one-adapter union). Never pass vague suffixes like ``q_proj``.
    """
    names = [str(name) for name, _ in selected_modules]
    dense_names = [str(name) for name in (dense_target_modules or ())]
    if len(names) != len(set(names)):
        raise ValueError("selected_modules contain duplicate names.")
    if len(dense_names) != len(set(dense_names)):
        raise ValueError("dense_target_modules contain duplicate names.")
    overlap = set(names) & set(dense_names)
    if overlap:
        raise ValueError(f"compressed/dense target overlap: {sorted(overlap)}")
    if include_lm_head and "lm_head" in dense_names:
        raise ValueError("lm_head already listed in dense_target_modules; do not also set include_lm_head.")
    if include_lm_head:
        lm_head = getattr(model, "lm_head", None)
        if not isinstance(lm_head, nn.Linear):
            raise TypeError(
                f"include_lm_head=true requires model.lm_head to be nn.Linear, got {type(lm_head)}."
            )

    if initial_low_rank_payloads is not None and not set(initial_low_rank_payloads).issubset(set(names)):
        raise RuntimeError("Full proxy low-rank payload keys must be a subset of selected targets.")
    rank_pattern: Dict[str, int] = {}
    if initial_low_rank_payloads is not None:
        for name, module in selected_modules:
            if module.has_low_rank_residual() and name not in initial_low_rank_payloads:
                raise RuntimeError(
                    f"{name}: existing full low-rank residual must be supplied for PEFT initialization."
                )
            if name not in initial_low_rank_payloads:
                continue
            module._validate_low_rank_payload_tensors(
                initial_low_rank_payloads[name][0],
                initial_low_rank_payloads[name][1],
            )
            payload_rank = int(initial_low_rank_payloads[name][0].shape[1])
            if bool(rank_explicit) and payload_rank != int(rank):
                raise ValueError(
                    f"lora_rank explicit value {int(rank)} conflicts with existing "
                    f"payload rank {payload_rank} for target {name!r}."
                )
            rank_pattern[f"{name}.{FullCompressedPeftProxy.CARRIER_NAME}"] = payload_rank
            module.register_parameter("low_rank_a", None)
            module.register_parameter("low_rank_b", None)
            module.clear_decoded_weight_cache()

    if selected_modules:
        wrap_full_compressed_peft_proxies(model, selected_modules)

    target_modules: List[str] = []
    if selected_modules:
        target_modules.append(FullCompressedPeftProxy.CARRIER_NAME)
    target_modules.extend(dense_names)
    if include_lm_head:
        target_modules.append("lm_head")
    if not target_modules:
        raise ValueError("build_full_compressed_peft_model requires at least one LoRA target.")

    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type=None,
            inference_mode=False,
            r=int(rank),
            target_modules=target_modules,
            lora_alpha=float(alpha),
            lora_dropout=float(dropout),
            rank_pattern=rank_pattern,
            bias="none",
            init_lora_weights=True,
        ),
    )
    assert_exact_adapter_target_set(
        peft_model,
        compressed_proxy_names=names,
        dense_module_names=dense_names,
        include_lm_head=include_lm_head,
    )
    if initial_low_rank_payloads is not None:
        init_names = [name for name in names if name in initial_low_rank_payloads]
        initialize_full_proxy_lora_from_low_rank(
            peft_model,
            initial_low_rank_payloads,
            module_names=init_names,
        )
    return peft_model


def collect_exact_peft_lora_config(
    model: nn.Module,
    *,
    default_rank: int,
    alpha: float,
    dropout: float,
) -> Optional[Dict[str, object]]:
    """Describe the exact live one-adapter topology for training-step resume."""
    target_ranks: Dict[str, int] = {}
    for peft_name, layer in iter_named_peft_lora_layers(model):
        logical_name = _logical_adapter_target_name(peft_name)
        adapter = _get_default_adapter_name(layer)
        target_ranks[logical_name] = int(layer.lora_A[adapter].weight.shape[0])
    if not target_ranks:
        return None
    return {
        "rank": int(default_rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "rank_pattern": {
            name: int(target_ranks[name])
            for name in sorted(target_ranks)
            if int(target_ranks[name]) != int(default_rank)
        },
        "target_modules": sorted(target_ranks),
    }


def _strip_peft_prefix(name: str) -> str:
    text = str(name)
    for prefix in ("base_model.model.", "base_model.", "model."):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _logical_adapter_target_name(peft_module_name: str) -> str:
    text = _strip_peft_prefix(peft_module_name)
    carrier = FullCompressedPeftProxy.CARRIER_NAME
    if text == carrier:
        raise RuntimeError(f"Unexpected top-level carrier target name: {peft_module_name!r}")
    suffix = f".{carrier}"
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def iter_named_peft_lora_layers(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yield raw PEFT module names exactly once.

    Logical-name normalization belongs to ``_logical_adapter_target_name``.  Do
    not strip here as well: nested targets such as ``model.layers.0.q_proj``
    would otherwise lose their leading ``model.`` on the second normalization.
    """
    for name, module in model.named_modules():
        if is_peft_lora_linear(module):
            yield str(name), module


def collect_logical_adapter_target_names(model: nn.Module) -> set[str]:
    return {_logical_adapter_target_name(name) for name, _module in iter_named_peft_lora_layers(model)}


def assert_exact_adapter_target_set(
    model: nn.Module,
    *,
    compressed_proxy_names: Sequence[str],
    dense_module_names: Sequence[str] = (),
    include_lm_head: bool = False,
) -> None:
    expected = {str(name) for name in compressed_proxy_names}
    expected.update(str(name) for name in dense_module_names)
    if include_lm_head:
        expected.add("lm_head")
    actual = collect_logical_adapter_target_names(model)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            "PEFT adapter target-set mismatch after injection: "
            f"missing={missing} extra={extra} expected={sorted(expected)} actual={sorted(actual)}."
        )


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
        if not is_peft_lora_linear(carrier):
            raise TypeError(f"{module_name}: expected plain PEFT LoRA carrier, got {type(carrier)}.")
        adapter = _get_default_adapter_name(carrier)
        low_rank_a, low_rank_b = payloads[module_name]
        proxy.base_layer._validate_low_rank_payload_tensors(low_rank_a, low_rank_b)
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
        if not is_peft_lora_linear(carrier):
            raise TypeError(f"{module_name}: expected plain PEFT LoRA carrier, got {type(carrier)}.")
        adapter = _get_default_adapter_name(carrier)
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
        proxy.base_layer._validate_low_rank_payload_tensors(low_rank_a, low_rank_b)
        payloads[module_name] = (low_rank_a, low_rank_b)
    return payloads


def unwrap_full_compressed_peft_proxies(
    model: nn.Module,
    *,
    module_names: Optional[Sequence[str]] = None,
) -> int:
    root = _proxy_root(model)
    if module_names is None:
        proxy_refs = list(iter_named_full_compressed_peft_proxies(model))
    else:
        proxy_refs = _select_refs(model, module_names)
    restored = 0
    for module_name, proxy in proxy_refs:
        base_layer = proxy.base_layer
        base_layer.clear_decoded_weight_cache()
        base_layer.train(proxy.training)
        set_module_by_name(root, module_name, base_layer)
        restored += 1
    return int(restored)


def _resolve_peft_delta_weight(lora_layer: nn.Module, adapter_name: str) -> torch.Tensor:
    getter = getattr(lora_layer, "get_delta_weight", None)
    if callable(getter):
        return getter(adapter_name)
    lora_a = lora_layer.lora_A[adapter_name].weight
    lora_b = lora_layer.lora_B[adapter_name].weight
    scaling = float(lora_layer.scaling[adapter_name])
    return (lora_b @ lora_a) * scaling


@torch.no_grad()
def _merge_dense_peft_lora_into_base(lora_layer: nn.Module) -> nn.Module:
    if not is_peft_lora_linear(lora_layer):
        raise TypeError(f"expected plain PEFT LoRA layer, got {type(lora_layer)}.")
    adapter = _get_default_adapter_name(lora_layer)
    base = lora_layer.get_base_layer() if hasattr(lora_layer, "get_base_layer") else lora_layer.base_layer
    if not isinstance(base, nn.Linear):
        raise TypeError(f"dense LoRA finalize expects nn.Linear base, got {type(base)}.")
    delta = _resolve_peft_delta_weight(lora_layer, adapter)
    base.weight.add_(delta.to(device=base.weight.device, dtype=base.weight.dtype))
    return base


@torch.no_grad()
def _write_payload_to_vae_linear(
    module: VAELinear,
    low_rank_a: torch.Tensor,
    low_rank_b: torch.Tensor,
) -> None:
    module._validate_low_rank_payload_tensors(low_rank_a, low_rank_b)
    device, _base_dtype = _resolve_float_device_dtype(module)
    current_a = getattr(module, "low_rank_a", None)
    current_b = getattr(module, "low_rank_b", None)
    replace_payload = (
        current_a is None
        or current_b is None
        or current_a.dtype != low_rank_a.dtype
        or current_b.dtype != low_rank_b.dtype
    )
    if replace_payload:
        module.low_rank_a = nn.Parameter(
            low_rank_a.detach().to(device=device).contiguous(),
            requires_grad=False,
        )
        module.low_rank_b = nn.Parameter(
            low_rank_b.detach().to(device=device).contiguous(),
            requires_grad=False,
        )
    else:
        module.low_rank_a.data.copy_(low_rank_a.to(device=module.low_rank_a.device, dtype=module.low_rank_a.dtype))
        module.low_rank_b.data.copy_(low_rank_b.to(device=module.low_rank_b.device, dtype=module.low_rank_b.dtype))
    module.clear_decoded_weight_cache()


def _vae_linear_base_fingerprint(module: VAELinear) -> Dict[str, torch.Tensor]:
    """Capture compressed base state that finalize must leave unchanged."""
    payload: Dict[str, torch.Tensor] = {}
    vq = getattr(module, "vq_weight", None)
    if isinstance(vq, torch.Tensor):
        payload["vq_weight"] = vq.detach().to("cpu").clone()
    decoder = getattr(module, "decoder", None)
    if isinstance(decoder, nn.Module):
        for name, tensor in decoder.state_dict().items():
            if torch.is_tensor(tensor):
                payload[f"decoder.{name}"] = tensor.detach().to("cpu").clone()
    return payload


def _assert_vae_linear_base_unchanged(module: VAELinear, fingerprint: Dict[str, torch.Tensor]) -> None:
    current = _vae_linear_base_fingerprint(module)
    if set(current) != set(fingerprint):
        raise RuntimeError(
            "VAELinear base fingerprint key mismatch: "
            f"missing={sorted(set(fingerprint) - set(current))} "
            f"extra={sorted(set(current) - set(fingerprint))}."
        )
    for key, expected in fingerprint.items():
        if not torch.equal(current[key], expected):
            raise RuntimeError(f"VAELinear base tensor changed during finalize: {key}")


@torch.no_grad()
def finalize_model_level_lora(
    model: nn.Module,
    *,
    compressed_proxy_names: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Finalize one model-level PEFT adapter without harming compressed carriers.

    Path A (any FullCompressedPeftProxy): write PEFT A/B back to VAELinear low-rank,
    selectively merge ordinary dense targets (e.g. lm_head), unwrap proxies, never
    call global merge_and_unload().

    Path B (no compressed proxy): standard PEFT merge_and_unload().
    """
    proxy_refs = list(iter_named_full_compressed_peft_proxies(model))
    if compressed_proxy_names is not None:
        wanted = {str(name) for name in compressed_proxy_names}
        proxy_refs = [(name, proxy) for name, proxy in proxy_refs if name in wanted]
        missing = sorted(wanted - {name for name, _ in proxy_refs})
        if missing:
            raise RuntimeError(f"finalize missing compressed proxies: {missing}")

    if not proxy_refs:
        merge_and_unload = getattr(model, "merge_and_unload", None)
        if not callable(merge_and_unload):
            raise TypeError("Path-B finalize requires a PEFT model with merge_and_unload().")
        finalized = merge_and_unload()
        remaining = collect_logical_adapter_target_names(finalized)
        if remaining:
            raise RuntimeError(f"PEFT layers still present after merge_and_unload: {sorted(remaining)}")
        return finalized

    # Path A: never call global merge_and_unload.
    root = _proxy_root(model)
    compressed_names = [name for name, _ in proxy_refs]
    payloads = extract_full_proxy_low_rank_payloads(model, module_names=compressed_names)

    # Snapshot base compressed state before writing low-rank (must remain unchanged).
    base_fingerprints = {
        name: _vae_linear_base_fingerprint(proxy.base_layer) for name, proxy in proxy_refs
    }

    for name, proxy in proxy_refs:
        low_rank_a, low_rank_b = payloads[name]
        _write_payload_to_vae_linear(proxy.base_layer, low_rank_a, low_rank_b)

    # Selectively merge ordinary dense PEFT targets (anything that is not a carrier).
    dense_lora_layers = []
    for peft_name, lora_layer in iter_named_peft_lora_layers(model):
        logical = _logical_adapter_target_name(peft_name)
        if logical in set(compressed_names):
            continue
        dense_lora_layers.append((logical, peft_name, lora_layer))

    for logical, peft_name, lora_layer in dense_lora_layers:
        merged_base = _merge_dense_peft_lora_into_base(lora_layer)
        set_module_by_name(root, logical, merged_base)

    unwrap_full_compressed_peft_proxies(model, module_names=compressed_names)

    # Drop PEFT wrapper if present, keeping the underlying module tree.
    get_base = getattr(model, "get_base_model", None)
    finalized = get_base() if callable(get_base) else model
    if finalized is model and hasattr(model, "base_model"):
        # PeftModel.base_model.model is the original root in common PEFT layouts.
        base_model = getattr(model, "base_model", None)
        inner = getattr(base_model, "model", None) if base_model is not None else None
        if isinstance(inner, nn.Module):
            finalized = inner

    # Assert compressed targets restored and base compressed state unchanged.
    for name, fingerprint in base_fingerprints.items():
        module = dict(finalized.named_modules()).get(name)
        if module is None:
            module = dict(_proxy_root(finalized).named_modules()).get(name)
        if not isinstance(module, VAELinear):
            raise RuntimeError(f"{name}: expected VAELinear after finalize, got {type(module)}.")
        if list(iter_named_full_compressed_peft_proxies(finalized)):
            raise RuntimeError("FullCompressedPeftProxy still present after finalize.")
        _assert_vae_linear_base_unchanged(module, fingerprint)
        if getattr(module, "low_rank_a", None) is None or getattr(module, "low_rank_b", None) is None:
            raise RuntimeError(f"{name}: missing low_rank_a/b after finalize.")
    remaining_lora = [
        name
        for name, module in finalized.named_modules()
        if is_peft_lora_linear(module)
    ]
    if remaining_lora:
        raise RuntimeError(f"PEFT LoRA layers still present after path-A finalize: {remaining_lora}")
    return finalized
