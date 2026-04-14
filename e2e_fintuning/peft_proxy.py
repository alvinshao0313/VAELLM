import math
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from peft import AdaLoraConfig, LoraConfig
from peft.mapping import inject_adapter_in_model
from peft.tuners.adalora.layer import RankAllocator, SVDLinear as PeftAdaLoraLinear
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from torch import nn

from litebsq.misc import set_module_by_name
from litebsq.vae_linear_prewarm import (
    NamedVAELinearDecodeTarget,
    decode_named_vae_linear_weights,
    resolve_grouped_decode_compute_device,
)
from litebsq.vae_linear import VAELinear


_DEFAULT_ADAPTER_NAME = "default"
_DEFAULT_CATEGORY_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_VALID_VAE_LORA_VARIANTS = {"plain", "rslora", "dora", "adalora"}
_ADALORA_RANKALLOCATOR_ATTR = "_peft_proxy_adalora_rankallocator"
_ADALORA_RUNTIME_PREFIX = "__peft_proxy_adalora_runtime__."


@dataclass(frozen=True)
class _ValidatedResidualSVDTarget:
    name: str
    category: str
    adapter_name: str
    peft_linear: PeftLoraLinear
    weight_A: torch.Tensor
    weight_B: torch.Tensor
    scaling: float
    expected_shape: Tuple[int, int]
    rank: int
    use_dora: bool
    base_weight: torch.Tensor
    magnitude_vector: Optional[torch.Tensor]


@dataclass(frozen=True)
class _TeacherResidualSVDSource:
    target: _ValidatedResidualSVDTarget
    teacher_weight: torch.Tensor
    decoded_weight: torch.Tensor


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


def is_peft_adalora_linear(module: nn.Module) -> bool:
    return isinstance(module, PeftAdaLoraLinear)


def is_peft_proxy_adapter_linear(module: nn.Module) -> bool:
    return is_peft_lora_linear(module) or is_peft_adalora_linear(module)


class PeftVAELinearProxy(nn.Module):
    def __init__(self, base_layer: VAELinear):
        if not isinstance(base_layer, VAELinear):
            raise TypeError(f"PeftVAELinearProxy expects VAELinear base_layer, got {type(base_layer)}")
        super().__init__()
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.temporary = bool(getattr(base_layer, "temporary", True))
        self.per_decoded_linear = self._build_placeholder_decoded_linear()
        self._dense_base_materialized = False

        self.base_layer.cache_decoded_weight = False
        self.base_layer.clear_decoded_weight_cache()
        setattr(self.base_layer, "_skip_global_cache_prewarm", True)

    @torch.no_grad()
    def _build_placeholder_decoded_linear(self) -> nn.Linear:
        target_dtype = _resolve_proxy_dtype(self.base_layer)
        target_device = next(self.base_layer.parameters(), None)
        target_device = torch.device("cpu") if target_device is None else target_device.device
        bias = self.base_layer.bias
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=bias is not None,
            device=target_device,
            dtype=target_dtype,
        )
        linear.weight.requires_grad = False
        linear.weight.zero_()
        if bias is not None:
            linear.bias.requires_grad = False
            linear.bias.zero_()
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
        if not bool(self._dense_base_materialized):
            raise RuntimeError("PeftVAELinearProxy dense base has not been materialized.")
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


def _sorted_named_peft_vae_proxies(model: nn.Module) -> List[Tuple[str, PeftVAELinearProxy]]:
    return sorted(iter_named_peft_vae_proxies(model), key=lambda item: str(item[0]))


def _resolve_proxy_base_linear(module_name: str, proxy: PeftVAELinearProxy) -> nn.Linear:
    decoded_linear = proxy.per_decoded_linear
    if is_peft_proxy_adapter_linear(decoded_linear):
        decoded_linear = decoded_linear.get_base_layer()
    if not isinstance(decoded_linear, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear under '{module_name}.per_decoded_linear', got {type(decoded_linear)}"
        )
    return decoded_linear


def _has_unmaterialized_proxy_refs(proxy_refs: List[Tuple[str, PeftVAELinearProxy]]) -> bool:
    for _name, proxy in proxy_refs:
        if not bool(getattr(proxy, "_dense_base_materialized", False)):
            return True
    return False


def _normalize_variant(variant: str) -> str:
    norm = str(variant or "").strip().lower()
    if norm not in _VALID_VAE_LORA_VARIANTS:
        raise ValueError(f"Unsupported VAE LoRA variant: {variant}")
    return norm


def _get_default_adapter_name(module: nn.Module) -> str:
    lora_A = getattr(module, "lora_A", None)
    if lora_A is None:
        raise TypeError(f"Module does not expose lora_A: {type(module)}")
    if _DEFAULT_ADAPTER_NAME in lora_A:
        return _DEFAULT_ADAPTER_NAME
    if len(lora_A) != 1:
        raise ValueError("Only single-adapter plain PEFT adapters are supported for VAELinear proxy export.")
    return next(iter(lora_A.keys()))


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


def _resolve_lora_variant_from_module(module: PeftLoraLinear, adapter_name: str) -> str:
    if bool(module.use_dora.get(adapter_name, False)):
        return "dora"
    if _resolve_use_rslora(module, adapter_name):
        return "rslora"
    return "plain"


def _resolve_proxy_category(module_name: str) -> str:
    return str(module_name).rsplit(".", 1)[-1]


def _category_sort_key(category: str) -> Tuple[int, str]:
    cat = str(category)
    try:
        return (_DEFAULT_CATEGORY_ORDER.index(cat), cat)
    except ValueError:
        return (len(_DEFAULT_CATEGORY_ORDER), cat)


def _resolve_model_peft_config(model: nn.Module):
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict):
        return None
    if _DEFAULT_ADAPTER_NAME not in peft_config:
        if len(peft_config) != 1:
            raise ValueError("Only single-adapter PEFT proxy checkpoints are supported.")
        return next(iter(peft_config.values()))
    return peft_config[_DEFAULT_ADAPTER_NAME]


def _validate_existing_lora_proxy_linear(
    module_name: str,
    peft_linear: PeftLoraLinear,
    *,
    variant: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    adapter_name = _get_default_adapter_name(peft_linear)
    actual_variant = _resolve_lora_variant_from_module(peft_linear, adapter_name)
    actual_rank = int(peft_linear.r[adapter_name])
    actual_alpha = float(peft_linear.lora_alpha[adapter_name])
    actual_dropout = float(_dropout_p(peft_linear.lora_dropout[adapter_name]))
    if actual_variant != str(variant):
        raise ValueError(
            f"Existing PEFT proxy adapter at '{module_name}' has variant={actual_variant}, "
            f"but requested {variant}."
        )
    if actual_rank != int(rank) or actual_alpha != float(alpha) or actual_dropout != float(dropout):
        raise ValueError(
            f"Existing PEFT proxy adapter at '{module_name}' has config "
            f"(rank={actual_rank}, alpha={actual_alpha}, dropout={actual_dropout}) "
            f"but requested (rank={rank}, alpha={alpha}, dropout={dropout})."
        )


def _validate_existing_adalora_proxy_linear(
    module_name: str,
    peft_linear: PeftAdaLoraLinear,
    *,
    alpha: float,
    dropout: float,
    init_r: int,
) -> None:
    adapter_name = _get_default_adapter_name(peft_linear)
    actual_rank = int(peft_linear.r[adapter_name])
    actual_alpha = float(peft_linear.lora_alpha[adapter_name])
    actual_dropout = float(_dropout_p(peft_linear.lora_dropout[adapter_name]))
    if actual_rank != int(init_r) or actual_alpha != float(alpha) or actual_dropout != float(dropout):
        raise ValueError(
            f"Existing PEFT AdaLoRA proxy at '{module_name}' has config "
            f"(init_r={actual_rank}, alpha={actual_alpha}, dropout={actual_dropout}) "
            f"but requested (init_r={init_r}, alpha={alpha}, dropout={dropout})."
        )


def _validate_existing_adalora_root_config(
    model: nn.Module,
    *,
    target_r: int,
    init_r: int,
    tinit: int,
    tfinal: int,
    delta_t: int,
    beta1: float,
    beta2: float,
    orth_reg_weight: float,
    total_step: Optional[int],
) -> AdaLoraConfig:
    peft_config = _resolve_model_peft_config(model)
    if not isinstance(peft_config, AdaLoraConfig):
        raise ValueError("Existing PEFT proxy adapters are not AdaLoRA.")
    if int(peft_config.target_r) != int(target_r):
        raise ValueError(f"Existing AdaLoRA target_r={peft_config.target_r} does not match requested {target_r}.")
    if int(peft_config.init_r) != int(init_r):
        raise ValueError(f"Existing AdaLoRA init_r={peft_config.init_r} does not match requested {init_r}.")
    if int(peft_config.tinit) != int(tinit):
        raise ValueError(f"Existing AdaLoRA tinit={peft_config.tinit} does not match requested {tinit}.")
    if int(peft_config.tfinal) != int(tfinal):
        raise ValueError(f"Existing AdaLoRA tfinal={peft_config.tfinal} does not match requested {tfinal}.")
    if int(peft_config.deltaT) != int(delta_t):
        raise ValueError(f"Existing AdaLoRA deltaT={peft_config.deltaT} does not match requested {delta_t}.")
    if float(peft_config.beta1) != float(beta1):
        raise ValueError(f"Existing AdaLoRA beta1={peft_config.beta1} does not match requested {beta1}.")
    if float(peft_config.beta2) != float(beta2):
        raise ValueError(f"Existing AdaLoRA beta2={peft_config.beta2} does not match requested {beta2}.")
    if float(peft_config.orth_reg_weight) != float(orth_reg_weight):
        raise ValueError(
            f"Existing AdaLoRA orth_reg_weight={peft_config.orth_reg_weight} "
            f"does not match requested {orth_reg_weight}."
        )
    if total_step is not None and peft_config.total_step is not None and int(peft_config.total_step) != int(total_step):
        raise ValueError(
            f"Existing AdaLoRA total_step={peft_config.total_step} does not match requested {total_step}."
        )
    return peft_config


def _build_residual_svd_target(
    module_name: str,
    peft_linear: PeftLoraLinear,
) -> _ValidatedResidualSVDTarget:
    adapter_name = _get_default_adapter_name(peft_linear)
    base_layer = peft_linear.get_base_layer()
    weight_A = peft_linear.lora_A[adapter_name].weight
    weight_B = peft_linear.lora_B[adapter_name].weight
    use_dora = bool(peft_linear.use_dora.get(adapter_name, False))
    magnitude_vector = peft_linear.lora_magnitude_vector[adapter_name] if use_dora else None
    return _ValidatedResidualSVDTarget(
        name=str(module_name),
        category=_resolve_proxy_category(module_name),
        adapter_name=adapter_name,
        peft_linear=peft_linear,
        weight_A=weight_A,
        weight_B=weight_B,
        scaling=float(peft_linear.scaling[adapter_name]),
        expected_shape=tuple(base_layer.weight.shape),
        rank=int(weight_A.shape[0]),
        use_dora=use_dora,
        base_weight=base_layer.weight.detach(),
        magnitude_vector=magnitude_vector,
    )


def _initialize_residual_svd_targets_batched(
    targets: List[_ValidatedResidualSVDTarget],
    residuals: List[torch.Tensor],
    *,
    batch_device: torch.device,
) -> None:
    if len(targets) == 0:
        raise ValueError("targets cannot be empty for batched residual_svd init.")
    if len(targets) != len(residuals):
        raise ValueError(
            f"targets/residuals length mismatch: targets={len(targets)} residuals={len(residuals)}"
        )

    batch = torch.stack(
        [residual.detach().to(device=batch_device, dtype=torch.float32) for residual in residuals],
        dim=0,
    )
    scales = torch.tensor(
        [float(target.scaling) for target in targets],
        device=batch_device,
        dtype=torch.float32,
    ).view(len(targets), 1, 1)
    scaled_batch = batch / scales
    u, s, vh = torch.linalg.svd(scaled_batch, full_matrices=False)

    for idx, target in enumerate(targets):
        target.weight_A.zero_()
        target.weight_B.zero_()
        k = min(int(target.rank), int(s.shape[1]))
        if k > 0:
            sqrt_s = torch.sqrt(s[idx, :k])
            b_factor = u[idx, :, :k] * sqrt_s.unsqueeze(0)
            a_factor = sqrt_s.unsqueeze(1) * vh[idx, :k, :]
            target.weight_B[:, :k].copy_(b_factor.to(device=target.weight_B.device, dtype=target.weight_B.dtype))
            target.weight_A[:k, :].copy_(a_factor.to(device=target.weight_A.device, dtype=target.weight_A.dtype))

        if target.use_dora:
            if target.magnitude_vector is None:
                raise RuntimeError(f"DoRA target '{target.name}' is missing lora_magnitude_vector.")
            combined_weight = target.base_weight.detach().to(device=batch_device, dtype=torch.float32)
            combined_weight = combined_weight + target.peft_linear.get_delta_weight(target.adapter_name).to(
                device=batch_device,
                dtype=torch.float32,
            )
            magnitude = torch.linalg.norm(combined_weight, dim=1)
            target.magnitude_vector.copy_(
                magnitude.to(device=target.magnitude_vector.device, dtype=target.magnitude_vector.dtype)
            )

    del batch, scales, scaled_batch, u, s, vh
    if batch_device.type == "cuda":
        torch.cuda.empty_cache()


def _collect_teacher_residual_sources(
    model: nn.Module,
    teacher_model: nn.Module,
) -> List[_TeacherResidualSVDSource]:
    items: List[_TeacherResidualSVDSource] = []
    for name, proxy in _sorted_named_peft_vae_proxies(model):
        peft_linear = proxy.per_decoded_linear
        if not is_peft_lora_linear(peft_linear):
            raise ValueError(f"residual_svd only supports LoRA-family PEFT proxy modules, got {type(peft_linear)} at {name}.")
        target = _build_residual_svd_target(name, peft_linear)

        teacher_module = _get_module_by_name(teacher_model, name)
        teacher_weight = teacher_module.weight

        decoded_weight = peft_linear.get_base_layer().weight
        if tuple(teacher_weight.shape) != tuple(decoded_weight.shape):
            raise ValueError(
                f"Teacher/student weight shape mismatch at '{name}': "
                f"teacher={tuple(teacher_weight.shape)} student={tuple(decoded_weight.shape)}."
            )
        items.append(
            _TeacherResidualSVDSource(
                target=target,
                teacher_weight=teacher_weight,
                decoded_weight=decoded_weight,
            )
        )
    return items


def _group_residual_targets_by_category(
    items: List[_TeacherResidualSVDSource],
) -> List[Tuple[str, List[_TeacherResidualSVDSource]]]:
    grouped: Dict[str, List[_TeacherResidualSVDSource]] = {}
    ordered_categories: List[str] = []
    for item in sorted(items, key=lambda one: (_category_sort_key(one.target.category), one.target.name)):
        target = item.target
        category = str(target.category)
        if category not in grouped:
            grouped[category] = []
            ordered_categories.append(category)
        grouped[category].append(item)
    return [(category, grouped[category]) for category in ordered_categories]


def _enable_peft_proxy_adapters(model: nn.Module) -> None:
    for _name, proxy in iter_named_peft_vae_proxies(model):
        peft_linear = proxy.per_decoded_linear
        if not is_peft_proxy_adapter_linear(peft_linear):
            continue
        adapter_name = _get_default_adapter_name(peft_linear)
        peft_linear.enable_adapters(True)
        peft_linear.set_adapter(adapter_name)


def _zero_initialize_adalora_modules(model: nn.Module) -> None:
    with torch.no_grad():
        for name, proxy in iter_named_peft_vae_proxies(model):
            peft_linear = proxy.per_decoded_linear
            if not is_peft_adalora_linear(peft_linear):
                continue
            adapter_name = _get_default_adapter_name(peft_linear)
            peft_linear.lora_E[adapter_name].zero_()
            if torch.count_nonzero(peft_linear.get_delta_weight(adapter_name)).item() != 0:
                raise RuntimeError(f"Failed to zero-initialize AdaLoRA delta at '{name}'.")


def _ensure_peft_proxy_adalora_runtime(
    model: nn.Module,
    *,
    total_step: Optional[int],
) -> bool:
    peft_config = _resolve_model_peft_config(model)
    if not isinstance(peft_config, AdaLoraConfig):
        return False
    if total_step is not None:
        peft_config.total_step = int(total_step)
    rankallocator = getattr(model, _ADALORA_RANKALLOCATOR_ATTR, None)
    if rankallocator is None:
        rankallocator = RankAllocator(model, peft_config, _DEFAULT_ADAPTER_NAME)
        setattr(model, _ADALORA_RANKALLOCATOR_ATTR, rankallocator)
    elif total_step is not None:
        rankallocator.set_total_step(int(total_step))
    return True


def _adalora_runtime_key(group_name: str, name: str) -> str:
    return f"{_ADALORA_RUNTIME_PREFIX}{group_name}.{name}"


def inject_peft_proxy_adalora_runtime_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> int:
    peft_config = _resolve_model_peft_config(model)
    if not isinstance(peft_config, AdaLoraConfig):
        return 0
    if not _ensure_peft_proxy_adalora_runtime(model, total_step=peft_config.total_step):
        return 0
    rankallocator = getattr(model, _ADALORA_RANKALLOCATOR_ATTR, None)
    if rankallocator is None:
        raise RuntimeError("AdaLoRA rankallocator is missing.")

    added = 0
    for group_name in ("ipt", "exp_avg_ipt", "exp_avg_unc"):
        group = getattr(rankallocator, group_name, {})
        if not isinstance(group, dict):
            raise TypeError(f"AdaLoRA runtime field {group_name} must be a dict, got {type(group)}")
        for name, tensor in group.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"AdaLoRA runtime tensor {group_name}.{name} must be a tensor, got {type(tensor)}")
            state_dict[_adalora_runtime_key(group_name, str(name))] = tensor.detach().clone()
            added += 1

    rank_pattern = getattr(peft_config, "rank_pattern", None)
    if isinstance(rank_pattern, dict):
        for name, value in rank_pattern.items():
            state_dict[_adalora_runtime_key("rank_pattern", str(name))] = torch.as_tensor(value, dtype=torch.bool)
            added += 1
    return added


def pop_peft_proxy_adalora_runtime_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    runtime_state: Dict[str, Dict[str, torch.Tensor]] = {
        "ipt": {},
        "exp_avg_ipt": {},
        "exp_avg_unc": {},
        "rank_pattern": {},
    }
    pop_keys = [key for key in state_dict.keys() if str(key).startswith(_ADALORA_RUNTIME_PREFIX)]
    for key in pop_keys:
        tensor = state_dict.pop(key)
        suffix = str(key)[len(_ADALORA_RUNTIME_PREFIX):]
        if "." not in suffix:
            raise ValueError(f"Malformed AdaLoRA runtime state key: {key}")
        group_name, name = suffix.split(".", 1)
        if group_name not in runtime_state:
            raise ValueError(f"Unsupported AdaLoRA runtime state group: {group_name}")
        runtime_state[group_name][name] = tensor
    if any(runtime_state[group_name] for group_name in runtime_state):
        return runtime_state
    return {}


def restore_peft_proxy_adalora_runtime_state_dict(
    model: nn.Module,
    runtime_state: Dict[str, Dict[str, torch.Tensor]],
) -> int:
    if not runtime_state:
        return 0
    peft_config = _resolve_model_peft_config(model)
    if not isinstance(peft_config, AdaLoraConfig):
        raise ValueError("AdaLoRA runtime state found, but model does not have AdaLoRA config.")
    if not _ensure_peft_proxy_adalora_runtime(model, total_step=peft_config.total_step):
        raise RuntimeError("Failed to rebuild AdaLoRA runtime before restoring state.")
    rankallocator = getattr(model, _ADALORA_RANKALLOCATOR_ATTR, None)
    if rankallocator is None:
        raise RuntimeError("AdaLoRA rankallocator is missing.")

    named_params = dict(model.named_parameters())
    restored = 0
    for group_name in ("ipt", "exp_avg_ipt", "exp_avg_unc"):
        restored_group = {}
        for name, tensor in runtime_state.get(group_name, {}).items():
            if name not in named_params:
                raise ValueError(f"AdaLoRA runtime tensor '{name}' is missing from current model parameters.")
            ref_param = named_params[name]
            restored_group[name] = tensor.to(device=ref_param.device, dtype=ref_param.dtype)
            restored += 1
        setattr(rankallocator, group_name, restored_group)

    rank_pattern_state = runtime_state.get("rank_pattern", {})
    if rank_pattern_state:
        peft_config.rank_pattern = {
            str(name): tensor.detach().view(-1).to(dtype=torch.bool).tolist()
            for name, tensor in rank_pattern_state.items()
        }
        restored += len(rank_pattern_state)
    return restored


def ensure_peft_vae_proxy_adapter(
    model: nn.Module,
    *,
    variant: str,
    rank: int,
    alpha: float,
    dropout: float,
    init_mode: str = "zero",
    total_step: Optional[int] = None,
    adalora_target_r: Optional[int] = None,
    adalora_init_r: Optional[int] = None,
    adalora_tinit: int = 0,
    adalora_tfinal: int = 0,
    adalora_delta_t: int = 1,
    adalora_beta1: float = 0.85,
    adalora_beta2: float = 0.85,
    adalora_orth_reg_weight: float = 0.5,
    materialize_before_inject: bool = True,
    materialize_group_size: int = 8,
    materialize_compute_device: Optional[object] = None,
    materialize_logger=None,
) -> int:
    variant = _normalize_variant(variant)
    init_mode = str(init_mode).strip().lower()
    proxy_refs = list(iter_named_peft_vae_proxies(model))
    if not proxy_refs:
        return 0

    injected_count = 0
    for module_name, proxy in proxy_refs:
        per_decoded_linear = proxy.per_decoded_linear
        if is_peft_lora_linear(per_decoded_linear):
            if variant == "adalora":
                raise ValueError(f"Proxy at '{module_name}' is already LoRA, but requested AdaLoRA.")
            _validate_existing_lora_proxy_linear(
                module_name,
                per_decoded_linear,
                variant=variant,
                rank=int(rank),
                alpha=float(alpha),
                dropout=float(dropout),
            )
            injected_count += 1
            continue
        if is_peft_adalora_linear(per_decoded_linear):
            if variant != "adalora":
                raise ValueError(f"Proxy at '{module_name}' is already AdaLoRA, but requested {variant}.")
            _validate_existing_adalora_proxy_linear(
                module_name,
                per_decoded_linear,
                alpha=float(alpha),
                dropout=float(dropout),
                init_r=int(adalora_init_r),
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
        if bool(materialize_before_inject) and _has_unmaterialized_proxy_refs(proxy_refs):
            materialize_peft_proxy_decoded_linears(
                model,
                group_size=int(materialize_group_size),
                compute_device=materialize_compute_device,
                logger=materialize_logger,
            )
        if variant == "adalora":
            inject_adapter_in_model(
                AdaLoraConfig(
                    task_type=None,
                    r=int(adalora_init_r),
                    init_r=int(adalora_init_r),
                    target_r=int(adalora_target_r),
                    tinit=int(adalora_tinit),
                    tfinal=int(adalora_tfinal),
                    deltaT=int(adalora_delta_t),
                    beta1=float(adalora_beta1),
                    beta2=float(adalora_beta2),
                    orth_reg_weight=float(adalora_orth_reg_weight),
                    total_step=None if total_step is None else int(total_step),
                    lora_alpha=float(alpha),
                    lora_dropout=float(dropout),
                    target_modules=["per_decoded_linear"],
                    bias="none",
                    inference_mode=False,
                    init_lora_weights="gaussian" if init_mode == "gaussian" else True,
                ),
                model,
            )
            for module_name, proxy in proxy_refs:
                if not is_peft_adalora_linear(proxy.per_decoded_linear):
                    raise RuntimeError(f"Failed to inject PEFT AdaLoRA into '{module_name}.per_decoded_linear'.")
            if init_mode == "zero":
                _zero_initialize_adalora_modules(model)
        else:
            inject_adapter_in_model(
                LoraConfig(
                    task_type=None,
                    r=int(rank),
                    lora_alpha=float(alpha),
                    lora_dropout=float(dropout),
                    target_modules=["per_decoded_linear"],
                    bias="none",
                    inference_mode=False,
                    use_rslora=variant == "rslora",
                    use_dora=variant == "dora",
                    init_lora_weights="gaussian" if init_mode == "gaussian" else True,
                ),
                model,
            )
            for module_name, proxy in proxy_refs:
                if not is_peft_lora_linear(proxy.per_decoded_linear):
                    raise RuntimeError(f"Failed to inject PEFT LoRA into '{module_name}.per_decoded_linear'.")
    elif variant == "adalora":
        _validate_existing_adalora_root_config(
            model,
            target_r=int(adalora_target_r),
            init_r=int(adalora_init_r),
            tinit=int(adalora_tinit),
            tfinal=int(adalora_tfinal),
            delta_t=int(adalora_delta_t),
            beta1=float(adalora_beta1),
            beta2=float(adalora_beta2),
            orth_reg_weight=float(adalora_orth_reg_weight),
            total_step=None if total_step is None else int(total_step),
        )

    _enable_peft_proxy_adapters(model)
    if variant == "adalora":
        _ensure_peft_proxy_adalora_runtime(model, total_step=total_step)
    return len(proxy_refs)


@torch.no_grad()
def initialize_peft_linear_from_residual_svd(
    peft_linear: PeftLoraLinear,
    residual: torch.Tensor,
    *,
    module_name: str,
) -> None:
    target = _build_residual_svd_target(module_name, peft_linear)
    _initialize_residual_svd_targets_batched(
        [target],
        [residual],
        batch_device=torch.device(residual.device),
    )


@torch.no_grad()
def initialize_peft_vae_proxy_lora_from_teacher_residual(
    model: nn.Module,
    teacher_model: nn.Module,
    *,
    batch_device: torch.device,
) -> int:
    if teacher_model is None:
        raise ValueError("teacher_model is required for residual_svd init.")
    device = torch.device(batch_device)
    items = _collect_teacher_residual_sources(model, teacher_model)
    grouped_items = _group_residual_targets_by_category(items)

    initialized = 0
    for _category, batch_items in grouped_items:
        targets = [item.target for item in batch_items]
        residuals = [
            item.teacher_weight.detach().to(dtype=torch.float32, device=device)
            - item.decoded_weight.detach().to(dtype=torch.float32, device=device)
            for item in batch_items
        ]
        _initialize_residual_svd_targets_batched(
            targets,
            residuals,
            batch_device=device,
        )
        initialized += len(targets)
    return initialized


@torch.no_grad()
def sync_peft_vae_proxy_lora_weights(
    model: nn.Module,
    *,
    sync_device: torch.device,
    src_rank: int = 0,
) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed is not initialized.")

    device = torch.device(sync_device)
    synced = 0
    rank = int(torch.distributed.get_rank())
    is_src_rank = rank == int(src_rank)
    for _name, proxy in _sorted_named_peft_vae_proxies(model):
        peft_linear = proxy.per_decoded_linear
        if not is_peft_lora_linear(peft_linear):
            raise ValueError("sync_peft_vae_proxy_lora_weights only supports LoRA-family PEFT proxies.")
        adapter_name = _get_default_adapter_name(peft_linear)
        tensors = [
            peft_linear.lora_A[adapter_name].weight,
            peft_linear.lora_B[adapter_name].weight,
        ]
        if bool(peft_linear.use_dora.get(adapter_name, False)):
            tensors.append(peft_linear.lora_magnitude_vector[adapter_name])
        for param in tensors:
            if is_src_rank:
                sync_buffer = param.detach().to(device=device, dtype=param.dtype)
            else:
                sync_buffer = torch.empty(tuple(param.shape), device=device, dtype=param.dtype)
            torch.distributed.broadcast(sync_buffer, src=int(src_rank))
            param.copy_(sync_buffer.to(device=param.device, dtype=param.dtype))
        synced += 1

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return synced


def collect_peft_vae_proxy_adapter_specs(
    model: nn.Module,
    *,
    train_mode: str,
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    peft_config = _resolve_model_peft_config(model)
    for name, proxy in iter_named_peft_vae_proxies(model):
        peft_linear = proxy.per_decoded_linear
        if is_peft_lora_linear(peft_linear):
            adapter_name = _get_default_adapter_name(peft_linear)
            specs.append(
                {
                    "name": name,
                    "adapter_type": "peft_proxy_lora",
                    "base_type": "PeftVAELinearProxy",
                    "r": int(peft_linear.r[adapter_name]),
                    "alpha": float(peft_linear.lora_alpha[adapter_name]),
                    "dropout": float(_dropout_p(peft_linear.lora_dropout[adapter_name])),
                    "use_rslora": bool(_resolve_use_rslora(peft_linear, adapter_name)),
                    "use_dora": bool(peft_linear.use_dora.get(adapter_name, False)),
                    "train_mode_at_save": str(train_mode),
                }
            )
            continue
        if is_peft_adalora_linear(peft_linear):
            if not isinstance(peft_config, AdaLoraConfig):
                raise ValueError("Missing AdaLoRA config while exporting PEFT proxy adapter specs.")
            adapter_name = _get_default_adapter_name(peft_linear)
            specs.append(
                {
                    "name": name,
                    "adapter_type": "peft_proxy_adalora",
                    "base_type": "PeftVAELinearProxy",
                    "r": int(peft_linear.r[adapter_name]),
                    "alpha": float(peft_linear.lora_alpha[adapter_name]),
                    "dropout": float(_dropout_p(peft_linear.lora_dropout[adapter_name])),
                    "target_r": int(peft_config.target_r),
                    "init_r": int(peft_config.init_r),
                    "tinit": int(peft_config.tinit),
                    "tfinal": int(peft_config.tfinal),
                    "delta_t": int(peft_config.deltaT),
                    "beta1": float(peft_config.beta1),
                    "beta2": float(peft_config.beta2),
                    "orth_reg_weight": float(peft_config.orth_reg_weight),
                    "total_step": None if peft_config.total_step is None else int(peft_config.total_step),
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
def materialize_peft_proxy_decoded_linears(
    model: nn.Module,
    *,
    group_size: int = 8,
    compute_device: Optional[object] = None,
    logger=None,
    log_prefix: str = "",
) -> Dict[str, object]:
    proxy_refs = _sorted_named_peft_vae_proxies(model)
    if not proxy_refs:
        resolved_compute_device = resolve_grouped_decode_compute_device(
            compute_device,
            logger=logger,
            log_prefix=log_prefix,
        )
        return {
            "total": 0,
            "refreshed": 0,
            "warmed": 0,
            "skipped": 0,
            "failed": 0,
            "group_size": int(group_size),
            "compute_device": str(resolved_compute_device),
            "writeback_device": "none",
            "duration_sec": 0.0,
        }

    requested_compute_device = compute_device
    if requested_compute_device is None:
        requested_compute_device = _resolve_proxy_base_linear(proxy_refs[0][0], proxy_refs[0][1]).weight.device
    resolved_compute_device = resolve_grouped_decode_compute_device(
        requested_compute_device,
        logger=logger,
        log_prefix=log_prefix,
    )

    writeback_devices = {
        str(_resolve_proxy_base_linear(name, proxy).weight.device)
        for name, proxy in proxy_refs
    }
    writeback_device_label = next(iter(writeback_devices)) if len(writeback_devices) == 1 else "mixed"
    if logger is not None:
        logger.info(
            "%sStart proxy materialize: total=%d group_size=%d compute_device=%s writeback_device=%s",
            log_prefix,
            len(proxy_refs),
            int(group_size),
            str(resolved_compute_device),
            writeback_device_label,
        )

    start_time = time.time()
    decode_targets = [
        NamedVAELinearDecodeTarget(
            name=name,
            base_layer=proxy.base_layer,
            target_dtype=_resolve_proxy_base_linear(name, proxy).weight.dtype,
        )
        for name, proxy in proxy_refs
    ]
    decoded_results = decode_named_vae_linear_weights(
        decode_targets,
        group_size=int(group_size),
        compute_device=resolved_compute_device,
        logger=logger,
        respect_cache_policy=False,
    )
    decoded_by_name = {item.name: item for item in decoded_results}
    if len(decoded_by_name) != len(proxy_refs):
        raise RuntimeError(
            f"Proxy materialize result count mismatch: decoded={len(decoded_by_name)} expected={len(proxy_refs)}."
        )

    refreshed = 0
    for name, proxy in proxy_refs:
        if name not in decoded_by_name:
            raise RuntimeError(f"Missing grouped decode result for proxy '{name}'.")
        decoded_item = decoded_by_name[name]
        decoded_linear = _resolve_proxy_base_linear(name, proxy)
        target_device = decoded_linear.weight.device
        target_dtype = decoded_linear.weight.dtype
        decoded_linear.weight.copy_(
            decoded_item.decoded_weight.to(device=target_device, dtype=target_dtype)
        )

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
        proxy.base_layer.clear_decoded_weight_cache()
        proxy._dense_base_materialized = True
        refreshed += 1

    duration_sec = float(time.time() - start_time)
    if logger is not None:
        logger.info(
            "%sFinished proxy materialize: total=%d refreshed=%d failed=%d group_size=%d compute_device=%s writeback_device=%s duration_sec=%.2f",
            log_prefix,
            len(proxy_refs),
            refreshed,
            0,
            int(group_size),
            str(resolved_compute_device),
            writeback_device_label,
            duration_sec,
        )
    return {
        "total": int(len(proxy_refs)),
        "refreshed": int(refreshed),
        "warmed": int(refreshed),
        "skipped": int(len(proxy_refs) - refreshed),
        "failed": 0,
        "group_size": int(group_size),
        "compute_device": str(resolved_compute_device),
        "writeback_device": writeback_device_label,
        "duration_sec": duration_sec,
    }


def update_peft_vae_proxy_adalora(
    model: nn.Module,
    *,
    global_step: int,
) -> bool:
    peft_config = _resolve_model_peft_config(model)
    if not isinstance(peft_config, AdaLoraConfig):
        return False
    if int(getattr(peft_config, "total_step", 0) or 0) <= 0:
        raise ValueError("AdaLoRA update requires total_step > 0.")
    if not _ensure_peft_proxy_adalora_runtime(model, total_step=peft_config.total_step):
        return False
    rankallocator = getattr(model, _ADALORA_RANKALLOCATOR_ATTR, None)
    if rankallocator is None:
        raise RuntimeError("AdaLoRA rankallocator is missing.")

    current_step = int(global_step)
    freeze_step = int(peft_config.total_step) - int(peft_config.tfinal)
    if current_step < freeze_step:
        _, rank_pattern = rankallocator.update_and_allocate(model, current_step)
        if rank_pattern:
            peft_config.rank_pattern = rank_pattern
    elif current_step == freeze_step:
        _, rank_pattern = rankallocator.update_and_allocate(model, current_step, force_mask=True)
        peft_config.rank_pattern = rank_pattern
        rankallocator.reset_ipt()
    elif current_step > freeze_step:
        rank_pattern = getattr(peft_config, "rank_pattern", None)
        if not isinstance(rank_pattern, dict) or len(rank_pattern) == 0:
            raise RuntimeError("AdaLoRA rank_pattern is missing. Resume checkpoint state is incomplete.")
        rankallocator.mask_using_rank_pattern(model, rank_pattern)
    return True
