from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set

from torch import nn

from e2e_common.peft_proxy import PeftVAELinearProxy, ensure_peft_vae_linear_proxy
from e2e_common.post_norm_head import resolve_post_norm_linear
from litebsq.vae_linear import VAELinear
from rotation.model_utils import get_model_type, get_pre_head_layernorm
from train_utils.utils import extract_layer_idx


@dataclass(frozen=True)
class VAEModuleRef:
    name: str
    module: nn.Module
    base_layer: VAELinear


@dataclass
class TrainableSelection:
    decoder_layer_ids: List[int]
    target_modules: List[str]
    adapter_modules: List[str]
    peft_proxy_modules: List[str]
    frozen_cacheable_vae_modules: List[str]
    final_norm_modules: List[str]
    post_norm_head_modules: List[str]


def iter_named_vae_module_refs(model: nn.Module) -> Iterator[VAEModuleRef]:
    skip_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if isinstance(module, PeftVAELinearProxy):
            skip_prefixes.append(f"{name}.base_layer")
            skip_prefixes.append(f"{name}.per_decoded_linear")
            yield VAEModuleRef(
                name=str(name),
                module=module,
                base_layer=module.base_layer,
            )
            continue
        if isinstance(module, VAELinear):
            yield VAEModuleRef(
                name=str(name),
                module=module,
                base_layer=module,
            )


def resolve_target_layer_ids(requested: Optional[Sequence[int]], num_layers: int) -> List[int]:
    if requested is None:
        return list(range(int(num_layers)))

    resolved = sorted(set(int(idx) for idx in requested))
    for idx in resolved:
        if idx < 0 or idx >= int(num_layers):
            raise ValueError(f"Invalid decoder layer id {idx}; valid range is [0, {int(num_layers) - 1}].")
    return resolved


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for ref in iter_named_vae_module_refs(model):
        ref.base_layer.cache_decoded_weight = not isinstance(ref.module, PeftVAELinearProxy)
        ref.base_layer.clear_decoded_weight_cache()


def _enable_module_trainable(module: nn.Module, module_name: str) -> List[str]:
    enabled_names: List[str] = []
    for param_name, param in module.named_parameters(recurse=True):
        param.requires_grad = True
        full_name = str(module_name) if not param_name else f"{module_name}.{param_name}"
        enabled_names.append(full_name)
    return enabled_names


def _find_module_name(model: nn.Module, target: nn.Module, fallback: str) -> str:
    for name, module in model.named_modules():
        if module is target:
            return str(name)
    return str(fallback)


def select_e2e_trainables_peft_proxy(
    model: nn.Module,
    *,
    decoder_layer_ids: Sequence[int],
    target_module_names: Optional[Sequence[str]] = None,
    tune_final_norm: bool = False,
    use_post_norm_head_linear: bool = False,
) -> TrainableSelection:
    _freeze_all(model)

    selected_layers: Set[int] = {int(idx) for idx in decoder_layer_ids}
    selected_module_names: Optional[Set[str]] = None
    if target_module_names is not None:
        selected_module_names = {str(name).strip().lower() for name in target_module_names if str(name).strip()}
    target_modules: List[str] = []
    adapter_modules: List[str] = []
    peft_proxy_modules: List[str] = []
    frozen_cacheable_vae_modules: List[str] = []
    final_norm_modules: List[str] = []
    post_norm_head_modules: List[str] = []

    refs = list(iter_named_vae_module_refs(model))
    for ref in refs:
        layer_idx = extract_layer_idx(ref.name)
        module_category = str(ref.name).rsplit(".", 1)[-1].lower()
        module = ref.module
        base_layer = ref.base_layer

        if layer_idx not in selected_layers:
            base_layer.cache_decoded_weight = True
            base_layer.clear_decoded_weight_cache()
            frozen_cacheable_vae_modules.append(ref.name)
            continue
        if selected_module_names is not None and module_category not in selected_module_names:
            base_layer.cache_decoded_weight = True
            base_layer.clear_decoded_weight_cache()
            frozen_cacheable_vae_modules.append(ref.name)
            continue

        target_modules.append(ref.name)
        proxy = ensure_peft_vae_linear_proxy(model, ref.name, module)
        proxy.train(module.training)
        peft_proxy_modules.append(ref.name)
        adapter_modules.append(ref.name)
        base_layer = proxy.base_layer
        base_layer.cache_decoded_weight = False
        base_layer.clear_decoded_weight_cache()

    if bool(tune_final_norm):
        model_type = get_model_type(model)
        final_norm = get_pre_head_layernorm(model, model_type)
        final_norm_name = _find_module_name(model, final_norm, "model.norm")
        final_norm_modules.extend(_enable_module_trainable(final_norm, final_norm_name))

    if bool(use_post_norm_head_linear):
        post_norm_linear = resolve_post_norm_linear(model)
        if post_norm_linear is None:
            raise ValueError(
                "--use_post_norm_head_linear=true but model.lm_head is not LMHeadWithPostNormLinear."
            )
        post_norm_head_modules.extend(_enable_module_trainable(post_norm_linear, "lm_head.post_norm_linear"))

    return TrainableSelection(
        decoder_layer_ids=sorted(selected_layers),
        target_modules=sorted(set(target_modules)),
        adapter_modules=sorted(set(adapter_modules)),
        peft_proxy_modules=sorted(set(peft_proxy_modules)),
        frozen_cacheable_vae_modules=sorted(set(frozen_cacheable_vae_modules)),
        final_norm_modules=sorted(set(final_norm_modules)),
        post_norm_head_modules=sorted(set(post_norm_head_modules)),
    )
