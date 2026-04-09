from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from torch import nn

from e2e_fintuning.lora import LoRAVAELinear, iter_named_vae_module_refs
from e2e_fintuning.peft_proxy import PeftVAELinearProxy, ensure_peft_vae_linear_proxy
from train_utils.utils import extract_layer_idx


@dataclass
class TrainableSelection:
    decoder_layer_ids: List[int]
    target_modules: List[str]
    adapter_modules: List[str]
    peft_proxy_modules: List[str]
    frozen_cacheable_vae_modules: List[str]


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
    for _name, module in model.named_modules():
        if isinstance(module, LoRAVAELinear):
            module.disable_adapter = not bool(getattr(module, "temporary", True))
    for ref in iter_named_vae_module_refs(model):
        ref.base_layer.cache_decoded_weight = not isinstance(ref.module, PeftVAELinearProxy)
        ref.base_layer.clear_decoded_weight_cache()


def select_e2e_trainables_peft_proxy(
    model: nn.Module,
    *,
    decoder_layer_ids: Sequence[int],
    target_module_names: Optional[Sequence[str]] = None,
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

    return TrainableSelection(
        decoder_layer_ids=sorted(selected_layers),
        target_modules=sorted(set(target_modules)),
        adapter_modules=sorted(set(adapter_modules)),
        peft_proxy_modules=sorted(set(peft_proxy_modules)),
        frozen_cacheable_vae_modules=sorted(set(frozen_cacheable_vae_modules)),
    )
