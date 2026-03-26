from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from torch import nn

from e2e_fintuning.lora import ensure_lora_vae_linear, iter_named_vae_module_refs
from train_utils.utils import extract_layer_idx


@dataclass
class TrainableSelection:
    decoder_layer_ids: List[int]
    lora_trainable_params: List[Tuple[str, nn.Parameter]]
    target_modules: List[str]
    adapter_modules: List[str]
    frozen_cacheable_vae_modules: List[str]

    @property
    def trainable_params(self) -> List[Tuple[str, nn.Parameter]]:
        return list(self.lora_trainable_params)

    @property
    def trainable_tensor_count(self) -> int:
        return len(self.trainable_params)

    @property
    def trainable_param_count(self) -> int:
        return int(sum(int(param.numel()) for _name, param in self.trainable_params))

    @property
    def lora_trainable_param_count(self) -> int:
        return int(sum(int(param.numel()) for _name, param in self.lora_trainable_params))


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
        ref.base_layer.cache_decoded_weight = True
        ref.base_layer.clear_decoded_weight_cache()
        if ref.adapter is not None:
            ref.adapter.disable_adapter = not bool(getattr(ref.adapter, "temporary", True))


def _append_param(
    output: List[Tuple[str, nn.Parameter]],
    seen: Set[int],
    name: str,
    param,
) -> None:
    if not isinstance(param, nn.Parameter):
        return
    if id(param) in seen:
        return
    seen.add(id(param))
    param.requires_grad = True
    output.append((name, param))


def _select_lora_trainables(
    module_name: str,
    adapter,
    *,
    output: List[Tuple[str, nn.Parameter]],
    seen: Set[int],
) -> None:
    for param_name, param in adapter.lora_A.named_parameters():
        _append_param(output, seen, f"{module_name}.lora_A.{param_name}", param)
    for param_name, param in adapter.lora_B.named_parameters():
        _append_param(output, seen, f"{module_name}.lora_B.{param_name}", param)


def select_e2e_trainables(
    model: nn.Module,
    *,
    decoder_layer_ids: Sequence[int],
    target_module_names: Optional[Sequence[str]] = None,
    vae_lora_rank: int,
    vae_lora_alpha: float,
    vae_lora_dropout: float,
) -> TrainableSelection:
    _freeze_all(model)

    selected_layers: Set[int] = {int(idx) for idx in decoder_layer_ids}
    selected_module_names: Optional[Set[str]] = None
    if target_module_names is not None:
        selected_module_names = {str(name).strip().lower() for name in target_module_names if str(name).strip()}
    lora_trainable_params: List[Tuple[str, nn.Parameter]] = []
    target_modules: List[str] = []
    adapter_modules: List[str] = []
    frozen_cacheable_vae_modules: List[str] = []
    seen: Set[int] = set()

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
        adapter = ensure_lora_vae_linear(
            model,
            ref.name,
            module,
            rank=int(vae_lora_rank),
            alpha=float(vae_lora_alpha),
            dropout=float(vae_lora_dropout),
        )
        adapter_modules.append(ref.name)
        _select_lora_trainables(
            ref.name,
            adapter,
            output=lora_trainable_params,
            seen=seen,
        )
        base_layer = adapter.base_layer
        base_layer.cache_decoded_weight = True
        base_layer.clear_decoded_weight_cache()
        frozen_cacheable_vae_modules.append(ref.name)

    return TrainableSelection(
        decoder_layer_ids=sorted(selected_layers),
        lora_trainable_params=lora_trainable_params,
        target_modules=sorted(set(target_modules)),
        adapter_modules=sorted(set(adapter_modules)),
        frozen_cacheable_vae_modules=sorted(set(frozen_cacheable_vae_modules)),
    )
