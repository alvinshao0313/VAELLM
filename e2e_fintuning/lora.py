import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import nn

from e2e_fintuning.peft_proxy import PeftVAELinearProxy
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear


def _validate_lora_hparams(rank: int, dropout: float) -> Tuple[int, float]:
    norm_rank = int(rank)
    norm_dropout = float(dropout)
    if norm_rank < 1:
        raise ValueError(f"LoRA rank must be >= 1, got {rank}")
    if norm_dropout < 0.0 or norm_dropout >= 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")
    return norm_rank, norm_dropout


def _resolve_reference_param(module: nn.Module) -> Optional[nn.Parameter]:
    for param in module.parameters():
        if param.is_floating_point():
            return param
    return None


class LoRAVAELinear(nn.Module):
    def __init__(
        self,
        base_layer: VAELinear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        if not isinstance(base_layer, VAELinear):
            raise TypeError(f"LoRAVAELinear expects VAELinear base_layer, got {type(base_layer)}")
        norm_rank, norm_dropout = _validate_lora_hparams(rank, dropout)

        super().__init__()
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(norm_rank)
        self.lora_alpha = float(alpha)
        self.lora_dropout_p = float(norm_dropout)
        self.scaling = float(alpha) / float(norm_rank)
        self.temporary = bool(getattr(base_layer, "temporary", True))
        self.disable_adapter = not self.temporary
        self.lora_dropout = nn.Dropout(p=float(norm_dropout)) if float(norm_dropout) > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        ref_param = _resolve_reference_param(base_layer)
        if ref_param is not None and ref_param.is_floating_point():
            self.lora_A.to(device=ref_param.device, dtype=ref_param.dtype)
            self.lora_B.to(device=ref_param.device, dtype=ref_param.dtype)

    @property
    def bias(self):
        return self.base_layer.bias

    def set_temporary(self, temporary: bool = True) -> None:
        self.base_layer.set_temporary(temporary)
        if bool(getattr(self.base_layer, "always_use_original", False)):
            self.temporary = False
        else:
            self.temporary = bool(temporary)
        self.disable_adapter = not self.temporary

    def merge_delta_weight(self) -> torch.Tensor:
        return (self.lora_B.weight @ self.lora_A.weight) * float(self.scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(x)
        if self.disable_adapter:
            return output
        delta_input = self.lora_dropout(x)
        lora_dtype = self.lora_A.weight.dtype
        if delta_input.dtype != lora_dtype:
            delta_input = delta_input.to(dtype=lora_dtype)
        delta = self.lora_B(self.lora_A(delta_input))
        if delta.dtype != output.dtype:
            delta = delta.to(dtype=output.dtype)
        return output + delta * float(self.scaling)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout_p}"
        )


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear base_layer, got {type(base_layer)}")
        norm_rank, norm_dropout = _validate_lora_hparams(rank, dropout)

        super().__init__()
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(norm_rank)
        self.lora_alpha = float(alpha)
        self.lora_dropout_p = float(norm_dropout)
        self.scaling = float(alpha) / float(norm_rank)
        self.temporary = True
        self.disable_adapter = False
        self.lora_dropout = nn.Dropout(p=float(norm_dropout)) if float(norm_dropout) > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        ref_param = _resolve_reference_param(base_layer)
        if ref_param is not None and ref_param.is_floating_point():
            self.lora_A.to(device=ref_param.device, dtype=ref_param.dtype)
            self.lora_B.to(device=ref_param.device, dtype=ref_param.dtype)

    @property
    def bias(self):
        return self.base_layer.bias

    def set_temporary(self, temporary: bool = True) -> None:
        self.temporary = bool(temporary)
        self.disable_adapter = not self.temporary

    def merge_delta_weight(self) -> torch.Tensor:
        return (self.lora_B.weight @ self.lora_A.weight) * float(self.scaling)

    def merge_and_unload(self) -> nn.Linear:
        delta = self.merge_delta_weight().to(
            dtype=self.base_layer.weight.dtype,
            device=self.base_layer.weight.device,
        )
        with torch.no_grad():
            self.base_layer.weight.add_(delta)
        self.disable_adapter = True
        return self.base_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(x)
        if self.disable_adapter:
            return output
        delta_input = self.lora_dropout(x)
        lora_dtype = self.lora_A.weight.dtype
        if delta_input.dtype != lora_dtype:
            delta_input = delta_input.to(dtype=lora_dtype)
        delta = self.lora_B(self.lora_A(delta_input))
        if delta.dtype != output.dtype:
            delta = delta.to(dtype=output.dtype)
        return output + delta * float(self.scaling)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout_p}"
        )


class LoRAEmbedding(nn.Module):
    def __init__(
        self,
        base_layer: nn.Embedding,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        if not isinstance(base_layer, nn.Embedding):
            raise TypeError(f"LoRAEmbedding expects nn.Embedding base_layer, got {type(base_layer)}")
        norm_rank, norm_dropout = _validate_lora_hparams(rank, dropout)

        super().__init__()
        self.base_layer = base_layer
        self.num_embeddings = int(base_layer.num_embeddings)
        self.embedding_dim = int(base_layer.embedding_dim)
        self.rank = int(norm_rank)
        self.lora_alpha = float(alpha)
        self.lora_dropout_p = float(norm_dropout)
        self.scaling = float(alpha) / float(norm_rank)
        self.temporary = True
        self.disable_adapter = False
        self.lora_dropout = nn.Dropout(p=float(norm_dropout)) if float(norm_dropout) > 0.0 else nn.Identity()
        self.lora_A = nn.Embedding(
            self.num_embeddings,
            self.rank,
            padding_idx=base_layer.padding_idx,
            sparse=bool(base_layer.sparse),
        )
        self.lora_B = nn.Linear(self.rank, self.embedding_dim, bias=False)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_A.padding_idx is not None:
            with torch.no_grad():
                self.lora_A.weight[self.lora_A.padding_idx].zero_()
        ref_param = _resolve_reference_param(base_layer)
        if ref_param is not None and ref_param.is_floating_point():
            self.lora_A.to(device=ref_param.device, dtype=ref_param.dtype)
            self.lora_B.to(device=ref_param.device, dtype=ref_param.dtype)

    @property
    def weight(self):
        return self.base_layer.weight

    def set_temporary(self, temporary: bool = True) -> None:
        self.temporary = bool(temporary)
        self.disable_adapter = not self.temporary

    def merge_delta_weight(self) -> torch.Tensor:
        return (self.lora_A.weight @ self.lora_B.weight.transpose(0, 1)) * float(self.scaling)

    def merge_and_unload(self) -> nn.Embedding:
        delta = self.merge_delta_weight().to(
            dtype=self.base_layer.weight.dtype,
            device=self.base_layer.weight.device,
        )
        with torch.no_grad():
            self.base_layer.weight.add_(delta)
        self.disable_adapter = True
        return self.base_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(x)
        if self.disable_adapter:
            return output
        delta = self.lora_A(x)
        delta = self.lora_dropout(delta)
        lora_dtype = self.lora_B.weight.dtype
        if delta.dtype != lora_dtype:
            delta = delta.to(dtype=lora_dtype)
        delta = self.lora_B(delta)
        if delta.dtype != output.dtype:
            delta = delta.to(dtype=output.dtype)
        return output + delta * float(self.scaling)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"rank={self.rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout_p}"
        )


@dataclass(frozen=True)
class VAEModuleRef:
    name: str
    module: nn.Module
    base_layer: VAELinear
    adapter: Optional[nn.Module]


def iter_named_vae_module_refs(model: nn.Module) -> Iterator[VAEModuleRef]:
    skip_prefixes = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if isinstance(module, LoRAVAELinear):
            skip_prefixes.append(f"{name}.base_layer")
            yield VAEModuleRef(
                name=name,
                module=module,
                base_layer=module.base_layer,
                adapter=module,
            )
            continue
        if isinstance(module, PeftVAELinearProxy):
            skip_prefixes.append(f"{name}.base_layer")
            skip_prefixes.append(f"{name}.per_decoded_linear")
            yield VAEModuleRef(
                name=name,
                module=module,
                base_layer=module.base_layer,
                adapter=None,
            )
            continue
        if isinstance(module, VAELinear):
            yield VAEModuleRef(
                name=name,
                module=module,
                base_layer=module,
                adapter=None,
            )


def ensure_lora_vae_linear(
    model: nn.Module,
    module_name: str,
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRAVAELinear:
    if isinstance(module, LoRAVAELinear):
        if (
            int(module.rank) != int(rank)
            or float(module.lora_alpha) != float(alpha)
            or float(module.lora_dropout_p) != float(dropout)
        ):
            raise ValueError(
                f"Existing LoRAVAELinear at '{module_name}' has config "
                f"(rank={module.rank}, alpha={module.lora_alpha}, dropout={module.lora_dropout_p}) "
                f"but requested (rank={rank}, alpha={alpha}, dropout={dropout})."
            )
        return module

    if not isinstance(module, VAELinear):
        raise TypeError(f"Expected VAELinear or LoRAVAELinear at '{module_name}', got {type(module)}")

    wrapper = LoRAVAELinear(
        base_layer=module,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
    )
    wrapper.train(module.training)
    set_module_by_name(model, module_name, wrapper)
    return wrapper


def ensure_lora_linear(
    model: nn.Module,
    module_name: str,
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRALinear:
    if isinstance(module, LoRALinear):
        if (
            int(module.rank) != int(rank)
            or float(module.lora_alpha) != float(alpha)
            or float(module.lora_dropout_p) != float(dropout)
        ):
            raise ValueError(
                f"Existing LoRALinear at '{module_name}' has config "
                f"(rank={module.rank}, alpha={module.lora_alpha}, dropout={module.lora_dropout_p}) "
                f"but requested (rank={rank}, alpha={alpha}, dropout={dropout})."
            )
        return module
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear or LoRALinear at '{module_name}', got {type(module)}")

    wrapper = LoRALinear(
        base_layer=module,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
    )
    wrapper.train(module.training)
    set_module_by_name(model, module_name, wrapper)
    return wrapper


def ensure_lora_embedding(
    model: nn.Module,
    module_name: str,
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRAEmbedding:
    if isinstance(module, LoRAEmbedding):
        if (
            int(module.rank) != int(rank)
            or float(module.lora_alpha) != float(alpha)
            or float(module.lora_dropout_p) != float(dropout)
        ):
            raise ValueError(
                f"Existing LoRAEmbedding at '{module_name}' has config "
                f"(rank={module.rank}, alpha={module.lora_alpha}, dropout={module.lora_dropout_p}) "
                f"but requested (rank={rank}, alpha={alpha}, dropout={dropout})."
            )
        return module
    if not isinstance(module, nn.Embedding):
        raise TypeError(f"Expected nn.Embedding or LoRAEmbedding at '{module_name}', got {type(module)}")

    wrapper = LoRAEmbedding(
        base_layer=module,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
    )
    wrapper.train(module.training)
    set_module_by_name(model, module_name, wrapper)
    return wrapper


def merge_and_unload_extra_lora_modules(model: nn.Module) -> Tuple[nn.Module, int]:
    merge_targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (LoRALinear, LoRAEmbedding))
    ]
    for name, module in merge_targets:
        set_module_by_name(model, name, module.merge_and_unload())
    return model, len(merge_targets)


def merge_extra_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int]:
    merged_state_dict = dict(state_dict)
    if not merged_state_dict:
        return merged_state_dict, 0
    merged_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            base_weight_key = f"{name}.base_layer.weight"
            lora_a_key = f"{name}.lora_A.weight"
            lora_b_key = f"{name}.lora_B.weight"
            if base_weight_key not in merged_state_dict or lora_a_key not in merged_state_dict or lora_b_key not in merged_state_dict:
                raise KeyError(f"Missing LoRALinear state_dict entries for '{name}'.")
            delta = (merged_state_dict[lora_b_key] @ merged_state_dict[lora_a_key]) * float(module.scaling)
            merged_state_dict[f"{name}.weight"] = merged_state_dict[base_weight_key] + delta.to(
                dtype=merged_state_dict[base_weight_key].dtype
            )
            base_bias_key = f"{name}.base_layer.bias"
            if base_bias_key in merged_state_dict:
                merged_state_dict[f"{name}.bias"] = merged_state_dict[base_bias_key]
                merged_state_dict.pop(base_bias_key)
            merged_state_dict.pop(base_weight_key)
            merged_state_dict.pop(lora_a_key)
            merged_state_dict.pop(lora_b_key)
            merged_count += 1
            continue

        if not isinstance(module, LoRAEmbedding):
            continue
        base_weight_key = f"{name}.base_layer.weight"
        lora_a_key = f"{name}.lora_A.weight"
        lora_b_key = f"{name}.lora_B.weight"
        if base_weight_key not in merged_state_dict or lora_a_key not in merged_state_dict or lora_b_key not in merged_state_dict:
            raise KeyError(f"Missing LoRAEmbedding state_dict entries for '{name}'.")
        delta = (
            merged_state_dict[lora_a_key] @ merged_state_dict[lora_b_key].transpose(0, 1)
        ) * float(module.scaling)
        merged_state_dict[f"{name}.weight"] = merged_state_dict[base_weight_key] + delta.to(
            dtype=merged_state_dict[base_weight_key].dtype
        )
        merged_state_dict.pop(base_weight_key)
        merged_state_dict.pop(lora_a_key)
        merged_state_dict.pop(lora_b_key)
        merged_count += 1
    return merged_state_dict, merged_count
