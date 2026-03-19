import math
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch import nn

from litebsq.bsq_linear import set_module_by_name
from litebsq.vae_linear import VAELinear


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
        if int(rank) < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

        super().__init__()
        self.base_layer = base_layer
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_dropout_p = float(dropout)
        self.scaling = float(alpha) / float(rank)
        self.temporary = bool(getattr(base_layer, "temporary", True))
        self.disable_adapter = not self.temporary
        self.lora_dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        ref_param = self._get_reference_param()
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

    def _get_reference_param(self) -> Optional[nn.Parameter]:
        for param in self.base_layer.parameters():
            if param.is_floating_point():
                return param
        return None


@dataclass(frozen=True)
class VAEModuleRef:
    name: str
    module: nn.Module
    base_layer: VAELinear
    adapter: Optional[LoRAVAELinear]


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
