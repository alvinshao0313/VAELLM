from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from torch import nn

from litebsq.vae_linear import VAELinear


@dataclass(frozen=True)
class VAEModuleRef:
    name: str
    module: nn.Module
    base_layer: VAELinear


def iter_named_vae_module_refs(model: nn.Module) -> Iterator[VAEModuleRef]:
    skip_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
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
