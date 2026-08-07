from typing import Iterator, Tuple

from torch import nn

from e2e_common.compressed_subspace_lora import CompressedSubspacePeftProxy
from e2e_common.peft_proxy import PeftVAELinearProxy


def _iter_named_temporary_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    skip_prefixes = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if isinstance(module, PeftVAELinearProxy):
            skip_prefixes.append(f"{name}.base_layer")
            skip_prefixes.append(f"{name}.per_decoded_linear")
        elif isinstance(module, CompressedSubspacePeftProxy):
            skip_prefixes.append(f"{name}.base_layer")
            skip_prefixes.append(f"{name}.{CompressedSubspacePeftProxy.CARRIER_NAME}")
        if callable(getattr(module, "set_temporary", None)):
            yield name, module


def set_model_temporary(model: nn.Module, temporary: bool) -> None:
    for _name, module in _iter_named_temporary_modules(model):
        module.set_temporary(bool(temporary))
