from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from litebsq.low_rank_scope import LOW_RANK_SCOPE_FULL, normalize_low_rank_scope
from litebsq.vae_linear import VAELinear


def strip_peft_module_prefix(name: str) -> str:
    text = str(name)
    for prefix in ("base_model.model.", "model."):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def iter_lora_target_modules(model: nn.Module):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield strip_peft_module_prefix(str(name)), module


def extract_low_rank_payloads_from_lora(
    peft_model: nn.Module,
    target_modules: Sequence[str],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    target_set = {str(name) for name in target_modules}
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for base_name, module in iter_lora_target_modules(peft_model):
        if base_name not in target_set:
            continue
        lora_a = module.lora_A["default"].weight.detach()
        lora_b = module.lora_B["default"].weight.detach()
        scaling = float(module.scaling["default"])
        low_rank_b = lora_a.to(device="cpu").contiguous()
        low_rank_a = (lora_b.to(device="cpu", dtype=torch.float32) * scaling).to(dtype=lora_b.dtype).contiguous()
        payloads[base_name] = (low_rank_a, low_rank_b)
    if len(payloads) != len(target_set):
        missing = sorted(target_set - set(payloads.keys()))
        raise RuntimeError(f"Missing trained LoRA payloads for low-rank export: {missing}")
    return payloads


def iter_named_vae_linears(model: nn.Module) -> Iterable[Tuple[str, VAELinear]]:
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            yield str(name), module


def write_low_rank_payloads_to_compressed_model(
    model: nn.Module,
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    *,
    expected_scope: Optional[str] = None,
) -> int:
    modules = dict(iter_named_vae_linears(model))
    resolved_expected_scope = None
    if expected_scope is not None:
        resolved_expected_scope = normalize_low_rank_scope(expected_scope)
    written = 0
    for name, (low_rank_a, low_rank_b) in payloads.items():
        module = modules.get(str(name))
        if module is None:
            raise RuntimeError(f"Cannot export low-rank payload; VAELinear not found: {name}")
        if getattr(module, "low_rank_a", None) is None or getattr(module, "low_rank_b", None) is None:
            raise RuntimeError(f"Cannot export low-rank payload; {name} has no low_rank_a/b.")
        if resolved_expected_scope is not None:
            module_scope = normalize_low_rank_scope(
                getattr(module, "low_rank_scope", LOW_RANK_SCOPE_FULL)
            )
            if module_scope != resolved_expected_scope:
                raise RuntimeError(
                    f"{name}: low_rank_scope={module_scope!r} != expected_scope={resolved_expected_scope!r}."
                )
        if tuple(module.low_rank_a.shape) != tuple(low_rank_a.shape):
            raise RuntimeError(f"{name}: low_rank_a shape mismatch: {tuple(module.low_rank_a.shape)} != {tuple(low_rank_a.shape)}.")
        if tuple(module.low_rank_b.shape) != tuple(low_rank_b.shape):
            raise RuntimeError(f"{name}: low_rank_b shape mismatch: {tuple(module.low_rank_b.shape)} != {tuple(low_rank_b.shape)}.")
        module.low_rank_a.data.copy_(low_rank_a.to(device=module.low_rank_a.device, dtype=module.low_rank_a.dtype))
        module.low_rank_b.data.copy_(low_rank_b.to(device=module.low_rank_b.device, dtype=module.low_rank_b.dtype))
        module.clear_decoded_weight_cache()
        written += 1
    return int(written)
