"""Parameter precision for CAT/E2E, independent of activation and export dtype."""

from __future__ import annotations

import logging
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def mixed_precision_dtype(args):
    bf16 = bool(getattr(args, "bf16", False))
    fp16 = bool(getattr(args, "fp16", False))
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled.")
    return torch.bfloat16 if bf16 else torch.float16 if fp16 else None


def _linear_forward(self, input):
    dtype = getattr(self, "_distill_compute_dtype", None)
    if dtype is None:
        dtype = torch.get_autocast_dtype(input.device.type) if torch.is_autocast_enabled(input.device.type) else input.dtype
    bias = None if self.bias is None else self.bias.to(dtype=dtype)
    return F.linear(input.to(dtype=dtype), self.weight.to(dtype=dtype), bias)


def _norm_output_dtype(module, inputs, output):
    return output.to(dtype=inputs[0].dtype)


def _torch_norm_forward(self, input):
    weight = None if self.weight is None else self.weight.to(dtype=input.dtype)
    bias = getattr(self, "bias", None)
    bias = None if bias is None else bias.to(dtype=input.dtype)
    if isinstance(self, nn.LayerNorm):
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
    return F.rms_norm(input, self.normalized_shape, weight, self.eps)


def install_precision_runtime(model, compute_dtype=None):
    """Keep FP32 leaf parameters; casts in forward remain differentiable.

    Runtime hooks do not add state-dict entries and are reinstalled on loading.
    """
    from litebsq.vae_linear import VAELinear
    from train_utils.model_level_trainables import NORM_TYPE_REGISTRY

    model._distill_model_compute_dtype = compute_dtype
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        getter()._distill_model_compute_dtype = compute_dtype
    for module in model.modules():
        if isinstance(module, VAELinear):
            if getattr(module, "_decoder_compute_dtype", None) != compute_dtype:
                module.clear_decoded_weight_cache()
            module._decoder_compute_dtype = compute_dtype
        if type(module) is nn.Linear and module.weight.dtype == torch.float32:
            module._distill_compute_dtype = compute_dtype
            module.forward = MethodType(_linear_forward, module)
        if isinstance(module, NORM_TYPE_REGISTRY):
            weight = getattr(module, "weight", None)
            if type(module) in (nn.LayerNorm, nn.RMSNorm) and weight is not None and weight.dtype == torch.float32:
                module.forward = MethodType(_torch_norm_forward, module)
            if weight is not None and weight.dtype == torch.float32 and not hasattr(module, "_distill_norm_handle"):
                module._distill_norm_handle = module.register_forward_hook(_norm_output_dtype)


def configure_distill_precision(selection, *, components, training_args, logger=None):
    from train_utils.config.configs import parse_distill_fp32_components

    components = parse_distill_fp32_components(components)
    dtype = mixed_precision_dtype(training_args)
    log = logger or logging.getLogger(__name__)
    counts = {}
    for component in components:
        inventory = getattr(selection, f"{component}_parameters")
        counts[component] = sum(p.numel() for p in inventory.values())
        for parameter in inventory.values():
            if not parameter.requires_grad:
                raise RuntimeError(f"Frozen parameter in FP32 {component} inventory.")
            parameter.data = parameter.data.to(dtype=torch.float32)
    if components:
        # BatchNorm keeps floating running statistics alongside its parameters.
        for module in selection.peft_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.weight is not None and module.weight.dtype == torch.float32:
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer.is_floating_point():
                        setattr(module, name, buffer.to(dtype=torch.float32))
    install_precision_runtime(selection.peft_model, dtype)
    if dtype == torch.float16:
        invalid = [name for name, p in selection.peft_model.named_parameters() if p.requires_grad and p.dtype == torch.float16]
        if invalid:
            from train_utils.config.configs import DISTILL_FP32_COMPONENTS

            affected = [component for component in DISTILL_FP32_COMPONENTS
                        if any(p.dtype == torch.float16 for p in getattr(selection, f"{component}_parameters").values())]
            raise ValueError(
                "FP16 GradScaler cannot unscale FP16 trainable gradients. Include their components "
                f"in --distill_fp32_components: components={affected}, parameters={invalid[:8]}"
            )
    effective = tuple(name for name, count in counts.items() if count)
    log.info(
        "Distill FP32 components: requested=%s effective=%s parameter_counts=%s parameter_dtype=float32 compute_dtype=%s",
        components, effective, counts, dtype,
    )
    return counts


@torch.no_grad()
def prepare_model_export(model, training_args):
    """Call only at a completed stage boundary, never for a training-step save."""
    dtype = mixed_precision_dtype(training_args)
    if dtype is not None:
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=dtype)
        # Quantization buffers have their own storage format; do not cast them.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer.is_floating_point():
                        setattr(module, name, buffer.to(dtype=dtype))
        if getattr(model, "config", None) is not None:
            model.config.torch_dtype = dtype
    # Update already-installed linear compute policies after export.
    for module in model.modules():
        if hasattr(module, "_distill_compute_dtype"):
            module._distill_compute_dtype = dtype
        clear = getattr(module, "clear_decoded_weight_cache", None)
        if callable(clear):
            clear()
    install_precision_runtime(model, dtype)
    return dtype


@torch.no_grad()
def restore_parameter_dtypes(model, state_dict):
    """load_state_dict copies values, so restore storage dtype before that copy."""
    seen = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        source = state_dict.get(name)
        if source is None or not source.is_floating_point():
            continue
        previous = seen.setdefault(id(parameter), source.dtype)
        if previous != source.dtype:
            raise ValueError(f"Conflicting saved dtypes for tied parameter {name}.")
        if parameter.dtype != source.dtype:
            parameter.data = parameter.data.to(dtype=source.dtype)
    for name, buffer in model.named_buffers():
        source = state_dict.get(name)
        if source is not None and buffer.is_floating_point() and buffer.dtype != source.dtype:
            buffer.data = buffer.data.to(dtype=source.dtype)
