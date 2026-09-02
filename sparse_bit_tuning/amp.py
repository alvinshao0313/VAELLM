from __future__ import annotations

from collections import defaultdict
from types import MethodType
from typing import Dict, Iterable, Sequence

import torch

try:
    from torch.amp.grad_scaler import OptState
except ImportError:  # pragma: no cover
    OptState = None  # type: ignore[assignment]


_BIT_GROUP_MARKER = "_sparse_bit_score_group"


class _OptimizerGroupView:
    def __init__(self, param_groups):
        self.param_groups = list(param_groups)


class SparseBitGradScaler(torch.amp.GradScaler):
    """GradScaler that permits FP16 grads only for marked Sparse-Bit score groups.

    PyTorch 2.6's public ``unscale_`` rejects FP16 gradients unconditionally.
    Sparse Bit scores are intentionally FP16, so we split optimizer groups and call
    the same PyTorch unscale implementation with ``allow_fp16=True`` only for the
    explicitly marked score groups.  Main parameter groups preserve stock behavior.
    """

    @classmethod
    def from_existing(cls, base_scaler) -> "SparseBitGradScaler":
        if isinstance(base_scaler, cls):
            return base_scaler
        required = ("_unscale_grads_", "_per_optimizer_states", "_check_scale_growth_tracker")
        missing = [name for name in required if not hasattr(base_scaler, name)]
        if missing:
            raise RuntimeError(
                "Sparse Bit FP16 AMP adapter unsupported for this torch version; "
                f"base GradScaler is missing {missing}. Use BF16 or update the adapter."
            )
        scaler = cls("cuda", enabled=bool(base_scaler.is_enabled()))
        scaler._sparse_bit_grad_scaler = True
        if bool(base_scaler.is_enabled()):
            scaler.load_state_dict(base_scaler.state_dict())
        return scaler

    @staticmethod
    def _merge_found_inf(*parts: Dict[torch.device, torch.Tensor]) -> Dict[torch.device, torch.Tensor]:
        merged: Dict[torch.device, torch.Tensor] = {}
        for part in parts:
            for device, value in part.items():
                if device in merged:
                    merged[device].add_(value.to(device=merged[device].device))
                else:
                    merged[device] = value
        return merged

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if not self._enabled:
            return
        if OptState is None:  # pragma: no cover
            raise RuntimeError("Sparse Bit FP16 AMP adapter cannot import torch GradScaler OptState.")
        self._check_scale_growth_tracker("unscale_")
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")
        assert self._scale is not None
        inv_scale = self._scale.double().reciprocal().float()
        found_inf_seed = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)
        main_groups = []
        bit_groups = []
        for group in optimizer.param_groups:
            if bool(group.get(_BIT_GROUP_MARKER, False)):
                bit_groups.append(group)
            else:
                main_groups.append(group)
        if not bit_groups:
            raise RuntimeError(
                "SparseBitGradScaler was installed but optimizer has no marked Sparse Bit score param groups."
            )
        main_found = {}
        if main_groups:
            native_main_params = []
            bf16_found: Dict[torch.device, torch.Tensor] = {}
            for group in main_groups:
                for param in group["params"]:
                    grad = param.grad
                    if grad is None:
                        continue
                    if grad.dtype == torch.float16:
                        raise ValueError(
                            "Attempting to unscale FP16 gradients in a non-Bit parameter group. "
                            "Keep ordinary trainable parameters in BF16/FP32 when using Sparse Bit FP16 AMP."
                        )
                    if grad.dtype != torch.bfloat16:
                        native_main_params.append(param)
                        continue
                    if grad.is_sparse:
                        grad = grad.coalesce()
                        param.grad = grad
                        value = grad._values()
                    else:
                        value = grad
                    device = value.device
                    bad = (~torch.isfinite(value)).any().to(dtype=torch.float32)
                    if device not in bf16_found:
                        bf16_found[device] = torch.zeros((), dtype=torch.float32, device=device)
                    bf16_found[device].add_(bad)
                    value.mul_(inv_scale.to(device=device, dtype=torch.float32))
            native_found = {}
            if native_main_params:
                native_found = self._unscale_grads_(
                    _OptimizerGroupView([{"params": native_main_params}]),
                    inv_scale,
                    found_inf_seed,
                    False,
                )
            main_found = self._merge_found_inf(native_found, bf16_found)
        bit_found = self._unscale_grads_(
            _OptimizerGroupView(bit_groups), inv_scale, found_inf_seed, True
        )
        # A marked score parameter is expected to participate in every valid Bit step.
        for group in bit_groups:
            for param in group["params"]:
                if param.grad is None:
                    raise RuntimeError(
                        "Sparse Bit FP16 score parameter has grad=None during GradScaler unscale; "
                        "the bit-aware autograd path is disconnected or the target did not execute."
                    )
                if param.grad.dtype != torch.float16:
                    raise RuntimeError(
                        f"Sparse Bit score grad must remain FP16, got {param.grad.dtype}."
                    )
        optimizer_state["found_inf_per_device"] = self._merge_found_inf(main_found, bit_found)
        optimizer_state["stage"] = OptState.UNSCALED


def install_sparse_bit_grad_scaler(accelerator):
    """Replace an existing FP16 scaler in-place; BF16/no-scaler is a no-op."""
    base = getattr(accelerator, "scaler", None)
    if base is None:
        return None
    if isinstance(base, SparseBitGradScaler):
        return base
    adapted = SparseBitGradScaler.from_existing(base)
    accelerator.scaler = adapted
    return adapted


def install_main_grad_clip_filter(accelerator, main_parameters: Sequence[torch.nn.Parameter]):
    """Keep HF clipping cadence/threshold while excluding Sparse-Bit score Parameters."""
    if getattr(accelerator, "_sparse_bit_original_clip_grad_norm", None) is not None:
        raise RuntimeError("Sparse Bit main-gradient clip filter is already installed.")
    original = accelerator.clip_grad_norm_
    main_params = tuple(param for param in main_parameters if bool(param.requires_grad))
    accelerator._sparse_bit_original_clip_grad_norm = original
    accelerator._sparse_bit_main_clip_parameter_ids = tuple(id(param) for param in main_params)

    def _clip_grad_norm_filtered(self, parameters, max_norm, norm_type=2):
        # Ignore HF's model.parameters() iterable and clip only continuous trainables.
        del parameters
        self.unscale_gradients()
        grads = [param for param in main_params if param.grad is not None]
        if not grads:
            device = getattr(self, "device", torch.device("cpu"))
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.nn.utils.clip_grad_norm_(grads, max_norm, norm_type=norm_type)

    accelerator.clip_grad_norm_ = MethodType(_clip_grad_norm_filtered, accelerator)
    return original


def restore_main_grad_clip_filter(accelerator) -> None:
    original = getattr(accelerator, "_sparse_bit_original_clip_grad_norm", None)
    if original is None:
        return
    accelerator.clip_grad_norm_ = original
    delattr(accelerator, "_sparse_bit_original_clip_grad_norm")
    if hasattr(accelerator, "_sparse_bit_main_clip_parameter_ids"):
        delattr(accelerator, "_sparse_bit_main_clip_parameter_ids")
