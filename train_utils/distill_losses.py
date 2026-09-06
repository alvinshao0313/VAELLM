"""Causal token-region helpers shared by model-level distillation tests/runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistillTokenRegions:
    response_mask: torch.Tensor
    prompt_mask: torch.Tensor


def _validate_distill_mask_shape(
    source: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    name: str,
) -> None:
    if source.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [batch, sequence], got {tuple(source.shape)}.")
    if reference_logits.ndim != 3:
        raise ValueError(
            "reference_logits must be rank-3 [batch, sequence, vocab], "
            f"got {tuple(reference_logits.shape)}."
        )
    if tuple(source.shape) != tuple(reference_logits.shape[:2]):
        raise ValueError(
            f"{name} shape {tuple(source.shape)} does not match logits "
            f"{tuple(reference_logits.shape[:2])}."
        )


def _apply_causal_shift(source_weights: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(source_weights, dtype=torch.float32)
    if source_weights.shape[1] > 1:
        shifted[:, :-1] = source_weights[:, 1:].to(dtype=torch.float32)
    return shifted


def build_distill_token_mask(
    *,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the causal response-token mask aligned with logit positions."""
    batch, sequence = reference_logits.shape[:2]
    device = reference_logits.device
    if labels is not None:
        _validate_distill_mask_shape(labels, reference_logits, name="labels")
        source = labels.ne(-100)
    elif attention_mask is not None:
        _validate_distill_mask_shape(attention_mask, reference_logits, name="attention_mask")
        source = attention_mask.ne(0)
    else:
        source = torch.ones((batch, sequence), dtype=torch.bool, device=device)
    return _apply_causal_shift(source.to(device=device))


def build_distill_token_regions(
    *,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    reference_logits: torch.Tensor,
) -> DistillTokenRegions:
    """Split valid causal positions into response and prompt regions."""
    response = build_distill_token_mask(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )
    valid = build_distill_token_mask(
        labels=None,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )
    prompt = (valid - response).clamp_min(0.0)
    return DistillTokenRegions(response_mask=response, prompt_mask=prompt)


def compute_forward_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL(teacher || student), reduced over valid tokens."""
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError("student_logits and teacher_logits must have identical shapes.")
    if mask is None:
        resolved_mask = torch.ones(
            student_logits.shape[:-1], device=student_logits.device, dtype=torch.float32
        )
    else:
        if tuple(mask.shape) != tuple(student_logits.shape[:-1]):
            raise ValueError("mask shape must match logits batch/sequence dimensions.")
        resolved_mask = mask.to(device=student_logits.device, dtype=torch.float32)
    student_log_prob = F.log_softmax(student_logits.float() / temp, dim=-1)
    teacher_prob = F.softmax(teacher_logits.detach().float() / temp, dim=-1)
    per_token = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    return (per_token * resolved_mask).sum() / resolved_mask.sum().clamp_min(1.0) * (temp * temp)


__all__ = [
    "DistillTokenRegions",
    "build_distill_token_mask",
    "build_distill_token_regions",
    "compute_forward_kl_loss",
]
