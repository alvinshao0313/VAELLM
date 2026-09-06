"""Shared model-level distill loss math for CAT and E2E trainers."""

from __future__ import annotations

import math
from typing import Optional, Tuple  # Tuple kept for mask helpers

import torch
import torch.nn.functional as F

MODEL_LEVEL_LOSS_TYPES = ("sft", "kl", "kl_top", "kd", "kd_top")


def _require_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}.")
    return value


def build_prediction_token_masks(
    *,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("labels and attention_mask must have shape [B, L].")
    if tuple(labels.shape) != tuple(attention_mask.shape):
        raise ValueError(
            "labels/attention_mask shape mismatch: "
            f"{tuple(labels.shape)} vs {tuple(attention_mask.shape)}."
        )
    if int(labels.shape[1]) < 2:
        raise ValueError("sequence length must be >= 2 for causal next-token loss.")

    target_labels = labels[:, 1:]
    target_attention = attention_mask[:, 1:].to(dtype=torch.bool).ne(0)
    response_mask = target_labels.ne(-100) & target_attention
    prompt_mask = target_labels.eq(-100) & target_attention
    return (
        response_mask.to(dtype=torch.float32),
        prompt_mask.to(dtype=torch.float32),
    )


def reduce_weighted_token_loss(
    token_loss: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    prompt_loss_weight: float,
) -> torch.Tensor:
    if token_loss.ndim != 2:
        raise ValueError(f"token_loss must have shape [B, L-1], got ndim={token_loss.ndim}.")
    if tuple(token_loss.shape) != tuple(response_mask.shape) or tuple(token_loss.shape) != tuple(
        prompt_mask.shape
    ):
        raise ValueError(
            "token_loss/response_mask/prompt_mask shape mismatch: "
            f"{tuple(token_loss.shape)} / {tuple(response_mask.shape)} / {tuple(prompt_mask.shape)}."
        )
    weight = float(prompt_loss_weight)
    if weight < 0.0:
        raise ValueError(f"prompt_loss_weight must be >= 0, got {prompt_loss_weight}.")
    loss_fp = token_loss.float()
    weights = response_mask.float() + weight * prompt_mask.float()
    denom = weights.sum().clamp_min(1.0)
    return (loss_fp * weights).sum() / denom


def compute_sft_token_loss(
    *,
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    if student_logits.ndim != 3:
        raise ValueError(f"student_logits must have shape [B, L, V], got ndim={student_logits.ndim}.")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, L], got ndim={input_ids.ndim}.")
    if int(student_logits.shape[0]) != int(input_ids.shape[0]) or int(student_logits.shape[1]) != int(
        input_ids.shape[1]
    ):
        raise ValueError(
            "student_logits/input_ids batch/length mismatch: "
            f"{tuple(student_logits.shape[:2])} vs {tuple(input_ids.shape)}."
        )
    if int(student_logits.shape[1]) < 2:
        raise ValueError("sequence length must be >= 2 for causal next-token loss.")

    shift_logits = student_logits[:, :-1, :].float().contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    vocab = int(shift_logits.shape[-1])
    return F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)


def compute_kl_token_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}."
        )
    temp = _require_temperature(temperature)
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    teacher_prob = F.softmax(teacher_scaled, dim=-1)
    student_log_prob = F.log_softmax(student_scaled, dim=-1)
    token_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    return token_kl * (temp * temp)


def compute_kl_top_token_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}."
        )
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}.")
    temp = _require_temperature(temperature)
    vocab = int(student_logits.shape[-1])
    k_eff = min(int(top_k), vocab)
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    _, indices = teacher_scaled.topk(k_eff, dim=-1, sorted=False)
    teacher_prob = F.softmax(teacher_scaled.gather(-1, indices), dim=-1)
    student_log_prob = F.log_softmax(student_scaled.gather(-1, indices), dim=-1)
    token_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    return token_kl * (temp * temp)


def compute_selected_kl_top_token_loss(
    *,
    student_selected_logits: torch.Tensor,
    teacher_selected_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Token-wise teacher-topK KL when logits are already gathered to K."""
    if tuple(student_selected_logits.shape) != tuple(teacher_selected_logits.shape):
        raise ValueError(
            "selected student/teacher top-k shape mismatch: "
            f"{tuple(student_selected_logits.shape)} vs {tuple(teacher_selected_logits.shape)}."
        )
    temp = _require_temperature(temperature)
    student_log_prob = F.log_softmax(student_selected_logits.float() / temp, dim=-1)
    teacher_prob = F.softmax(teacher_selected_logits.detach().float() / temp, dim=-1)
    token_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    return token_kl * (temp * temp)


def compute_selected_kl_top_model_level_loss(
    *,
    student_selected_logits: torch.Tensor,
    teacher_selected_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
    prompt_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Model-level reduction for selective_student_topk kl_top path."""
    response_mask, prompt_mask = build_prediction_token_masks(
        labels=labels,
        attention_mask=attention_mask,
    )
    if int(student_selected_logits.shape[1]) < 2:
        raise ValueError("sequence length must be >= 2 for causal next-token loss.")
    token_loss = compute_selected_kl_top_token_loss(
        student_selected_logits=student_selected_logits[:, :-1, :],
        teacher_selected_logits=teacher_selected_logits[:, :-1, :],
        temperature=temperature,
    )
    return reduce_weighted_token_loss(
        token_loss,
        response_mask=response_mask,
        prompt_mask=prompt_mask,
        prompt_loss_weight=prompt_loss_weight,
    )


def normalize_model_level_loss_type(loss_type: str) -> str:
    """Return exact canonical type. Rejects deleted types and suffix encodings.

    Shared core accepts only ``sft|kl|kl_top|kd|kd_top``. Encoded forms such as
    ``kl_top_100`` / ``kd_top_100`` must be split at a legacy parser/wrapper
    caller before invoking the shared core.
    """
    norm = str(loss_type or "").strip().lower()
    if norm in MODEL_LEVEL_LOSS_TYPES:
        return norm
    raise ValueError(
        f"Unsupported model-level loss_type={loss_type!r}. "
        f"Supported: {', '.join(MODEL_LEVEL_LOSS_TYPES)}. "
        "Do not encode top-k in the type string; pass top_k separately. "
        "Legacy kl_top_<K>/kd_top_<K> must be parsed only at the legacy wrapper boundary."
    )


def compute_model_level_loss(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
    top_k: int = 100,
    prompt_loss_weight: float = 0.0,
) -> torch.Tensor:
    norm = normalize_model_level_loss_type(loss_type)
    resolved_top_k = int(top_k)

    response_mask, prompt_mask = build_prediction_token_masks(
        labels=labels,
        attention_mask=attention_mask,
    )
    pred_student = student_logits[:, :-1, :]
    pred_teacher = None if teacher_logits is None else teacher_logits[:, :-1, :]

    if norm == "sft":
        token_loss = compute_sft_token_loss(student_logits=student_logits, input_ids=input_ids)
        return reduce_weighted_token_loss(
            token_loss,
            response_mask=response_mask,
            prompt_mask=prompt_mask,
            prompt_loss_weight=prompt_loss_weight,
        )

    if teacher_logits is None or pred_teacher is None:
        raise ValueError(f"loss_type={norm} requires teacher_logits.")

    if norm == "kl":
        token_loss = compute_kl_token_loss(
            student_logits=pred_student,
            teacher_logits=pred_teacher,
            temperature=temperature,
        )
        return reduce_weighted_token_loss(
            token_loss,
            response_mask=response_mask,
            prompt_mask=prompt_mask,
            prompt_loss_weight=prompt_loss_weight,
        )

    if norm == "kl_top":
        token_loss = compute_kl_top_token_loss(
            student_logits=pred_student,
            teacher_logits=pred_teacher,
            temperature=temperature,
            top_k=resolved_top_k,
        )
        return reduce_weighted_token_loss(
            token_loss,
            response_mask=response_mask,
            prompt_mask=prompt_mask,
            prompt_loss_weight=prompt_loss_weight,
        )

    alpha_f = float(alpha)
    if alpha_f < 0.0 or alpha_f > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    ce_token = compute_sft_token_loss(student_logits=student_logits, input_ids=input_ids)
    if norm == "kd":
        kl_token = compute_kl_token_loss(
            student_logits=pred_student,
            teacher_logits=pred_teacher,
            temperature=temperature,
        )
    else:
        kl_token = compute_kl_top_token_loss(
            student_logits=pred_student,
            teacher_logits=pred_teacher,
            temperature=temperature,
            top_k=resolved_top_k,
        )
    ce = reduce_weighted_token_loss(
        ce_token,
        response_mask=response_mask,
        prompt_mask=prompt_mask,
        prompt_loss_weight=prompt_loss_weight,
    )
    kl = reduce_weighted_token_loss(
        kl_token,
        response_mask=response_mask,
        prompt_mask=prompt_mask,
        prompt_loss_weight=prompt_loss_weight,
    )
    return (1.0 - alpha_f) * ce + alpha_f * kl
