from typing import Optional

import math

import torch
import torch.nn.functional as F


DEFAULT_DUAL_SCALE_EPS = 1e-6


def build_distill_token_mask(
    *,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    if reference_logits.ndim < 3:
        raise ValueError(
            f"reference_logits must have shape [B, L, V], got ndim={reference_logits.ndim}"
        )

    expected_shape = tuple(int(dim) for dim in reference_logits.shape[:2])
    mask_tensor: Optional[torch.Tensor] = None

    if isinstance(labels, torch.Tensor):
        mask_tensor = labels.ne(-100)
    elif isinstance(attention_mask, torch.Tensor):
        mask_tensor = attention_mask.ne(0)

    if mask_tensor is None:
        return torch.ones(
            expected_shape,
            dtype=torch.float32,
            device=reference_logits.device,
        )

    if tuple(int(dim) for dim in mask_tensor.shape) != expected_shape:
        raise ValueError(
            f"mask shape mismatch: expected {expected_shape}, got {tuple(mask_tensor.shape)}"
        )

    return mask_tensor.to(device=reference_logits.device, dtype=torch.float32)


def _dual_scale_logits(logits: torch.Tensor, eps: float) -> torch.Tensor:
    logits_fp32 = logits.float()
    scale = logits_fp32.std(dim=-1, keepdim=True, unbiased=False)
    return logits_fp32 / (scale + float(eps))


def _masked_token_kl_mean(
    *,
    student_log_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    token_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
    if mask is None:
        denom = torch.tensor(
            float(token_kl.numel()),
            dtype=token_kl.dtype,
            device=token_kl.device,
        ).clamp_min(1.0)
        return token_kl.sum() / denom

    mask_fp32 = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    denom = mask_fp32.sum().clamp_min(1.0)
    return (token_kl * mask_fp32).sum() / denom


def _resolve_distill_temperature(temperature: float) -> float:
    return max(float(temperature), 0.1)


def compute_teacher_entropy_gamma(
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    confidence_k: int = 16,
) -> torch.Tensor:
    resolved_k = int(confidence_k)
    if resolved_k < 2:
        raise ValueError(f"confidence_k must be >= 2, got {resolved_k}.")

    teacher_probs = F.softmax(teacher_logits.detach().float(), dim=-1)
    entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-8))).sum(dim=-1)
    max_entropy = math.log(float(resolved_k))

    if mask is not None:
        mask_bool = mask.to(device=entropy.device, dtype=torch.bool)
        entropy = entropy.masked_fill(~mask_bool, 0.0)
        valid_lengths = mask.to(device=entropy.device, dtype=torch.float32).sum(dim=-1).clamp_min(1.0)
        sample_avg_entropy = entropy.sum(dim=-1) / valid_lengths
    else:
        sample_avg_entropy = entropy.mean(dim=-1)

    normalized_entropy = sample_avg_entropy / float(max_entropy)
    gamma = 1.0 - normalized_entropy.mean()
    return gamma.clamp(0.0, 1.0)


def compute_forward_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    temp = _resolve_distill_temperature(temperature)
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    kl = _masked_token_kl_mean(
        student_log_prob=F.log_softmax(student_scaled, dim=-1),
        teacher_prob=F.softmax(teacher_scaled, dim=-1),
        mask=mask,
    )
    return kl * (temp * temp)


def compute_reverse_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    temp = _resolve_distill_temperature(temperature)
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    kl = _masked_token_kl_mean(
        student_log_prob=F.log_softmax(teacher_scaled, dim=-1),
        teacher_prob=F.softmax(student_scaled, dim=-1),
        mask=mask,
    )
    return kl * (temp * temp)


def compute_entropy_aware_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
    confidence_k: int = 16,
) -> torch.Tensor:
    gamma = compute_teacher_entropy_gamma(
        teacher_logits,
        mask,
        confidence_k=int(confidence_k),
    )
    reverse_kl = compute_reverse_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=float(temperature),
    )
    forward_kl = compute_forward_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=float(temperature),
    )
    return gamma * reverse_kl + (1.0 - gamma) * forward_kl


def compute_dual_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    eps: float = DEFAULT_DUAL_SCALE_EPS,
) -> torch.Tensor:
    teacher_scaled = _dual_scale_logits(teacher_logits, eps=float(eps))
    student_scaled = _dual_scale_logits(student_logits, eps=float(eps))
    return _masked_token_kl_mean(
        student_log_prob=F.log_softmax(student_scaled, dim=-1),
        teacher_prob=F.softmax(teacher_scaled, dim=-1),
        mask=mask,
    )


def compute_dual_rkl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    eps: float = DEFAULT_DUAL_SCALE_EPS,
) -> torch.Tensor:
    teacher_scaled = _dual_scale_logits(teacher_logits, eps=float(eps))
    student_scaled = _dual_scale_logits(student_logits, eps=float(eps))
    return _masked_token_kl_mean(
        student_log_prob=F.log_softmax(teacher_scaled, dim=-1),
        teacher_prob=F.softmax(student_scaled, dim=-1),
        mask=mask,
    )


def compute_dual_kl_topk_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    post_attn: bool = False,
    eps: float = DEFAULT_DUAL_SCALE_EPS,
) -> torch.Tensor:
    resolved_k = int(k)
    if resolved_k <= 0:
        raise ValueError(f"k must be > 0, got {resolved_k}")

    teacher_logits_fp32 = teacher_logits.float()
    student_scaled = _dual_scale_logits(student_logits, eps=float(eps))
    teacher_scaled = _dual_scale_logits(teacher_logits_fp32, eps=float(eps))

    resolved_k = min(resolved_k, int(teacher_logits_fp32.shape[-1]))
    _, indices = teacher_logits_fp32.topk(resolved_k, dim=-1, sorted=False)
    if bool(post_attn):
        top_teacher_prob = F.softmax(teacher_scaled, dim=-1).gather(-1, indices)
        top_student_log_prob = F.log_softmax(student_scaled, dim=-1).gather(-1, indices)
        return _masked_token_kl_mean(
            student_log_prob=top_student_log_prob,
            teacher_prob=top_teacher_prob,
            mask=mask,
        )

    top_teacher_scaled = teacher_scaled.gather(-1, indices)
    top_student_scaled = student_scaled.gather(-1, indices)
    return _masked_token_kl_mean(
        student_log_prob=F.log_softmax(top_student_scaled, dim=-1),
        teacher_prob=F.softmax(top_teacher_scaled, dim=-1),
        mask=mask,
    )


def compute_dual_rkl_topk_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    post_attn: bool = False,
    eps: float = DEFAULT_DUAL_SCALE_EPS,
) -> torch.Tensor:
    resolved_k = int(k)
    if resolved_k <= 0:
        raise ValueError(f"k must be > 0, got {resolved_k}")

    student_logits_fp32 = student_logits.float()
    student_scaled = _dual_scale_logits(student_logits_fp32, eps=float(eps))
    teacher_scaled = _dual_scale_logits(teacher_logits, eps=float(eps))

    resolved_k = min(resolved_k, int(student_logits_fp32.shape[-1]))
    _, indices = student_logits_fp32.topk(resolved_k, dim=-1, sorted=False)
    if bool(post_attn):
        top_student_prob = F.softmax(student_scaled, dim=-1).gather(-1, indices)
        top_teacher_log_prob = F.log_softmax(teacher_scaled, dim=-1).gather(-1, indices)
        return _masked_token_kl_mean(
            student_log_prob=top_teacher_log_prob,
            teacher_prob=top_student_prob,
            mask=mask,
        )

    top_student_scaled = student_scaled.gather(-1, indices)
    top_teacher_scaled = teacher_scaled.gather(-1, indices)
    return _masked_token_kl_mean(
        student_log_prob=F.log_softmax(top_teacher_scaled, dim=-1),
        teacher_prob=F.softmax(top_student_scaled, dim=-1),
        mask=mask,
    )
