from typing import Optional

import math

import torch
import torch.nn.functional as F


DEFAULT_DUAL_SCALE_EPS = 1e-6
DEFAULT_TEACHER_ENTROPY_SEQUENCE_CHUNK_SIZE = 16


def parse_eakld_top_k(value: str, *, default_k: int = 1000) -> int:
    norm = str(value).strip().lower()
    if norm in {"eakld_topk", "eakld_top"}:
        return int(default_k)
    if norm.startswith("eakld_topk_"):
        return max(1, int(norm[len("eakld_topk_") :]))
    if norm.startswith("eakld_top_"):
        return max(1, int(norm[len("eakld_top_") :]))
    raise ValueError(f"Unsupported eakld top-k loss type: {value}")


def is_eakld_top_loss(value: str) -> bool:
    norm = str(value).strip().lower()
    return (
        norm in {"eakld_topk", "eakld_top"}
        or norm.startswith("eakld_topk_")
        or norm.startswith("eakld_top_")
    )


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


def _default_token_mask(reference: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        return mask.to(device=reference.device, dtype=torch.float32)
    return torch.ones(reference.shape[:2], device=reference.device, dtype=torch.float32)


def _dual_scale_logits(logits: torch.Tensor, eps: float) -> torch.Tensor:
    logits_fp32 = logits.float()
    scale = logits_fp32.std(dim=-1, keepdim=True, unbiased=False)
    return logits_fp32 / (scale + float(eps))


def _resolve_distill_temperature(temperature: float) -> float:
    return max(float(temperature), 0.1)


def _masked_token_kl_mean(
    *,
    student_log_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Forward KL: KL(teacher || student); student log_prob is differentiable."""
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


def _masked_token_reverse_kl_mean(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reverse KL: KL(student || teacher) = Σ p_s (log p_s - log p_t).

    Do not use kl_div(log_t, softmax(s)): PyTorch does not differentiate target.
    """
    log_s = F.log_softmax(student_logits, dim=-1)
    log_t = F.log_softmax(teacher_logits, dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    if mask is None:
        return token_kl.mean()
    mask_fp32 = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    denom = mask_fp32.sum().clamp_min(1.0)
    return (token_kl * mask_fp32).sum() / denom


def accumulate_teacher_entropy_stats(
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    sequence_chunk_size: int = DEFAULT_TEACHER_ENTROPY_SEQUENCE_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_logits.ndim != 3:
        raise ValueError(
            "teacher_logits must have shape [B, L, V], "
            f"got shape={tuple(teacher_logits.shape)}"
        )

    expected_mask_shape = tuple(int(dim) for dim in teacher_logits.shape[:2])
    if tuple(int(dim) for dim in mask.shape) != expected_mask_shape:
        raise ValueError(
            f"mask shape mismatch: expected {expected_mask_shape}, got {tuple(mask.shape)}"
        )

    resolved_chunk_size = int(sequence_chunk_size)
    if resolved_chunk_size <= 0:
        raise ValueError(
            f"sequence_chunk_size must be > 0, got {sequence_chunk_size}"
        )

    total_entropy = torch.zeros(
        (),
        device=teacher_logits.device,
        dtype=torch.float32,
    )
    total_valid = torch.zeros(
        (),
        device=teacher_logits.device,
        dtype=torch.float32,
    )

    sequence_length = int(teacher_logits.shape[1])
    with torch.no_grad():
        for start in range(0, sequence_length, resolved_chunk_size):
            end = min(sequence_length, start + resolved_chunk_size)
            logits_chunk = teacher_logits[:, start:end, :].detach().float()
            teacher_probs = F.softmax(logits_chunk, dim=-1)
            log_probs = teacher_probs.clamp_min(1e-8).log_()
            teacher_probs.mul_(log_probs)
            entropy = -teacher_probs.sum(dim=-1)
            mask_chunk = mask[:, start:end].to(
                device=entropy.device,
                dtype=torch.float32,
            )

            total_entropy.add_((entropy * mask_chunk).sum())
            total_valid.add_(mask_chunk.sum())

            del logits_chunk, teacher_probs, log_probs, entropy, mask_chunk

    return total_entropy, total_valid


def gamma_from_entropy_sums(
    sum_entropy: torch.Tensor,
    sum_valid: torch.Tensor,
    *,
    confidence_k: int = 16,
) -> torch.Tensor:
    resolved_k = int(confidence_k)
    if resolved_k < 2:
        raise ValueError(f"confidence_k must be >= 2, got {resolved_k}.")
    max_entropy = math.log(float(resolved_k))
    avg = sum_entropy / sum_valid.clamp_min(1.0)
    return (1.0 - avg / float(max_entropy)).clamp(0.0, 1.0)


def compute_teacher_entropy_gamma(
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    confidence_k: int = 16,
) -> torch.Tensor:
    """Global token-mean teacher entropy -> gamma (aligned with loss.py)."""
    resolved_mask = _default_token_mask(teacher_logits, mask)
    sum_e, sum_v = accumulate_teacher_entropy_stats(teacher_logits, resolved_mask)
    return gamma_from_entropy_sums(sum_e, sum_v, confidence_k=int(confidence_k))


def _topk_forward_kl_mean(
    *,
    student_scaled: torch.Tensor,
    teacher_scaled: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    post_attn: bool,
) -> torch.Tensor:
    """Forward KL on teacher top-k. Inputs must already be temperature-scaled."""
    resolved_k = min(int(k), int(student_scaled.shape[-1]))
    _, indices = teacher_scaled.topk(resolved_k, dim=-1, sorted=False)
    if bool(post_attn):
        teacher_prob = F.softmax(teacher_scaled, dim=-1).gather(-1, indices)
        student_log_prob = F.log_softmax(student_scaled, dim=-1).gather(-1, indices)
    else:
        teacher_prob = F.softmax(teacher_scaled.gather(-1, indices), dim=-1)
        student_log_prob = F.log_softmax(student_scaled.gather(-1, indices), dim=-1)
    return _masked_token_kl_mean(
        student_log_prob=student_log_prob,
        teacher_prob=teacher_prob,
        mask=mask,
    )


def _topk_reverse_kl_mean(
    *,
    student_scaled: torch.Tensor,
    teacher_scaled: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    post_attn: bool,
) -> torch.Tensor:
    """Reverse KL on student top-k. Inputs must already be temperature-scaled."""
    resolved_k = min(int(k), int(student_scaled.shape[-1]))
    _, indices = student_scaled.topk(resolved_k, dim=-1, sorted=False)
    if bool(post_attn):
        log_s = F.log_softmax(student_scaled, dim=-1).gather(-1, indices)
        log_t = F.log_softmax(teacher_scaled, dim=-1).gather(-1, indices)
    else:
        log_s = F.log_softmax(student_scaled.gather(-1, indices), dim=-1)
        log_t = F.log_softmax(teacher_scaled.gather(-1, indices), dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    mask_fp = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    denom = mask_fp.sum().clamp_min(1.0)
    return (token_kl * mask_fp).sum() / denom


def compute_forward_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL(teacher || student), global token mean."""
    temp = _resolve_distill_temperature(temperature)
    resolved_mask = _default_token_mask(student_logits, mask)
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    kl = _masked_token_kl_mean(
        student_log_prob=F.log_softmax(student_scaled, dim=-1),
        teacher_prob=F.softmax(teacher_scaled, dim=-1),
        mask=resolved_mask,
    )
    return kl * (temp * temp)


def compute_reverse_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reverse KL(student || teacher), global token mean, student-differentiable."""
    temp = _resolve_distill_temperature(temperature)
    resolved_mask = _default_token_mask(student_logits, mask)
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    kl = _masked_token_reverse_kl_mean(
        student_logits=student_scaled,
        teacher_logits=teacher_scaled,
        mask=resolved_mask,
    )
    return kl * (temp * temp)


def compute_eakld(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
    confidence_k: int = 16,
) -> torch.Tensor:
    resolved_mask = _default_token_mask(student_logits, mask)
    temp = _resolve_distill_temperature(temperature)
    sum_e, sum_v = accumulate_teacher_entropy_stats(teacher_logits, resolved_mask)
    gamma = gamma_from_entropy_sums(sum_e, sum_v, confidence_k=int(confidence_k))
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    reverse_kl = _masked_token_reverse_kl_mean(
        student_logits=student_scaled,
        teacher_logits=teacher_scaled,
        mask=resolved_mask,
    ) * (temp * temp)
    forward_kl = _masked_token_kl_mean(
        student_log_prob=F.log_softmax(student_scaled, dim=-1),
        teacher_prob=F.softmax(teacher_scaled, dim=-1),
        mask=resolved_mask,
    ) * (temp * temp)
    return gamma * reverse_kl + (1.0 - gamma) * forward_kl


def compute_entropy_aware_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
    confidence_k: int = 16,
) -> torch.Tensor:
    """Backward-compatible alias for compute_eakld."""
    return compute_eakld(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=float(temperature),
        confidence_k=int(confidence_k),
    )


def compute_kl_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    post_attn: bool = False,
) -> torch.Tensor:
    """Forward KL on teacher top-k."""
    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    resolved_mask = _default_token_mask(student_logits, mask)
    temp = _resolve_distill_temperature(temperature)
    return _topk_forward_kl_mean(
        student_scaled=student_logits.float() / temp,
        teacher_scaled=teacher_logits.detach().float() / temp,
        mask=resolved_mask,
        k=int(k),
        post_attn=bool(post_attn),
    ) * (temp * temp)


def compute_rkl_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    post_attn: bool = False,
) -> torch.Tensor:
    """Reverse KL on student top-k."""
    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    resolved_mask = _default_token_mask(student_logits, mask)
    temp = _resolve_distill_temperature(temperature)
    return _topk_reverse_kl_mean(
        student_scaled=student_logits.float() / temp,
        teacher_scaled=teacher_logits.detach().float() / temp,
        mask=resolved_mask,
        k=int(k),
        post_attn=bool(post_attn),
    ) * (temp * temp)


def compute_eakld_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    confidence_k: int = 16,
    post_attn: bool = False,
) -> torch.Tensor:
    """EAKLD top-k: gamma from full-vocab teacher entropy; FKL/RKL use teacher/student top-k."""
    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    resolved_mask = _default_token_mask(student_logits, mask)
    temp = _resolve_distill_temperature(temperature)

    sum_e, sum_v = accumulate_teacher_entropy_stats(teacher_logits, resolved_mask)
    gamma = gamma_from_entropy_sums(sum_e, sum_v, confidence_k=int(confidence_k))

    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    reverse_kl = _topk_reverse_kl_mean(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=resolved_mask,
        k=int(k),
        post_attn=bool(post_attn),
    ) * (temp * temp)
    forward_kl = _topk_forward_kl_mean(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=resolved_mask,
        k=int(k),
        post_attn=bool(post_attn),
    ) * (temp * temp)
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
    return _masked_token_reverse_kl_mean(
        student_logits=student_scaled,
        teacher_logits=teacher_scaled,
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
        log_s = F.log_softmax(student_scaled, dim=-1).gather(-1, indices)
        log_t = F.log_softmax(teacher_scaled, dim=-1).gather(-1, indices)
    else:
        log_s = F.log_softmax(student_scaled.gather(-1, indices), dim=-1)
        log_t = F.log_softmax(teacher_scaled.gather(-1, indices), dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    if mask is None:
        return token_kl.mean()
    mask_fp32 = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    denom = mask_fp32.sum().clamp_min(1.0)
    return (token_kl * mask_fp32).sum() / denom
