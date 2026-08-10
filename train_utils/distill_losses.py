from dataclasses import dataclass
from typing import MutableMapping, Optional

import math

import torch
import torch.nn.functional as F
from torch.utils import checkpoint as torch_checkpoint

from compressed_e2e_fintuning.teacher_targets import (
    copy_teacher_logit_chunk_to_device,
    iter_token_chunk_ranges,
)


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


@dataclass(frozen=True)
class DistillTokenRegions:
    response_mask: torch.Tensor
    prompt_mask: torch.Tensor


def _validate_distill_mask_shape(
    *,
    expected_shape: tuple[int, ...],
    tensor: torch.Tensor,
    tensor_name: str,
) -> None:
    if tuple(int(dim) for dim in tensor.shape) != expected_shape:
        raise ValueError(
            f"mask shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}"
        )


def _apply_causal_shift(source_weights: torch.Tensor) -> torch.Tensor:
    expected_shape = tuple(int(dim) for dim in source_weights.shape)
    device = source_weights.device
    causal_mask = torch.zeros(
        expected_shape,
        dtype=torch.float32,
        device=device,
    )
    sequence_length = int(expected_shape[1])
    if sequence_length > 1:
        causal_mask[:, :-1] = source_weights[:, 1:]
    return causal_mask


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
    device = reference_logits.device
    source_weights: Optional[torch.Tensor] = None

    if isinstance(labels, torch.Tensor):
        _validate_distill_mask_shape(
            expected_shape=expected_shape,
            tensor=labels,
            tensor_name="labels",
        )
        labels = labels.to(device=device)
        source_weights = labels.ne(-100).to(dtype=torch.float32)
    elif isinstance(attention_mask, torch.Tensor):
        _validate_distill_mask_shape(
            expected_shape=expected_shape,
            tensor=attention_mask,
            tensor_name="attention_mask",
        )
        source_weights = attention_mask.to(device=device, dtype=torch.bool).ne(0).to(
            dtype=torch.float32
        )
    else:
        source_weights = torch.ones(
            expected_shape,
            dtype=torch.float32,
            device=device,
        )

    return _apply_causal_shift(source_weights)


def build_distill_token_regions(
    *,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    reference_logits: torch.Tensor,
) -> DistillTokenRegions:
    response_mask = build_distill_token_mask(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )
    expected_shape = tuple(int(dim) for dim in reference_logits.shape[:2])
    device = reference_logits.device

    if isinstance(labels, torch.Tensor):
        _validate_distill_mask_shape(
            expected_shape=expected_shape,
            tensor=labels,
            tensor_name="labels",
        )
        labels = labels.to(device=device)
        prompt_validity = labels.eq(-100)
        if isinstance(attention_mask, torch.Tensor):
            _validate_distill_mask_shape(
                expected_shape=expected_shape,
                tensor=attention_mask,
                tensor_name="attention_mask",
            )
            attention_validity = attention_mask.to(
                device=device,
                dtype=torch.bool,
            ).ne(0)
            prompt_validity = prompt_validity & attention_validity
        prompt_source = prompt_validity.to(dtype=torch.float32)
        prompt_mask = _apply_causal_shift(prompt_source)
    else:
        prompt_mask = torch.zeros(
            expected_shape,
            dtype=torch.float32,
            device=device,
        )

    return DistillTokenRegions(
        response_mask=response_mask,
        prompt_mask=prompt_mask,
    )


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


def compute_teacher_entropy_mean_and_gamma(
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    confidence_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        teacher_entropy_mean: scalar fp32
        gamma_reverse: scalar fp32
        valid_count: scalar fp32
    """
    resolved_mask = _default_token_mask(teacher_logits, mask)
    entropy_sum, valid_count = accumulate_teacher_entropy_stats(
        teacher_logits,
        resolved_mask,
    )
    entropy_mean = entropy_sum / valid_count.clamp_min(1.0)
    gamma_reverse = gamma_from_entropy_sums(
        entropy_sum,
        valid_count,
        confidence_k=confidence_k,
    )
    return entropy_mean, gamma_reverse, valid_count


def compute_teacher_entropy_gamma(
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    confidence_k: int = 16,
) -> torch.Tensor:
    """Global token-mean teacher entropy -> gamma (aligned with loss.py)."""
    _entropy_mean, gamma_reverse, _valid_count = compute_teacher_entropy_mean_and_gamma(
        teacher_logits,
        mask,
        confidence_k=int(confidence_k),
    )
    return gamma_reverse


def _write_eakld_telemetry(
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]],
    *,
    teacher_entropy_mean: torch.Tensor,
    gamma_reverse: torch.Tensor,
    forward_kl: torch.Tensor,
    reverse_kl: torch.Tensor,
    eakld_total: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> None:
    if telemetry_out is None:
        return
    gamma_scalar = gamma_reverse.detach().reshape(()).to(dtype=torch.float32)
    telemetry_out.update(
        {
            "teacher_entropy_mean": teacher_entropy_mean.detach().reshape(()).to(
                dtype=torch.float32
            ),
            "gamma_reverse": gamma_scalar,
            "lambda_forward": (1.0 - gamma_scalar).detach().reshape(()).to(
                dtype=torch.float32
            ),
            "forward_kl": forward_kl.detach().reshape(()).to(dtype=torch.float32),
            "reverse_kl": reverse_kl.detach().reshape(()).to(dtype=torch.float32),
            "eakld_total": eakld_total.detach().reshape(()).to(dtype=torch.float32),
            "valid_tokens": valid_tokens.detach().reshape(()).to(dtype=torch.float32),
        }
    )


def _topk_forward_kl_mean(
    *,
    student_scaled: torch.Tensor,
    teacher_scaled: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Forward KL on teacher top-k. Inputs must already be temperature-scaled."""
    resolved_k = min(int(k), int(student_scaled.shape[-1]))
    _, indices = teacher_scaled.topk(resolved_k, dim=-1, sorted=False)
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
) -> torch.Tensor:
    """Reverse KL on student top-k. Inputs must already be temperature-scaled."""
    resolved_k = min(int(k), int(student_scaled.shape[-1]))
    _, indices = student_scaled.topk(resolved_k, dim=-1, sorted=False)
    log_s = F.log_softmax(student_scaled.gather(-1, indices), dim=-1)
    log_t = F.log_softmax(teacher_scaled.gather(-1, indices), dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    mask_fp = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    denom = mask_fp.sum().clamp_min(1.0)
    return (token_kl * mask_fp).sum() / denom


def compute_masked_logit_mse_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-token vocab-mean MSE, then masked global token mean. No temperature."""
    student_f = student_logits.float()
    teacher_f = teacher_logits.detach().float()
    token_mse = (student_f - teacher_f).pow(2).mean(dim=-1)
    mask_f = _default_token_mask(token_mse, mask)
    numerator = (token_mse * mask_f).sum()
    denominator = mask_f.sum()
    return numerator / denominator.clamp_min(1.0)


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
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    resolved_mask = _default_token_mask(student_logits, mask)
    temp = _resolve_distill_temperature(temperature)
    entropy_mean, gamma, valid_count = compute_teacher_entropy_mean_and_gamma(
        teacher_logits,
        resolved_mask,
        confidence_k=int(confidence_k),
    )
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
    eakld_total = gamma * reverse_kl + (1.0 - gamma) * forward_kl
    _write_eakld_telemetry(
        telemetry_out,
        teacher_entropy_mean=entropy_mean,
        gamma_reverse=gamma,
        forward_kl=forward_kl,
        reverse_kl=reverse_kl,
        eakld_total=eakld_total,
        valid_tokens=valid_count,
    )
    return eakld_total


def compute_entropy_aware_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float = 1.0,
    confidence_k: int = 16,
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Backward-compatible alias for compute_eakld."""
    return compute_eakld(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=float(temperature),
        confidence_k=int(confidence_k),
        telemetry_out=telemetry_out,
    )


def compute_kl_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
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
    ) * (temp * temp)


def compute_rkl_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
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
    ) * (temp * temp)


def compute_eakld_topk(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    confidence_k: int = 16,
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """EAKLD top-k: gamma from full-vocab teacher entropy; FKL/RKL use teacher/student top-k."""
    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    resolved_mask = _default_token_mask(student_logits, mask)
    temp = _resolve_distill_temperature(temperature)

    entropy_mean, gamma, valid_count = compute_teacher_entropy_mean_and_gamma(
        teacher_logits,
        resolved_mask,
        confidence_k=int(confidence_k),
    )

    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    reverse_kl = _topk_reverse_kl_mean(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=resolved_mask,
        k=int(k),
    ) * (temp * temp)
    forward_kl = _topk_forward_kl_mean(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=resolved_mask,
        k=int(k),
    ) * (temp * temp)
    eakld_total = gamma * reverse_kl + (1.0 - gamma) * forward_kl
    _write_eakld_telemetry(
        telemetry_out,
        teacher_entropy_mean=entropy_mean,
        gamma_reverse=gamma,
        forward_kl=forward_kl,
        reverse_kl=reverse_kl,
        eakld_total=eakld_total,
        valid_tokens=valid_count,
    )
    return eakld_total



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
    log_s = F.log_softmax(student_scaled.gather(-1, indices), dim=-1)
    log_t = F.log_softmax(teacher_scaled.gather(-1, indices), dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    if mask is None:
        return token_kl.mean()
    mask_fp32 = mask.to(device=token_kl.device, dtype=token_kl.dtype)
    denom = mask_fp32.sum().clamp_min(1.0)
    return (token_kl * mask_fp32).sum() / denom


def _validate_cpu_teacher_distill_inputs(
    *,
    student_logits: torch.Tensor,
    teacher_logits_cpu: torch.Tensor,
    gamma: torch.Tensor,
    sequence_chunk_size: int,
) -> None:
    if student_logits.ndim != 3:
        raise ValueError(
            "student_logits must have shape [B, L, V], "
            f"got {tuple(student_logits.shape)}."
        )
    if teacher_logits_cpu.device.type != "cpu":
        raise ValueError("teacher_logits_cpu must reside on CPU.")
    if teacher_logits_cpu.ndim != 3:
        raise ValueError(
            "teacher_logits_cpu must have shape [B, L, V], "
            f"got {tuple(teacher_logits_cpu.shape)}."
        )
    if tuple(student_logits.shape) != tuple(teacher_logits_cpu.shape):
        raise ValueError(
            "student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits_cpu.shape)}."
        )
    if int(sequence_chunk_size) < 1:
        raise ValueError(
            f"sequence_chunk_size must be >= 1, got {sequence_chunk_size}."
        )
    if not torch.is_tensor(gamma) or int(gamma.numel()) != 1:
        raise ValueError("gamma must be a scalar tensor.")


def _full_eakld_chunk_sums(
    *,
    student_logits_chunk: torch.Tensor,
    teacher_logits_chunk: torch.Tensor,
    mask_chunk: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temp = _resolve_distill_temperature(temperature)
    student_scaled = student_logits_chunk.float() / temp
    teacher_scaled = teacher_logits_chunk.detach().float() / temp

    log_student = F.log_softmax(student_scaled, dim=-1)
    log_teacher = F.log_softmax(teacher_scaled, dim=-1)

    reverse_token = (
        log_student.exp() * (log_student - log_teacher)
    ).sum(dim=-1)
    forward_token = F.kl_div(
        log_student,
        F.softmax(teacher_scaled, dim=-1),
        reduction="none",
    ).sum(dim=-1)

    mask_fp32 = mask_chunk.to(
        device=student_logits_chunk.device,
        dtype=torch.float32,
    )
    return torch.stack(
        (
            (reverse_token * mask_fp32).sum(),
            (forward_token * mask_fp32).sum(),
        )
    )


def _topk_eakld_chunk_sums(
    *,
    student_logits_chunk: torch.Tensor,
    teacher_logits_chunk: torch.Tensor,
    mask_chunk: torch.Tensor,
    k: int,
    temperature: float,
) -> torch.Tensor:
    temp = _resolve_distill_temperature(temperature)
    student_scaled = student_logits_chunk.float() / temp
    teacher_scaled = teacher_logits_chunk.detach().float() / temp
    resolved_k = min(int(k), int(student_scaled.shape[-1]))

    _, student_indices = student_scaled.topk(
        resolved_k,
        dim=-1,
        sorted=False,
    )
    reverse_log_student = F.log_softmax(
        student_scaled.gather(-1, student_indices),
        dim=-1,
    )
    reverse_log_teacher = F.log_softmax(
        teacher_scaled.gather(-1, student_indices),
        dim=-1,
    )
    reverse_token = (
        reverse_log_student.exp()
        * (reverse_log_student - reverse_log_teacher)
    ).sum(dim=-1)

    _, teacher_indices = teacher_scaled.topk(
        resolved_k,
        dim=-1,
        sorted=False,
    )
    forward_teacher_prob = F.softmax(
        teacher_scaled.gather(-1, teacher_indices),
        dim=-1,
    )
    forward_student_log_prob = F.log_softmax(
        student_scaled.gather(-1, teacher_indices),
        dim=-1,
    )
    forward_token = F.kl_div(
        forward_student_log_prob,
        forward_teacher_prob,
        reduction="none",
    ).sum(dim=-1)

    mask_fp32 = mask_chunk.to(
        device=student_logits_chunk.device,
        dtype=torch.float32,
    )
    return torch.stack(
        (
            (reverse_token * mask_fp32).sum(),
            (forward_token * mask_fp32).sum(),
        )
    )


def _make_checkpointed_eakld_chunk_forward(
    *,
    teacher_logits_cpu: torch.Tensor,
    start: int,
    end: int,
    mask_chunk: torch.Tensor,
    temperature: float,
    k: Optional[int],
):
    fixed_start = int(start)
    fixed_end = int(end)

    def chunk_forward(active_student_chunk: torch.Tensor) -> torch.Tensor:
        teacher_chunk = copy_teacher_logit_chunk_to_device(
            teacher_logits_cpu,
            start=fixed_start,
            end=fixed_end,
            target_device=active_student_chunk.device,
        )
        if k is None:
            return _full_eakld_chunk_sums(
                student_logits_chunk=active_student_chunk,
                teacher_logits_chunk=teacher_chunk,
                mask_chunk=mask_chunk,
                temperature=temperature,
            )
        return _topk_eakld_chunk_sums(
            student_logits_chunk=active_student_chunk,
            teacher_logits_chunk=teacher_chunk,
            mask_chunk=mask_chunk,
            k=int(k),
            temperature=temperature,
        )

    return chunk_forward


def _compute_eakld_from_cpu_teacher_logits_impl(
    *,
    student_logits: torch.Tensor,
    teacher_logits_cpu: torch.Tensor,
    mask: Optional[torch.Tensor],
    gamma: torch.Tensor,
    temperature: float,
    sequence_chunk_size: int,
    k: Optional[int],
    teacher_entropy_mean: Optional[torch.Tensor] = None,
    teacher_valid_token_count: Optional[torch.Tensor] = None,
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    _validate_cpu_teacher_distill_inputs(
        student_logits=student_logits,
        teacher_logits_cpu=teacher_logits_cpu,
        gamma=gamma,
        sequence_chunk_size=sequence_chunk_size,
    )
    if k is not None and int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if telemetry_out is not None:
        if teacher_entropy_mean is None or teacher_valid_token_count is None:
            raise ValueError(
                "CPU EAKLD telemetry requires teacher_entropy_mean and "
                "teacher_valid_token_count."
            )

    resolved_mask = _default_token_mask(student_logits, mask)
    denominator = resolved_mask.sum().clamp_min(1.0)
    chunk_results = []

    for start, end in iter_token_chunk_ranges(
        int(student_logits.shape[1]),
        int(sequence_chunk_size),
    ):
        student_chunk = student_logits[:, start:end, :]
        mask_chunk = resolved_mask[:, start:end]
        chunk_forward = _make_checkpointed_eakld_chunk_forward(
            teacher_logits_cpu=teacher_logits_cpu,
            start=start,
            end=end,
            mask_chunk=mask_chunk,
            temperature=float(temperature),
            k=k,
        )
        if torch.is_grad_enabled() and student_chunk.requires_grad:
            chunk_result = torch_checkpoint.checkpoint(
                chunk_forward,
                student_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_result = chunk_forward(student_chunk)
        chunk_results.append(chunk_result)

    total_sums = torch.stack(chunk_results, dim=0).sum(dim=0)
    temp = _resolve_distill_temperature(temperature)
    reverse_kl = total_sums[0] / denominator * (temp * temp)
    forward_kl = total_sums[1] / denominator * (temp * temp)
    gamma_device = gamma.detach().reshape(()).to(
        device=student_logits.device,
        dtype=torch.float32,
    )
    eakld_total = gamma_device * reverse_kl + (1.0 - gamma_device) * forward_kl
    if telemetry_out is not None:
        _write_eakld_telemetry(
            telemetry_out,
            teacher_entropy_mean=teacher_entropy_mean,
            gamma_reverse=gamma_device,
            forward_kl=forward_kl,
            reverse_kl=reverse_kl,
            eakld_total=eakld_total,
            valid_tokens=teacher_valid_token_count,
        )
    return eakld_total


def compute_eakld_from_cpu_teacher_logits(
    *,
    student_logits: torch.Tensor,
    teacher_logits_cpu: torch.Tensor,
    mask: Optional[torch.Tensor],
    gamma: torch.Tensor,
    temperature: float = 1.0,
    sequence_chunk_size: int = 1,
    teacher_entropy_mean: Optional[torch.Tensor] = None,
    teacher_valid_token_count: Optional[torch.Tensor] = None,
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    return _compute_eakld_from_cpu_teacher_logits_impl(
        student_logits=student_logits,
        teacher_logits_cpu=teacher_logits_cpu,
        mask=mask,
        gamma=gamma,
        temperature=float(temperature),
        sequence_chunk_size=int(sequence_chunk_size),
        k=None,
        teacher_entropy_mean=teacher_entropy_mean,
        teacher_valid_token_count=teacher_valid_token_count,
        telemetry_out=telemetry_out,
    )


def compute_eakld_topk_from_cpu_teacher_logits(
    *,
    student_logits: torch.Tensor,
    teacher_logits_cpu: torch.Tensor,
    mask: Optional[torch.Tensor],
    gamma: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    sequence_chunk_size: int = 1,
    teacher_entropy_mean: Optional[torch.Tensor] = None,
    teacher_valid_token_count: Optional[torch.Tensor] = None,
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    return _compute_eakld_from_cpu_teacher_logits_impl(
        student_logits=student_logits,
        teacher_logits_cpu=teacher_logits_cpu,
        mask=mask,
        gamma=gamma,
        temperature=float(temperature),
        sequence_chunk_size=int(sequence_chunk_size),
        k=int(k),
        teacher_entropy_mean=teacher_entropy_mean,
        teacher_valid_token_count=teacher_valid_token_count,
        telemetry_out=telemetry_out,
    )
