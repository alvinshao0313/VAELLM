from typing import Optional

import torch
import torch.nn.functional as F

from train_utils.distill_losses import (
    compute_dual_kl_loss,
    compute_dual_kl_topk_loss,
    compute_dual_rkl_loss,
    compute_dual_rkl_topk_loss,
    compute_eakld,
    compute_eakld_from_cpu_teacher_logits,
    compute_eakld_topk,
    compute_eakld_topk_from_cpu_teacher_logits,
    compute_entropy_aware_kl_loss,
    compute_forward_kl_loss,
    compute_kl_topk,
    compute_reverse_kl_loss,
    compute_rkl_topk,
    is_eakld_top_loss,
    parse_eakld_top_k,
)


def get_output_logits(outputs) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    raise AttributeError("Model outputs do not contain `logits`.")


def parse_topk(value: str, *, prefix: str, default_k: int) -> int:
    if value == prefix:
        return int(default_k)
    suffix = value[len(prefix):]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if not suffix:
        return int(default_k)
    return max(1, int(suffix))


def token_mean_kl_div(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    student_log_probs: bool = False,
    teacher_probs: bool = False,
) -> torch.Tensor:
    """Mean KL over tokens. Always reduce on reshaped [N, C], never on raw [B, L, C]."""
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "student/teacher logits shape mismatch for KL: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}."
        )
    student = student_logits.float().reshape(-1, int(student_logits.shape[-1]))
    teacher = teacher_logits.float().reshape(-1, int(teacher_logits.shape[-1]))
    log_p = student if student_log_probs else F.log_softmax(student, dim=-1)
    target = teacher if teacher_probs else F.softmax(teacher, dim=-1)
    token_kl = F.kl_div(log_p, target, reduction="none").sum(dim=-1)
    return token_kl.mean()


def compute_dense_loss_from_logits(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    ce_loss: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
    post_attn: bool = False,
    eakld_confidence_k: int = 16,
) -> torch.Tensor:
    norm = str(loss_type or "").strip().lower()
    if norm in {"sft", "origin"}:
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        return ce_loss
    if teacher_logits is None:
        raise ValueError(f"loss_type={norm} requires teacher_logits.")

    if norm == "rkl":
        return compute_reverse_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=float(temperature),
        )
    if norm == "dual_rkl":
        return compute_dual_rkl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
    if norm == "kl":
        return compute_forward_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=float(temperature),
        )
    if norm == "dual_kl":
        return compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
    if norm.startswith("r_kl_top"):
        k = parse_topk(norm, prefix="r_kl_top", default_k=1000)
        return compute_rkl_topk(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            temperature=float(temperature),
            post_attn=bool(post_attn),
        )
    if norm.startswith("dual_r_kl_top"):
        k = parse_topk(norm, prefix="dual_r_kl_top", default_k=1000)
        return compute_dual_rkl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
    if norm.startswith("kl_top"):
        k = parse_topk(norm, prefix="kl_top", default_k=1000)
        return compute_kl_topk(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            temperature=float(temperature),
            post_attn=bool(post_attn),
        )
    if norm.startswith("kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = parse_topk(norm, prefix="kd_top", default_k=1000)
        temperature = float(temperature)
        kd_loss = compute_kl_topk(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            temperature=temperature,
            post_attn=bool(post_attn),
        )
        # compute_kl_topk already multiplies by T².
        return ce_loss * (1.0 - float(alpha)) + kd_loss * float(alpha)
    if is_eakld_top_loss(norm):
        k = parse_eakld_top_k(norm, default_k=1000)
        return compute_eakld_topk(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            temperature=float(temperature),
            confidence_k=int(eakld_confidence_k),
            post_attn=bool(post_attn),
        )
    if norm == "eakld":
        return compute_eakld(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=float(temperature),
            confidence_k=int(eakld_confidence_k),
        )
    if norm == "eakld_kd":
        if ce_loss is None:
            raise ValueError("loss_type=eakld_kd requires ce_loss.")
        temperature = float(temperature)
        eakld_loss = compute_entropy_aware_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=temperature,
            confidence_k=int(eakld_confidence_k),
        )
        # T² is already applied inside compute_eakld.
        return ce_loss * (1.0 - float(alpha)) + eakld_loss * float(alpha)
    if norm.startswith("dual_kl_top"):
        k = parse_topk(norm, prefix="dual_kl_top", default_k=1000)
        return compute_dual_kl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
    if norm == "mse":
        return F.mse_loss(student_logits, teacher_logits)
    if norm == "kd":
        if ce_loss is None:
            raise ValueError("loss_type=kd requires ce_loss.")
        temperature = float(temperature)
        kd_loss = compute_forward_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=temperature,
        )
        # compute_forward_kl_loss already multiplies by T².
        return ce_loss * (1.0 - float(alpha)) + kd_loss * float(alpha)
    if norm == "dual_kd":
        if ce_loss is None:
            raise ValueError("loss_type=dual_kd requires ce_loss.")
        kd_loss = compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * float(alpha)
    if norm.startswith("dual_kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = parse_topk(norm, prefix="dual_kd_top", default_k=1000)
        kd_loss = compute_dual_kl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            post_attn=bool(post_attn),
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * float(alpha)

    raise ValueError(
        f"Unsupported dense loss type: {loss_type}. "
        "Supported: sft/origin, kl, rkl, dual_rkl, mse, kd, kd_top[_K], eakld, eakld_kd, "
        "eakld_top[_K]/eakld_topk[_K], dual_kd_top[_K], dual_kl, dual_kd, kl_top[_K], "
        "r_kl_top[_K], dual_r_kl_top[_K], dual_kl_top[_K]."
    )


def compute_dense_loss_from_offloaded_teacher(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits_cpu: torch.Tensor,
    teacher_gamma_cpu: torch.Tensor,
    ce_loss: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
    post_attn: bool = False,
    eakld_confidence_k: int = 16,
    sequence_chunk_size: int = 16,
) -> torch.Tensor:
    norm = str(loss_type or "").strip().lower()
    if int(eakld_confidence_k) < 2:
        raise ValueError("eakld_confidence_k must be >= 2.")

    if norm == "eakld" or norm == "eakld_kd":
        eakld_loss = compute_eakld_from_cpu_teacher_logits(
            student_logits=student_logits,
            teacher_logits_cpu=teacher_logits_cpu,
            mask=mask,
            gamma=teacher_gamma_cpu,
            temperature=float(temperature),
            sequence_chunk_size=int(sequence_chunk_size),
        )
    elif is_eakld_top_loss(norm):
        eakld_loss = compute_eakld_topk_from_cpu_teacher_logits(
            student_logits=student_logits,
            teacher_logits_cpu=teacher_logits_cpu,
            mask=mask,
            gamma=teacher_gamma_cpu,
            k=parse_eakld_top_k(norm, default_k=1000),
            temperature=float(temperature),
            post_attn=bool(post_attn),
            sequence_chunk_size=int(sequence_chunk_size),
        )
    else:
        raise ValueError(
            "teacher_output_offload=cpu supports only EAKLD-family losses."
        )

    if norm != "eakld_kd":
        return eakld_loss
    if ce_loss is None:
        raise ValueError("loss_type=eakld_kd requires ce_loss.")
    return ce_loss * (1.0 - float(alpha)) + eakld_loss * float(alpha)
