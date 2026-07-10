from typing import Optional

import torch
import torch.nn.functional as F

from train_utils.distill_losses import (
    compute_dual_kl_loss,
    compute_dual_kl_topk_loss,
    compute_dual_rkl_loss,
    compute_dual_rkl_topk_loss,
    compute_entropy_aware_kl_loss,
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
        return F.kl_div(
            F.log_softmax(teacher_logits.flatten(0, -2), dim=-1),
            F.softmax(student_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm == "dual_rkl":
        return compute_dual_rkl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
    if norm == "kl":
        return F.kl_div(
            F.log_softmax(student_logits.flatten(0, -2), dim=-1),
            F.softmax(teacher_logits.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm == "dual_kl":
        return compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
    if norm.startswith("r_kl_top"):
        k = parse_topk(norm, prefix="r_kl_top", default_k=1000)
        k = min(int(k), int(student_logits.shape[-1]))
        top_student, indices = student_logits.topk(k, dim=-1, sorted=False)
        top_teacher = teacher_logits.gather(-1, indices)
        return F.kl_div(
            F.log_softmax(top_teacher.flatten(0, -2), dim=-1),
            F.softmax(top_student.flatten(0, -2), dim=-1),
            reduction="batchmean",
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
        k = min(int(k), int(teacher_logits.shape[-1]))
        top_teacher, indices = teacher_logits.topk(k, dim=-1, sorted=False)
        if bool(post_attn):
            ref = F.softmax(teacher_logits, dim=-1).gather(-1, indices).flatten(0, -2)
            can = F.log_softmax(student_logits, dim=-1).gather(-1, indices).flatten(0, -2)
            return F.kl_div(can, ref, reduction="batchmean")
        top_student = student_logits.gather(-1, indices)
        return F.kl_div(
            F.log_softmax(top_student.flatten(0, -2), dim=-1),
            F.softmax(top_teacher.flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
    if norm.startswith("kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = parse_topk(norm, prefix="kd_top", default_k=1000)
        k = min(int(k), int(teacher_logits.shape[-1]))
        top_teacher, indices = teacher_logits.topk(k, dim=-1, sorted=False)
        temperature = float(temperature)
        if bool(post_attn):
            ref = F.softmax(teacher_logits / temperature, dim=-1).gather(-1, indices).flatten(0, -2)
            can = F.log_softmax(student_logits / temperature, dim=-1).gather(-1, indices).flatten(0, -2)
            kd_loss = F.kl_div(can, ref, reduction="batchmean")
        else:
            top_student = student_logits.gather(-1, indices)
            kd_loss = F.kl_div(
                F.log_softmax((top_student / temperature).flatten(0, -2), dim=-1),
                F.softmax((top_teacher / temperature).flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * (float(alpha) * temperature * temperature)
    if norm == "eakld":
        return compute_entropy_aware_kl_loss(
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
        return ce_loss * (1.0 - float(alpha)) + eakld_loss * (float(alpha) * temperature * temperature)
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
        kd_loss = F.kl_div(
            F.log_softmax((student_logits / temperature).flatten(0, -2), dim=-1),
            F.softmax((teacher_logits / temperature).flatten(0, -2), dim=-1),
            reduction="batchmean",
        )
        return ce_loss * (1.0 - float(alpha)) + kd_loss * (float(alpha) * temperature * temperature)
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
        "Supported: sft/origin, kl, rkl, dual_rkl, mse, kd, kd_top[_K], eakld, eakld_kd, dual_kd_top[_K], "
        "dual_kl, dual_kd, kl_top[_K], r_kl_top[_K], dual_r_kl_top[_K], dual_kl_top[_K]."
    )
