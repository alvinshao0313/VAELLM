from typing import MutableMapping, Optional

import torch
import torch.nn.functional as F

from train_utils.distill_losses import (
    compute_chunked_token_mean_from_cpu_teacher_logits,
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
    compute_masked_logit_mse_loss,
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


def _validate_prompt_weight(
    *,
    prompt_mask: Optional[torch.Tensor],
    prompt_kd_weight: float,
) -> float:
    weight = float(prompt_kd_weight)
    if weight < 0.0:
        raise ValueError(f"prompt_kd_weight must be >= 0.0, got {weight}.")
    if weight > 0.0 and prompt_mask is None:
        raise ValueError("prompt_kd_weight > 0 requires prompt_mask.")
    return weight


def _combine_region_loss(
    *,
    response_loss: torch.Tensor,
    prompt_loss_fn,
    prompt_mask: Optional[torch.Tensor],
    weight: float,
) -> torch.Tensor:
    if weight == 0.0 or prompt_mask is None:
        return response_loss
    return response_loss + weight * prompt_loss_fn()


def _is_eakld_family_loss(loss_type: str) -> bool:
    norm = str(loss_type or "").strip().lower()
    return norm in {"eakld", "eakld_kd"} or is_eakld_top_loss(norm)


def _offloaded_region_loss_type(loss_type: str) -> str:
    norm = str(loss_type or "").strip().lower()
    if norm == "kd":
        return "kl"
    if norm.startswith("kd_top"):
        return "kl_top" + norm[len("kd_top"):]
    if norm == "dual_kd":
        return "dual_kl"
    if norm.startswith("dual_kd_top"):
        return "dual_kl_top" + norm[len("dual_kd_top"):]
    return norm


def _is_ce_blended_dense_loss(loss_type: str) -> bool:
    norm = str(loss_type or "").strip().lower()
    return (
        norm == "kd"
        or norm.startswith("kd_top")
        or norm == "dual_kd"
        or norm.startswith("dual_kd_top")
        or norm == "eakld_kd"
    )


def compute_dense_loss_from_logits(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    ce_loss: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
    eakld_confidence_k: int = 16,
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]] = None,
    prompt_mask: Optional[torch.Tensor] = None,
    prompt_kd_weight: float = 0.0,
) -> torch.Tensor:
    weight = _validate_prompt_weight(
        prompt_mask=prompt_mask, prompt_kd_weight=prompt_kd_weight
    )
    norm = str(loss_type or "").strip().lower()
    if norm in {"sft", "origin"}:
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        return ce_loss
    if teacher_logits is None:
        raise ValueError(f"loss_type={norm} requires teacher_logits.")

    temperature = float(temperature)
    alpha_f = float(alpha)
    confidence_k = int(eakld_confidence_k)

    if norm == "rkl":
        response = compute_reverse_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=temperature,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_reverse_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                temperature=temperature,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm == "dual_rkl":
        response = compute_dual_rkl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_dual_rkl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm == "kl":
        response = compute_forward_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=temperature,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_forward_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                temperature=temperature,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm == "dual_kl":
        response = compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_dual_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm.startswith("r_kl_top"):
        k = parse_topk(norm, prefix="r_kl_top", default_k=1000)
        response = compute_rkl_topk(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            temperature=temperature,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_rkl_topk(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                k=k,
                temperature=temperature,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm.startswith("dual_r_kl_top"):
        k = parse_topk(norm, prefix="dual_r_kl_top", default_k=1000)
        response = compute_dual_rkl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_dual_rkl_topk_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                k=k,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm.startswith("kl_top"):
        k = parse_topk(norm, prefix="kl_top", default_k=1000)
        response = compute_kl_topk(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            temperature=temperature,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_kl_topk(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                k=k,
                temperature=temperature,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm.startswith("kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = parse_topk(norm, prefix="kd_top", default_k=1000)
        kd_response = compute_kl_topk(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            temperature=temperature,
        )
        kd_region = _combine_region_loss(
            response_loss=kd_response,
            prompt_loss_fn=lambda: compute_kl_topk(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                k=k,
                temperature=temperature,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
        # compute_kl_topk already multiplies by T².
        return ce_loss * (1.0 - alpha_f) + kd_region * alpha_f
    if is_eakld_top_loss(norm):
        k = parse_eakld_top_k(norm, default_k=1000)
        response = compute_eakld_topk(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
            temperature=temperature,
            confidence_k=confidence_k,
            telemetry_out=telemetry_out,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_eakld_topk(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                k=k,
                temperature=temperature,
                confidence_k=confidence_k,
                telemetry_out=None,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm == "eakld":
        response = compute_eakld(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=temperature,
            confidence_k=confidence_k,
            telemetry_out=telemetry_out,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_eakld(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                temperature=temperature,
                confidence_k=confidence_k,
                telemetry_out=None,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm == "eakld_kd":
        if ce_loss is None:
            raise ValueError("loss_type=eakld_kd requires ce_loss.")
        eakld_response = compute_entropy_aware_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=temperature,
            confidence_k=confidence_k,
            telemetry_out=telemetry_out,
        )
        eakld_region = _combine_region_loss(
            response_loss=eakld_response,
            prompt_loss_fn=lambda: compute_entropy_aware_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                temperature=temperature,
                confidence_k=confidence_k,
                telemetry_out=None,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
        # T² is already applied inside compute_eakld.
        return ce_loss * (1.0 - alpha_f) + eakld_region * alpha_f
    if norm.startswith("dual_kl_top"):
        k = parse_topk(norm, prefix="dual_kl_top", default_k=1000)
        response = compute_dual_kl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_dual_kl_topk_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                k=k,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm == "mse":
        response = compute_masked_logit_mse_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
        return _combine_region_loss(
            response_loss=response,
            prompt_loss_fn=lambda: compute_masked_logit_mse_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
    if norm == "kd":
        if ce_loss is None:
            raise ValueError("loss_type=kd requires ce_loss.")
        kd_response = compute_forward_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=temperature,
        )
        kd_region = _combine_region_loss(
            response_loss=kd_response,
            prompt_loss_fn=lambda: compute_forward_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                temperature=temperature,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
        # compute_forward_kl_loss already multiplies by T².
        return ce_loss * (1.0 - alpha_f) + kd_region * alpha_f
    if norm == "dual_kd":
        if ce_loss is None:
            raise ValueError("loss_type=dual_kd requires ce_loss.")
        kd_response = compute_dual_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
        )
        kd_region = _combine_region_loss(
            response_loss=kd_response,
            prompt_loss_fn=lambda: compute_dual_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
        return ce_loss * (1.0 - alpha_f) + kd_region * alpha_f
    if norm.startswith("dual_kd_top"):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        k = parse_topk(norm, prefix="dual_kd_top", default_k=1000)
        kd_response = compute_dual_kl_topk_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            k=k,
        )
        kd_region = _combine_region_loss(
            response_loss=kd_response,
            prompt_loss_fn=lambda: compute_dual_kl_topk_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                mask=prompt_mask,
                k=k,
            ),
            prompt_mask=prompt_mask,
            weight=weight,
        )
        return ce_loss * (1.0 - alpha_f) + kd_region * alpha_f

    raise ValueError(
        f"Unsupported dense loss type: {loss_type}. "
        "Supported: sft/origin, kl, rkl, dual_rkl, mse, kd, kd_top[_K], eakld, eakld_kd, "
        "eakld_top[_K]/eakld_topk[_K], dual_kd_top[_K], dual_kl, dual_kd, kl_top[_K], "
        "r_kl_top[_K], dual_r_kl_top[_K], dual_kl_top[_K]."
    )


def _compute_offloaded_non_eakld_region_loss(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits_cpu: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float,
    sequence_chunk_size: int,
) -> torch.Tensor:
    region_loss_type = _offloaded_region_loss_type(loss_type)

    def chunk_loss_fn(
        student_chunk: torch.Tensor,
        teacher_chunk: torch.Tensor,
        mask_chunk: torch.Tensor,
    ) -> torch.Tensor:
        return compute_dense_loss_from_logits(
            loss_type=region_loss_type,
            student_logits=student_chunk,
            teacher_logits=teacher_chunk,
            ce_loss=None,
            mask=mask_chunk,
            temperature=temperature,
            prompt_mask=None,
            prompt_kd_weight=0,
            telemetry_out=None,
        )

    return compute_chunked_token_mean_from_cpu_teacher_logits(
        student_logits=student_logits,
        teacher_logits_cpu=teacher_logits_cpu,
        mask=mask,
        sequence_chunk_size=sequence_chunk_size,
        chunk_loss_fn=chunk_loss_fn,
    )


def compute_dense_loss_from_offloaded_teacher(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits_cpu: torch.Tensor,
    teacher_gamma_cpu: Optional[torch.Tensor] = None,
    teacher_entropy_mean_cpu: Optional[torch.Tensor] = None,
    teacher_valid_token_count_cpu: Optional[torch.Tensor] = None,
    ce_loss: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
    eakld_confidence_k: int = 16,
    sequence_chunk_size: int = 16,
    telemetry_out: Optional[MutableMapping[str, torch.Tensor]] = None,
    prompt_mask: Optional[torch.Tensor] = None,
    prompt_kd_weight: float = 0.0,
    teacher_prompt_gamma_cpu: Optional[torch.Tensor] = None,
    teacher_prompt_entropy_mean_cpu: Optional[torch.Tensor] = None,
    teacher_prompt_valid_token_count_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weight = _validate_prompt_weight(
        prompt_mask=prompt_mask, prompt_kd_weight=prompt_kd_weight
    )
    norm = str(loss_type or "").strip().lower()
    is_eakld = _is_eakld_family_loss(norm)
    temperature = float(temperature)
    alpha_f = float(alpha)
    chunk_size = int(sequence_chunk_size)

    if is_eakld:
        if teacher_gamma_cpu is None:
            raise ValueError("EAKLD-family loss requires teacher_gamma_cpu.")
        if telemetry_out is not None and (
            teacher_entropy_mean_cpu is None
            or teacher_valid_token_count_cpu is None
        ):
            raise ValueError(
                "EAKLD-family telemetry requires teacher_entropy_mean_cpu and "
                "teacher_valid_token_count_cpu."
            )
        if weight > 0.0 and (
            teacher_prompt_gamma_cpu is None
            or teacher_prompt_entropy_mean_cpu is None
            or teacher_prompt_valid_token_count_cpu is None
        ):
            raise ValueError(
                "prompt_kd_weight > 0 requires teacher_prompt_gamma_cpu, "
                "teacher_prompt_entropy_mean_cpu, and "
                "teacher_prompt_valid_token_count_cpu."
            )
        if int(eakld_confidence_k) < 2:
            raise ValueError("eakld_confidence_k must be >= 2.")

        def _response_eakld() -> torch.Tensor:
            if norm == "eakld" or norm == "eakld_kd":
                return compute_eakld_from_cpu_teacher_logits(
                    student_logits=student_logits,
                    teacher_logits_cpu=teacher_logits_cpu,
                    mask=mask,
                    gamma=teacher_gamma_cpu,
                    temperature=temperature,
                    sequence_chunk_size=chunk_size,
                    teacher_entropy_mean=teacher_entropy_mean_cpu,
                    teacher_valid_token_count=teacher_valid_token_count_cpu,
                    telemetry_out=telemetry_out,
                )
            k = parse_eakld_top_k(norm, default_k=1000)
            return compute_eakld_topk_from_cpu_teacher_logits(
                student_logits=student_logits,
                teacher_logits_cpu=teacher_logits_cpu,
                mask=mask,
                gamma=teacher_gamma_cpu,
                k=k,
                temperature=temperature,
                sequence_chunk_size=chunk_size,
                teacher_entropy_mean=teacher_entropy_mean_cpu,
                teacher_valid_token_count=teacher_valid_token_count_cpu,
                telemetry_out=telemetry_out,
            )

        def _prompt_eakld() -> torch.Tensor:
            # Uses precomputed prompt-region gamma; does not overwrite response
            # telemetry (telemetry_out=None) and does not recompute gamma from
            # teacher_logits_cpu.
            if norm == "eakld" or norm == "eakld_kd":
                return compute_eakld_from_cpu_teacher_logits(
                    student_logits=student_logits,
                    teacher_logits_cpu=teacher_logits_cpu,
                    mask=prompt_mask,
                    gamma=teacher_prompt_gamma_cpu,
                    temperature=temperature,
                    sequence_chunk_size=chunk_size,
                    teacher_entropy_mean=None,
                    teacher_valid_token_count=None,
                    telemetry_out=None,
                )
            k = parse_eakld_top_k(norm, default_k=1000)
            return compute_eakld_topk_from_cpu_teacher_logits(
                student_logits=student_logits,
                teacher_logits_cpu=teacher_logits_cpu,
                mask=prompt_mask,
                gamma=teacher_prompt_gamma_cpu,
                k=k,
                temperature=temperature,
                sequence_chunk_size=chunk_size,
                teacher_entropy_mean=None,
                teacher_valid_token_count=None,
                telemetry_out=None,
            )

        eakld_response = _response_eakld()
        eakld_region = _combine_region_loss(
            response_loss=eakld_response,
            prompt_loss_fn=_prompt_eakld,
            prompt_mask=prompt_mask,
            weight=weight,
        )
        if norm != "eakld_kd":
            return eakld_region
        if ce_loss is None:
            raise ValueError("loss_type=eakld_kd requires ce_loss.")
        return ce_loss * (1.0 - alpha_f) + eakld_region * alpha_f

    response_loss = _compute_offloaded_non_eakld_region_loss(
        loss_type=norm,
        student_logits=student_logits,
        teacher_logits_cpu=teacher_logits_cpu,
        mask=mask,
        temperature=temperature,
        sequence_chunk_size=chunk_size,
    )
    region_loss = _combine_region_loss(
        response_loss=response_loss,
        prompt_loss_fn=lambda: _compute_offloaded_non_eakld_region_loss(
            loss_type=norm,
            student_logits=student_logits,
            teacher_logits_cpu=teacher_logits_cpu,
            mask=prompt_mask,
            temperature=temperature,
            sequence_chunk_size=chunk_size,
        ),
        prompt_mask=prompt_mask,
        weight=weight,
    )
    if _is_ce_blended_dense_loss(norm):
        if ce_loss is None:
            raise ValueError(f"loss_type={norm} requires ce_loss.")
        return ce_loss * (1.0 - alpha_f) + region_loss * alpha_f
    return region_loss
