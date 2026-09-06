from typing import Optional

import torch

from train_utils.distill_loss_core import (
    MODEL_LEVEL_LOSS_TYPES,
    compute_model_level_loss,
    normalize_model_level_loss_type,
)


def get_output_logits(outputs) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    raise AttributeError("Model outputs do not contain `logits`.")


def compute_dense_loss_from_logits(
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
    prompt_kd_weight: Optional[float] = None,
) -> torch.Tensor:
    resolved_prompt_weight = (
        float(prompt_loss_weight)
        if prompt_kd_weight is None
        else float(prompt_kd_weight)
    )
    # Consume canonical loss_type + explicit top_k only; no suffix re-parse here.
    resolved_type = normalize_model_level_loss_type(loss_type)
    return compute_model_level_loss(
        loss_type=resolved_type,
        student_logits=student_logits,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        teacher_logits=teacher_logits,
        temperature=temperature,
        alpha=alpha,
        top_k=int(top_k),
        prompt_loss_weight=resolved_prompt_weight,
    )


def compute_dense_loss_from_offloaded_teacher(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits_cpu: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = 0.5,
    top_k: int = 100,
    prompt_loss_weight: float = 0.0,
    prompt_kd_weight: Optional[float] = None,
    teacher_output_chunk_tokens: int = 8,
) -> torch.Tensor:
    """Offload path: move teacher logits to student device then use shared core.

    Chunked CPU->GPU transfer remains a teacher-runtime concern; Task 5 keeps math
    identical to the dense path via a full transfer into the shared core.
    """
    del teacher_output_chunk_tokens
    if not isinstance(teacher_logits_cpu, torch.Tensor):
        raise TypeError("teacher_logits_cpu must be a torch.Tensor.")
    teacher_logits = teacher_logits_cpu.to(device=student_logits.device, dtype=torch.float32)
    return compute_dense_loss_from_logits(
        loss_type=loss_type,
        student_logits=student_logits,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        teacher_logits=teacher_logits,
        temperature=temperature,
        alpha=alpha,
        top_k=top_k,
        prompt_loss_weight=prompt_loss_weight,
        prompt_kd_weight=prompt_kd_weight,
    )


# Re-export for callers that only need the supported type list.
__all__ = [
    "MODEL_LEVEL_LOSS_TYPES",
    "compute_dense_loss_from_logits",
    "compute_dense_loss_from_offloaded_teacher",
    "get_output_logits",
]
