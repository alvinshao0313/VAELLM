"""Minimal numerical smoke tests for model-level distill loss core paths.

No large model download. Uses tiny synthetic logits (B=2, L=6, V=31).
"""

from __future__ import annotations

import pytest
import torch

from e2e_common.dense_loss import (
    compute_dense_loss_from_logits,
    compute_dense_loss_from_offloaded_teacher,
)
from train_utils.distill_loss_core import MODEL_LEVEL_LOSS_TYPES


def _make_tiny_batch(*, seed: int = 12):
    torch.manual_seed(seed)
    batch, seq_len, vocab = 2, 6, 31
    teacher_logits = torch.randn(batch, seq_len, vocab, dtype=torch.float32)
    student_base = teacher_logits + torch.randn(batch, seq_len, vocab, dtype=torch.float32) * 0.75
    input_ids = torch.randint(0, vocab, (batch, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels[0, -1] = -100
    labels[1, -2:] = -100
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    return student_base, teacher_logits, input_ids, labels, attention_mask


@pytest.mark.parametrize("loss_type", MODEL_LEVEL_LOSS_TYPES)
def test_dense_dispatcher_loss_pipeline_smoke(loss_type: str) -> None:
    student_base, teacher_logits, input_ids, labels, attention_mask = _make_tiny_batch()
    student_logits = student_base.detach().clone().requires_grad_(True)
    teacher = None if loss_type == "sft" else teacher_logits
    loss = compute_dense_loss_from_logits(
        loss_type=loss_type,
        student_logits=student_logits,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        teacher_logits=teacher,
        temperature=1.0,
        alpha=0.5,
        top_k=7,
        prompt_loss_weight=0.03,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    # Final prediction position is padded out of both masks -> zero grad on last step.
    assert torch.equal(
        student_logits.grad[:, -1, :],
        torch.zeros_like(student_logits.grad[:, -1, :]),
    )


def test_offload_cpu_kl_matches_dense() -> None:
    student_base, teacher_logits, input_ids, labels, attention_mask = _make_tiny_batch(seed=21)
    dense_student = student_base.detach().clone().requires_grad_(True)
    dense_loss = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=dense_student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        teacher_logits=teacher_logits,
        temperature=1.0,
        prompt_loss_weight=0.03,
    )
    dense_loss.backward()

    off_student = student_base.detach().clone().requires_grad_(True)
    off_loss = compute_dense_loss_from_offloaded_teacher(
        loss_type="kl",
        student_logits=off_student,
        teacher_logits_cpu=teacher_logits.detach().cpu(),
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        temperature=1.0,
        prompt_loss_weight=0.03,
    )
    off_loss.backward()
    torch.testing.assert_close(off_loss, dense_loss.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(off_student.grad, dense_student.grad, rtol=1e-5, atol=1e-6)


def test_deleted_model_level_types_are_rejected() -> None:
    student_base, teacher_logits, input_ids, labels, attention_mask = _make_tiny_batch()
    for removed in ("origin", "rkl", "dual_kl", "eakld", "mse", "choice_kd"):
        with pytest.raises(ValueError, match="Unsupported"):
            compute_dense_loss_from_logits(
                loss_type=removed,
                student_logits=student_base,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                teacher_logits=teacher_logits,
            )
