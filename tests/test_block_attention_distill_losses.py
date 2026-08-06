from __future__ import annotations

from unittest import mock

import torch
from torch import nn

from train_utils import block_distill
from train_utils.block_distill import (
    _attention_map_kl_chunk_losses,
    entropy_aware_attention_map_kl_loss,
)


def _manual_forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    teacher_log = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_log = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_prob = teacher_log.exp()
    return (teacher_prob * (teacher_log - student_log)).sum(dim=-1)


def _manual_reverse_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    teacher_log = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_log = torch.log_softmax(student_logits.float(), dim=-1)
    student_prob = student_log.exp()
    return (student_prob * (student_log - teacher_log)).sum(dim=-1)


def test_attention_chunk_helper_names_forward_and_reverse_kl_correctly() -> None:
    teacher = torch.tensor([[[[4.0, 1.0, -2.0]]]], dtype=torch.float32)
    student = torch.tensor([[[[-1.0, 2.0, 1.0]]]], dtype=torch.float32)
    valid = torch.ones_like(teacher, dtype=torch.bool)

    forward, reverse, _entropy, valid_query = (
        _attention_map_kl_chunk_losses(
            teacher_logits=teacher,
            student_logits=student,
            valid_key_mask=valid,
        )
    )

    expected_forward = _manual_forward_kl(teacher, student)
    expected_reverse = _manual_reverse_kl(teacher, student)

    assert torch.equal(valid_query, torch.ones_like(valid_query, dtype=torch.bool))
    assert torch.allclose(forward, expected_forward, rtol=1e-6, atol=1e-7)
    assert torch.allclose(reverse, expected_reverse, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(forward, reverse)


def test_attention_chunk_helper_ignores_masked_keys_exactly() -> None:
    teacher = torch.tensor(
        [[[[2.0, 0.0, 10000.0, -10000.0]]]],
        dtype=torch.bfloat16,
    )
    student = torch.tensor(
        [[[[0.0, 2.0, -10000.0, 10000.0]]]],
        dtype=torch.bfloat16,
    )
    valid = torch.tensor([[[[True, True, False, False]]]])

    forward, reverse, entropy, valid_query = (
        _attention_map_kl_chunk_losses(
            teacher_logits=teacher,
            student_logits=student,
            valid_key_mask=valid,
        )
    )

    teacher_ref = teacher[..., :2].float()
    student_ref = student[..., :2].float()
    expected_forward = _manual_forward_kl(teacher_ref, student_ref)
    expected_reverse = _manual_reverse_kl(teacher_ref, student_ref)
    teacher_log = torch.log_softmax(teacher_ref, dim=-1)
    expected_entropy = -(teacher_log.exp() * teacher_log).sum(dim=-1)

    assert valid_query.item() is True
    assert torch.allclose(forward, expected_forward, rtol=1e-6, atol=1e-7)
    assert torch.allclose(reverse, expected_reverse, rtol=1e-6, atol=1e-7)
    assert torch.allclose(entropy, expected_entropy, rtol=1e-6, atol=1e-7)


def test_attention_chunk_helper_marks_query_invalid_when_no_key_is_valid() -> None:
    teacher = torch.randn(2, 3, 4, 5)
    student = torch.randn(2, 3, 4, 5)
    valid = torch.zeros_like(teacher, dtype=torch.bool)

    forward, reverse, entropy, valid_query = (
        _attention_map_kl_chunk_losses(
            teacher_logits=teacher,
            student_logits=student,
            valid_key_mask=valid,
        )
    )

    assert not valid_query.any()
    assert torch.equal(forward, torch.zeros_like(forward))
    assert torch.equal(reverse, torch.zeros_like(reverse))
    assert torch.equal(entropy, torch.zeros_like(entropy))


def test_attention_chunk_helper_computes_fp32_finite_losses_from_bf16() -> None:
    torch.manual_seed(31)
    teacher = (torch.randn(2, 4, 7, 257) * 8.0).to(torch.bfloat16)
    student = (torch.randn(2, 4, 7, 257) * 8.0).to(torch.bfloat16)
    valid = torch.ones_like(teacher, dtype=torch.bool)
    valid[..., -17:] = False

    forward, reverse, entropy, valid_query = (
        _attention_map_kl_chunk_losses(
            teacher_logits=teacher,
            student_logits=student,
            valid_key_mask=valid,
        )
    )

    assert forward.dtype == torch.float32
    assert reverse.dtype == torch.float32
    assert entropy.dtype == torch.float32
    assert valid_query.dtype == torch.bool
    assert torch.isfinite(forward).all()
    assert torch.isfinite(reverse).all()
    assert torch.isfinite(entropy).all()
    assert (forward >= -1e-6).all()
    assert (reverse >= -1e-6).all()


def test_attention_chunk_helper_returns_zero_when_distributions_match() -> None:
    torch.manual_seed(41)
    logits = torch.randn(2, 3, 5, 11)
    valid = torch.ones_like(logits, dtype=torch.bool)

    forward, reverse, _entropy, valid_query = (
        _attention_map_kl_chunk_losses(
            teacher_logits=logits,
            student_logits=logits.clone(),
            valid_key_mask=valid,
        )
    )

    assert valid_query.all()
    assert torch.allclose(forward, torch.zeros_like(forward), atol=1e-7)
    assert torch.allclose(reverse, torch.zeros_like(reverse), atol=1e-7)


def test_entropy_aware_attention_eakld_telemetry_directions() -> None:
    torch.manual_seed(53)
    teacher_q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    teacher_k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    student_q = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)
    student_k = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.float32)

    class _FakeAttn:
        scaling = 1.0

    class _FakeLayer:
        self_attn = _FakeAttn()

    class _FakeInner:
        layers = [_FakeLayer()]

    class _FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _FakeInner()

    model = _FakeModel()
    calls = {"n": 0}

    def fake_qk(_model, _layer_idx, _hidden):
        calls["n"] += 1
        if calls["n"] == 1:
            return teacher_q, teacher_k, mask
        return student_q, student_k, mask

    telemetry: dict[str, torch.Tensor] = {}
    with mock.patch.object(
        block_distill,
        "_qk_states_for_attention",
        side_effect=fake_qk,
    ):
        loss = entropy_aware_attention_map_kl_loss(
            model,
            0,
            student_hidden=torch.zeros(1, 2, 2),
            teacher_hidden=torch.zeros(1, 2, 2),
            query_chunk_size=2,
            confidence_k=16,
            telemetry_out=telemetry,
        )

    teacher_logits = torch.matmul(teacher_q, teacher_k.transpose(2, 3))
    student_logits = torch.matmul(student_q, student_k.transpose(2, 3))
    expected_kl_teacher_student = _manual_forward_kl(
        teacher_logits,
        student_logits,
    ).mean()
    expected_kl_student_teacher = _manual_reverse_kl(
        teacher_logits,
        student_logits,
    ).mean()

    assert torch.allclose(
        telemetry["forward_kl"],
        expected_kl_teacher_student,
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.allclose(
        telemetry["reverse_kl"],
        expected_kl_student_teacher,
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.allclose(
        telemetry["eakld_total"],
        (
            telemetry["gamma_reverse"] * telemetry["reverse_kl"]
            + telemetry["lambda_forward"] * telemetry["forward_kl"]
        ),
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.allclose(loss, telemetry["eakld_total"], rtol=1e-6, atol=1e-7)
    assert set(telemetry) == {
        "teacher_entropy_mean",
        "gamma_reverse",
        "lambda_forward",
        "forward_kl",
        "reverse_kl",
        "eakld_total",
        "valid_queries",
    }
