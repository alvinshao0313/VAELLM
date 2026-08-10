from __future__ import annotations

import math
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from e2e_common.dense_loss import (
    compute_dense_loss_from_logits,
    compute_dense_loss_from_offloaded_teacher,
)
from train_utils import distill_losses


def _dense_teacher_entropy_stats_reference(
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_probs = F.softmax(teacher_logits.detach().float(), dim=-1)
    entropy = -(
        teacher_probs * torch.log(teacher_probs.clamp_min(1e-8))
    ).sum(dim=-1)
    mask_fp32 = mask.to(device=entropy.device, dtype=torch.float32)
    return (entropy * mask_fp32).sum(), mask_fp32.sum()


@pytest.mark.parametrize("sequence_chunk_size", [1, 2, 3, 16])
def test_chunked_teacher_entropy_matches_dense_fp32_reference(
    sequence_chunk_size: int,
) -> None:
    torch.manual_seed(31)
    teacher_logits = torch.randn(3, 7, 19, dtype=torch.bfloat16) * 3.0
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0, 1],
        ],
        dtype=torch.float32,
    )

    expected_sum, expected_valid = _dense_teacher_entropy_stats_reference(
        teacher_logits,
        mask,
    )
    actual_sum, actual_valid = distill_losses.accumulate_teacher_entropy_stats(
        teacher_logits,
        mask,
        sequence_chunk_size=sequence_chunk_size,
    )

    assert actual_sum.dtype == torch.float32
    assert actual_valid.dtype == torch.float32
    assert torch.allclose(actual_sum, expected_sum, rtol=1e-6, atol=1e-5)
    assert torch.equal(actual_valid, expected_valid)


def test_chunked_teacher_entropy_rejects_invalid_inputs() -> None:
    logits = torch.randn(2, 3, 5)
    mask = torch.ones(2, 3)

    with pytest.raises(ValueError, match="sequence_chunk_size must be > 0"):
        distill_losses.accumulate_teacher_entropy_stats(
            logits,
            mask,
            sequence_chunk_size=0,
        )

    with pytest.raises(ValueError, match="mask shape mismatch"):
        distill_losses.accumulate_teacher_entropy_stats(
            logits,
            torch.ones(2, 2),
            sequence_chunk_size=2,
        )

    with pytest.raises(ValueError, match=r"shape \[B, L, V\]"):
        distill_losses.accumulate_teacher_entropy_stats(
            torch.randn(6, 5),
            torch.ones(6),
            sequence_chunk_size=2,
        )


def test_teacher_entropy_never_runs_softmax_on_full_sequence() -> None:
    torch.manual_seed(7)
    teacher_logits = torch.randn(4, 41, 257, dtype=torch.bfloat16)
    mask = torch.ones(4, 41, dtype=torch.float32)
    observed_shapes: list[tuple[int, ...]] = []
    observed_dtypes: list[torch.dtype] = []
    real_softmax = F.softmax

    def recording_softmax(
        input_tensor: torch.Tensor,
        dim: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        observed_shapes.append(tuple(int(value) for value in input_tensor.shape))
        observed_dtypes.append(input_tensor.dtype)
        return real_softmax(input_tensor, dim=dim, *args, **kwargs)

    with mock.patch.object(
        distill_losses.F,
        "softmax",
        side_effect=recording_softmax,
    ):
        entropy_sum, valid_count = distill_losses.accumulate_teacher_entropy_stats(
            teacher_logits,
            mask,
        )

    assert torch.isfinite(entropy_sum)
    assert valid_count.item() == pytest.approx(4 * 41)
    assert len(observed_shapes) == 3
    assert all(shape[0] == 4 for shape in observed_shapes)
    assert all(
        shape[1] <= distill_losses.DEFAULT_TEACHER_ENTROPY_SEQUENCE_CHUNK_SIZE
        for shape in observed_shapes
    )
    assert all(shape[2] == 257 for shape in observed_shapes)
    assert all(dtype == torch.float32 for dtype in observed_dtypes)
    assert not any(shape == (4, 41, 257) for shape in observed_shapes)


def test_eakld_teacher_entropy_gamma_uses_fractional_mask() -> None:
    torch.manual_seed(55)
    teacher_logits = torch.randn(2, 5, 31, dtype=torch.float32)
    mask = torch.tensor(
        [
            [0.1, 1.0, 1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    entropy_mean, gamma_reverse, valid_count = (
        distill_losses.compute_teacher_entropy_mean_and_gamma(
            teacher_logits,
            mask,
            confidence_k=16,
        )
    )

    expected_entropy_sum, expected_valid = _dense_teacher_entropy_stats_reference(
        teacher_logits,
        mask,
    )
    expected_entropy_mean = expected_entropy_sum / expected_valid.clamp_min(1.0)
    expected_gamma = distill_losses.gamma_from_entropy_sums(
        expected_entropy_sum,
        expected_valid,
        confidence_k=16,
    )

    assert valid_count.dtype == torch.float32
    assert torch.allclose(valid_count, expected_valid, rtol=1e-6, atol=1e-6)
    assert valid_count.item() == pytest.approx(mask.sum().item())
    assert torch.allclose(entropy_mean, expected_entropy_mean, rtol=1e-6, atol=1e-5)
    assert torch.allclose(gamma_reverse, expected_gamma, rtol=1e-6, atol=1e-5)


def test_eakld_telemetry_valid_tokens_is_fractional_mask_sum() -> None:
    torch.manual_seed(61)
    teacher = torch.randn(2, 5, 19)
    student = torch.randn(2, 5, 19, requires_grad=True)
    mask = torch.tensor(
        [
            [0.1, 1.0, 1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    telemetry: dict[str, torch.Tensor] = {}

    loss = distill_losses.compute_eakld(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
        temperature=1.0,
        confidence_k=16,
        telemetry_out=telemetry,
    )
    loss.backward()

    assert telemetry["valid_tokens"].item() == pytest.approx(mask.sum().item())
    assert telemetry["valid_tokens"].item() == pytest.approx(4.3)


def test_eakld_gamma_uses_full_vocab_entropy_and_log_confidence_k() -> None:
    teacher_logits = torch.zeros(1, 2, 32, dtype=torch.float32)
    mask = torch.ones(1, 2, dtype=torch.float32)

    entropy_mean, gamma_reverse, valid_count = (
        distill_losses.compute_teacher_entropy_mean_and_gamma(
            teacher_logits,
            mask,
            confidence_k=16,
        )
    )

    expected_entropy = torch.tensor(
        math.log(32.0),
        dtype=torch.float32,
    )
    expected_gamma = torch.clamp(
        1.0 - expected_entropy / math.log(16.0),
        min=0.0,
        max=1.0,
    )

    assert torch.allclose(entropy_mean, expected_entropy, atol=1e-6)
    assert torch.allclose(gamma_reverse, expected_gamma, atol=1e-6)
    assert valid_count.item() == pytest.approx(2.0)


def test_eakld_telemetry_reuses_existing_entropy_and_kl_computation() -> None:
    torch.manual_seed(47)
    teacher = torch.randn(2, 4, 19)
    mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    original_softmax = distill_losses.F.softmax

    baseline_student = torch.randn(2, 4, 19, requires_grad=True)
    with mock.patch.object(
        distill_losses.F,
        "softmax",
        wraps=original_softmax,
    ) as baseline_softmax:
        baseline_loss = distill_losses.compute_eakld(
            student_logits=baseline_student,
            teacher_logits=teacher,
            mask=mask,
            temperature=1.0,
            confidence_k=16,
            telemetry_out=None,
        )
    baseline_loss.backward()
    baseline_grad = baseline_student.grad.detach().clone()
    baseline_calls = baseline_softmax.call_count

    telemetry_student = baseline_student.detach().clone().requires_grad_(True)
    telemetry: dict[str, torch.Tensor] = {}
    with mock.patch.object(
        distill_losses.F,
        "softmax",
        wraps=original_softmax,
    ) as telemetry_softmax:
        telemetry_loss = distill_losses.compute_eakld(
            student_logits=telemetry_student,
            teacher_logits=teacher,
            mask=mask,
            temperature=1.0,
            confidence_k=16,
            telemetry_out=telemetry,
        )
    telemetry_loss.backward()

    assert telemetry_softmax.call_count == baseline_calls
    assert torch.allclose(telemetry_loss, baseline_loss, rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        telemetry_student.grad,
        baseline_grad,
        rtol=1e-6,
        atol=1e-7,
    )
    assert set(telemetry) == {
        "teacher_entropy_mean",
        "gamma_reverse",
        "lambda_forward",
        "forward_kl",
        "reverse_kl",
        "eakld_total",
        "valid_tokens",
    }
    for value in telemetry.values():
        assert value.ndim == 0
        assert value.requires_grad is False


def _dense_eakld_topk_reference(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    temperature: float,
    confidence_k: int,
) -> torch.Tensor:
    temp = max(float(temperature), 0.1)
    expected_entropy_sum, expected_valid = _dense_teacher_entropy_stats_reference(
        teacher_logits,
        mask,
    )
    gamma = distill_losses.gamma_from_entropy_sums(
        expected_entropy_sum,
        expected_valid,
        confidence_k=int(confidence_k),
    )
    student_scaled = student_logits.float() / temp
    teacher_scaled = teacher_logits.detach().float() / temp
    reverse_kl = distill_losses._topk_reverse_kl_mean(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=mask,
        k=int(k),
    ) * (temp * temp)
    forward_kl = distill_losses._topk_forward_kl_mean(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=mask,
        k=int(k),
    ) * (temp * temp)
    return gamma * reverse_kl + (1.0 - gamma) * forward_kl


def test_eakld_topk_fractional_mask_matches_dense_output_and_gradient() -> None:
    torch.manual_seed(19)
    teacher_logits = torch.randn(2, 5, 23, dtype=torch.bfloat16)
    mask = torch.tensor(
        [
            [0.1, 1.0, 1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    student_expected = torch.randn(
        2,
        5,
        23,
        dtype=torch.float32,
        requires_grad=True,
    )
    expected = _dense_eakld_topk_reference(
        student_logits=student_expected,
        teacher_logits=teacher_logits,
        mask=mask,
        k=7,
        temperature=1.3,
        confidence_k=16,
    )
    expected.backward()
    expected_grad = student_expected.grad.detach().clone()

    student_actual = student_expected.detach().clone().requires_grad_(True)
    actual = distill_losses.compute_eakld_topk(
        student_logits=student_actual,
        teacher_logits=teacher_logits,
        mask=mask,
        k=7,
        temperature=1.3,
        confidence_k=16,
    )
    actual.backward()

    assert torch.allclose(actual, expected.detach(), rtol=1e-6, atol=1e-6)
    assert student_actual.grad is not None
    assert torch.allclose(
        student_actual.grad,
        expected_grad,
        rtol=2e-5,
        atol=2e-6,
    )


def test_eakld_topk_chunked_entropy_matches_dense_output_and_gradient() -> None:
    torch.manual_seed(17)
    teacher_logits = torch.randn(2, 5, 23, dtype=torch.bfloat16)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )

    student_expected = torch.randn(
        2,
        5,
        23,
        dtype=torch.float32,
        requires_grad=True,
    )
    expected = _dense_eakld_topk_reference(
        student_logits=student_expected,
        teacher_logits=teacher_logits,
        mask=mask,
        k=7,
        temperature=1.3,
        confidence_k=16,
    )
    expected.backward()
    expected_grad = student_expected.grad.detach().clone()

    student_actual = student_expected.detach().clone().requires_grad_(True)
    actual = distill_losses.compute_eakld_topk(
        student_logits=student_actual,
        teacher_logits=teacher_logits,
        mask=mask,
        k=7,
        temperature=1.3,
        confidence_k=16,
    )
    actual.backward()

    assert torch.allclose(actual, expected.detach(), rtol=1e-6, atol=1e-6)
    assert student_actual.grad is not None
    assert torch.allclose(
        student_actual.grad,
        expected_grad,
        rtol=2e-5,
        atol=2e-6,
    )
    assert teacher_logits.grad is None


def test_dense_loss_dispatches_eakld_top_100_with_finite_backward() -> None:
    torch.manual_seed(23)
    student_logits = torch.randn(
        2,
        6,
        127,
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_logits = torch.randn(2, 6, 127, dtype=torch.bfloat16)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    loss = compute_dense_loss_from_logits(
        loss_type="eakld_top_100",
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=1.0,
        alpha=0.5,
        eakld_confidence_k=16,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_distill_mask_shifts_labels_for_causal_logits() -> None:
    reference_logits = torch.zeros(2, 5, 7, dtype=torch.bfloat16)
    labels = torch.tensor(
        [
            [-100, -100, 10, 11, 2],
            [-100, 20, -100, 21, -100],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    actual = distill_losses.build_distill_token_mask(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )

    expected = torch.tensor(
        [
            [0, 1, 1, 1, 0],
            [1, 0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    assert actual.dtype == torch.float32
    assert actual.device == reference_logits.device
    assert torch.equal(actual, expected)


def test_distill_mask_shifts_attention_mask_and_masks_final_logit() -> None:
    reference_logits = torch.zeros(2, 5, 7)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    actual = distill_losses.build_distill_token_mask(
        labels=None,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )

    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(actual, expected)


def test_distill_mask_defaults_to_positions_with_next_token() -> None:
    reference_logits = torch.zeros(2, 4, 9)

    actual = distill_losses.build_distill_token_mask(
        labels=None,
        attention_mask=None,
        reference_logits=reference_logits,
    )

    expected = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(actual, expected)


def test_distill_mask_single_token_sequence_has_no_target() -> None:
    reference_logits = torch.zeros(2, 1, 5)
    labels = torch.tensor([[2], [-100]], dtype=torch.long)

    from_labels = distill_losses.build_distill_token_mask(
        labels=labels,
        attention_mask=torch.ones(2, 1, dtype=torch.long),
        reference_logits=reference_logits,
    )
    without_metadata = distill_losses.build_distill_token_mask(
        labels=None,
        attention_mask=None,
        reference_logits=reference_logits,
    )

    expected = torch.zeros(2, 1, dtype=torch.float32)
    assert torch.equal(from_labels, expected)
    assert torch.equal(without_metadata, expected)


def test_forward_kl_gradient_uses_shifted_causal_positions() -> None:
    torch.manual_seed(41)
    student_logits = torch.randn(
        1,
        5,
        13,
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_logits = torch.randn(1, 5, 13, dtype=torch.float32)
    labels = torch.tensor([[-100, -100, 3, 4, 2]], dtype=torch.long)

    actual_mask = distill_losses.build_distill_token_mask(
        labels=labels,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        reference_logits=student_logits,
    )
    loss = distill_losses.compute_forward_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=actual_mask,
        temperature=1.0,
    )
    loss.backward()

    assert student_logits.grad is not None
    gradient_by_position = student_logits.grad.abs().sum(dim=-1).squeeze(0)
    assert gradient_by_position[0].item() == pytest.approx(0.0, abs=0.0)
    assert gradient_by_position[1].item() > 0.0
    assert gradient_by_position[2].item() > 0.0
    assert gradient_by_position[3].item() > 0.0
    assert gradient_by_position[4].item() == pytest.approx(0.0, abs=0.0)


def test_masked_logit_mse_matches_manual_valid_token_mean() -> None:
    student = torch.tensor(
        [[[1.0, 2.0], [9.0, 9.0], [3.0, 5.0]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[0.0, 0.0], [-99.0, 99.0], [1.0, 1.0]]],
    )
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    actual = distill_losses.compute_masked_logit_mse_loss(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
    )

    token0 = ((student[0, 0] - teacher[0, 0]) ** 2).mean()
    token2 = ((student[0, 2] - teacher[0, 2]) ** 2).mean()
    expected = (token0 + token2) / 2.0

    assert torch.allclose(actual, expected)


def test_mse_gradient_uses_shifted_causal_positions() -> None:
    student = torch.randn(1, 5, 13, requires_grad=True)
    teacher = torch.randn(1, 5, 13)
    labels = torch.tensor([[-100, -100, 3, 4, 2]])

    mask = distill_losses.build_distill_token_mask(
        labels=labels,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        reference_logits=student,
    )
    loss = distill_losses.compute_masked_logit_mse_loss(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
    )
    loss.backward()

    grad_by_position = student.grad.abs().sum(dim=-1).squeeze(0)
    assert grad_by_position[0].item() == pytest.approx(0.0, abs=0.0)
    assert grad_by_position[1].item() > 0.0
    assert grad_by_position[2].item() > 0.0
    assert grad_by_position[3].item() > 0.0
    assert grad_by_position[4].item() == pytest.approx(0.0, abs=0.0)


def test_masked_logit_mse_all_zero_mask_returns_differentiable_zero() -> None:
    student = torch.randn(2, 4, 7, requires_grad=True)
    teacher = torch.randn(2, 4, 7)
    mask = torch.zeros(2, 4)

    loss = distill_losses.compute_masked_logit_mse_loss(
        student_logits=student,
        teacher_logits=teacher,
        mask=mask,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=0.0)
    assert student.grad is not None
    assert torch.equal(student.grad, torch.zeros_like(student.grad))


def test_distill_mask_exactly_matches_next_label_validity() -> None:
    torch.manual_seed(53)
    batch_size = 4
    sequence_length = 9
    reference_logits = torch.zeros(batch_size, sequence_length, 17)
    labels = torch.randint(0, 17, (batch_size, sequence_length), dtype=torch.long)
    labels[0, :3] = -100
    labels[1, 2:5] = -100
    labels[2, -2:] = -100
    labels[3, :] = -100

    actual = distill_losses.build_distill_token_mask(
        labels=labels,
        attention_mask=torch.ones_like(labels),
        reference_logits=reference_logits,
    )

    expected = torch.zeros(batch_size, sequence_length, dtype=torch.float32)
    expected[:, :-1] = labels[:, 1:].ne(-100).to(dtype=torch.float32)
    assert torch.equal(actual, expected)


def _build_distill_regions(
    *,
    labels: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    reference_logits: torch.Tensor,
) -> distill_losses.DistillTokenRegions:
    return distill_losses.build_distill_token_regions(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )


def test_distill_regions_single_turn_splits_prompt_and_response() -> None:
    reference_logits = torch.zeros(1, 6, 11)
    labels = torch.tensor([[-100, -100, -100, 10, 11, 2]], dtype=torch.long)
    attention_mask = torch.ones(1, 6, dtype=torch.long)

    regions = _build_distill_regions(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )

    expected_response = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32)
    expected_prompt = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    assert torch.equal(regions.response_mask, expected_response)
    assert torch.equal(regions.prompt_mask, expected_prompt)


def test_distill_regions_padding_excludes_prompt_and_response() -> None:
    reference_logits = torch.zeros(1, 6, 11)
    labels = torch.tensor([[-100, -100, 10, 2, -100, -100]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)

    regions = _build_distill_regions(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )

    expected_response = torch.tensor([[0, 1, 1, 0, 0, 0]], dtype=torch.float32)
    expected_prompt = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    assert torch.equal(regions.response_mask, expected_response)
    assert torch.equal(regions.prompt_mask, expected_prompt)


def test_distill_regions_interleaved_prompt_and_response() -> None:
    reference_logits = torch.zeros(1, 5, 11)
    labels = torch.tensor([[-100, 10, -100, 11, 2]], dtype=torch.long)
    attention_mask = torch.ones(1, 5, dtype=torch.long)

    regions = _build_distill_regions(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )

    expected_response = torch.tensor([[1, 0, 1, 1, 0]], dtype=torch.float32)
    expected_prompt = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32)
    assert torch.equal(regions.response_mask, expected_response)
    assert torch.equal(regions.prompt_mask, expected_prompt)


def test_distill_regions_labels_none_keeps_response_and_zero_prompt() -> None:
    reference_logits = torch.zeros(2, 5, 7)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    regions = _build_distill_regions(
        labels=None,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )
    expected_response = distill_losses.build_distill_token_mask(
        labels=None,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )
    expected_prompt = torch.zeros(2, 5, dtype=torch.float32)

    assert torch.equal(regions.response_mask, expected_response)
    assert torch.equal(regions.prompt_mask, expected_prompt)

    reference_logits_no_metadata = torch.zeros(2, 4, 9)
    regions_no_metadata = _build_distill_regions(
        labels=None,
        attention_mask=None,
        reference_logits=reference_logits_no_metadata,
    )
    expected_response_no_metadata = distill_losses.build_distill_token_mask(
        labels=None,
        attention_mask=None,
        reference_logits=reference_logits_no_metadata,
    )
    expected_prompt_no_metadata = torch.zeros(2, 4, dtype=torch.float32)

    assert torch.equal(regions_no_metadata.response_mask, expected_response_no_metadata)
    assert torch.equal(regions_no_metadata.prompt_mask, expected_prompt_no_metadata)


def test_distill_regions_masks_are_binary_disjoint_with_zero_final() -> None:
    reference_logits = torch.zeros(2, 6, 11)
    labels = torch.tensor(
        [
            [-100, -100, -100, 10, 11, 2],
            [-100, 10, -100, 11, 2, -100],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    regions = _build_distill_regions(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=reference_logits,
    )

    for mask in (regions.response_mask, regions.prompt_mask):
        assert mask.dtype == torch.float32
        assert mask.device == reference_logits.device
        assert mask.shape == reference_logits.shape[:2]
        assert torch.all((mask == 0.0) | (mask == 1.0))
        assert torch.all(mask[:, -1] == 0.0)

    assert torch.all((regions.response_mask + regions.prompt_mask) <= 1.0)


@pytest.mark.parametrize("sequence_chunk_size", [1, 2, 5])
def test_cpu_teacher_eakld_fractional_mask_matches_dense_value_and_gradient(
    sequence_chunk_size: int,
) -> None:
    torch.manual_seed(102)
    teacher_logits = torch.randn(2, 7, 23, dtype=torch.float32)
    initial_student = torch.randn(2, 7, 23, dtype=torch.float32)
    mask = torch.tensor(
        [
            [0.1, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0],
            [1.0, 0.1, 0.0, 1.0, 1.0, 1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    gamma_cpu = distill_losses.compute_teacher_entropy_gamma(
        teacher_logits,
        mask,
        confidence_k=16,
    ).detach().cpu()

    dense_student = initial_student.detach().clone().requires_grad_(True)
    chunk_student = initial_student.detach().clone().requires_grad_(True)

    dense_loss = distill_losses.compute_eakld(
        student_logits=dense_student,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=1.3,
        confidence_k=16,
    )
    chunk_loss = distill_losses.compute_eakld_from_cpu_teacher_logits(
        student_logits=chunk_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        mask=mask,
        gamma=gamma_cpu,
        temperature=1.3,
        sequence_chunk_size=sequence_chunk_size,
    )

    dense_grad = torch.autograd.grad(dense_loss, dense_student)[0]
    chunk_grad = torch.autograd.grad(chunk_loss, chunk_student)[0]

    assert mask.sum().item() == pytest.approx(7.4)
    assert torch.allclose(chunk_loss, dense_loss, rtol=5e-6, atol=5e-6)
    assert torch.allclose(chunk_grad, dense_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sequence_chunk_size", [1, 2, 5])
def test_cpu_teacher_eakld_matches_dense_value_and_gradient(
    sequence_chunk_size: int,
) -> None:
    torch.manual_seed(101)
    teacher_logits = torch.randn(2, 7, 23, dtype=torch.float32)
    initial_student = torch.randn(2, 7, 23, dtype=torch.float32)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    gamma_cpu = distill_losses.compute_teacher_entropy_gamma(
        teacher_logits,
        mask,
        confidence_k=16,
    ).detach().cpu()

    dense_student = initial_student.detach().clone().requires_grad_(True)
    chunk_student = initial_student.detach().clone().requires_grad_(True)

    dense_loss = distill_losses.compute_eakld(
        student_logits=dense_student,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=1.3,
        confidence_k=16,
    )
    chunk_loss = distill_losses.compute_eakld_from_cpu_teacher_logits(
        student_logits=chunk_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        mask=mask,
        gamma=gamma_cpu,
        temperature=1.3,
        sequence_chunk_size=sequence_chunk_size,
    )

    dense_grad = torch.autograd.grad(dense_loss, dense_student)[0]
    chunk_grad = torch.autograd.grad(chunk_loss, chunk_student)[0]

    assert torch.allclose(chunk_loss, dense_loss, rtol=5e-6, atol=5e-6)
    assert torch.allclose(chunk_grad, dense_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sequence_chunk_size", [1, 3, 8])
def test_cpu_teacher_eakld_topk_fractional_mask_matches_dense_value_and_gradient(
    sequence_chunk_size: int,
) -> None:
    torch.manual_seed(104)
    teacher_logits = torch.randn(2, 6, 29, dtype=torch.float32)
    initial_student = torch.randn(2, 6, 29, dtype=torch.float32)
    mask = torch.tensor(
        [
            [0.1, 1.0, 1.0, 0.0, 1.0, 0.1],
            [1.0, 0.0, 0.1, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gamma_cpu = distill_losses.compute_teacher_entropy_gamma(
        teacher_logits,
        mask,
        confidence_k=16,
    ).detach().cpu()

    dense_student = initial_student.detach().clone().requires_grad_(True)
    chunk_student = initial_student.detach().clone().requires_grad_(True)

    dense_loss = distill_losses.compute_eakld_topk(
        student_logits=dense_student,
        teacher_logits=teacher_logits,
        mask=mask,
        k=7,
        temperature=0.9,
        confidence_k=16,
    )
    chunk_loss = distill_losses.compute_eakld_topk_from_cpu_teacher_logits(
        student_logits=chunk_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        mask=mask,
        gamma=gamma_cpu,
        k=7,
        temperature=0.9,
        sequence_chunk_size=sequence_chunk_size,
    )

    dense_grad = torch.autograd.grad(dense_loss, dense_student)[0]
    chunk_grad = torch.autograd.grad(chunk_loss, chunk_student)[0]

    assert mask.sum().item() == pytest.approx(6.3)
    assert torch.allclose(chunk_loss, dense_loss, rtol=5e-6, atol=5e-6)
    assert torch.allclose(chunk_grad, dense_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sequence_chunk_size", [1, 3, 8])
def test_cpu_teacher_eakld_topk_matches_dense_value_and_gradient(
    sequence_chunk_size: int,
) -> None:
    torch.manual_seed(103)
    teacher_logits = torch.randn(2, 6, 29, dtype=torch.float32)
    initial_student = torch.randn(2, 6, 29, dtype=torch.float32)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    gamma_cpu = distill_losses.compute_teacher_entropy_gamma(
        teacher_logits,
        mask,
        confidence_k=16,
    ).detach().cpu()

    dense_student = initial_student.detach().clone().requires_grad_(True)
    chunk_student = initial_student.detach().clone().requires_grad_(True)

    dense_loss = distill_losses.compute_eakld_topk(
        student_logits=dense_student,
        teacher_logits=teacher_logits,
        mask=mask,
        k=7,
        temperature=0.9,
        confidence_k=16,
    )
    chunk_loss = distill_losses.compute_eakld_topk_from_cpu_teacher_logits(
        student_logits=chunk_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        mask=mask,
        gamma=gamma_cpu,
        k=7,
        temperature=0.9,
        sequence_chunk_size=sequence_chunk_size,
    )

    dense_grad = torch.autograd.grad(dense_loss, dense_student)[0]
    chunk_grad = torch.autograd.grad(chunk_loss, chunk_student)[0]

    assert torch.allclose(chunk_loss, dense_loss, rtol=5e-6, atol=5e-6)
    assert torch.allclose(chunk_grad, dense_grad, rtol=1e-5, atol=1e-5)


def test_cpu_teacher_eakld_transfers_only_chunks_and_recomputation() -> None:
    torch.manual_seed(107)
    teacher_logits_cpu = torch.randn(2, 13, 11, dtype=torch.float32)
    student_logits = torch.randn(2, 13, 11, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(2, 13, dtype=torch.float32)
    gamma_cpu = distill_losses.compute_teacher_entropy_gamma(
        teacher_logits_cpu,
        mask,
        confidence_k=16,
    ).detach().cpu()

    transfer_calls: list[tuple[int, int, tuple[int, ...]]] = []
    original = distill_losses.copy_teacher_logit_chunk_to_device

    def wrapped(teacher_logits_cpu, *, start, end, target_device):
        out = original(
            teacher_logits_cpu,
            start=start,
            end=end,
            target_device=target_device,
        )
        transfer_calls.append((int(start), int(end), tuple(out.shape)))
        return out

    with mock.patch.object(
        distill_losses,
        "copy_teacher_logit_chunk_to_device",
        side_effect=wrapped,
    ):
        loss = distill_losses.compute_eakld_from_cpu_teacher_logits(
            student_logits=student_logits,
            teacher_logits_cpu=teacher_logits_cpu,
            mask=mask,
            gamma=gamma_cpu,
            temperature=1.0,
            sequence_chunk_size=4,
        )
        assert len(transfer_calls) == 4
        loss.backward()
        assert len(transfer_calls) == 8

    full_shape = tuple(teacher_logits_cpu.shape)
    for start, end, shape in transfer_calls:
        assert end - start <= 4
        assert shape[1] <= 4
        assert shape != full_shape


def test_cpu_teacher_eakld_rejects_invalid_inputs() -> None:
    teacher_cpu = torch.randn(2, 5, 7)
    student = torch.randn(2, 5, 7, requires_grad=True)
    mask = torch.ones(2, 5)
    gamma = torch.tensor(0.5)
    non_cpu_teacher = (
        teacher_cpu.cuda() if torch.cuda.is_available() else teacher_cpu.to("meta")
    )

    with pytest.raises(ValueError, match="reside on CPU"):
        distill_losses.compute_eakld_from_cpu_teacher_logits(
            student_logits=student,
            teacher_logits_cpu=non_cpu_teacher,
            mask=mask,
            gamma=gamma,
            temperature=1.0,
            sequence_chunk_size=2,
        )

    with pytest.raises(ValueError, match="shape mismatch"):
        distill_losses.compute_eakld_from_cpu_teacher_logits(
            student_logits=torch.randn(2, 4, 7, requires_grad=True),
            teacher_logits_cpu=teacher_cpu,
            mask=mask,
            gamma=gamma,
            temperature=1.0,
            sequence_chunk_size=2,
        )

    with pytest.raises(ValueError, match=r"\[B, L, V\]"):
        distill_losses.compute_eakld_from_cpu_teacher_logits(
            student_logits=torch.randn(2, 5, requires_grad=True),
            teacher_logits_cpu=torch.randn(2, 5),
            mask=None,
            gamma=gamma,
            temperature=1.0,
            sequence_chunk_size=2,
        )

    with pytest.raises(ValueError, match="sequence_chunk_size"):
        distill_losses.compute_eakld_from_cpu_teacher_logits(
            student_logits=student,
            teacher_logits_cpu=teacher_cpu,
            mask=mask,
            gamma=gamma,
            temperature=1.0,
            sequence_chunk_size=0,
        )

    with pytest.raises(ValueError, match="gamma"):
        distill_losses.compute_eakld_from_cpu_teacher_logits(
            student_logits=student,
            teacher_logits_cpu=teacher_cpu,
            mask=mask,
            gamma=torch.tensor([0.1, 0.2]),
            temperature=1.0,
            sequence_chunk_size=2,
        )

    with pytest.raises(ValueError, match="k must be"):
        distill_losses.compute_eakld_topk_from_cpu_teacher_logits(
            student_logits=student,
            teacher_logits_cpu=teacher_cpu,
            mask=mask,
            gamma=gamma,
            k=0,
            temperature=1.0,
            sequence_chunk_size=2,
        )


def _offloaded_teacher_loss_fixtures(
    *,
    loss_type: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    torch.manual_seed(211)
    teacher_logits = torch.randn(2, 6, 31, dtype=torch.float32)
    student_logits = torch.randn(2, 6, 31, dtype=torch.float32, requires_grad=True)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    entropy_mean, teacher_gamma, valid_count = (
        distill_losses.compute_teacher_entropy_mean_and_gamma(
            teacher_logits,
            mask,
            confidence_k=16,
        )
    )
    teacher_gamma_cpu = teacher_gamma.detach().cpu()
    teacher_entropy_mean_cpu = entropy_mean.detach().cpu()
    teacher_valid_token_count_cpu = valid_count.detach().cpu()
    ce_loss = None
    if loss_type == "eakld_kd":
        ce_loss = torch.tensor(1.25, dtype=torch.float32, requires_grad=True)
    return (
        student_logits,
        teacher_logits.cpu(),
        teacher_gamma_cpu,
        teacher_entropy_mean_cpu,
        teacher_valid_token_count_cpu,
        ce_loss,
    )


@pytest.mark.parametrize(
    "loss_type",
    ["eakld", "eakld_kd", "eakld_top_7", "eakld_topk_7"],
)
def test_offloaded_teacher_dense_loss_finite_backward(loss_type: str) -> None:
    (
        student_logits,
        teacher_logits_cpu,
        teacher_gamma_cpu,
        teacher_entropy_mean_cpu,
        teacher_valid_token_count_cpu,
        ce_loss,
    ) = _offloaded_teacher_loss_fixtures(loss_type=loss_type)

    loss = compute_dense_loss_from_offloaded_teacher(
        loss_type=loss_type,
        student_logits=student_logits,
        teacher_logits_cpu=teacher_logits_cpu,
        teacher_gamma_cpu=teacher_gamma_cpu,
        teacher_entropy_mean_cpu=teacher_entropy_mean_cpu,
        teacher_valid_token_count_cpu=teacher_valid_token_count_cpu,
        ce_loss=ce_loss,
        mask=torch.tensor(
            [
                [1, 1, 1, 1, 0, 1],
                [1, 0, 1, 1, 1, 0],
            ],
            dtype=torch.float32,
        ),
        temperature=1.1,
        alpha=0.4,
        eakld_confidence_k=16,
        sequence_chunk_size=3,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    if ce_loss is not None:
        assert ce_loss.grad is not None
        assert torch.isfinite(ce_loss.grad)


def test_offloaded_teacher_dense_loss_rejects_non_eakld() -> None:
    torch.manual_seed(213)
    student_logits = torch.randn(2, 4, 17, dtype=torch.float32, requires_grad=True)
    teacher_logits_cpu = torch.randn(2, 4, 17, dtype=torch.float32)
    teacher_gamma_cpu = torch.tensor(0.5, dtype=torch.float32)
    mask = torch.ones(2, 4, dtype=torch.float32)

    with pytest.raises(
        ValueError,
        match="teacher_output_offload=cpu supports only EAKLD-family losses.",
    ):
        compute_dense_loss_from_offloaded_teacher(
            loss_type="kl",
            student_logits=student_logits,
            teacher_logits_cpu=teacher_logits_cpu,
            teacher_gamma_cpu=teacher_gamma_cpu,
            teacher_entropy_mean_cpu=None,
            teacher_valid_token_count_cpu=None,
            mask=mask,
            temperature=1.0,
            alpha=0.5,
                eakld_confidence_k=16,
            sequence_chunk_size=2,
        )


# ---------------------------------------------------------------------------
# Task 2: region-level prompt KD dispatch tests
# ---------------------------------------------------------------------------

EAKLD_TELEMETRY_KEYS = {
    "teacher_entropy_mean",
    "gamma_reverse",
    "lambda_forward",
    "forward_kl",
    "reverse_kl",
    "eakld_total",
    "valid_tokens",
}


def _region_kl_reference(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    temperature: float,
    weight: float,
) -> torch.Tensor:
    response = distill_losses.compute_forward_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=response_mask,
        temperature=temperature,
    )
    prompt = distill_losses.compute_forward_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        mask=prompt_mask,
        temperature=temperature,
    )
    return response + weight * prompt


def test_forward_kl_region_combination_matches_manual_means() -> None:
    torch.manual_seed(301)
    student = torch.randn(2, 6, 19, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 6, 19, dtype=torch.float32)
    response_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float32
    )
    prompt_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=torch.float32
    )
    weight = 0.03

    actual = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=student,
        teacher_logits=teacher,
        mask=response_mask,
        temperature=1.0,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
    )
    expected = _region_kl_reference(
        student_logits=student,
        teacher_logits=teacher,
        response_mask=response_mask,
        prompt_mask=prompt_mask,
        temperature=1.0,
        weight=weight,
    )
    assert torch.allclose(actual, expected.detach(), rtol=1e-6, atol=1e-6)


def test_forward_kl_prompt_region_mean_invariant_to_prompt_repetition() -> None:
    """Prompt-region mean must not change when prompt tokens are repeated.

    This distinguishes the region-normalized formula (each region divides by its
    own token count) from a shared weighted-denominator formula. The response
    region has non-empty tokens whose per-token KL differs from the prompt
    region's per-token KL, so the old shared-denominator loss would drift as the
    prompt length P changes, while the new region-normalized loss stays constant.
    """
    torch.manual_seed(302)
    vocab = 17
    # Distinct logits for response vs prompt positions so per-token KL differs
    # between the two regions.
    response_student = torch.randn(1, 1, vocab, dtype=torch.float32) * 2.0
    response_teacher = torch.randn(1, 1, vocab, dtype=torch.float32) * 2.0
    prompt_student = torch.randn(1, 1, vocab, dtype=torch.float32) * 0.5
    prompt_teacher = torch.randn(1, 1, vocab, dtype=torch.float32) * 0.5
    n_response = 2
    short_n_prompt = 3
    long_n_prompt = 30

    def _build(n_prompt):
        resp_s = response_student.expand(1, n_response, vocab).clone()
        resp_t = response_teacher.expand(1, n_response, vocab).clone()
        prom_s = prompt_student.expand(1, n_prompt, vocab).clone()
        prom_t = prompt_teacher.expand(1, n_prompt, vocab).clone()
        student = torch.cat([resp_s, prom_s], dim=1).requires_grad_(True)
        teacher = torch.cat([resp_t, prom_t], dim=1)
        response_mask = torch.cat(
            [torch.ones(1, n_response), torch.zeros(1, n_prompt)], dim=1
        ).to(dtype=torch.float32)
        prompt_mask = torch.cat(
            [torch.zeros(1, n_response), torch.ones(1, n_prompt)], dim=1
        ).to(dtype=torch.float32)
        return student, teacher, response_mask, prompt_mask

    short_student, short_teacher, short_response, short_prompt_mask = _build(
        short_n_prompt
    )
    long_student, long_teacher, long_response, long_prompt_mask = _build(
        long_n_prompt
    )

    weight = 0.03
    short_loss = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=short_student,
        teacher_logits=short_teacher,
        mask=short_response,
        temperature=1.0,
        prompt_mask=short_prompt_mask,
        prompt_kd_weight=weight,
    )
    long_loss = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=long_student,
        teacher_logits=long_teacher,
        mask=long_response,
        temperature=1.0,
        prompt_mask=long_prompt_mask,
        prompt_kd_weight=weight,
    )

    # New region-normalized loss is invariant to prompt length P.
    assert torch.isfinite(short_loss)
    assert torch.isfinite(long_loss)
    assert torch.allclose(short_loss, long_loss, rtol=1e-6, atol=1e-6)

    # And it equals response_mean + w * prompt_mean computed independently.
    response_mean = distill_losses.compute_forward_kl_loss(
        student_logits=short_student,
        teacher_logits=short_teacher,
        mask=short_response,
        temperature=1.0,
    )
    prompt_mean = distill_losses.compute_forward_kl_loss(
        student_logits=short_student,
        teacher_logits=short_teacher,
        mask=short_prompt_mask,
        temperature=1.0,
    )
    expected = response_mean + weight * prompt_mean
    assert torch.allclose(short_loss, expected.detach(), rtol=1e-6, atol=1e-6)

    # Sanity check: the old shared-denominator formula WOULD change with P.
    def _old_shared_denominator(student, teacher, response_mask, prompt_mask, w):
        log_p = F.log_softmax(student.float(), dim=-1)
        t_p = F.softmax(teacher.float(), dim=-1)
        token_kl = F.kl_div(log_p, t_p, reduction="none").sum(dim=-1)
        weighted_mask = response_mask + w * prompt_mask
        denom = weighted_mask.sum().clamp_min(1.0)
        return (token_kl * weighted_mask).sum() / denom

    old_short = _old_shared_denominator(
        short_student, short_teacher, short_response, short_prompt_mask, weight
    )
    old_long = _old_shared_denominator(
        long_student, long_teacher, long_response, long_prompt_mask, weight
    )
    assert not torch.allclose(old_short, old_long, rtol=1e-4, atol=1e-4)


def test_zero_prompt_weight_matches_response_only_value_and_gradient() -> None:
    torch.manual_seed(303)
    student_base = torch.randn(2, 6, 19, dtype=torch.float32)
    teacher = torch.randn(2, 6, 19, dtype=torch.float32)
    response_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float32
    )
    prompt_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=torch.float32
    )

    response_only_student = student_base.detach().clone().requires_grad_(True)
    response_only = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=response_only_student,
        teacher_logits=teacher,
        mask=response_mask,
        temperature=1.0,
    )
    response_only.backward()
    response_only_grad = response_only_student.grad.detach().clone()

    region_student = student_base.detach().clone().requires_grad_(True)
    region = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=region_student,
        teacher_logits=teacher,
        mask=response_mask,
        temperature=1.0,
        prompt_mask=prompt_mask,
        prompt_kd_weight=0.0,
    )
    region.backward()
    region_grad = region_student.grad.detach().clone()

    assert torch.allclose(region, response_only.detach(), rtol=1e-6, atol=1e-7)
    assert torch.allclose(region_grad, response_only_grad, rtol=1e-6, atol=1e-7)


def test_empty_prompt_mask_with_positive_weight_contributes_zero() -> None:
    torch.manual_seed(304)
    student = torch.randn(2, 6, 19, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 6, 19, dtype=torch.float32)
    response_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float32
    )
    empty_prompt = torch.zeros(2, 6, dtype=torch.float32)

    loss = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=student,
        teacher_logits=teacher,
        mask=response_mask,
        temperature=1.0,
        prompt_mask=empty_prompt,
        prompt_kd_weight=0.03,
    )
    response_only = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=student.detach().clone().requires_grad_(True),
        teacher_logits=teacher,
        mask=response_mask,
        temperature=1.0,
    )
    assert torch.isfinite(loss)
    assert torch.allclose(loss, response_only.detach(), rtol=1e-6, atol=1e-7)


def test_empty_response_mask_leaves_only_weighted_prompt_loss_kl() -> None:
    torch.manual_seed(305)
    student = torch.randn(2, 6, 19, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 6, 19, dtype=torch.float32)
    empty_response = torch.zeros(2, 6, dtype=torch.float32)
    prompt_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0]], dtype=torch.float32
    )
    weight = 0.03

    loss = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=student,
        teacher_logits=teacher,
        mask=empty_response,
        temperature=1.0,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
    )
    prompt_only = distill_losses.compute_forward_kl_loss(
        student_logits=student,
        teacher_logits=teacher,
        mask=prompt_mask,
        temperature=1.0,
    )
    assert torch.isfinite(loss)
    assert torch.allclose(loss, (weight * prompt_only).detach(), rtol=1e-6, atol=1e-7)


def test_empty_response_mask_leaves_only_weighted_prompt_loss_eakld() -> None:
    torch.manual_seed(306)
    student = torch.randn(2, 6, 19, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 6, 19, dtype=torch.float32)
    empty_response = torch.zeros(2, 6, dtype=torch.float32)
    prompt_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0]], dtype=torch.float32
    )
    weight = 0.03

    loss = compute_dense_loss_from_logits(
        loss_type="eakld",
        student_logits=student,
        teacher_logits=teacher,
        mask=empty_response,
        temperature=1.0,
        eakld_confidence_k=16,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
    )
    prompt_only = distill_losses.compute_eakld(
        student_logits=student,
        teacher_logits=teacher,
        mask=prompt_mask,
        temperature=1.0,
        confidence_k=16,
        telemetry_out=None,
    )
    assert torch.isfinite(loss)
    assert torch.allclose(loss, (weight * prompt_only).detach(), rtol=1e-5, atol=1e-6)


def test_empty_prompt_mask_eakld_positive_weight_remains_finite() -> None:
    torch.manual_seed(307)
    student = torch.randn(2, 6, 19, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 6, 19, dtype=torch.float32)
    response_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float32
    )
    empty_prompt = torch.zeros(2, 6, dtype=torch.float32)

    loss = compute_dense_loss_from_logits(
        loss_type="eakld",
        student_logits=student,
        teacher_logits=teacher,
        mask=response_mask,
        temperature=1.0,
        eakld_confidence_k=16,
        prompt_mask=empty_prompt,
        prompt_kd_weight=0.03,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_negative_prompt_weight_is_error() -> None:
    student = torch.randn(2, 4, 7, requires_grad=True)
    teacher = torch.randn(2, 4, 7)
    mask = torch.ones(2, 4)
    prompt_mask = torch.ones(2, 4)
    with pytest.raises(ValueError, match="prompt_kd_weight must be >= 0.0"):
        compute_dense_loss_from_logits(
            loss_type="kl",
            student_logits=student,
            teacher_logits=teacher,
            mask=mask,
            temperature=1.0,
            prompt_mask=prompt_mask,
            prompt_kd_weight=-0.01,
        )


def test_positive_weight_without_prompt_mask_is_error() -> None:
    student = torch.randn(2, 4, 7, requires_grad=True)
    teacher = torch.randn(2, 4, 7)
    mask = torch.ones(2, 4)
    with pytest.raises(ValueError, match="prompt_kd_weight > 0 requires prompt_mask"):
        compute_dense_loss_from_logits(
            loss_type="kl",
            student_logits=student,
            teacher_logits=teacher,
            mask=mask,
            temperature=1.0,
            prompt_kd_weight=0.03,
        )


def test_kd_ce_not_double_counted_across_regions() -> None:
    torch.manual_seed(308)
    student = torch.randn(2, 6, 19, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 6, 19, dtype=torch.float32)
    response_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float32
    )
    prompt_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=torch.float32
    )
    alpha = 0.5
    weight = 0.03

    ce_tensor = torch.tensor(1000.0, dtype=torch.float32, requires_grad=True)
    loss = compute_dense_loss_from_logits(
        loss_type="kd",
        student_logits=student,
        teacher_logits=teacher,
        ce_loss=ce_tensor,
        mask=response_mask,
        temperature=1.0,
        alpha=alpha,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
    )
    loss.backward()
    # CE appears exactly once: gradient of loss w.r.t. ce_tensor is (1 - alpha).
    assert ce_tensor.grad is not None
    assert torch.allclose(ce_tensor.grad, torch.tensor(1.0 - alpha), rtol=1e-6, atol=1e-7)


def test_eakld_prompt_call_does_not_overwrite_response_telemetry() -> None:
    torch.manual_seed(309)
    student = torch.randn(2, 6, 19, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 6, 19, dtype=torch.float32)
    response_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float32
    )
    prompt_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=torch.float32
    )

    telemetry: dict[str, torch.Tensor] = {}
    loss = compute_dense_loss_from_logits(
        loss_type="eakld",
        student_logits=student,
        teacher_logits=teacher,
        mask=response_mask,
        temperature=1.0,
        eakld_confidence_k=16,
        telemetry_out=telemetry,
        prompt_mask=prompt_mask,
        prompt_kd_weight=0.03,
    )
    loss.backward()

    # Telemetry must reflect the response-region EAKLD only.
    response_only_telemetry: dict[str, torch.Tensor] = {}
    distill_losses.compute_eakld(
        student_logits=student.detach().clone().requires_grad_(True),
        teacher_logits=teacher,
        mask=response_mask,
        temperature=1.0,
        confidence_k=16,
        telemetry_out=response_only_telemetry,
    )
    assert set(telemetry) == EAKLD_TELEMETRY_KEYS
    for key in EAKLD_TELEMETRY_KEYS:
        assert torch.allclose(
            telemetry[key], response_only_telemetry[key], rtol=1e-6, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Task 5: dense-vs-offload equality with positive prompt weight
# ---------------------------------------------------------------------------

def _prompt_region_scalar_fixtures(
    *,
    seed: int,
    batch: int,
    seq_len: int,
    vocab: int,
):
    torch.manual_seed(seed)
    teacher_logits = torch.randn(batch, seq_len, vocab, dtype=torch.float32)
    student_base = torch.randn(batch, seq_len, vocab, dtype=torch.float32)
    response_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float32
    )[:batch, :seq_len]
    prompt_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=torch.float32
    )[:batch, :seq_len]

    (
        resp_entropy,
        resp_gamma,
        resp_valid,
    ) = distill_losses.compute_teacher_entropy_mean_and_gamma(
        teacher_logits, response_mask, confidence_k=16
    )
    (
        prompt_entropy,
        prompt_gamma,
        prompt_valid,
    ) = distill_losses.compute_teacher_entropy_mean_and_gamma(
        teacher_logits, prompt_mask, confidence_k=16
    )
    return (
        student_base,
        teacher_logits,
        response_mask,
        prompt_mask,
        resp_gamma.detach().cpu(),
        resp_entropy.detach().cpu(),
        resp_valid.detach().cpu(),
        prompt_gamma.detach().cpu(),
        prompt_entropy.detach().cpu(),
        prompt_valid.detach().cpu(),
    )


@pytest.mark.parametrize("sequence_chunk_size", [1, 2, 3])
def test_offloaded_eakld_positive_prompt_weight_matches_dense_value_and_gradient(
    sequence_chunk_size: int,
) -> None:
    weight = 0.07
    (
        student_base,
        teacher_logits,
        response_mask,
        prompt_mask,
        resp_gamma_cpu,
        resp_entropy_cpu,
        resp_valid_cpu,
        prompt_gamma_cpu,
        prompt_entropy_cpu,
        prompt_valid_cpu,
    ) = _prompt_region_scalar_fixtures(
        seed=401, batch=2, seq_len=6, vocab=23
    )

    dense_student = student_base.detach().clone().requires_grad_(True)
    dense_loss = compute_dense_loss_from_logits(
        loss_type="eakld",
        student_logits=dense_student,
        teacher_logits=teacher_logits,
        mask=response_mask,
        temperature=1.3,
        eakld_confidence_k=16,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
    )
    dense_grad = torch.autograd.grad(dense_loss, dense_student)[0]

    offload_student = student_base.detach().clone().requires_grad_(True)
    offload_loss = compute_dense_loss_from_offloaded_teacher(
        loss_type="eakld",
        student_logits=offload_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        teacher_gamma_cpu=resp_gamma_cpu,
        teacher_entropy_mean_cpu=resp_entropy_cpu,
        teacher_valid_token_count_cpu=resp_valid_cpu,
        mask=response_mask,
        temperature=1.3,
        eakld_confidence_k=16,
        sequence_chunk_size=sequence_chunk_size,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
        teacher_prompt_gamma_cpu=prompt_gamma_cpu,
        teacher_prompt_entropy_mean_cpu=prompt_entropy_cpu,
        teacher_prompt_valid_token_count_cpu=prompt_valid_cpu,
    )
    offload_grad = torch.autograd.grad(offload_loss, offload_student)[0]

    assert torch.allclose(offload_loss, dense_loss.detach(), rtol=5e-6, atol=5e-6)
    assert torch.allclose(offload_grad, dense_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sequence_chunk_size", [1, 3, 8])
def test_offloaded_eakld_topk_positive_prompt_weight_matches_dense_value_and_gradient(
    sequence_chunk_size: int,
) -> None:
    weight = 0.11
    (
        student_base,
        teacher_logits,
        response_mask,
        prompt_mask,
        resp_gamma_cpu,
        resp_entropy_cpu,
        resp_valid_cpu,
        prompt_gamma_cpu,
        prompt_entropy_cpu,
        prompt_valid_cpu,
    ) = _prompt_region_scalar_fixtures(
        seed=402, batch=2, seq_len=6, vocab=29
    )

    dense_student = student_base.detach().clone().requires_grad_(True)
    dense_loss = compute_dense_loss_from_logits(
        loss_type="eakld_top_7",
        student_logits=dense_student,
        teacher_logits=teacher_logits,
        mask=response_mask,
        temperature=0.9,
        eakld_confidence_k=16,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
    )
    dense_grad = torch.autograd.grad(dense_loss, dense_student)[0]

    offload_student = student_base.detach().clone().requires_grad_(True)
    offload_loss = compute_dense_loss_from_offloaded_teacher(
        loss_type="eakld_top_7",
        student_logits=offload_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        teacher_gamma_cpu=resp_gamma_cpu,
        teacher_entropy_mean_cpu=resp_entropy_cpu,
        teacher_valid_token_count_cpu=resp_valid_cpu,
        mask=response_mask,
        temperature=0.9,
        eakld_confidence_k=16,
        sequence_chunk_size=sequence_chunk_size,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
        teacher_prompt_gamma_cpu=prompt_gamma_cpu,
        teacher_prompt_entropy_mean_cpu=prompt_entropy_cpu,
        teacher_prompt_valid_token_count_cpu=prompt_valid_cpu,
    )
    offload_grad = torch.autograd.grad(offload_loss, offload_student)[0]

    assert torch.allclose(offload_loss, dense_loss.detach(), rtol=5e-6, atol=5e-6)
    assert torch.allclose(offload_grad, dense_grad, rtol=1e-5, atol=1e-5)


def test_offloaded_zero_prompt_weight_matches_response_only_value_and_gradient() -> None:
    """Zero weight: prompt scalar fields unnecessary; matches response-only."""
    (
        student_base,
        teacher_logits,
        response_mask,
        prompt_mask,
        resp_gamma_cpu,
        resp_entropy_cpu,
        resp_valid_cpu,
        _prompt_gamma_cpu,
        _prompt_entropy_cpu,
        _prompt_valid_cpu,
    ) = _prompt_region_scalar_fixtures(
        seed=403, batch=2, seq_len=6, vocab=23
    )

    response_only_student = student_base.detach().clone().requires_grad_(True)
    response_only = compute_dense_loss_from_offloaded_teacher(
        loss_type="eakld",
        student_logits=response_only_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        teacher_gamma_cpu=resp_gamma_cpu,
        teacher_entropy_mean_cpu=resp_entropy_cpu,
        teacher_valid_token_count_cpu=resp_valid_cpu,
        mask=response_mask,
        temperature=1.3,
        eakld_confidence_k=16,
        sequence_chunk_size=2,
    )
    response_only.backward()
    response_only_grad = response_only_student.grad.detach().clone()

    region_student = student_base.detach().clone().requires_grad_(True)
    region = compute_dense_loss_from_offloaded_teacher(
        loss_type="eakld",
        student_logits=region_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        teacher_gamma_cpu=resp_gamma_cpu,
        teacher_entropy_mean_cpu=resp_entropy_cpu,
        teacher_valid_token_count_cpu=resp_valid_cpu,
        mask=response_mask,
        temperature=1.3,
        eakld_confidence_k=16,
        sequence_chunk_size=2,
        prompt_mask=prompt_mask,
        prompt_kd_weight=0.0,
    )
    region.backward()
    region_grad = region_student.grad.detach().clone()

    assert torch.allclose(region, response_only.detach(), rtol=1e-6, atol=1e-7)
    assert torch.allclose(region_grad, response_only_grad, rtol=1e-6, atol=1e-7)


def test_offloaded_empty_prompt_mask_positive_weight_remains_finite() -> None:
    """Empty prompt mask + positive weight + scalars computed on empty mask."""
    torch.manual_seed(404)
    teacher_logits = torch.randn(2, 6, 19, dtype=torch.float32)
    student = torch.randn(2, 6, 19, dtype=torch.float32, requires_grad=True)
    response_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float32
    )
    empty_prompt = torch.zeros(2, 6, dtype=torch.float32)

    (
        resp_entropy,
        resp_gamma,
        resp_valid,
    ) = distill_losses.compute_teacher_entropy_mean_and_gamma(
        teacher_logits, response_mask, confidence_k=16
    )
    (
        prompt_entropy,
        prompt_gamma,
        prompt_valid,
    ) = distill_losses.compute_teacher_entropy_mean_and_gamma(
        teacher_logits, empty_prompt, confidence_k=16
    )

    loss = compute_dense_loss_from_offloaded_teacher(
        loss_type="eakld",
        student_logits=student,
        teacher_logits_cpu=teacher_logits.cpu(),
        teacher_gamma_cpu=resp_gamma.detach().cpu(),
        teacher_entropy_mean_cpu=resp_entropy.detach().cpu(),
        teacher_valid_token_count_cpu=resp_valid.detach().cpu(),
        mask=response_mask,
        temperature=1.0,
        eakld_confidence_k=16,
        sequence_chunk_size=2,
        prompt_mask=empty_prompt,
        prompt_kd_weight=0.03,
        teacher_prompt_gamma_cpu=prompt_gamma.detach().cpu(),
        teacher_prompt_entropy_mean_cpu=prompt_entropy.detach().cpu(),
        teacher_prompt_valid_token_count_cpu=prompt_valid.detach().cpu(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_offloaded_positive_weight_without_prompt_scalars_is_error() -> None:
    torch.manual_seed(405)
    student = torch.randn(2, 4, 11, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(2, 4, 11, dtype=torch.float32)
    mask = torch.ones(2, 4, dtype=torch.float32)
    prompt_mask = torch.ones(2, 4, dtype=torch.float32)
    gamma_cpu = distill_losses.compute_teacher_entropy_mean_and_gamma(
        teacher, mask, confidence_k=16
    )[1].detach().cpu()

    with pytest.raises(
        ValueError,
        match="prompt_kd_weight > 0 requires teacher_prompt_gamma_cpu",
    ):
        compute_dense_loss_from_offloaded_teacher(
            loss_type="eakld",
            student_logits=student,
            teacher_logits_cpu=teacher.cpu(),
            teacher_gamma_cpu=gamma_cpu,
            teacher_entropy_mean_cpu=None,
            teacher_valid_token_count_cpu=None,
            mask=mask,
            temperature=1.0,
            eakld_confidence_k=16,
            sequence_chunk_size=2,
            prompt_mask=prompt_mask,
            prompt_kd_weight=0.03,
        )


def test_offloaded_eakld_kd_positive_prompt_weight_mixes_ce_once() -> None:
    """eakld_kd mixes CE exactly once after regional EAKLD combination."""
    weight = 0.07
    alpha = 0.4
    (
        student_base,
        teacher_logits,
        response_mask,
        prompt_mask,
        resp_gamma_cpu,
        resp_entropy_cpu,
        resp_valid_cpu,
        prompt_gamma_cpu,
        prompt_entropy_cpu,
        prompt_valid_cpu,
    ) = _prompt_region_scalar_fixtures(
        seed=406, batch=2, seq_len=6, vocab=23
    )

    ce_tensor = torch.tensor(1000.0, dtype=torch.float32, requires_grad=True)
    offload_student = student_base.detach().clone().requires_grad_(True)
    loss = compute_dense_loss_from_offloaded_teacher(
        loss_type="eakld_kd",
        student_logits=offload_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        teacher_gamma_cpu=resp_gamma_cpu,
        teacher_entropy_mean_cpu=resp_entropy_cpu,
        teacher_valid_token_count_cpu=resp_valid_cpu,
        ce_loss=ce_tensor,
        mask=response_mask,
        temperature=1.3,
        alpha=alpha,
        eakld_confidence_k=16,
        sequence_chunk_size=2,
        prompt_mask=prompt_mask,
        prompt_kd_weight=weight,
        teacher_prompt_gamma_cpu=prompt_gamma_cpu,
        teacher_prompt_entropy_mean_cpu=prompt_entropy_cpu,
        teacher_prompt_valid_token_count_cpu=prompt_valid_cpu,
    )
    loss.backward()
    assert ce_tensor.grad is not None
    assert torch.allclose(
        ce_tensor.grad, torch.tensor(1.0 - alpha), rtol=1e-6, atol=1e-7
    )

