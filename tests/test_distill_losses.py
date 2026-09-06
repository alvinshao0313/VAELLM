from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from e2e_common.dense_loss import (
    compute_dense_loss_from_logits,
    compute_dense_loss_from_offloaded_teacher,
)
from train_utils import distill_losses
from train_utils.distill_loss_core import MODEL_LEVEL_LOSS_TYPES, compute_model_level_loss


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


def _tiny_batch(seed: int = 0):
    torch.manual_seed(seed)
    student = torch.randn(2, 5, 11, dtype=torch.float32)
    teacher = torch.randn(2, 5, 11, dtype=torch.float32)
    input_ids = torch.randint(0, 11, (2, 5), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100
    attention = torch.ones(2, 5, dtype=torch.long)
    attention[:, -1] = 0
    return student, teacher, input_ids, labels, attention


@pytest.mark.parametrize("loss_type", MODEL_LEVEL_LOSS_TYPES)
def test_dense_wrapper_five_loss_types_finite(loss_type: str):
    student, teacher, input_ids, labels, attention = _tiny_batch()
    student = student.detach().requires_grad_(True)
    loss = compute_dense_loss_from_logits(
        loss_type=loss_type,
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        teacher_logits=None if loss_type == "sft" else teacher,
        temperature=1.5,
        alpha=0.4,
        top_k=3,
        prompt_loss_weight=0.5,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None


def test_weighted_prompt_reduction_matches_core():
    student, teacher, input_ids, labels, attention = _tiny_batch(3)
    dense = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        teacher_logits=teacher,
        prompt_loss_weight=2.0,
    )
    core = compute_model_level_loss(
        loss_type="kl",
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        teacher_logits=teacher,
        prompt_loss_weight=2.0,
    )
    torch.testing.assert_close(dense, core)


def test_offloaded_kl_matches_dense():
    student, teacher, input_ids, labels, attention = _tiny_batch(4)
    student_d = student.detach().requires_grad_(True)
    dense = compute_dense_loss_from_logits(
        loss_type="kl",
        student_logits=student_d,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        teacher_logits=teacher,
        prompt_loss_weight=0.25,
    )
    dense.backward()
    student_o = student.detach().requires_grad_(True)
    off = compute_dense_loss_from_offloaded_teacher(
        loss_type="kl",
        student_logits=student_o,
        teacher_logits_cpu=teacher.cpu(),
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        prompt_loss_weight=0.25,
    )
    off.backward()
    torch.testing.assert_close(off, dense.detach())
    torch.testing.assert_close(student_o.grad, student_d.grad)


def test_kd_blends_separately_reduced_ce_and_kl():
    student, teacher, input_ids, labels, attention = _tiny_batch(5)
    alpha = 0.3
    kd = compute_model_level_loss(
        loss_type="kd",
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        teacher_logits=teacher,
        alpha=alpha,
        prompt_loss_weight=1.0,
    )
    ce = compute_model_level_loss(
        loss_type="sft",
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        prompt_loss_weight=1.0,
    )
    kl = compute_model_level_loss(
        loss_type="kl",
        student_logits=student,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        teacher_logits=teacher,
        prompt_loss_weight=1.0,
    )
    torch.testing.assert_close(kd, (1 - alpha) * ce + alpha * kl)


def test_model_level_deleted_types_rejected_by_dense_wrapper():
    student, teacher, input_ids, labels, attention = _tiny_batch()
    for removed in ("origin", "rkl", "dual_rkl", "eakld", "mse", "choice_kd"):
        with pytest.raises(ValueError, match="Unsupported"):
            compute_dense_loss_from_logits(
                loss_type=removed,
                student_logits=student,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention,
                teacher_logits=teacher,
            )
