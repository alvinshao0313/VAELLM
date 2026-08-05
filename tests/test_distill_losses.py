from __future__ import annotations

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from e2e_common.dense_loss import compute_dense_loss_from_logits
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

def _dense_eakld_topk_reference(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    temperature: float,
    confidence_k: int,
    post_attn: bool,
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
        post_attn=bool(post_attn),
    ) * (temp * temp)
    forward_kl = distill_losses._topk_forward_kl_mean(
        student_scaled=student_scaled,
        teacher_scaled=teacher_scaled,
        mask=mask,
        k=int(k),
        post_attn=bool(post_attn),
    ) * (temp * temp)
    return gamma * reverse_kl + (1.0 - gamma) * forward_kl

@pytest.mark.parametrize("post_attn", [False, True])
def test_eakld_topk_chunked_entropy_matches_dense_output_and_gradient(
    post_attn: bool,
) -> None:
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
        post_attn=post_attn,
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
        post_attn=post_attn,
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
        post_attn=False,
        eakld_confidence_k=16,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()

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

@pytest.mark.parametrize("post_attn", [False, True])
@pytest.mark.parametrize("sequence_chunk_size", [1, 3, 8])
def test_cpu_teacher_eakld_topk_matches_dense_value_and_gradient(
    post_attn: bool,
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
        post_attn=post_attn,
    )
    chunk_loss = distill_losses.compute_eakld_topk_from_cpu_teacher_logits(
        student_logits=chunk_student,
        teacher_logits_cpu=teacher_logits.cpu(),
        mask=mask,
        gamma=gamma_cpu,
        k=7,
        temperature=0.9,
        post_attn=post_attn,
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
