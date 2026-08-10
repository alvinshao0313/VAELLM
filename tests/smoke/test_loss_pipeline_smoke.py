"""Minimal numerical smoke tests for distill loss dispatcher paths.

No large model download. Uses tiny synthetic logits (B=2, L=6, V=31).
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from e2e_common.dense_loss import (
    compute_dense_loss_from_logits,
    compute_dense_loss_from_offloaded_teacher,
)
from train_utils import distill_losses
from train_utils.block_distill import _attention_map_kl_chunk_losses
from train_utils.distill_losses import build_distill_token_mask


EAKLD_TELEMETRY_KEYS = {
    "teacher_entropy_mean",
    "gamma_reverse",
    "lambda_forward",
    "forward_kl",
    "reverse_kl",
    "eakld_total",
    "valid_tokens",
}

DENSE_LOSS_TYPES = (
    "kl",
    "rkl",
    "dual_kl",
    "dual_rkl",
    "eakld",
    "eakld_top_7",
    "mse",
)

def _make_tiny_logits_and_labels(
    *,
    seed: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build B=2, L=6, V=31 logits with prompt/completion/pad labels."""
    torch.manual_seed(seed)
    batch, seq_len, vocab = 2, 6, 31
    teacher_logits = torch.randn(batch, seq_len, vocab, dtype=torch.float32)
    student_logits = (
        teacher_logits + torch.randn(batch, seq_len, vocab, dtype=torch.float32) * 0.75
    ).requires_grad_(True)

    # labels: prompt (-100), completion (token ids), padding (-100)
    # positions: 0 1 2 3 4 5
    labels = torch.tensor(
        [
            [-100, -100, 3, 7, 11, -100],
            [-100, 5, 9, 13, -100, -100],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    assert not torch.equal(student_logits.detach(), teacher_logits)
    assert student_logits.requires_grad
    assert not teacher_logits.requires_grad
    return student_logits, teacher_logits, labels, attention_mask


def test_dense_dispatcher_loss_pipeline_smoke() -> None:
    student_base, teacher_logits, labels, attention_mask = _make_tiny_logits_and_labels()
    mask = build_distill_token_mask(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=student_base,
        prompt_kd_weight=0.1,
    )
    assert mask.shape == (2, 6)
    assert float(mask.sum().item()) > 0.0
    assert bool(((mask > 0.0) & (mask < 1.0)).any().item())
    zero_weight_positions = mask.eq(0)

    for loss_type in DENSE_LOSS_TYPES:
        student_logits = student_base.detach().clone().requires_grad_(True)
        telemetry: dict[str, torch.Tensor] = {}
        loss = compute_dense_loss_from_logits(
            loss_type=loss_type,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            temperature=1.0,
            eakld_confidence_k=16,
            telemetry_out=telemetry if loss_type.startswith("eakld") else None,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)

        loss.backward()
        assert student_logits.grad is not None
        assert torch.isfinite(student_logits.grad).all()
        assert torch.equal(
            student_logits.grad[zero_weight_positions],
            torch.zeros_like(student_logits.grad[zero_weight_positions]),
        )
        assert teacher_logits.grad is None
        assert not teacher_logits.requires_grad

        if loss_type.startswith("eakld"):
            assert set(telemetry) == EAKLD_TELEMETRY_KEYS
            for key, value in telemetry.items():
                assert value.ndim == 0
                assert value.requires_grad is False
                assert torch.isfinite(value)


def test_offload_cpu_eakld_dispatcher_smoke() -> None:
    student_base, teacher_logits, labels, attention_mask = _make_tiny_logits_and_labels(
        seed=21,
    )
    mask = build_distill_token_mask(
        labels=labels,
        attention_mask=attention_mask,
        reference_logits=student_base,
        prompt_kd_weight=0.1,
    )
    assert bool(((mask > 0.0) & (mask < 1.0)).any().item())
    entropy_mean, gamma, valid_count = (
        distill_losses.compute_teacher_entropy_mean_and_gamma(
            teacher_logits,
            mask,
            confidence_k=16,
        )
    )
    teacher_logits_cpu = teacher_logits.detach().cpu()
    teacher_gamma_cpu = gamma.detach().cpu()
    teacher_entropy_mean_cpu = entropy_mean.detach().cpu()
    teacher_valid_token_count_cpu = valid_count.detach().cpu()
    assert teacher_logits_cpu.device.type == "cpu"
    assert teacher_gamma_cpu.device.type == "cpu"
    assert teacher_entropy_mean_cpu.ndim == 0
    assert teacher_valid_token_count_cpu.ndim == 0

    dense_student = student_base.detach().clone().requires_grad_(True)
    dense_telemetry: dict[str, torch.Tensor] = {}
    dense_loss = compute_dense_loss_from_logits(
        loss_type="eakld",
        student_logits=dense_student,
        teacher_logits=teacher_logits,
        mask=mask,
        temperature=1.0,
        eakld_confidence_k=16,
        telemetry_out=dense_telemetry,
    )
    dense_loss.backward()

    chunk_size = 2
    seq_len = int(student_base.shape[1])
    expected_chunks = (seq_len + chunk_size - 1) // chunk_size
    transfer_calls: list[tuple[int, int, tuple[int, ...]]] = []
    original = distill_losses.copy_teacher_logit_chunk_to_device

    def wrapped(teacher_logits_cpu_arg, *, start, end, target_device):
        out = original(
            teacher_logits_cpu_arg,
            start=start,
            end=end,
            target_device=target_device,
        )
        transfer_calls.append((int(start), int(end), tuple(out.shape)))
        return out

    offload_student = student_base.detach().clone().requires_grad_(True)
    offload_telemetry: dict[str, torch.Tensor] = {}
    with mock.patch.object(
        distill_losses,
        "copy_teacher_logit_chunk_to_device",
        side_effect=wrapped,
    ):
        offload_loss = compute_dense_loss_from_offloaded_teacher(
            loss_type="eakld",
            student_logits=offload_student,
            teacher_logits_cpu=teacher_logits_cpu,
            teacher_gamma_cpu=teacher_gamma_cpu,
            teacher_entropy_mean_cpu=teacher_entropy_mean_cpu,
            teacher_valid_token_count_cpu=teacher_valid_token_count_cpu,
            mask=mask,
            temperature=1.0,
            eakld_confidence_k=16,
            sequence_chunk_size=chunk_size,
            telemetry_out=offload_telemetry,
        )
        assert len(transfer_calls) == expected_chunks
        offload_loss.backward()
        # checkpoint recomputation transfers each chunk again on backward
        assert len(transfer_calls) == expected_chunks * 2

    full_shape = tuple(teacher_logits_cpu.shape)
    for start, end, shape in transfer_calls:
        assert end - start <= chunk_size
        assert shape[1] <= chunk_size
        assert shape != full_shape

    assert offload_loss.ndim == 0
    assert torch.isfinite(offload_loss)
    assert torch.allclose(offload_loss, dense_loss, rtol=5e-6, atol=5e-6)
    assert set(offload_telemetry) == EAKLD_TELEMETRY_KEYS
    for key in EAKLD_TELEMETRY_KEYS:
        assert torch.allclose(
            offload_telemetry[key],
            dense_telemetry[key],
            rtol=5e-6,
            atol=5e-6,
        )
    assert dense_student.grad is not None
    assert offload_student.grad is not None
    assert torch.isfinite(offload_student.grad).all()
    assert torch.allclose(
        offload_student.grad,
        dense_student.grad,
        rtol=5e-6,
        atol=5e-6,
    )
    assert teacher_logits.grad is None


def test_block_attention_helper_smoke() -> None:
    torch.manual_seed(33)
    # [B, H, Q, K] with causal valid key mask
    batch, heads, seq_len = 2, 2, 6
    teacher = torch.randn(batch, heads, seq_len, seq_len, dtype=torch.float32)
    student = teacher + torch.randn_like(teacher) * 0.5
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    valid = causal.view(1, 1, seq_len, seq_len).expand(batch, heads, seq_len, seq_len)

    forward, reverse, entropy, valid_query = _attention_map_kl_chunk_losses(
        teacher_logits=teacher,
        student_logits=student,
        valid_key_mask=valid,
    )
    assert torch.isfinite(forward).all()
    assert torch.isfinite(reverse).all()
    assert torch.isfinite(entropy).all()
    assert valid_query.dtype == torch.bool
    assert valid_query.all()

    match_forward, match_reverse, _entropy, match_valid = (
        _attention_map_kl_chunk_losses(
            teacher_logits=teacher,
            student_logits=teacher.clone(),
            valid_key_mask=valid,
        )
    )
    assert match_valid.all()
    assert torch.allclose(match_forward, torch.zeros_like(match_forward), atol=1e-7)
    assert torch.allclose(match_reverse, torch.zeros_like(match_reverse), atol=1e-7)

    teacher_extreme = teacher.clone()
    student_extreme = student.clone()
    future = ~causal
    teacher_extreme[..., future] = 1.0e4
    student_extreme[..., future] = -1.0e4
    extreme_forward, extreme_reverse, extreme_entropy, extreme_valid = (
        _attention_map_kl_chunk_losses(
            teacher_logits=teacher_extreme,
            student_logits=student_extreme,
            valid_key_mask=valid,
        )
    )
    assert extreme_valid.equal(valid_query)
    assert torch.allclose(extreme_forward, forward, rtol=1e-6, atol=1e-7)
    assert torch.allclose(extreme_reverse, reverse, rtol=1e-6, atol=1e-7)
    assert torch.allclose(extreme_entropy, entropy, rtol=1e-6, atol=1e-7)
