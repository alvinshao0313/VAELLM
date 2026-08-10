from __future__ import annotations

import pytest
import torch

from train_utils.distill_token_stats import (
    DistillTokenStatsAccumulator,
    DistillWindowTokenStats,
)


class FakeAccelerator:
    def __init__(
        self,
        device: torch.device,
        *,
        reduced: torch.Tensor | None = None,
    ) -> None:
        self.device = device
        self._reduced = reduced
        self.reduce_calls: list[tuple[torch.Tensor, str]] = []

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        self.reduce_calls.append((tensor.clone(), reduction))
        if self._reduced is not None:
            return self._reduced.to(device=tensor.device, dtype=tensor.dtype)
        return tensor.clone()


def _consume(
    accumulator: DistillTokenStatsAccumulator,
    device: torch.device | None = None,
    *,
    reduced: torch.Tensor | None = None,
) -> DistillWindowTokenStats | None:
    dev = device or torch.device("cpu")
    return accumulator.consume_global(FakeAccelerator(dev, reduced=reduced))


def test_single_sample_prompt_and_response_without_causal_shift() -> None:
    labels = torch.tensor([[-100, -100, -100, 1, 2, 3]])
    attention = torch.ones(1, 6, dtype=torch.long)

    accumulator = DistillTokenStatsAccumulator()
    accumulator.update(labels, attention)
    stats = _consume(accumulator)

    assert stats is not None
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(3.0)
    assert stats.avg_response_tokens_per_sample == pytest.approx(3.0)
    assert stats.global_samples == 1


def test_padding_with_label_minus100_and_zero_attention_is_not_prompt() -> None:
    labels = torch.tensor([[-100, -100, 5, 6]])
    attention = torch.tensor([[1, 0, 1, 1]])

    accumulator = DistillTokenStatsAccumulator()
    accumulator.update(labels, attention)
    stats = _consume(accumulator)

    assert stats is not None
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(1.0)
    assert stats.avg_response_tokens_per_sample == pytest.approx(2.0)
    assert stats.global_samples == 1


def test_asymmetric_micro_batches_use_global_weighted_average() -> None:
    labels_batch_a = torch.tensor([[-100] * 8 + [1, 2]])
    labels_batch_b = torch.tensor(
        [
            [-100, -100, -100, 9],
            [-100, -100, -100, 10],
            [-100, -100, -100, 11],
        ]
    )
    attention_a = torch.ones(1, 10, dtype=torch.long)
    attention_b = torch.ones(3, 4, dtype=torch.long)

    accumulator = DistillTokenStatsAccumulator()
    accumulator.update(labels_batch_a, attention_a)
    accumulator.update(labels_batch_b, attention_b)
    stats = _consume(accumulator)

    assert stats is not None
    assert stats.global_samples == 4
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(4.25)
    assert stats.avg_response_tokens_per_sample == pytest.approx(1.25)

    wrong_prompt_avg = (8.0 + 3.0) / 2.0
    wrong_response_avg = (2.0 + 1.0) / 2.0
    assert stats.avg_prompt_tokens_per_sample != pytest.approx(wrong_prompt_avg)
    assert stats.avg_response_tokens_per_sample != pytest.approx(wrong_response_avg)


def test_ten_optimizer_steps_accumulate_before_single_consume() -> None:
    labels = torch.tensor([[-100, -100, -100, 1, 2, 3]])
    attention = torch.ones(1, 6, dtype=torch.long)
    accumulator = DistillTokenStatsAccumulator()

    for _ in range(10):
        accumulator.update(labels, attention)

    stats = _consume(accumulator)

    assert stats is not None
    assert stats.global_samples == 10
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(3.0)
    assert stats.avg_response_tokens_per_sample == pytest.approx(3.0)


def test_consume_global_uses_distributed_totals_not_rank_local() -> None:
    labels = torch.tensor([[-100, -100, -100, 1, 2, 3]])
    attention = torch.ones(1, 6, dtype=torch.long)
    reduced = torch.tensor([15.0, 15.0, 3.0], dtype=torch.float32)

    accumulator = DistillTokenStatsAccumulator()
    accumulator.update(labels, attention)
    fake = FakeAccelerator(torch.device("cpu"), reduced=reduced)
    stats = accumulator.consume_global(fake)

    assert stats is not None
    assert stats.global_samples == 3
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(5.0)
    assert stats.avg_response_tokens_per_sample == pytest.approx(5.0)
    assert len(fake.reduce_calls) == 1
    local_tensor = fake.reduce_calls[0][0]
    assert torch.allclose(local_tensor, torch.tensor([3.0, 3.0, 1.0]))


def test_consume_global_with_no_local_updates_still_reduces_global_totals() -> None:
    reduced = torch.tensor([30.0, 30.0, 10.0], dtype=torch.float32)
    accumulator = DistillTokenStatsAccumulator()
    fake = FakeAccelerator(torch.device("cpu"), reduced=reduced)

    stats = accumulator.consume_global(fake)

    assert stats is not None
    assert stats.global_samples == 10
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(3.0)
    assert stats.avg_response_tokens_per_sample == pytest.approx(3.0)
    assert len(fake.reduce_calls) == 1
    assert torch.allclose(fake.reduce_calls[0][0], torch.zeros(3))


def test_second_consume_without_updates_returns_none() -> None:
    labels = torch.tensor([[-100, -100, -100, 1, 2, 3]])
    attention = torch.ones(1, 6, dtype=torch.long)
    accumulator = DistillTokenStatsAccumulator()
    accumulator.update(labels, attention)

    first = _consume(accumulator)
    second = _consume(accumulator)

    assert first is not None
    assert second is None


def test_update_rejects_invalid_labels_rank() -> None:
    accumulator = DistillTokenStatsAccumulator()

    with pytest.raises(ValueError, match="labels must be a rank-2 tensor"):
        accumulator.update(torch.tensor([1, 2, 3]))

    with pytest.raises(ValueError, match="labels must be a rank-2 tensor"):
        accumulator.update(torch.ones(2, 3, 4))


def test_update_rejects_attention_shape_or_device_mismatch() -> None:
    accumulator = DistillTokenStatsAccumulator()
    labels = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="attention_mask shape mismatch"):
        accumulator.update(labels, torch.ones(1, 4))

    with pytest.raises(ValueError, match="attention_mask shape mismatch"):
        accumulator.update(labels, torch.ones(2, 3))

    if torch.cuda.is_available():
        labels_cuda = labels.to("cuda")
        attention_cpu = torch.ones(1, 3)
        with pytest.raises(ValueError, match="attention_mask device mismatch"):
            accumulator.update(labels_cuda, attention_cpu)


def test_update_with_none_attention_treats_all_positions_as_valid() -> None:
    labels = torch.tensor([[-100, -100, 1, 2]])
    accumulator = DistillTokenStatsAccumulator()
    accumulator.update(labels, None)
    stats = _consume(accumulator)

    assert stats is not None
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(2.0)
    assert stats.avg_response_tokens_per_sample == pytest.approx(2.0)
    assert stats.global_samples == 1
