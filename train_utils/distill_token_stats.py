from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class DistillWindowTokenStats:
    avg_prompt_tokens_per_sample: float
    avg_response_tokens_per_sample: float
    global_samples: int


class DistillTokenStatsAccumulator:
    def __init__(self) -> None:
        self._accumulator: Optional[torch.Tensor] = None

    def update(
        self,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            raise ValueError("labels must be a rank-2 tensor")

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
                raise ValueError("attention_mask must be a rank-2 tensor")
            if tuple(labels.shape) != tuple(attention_mask.shape):
                raise ValueError(
                    "attention_mask shape mismatch: "
                    f"expected {tuple(labels.shape)}, got {tuple(attention_mask.shape)}"
                )
            if labels.device != attention_mask.device:
                raise ValueError(
                    "attention_mask device mismatch: "
                    f"expected {labels.device}, got {attention_mask.device}"
                )

        device = labels.device
        if attention_mask is None:
            valid = torch.ones_like(labels, dtype=torch.bool)
        else:
            valid = attention_mask.ne(0)

        prompt_count = (valid & labels.eq(-100)).sum(dtype=torch.float32)
        response_count = (valid & labels.ne(-100)).sum(dtype=torch.float32)
        sample_count = torch.tensor(
            float(labels.shape[0]),
            dtype=torch.float32,
            device=device,
        )
        batch_stats = torch.stack(
            [prompt_count, response_count, sample_count]
        ).detach()

        if self._accumulator is None:
            self._accumulator = batch_stats
            return

        if self._accumulator.device != device:
            self._accumulator = self._accumulator.to(device)
        self._accumulator = self._accumulator + batch_stats

    def consume_global(self, accelerator: Any) -> Optional[DistillWindowTokenStats]:
        device = accelerator.device

        if self._accumulator is None:
            local = torch.zeros(3, dtype=torch.float32, device=device)
        else:
            local = self._accumulator.to(device=device, dtype=torch.float32)

        global_stats = accelerator.reduce(local, reduction="sum")
        self._accumulator = None

        prompt_total = float(global_stats[0].item())
        response_total = float(global_stats[1].item())
        global_samples = int(global_stats[2].item())
        if global_samples == 0:
            return None

        return DistillWindowTokenStats(
            avg_prompt_tokens_per_sample=prompt_total / global_samples,
            avg_response_tokens_per_sample=response_total / global_samples,
            global_samples=global_samples,
        )
