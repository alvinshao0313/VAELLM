from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LMHeadWithPostNormLinear(nn.Module):
    """Apply a trainable post-norm linear before the original lm_head."""

    def __init__(self, lm_head: nn.Module):
        if not isinstance(lm_head, nn.Linear):
            raise TypeError(f"LMHeadWithPostNormLinear expects nn.Linear lm_head, got {type(lm_head)}")
        super().__init__()
        hidden_size = int(lm_head.in_features)
        if int(lm_head.out_features) <= 0:
            raise ValueError(f"Invalid lm_head out_features={lm_head.out_features}")

        self.post_norm_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        # Identity init keeps logits unchanged at step 0.
        with torch.no_grad():
            self.post_norm_linear.weight.copy_(torch.eye(hidden_size, dtype=self.post_norm_linear.weight.dtype))

        self.lm_head = lm_head

    @property
    def weight(self):
        return self.lm_head.weight

    @property
    def bias(self):
        return self.lm_head.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.post_norm_linear(hidden_states)
        return self.lm_head(hidden_states)


def has_post_norm_head_linear(model: nn.Module) -> bool:
    return isinstance(getattr(model, "lm_head", None), LMHeadWithPostNormLinear)


def ensure_post_norm_head_linear(model: nn.Module) -> bool:
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, LMHeadWithPostNormLinear):
        return False
    if not isinstance(lm_head, nn.Linear):
        raise TypeError(f"Model lm_head must be nn.Linear to attach post-norm linear, got {type(lm_head)}")

    wrapped = LMHeadWithPostNormLinear(lm_head)
    wrapped.train(lm_head.training)
    wrapped.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
    model.lm_head = wrapped
    return True


def resolve_post_norm_linear(model: nn.Module) -> Optional[nn.Linear]:
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, LMHeadWithPostNormLinear):
        return None
    return lm_head.post_norm_linear
