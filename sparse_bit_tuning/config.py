from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union


_VALID_BIT_OPTIMIZERS = {"rms_sgd", "adam", "adamw"}
_AUTO_BIT_LR = {
    "rms_sgd": 0.05,
    "adam": 0.02,
    "adamw": 0.02,
}


def normalize_bit_optimizer(value: str) -> str:
    name = str(value or "").strip().lower()
    if name not in _VALID_BIT_OPTIMIZERS:
        raise ValueError(
            f"bit_optimizer must be one of {sorted(_VALID_BIT_OPTIMIZERS)}, got {value!r}."
        )
    return name


def resolve_bit_lr(value: Union[str, float, int], *, optimizer: str) -> float:
    opt = normalize_bit_optimizer(optimizer)
    text = str(value).strip().lower()
    if text == "auto":
        return float(_AUTO_BIT_LR[opt])
    try:
        lr = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bit_lr must be 'auto' or a positive float, got {value!r}.") from exc
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"bit_lr must be finite and > 0, got {lr!r}.")
    return lr


def normalize_round_steps(value: Union[str, int]) -> Union[str, int]:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bit_round_steps must be 'auto' or an integer >= 1, got {value!r}.") from exc
    if steps < 1:
        raise ValueError(f"bit_round_steps must be >= 1, got {steps}.")
    return steps


def resolve_round_steps(value: Union[str, int], *, total_optimizer_steps: int, active_ratio: float) -> int:
    normalized = normalize_round_steps(value)
    if normalized != "auto":
        return int(normalized)
    total = int(total_optimizer_steps)
    if total < 1:
        raise ValueError(f"total_optimizer_steps must be >= 1, got {total}.")
    ratio = float(active_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"active_ratio must satisfy 0 < ratio <= 1, got {ratio}.")
    coverage_rounds = int(math.ceil(1.0 / ratio))
    return max(1, int(math.floor(total / coverage_rounds)))


def resolve_stable_steps(bit_round_steps: int) -> int:
    steps = int(bit_round_steps)
    if steps < 1:
        raise ValueError(f"bit_round_steps must be >= 1, got {steps}.")
    return max(3, min(10, int(math.ceil(0.2 * steps))))


def active_count(n_bits: int, ratio: float) -> int:
    n = int(n_bits)
    r = float(ratio)
    if n < 1:
        raise ValueError(f"n_bits must be >= 1, got {n}.")
    if not (0.0 < r <= 1.0):
        raise ValueError(f"active_ratio must satisfy 0 < ratio <= 1, got {r}.")
    return min(n, max(1, int(math.ceil(n * r))))


@dataclass(frozen=True)
class SparseBitTuningConfig:
    enabled: bool = False
    active_ratio: float = 0.01
    optimizer: str = "rms_sgd"
    bit_lr: Union[str, float] = "auto"
    weight_decay: float = 0.0
    round_steps: Union[str, int] = "auto"

    def normalized(self) -> "SparseBitTuningConfig":
        opt = normalize_bit_optimizer(self.optimizer)
        ratio = float(self.active_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"bit_active_ratio must satisfy 0 < ratio <= 1, got {ratio}.")
        wd = float(self.weight_decay)
        if not math.isfinite(wd) or wd < 0.0:
            raise ValueError(f"bit_weight_decay must be finite and >= 0, got {wd}.")
        if opt != "adamw" and wd != 0.0:
            raise ValueError("bit_weight_decay is only supported by bit_optimizer=adamw.")
        normalize_round_steps(self.round_steps)
        resolve_bit_lr(self.bit_lr, optimizer=opt)
        return SparseBitTuningConfig(
            enabled=bool(self.enabled),
            active_ratio=ratio,
            optimizer=opt,
            bit_lr=self.bit_lr,
            weight_decay=wd,
            round_steps=self.round_steps,
        )

    def resolved_lr(self) -> float:
        return resolve_bit_lr(self.bit_lr, optimizer=self.optimizer)
