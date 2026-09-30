"""Finite, CPU-only decisions for the authorized rank-8 DP search.

Scores are percentage points. Promotion thresholds allocate training budget;
they are not tests of statistical significance. Each new configuration starts
from the same input model, and completed candidates share a 5000-step schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


EARLY_STAGE_STEPS = 2500
FINAL_STEPS = 5000
PILOT_ADDITIONAL_STEPS = 3 * (FINAL_STEPS - 1000)
MAX_EXTRA_PROMOTIONS = 3
NEAR_TIE_PP = 0.25
RISING_GAIN_PP = 0.25
RISING_MAX_GAP_PP = 0.75


@dataclass(frozen=True)
class Coordinate:
    name: str
    key: str
    values: tuple[float | int, ...]


HEAD = Coordinate("head_lr", "lm_head_lr", (3e-5, 3e-4))
NORM = Coordinate("norm_lr", "norm_lr", (3e-5, 3e-4))
DROPOUT = Coordinate("dropout", "lora_dropout", (0.0, 0.05))
LORA_ALPHA = Coordinate("lora_alpha", "lora_alpha", (8, 32))
WEIGHT_DECAY = Coordinate("weight_decay", "weight_decay", (0.0, 0.01))
WARMUP = Coordinate("warmup", "warmup_steps", (50, 300))
HIDDEN = Coordinate("hidden_weight", "hidden_loss_weight", (0.0, 0.03))
PRE_MLP = Coordinate("pre_mlp_weight", "pre_mlp_hidden_loss_weight", (0.0, 0.03))
KD_ALPHA = Coordinate("kd_alpha", "alpha", (0.99, 1.0))


def coordinates(scope: str, days: int) -> list[Coordinate]:
    """Return one finite pass; loss weights require an explicit scope choice.

    ``days`` chooses a plan, not a promise about elapsed execution time. A
    three-day plan prioritizes fewer coordinates; the controller owns the
    wall-clock limit and reserves time for final model export and evaluation.
    """
    if scope not in ("optimization", "loss_weights"):
        raise ValueError("scope must be optimization or loss_weights")
    if days not in (3, 7, 14):
        raise ValueError("days must be 3, 7, or 14")
    if days == 3:
        return [HEAD, NORM, HIDDEN if scope == "loss_weights" else DROPOUT]
    if scope == "optimization":
        return [HEAD, NORM, DROPOUT, LORA_ALPHA, WEIGHT_DECAY, WARMUP]
    result = [HEAD, NORM, DROPOUT, HIDDEN, PRE_MLP, KD_ALPHA]
    if days == 14:
        result.extend((LORA_ALPHA, WEIGHT_DECAY, WARMUP))
    return result


def estimate_base_additional_steps(scope: str, days: int) -> int:
    """Exclude the running pilot prefixes, extra promotions, and export cost."""
    per_coordinate = 2 * EARLY_STAGE_STEPS + FINAL_STEPS - EARLY_STAGE_STEPS
    return PILOT_ADDITIONAL_STEPS + per_coordinate * len(coordinates(scope, days))


def _prefix(evaluations: list[dict], through_step: int) -> list[dict]:
    if not isinstance(through_step, int) or isinstance(through_step, bool) or through_step <= 0:
        raise ValueError("through_step must be a positive integer")
    selected = []
    seen = set()
    for row in evaluations:
        step = row["step"]
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError(f"Invalid evaluation step: {step!r}")
        if step > through_step:
            continue
        if step in seen:
            raise ValueError(f"Duplicate evaluation step: {step}")
        score = float(row["mean_percent"])
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(f"Invalid mean_percent at step {step}: {score}")
        seen.add(step)
        selected.append(row)
    if through_step not in seen:
        raise ValueError(f"Missing completed evaluation at common step {through_step}")
    return selected


def best_evaluation(evaluations: list[dict], *, through_step: int) -> dict:
    """Rank recoverable rows after verifying the common completed prefix.

    The controller maintains checkpoint availability. Missing availability means
    an abstract policy input; an explicit False excludes a rotated snapshot.
    The common endpoint is still required even when its weights were retired.
    """
    rows = [row for row in _prefix(evaluations, through_step) if row.get("available") is not False]
    if not rows:
        raise ValueError(f"No recoverable evaluated checkpoint through step {through_step}")
    return min(rows, key=lambda row: (-float(row["mean_percent"]), row["step"]))


def improvement_over_last_steps(
    evaluations: list[dict], *, through_step: int, span: int = 1000,
) -> float:
    """Use the exact endpoints, not the difference between prefix maxima."""
    rows = {row["step"]: row for row in _prefix(evaluations, through_step)}
    if not isinstance(span, int) or isinstance(span, bool) or span <= 0:
        raise ValueError("span must be a positive integer")
    start = through_step - span
    if start not in rows:
        raise ValueError(f"Missing trend evaluation at step {start}")
    return float(rows[through_step]["mean_percent"]) - float(rows[start]["mean_percent"])


def _ranked(candidates: list[dict], through_step: int) -> list[tuple[str, float, list[dict]]]:
    if not candidates:
        raise ValueError("At least one candidate is required")
    seen = set()
    ranked = []
    for candidate in candidates:
        name = candidate["name"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"Candidate names must be nonempty and unique: {name!r}")
        seen.add(name)
        rows = candidate["evaluations"]
        score = float(best_evaluation(rows, through_step=through_step)["mean_percent"])
        ranked.append((name, score, rows))
    # Python's stable sort preserves caller order for exact score ties.
    return sorted(ranked, key=lambda candidate: -candidate[1])


def coordinate_winner(candidates: list[dict], *, through_step: int = FINAL_STEPS) -> str:
    """Rank a common completed budget; put the incumbent first to retain ties."""
    return _ranked(candidates, through_step)[0][0]


def select_promotions(
    candidates: list[dict], *, through_step: int = EARLY_STAGE_STEPS, extra_slots: int = 0,
) -> list[str]:
    """Promote the best challenger plus qualifying, explicitly budgeted extras.

    ``candidates`` contains challengers only. ``extra_slots`` is the controller's
    remaining search-wide allowance, at most three; this pure function does not
    track allowance across calls. A weaker challenger can qualify by a rising
    final 1000-step trajectory, even when its current prefix maximum is lower.
    """
    if (
        not isinstance(extra_slots, int)
        or isinstance(extra_slots, bool)
        or not 0 <= extra_slots <= MAX_EXTRA_PROMOTIONS
    ):
        raise ValueError(f"extra_slots must be an integer in [0, {MAX_EXTRA_PROMOTIONS}]")
    ranked = _ranked(candidates, through_step)
    promoted = [ranked[0][0]]
    best_score = ranked[0][1]
    for name, score, rows in ranked[1:]:
        if len(promoted) - 1 >= extra_slots:
            break
        gap = best_score - score
        near = gap <= NEAR_TIE_PP + 1e-12
        rising = False
        if not near and gap <= RISING_MAX_GAP_PP + 1e-12:
            gain = improvement_over_last_steps(rows, through_step=through_step)
            rising = gain >= RISING_GAIN_PP - 1e-12
        if near or rising:
            promoted.append(name)
    return promoted
