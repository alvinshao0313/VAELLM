"""Per-bit observations and score-only interventions for isolated native trials."""
from __future__ import annotations

import math
import random

import torch


CHUNK = 1 << 20
FLIPBACK_WINDOWS = (1, 2, 4, 8, 16, 32)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class BankOscillation:
    """Measure natural optimizer travel in u=p/s, excluding interventions."""

    def __init__(self, score, scale, *, sample_count=128):
        _require(score.ndim == 1 and score.dtype == torch.float32, "Expected one FP32 score bank.")
        _require(math.isfinite(scale) and scale > 0, "A fixed positive bank scale is required.")
        _require(bool(torch.isfinite(score).all()) and bool((score.abs() <= scale / 2).all()),
                 "Initial scores must be finite and within the bank radius.")
        self.scale = float(scale)
        self.previous = score.detach().clone()
        self.travel = torch.zeros_like(score)
        self.flips = torch.zeros_like(score, dtype=torch.int16)
        self.last_flip = torch.full_like(score, -1, dtype=torch.int32)
        self.sample_indices = torch.arange(min(sample_count, score.numel()), device=score.device)
        self.step = 0
        self.window_start = 0

    @torch.no_grad()
    def observe(self, score, *, step):
        _require(step == self.step + 1, "Observe every optimizer update exactly once.")
        _require(step - self.window_start < 32767, "The int16 window counter requires fewer than32767 updates.")
        _require(score.shape == self.previous.shape and score.device == self.previous.device,
                 "The logical score bank must remain fixed.")
        counts = torch.zeros(2 + len(FLIPBACK_WINDOWS), dtype=torch.int64, device=score.device)
        travel_sum = torch.zeros((), dtype=torch.float64, device=score.device)
        for start in range(0, score.numel(), CHUNK):
            stop = min(start + CHUNK, score.numel())
            current, old = score[start:stop], self.previous[start:stop]
            flipped = (current >= 0) != (old >= 0)
            prior = self.flips[start:stop]
            last = self.last_flip[start:stop]
            counts[0] += flipped.sum()
            counts[1] += (flipped & ((prior % 2) == 1)).sum()
            interval = step - last
            for i, bound in enumerate(FLIPBACK_WINDOWS):
                counts[i + 2] += (flipped & (last >= 0) & (interval <= bound)).sum()
            distance = (current - old).abs() / self.scale
            self.travel[start:stop].add_(distance)
            travel_sum += distance.sum(dtype=torch.float64)
            prior.add_(flipped.to(torch.int16))
            last.copy_(torch.where(flipped, step, last))
            old.copy_(current)
        self.step = step
        values = counts.cpu().tolist()
        return {"step": step, "step_flip_events": values[0], "returns_to_window_start": values[1],
                "flipbacks_within_updates": dict(zip(map(str, FLIPBACK_WINDOWS), values[2:])),
                "natural_u_travel": float(travel_sum), "sample": self.sample(score)}

    @torch.no_grad()
    def sample(self, score):
        ids = self.sample_indices
        return {"active_ordinals": ids.cpu().tolist(), "u": (score[ids] / self.scale).cpu().tolist(),
                "window_flips": self.flips[ids].cpu().tolist(),
                "window_u_travel": self.travel[ids].cpu().tolist(),
                "last_flip_step": self.last_flip[ids].cpu().tolist()}

    @torch.no_grad()
    def risk_mask(self, threshold):
        _require(math.isfinite(threshold) and threshold > 0, "A fixed positive risk threshold is required.")
        # Two crossings certify a return in this window; one useful crossing alone
        # cannot become a risk candidate, even when its travel ratio is large.
        return (self.flips >= 2) & (self.flips.float() >= threshold * self.travel)

    @torch.no_grad()
    def reanchor(self, score, mask):
        _require(mask.shape == score.shape and mask.dtype == torch.bool, "Expected a per-bit intervention mask.")
        count = changed = 0
        travel = 0.0
        for start in range(0, score.numel(), CHUNK):
            stop = min(start + CHUNK, score.numel())
            selected = mask[start:stop]
            current = score[start:stop]
            before = current[selected]
            hard = before >= 0
            after = torch.where(hard, self.scale / 2, -self.scale / 2)
            current[selected] = after
            count += int(selected.sum())
            changed += int((before != after).sum())
            travel += float(((after - before).abs() / self.scale).sum(dtype=torch.float64))
            _require(torch.equal(current[selected] >= 0, hard), "Score reanchoring must preserve every hard bit.")
            # Only the observation snapshot follows the intervention. Its movement
            # is deliberately absent from the natural-travel denominator.
            self.previous[start:stop].copy_(current)
        return {"selected_bits": count, "score_values_changed": changed,
                "intervention_u_travel": travel, "hard_bits_unchanged": True,
                "Adam_moments_touched": False}

    @torch.no_grad()
    def finish_window(self, score):
        result = {"start_step": self.window_start, "end_step": self.step,
                  "flip_events": int(self.flips.sum(dtype=torch.int64)),
                  "net_bits": int((self.flips % 2).sum(dtype=torch.int64)),
                  "return_pairs": int((self.flips // 2).sum(dtype=torch.int64)),
                  "natural_u_travel": float(self.travel.sum(dtype=torch.float64))}
        self.previous.copy_(score)
        self.travel.zero_()
        self.flips.zero_()
        self.window_start = self.step
        return result

    def state_dict(self):
        return {"scale": self.scale, "step": self.step, "window_start": self.window_start,
                **{name: getattr(self, name).detach().to(device="cpu", copy=True)
                   for name in ("previous", "travel", "flips", "last_flip", "sample_indices")}}

    @torch.no_grad()
    def load_state_dict(self, state):
        _require(state["scale"] == self.scale, "Observation scale changed across the shared anchor.")
        for name in ("previous", "travel", "flips", "last_flip", "sample_indices"):
            target, source = getattr(self, name), state[name]
            _require(source.shape == target.shape and source.dtype == target.dtype,
                     "Observation anchor shape/dtype mismatch: " + name)
            target.copy_(source)
        self.step, self.window_start = int(state["step"]), int(state["window_start"])


@torch.no_grad()
def matched_masks(score, scale, risk, *, distance_bins, seed):
    """Keep only risk/control pairs available in the same predefined distance bin."""
    _require(distance_bins >= 1 and risk.dtype == torch.bool and risk.shape == score.shape,
             "Expected fixed distance bins and a bank-local risk mask.")
    bins = ((score.abs() / (scale / 2)) * distance_bins).long().clamp_(max=distance_bins - 1)
    selected_risk, control = torch.zeros_like(risk), torch.zeros_like(risk)
    generator = random.Random(seed)
    matches = []
    for bucket in range(distance_bins):
        in_bucket = bins == bucket
        risk_ids = torch.nonzero(risk & in_bucket, as_tuple=False).flatten()
        requested = risk_ids.numel()
        if not requested:
            continue
        candidates = torch.nonzero(~risk & in_bucket, as_tuple=False).flatten()
        size = candidates.numel()
        count = min(requested, size)
        # An affine permutation samples distinct candidate positions without a
        # bank-sized randperm or changing the production bit sampler.
        for pool, mask in ((risk_ids, selected_risk), (candidates, control)):
            if not count:
                continue
            offset = generator.randrange(pool.numel())
            stride = generator.randrange(1, pool.numel() + 1)
            while math.gcd(stride, pool.numel()) != 1:
                stride = generator.randrange(1, pool.numel() + 1)
            positions = (torch.arange(count, device=score.device) * stride + offset) % pool.numel()
            mask[pool[positions]] = True
        matches.append({"distance_bin": bucket, "risk_candidates": requested,
                        "selected_pairs": count, "available_nonrisk": size})
    _require(not bool((control & risk).any()) and int(control.sum()) == int(selected_risk.sum()),
             "Matched controls must exclude risk bits and preserve exact bank counts.")
    def distances(mask):
        values = score[mask].abs() / scale
        return {"count": values.numel(), "minimum": float(values.min()) if values.numel() else None,
                "maximum": float(values.max()) if values.numel() else None,
                "mean": float(values.mean(dtype=torch.float64)) if values.numel() else None,
                "second_moment": float(values.double().square().mean()) if values.numel() else None}
    eligible, selected = int(risk.sum()), int(selected_risk.sum())
    return selected_risk, control, {"distance_bins": distance_bins, "max_abs_u_bin_width": .5 / distance_bins,
                    "seed": seed, "raw_risk_count": eligible, "matched_pair_count": selected,
                    "matched_coverage": selected / eligible if eligible else None, "counts": matches,
                    "risk_abs_u_distribution": distances(selected_risk),
                    "control_abs_u_distribution": distances(control)}
