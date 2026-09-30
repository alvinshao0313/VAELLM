"""Training-only coordinates calibrated to the existing decoder's bit sensitivity.

p = s * (b - 1/2), b(p) = clamp(round(p/s + 1/2), 0, 1).
The fixed s is measured before training; it is NOT a new model parameter.
"""
import math

import torch

from litebsq.bit_sensitivity import measure_scale


def hard_bits(proxy, scale):
    # Keep the exact round-to-even tie rule and saturation of the specified STE.
    return (proxy / scale + 0.5).round().clamp_(0, 1).bool()


def ste_gate(proxy, scale):
    rounded = (proxy / scale + 0.5).round()
    return (rounded >= 0) & (rounded <= 1)


def dense_ste(proxy, scale):
    normalized = proxy / scale + 0.5
    rounded = normalized + (normalized.round() - normalized).detach()
    return rounded.clamp(0, 1)


def motion_budget(initial_lr, steps, margin):
    if min(initial_lr, steps, margin) <= 0:
        raise ValueError('Expected a positive lr, step budget and margin.')
    b1, b2 = .9, .999
    ratio_bound = math.sqrt((1-b1)**2 / ((1-b2)*(1-b1*b1/b2)))
    # Standard cosine upper-bounds the pinned upstream step()+get_lr() sequence.
    lr_sum_upper = initial_lr * (.525*steps + .475)
    bound = lr_sum_upper * ratio_bound
    return dict(steps=steps, initial_lr=initial_lr, margin=margin,
                adam_total_motion_upper_bound=bound,
                upper_bound_to_margin=bound/margin,
                structurally_unreachable=bound < margin,
                constant_gradient_motion_to_margin_upper=lr_sum_upper/margin,
                meaning='A bound above the margin permits crossing; it does not guarantee a flip or improvement.')
