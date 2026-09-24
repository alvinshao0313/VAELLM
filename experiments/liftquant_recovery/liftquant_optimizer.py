"""Representation-independent optimizer policy extracted from LiftQuant Stage2.

Source: Heliulu/LiftQuant@72b3875c770e4579639931fed89dc95e4067edac,
quantize/liftq.py Stage2. Preserve the upstream step()+get_lr() semantics.
Parameter groups and their initial learning rates are supplied by the VAE adapter;
this module makes no claim that those representation-specific rates are equivalent.
"""
import torch


def build_optimizer(groups, steps):
    if steps <= 0:
        raise ValueError("Expected a positive optimizer-step budget.")
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    empty_optimizers = [
        torch.optim.AdamW([torch.tensor(0)], lr=group["lr"])
        for group in optimizer.param_groups
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            empty, T_max=steps, eta_min=group["lr"] / 20,
        )
        for empty, group in zip(empty_optimizers, optimizer.param_groups)
    ]
    return optimizer, schedulers


def advance_learning_rates(optimizer, schedulers):
    if len(optimizer.param_groups) != len(schedulers):
        raise ValueError("Each parameter group must have its own scheduler.")
    for group, scheduler in zip(optimizer.param_groups, schedulers):
        scheduler.step()
        # Deliberately get_lr(), as called by the pinned reference, not get_last_lr().
        group["lr"] = scheduler.get_lr()[0]


def verify_reference_sequence():
    """Regression for the previously observed non-equivalent cosine replacement."""
    import warnings
    expected = [0.0002, 0.00014842514421272202,
                0.00006564971157455597, 0.000018149711574555977]
    groups = [dict(params=[torch.nn.Parameter(torch.zeros(()))], lr=2e-4),
              dict(params=[torch.nn.Parameter(torch.zeros(()))], lr=2e-5)]
    optimizer, schedulers = build_optimizer(groups, 4)
    used = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for value in expected:
            pair = [group["lr"] for group in optimizer.param_groups]
            assert abs(pair[0] - value) < 1e-16
            assert abs(pair[1] - value / 10) < 1e-16
            used.append(pair)
            optimizer.step()
            advance_learning_rates(optimizer, schedulers)
    assert optimizer.defaults["weight_decay"] == 0.0
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    return dict(status="PASS", source="pinned upstream 4-step sequence", used_lrs=used)


if __name__ == "__main__":
    import json
    print(json.dumps(verify_reference_sequence(), indent=2))
