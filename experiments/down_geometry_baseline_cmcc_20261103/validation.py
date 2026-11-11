"""Observe only the two genuine new-shape optimizer updates per stage."""
from contextlib import contextmanager

from . import bindings as b


@contextmanager
def updates(native, *, enabled):
    rows,original = [],native.create_optimizer

    def create(*args,**kwargs):
        optimizer = original(*args,**kwargs)
        row = dict(stage=len(rows),updates=[])
        rows.append(row)
        step = optimizer.step

        def observe(_optimizer,*step_args,**step_kwargs):
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            gradients = [p.grad for p in parameters if p.grad is not None]
            b.require(gradients and all(b.torch.isfinite(value).all().item() for value in gradients),
                'New down shape has missing or nonfinite gradients.')
            norm = sum(value.detach().double().square().sum().item() for value in gradients)
            b.require(norm > 0,'New down shape has zero total gradient.')
            before = [p.detach().clone() for p in parameters]
            result = step(*step_args,**step_kwargs)
            changed = sum(b.torch.count_nonzero(p.detach() != old).item() for p,old in zip(parameters,before,strict=True))
            b.require(changed > 0 and all(b.torch.isfinite(p).all().item() for p in parameters),
                'New down shape did not update finite parameters.')
            row['updates'].append(dict(step=len(row['updates'])+1,gradient_squared_norm=norm,
                changed_parameter_elements=changed,finite_gradients=True,finite_parameters=True))
            return result

        optimizer.step = observe.__get__(optimizer,type(optimizer))
        return optimizer

    if enabled:
        native.create_optimizer=create
    try:
        yield rows
    finally:
        native.create_optimizer=original
