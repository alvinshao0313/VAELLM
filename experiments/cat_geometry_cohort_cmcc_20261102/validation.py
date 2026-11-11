"""Read-only observations of the two real updates for each new geometry."""
from contextlib import contextmanager

from . import bindings as b


@contextmanager
def updates(native, *, enabled):
    rows = []
    original = native.create_optimizer

    def create(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        row = dict(stage=len(rows), updates=[])
        rows.append(row)
        step = optimizer.step

        def observe(_optimizer, *step_args, **step_kwargs):
            parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            b.require(gradients and all(b.torch.isfinite(value).all().item() for value in gradients),
                'New geometry has missing or nonfinite gradients.')
            norm = sum(value.detach().double().square().sum().item() for value in gradients)
            b.require(norm > 0, 'New geometry has zero total gradient.')
            before = [parameter.detach().clone() for parameter in parameters]
            result = step(*step_args, **step_kwargs)
            changed = sum(b.torch.count_nonzero(parameter.detach() != previous).item()
                for parameter, previous in zip(parameters, before))
            b.require(changed > 0 and all(b.torch.isfinite(value).all().item() for value in parameters),
                'New geometry did not update finite real parameters.')
            row['updates'].append(dict(step=len(row['updates'])+1, gradient_squared_norm=norm,
                changed_parameter_elements=changed, finite_gradients=True, finite_parameters=True))
            return result

        optimizer.step = observe.__get__(optimizer, type(optimizer))
        return optimizer

    if enabled:
        native.create_optimizer = create
    try:
        yield rows
    finally:
        native.create_optimizer = original
