"""Read-only evidence of the actual N fitter parameters at optimizer creation."""
from contextlib import contextmanager

from experiments.adaptive_vector_rvq import common as c


@contextmanager
def observe(native):
    original = native.create_optimizer
    rows = []

    def capture(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        tensors = {f'group{index}.parameter{position}': parameter.detach()
            for index, group in enumerate(optimizer.param_groups) for position, parameter in enumerate(group['params'])}
        rows.append(dict(actual_stage_index=len(rows), initial_parameter_digest=c.digest(tensors),
            parameter_elements=sum(value.numel() for value in tensors.values()),
            parameter_dtypes=sorted({str(value.dtype) for value in tensors.values()}),
            parameter_shapes={name: list(value.shape) for name, value in tensors.items()},
            initial_learning_rates=[group['lr'] for group in optimizer.param_groups],
            seed=31, timing='Actual original optimizer constructed, before any VAE update; no parameter/RNG mutation'))
        return optimizer

    native.create_optimizer = capture
    try:
        yield rows
    finally:
        native.create_optimizer = original
