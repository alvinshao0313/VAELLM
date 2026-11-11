"""Reuse the unchanged fitter with private bindings and read-only Adam telemetry."""
from contextlib import contextmanager
from types import FunctionType

from . import bindings as b
from experiments.down_geometry_baseline_cmcc_20261103 import fit as mature


@contextmanager
def observe_lr(native, lr):
    factory, rows, handles = native.create_optimizer, [], []

    def create(*args, **kwargs):
        optimizer = factory(*args, **kwargs)
        b.require(type(optimizer) is b.torch.optim.AdamW and all(group['lr'] == lr
            and group['betas'] == (.9, .95) and group['eps'] == 1e-8 and group['weight_decay'] == 0
            for group in optimizer.param_groups), 'The actual original BF16 AdamW groups changed beyond LR.')
        row = dict(stage_index=len(rows), base_lr=lr, actual_steps=0, observations=[])
        rows.append(row)

        def observed(actual, _args, _kwargs):
            row['actual_steps'] += 1
            step = row['actual_steps']
            if step not in (1, 2, 10000):
                return
            rates = [group['lr'] for group in actual.param_groups]
            expected = lr * (1. - (step - 1) / 10000)
            b.require(all(abs(rate - expected) <= max(1e-15, abs(expected) * 1e-10) for rate in rates),
                'The actual scheduled update LR differs from the original linear10000 horizon.')
            states = [actual.state[p] for group in actual.param_groups for p in group['params']]
            b.require(states and all(int(state['step']) == step and state['exp_avg'].dtype == b.torch.bfloat16
                and state['exp_avg_sq'].dtype == b.torch.bfloat16
                and b.torch.isfinite(state['exp_avg']).all().item()
                and b.torch.isfinite(state['exp_avg_sq']).all().item() for state in states),
                'Actual BF16 Adam state must remain finite at the declared real step.')
            item = dict(step=step, learning_rates=rates, original_linear10000_expected=expected,
                BF16_moments_finite=True)
            row['observations'].append(item)
            print(dict(event='actual_down_LR_update', stage_index=row['stage_index'], **item), flush=True)
        handles.append(optimizer.register_step_post_hook(observed))
        return optimizer

    native.create_optimizer = create
    try:
        yield rows
        b.require(len(rows) == 2 and all(row['actual_steps'] == 10000
            and len(row['observations']) == 3 for row in rows), 'Both real stages must complete10000 actual Adam updates.')
    finally:
        native.create_optimizer = factory
        for handle in handles:
            handle.remove()


def execute(model, parsed, activation_runtime, plan, *, arm, output, short):
    b.require(not short, 'Normal scalar-LR candidates reuse applicable mature evidence and run formal directly.')
    function = FunctionType(mature.execute.__code__, {**mature.execute.__globals__, 'b': b},
        name=mature.execute.__name__, argdefs=mature.execute.__defaults__, closure=mature.execute.__closure__)
    function.__kwdefaults__ = mature.execute.__kwdefaults__
    with observe_lr(b.recovery.native, b.ARMS[arm]['lr']) as actual:
        result = function(model, parsed, activation_runtime, plan, arm=arm, output=output, short=False)
    reference = b.initialization_reference()
    matches = [row['initial_parameter_digest'] == expected['initial_parameter_digest']
        for row, expected in zip(result['actual_optimizer_initializations'], reference, strict=True)]
    b.require(len(matches) == 2 and all(matches), 'Both stage initial tensors must match the actual original D16 control.')
    result.update(comparison_D16_initialization_match=matches, actual_Adam_stages=actual,
        LR_override_scope='Only newly fitted down; all historical prefix weights/configuration remain from the bound parent.',
        actual_base_lr=b.ARMS[arm]['lr'], scalar_LR_multiplier=b.ARMS[arm]['lr'] / .003)
    return result
