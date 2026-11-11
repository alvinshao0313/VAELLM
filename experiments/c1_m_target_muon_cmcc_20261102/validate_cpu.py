"""Check isolated imports/CLI routing and the target coefficients before real M32."""
import ast
import os
from pathlib import Path
import sys

from . import bindings as b
from . import objective, fit, run, fresh, evaluate, sequence


def main():
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized(), 'Use CPU-only bitvae.')
    b.configure()
    parsed = []
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        parsed.append(str(path))
    b.require(fit.b is b and objective.b is b and run.b is b and fresh.b is b and evaluate.r is b
        and sequence.b is b and b.mature.STEPS == b.STEPS,
        'Every new entrypoint must retain the same isolated target-study binding.')
    fit_tree = ast.parse(Path(fit.__file__).read_text(encoding='utf-8'))
    optimizer_loops = [node for node in ast.walk(fit_tree) if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Attribute) and node.iter.attr == 'param_groups']
    b.require(len(optimizer_loops) == 1 and isinstance(optimizer_loops[0].target, ast.Name)
        and optimizer_loops[0].target.id == 'optimizer_group',
        'Optimizer LR scaling must preserve the training data group used by the update log.')
    b.require(b.PREPARED.parent == b.RUN_OUTPUT and b.RUN_OUTPUT.parent == b.OUTPUT
        and b.RUN_OUTPUT.name == b.EVIDENCE.name == b.REVISION,
        'Bind fresh preparation, arm outputs and evidence to the same independent revision.')
    # A mixed-target KL and the half KL/half CE objective have identical gradients;
    # this verifies why the completed mixed-target control remains comparable.
    logits = b.torch.tensor([[1., -.3, .4, -1.]], dtype=b.torch.float64, requires_grad=True)
    teacher = b.torch.tensor([[.1, .2, .3, .4]], dtype=b.torch.float64)
    label = b.torch.tensor([[0., 1., 0., 0.]], dtype=b.torch.float64)
    student = logits.log_softmax(-1)
    mixed = .5 * teacher + .5 * label
    control = .5 * b.torch.nn.functional.kl_div(student, mixed, reduction='sum')
    explicit = .5 * (.5 * b.torch.nn.functional.kl_div(student, teacher, reduction='sum') - .5 * student[0, 1])
    first = b.torch.autograd.grad(control, logits, retain_graph=True)[0]
    second = b.torch.autograd.grad(explicit, logits)[0]
    b.require(b.torch.allclose(first, second, rtol=0, atol=1e-15), 'Teacher/gold fractions or outer M .5 are inconsistent.')
    destination = b.EVIDENCE / 'cpu'
    destination.mkdir(parents=True, exist_ok=True)
    b.require(not (destination / 'validation_parallel.json').exists(), 'Never overwrite parallel CPU validation.')
    b.require(b.ARMS == ('full_teacher', 'pure_gold', 'conditional_lr_half', 'conditional_lr_double')
        and b.LR_MULTIPLIERS == dict(full_teacher=1., pure_gold=1., conditional_lr_half=.5, conditional_lr_double=2.),
        'Keep the four reviewed arms and overall schedule multipliers.')
    b.core.write(destination / 'validation_parallel.json', dict(status='verified_target_math_imports_and_syntax',
        syntax=parsed, old_mixed_vs_explicit_gradient_max_difference=float((first - second).abs().max()),
        data_group_not_shadowed_by_optimizer_loop=True, revision=b.REVISION,
        CUDA_initialized=b.torch.cuda.is_initialized(), implementation=b.code_bindings(),
        boundary='CPU algebra/imports and revision regression only; reuse the completed independent fixed-M32 real backward probe.'))
    print(dict(status='verified_target_math_imports_and_syntax', files=len(parsed)), flush=True)


if __name__ == '__main__':
    main()
