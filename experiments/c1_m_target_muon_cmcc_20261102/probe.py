"""One fixed first-train-M32 real backward for each objective; no update or export."""
import os
import time

from . import bindings as b
from . import objective


def main():
    b.configure()
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '4' and b.torch.cuda.device_count() == 1,
        'Use the assigned single physical GPU4 for the bounded fixed-M32 probe.')
    began = time.monotonic()
    prepared = b.read_prepared(b.PREPARED)
    b.require(not (b.OUTPUT / 'probe.json').exists(), 'Never overwrite a target probe.')
    targets, manifest = b.load_full_targets(prepared)
    dataset, recovery, _, _ = b.inputs.load_inputs()
    cases = [recovery[key]['case'] for key in dataset['groups'][0]['case_ids'][:32]]
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    model, meta, strict, budget = b.checkpoint.load(prepared['source_binding']['checkpoint_dir'])
    b.require(meta['checkpoint_id'] == prepared['record']['source_checkpoint_id']
        and not strict.missing_keys and not strict.unexpected_keys and budget == prepared['source_budget'], 'Use original C1 raw.')
    model.to('cuda:0')
    modules = b.mature.selected_modules(model)
    active = b.lifecycle.training_mode(model, modules)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.config.use_cache = False
    frozen = b.lifecycle.frozen_proof(model, set(active))
    before = b.lifecycle.copy_parameters(modules)
    records, gradients = {}, {}
    # The conditional mixture control is computed from its original target cache.
    for arm in ('conditional_control', *b.TARGET_ARMS):
        model.zero_grad(set_to_none=True)
        if arm == 'conditional_control':
            distribution = b.mature.mmlu.log_distribution(model, cases)
            target = b.torch.cat([recovery[case['case_id']]['target_log_probs'] for case in cases])
            loss, _ = b.mature.mmlu.weighted_loss(distribution, target)
            detail = dict(weighted_mixed_target_KL=float(loss.detach()))
        else:
            loss, detail = objective.loss(model, cases, targets, arm, diagnostics=True)
        loss.backward()
        b.require(all(p.grad is not None and p.grad.dtype == b.torch.float32
            and b.torch.isfinite(p.grad).all().item() for p in active.values())
            and all(p.grad is None for name, p in model.named_parameters() if name not in active),
            'Every real D gradient must be finite, with the frozen complement untouched.')
        norm = sum(float(p.grad.double().square().sum()) for p in active.values()) ** .5
        b.require(norm > 0, 'The declared target must produce a real nonzero full-D gradient.')
        gradients[arm] = {name: p.grad.detach().cpu().clone() for name, p in active.items()}
        records[arm] = dict(weighted_loss=float(loss.detach()), gradient_norm=norm,
            trainable_tensors=len(active), diagnostics=detail, memory=b.rollout.memory())
        print(dict(event='target_M32_backward', arm=arm, loss=float(loss.detach()), gradient_norm=norm), flush=True)
        del loss
    reference = gradients['conditional_control']
    for arm in b.TARGET_ARMS:
        dot = sum(float((reference[name].double() * gradients[arm][name].double()).sum()) for name in active)
        records[arm]['gradient_cosine_to_control'] = dot / (records[arm]['gradient_norm'] * records['conditional_control']['gradient_norm'])
    b.require(all(b.torch.equal(value, before[name]) for name, value in b.lifecycle.copy_parameters(modules).items())
        and b.lifecycle.frozen_proof(model, set(active)) == frozen == prepared['record']['validated_frozen_state'],
        'Readout/backward probe must leave every raw C1 parameter unchanged.')
    model.zero_grad(set_to_none=True)
    b.core.write(b.OUTPUT / 'probe.json', dict(status='verified_fixed_first_train_M32_real_backward',
        source_checkpoint_id=meta['checkpoint_id'], records=records, group_index=0, optimizer_updates=0,
        teacher_cache=b.binding(b.OUTPUT / 'teacher/cache.json'), implementation=b.code_bindings(),
        all_source_parameters_unchanged=True, physical_GPU=4, elapsed_seconds=time.monotonic() - began,
        boundary='Target support/positions/finite full-D backward only; quality and convergence remain untested.'))


if __name__ == '__main__':
    main()
