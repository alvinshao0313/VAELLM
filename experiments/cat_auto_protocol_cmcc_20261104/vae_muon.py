"""Actual-owner generalization of cat_down_vae_muon's fixed BF16 update rule.

Only owner discovery is generalized. No import of the historical N-family package.
"""
from copy import deepcopy
import hashlib
import math

import torch
from litebsq.parallel_layers import Normalize, ParallelLinear
from .contracts import require

SCHEMA = 'cat_auto_actual_owner_muon_v0'


def tensor_digest(values):
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        require(isinstance(value, torch.Tensor), 'Digest requires real tensor leaves.')
        tensor = value.detach().cpu().contiguous()
        result.update(str((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        result.update(tensor.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def derive_layout(vae):
    named, layout, affine = dict(vae.named_parameters()), {}, []
    for role in ('encoder', 'decoder'):
        module = getattr(vae.model, role)
        require(module.num_res_blocks == 0, 'The v0 owner contract is R0.')
        for part in ('linear_in', 'linear_out'):
            linear = getattr(module, part)
            require(isinstance(linear, ParallelLinear) and linear.conv.groups == linear.num_models
                and linear.conv.kernel_size == (1,), 'Use actual ParallelLinear ownership.')
            name = f'model.{role}.{part}.conv.weight'
            shape = [linear.num_models, linear.out_features, linear.in_features]
            require(named[name] is linear.conv.weight and list(named[name].shape) == [shape[0] * shape[1], shape[2], 1], 'Actual packed owner mismatch.')
            layout[name] = dict(role=role, num_models=shape[0], matrix_shape=shape, packed_shape=list(named[name].shape))
            bias = f'model.{role}.{part}.conv.bias'
            require(named[bias] is linear.conv.bias, 'Auxiliary bias owner mismatch.')
            affine.append(bias)
        norm = module.norm_out
        require(isinstance(norm, Normalize) and norm.norm_type == 'layer', 'LayerNorm leaves remain auxiliary Adam parameters.')
        for leaf in ('weight', 'bias'):
            name = f'model.{role}.norm_out.{leaf}'
            require(named[name] is getattr(norm, leaf) and named[name].shape[0] == module.num_models, 'Actual normalization owner mismatch.')
            affine.append(name)
    require(set(named) == set(layout) | set(affine) and len({id(value) for value in named.values()}) == len(named)
        and all(value.dtype == torch.bfloat16 for value in named.values()), 'Every BF16 leaf needs exactly one owner.')
    return named, layout, affine


def matrices(value, spec):
    require(list(value.shape) == spec['packed_shape'], 'Packed owner storage changed.')
    return value[..., 0].reshape(spec['matrix_shape'])


def zeropower(value):
    require(value.ndim == 3, 'Keep actual per-linear matrices independent.')
    x = value.to(torch.bfloat16)
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        a = x @ x.mT
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    return x.mT if transpose else x


def direction(gradient, momentum, spec):
    momentum.lerp_(gradient.float(), .05)
    nesterov = gradient.float().lerp(momentum, .95)
    return (zeropower(matrices(nesterov, spec)).float() * (.2 * math.sqrt(max(spec['matrix_shape'][-2:])))).reshape_as(gradient)


def candidate(parameter, update, lr):
    return (parameter.float() - float(lr) * update).to(torch.bfloat16)


def solve_kappa(parameters, updates, reference_norm, base_lr=.003):
    unorm = math.sqrt(float(torch.stack([value.double().square().sum() for value in updates]).sum()))
    require(reference_norm > 0 and unorm > 0 and math.isfinite(reference_norm) and math.isfinite(unorm), 'Fresh BF16 Adam reference and Muon direction must be finite and nonzero.')
    initial, observed = reference_norm / (base_lr * unorm), {0.: 0.}

    def evaluate(kappa):
        if kappa not in observed:
            observed[kappa] = math.sqrt(float(torch.stack([(candidate(p, u, base_lr * kappa).float() - p.float()).double().square().sum()
                for p, u in zip(parameters, updates, strict=True)]).sum()))
        return observed[kappa]

    evaluate(initial)
    low, high, doublings = 0., 2 * initial, 0
    while evaluate(high) < reference_norm and doublings < 2:
        high *= 2
        doublings += 1
    upper_covers = evaluate(high) >= reference_norm
    for _ in range(12):
        middle = (low + high) / 2
        if evaluate(middle) < reference_norm:
            low = middle
        else:
            high = middle
    best = min(observed, key=lambda value: abs(observed[value] / reference_norm - 1))
    return dict(kappa=best, reference_BF16_displacement_norm=reference_norm, fresh_Muon_FP32_direction_norm=unorm,
        kappa0=initial, matched_BF16_displacement_norm=observed[best], relative_error=abs(observed[best] / reference_norm - 1),
        maximum_relative_error=.01, upper_doublings=doublings, bisections=12, upper_covers_reference=upper_covers,
        observed_amplitudes=[dict(kappa=key, norm=value) for key, value in observed.items()])


class VAEMuon(torch.optim.Optimizer):
    def __init__(self, vae, *, multiplier, evidence):
        named, self.layout, affine = derive_layout(vae)
        self.named, self.evidence, self.actual_step = named, evidence, 0
        self.matrix_names = list(self.layout)
        self.multiplier = float(multiplier)
        super().__init__([dict(params=[named[name] for name in self.matrix_names], group_name='muon_matrix', kappa=None),
            dict(params=[named[name] for name in affine], group_name='aux_adam')], dict(lr=.003 * self.multiplier, weight_decay=0.))
        self._adam = torch.optim.AdamW([self.param_groups[1]], lr=.003 * self.multiplier, betas=(.9, .95), eps=1e-8, weight_decay=0.)
        self._adam.state = self.state
        evidence['owner_layout'] = deepcopy(self.layout)

    @torch.no_grad()
    def calibrate(self):
        require(not self.state and self.actual_step == 0, 'Calibrate each genuinely fresh stage once.')
        params = self.param_groups[0]['params']
        before = tensor_digest(self.named)
        grads = tensor_digest({name: value.grad for name, value in self.named.items()})
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(params[0].device) if params[0].is_cuda else None
        clones = [value.detach().clone().requires_grad_() for value in params]
        for parameter, clone in zip(params, clones, strict=True):
            clone.grad = parameter.grad.detach().clone()
        reference = torch.optim.AdamW(clones, lr=.003, betas=(.9, .95), eps=1e-8, weight_decay=0.)
        reference.step()
        require(all(reference.state[value]['exp_avg'].dtype == reference.state[value]['exp_avg_sq'].dtype == torch.bfloat16 for value in clones), 'Reference moments must remain native BF16.')
        norm = math.sqrt(float(torch.stack([(clone.float() - parameter.float()).double().square().sum()
            for parameter, clone in zip(params, clones, strict=True)]).sum()))
        updates = [direction(self.named[name].grad, torch.zeros_like(self.named[name], dtype=torch.float32), self.layout[name]) for name in self.matrix_names]
        result = solve_kappa(params, updates, norm)
        require(result['relative_error'] <= .01, 'Bounded actual BF16 amplitude calibration exceeds1%; no automatic LR/precision fallback.')
        require(before == tensor_digest(self.named) and grads == tensor_digest({name: value.grad for name, value in self.named.items()})
            and torch.equal(rng, torch.get_rng_state()) and (cuda_rng is None or torch.equal(cuda_rng, torch.cuda.get_rng_state(params[0].device)))
            and not self.state, 'Calibration changed live parameters, gradients, RNG or optimizer state.')
        result.update(reference_lr=.003, actual_multiplier_after_calibration=self.multiplier, state_unchanged=True)
        self.evidence['calibration'] = result
        self.param_groups[0]['kappa'] = result['kappa']

    @torch.no_grad()
    def step(self, closure=None):
        require(closure is None and all(value.grad is not None and value.dtype == value.grad.dtype == torch.bfloat16
            for value in self.named.values()) and torch.stack([torch.isfinite(value.grad).all() for value in self.named.values()]).all().item(),
            'Muon requires finite actual BF16 gradients.')
        if self.param_groups[0]['kappa'] is None:
            self.calibrate()
        self.actual_step += 1
        updates = []
        for name, parameter in zip(self.matrix_names, self.param_groups[0]['params'], strict=True):
            state = self.state[parameter]
            if not state:
                state.update(momentum_buffer=torch.zeros_like(parameter, dtype=torch.float32), step=0)
            update = direction(parameter.grad, state['momentum_buffer'], self.layout[name])
            updates.append(update)
            state['step'] += 1
        require(torch.stack([torch.isfinite(value).all() for value in updates]).all().item(), 'Nonfinite Muon direction.')
        for parameter, update in zip(self.param_groups[0]['params'], updates, strict=True):
            parameter.copy_(candidate(parameter, update, self.param_groups[0]['lr'] * self.param_groups[0]['kappa']))
        self._adam.step()
        require(torch.stack([torch.isfinite(value).all() for value in self.named.values()]).all().item(), 'Nonfinite updated parameters.')

    def state_dict(self):
        result = super().state_dict()
        result.update(schema=SCHEMA, layout=deepcopy(self.layout), actual_step=self.actual_step, multiplier=self.multiplier)
        return result

    def load_state_dict(self, state_dict):
        require(state_dict.get('schema') == SCHEMA and state_dict.get('layout') == self.layout
            and state_dict.get('multiplier') == self.multiplier, 'Only the exact actual-owner optimizer contract may resume.')
        saved = state_dict['param_groups'][0]['params']
        momentum = {parameter: state_dict['state'][key]['momentum_buffer'].to(device=parameter.device, dtype=torch.float32).clone()
            for parameter, key in zip(self.param_groups[0]['params'], saved, strict=True)}
        require(all(state_dict['state'][key]['momentum_buffer'].dtype == torch.float32 for key in saved), 'Saved matrix momentum must already be FP32.')
        super().load_state_dict(state_dict)
        for parameter, value in momentum.items():
            self.state[parameter]['momentum_buffer'] = value
        self.actual_step = int(state_dict['actual_step'])
        self._adam.param_groups, self._adam.state = [self.param_groups[1]], self.state
