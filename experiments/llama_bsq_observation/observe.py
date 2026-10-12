"""Read-only hooks on the real CAT graph; no replacement loss or optimizer."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

from litebsq.autoencoder import MultiLayerVAE
from litebsq.bsq import BSQ


class ObservationComplete(Exception):
    """The requested native optimizer prefix completed."""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def norms(grads, encoder_count):
    def norm(values):
        return sum(float(g.detach().double().square().sum()) for g in values if g is not None) ** .5
    return {'encoder_l2': norm(grads[:encoder_count]), 'decoder_l2': norm(grads[encoder_count:]),
            'unused_encoder': sum(g is None for g in grads[:encoder_count]),
            'unused_decoder': sum(g is None for g in grads[encoder_count:])}


class Observer:
    def __init__(self, output: Path):
        self.output = output
        self.calls = self.updates = 0
        self.selected = (1, 16, 64, 128)
        self.model = None
        self.params = []
        self.terms = self.row = self.expected_grads = None
        self.rows = []
        self.handles = []

    def __enter__(self):
        self.handles = [register_module_forward_pre_hook(self.before),
                        register_module_forward_hook(self.after),
                        register_optimizer_step_pre_hook(self.before_step),
                        register_optimizer_step_post_hook(self.after_step)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def before(self, module, inputs):
        if not isinstance(module, MultiLayerVAE):
            return
        require(module.training and torch.is_grad_enabled(), 'Only the first native training stage is observed.')
        if self.model is None:
            self.model = module
            self.encoder = tuple(module.model.encoder.parameters())
            self.decoder = tuple(module.model.decoder.parameters())
            self.params = [*self.encoder, *self.decoder]
            require(set(map(id, self.params)) == set(map(id, module.parameters())), 'Unexpected trainable VAE state.')
            require(module.num_models == 32 and module.model.chunk_size == module.model.latent_dim == 32
                    and module.model.encoder.hidden_dim == module.model.decoder.hidden_dim == 128
                    and module.model.encoder.num_res_blocks == module.model.decoder.num_res_blocks == 0
                    and module.model.recon_loss_type == 'mse', 'Unexpected native gate geometry.')
        require(module is self.model and self.calls == self.updates, 'Unexpected VAE stage or nested training call.')
        require(tuple(inputs[0].shape) == (8192, 32, 32), 'Do not reduce the native batch or Linear count.')
        self.calls += 1

    def after(self, module, inputs, output):
        if self.calls not in self.selected or self.model is None:
            return
        if isinstance(module, BSQ) and module is self.model.model.quantizer:
            require(self.terms is None and module.training and module.new_quant, 'Unexpected BSQ call.')
            rng = torch.get_rng_state(), torch.cuda.get_rng_state(inputs[0].device)
            # Invoke the unchanged native method on the SAME encoder activation. This
            # bypasses hooks on this module, not its computation, and never replaces output.
            replay, terms = module.forward(inputs[0], return_loss_breakdown=True)
            require(torch.equal(replay.quantized, output.quantized)
                    and torch.equal(replay.bit_indices, output.bit_indices)
                    and torch.equal(replay.entropy_aux_loss, output.entropy_aux_loss),
                    'Native diagnostic replay differs from the actual training quantizer.')
            require(torch.equal(rng[0], torch.get_rng_state())
                    and torch.equal(rng[1], torch.cuda.get_rng_state(inputs[0].device)), 'BSQ replay consumed RNG.')
            self.terms = terms
            with torch.no_grad():
                fraction = output.bit_indices.float().mean(dim=0)
                require(tuple(fraction.shape) == (32, 32), 'Expected native hard bit frequencies per Linear.')
                probabilities = torch.stack([fraction, 1 - fraction], dim=-1)
                hard_entropy = module.get_entropy(probabilities, normalize=False).sum(dim=-1)
                self.row = {'step': self.calls, 'hard_replay_exact': True, 'aux_replay_exact': True,
                    'native_latent_dtype': str(inputs[0].dtype), 'native_bits_shape': list(output.bit_indices.shape),
                    'C': float(terms.commitment.detach()), 'Hsample': float(terms.per_sample_entropy.detach()),
                    'Hpool': float(terms.batch_entropy.detach()),
                    'hard_one_fraction_per_linear_bit': fraction.cpu().tolist(),
                    'hard_marginal_entropy_per_linear': hard_entropy.cpu().tolist(),
                    'hard_constant_bit_count': int(((fraction == 0) | (fraction == 1)).sum()),
                    'hard_bit_abs_bias_mean': float((fraction - .5).abs().mean()),
                    'hard_bit_abs_bias_max': float((fraction - .5).abs().max())}
        elif module is self.model:
            self.measure(output[1])

    def measure(self, losses):
        require(self.terms is not None and all(p.grad is None for p in self.params), 'Observe before original backward.')
        versions = [p._version for p in self.params]
        device = self.params[0].device
        rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        q, ae, terms = self.model.model.quantizer, self.model.model, self.terms
        factor = ae.lfq_weight * ae.num_models
        components = {'reconstruction': losses['train/recon_loss'],
            'commitment': terms.commitment * q.commitment_loss_weight * factor,
            'sample_entropy': (q.zeta * q.gamma0 * terms.per_sample_entropy / q.inv_temperature) * q.entropy_loss_weight * factor,
            'pool_entropy': -(q.zeta * q.gamma * terms.batch_entropy / q.inv_temperature) * q.entropy_loss_weight * factor}
        self.row['weighted_losses'] = {name: float(loss.detach()) for name, loss in components.items()}
        self.row['native_aux'] = float(losses['train/commitment_loss'].detach())
        self.row['native_total'] = float(losses['loss'].detach())
        self.row['gradient_norms'] = {}
        component_sum = [torch.zeros_like(p) for p in self.params]
        for name, loss in components.items():
            grads = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
            require(all(g is None or torch.isfinite(g).all() for g in grads), 'Nonfinite diagnostic gradient.')
            self.row['gradient_norms'][name] = norms(grads, len(self.encoder))
            for total, grad in zip(component_sum, grads):
                if grad is not None:
                    total.add_(grad.detach())
        native = torch.autograd.grad(losses['loss'], self.params, retain_graph=True)
        self.expected_grads = [g.detach().clone() for g in native]
        self.row['gradient_norms']['native_total'] = norms(native, len(self.encoder))
        self.row['gradient_norms']['component_sum_minus_native'] = norms(
            [a - b.detach() for a, b in zip(component_sum, native)], len(self.encoder))
        # BF16 accumulation may make the sum of separate backward calls differ slightly;
        # report this decomposition residual, never replace the native backward result.
        require(versions == [p._version for p in self.params] and all(p.grad is None for p in self.params),
                'Observer changed parameters or accumulated parameter gradients.')
        require(torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state(device)),
                'Gradient observation consumed RNG.')
        self.row['observer_rng_parameters_and_grad_unchanged'] = True
        self.terms = None

    def before_step(self, optimizer, _args, _kwargs):
        require(self.model is not None and self.calls == self.updates + 1, 'Unexpected optimizer call.')
        require(set(map(id, self.params)) == {id(p) for g in optimizer.param_groups for p in g['params']},
                'Optimizer must contain exactly the real encoder and decoder parameters.')
        if self.calls not in self.selected:
            return
        require(self.expected_grads is not None, 'Missing original-graph gradient reference.')
        deltas, exact = [], True
        for param, expected in zip(self.params, self.expected_grads):
            require(param.grad is not None and torch.isfinite(param.grad).all(), 'Invalid native backward.')
            actual = param.grad.detach()
            exact = exact and torch.equal(actual, expected)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            deltas.append(actual - expected)
        self.row['native_backward_vs_observed_total'] = {**norms(deltas, len(self.encoder)), 'exact': exact}
        self.row['lr'] = [float(group['lr']) for group in optimizer.param_groups]
        self.before_update = [p.detach().clone() for p in self.params]
        self.expected_grads = None

    def after_step(self, _optimizer, _args, _kwargs):
        self.updates += 1
        if self.calls in self.selected:
            self.row['actual_parameter_update'] = norms(
                [p.detach() - old for p, old in zip(self.params, self.before_update)], len(self.encoder))
            self.before_update = None
            self.rows.append(self.row)
            with (self.output / 'observations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(self.row, ensure_ascii=False, sort_keys=True) + '\n')
            print('[native_bsq_observer] ' + json.dumps({key: self.row[key] for key in
                ('step', 'C', 'Hsample', 'Hpool', 'weighted_losses', 'gradient_norms', 'actual_parameter_update')}), flush=True)
            self.row = None
        if self.updates == 128:
            raise ObservationComplete('128 native stage-1 optimizer updates completed; no checkpoint export.')
