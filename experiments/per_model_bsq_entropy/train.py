"""Isolated CAT regularizer comparison; the native BSQ and CAT sources stay unchanged.

Training: python -m experiments.per_model_bsq_entropy.train <native CAT arguments>
CPU check: CUDA_VISIBLE_DEVICES='' python -m experiments.per_model_bsq_entropy.train \
    --validate-cpu /absolute/path/to/fresh_report.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import weakref

import torch

from litebsq.bsq import BSQ


OBSERVED_CALLS = (1, 100, 1000, 10000)


@contextmanager
def per_model_marginal_entropy(*, telemetry: bool = True):
    """Pool batch samples within each independent CAT model, then average models."""
    original = BSQ.soft_entropy_loss
    counters = weakref.WeakKeyDictionary()
    next_instance = 0

    def loss(self, z):
        nonlocal next_instance
        if z.ndim != 4 or z.shape[0] < 1 or z.shape[1] < 1:
            raise ValueError(f'Expected CAT latent [B,M,C,K], got {tuple(z.shape)}.')
        if self.num_codebooks != 1 or z.shape[2] != 1 or z.shape[3] != self.codebook_dims:
            raise ValueError('This isolated CAT comparison requires C=1 and K=codebook_dims.')
        p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
        prob = torch.stack([p, 1 - p], dim=-1)  # [B,M,C,K,2]
        per_sample = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        per_model_prob = prob.mean(dim=0)  # [M,C,K,2]; never reduce the bit axis.
        per_model_entropy = self.get_entropy(per_model_prob, dim=-1, normalize=False).sum(dim=-1)
        marginal = per_model_entropy.mean()  # mean M; C is explicitly one.
        # Preserve the third return's native [K,2] shape; CAT forward ignores it.
        global_prob = per_model_prob.mean(dim=(0, 1))
        if telemetry:
            if self not in counters:
                next_instance += 1
                counters[self] = (next_instance, 0)
            instance, previous = counters[self]
            call = previous + 1
            counters[self] = (instance, call)
            if call in OBSERVED_CALLS:
                with torch.no_grad():
                    global_entropy = self.get_entropy(global_prob.detach(), dim=-1, normalize=False).sum()
                    marginal_values = per_model_entropy.detach().flatten()
                    bias = (per_model_prob.detach()[..., 0] - .5).abs()
                    print('[per_model_bsq_entropy] ' + json.dumps({
                        'instance': instance, 'call': call, 'rank': os.environ.get('RANK', '0'),
                        'latent_shape': list(z.shape), 'latent_dtype': str(z.dtype),
                        'sample_entropy': float(per_sample.detach()),
                        'global_marginal_entropy': float(global_entropy),
                        'mean_model_marginal_entropy': float(marginal.detach()),
                        'global_minus_mean_model_entropy': float(global_entropy - marginal.detach()),
                        'model_marginal_entropy_min': float(marginal_values.min()),
                        'model_marginal_entropy_max': float(marginal_values.max()),
                        'model_bit_probability_bias_abs_mean': float(bias.mean()),
                        'model_bit_probability_bias_abs_max': float(bias.max()),
                        'entropy_loss_weight': self.entropy_loss_weight,
                        'inv_temperature': self.inv_temperature,
                        'gamma0': self.gamma0, 'gamma': self.gamma, 'zeta': self.zeta,
                    }, sort_keys=True), flush=True)
        return per_sample, marginal, global_prob

    BSQ.soft_entropy_loss = loss
    try:
        yield
    finally:
        BSQ.soft_entropy_loss = original


def validate_cpu(output: Path):
    """Compare real BSQ forwards/backwards with a sum of native single-model calls."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('CPU validation requires CUDA_VISIBLE_DEVICES empty.')
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(31)
    ones = torch.ones(64, 1, 32)
    uneven = torch.linspace(.02, 1., 32).view(1, 1, 32).expand(64, 1, 32)
    cases = {
        'constant_complement': torch.cat((ones, -ones), 1),
        'uneven_complement': torch.cat((uneven, -uneven), 1),
        'random_B8192_M2': torch.randn(8192, 2, 32, generator=generator),
        'biased_B64_M2': torch.randn(64, 2, 32, generator=generator) + torch.tensor([.75, -.75]).view(1, 2, 1),
        'random_B64_M32': torch.randn(64, 32, 32, generator=generator),
        'random_B64_M36': torch.randn(64, 36, 32, generator=generator),
    }
    kwargs = dict(dim=32, entropy_loss_weight=.01, commitment_loss_weight=.25,
                  has_projections=False, spherical=True, new_quant=True,
                  gamma0=1., gamma=1., zeta=1., inv_temperature=100.)
    original = BSQ.soft_entropy_loss
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        for name, values in cases.items():
            x = values.to(dtype).detach().requires_grad_(True)
            count = x.shape[1]
            quantizer = BSQ(**kwargs).train()
            baseline, baseline_terms = quantizer(x, return_loss_breakdown=True)
            rng_before = torch.get_rng_state().clone()
            with per_model_marginal_entropy(telemetry=False):
                candidate, terms = quantizer(x, return_loss_breakdown=True)
                candidate_loss = candidate.entropy_aux_loss * (2.5 * count)
                candidate_grad = torch.autograd.grad(candidate_loss, x)[0]
            assert BSQ.soft_entropy_loss is original
            assert torch.equal(torch.get_rng_state(), rng_before)
            assert torch.equal(candidate.quantized, baseline.quantized)
            assert torch.equal(candidate.bit_indices, baseline.bit_indices)
            assert torch.equal(terms.per_sample_entropy, baseline_terms.per_sample_entropy)
            assert torch.equal(terms.commitment, baseline_terms.commitment)
            independent_loss = 0.
            gradients, marginals = [], []
            for index in range(count):
                one = values[:, index:index + 1].to(dtype).detach().requires_grad_(True)
                native, native_terms = quantizer(one, return_loss_breakdown=True)
                weighted = native.entropy_aux_loss * 2.5
                independent_loss += float(weighted.detach())
                gradients.append(torch.autograd.grad(weighted, one)[0])
                marginals.append(float(native_terms.batch_entropy))
            reference_grad = torch.cat(gradients, dim=1)
            torch.testing.assert_close(candidate_loss.detach(), torch.tensor(independent_loss), rtol=2e-5, atol=2e-8)
            torch.testing.assert_close(terms.batch_entropy.detach(), torch.tensor(sum(marginals) / count), rtol=2e-6, atol=2e-6)
            tolerance = dict(rtol=2e-5, atol=1e-9) if dtype == torch.float32 else dict(rtol=.02, atol=1e-7)
            torch.testing.assert_close(candidate_grad.float(), reference_grad.float(), **tolerance)
            gradient_delta = candidate_grad.float() - reference_grad.float()
            records.append({'name': name, 'dtype': str(dtype), 'shape': list(x.shape),
                'candidate_scaled_aux': float(candidate_loss.detach()), 'native_separate_scaled_sum': independent_loss,
                'loss_abs_difference': abs(float(candidate_loss.detach()) - independent_loss),
                'gradient_max_abs_difference': float(gradient_delta.abs().max()),
                'gradient_delta_l2': float(gradient_delta.double().norm()), 'gradient_tolerance': tolerance,
                'hard_output_exact': True, 'commitment_and_sample_entropy_exact': True,
                'rng_unchanged': True, 'scoped_override_restored': True})
    # Also exercise the detached observation path and restoration after an invalid shape.
    quantizer = BSQ(**kwargs).train()
    observation_input = cases['biased_B64_M2'].clone().requires_grad_(True)
    rng_before = torch.get_rng_state().clone()
    with per_model_marginal_entropy(telemetry=True):
        observed = quantizer(observation_input)
        observed_grad = torch.autograd.grad(observed.entropy_aux_loss, observation_input)[0]
        try:
            quantizer.soft_entropy_loss(torch.ones(2, 2, 32))
        except ValueError:
            pass
        else:
            raise AssertionError('Expected shape rejection.')
    with per_model_marginal_entropy(telemetry=False):
        silent_input = observation_input.detach().clone().requires_grad_(True)
        silent = quantizer(silent_input)
        silent_grad = torch.autograd.grad(silent.entropy_aux_loss, silent_input)[0]
    assert torch.equal(observed.entropy_aux_loss, silent.entropy_aux_loss)
    assert torch.equal(observed_grad, silent_grad)
    assert torch.equal(torch.get_rng_state(), rng_before) and BSQ.soft_entropy_loss is original
    assert not torch.cuda.is_initialized()
    source_paths = [Path(__file__).resolve(), Path(sys.modules[BSQ.__module__].__file__).resolve()]
    record = {'status': 'verified', 'cuda_initialized': False, 'python': sys.executable,
        'torch_version': torch.__version__, 'seed': 31, 'kwargs': kwargs, 'lfq_weight': 2.5,
        'scope': 'Native BSQ CPU equivalence to independent per-model calls; no CAT training or quality claim.',
        'semantics': 'CAT C=1: Hsample unchanged; Hmarginal=mean_M(sum_K(H(mean_B(prob)))); native commitment unchanged.',
        'telemetry_rng_loss_gradient_exact': True, 'cases': records,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'output': str(output), 'cases': len(records)}), flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--validate-cpu':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--validate-cpu', type=Path, required=True)
        args = parser.parse_args(argv)
        validate_cpu(args.validate_cpu.resolve())
        return
    from tools.cat_train import main as native_cat_main
    print('[per_model_bsq_entropy] isolated CAT marginal regularizer; C=1; native CLI unchanged', flush=True)
    with per_model_marginal_entropy():
        native_cat_main(argv)


if __name__ == '__main__':
    main()
