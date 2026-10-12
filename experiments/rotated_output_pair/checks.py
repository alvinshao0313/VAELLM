"""Small CPU algebra checks establish scaling semantics, not compression quality."""
from copy import deepcopy
import torch

from experiments.linear_output.objectives import full_vae_forward
from experiments.linear_output.output_kernel import output_mse
from experiments.rotation_factor_probe.capture import require
from litebsq.llm_vae import MultiLayerVAE
from rotation.weight_preprocess import build_weight_rotation, rotation_spec


def verify(args):
    gen = torch.Generator().manual_seed(31)
    rotation = build_weight_rotation(32, 32, rotation_spec('two_sided', 32, 31, 'cpu_scaling'))
    target = torch.randn(32, 32, generator=gen)
    normalized = torch.randn(32, 32, generator=gen, requires_grad=True)
    scale = 0.3
    qp_error = torch.randn(32, 32, generator=gen) * .02
    reconstructed = rotation(normalized * scale + target, inverse=True)
    weight = torch.cat([rotation(target, inverse=True), torch.zeros_like(target)], dim=1)
    actual = torch.cat([reconstructed, qp_error], dim=1)
    x = torch.eye(64) * 8
    loss = output_mse(actual, weight, x, use_triton=False) / (scale ** 2 * 32)
    reference = normalized.square().mean() + qp_error.square().sum() / (32 * 32 * scale ** 2)
    actual_grad = torch.autograd.grad(loss, normalized, retain_graph=True)[0]
    reference_grad = 2 * normalized / normalized.numel()
    torch.testing.assert_close(loss, reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-5, atol=2e-6)
    x = torch.randn(47, 64, generator=gen); x[:, 32:] += x[:, :32]
    trace = float(x[:, :32].double().square().sum() / len(x))
    complete = output_mse(actual, weight, x, use_triton=False) / (scale ** 2 * trace)
    error = x @ (actual - weight).T
    reference = error.square().mean() / (scale ** 2 * trace)
    ga = torch.autograd.grad(complete, normalized, retain_graph=True)[0]
    gr = torch.autograd.grad(reference, normalized)[0]
    torch.testing.assert_close(complete, reference); torch.testing.assert_close(ga, gr, rtol=2e-5, atol=2e-6)
    separate = (x[:, :32] @ (reconstructed - weight[:, :32]).T).square().mean()
    separate += (x[:, 32:] @ qp_error.T).square().mean()
    cross = float(error.square().mean() - separate)
    require(abs(cross) > 1e-7, 'Correlated protected cross term was not exercised.')
    torch.manual_seed(31)
    vae = MultiLayerVAE(deepcopy(args)).train()
    blocks = torch.randn(128, 1, 32, generator=gen).to(torch.bfloat16)
    split, aux_split, bits_split = full_vae_forward(vae, blocks, chunk_vectors=32, recompute=False)
    full, aux_full, bits_full = full_vae_forward(vae, blocks, chunk_vectors=128, recompute=False)
    require(torch.equal(bits_split, bits_full), 'Memory chunking changed these actual native test codes.')
    torch.testing.assert_close(split, full, rtol=.01, atol=.002)
    torch.testing.assert_close(aux_split, aux_full, rtol=1e-6, atol=1e-6)
    (split.float().square().mean() + aux_split).backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in vae.parameters()),
            'CPU native BF16 checkpointed encoder/decoder backward failed.')
    return {'isotropic_with_protection_constant_matches_normalized_wmse': True,
            'isotropic_gradient_matches_normalized_wmse': True, 'correlated_full_output_value_and_gradient': True,
            'observed_nonzero_protected_cross_term': cross, 'native_bf16_chunked_global_bsq_and_backward': True,
            'scope': 'Small CPU algebra and native module checks only; no GPU, true-matrix resource or quality proof.'}
