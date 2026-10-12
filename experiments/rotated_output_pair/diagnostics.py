"""Read-only reconstruction observations; never change optimizer inputs or states."""
from copy import deepcopy

import torch

from experiments.linear_output.objectives import full_vae_forward
from experiments.rotation_factor_probe.capture import require
from litebsq.llm_vae import MultiLayerVAE
from train_utils.cat_train_data import restore_stage_norm


@torch.no_grad()
def reconstruct_readonly(vae, blocks, mean, scale, *, chunk_vectors):
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(blocks.device) if blocks.is_cuda else None
    modes = [(m, m.training) for m in vae.modules()]
    tensors = list(vae.parameters()) + list(vae.buffers())
    versions = [v._version for v in tensors]
    gradients = [None if p.grad is None else p.grad.clone() for p in vae.parameters()]
    values = []
    try:
        vae.eval()
        for chunk in blocks.split(chunk_vectors):
            value, _ = vae(chunk, is_train=False)
            values.append(value.float())
        result = restore_stage_norm(torch.cat(values), mean=mean, scale=scale)
    finally:
        for module, mode in modes:
            module.training = mode
    require(versions == [v._version for v in tensors], 'Read-only decode mutated VAE state.')
    require(torch.equal(cpu_rng, torch.get_rng_state()), 'Read-only decode changed CPU RNG.')
    if cuda_rng is not None:
        require(torch.equal(cuda_rng, torch.cuda.get_rng_state(blocks.device)), 'Read-only decode changed CUDA RNG.')
    for before, parameter in zip(gradients, vae.parameters()):
        require((before is None and parameter.grad is None) or
                (before is not None and parameter.grad is not None and torch.equal(before, parameter.grad)),
                'Read-only decode changed accumulated gradients.')
    return result


@torch.no_grad()
def physical_sse(actual, reference, inputs):
    result = {}
    delta = actual.float() - reference.float()
    for domain, x in inputs.items():
        error, energy = 0., 0.
        for chunk in x.split(256):
            x32 = chunk.to(device=actual.device, dtype=torch.float32)
            error += float((x32 @ delta.T).double().square().sum())
            energy += float((x32 @ reference.float().T).double().square().sum())
        result[domain] = {'valid_tokens': len(x), 'sse': error, 'reference_energy': energy, 'nmse': error / energy}
    return result


def verify_readonly(args):
    """CPU native BF16 check: inserting observations leaves the next update exact."""
    torch.manual_seed(31)
    left = MultiLayerVAE(deepcopy(args)).train()
    right = deepcopy(left)
    blocks = torch.randn(128, 1, 32).to(torch.bfloat16)
    x = torch.randn(17, 32)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.003, weight_decay=0.) for m in (left, right)]
    for step in range(2):
        for model, optimizer in zip((left, right), optimizers):
            optimizer.zero_grad(set_to_none=True)
            decoded, auxiliary, _ = full_vae_forward(model, blocks, chunk_vectors=32, recompute=False)
            ((decoded.float() - blocks.float()).square().mean() + auxiliary).backward()
            optimizer.step()
            if model is right:
                reconstructed = reconstruct_readonly(model, blocks, torch.tensor(0.), torch.tensor(1.), chunk_vectors=32)
                observed = physical_sse(reconstructed.reshape(128, 32), blocks.float().reshape(128, 32), {'train': x})
                require(observed['train']['valid_tokens'] == 17, 'CPU SSE did not use all rows.')
        require(all(torch.equal(a, b) for a, b in zip(left.state_dict().values(), right.state_dict().values())),
                'Inserted observations changed a native BF16 update.')
        require(all(torch.equal(a.grad, b.grad) for a, b in zip(left.parameters(), right.parameters())),
                'Inserted observations changed a native BF16 gradient.')
    return {'cpu_two_native_bf16_updates_with_without_observation_exact': True,
            'rng_state_versions_modes_and_gradients_unchanged': True,
            'scope': 'Small actual native CPU module; no GPU or true-matrix throughput claim.'}
