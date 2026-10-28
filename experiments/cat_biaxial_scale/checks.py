"""Small CPU checks for axis mapping, physical loss, gradients and scalar fusion."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from litebsq.llm_vae import MultiLayerVAE
from litebsq.vae_args import apply_autoencoder_arch_defaults
from train_utils.cat_train_data import apply_stage_norm, compute_stage_norm_stats
from train_utils.cat_train_pipeline import _fuse_norm_into_decoder

from .scaling import BiaxialScale, weighted_reconstruction_loss
from .learned import LearnedBiaxialScale
from .training import recipe


def main():
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(31)
    matrix = torch.randn(64, 96, generator=generator) * torch.linspace(.05, 2., 64)[:, None]
    matrix[0] = 0.; matrix[:, 2] = 0.
    scale = BiaxialScale.from_rotated(matrix)
    assert scale.row.dtype == scale.column.dtype == torch.bfloat16
    assert scale.row[0] == 1 and scale.column[2] == 1
    torch.testing.assert_close(scale.restore(scale.normalize(matrix)), matrix, rtol=3e-7, atol=3e-7)
    for transpose in (False, True):
        expected = scale.row.float()[:, None] * scale.column.float()[None, :]
        if transpose:
            expected = expected.T.contiguous()
        actual = scale.block_factors(torch.arange(matrix.numel() // 32), transpose=transpose, dim=32)
        torch.testing.assert_close(actual[:, 0], expected.reshape(-1, 32), rtol=0, atol=0)
    for bad in (torch.zeros(32, 32), torch.full((32, 32), float('nan'))):
        try:
            BiaxialScale.from_rotated(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid initialization was silently accepted.')
    blocks = matrix.reshape(-1, 1, 32)
    mean, std = compute_stage_norm_stats(blocks)
    x = apply_stage_norm(blocks, mean=mean, scale=std).to(torch.bfloat16)
    y = torch.randn(x.shape, generator=generator, requires_grad=True)
    factors = scale.block_factors(torch.arange(len(x)), transpose=False, dim=32)
    physical_std = scale.restore(matrix).std().reshape(1, 1)
    actual = weighted_reconstruction_loss(y, x, factors, std, physical_std)
    physical_error = (y - x.float()) * std.reshape(1, 1, 1) * factors
    expected = physical_error.square().mean() / physical_std.square().squeeze()
    ga = torch.autograd.grad(actual, y, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, y, retain_graph=True)[0]
    torch.testing.assert_close(actual, expected); torch.testing.assert_close(ga, ge)
    identity = weighted_reconstruction_loss(y, x, torch.ones_like(factors), std, std)
    native = torch.nn.functional.mse_loss(y, x.float())
    torch.testing.assert_close(identity, native, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(identity, y)[0], torch.autograd.grad(native, y)[0])
    learner = LearnedBiaxialScale(scale)
    initial_row, initial_column = learner.effective()
    torch.testing.assert_close(initial_row, scale.row.float(), rtol=0, atol=0)
    torch.testing.assert_close(initial_column, scale.column.float(), rtol=0, atol=0)
    live_factors = learner.block_factors(torch.arange(len(blocks)), transpose=False)
    live_input = blocks / live_factors
    input_grads = torch.autograd.grad(live_input.square().mean(), tuple(learner.parameters()), retain_graph=True)
    assert all(bool(torch.isfinite(value).all()) and bool(value.abs().sum() > 0) for value in input_grads)
    prior = torch.randn(blocks.shape, generator=generator) * .03
    prediction = torch.randn(blocks.shape, generator=generator)
    physical = live_factors * (prior + mean.reshape(1, 1, 1) + std.reshape(1, 1, 1) * prediction)
    loss_learned = ((physical - blocks) / physical_std.reshape(1, 1, 1)).square().mean()
    row, column = learner.effective()
    reference = ((row[:, None] * column[None, :] * (prior + mean.reshape(1, 1, 1)
        + std.reshape(1, 1, 1) * prediction).reshape_as(matrix) - matrix) / physical_std.squeeze()).square().mean()
    learned_gradients = torch.autograd.grad(loss_learned, tuple(learner.parameters()), retain_graph=True)
    reference_gradients = torch.autograd.grad(reference, tuple(learner.parameters()))
    torch.testing.assert_close(loss_learned, reference)
    for actual_gradient, expected_gradient in zip(learned_gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    with torch.no_grad():
        learner.row_delta.add_(.1)
        learner.column_delta.sub_(.3)
    shifted_row, shifted_column = learner.effective()
    torch.testing.assert_close(shifted_row, initial_row, rtol=0, atol=0)
    torch.testing.assert_close(shifted_column, initial_column, rtol=0, atol=0)
    _, config, _, args = recipe('/unused', 'gate_proj', 2, False)
    for key in ('codebook_bits', 'codebook_dim', 'base_ch', 'num_res_blocks', 'norm_type',
                'activation_type', 'decoder_type', 'decoder_base_ch', 'decoder_num_res_blocks', 'recon_loss_type'):
        setattr(args, key, getattr(config, key))
    args.parallel_layers = 3; args.vae_weight_dtype = 'fp32'; args.vae_autocast_dtype = 'fp32'
    apply_autoencoder_arch_defaults(args)
    torch.manual_seed(31)
    vae = MultiLayerVAE(args).train()
    values = torch.randn(128, 3, 32, generator=generator)
    decoded, losses = vae(values, is_train=True)
    unit = weighted_reconstruction_loss(decoded, values, torch.ones_like(values), torch.ones(3, 1), torch.ones(3, 1))
    loss = 3 * unit * args.l1_weight + losses['train/commitment_loss']
    torch.testing.assert_close(loss, losses['loss'])
    native_grads = torch.autograd.grad(losses['loss'], tuple(vae.parameters()), retain_graph=True)
    unit_grads = torch.autograd.grad(loss, tuple(vae.parameters()))
    for a, b in zip(native_grads, unit_grads):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)
    decoder = deepcopy(vae.model.decoder.get_sub_decoder(0)).eval()
    code = torch.randn(97, 1, 32, generator=generator)
    before = decoder(code)
    _fuse_norm_into_decoder(decoder, mean=.013, std=.074)
    torch.testing.assert_close(decoder(code), before * .074 + .013, rtol=3e-5, atol=3e-7)
    result = dict(status='passed_cpu_algebra', bf16_scale_storage=True,
        zero_axes_and_invalid_matrix=True, original_and_transposed_block_mapping=True,
        weighted_native_bf16_target_physical_loss_and_gradient=True,
        identity_native_m3_bsq_loss_and_parameter_gradients=True, scalar_decoder_fusion=True,
        learned_dynamic_input_gradients_and_fp32_physical_loss=True, learned_centered_axis_gauge=True,
        cuda_initialized=torch.cuda.is_initialized(), scope='Small CPU algebra/native M3 module checks; real BF16 GPU and packed endpoint proof remain separate.')
    assert not result['cuda_initialized']
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
