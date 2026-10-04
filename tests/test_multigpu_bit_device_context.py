"""Triton launches must follow tensor placement and restore the caller's device."""
from copy import deepcopy

import pytest
import torch

from litebsq.autoencoder import Decoder
from litebsq.fused_multistage_decoder import (
    fused_decode_batched_same_shape_decoders,
    fused_decode_packed_symmetric_decoder,
)
from litebsq.packed_bit_linear import packed_u8_linear
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import SparseBitTuningConfig
from sparse_bit_tuning.coordinates import calibrate_bank_scale
from sparse_bit_tuning.module import BankSpec, SparseBitTuningModule
from sparse_bit_tuning.optimizer import BitOptimizerManager
from sparse_bit_tuning.packed_ops import (
    PackedBitRuntimeMeta,
    bit_aware_packed_u8_linear,
    initialize_scores_from_packed,
    project_scores_to_packed,
)
from sparse_bit_tuning.reference import (
    apply_bank_scores_reference,
    dense_active_score_grad_reference,
    pack_bool_bits,
    read_bank_hard_bits,
    scores_from_hard_bits,
)
from sparse_bit_tuning.sampler import AffineSamplerState


pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two real CUDA devices required")


@pytest.fixture
def other_device():
    with torch.cuda.device(0):
        yield torch.device("cuda:1")
        assert torch.cuda.current_device() == 0


def test_calibration_preserves_rng_state_and_current_device(other_device):
    torch.manual_seed(47)
    decoder = Decoder(
        in_dim=32, out_dim=32, hidden_dim=128, num_res_blocks=0,
        norm_type="layer", decoder_type="symmetric", use_checkpoint=False,
    )
    layer = VAELinear(
        in_features=32, out_features=64, bias=None, original_weight=None,
        vq_weight=torch.randint(0, 2, (64, 1, 32), dtype=torch.bool),
        decoder=decoder, codebook_dim=32, transpose=False,
    ).to(device=other_device, dtype=torch.float32)
    spec = BankSpec("layer|stage=0|part=0", "layer", 0, 0, (64, 1, 32), 2048, 2048, other_device)
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    gpu_rng = [torch.cuda.get_rng_state(device) for device in (0, 1)]
    actual = calibrate_bank_scale(layer, spec, compute_dtype=torch.bfloat16)
    assert torch.cuda.current_device() == 0
    with torch.cuda.device(other_device):
        expected = calibrate_bank_scale(layer, spec, compute_dtype=torch.bfloat16)
    assert actual == expected and actual > 0
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    for device, state in enumerate(gpu_rng):
        assert torch.equal(torch.cuda.get_rng_state(device), state)
    for name, value in layer.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_packed_and_sparse_gradients_and_full_code_commit(other_device):
    torch.manual_seed(53)
    rows, latent, hidden = 5, 17, 32
    bits = torch.randint(0, 2, (rows, 1, latent), device=other_device, dtype=torch.bool)
    packed = pack_bool_bits(bits)
    # Fifteen packed bytes exercise both full-word SET and its tail kernel.
    assert packed.numel() % 4 == 3
    state = AffineSamplerState.create(
        canonical_key="layer|stage=0|part=0", training_seed=17,
        n_bits=rows * latent, n_active=rows * latent,
    )
    scale = 0.004
    meta = PackedBitRuntimeMeta.build(
        states=(state,), model_indices=(0,), score_offsets=(0,),
        logical_in_dim=latent, device=other_device, proxy_scales=(scale,),
    )
    score = torch.empty(rows * latent, device=other_device, dtype=torch.float32, requires_grad=True)
    initialize_scores_from_packed(packed, score, meta)
    assert torch.cuda.current_device() == 0
    expected_score = scores_from_hard_bits(
        read_bank_hard_bits(packed, state=state, model_idx=0, logical_in_dim=latent), proxy_scale=scale,
    )
    assert torch.equal(score, expected_score)
    weight = torch.randn(1, hidden, latent, device=other_device, requires_grad=True)
    bias = torch.randn(1, hidden, device=other_device, requires_grad=True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_bias = bias.detach().clone().requires_grad_(True)
    upstream = torch.randn(rows, 1, hidden, device=other_device, dtype=torch.bfloat16)
    actual = bit_aware_packed_u8_linear(
        packed, weight, bias, score, meta, logical_in_dim=latent, activation_dtype=torch.bfloat16,
    )
    assert torch.cuda.current_device() == 0
    reference = packed_u8_linear(
        packed, reference_weight, reference_bias, logical_in_dim=latent, activation_dtype=torch.bfloat16,
    )
    assert torch.cuda.current_device() == 0
    actual.backward(upstream)
    reference.backward(upstream)
    assert torch.cuda.current_device() == 0
    assert torch.equal(actual, reference)
    assert torch.equal(weight.grad, reference_weight.grad)
    assert torch.equal(bias.grad, reference_bias.grad)
    expected_grad = dense_active_score_grad_reference(
        upstream, weight.detach().to(torch.bfloat16), states=(state,), model_indices=(0,),
        proxy_scales=(scale,),
    )[0]
    torch.testing.assert_close(score.grad, expected_grad, rtol=2e-6, atol=2e-5)
    with torch.no_grad():
        score.neg_()
    expected_packed, expected_flips = apply_bank_scores_reference(
        packed, state=state, scores=score, model_idx=0, logical_in_dim=latent,
    )
    flipped = project_scores_to_packed(packed, score, meta)
    assert torch.cuda.current_device() == 0
    assert torch.equal(packed, expected_packed)
    assert flipped.item() == expected_flips == rows * latent


@pytest.mark.parametrize("optimizer", ["adam", "rms_sgd"])
def test_bit_optimizer_uses_score_device_and_preserves_update(other_device, optimizer):
    spec = BankSpec("layer|stage=0|part=0", "layer", 0, 0, (1, 1, 8), 8, 8, other_device)
    module = SparseBitTuningModule([spec], proxy_coordinates="decoder_sensitivity")
    module.coordinate_scales[spec.canonical_key] = 0.004
    score = module.score_chunks[0]
    initial = torch.linspace(-0.001, 0.001, 8, device=other_device)
    with torch.no_grad():
        score.copy_(initial)
    manager = BitOptimizerManager(module, SparseBitTuningConfig(
        enabled=True, active_ratio=1.0, optimizer=optimizer,
        bit_lr=4e-5, proxy_coordinates="decoder_sensitivity",
    ))
    manager.refresh_coordinate_scales()
    expected = initial.clone()
    moment = torch.zeros_like(initial)
    second_moment = torch.zeros_like(initial)
    for step in (1, 2):
        grad = torch.tensor([4, -3, 2, -1, 1, -2, 3, -4], device=other_device, dtype=torch.float32)
        grad = grad * (1 if step == 1 else -0.25)
        score.grad = grad.clone()
        old = expected.clone()
        if optimizer == "adam":
            moment = 0.9 * moment + 0.1 * grad
            second_moment = 0.999 * second_moment + 0.001 * grad.square()
            update = (moment / (1 - 0.9 ** step)) / ((second_moment / (1 - 0.999 ** step)).sqrt() + 1e-8)
        else:
            update = grad / (grad.square().mean() + 1e-8).sqrt()
        expected = (expected - 4e-5 * update).clamp(-0.002, 0.002)
        counters = manager.step_scores(optimizer_step_in_round=step)
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(score, expected, rtol=2e-6, atol=2e-10)
        assert sum(counter.item() for counter in counters) == ((old >= 0) != (expected >= 0)).sum().item()
    if optimizer == "adam":
        actual_moment, actual_second = tuple(manager.state_tensors())
        torch.testing.assert_close(actual_moment, moment, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(actual_second, second_moment, rtol=2e-6, atol=1e-7)


def test_fused_deployment_uses_tensor_device_for_single_and_grouped_calls(other_device):
    torch.manual_seed(61)
    decoder = Decoder(
        in_dim=32, out_dim=32, hidden_dim=128, num_res_blocks=0,
        norm_type="layer", decoder_type="symmetric", use_checkpoint=False, num_models=2,
    ).to(device=other_device, dtype=torch.bfloat16).eval()
    bits = torch.randint(0, 2, (64, 2, 32), device=other_device, dtype=torch.bool).to(torch.bfloat16)
    other_decoder = deepcopy(decoder)
    with torch.no_grad():
        with torch.cuda.device(other_device):
            expected = fused_decode_packed_symmetric_decoder(decoder, bits)
        actual = fused_decode_packed_symmetric_decoder(decoder, bits)
        assert torch.cuda.current_device() == 0
        grouped = fused_decode_batched_same_shape_decoders([decoder, other_decoder], [bits, bits])
        assert torch.cuda.current_device() == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(grouped) == 2
    for output in grouped:
        # Grouped and serial kernels use the established BF16 parity tolerance.
        epsilon = torch.finfo(torch.bfloat16).eps
        torch.testing.assert_close(output, expected, rtol=2 * epsilon, atol=2 * epsilon)
