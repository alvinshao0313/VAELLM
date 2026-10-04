import torch

from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.bit_direction_diagnostic import (
    _output_measurement, candidate_coordinates, fixed_probe_groups,
    stable_output_delta, ste_flip_prediction,
)
from sparse_bit_tuning.config import active_count
from sparse_bit_tuning.reference import pack_bool_bits, unpack_packed_bits
from sparse_bit_tuning.sampler import AffineSamplerState


def test_sampler_little_endian_mapping_and_restore():
    # The five logical bank sizes in the fixed 552282 source metadata.
    for n_bits in (4128768, 16515072, 24772608, 49545216, 73924608):
        assert active_count(n_bits, 16 / n_bits) == 16
    original = torch.zeros((5, 1, 13), dtype=torch.bool)
    original[1, 0, 12] = True
    packed = pack_bool_bits(original)
    state = AffineSamplerState.create(canonical_key="layer|stage=0|part=0", training_seed=31,
                                     n_bits=65, n_active=16)
    for coordinate in candidate_coordinates(state.active_indices(), original.shape):
        trial = packed.clone()
        row, byte, shift = [coordinate[key] for key in ("block_index", "byte_index", "byte_bit")]
        trial[row, 0, byte] ^= 1 << shift
        dense = unpack_packed_bits(trial, logical_in_dim=13)
        assert int(torch.count_nonzero(dense != original)) == 1
        assert bool(dense[row, 0, coordinate["latent_bit"]]) != bool(original[row, 0, coordinate["latent_bit"]])
        trial[row, 0, byte] ^= 1 << shift
        assert torch.equal(trial, packed)


def test_sensitivity_coordinate_flip_prediction_matches_bit_gradient():
    bits = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    loss = (torch.sin(bits @ torch.tensor([1.3, -0.7], dtype=torch.float64)) - 0.4).square()
    gradient, = torch.autograd.grad(loss, bits)
    scale = 0.007
    for index in range(2):
        predicted = ste_flip_prediction(gradient[index] / scale, scale, int(bits[index]))
        expected = float(gradient[index] * (1 - 2 * bits[index]))
        assert abs(predicted - expected) < 1e-15


def test_shadow_delta_uses_actual_fp32_residual_outputs_and_objective():
    torch.manual_seed(11)
    x = torch.randn(7, 3)
    before_weight = torch.randn(2, 3)
    after_weight = before_weight + torch.tensor([[0.2, -0.3, 0.0], [0.1, 0.0, -0.4]])
    before, before_metric = _output_measurement(before_weight, x, 3)
    after, after_metric = _output_measurement(after_weight, x, 3)
    assert before_metric["loss_fp32"] == float(linear_output_mse(before_weight, x, chunk_size=3, recompute=False))
    assert after_metric["loss_fp32"] == float(linear_output_mse(after_weight, x, chunk_size=3, recompute=False))
    actual = stable_output_delta(before, after)["shadow_delta_mse_fp64"]
    expected = float(after.double().square().mean() - before.double().square().mean())
    assert abs(actual - expected) < 1e-14
    # A representable tiny output change survives shadow reduction even where
    # subtracting rounded scalar FP32 losses loses information.
    large = torch.tensor([[1000.0, 1000.0]], dtype=torch.float32)
    changed = large.clone()
    changed[0, 0] = torch.nextafter(changed[0, 0], torch.tensor(float("inf")))
    assert stable_output_delta(large, changed)["shadow_delta_mse_fp64"] > 0


def test_finite_flip_can_reverse_gradient_prediction_and_groups_are_fixed():
    # For (b-.1)^2 at b=0, the local derivative is negative but flipping to1 hurts.
    bit = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad((bit - 0.1).square(), bit)
    assert ste_flip_prediction(gradient / 0.01, 0.01, 0) < 0
    assert stable_output_delta(torch.tensor([[-0.1]], dtype=torch.float64),
                               torch.tensor([[0.9]], dtype=torch.float64))["shadow_delta_mse_fp64"] > 0
    probes = fixed_probe_groups(16)
    assert len(probes) == 35
    assert probes[-1][1] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len({tuple(bits) for _, bits in probes}) == 35
