import math

import pytest
import torch

from experiments.qwen3_w25.block_ldlq import (
    block_ldl,
    build_noncentered_hessian,
    prepare_protected_objective,
    quantize_reverse_blocks,
    two_stage_single_bit_candidates,
)
from litebsq.autoencoder import Decoder
from rotation.weight_preprocess import TwoSidedHadamard, rotation_spec


def _quadratic(error, hessian):
    return ((error @ hessian) * error).sum()


def _linear_stages(bits, width, seed=31, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    decoders = []
    for stage in range(2):
        decoder = Decoder(in_dim=bits, out_dim=width, hidden_dim=8, num_res_blocks=0,
                          norm_type="layer", decoder_type="linear").to(dtype).eval()
        with torch.no_grad():
            decoder.linear.linear.weight.copy_(torch.randn(width, bits, generator=generator).to(dtype))
            decoder.linear.linear.bias.copy_(torch.randn(width, generator=generator).to(dtype) * 0.1)
        decoder.requires_grad_(False)
        decoders.append(decoder)

    def decode(stage, codes):
        return decoders[stage](codes.to(dtype))

    return decoders, decode


def test_noncentered_hessian_keeps_mean_and_every_token_without_mutation():
    generator = torch.Generator().manual_seed(39)
    inputs = (torch.randn(19, 7, generator=generator) + 3).to(torch.bfloat16)
    before = inputs.clone()
    hessian = build_noncentered_hessian(inputs, chunk_size=4)
    expected = inputs.double().T @ inputs.double() / len(inputs)
    torch.testing.assert_close(hessian, expected, rtol=2e-15, atol=2e-15)
    centered = inputs.double() - inputs.double().mean(0)
    assert not torch.allclose(hessian, centered.T @ centered / len(inputs))
    assert torch.equal(inputs, before) and hessian.dtype == torch.float64


def test_protected_cross_term_and_actual_block_rotation_complete_the_square():
    generator = torch.Generator().manual_seed(41)
    inputs = torch.randn(29, 12, generator=generator, dtype=torch.float64) + 1
    hessian = build_noncentered_hessian(inputs, chunk_size=8)
    weight = torch.randn(4, 12, generator=generator, dtype=torch.float64)
    protected = torch.tensor([9, 0, 7, 3])
    compressed = torch.tensor([1, 2, 4, 5, 6, 8, 10, 11])
    q_p = weight[:, protected] + 0.3 * torch.randn(4, 4, generator=generator, dtype=torch.float64)
    original = [value.clone() for value in (weight, hessian, q_p)]
    objective = prepare_protected_objective(weight, q_p, hessian, compressed, protected)
    assert objective.damping == pytest.approx(0.01 * float(hessian[compressed][:, compressed].diagonal().mean()))
    assert not torch.equal(objective.target, weight[:, compressed])
    rotation = TwoSidedHadamard(4, 8, rotation_spec("two_sided", 2, 31, "ldl_test"))
    had = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64) / math.sqrt(2)
    u = torch.block_diag(had, had) * rotation.output_signs.double()[None, :]
    v = torch.block_diag(had, had, had, had) * rotation.input_signs.double()[None, :]
    target_rotated = rotation(objective.target)
    hessian_rotated = v @ objective.hessian @ v.T
    torch.testing.assert_close(target_rotated, u @ objective.target @ v.T, rtol=1e-14, atol=1e-14)
    constants = []
    for _ in range(3):
        q_c = torch.randn(4, 8, generator=generator, dtype=torch.float64)
        q = torch.empty_like(weight)
        q[:, compressed], q[:, protected] = q_c, q_p
        error = weight - q
        direct = _quadratic(error, hessian) + objective.damping * error[:, compressed].square().sum()
        proxy = _quadratic(objective.target - q_c, objective.hessian)
        rotated_proxy = _quadratic(target_rotated - rotation(q_c), hessian_rotated)
        torch.testing.assert_close(proxy, rotated_proxy, rtol=3e-14, atol=3e-13)
        constants.append(direct - proxy)
    torch.testing.assert_close(torch.stack(constants), constants[0].expand(3), rtol=1e-12, atol=1e-12)
    for value, before in zip((weight, hessian, q_p), original):
        assert torch.equal(value, before)


def test_block_ldl_reconstructs_dense_hessian_with_unit_diagonal_blocks():
    generator = torch.Generator().manual_seed(51)
    x = torch.randn(17, 12, generator=generator, dtype=torch.float64)
    hessian = x.T @ x + torch.eye(12, dtype=torch.float64) * 0.1
    before = hessian.clone()
    factors = block_ldl(hessian, block_size=3)
    full_d = torch.block_diag(*factors.diagonal.unbind())
    torch.testing.assert_close(factors.lower @ full_d @ factors.lower.T, hessian, rtol=1e-13, atol=1e-13)
    for start in range(0, 12, 3):
        torch.testing.assert_close(factors.lower[start:start + 3, start:start + 3],
                                   torch.eye(3, dtype=torch.float64), rtol=1e-14, atol=1e-14)
        assert torch.count_nonzero(factors.lower[:start, start:start + 3]) == 0
    assert torch.equal(hessian, before)


def test_joint_neighborhood_has_only_single_flips_and_native_bf16_sum():
    generator = torch.Generator().manual_seed(63)
    codes = torch.randint(0, 2, (2, 5, 24), generator=generator).bool()
    before = codes.clone()
    decoders, decode = _linear_stages(24, 16, dtype=torch.bfloat16)
    state = [{name: value.clone() for name, value in decoder.state_dict().items()} for decoder in decoders]
    candidates = two_stage_single_bit_candidates(codes, decode)
    assert candidates.shape == (5, 49, 16) and candidates.dtype == torch.bfloat16
    expected = []
    for candidate in range(49):
        changed = codes.clone()
        if candidate:
            stage, bit = divmod(candidate - 1, 24)
            changed[stage, :, bit] = ~changed[stage, :, bit]
        assert int((codes != changed).sum()) == (0 if candidate == 0 else 5)
        expected.append(decode(0, changed[0]) + decode(1, changed[1]))
    torch.testing.assert_close(candidates, torch.stack(expected, dim=1), rtol=0, atol=0)
    assert torch.equal(codes, before)
    for decoder, before_state in zip(decoders, state):
        for name, value in decoder.state_dict().items():
            assert torch.equal(value, before_state[name])


@pytest.mark.parametrize("diagonal_blocks", [False, True])
def test_identity_and_block_diagonal_make_feedback_on_off_identical(diagonal_blocks):
    generator = torch.Generator().manual_seed(71)
    blocks, rows, width, bits = 4, 3, 2, 3
    hessian = torch.eye(blocks * width, dtype=torch.float64)
    if diagonal_blocks:
        block = torch.tensor([[2., 0.7], [0.7, 0.5]], dtype=torch.float64)
        hessian = torch.block_diag(*[block * (i + 1) for i in range(blocks)])
    factors = block_ldl(hessian, width)
    weight = torch.randn(rows, blocks * width, generator=generator, dtype=torch.float64)
    codes = torch.randint(0, 2, (2, rows, blocks, bits), generator=generator).bool()
    _, decode = _linear_stages(bits, width)
    off = quantize_reverse_blocks(weight, factors, codes, decode, feedback=False)
    on = quantize_reverse_blocks(weight, factors, codes, decode, feedback=True)
    assert torch.equal(on.codes, off.codes)
    assert torch.equal(on.reconstructed_weight, off.reconstructed_weight)
    assert [report["block"] for report in on.statistics["block_reports"]] == [3, 2, 1, 0]


def test_reverse_feedback_matches_explicit_original_residual_formula_and_changes_only_one_bit():
    generator = torch.Generator().manual_seed(79)
    rows, blocks, width, bits = 4, 4, 2, 3
    x = torch.randn(11, blocks * width, generator=generator, dtype=torch.float64)
    hessian = x.T @ x + torch.eye(blocks * width, dtype=torch.float64) * 0.01
    factors = block_ldl(hessian, width)
    weight = torch.randn(rows, blocks * width, generator=generator, dtype=torch.float64)
    codes = torch.randint(0, 2, (2, rows, blocks, bits), generator=generator).bool()
    before = codes.clone()
    _, decode = _linear_stages(bits, width, seed=19)
    result = quantize_reverse_blocks(weight, factors, codes, decode)
    expected = torch.empty_like(weight)
    for block in range(blocks - 1, -1, -1):
        start, end = block * width, (block + 1) * width
        target = weight[:, start:end].clone()
        for later in range(block + 1, blocks):
            later_slice = slice(later * width, (later + 1) * width)
            target += (weight[:, later_slice] - expected[:, later_slice]) @ factors.lower[later_slice, start:end]
        # Deliberately enumerate all joint pairs independently of the batched oracle.
        candidates = []
        for candidate in range(1 + 2 * bits):
            changed = codes[:, :, block, :].clone()
            if candidate:
                stage, bit = divmod(candidate - 1, bits)
                changed[stage, :, bit] = ~changed[stage, :, bit]
            candidates.append(decode(0, changed[0]) + decode(1, changed[1]))
        values = torch.stack(candidates, dim=1)
        errors = target[:, None, :] - values
        scores = torch.einsum("rcd,de,rce->rc", errors, factors.diagonal[block], errors)
        expected[:, start:end] = values[torch.arange(rows), scores.argmin(1)]
    torch.testing.assert_close(result.reconstructed_weight, expected, rtol=0, atol=0)
    assert torch.equal(codes, before)
    assert bool(((result.codes != before).sum((0, 3)) <= 1).all())
    decoded_result = torch.stack([
        decode(0, result.codes[0, :, block]) + decode(1, result.codes[1, :, block])
        for block in range(blocks)
    ], dim=1).reshape(rows, blocks * width)
    torch.testing.assert_close(result.reconstructed_weight, decoded_result, rtol=0, atol=0)
    off = quantize_reverse_blocks(weight, factors, codes, decode, feedback=False)
    assert not torch.equal(result.codes, off.codes)


def test_current_wins_ties_and_local_improvement_does_not_promise_full_objective_improvement():
    def binary(stage, values):
        return values.double()

    hessian = torch.tensor([[1., 0.9], [0.9, 1.]], dtype=torch.float64)
    weight = torch.tensor([[0.4, 0.49]], dtype=torch.float64)
    codes = torch.tensor([[[[False], [True]]], [[[False], [False]]]])
    result = quantize_reverse_blocks(weight, block_ldl(hessian, 1), codes, binary)
    before = codes.double().sum(0).reshape(1, 2)
    assert _quadratic(weight - result.reconstructed_weight, hessian) > _quadratic(weight - before, hessian)
    assert all(report["selected_local_score_sum"] <= report["current_local_score_sum"]
               for report in result.statistics["block_reports"])
    tie_codes = torch.zeros(2, 1, 1, 1, dtype=torch.bool)
    tie = quantize_reverse_blocks(torch.tensor([[0.5]], dtype=torch.float64),
                                  block_ldl(torch.ones(1, 1, dtype=torch.float64), 1),
                                  tie_codes, binary)
    assert torch.equal(tie.codes, tie_codes)
    assert tie.statistics["stage_flips"] == [0, 0]


def test_invalid_geometry_precision_partition_and_non_spd_are_rejected():
    with pytest.raises(ValueError, match="FP64"):
        block_ldl(torch.eye(4), 2)
    with pytest.raises(ValueError, match="dividing"):
        block_ldl(torch.eye(4, dtype=torch.float64), 3)
    with pytest.raises(ValueError, match="symmetric"):
        block_ldl(torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64), 1)
    with pytest.raises(torch.linalg.LinAlgError):
        block_ldl(torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64), 1)
    with pytest.raises(ValueError, match="partition"):
        prepare_protected_objective(torch.zeros(2, 4, dtype=torch.float64),
                                    torch.zeros(2, 2, dtype=torch.float64),
                                    torch.eye(4, dtype=torch.float64),
                                    torch.tensor([0, 1]), torch.tensor([1, 3]))
