import pytest
import torch

from litebsq.packed_bit_linear import packed_u8_linear
from sparse_bit_tuning.packed_ops import (
    PackedBitRuntimeMeta,
    bit_aware_packed_u8_linear,
    initialize_scores_from_packed,
)
from sparse_bit_tuning.reference import pack_bool_bits
from sparse_bit_tuning.sampler import AffineSamplerState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _states(B, IN):
    return (
        AffineSamplerState.create(
            canonical_key="m|stage=0|part=0", training_seed=3, n_bits=B * IN, n_active=11
        ),
        AffineSamplerState.create(
            canonical_key="m|stage=1|part=0", training_seed=3, n_bits=B * IN, n_active=9
        ),
    )


def _expected_score_grad(upstream, weight, states, model_indices):
    rows = []
    up = upstream.float()
    w = weight.float()
    for state, model_idx in zip(states, model_indices):
        vals = []
        for logical in state.active_indices():
            block = logical // (state.n_bits // up.shape[0])
            latent = logical % (state.n_bits // up.shape[0])
            vals.append((up[block, model_idx] * w[model_idx, :, latent]).sum())
        rows.extend(vals)
    return torch.stack(rows).to(torch.float16)


def test_bit_aware_dscore_grouped_matches_dense_formula_frozen_decoder():
    device = torch.device("cuda:0")
    B, M, IN, H = 7, 2, 13, 19
    torch.manual_seed(11)
    bits = torch.randint(0, 2, (B, M, IN), dtype=torch.bool, device=device)
    packed = pack_bool_bits(bits).contiguous()
    weight = torch.randn(M, H, IN, device=device, dtype=torch.bfloat16)
    bias = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    states = _states(B, IN)
    meta = PackedBitRuntimeMeta.build(
        states=states,
        model_indices=(0, 1),
        score_offsets=(0, 11),
        logical_in_dim=IN,
        device=device,
    )
    score = torch.empty((20,), device=device, dtype=torch.float16, requires_grad=True)
    initialize_scores_from_packed(packed, score, meta)
    upstream = torch.randn(B, M, H, device=device, dtype=torch.bfloat16)
    out = bit_aware_packed_u8_linear(
        packed, weight, bias, score, meta, logical_in_dim=IN, activation_dtype=torch.bfloat16
    )
    out.backward(upstream)
    expected = _expected_score_grad(upstream, weight, states, (0, 1))
    assert score.grad is not None
    assert torch.equal(score.grad, expected)
    assert weight.grad is None
    assert bias.grad is None


def test_bit_aware_forward_and_decoder_grads_match_existing_packed_path():
    device = torch.device("cuda:0")
    B, M, IN, H = 5, 2, 16, 17
    torch.manual_seed(23)
    bits = torch.randint(0, 2, (B, M, IN), dtype=torch.bool, device=device)
    packed = pack_bool_bits(bits).contiguous()
    weight_a = torch.randn(M, H, IN, device=device, dtype=torch.bfloat16, requires_grad=True)
    bias_a = torch.randn(M, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight_b = weight_a.detach().clone().requires_grad_(True)
    bias_b = bias_a.detach().clone().requires_grad_(True)
    states = (
        AffineSamplerState.create(
            canonical_key="m|stage=0|part=0", training_seed=9, n_bits=B * IN, n_active=10
        ),
        AffineSamplerState.create(
            canonical_key="m|stage=1|part=0", training_seed=9, n_bits=B * IN, n_active=10
        ),
    )
    meta = PackedBitRuntimeMeta.build(
        states=states, model_indices=(0, 1), score_offsets=(0, 10), logical_in_dim=IN, device=device
    )
    score = torch.empty((20,), device=device, dtype=torch.float16, requires_grad=True)
    initialize_scores_from_packed(packed, score, meta)
    upstream = torch.randn(B, M, H, device=device, dtype=torch.bfloat16)

    out_a = bit_aware_packed_u8_linear(
        packed, weight_a, bias_a, score, meta, logical_in_dim=IN, activation_dtype=torch.bfloat16
    )
    out_b = packed_u8_linear(
        packed, weight_b, bias_b, logical_in_dim=IN, activation_dtype=torch.bfloat16
    )
    assert torch.equal(out_a, out_b)
    out_a.backward(upstream)
    out_b.backward(upstream)
    assert torch.equal(weight_a.grad, weight_b.grad)
    assert torch.equal(bias_a.grad, bias_b.grad)
    expected = _expected_score_grad(upstream, weight_a.detach(), states, (0, 1))
    assert torch.equal(score.grad, expected)
