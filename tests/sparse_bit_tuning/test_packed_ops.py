import pytest
import torch

from sparse_bit_tuning.packed_ops import (
    PackedBitRuntimeMeta,
    initialize_scores_from_packed,
    project_scores_to_packed,
)
from sparse_bit_tuning.reference import (
    apply_bank_scores_reference,
    pack_bool_bits,
    read_bank_hard_bits,
    scores_from_hard_bits,
)
from sparse_bit_tuning.sampler import AffineSamplerState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _state(key, *, n_bits, n_active, cursor=0):
    return AffineSamplerState.create(
        canonical_key=key,
        training_seed=17,
        n_bits=n_bits,
        n_active=n_active,
        cursor=cursor,
    )


def test_init_and_set_grouped_matches_reference():
    device = torch.device("cuda:0")
    B, M, IN = 5, 2, 13
    torch.manual_seed(1)
    dense = torch.randint(0, 2, (B, M, IN), dtype=torch.bool, device=device)
    packed = pack_bool_bits(dense).contiguous()
    states = (
        _state("m|stage=0|part=0", n_bits=B * IN, n_active=9),
        _state("m|stage=1|part=0", n_bits=B * IN, n_active=9),
    )
    meta = PackedBitRuntimeMeta.build(
        states=states,
        model_indices=(0, 1),
        score_offsets=(0, 9),
        logical_in_dim=IN,
        device=device,
    )
    scores = torch.empty((18,), dtype=torch.float16, device=device, requires_grad=True)
    initialize_scores_from_packed(packed, scores, meta)
    expected0 = scores_from_hard_bits(read_bank_hard_bits(packed, state=states[0], model_idx=0, logical_in_dim=IN))
    expected1 = scores_from_hard_bits(read_bank_hard_bits(packed, state=states[1], model_idx=1, logical_in_dim=IN))
    assert torch.equal(scores[:9], expected0)
    assert torch.equal(scores[9:], expected1)

    with torch.no_grad():
        scores[1::2].mul_(-1)
    ref, flips0 = apply_bank_scores_reference(
        packed.detach().clone(), state=states[0], scores=scores[:9], model_idx=0, logical_in_dim=IN
    )
    ref, flips1 = apply_bank_scores_reference(
        ref, state=states[1], scores=scores[9:], model_idx=1, logical_in_dim=IN
    )
    actual = packed.detach().clone()
    flip = project_scores_to_packed(actual, scores, meta)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, ref)
    assert int(flip.item()) == flips0 + flips1

    second = project_scores_to_packed(actual, scores, meta)
    torch.cuda.synchronize(device)
    assert int(second.item()) == 0
    assert torch.equal(actual, ref)


def test_tail_allocation_non_multiple_of_four_bytes():
    device = torch.device("cuda:0")
    B, M, IN = 1, 1, 17  # packed numel = 3: entire allocation is tail bytes
    dense = torch.tensor(
        [[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]],
        dtype=torch.bool,
        device=device,
    )
    packed = pack_bool_bits(dense).contiguous()
    state = _state("tail|stage=0|part=0", n_bits=B * IN, n_active=IN)
    meta = PackedBitRuntimeMeta.build(
        states=(state,), model_indices=(0,), score_offsets=(0,), logical_in_dim=IN, device=device
    )
    scores = torch.empty((IN,), dtype=torch.float16, device=device, requires_grad=True)
    initialize_scores_from_packed(packed, scores, meta)
    with torch.no_grad():
        scores.neg_()
    expected, expected_flips = apply_bank_scores_reference(
        packed.detach().clone(), state=state, scores=scores, model_idx=0, logical_in_dim=IN
    )
    actual = packed.detach().clone()
    flip = project_scores_to_packed(actual, scores, meta)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)
    assert int(flip.item()) == expected_flips


@pytest.mark.parametrize("B,M,IN,scales", [(5, 2, 13, (0.002, 0.008)), (1, 1, 17, (0.004,))])
def test_sensitivity_coordinates_init_set_preserve_inactive_bits_and_zero_tie(B, M, IN, scales):
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    original = pack_bool_bits(torch.randint(0, 2, (B, M, IN), device=device, dtype=torch.bool))
    states = tuple(_state(f"m{i}", n_bits=B * IN, n_active=min(9, B * IN)) for i in range(M))
    meta = PackedBitRuntimeMeta.build(
        states=states, model_indices=tuple(range(M)), score_offsets=None,
        logical_in_dim=IN, device=device, proxy_scales=scales,
    )
    scores = torch.empty(meta.total_active, device=device, dtype=torch.float32, requires_grad=True)
    initialize_scores_from_packed(original, scores, meta)
    expected_scores = torch.cat([
        scores_from_hard_bits(
            read_bank_hard_bits(original, state=state, model_idx=i, logical_in_dim=IN),
            proxy_scale=scales[i],
        ) for i, state in enumerate(states)
    ])
    assert torch.equal(scores, expected_scores)
    actual = original.clone()
    assert project_scores_to_packed(actual, scores, meta).item() == 0
    assert torch.equal(actual, original)
    with torch.no_grad():
        scores[::2].neg_()
        scores[1::3] = 0  # Sparse Bit keeps its existing zero-to-one tie rule.
    expected = original.clone()
    expected_flips = 0
    for i, (state, start) in enumerate(zip(states, meta.score_offsets_py)):
        expected, flips = apply_bank_scores_reference(
            expected, state=state, scores=scores[start:start + state.n_active],
            model_idx=i, logical_in_dim=IN,
        )
        expected_flips += flips
    actual_flips = project_scores_to_packed(actual, scores, meta)
    assert torch.equal(actual, expected)
    assert actual_flips.item() == expected_flips
    assert project_scores_to_packed(actual, scores, meta).item() == 0
