import math

from sparse_bit_tuning.sampler import AffineSamplerState, bank_coverage_seed


def _state(n=101, active=7, seed=123, key="model.layers.0.mlp.down_proj|stage=0|part=0", coverage=0, cursor=0):
    return AffineSamplerState.create(
        canonical_key=key,
        training_seed=seed,
        n_bits=n,
        n_active=active,
        coverage_id=coverage,
        cursor=cursor,
    )


def test_deterministic_seed_and_key():
    a = _state()
    b = _state()
    assert a == b
    assert a.active_indices() == b.active_indices()
    assert bank_coverage_seed(1, "x", 0) == bank_coverage_seed(1, "x", 0)
    assert bank_coverage_seed(1, "x", 0) != bank_coverage_seed(1, "y", 0)
    assert _state(coverage=0).active_indices() != _state(coverage=1).active_indices()


def test_primary_permutation_covers_without_duplicates():
    state = _state(n=97, active=11)
    seq = [(state.stride * k + state.offset) % state.n_bits for k in range(state.n_bits)]
    assert len(set(seq)) == state.n_bits
    assert math.gcd(state.stride, state.n_bits) == 1


def test_tail_fill_fixed_size_unique_and_contains_tail():
    state = _state(n=23, active=7, cursor=21)
    meta = state.subset_meta()
    assert meta.tail
    active = state.active_indices()
    assert len(active) == 7
    assert len(set(active)) == 7
    tail = {
        (state.stride * k + state.offset) % state.n_bits
        for k in range(state.cursor, state.n_bits)
    }
    assert tail.issubset(set(active))
    prefix = {
        (state.stride * k + state.offset) % state.n_bits
        for k in range(state.cursor)
    }
    assert set(active) - tail <= prefix
    next_state = state.advance()
    assert next_state.coverage_id == state.coverage_id + 1
    assert next_state.cursor == 0


def test_exact_divisible_boundary_rolls_coverage():
    state = _state(n=20, active=5, cursor=15)
    assert not state.subset_meta().tail
    next_state = state.advance()
    assert next_state.coverage_id == state.coverage_id + 1
    assert next_state.cursor == 0


def test_ratio_one_shape_and_n_one():
    state = _state(n=13, active=13)
    assert len(state.active_indices()) == 13
    assert len(set(state.active_indices())) == 13
    next_state = state.advance()
    assert next_state.coverage_id == 1 and next_state.cursor == 0

    single = _state(n=1, active=1)
    assert single.stride == 0
    assert single.offset == 0
    assert single.active_indices() == [0]
    assert single.advance().coverage_id == 1


def test_metadata_round_trip():
    state = _state(n=29, active=6, coverage=3, cursor=12)
    restored = AffineSamplerState.from_metadata(state.to_metadata())
    assert restored == state
    assert restored.active_indices() == state.active_indices()
