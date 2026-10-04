import pytest
import torch

from experiments.qwen3_w25.kl_gradient_diagnostic import batch_manifest
from experiments.qwen3_w25.kl_round_diagnostic import logical_ordinals, timing_summary, validate_batch_archive
from sparse_bit_tuning.sampler import AffineSamplerState


def test_full_coverage_observer_follows_fixed_bits_across_affine_rounds():
    state = AffineSamplerState.create(canonical_key="stage0", training_seed=0, n_bits=101, n_active=101)
    fixed = [state.logical_index(i) for i in (0, 3, 17, 99)]
    assert logical_ordinals(state, fixed) == [0, 3, 17, 99]
    next_state = state.advance()
    assert next_state.coverage_id == 1 and next_state.n_active == 101
    assert [next_state.logical_index(i) for i in logical_ordinals(next_state, fixed)] == fixed
    with pytest.raises(ValueError):
        logical_ordinals(AffineSamplerState.create(canonical_key="partial", training_seed=0, n_bits=101, n_active=5), fixed)


def test_archive_rejects_heldout_leakage_without_resampling():
    batches = []
    for index in range(264):
        ids = torch.arange(index*12, index*12+12, dtype=torch.int64).reshape(4, 3)
        batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()})
    archive = {"format": "qwen3_formal_kl_batches_v1", "batches": batches,
               "manifests": [batch_manifest(b, i, sample_offset=i*4) for i,b in enumerate(batches)]}
    result = validate_batch_archive(archive)
    assert result['training_unique_hashes'] == 1024 and result['heldout_unique_hashes'] == 32
    batches[256] = {key: value.clone() for key,value in batches[0].items()}
    archive['manifests'][256] = batch_manifest(batches[256], 256, sample_offset=1024)
    with pytest.raises(ValueError, match="overlap"):
        validate_batch_archive(archive)


def test_timing_keeps_cold_start_separate_and_reports_two_arm_budget():
    result = timing_summary([{'training_step_seconds': x} for x in [80, 14, 15, 16]])
    assert result['first_step_seconds'] == 80 and result['steady_median_seconds'] == 15
    assert result['one_256_step_arm_median_minutes'] == 64
    assert result['two_256_step_arms_median_minutes'] == 128
