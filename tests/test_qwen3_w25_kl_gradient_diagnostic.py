import torch
from experiments.qwen3_w25.kl_gradient_diagnostic import batch_manifest, gradient_statistics, score_epsilon_statistics


def test_batch_manifest_matches_causal_weighted_loss_denominator():
    labels = torch.tensor([[-100, -100, 4, 5, -100], [-100, 3, 4, -100, -100]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    batch = dict(input_ids=torch.arange(10).reshape(2, 5), labels=labels, attention_mask=mask)
    record = batch_manifest(batch, 0, sample_offset=0)
    assert record['causal_response_tokens'] == 4 and record['causal_prompt_tokens'] == 1
    assert abs(record['loss_denominator_fp32'] - 4.3) < 1e-6
    assert [x['iterator_sample_ordinal'] for x in record['samples']] == [0, 1]
    altered = {k: v.clone() for k, v in batch.items()}
    altered['input_ids'][0, 4] = 999
    assert batch_manifest(altered, 0, sample_offset=0)['samples'] == record['samples']


def test_equal_batch_and_token_weighted_gradient_aggregation_are_distinct():
    g = torch.tensor([[2., 0.], [-1., 0.]])
    equal = gradient_statistics(g, [1, 1])
    weighted = gradient_statistics(g, [1, 4])
    assert equal['coordinate_mean_over_rms'][0] > 0
    assert weighted['coordinate_mean_over_rms'][0] < 0
    assert equal['cosine'][0][1] == -1
    assert equal['cancellation_ratio'] == 1/3
    assert equal['nonzero_counts'] == [2, 0]
    assert equal['cosine_valid'][0][0]


def test_same_direction_zero_and_repeated_gradients():
    stats = gradient_statistics(torch.tensor([[1., -2.], [1., -2.]]), [1, 2])
    assert stats['cancellation_ratio'] == 1 and stats['zero_noise']
    assert stats['unweighted_nonzero_sign_majority_fraction'] == [1, 1]
    zero = gradient_statistics(torch.zeros(2, 3), [1, 1])
    assert zero['cancellation_ratio'] is None and zero['vector_snr'] is None
    assert zero['cosine_valid'] == [[False, False], [False, False]]


def test_fresh_adam_epsilon_reference_keeps_threshold_and_zero_gradient():
    stats = score_epsilon_statistics(torch.tensor([0., 1e-8, 1e-6], dtype=torch.float64), scale=.006)
    assert stats['initial_threshold_distance'] == .003
    assert stats['fraction_below_eps'] == 1/3 and stats['fraction_equal_eps'] == 1/3
    assert stats['fraction_above_eps'] == 1/3
    assert stats['fresh_first_step_unclamped_magnitude_quantiles'][0] == 0
    assert abs(stats['fresh_first_step_unclamped_magnitude_quantiles'][4] - 2e-5) < 1e-18
