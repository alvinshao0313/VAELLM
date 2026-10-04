import pytest
import torch

from experiments.qwen3_w25.bit_gradient_accumulation_diagnostic import (
    direction_statistics, weighted_score_gradients,
)


def test_token_weighted_batch_gradients_match_concatenated_objective():
    parameter = torch.tensor([0.2, -0.7], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[2.0, -1.0], [3.0, 0.5], [-1.0, 4.0], [0.2, 3.0]], dtype=torch.float64)
    y = torch.tensor([1.0, -2.0, 3.0, 0.7], dtype=torch.float64)
    gradients = []
    for indices in ([0], [1, 2, 3]):
        loss = (x[indices] @ parameter - y[indices]).square().mean()
        gradients.append(torch.autograd.grad(loss, parameter)[0])
    gradient, = torch.autograd.grad((x @ parameter - y).square().mean(), parameter)
    weighted = weighted_score_gradients(torch.stack(gradients), [1, 3], 2)
    torch.testing.assert_close(weighted, gradient, atol=1e-14, rtol=1e-14)
    assert not torch.allclose(torch.stack(gradients).mean(0), gradient)
    torch.testing.assert_close(weighted_score_gradients(torch.stack(gradients), [1, 3], 1), gradients[0])


def test_benefit_precision_recall_and_resolution_are_not_overall_accuracy():
    prediction = [-1.0, -1.0, 1.0, 0.0, -1.0]
    actual = [-2.0, 2.0, -1.0, 3.0, 0.0]
    all_nonzero = direction_statistics(prediction, actual, resolution=0.0)
    assert all_nonzero["beneficial_count"] == all_nonzero["harmful_count"] == 2
    assert all_nonzero["decrease_precision"] == all_nonzero["decrease_recall"] == 0.5
    assert all_nonzero["same_sign_fraction"] == 0.25
    resolved = direction_statistics(prediction, actual, resolution=1.5)
    assert resolved["resolved_count"] == 3 and resolved["zero_or_below_resolution_count"] == 2
    assert resolved["decrease_precision"] == 0.5 and resolved["decrease_recall"] == 1.0
    all_harm = direction_statistics([1, 1, 1], [-1, 1, 1], resolution=0.0)
    assert all_harm["same_sign_fraction"] == 2 / 3
    assert all_harm["decrease_precision"] is None and all_harm["decrease_recall"] == 0


def test_bad_token_counts_or_absent_resolved_changes_are_explicit():
    with pytest.raises(ValueError, match="positive token"):
        weighted_score_gradients(torch.ones(2, 3), [2, 0], 2)
    report = direction_statistics([1, -1], [0.0, 1e-8], resolution=1e-7)
    assert report["resolved_count"] == 0 and report["same_sign_fraction"] is None
    assert report["decrease_precision"] is None and report["decrease_recall"] is None
