import pytest
import torch

from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.full_linear_gram import build_full_linear_gram, full_linear_gram_mse
from experiments.qwen3_w25.gram_validation import measure_error_gradient, tensor_difference


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_complete_error_gradient_measurement_is_read_only_and_normalized(dtype):
    generator = torch.Generator().manual_seed(31)
    inputs = torch.randn(37, 11, generator=generator).to(torch.bfloat16)
    error = torch.randn(7, 11, generator=generator)
    original_error, original_inputs = error.clone(), inputs.clone()
    stream, stream_gradients = measure_error_gradient(
        error, lambda value: linear_output_mse(value, inputs, chunk_size=8, recompute=True))
    gram = build_full_linear_gram(inputs, dtype=dtype)
    actual, actual_gradients = measure_error_gradient(error, lambda value: full_linear_gram_mse(value, gram))
    difference = tensor_difference(actual_gradients[0], stream_gradients[0])
    expected = (2 * error.double() @ (inputs.double().T @ inputs.double())) / (len(inputs) * len(error))
    torch.testing.assert_close(actual_gradients[0].double(), expected, atol=3e-7, rtol=3e-6)
    assert difference["numel"] == error.numel() and difference["relative_l2"] < 1e-6
    assert actual["loss"] == pytest.approx(stream["loss"], rel=1e-6)
    for report in (stream, actual):
        assert report["warmup_passes"] == 1 and report["measured_passes"] == 2
        assert len(report["timings"]) == 2 and len(report["repeat_losses"]) == 3
        assert len(set(report["repeat_losses"])) == 1
    assert error.grad is None and torch.equal(error, original_error) and torch.equal(inputs, original_inputs)


def test_full_tensor_difference_does_not_ignore_zero_or_tail_coordinates():
    reference = torch.zeros(1_048_577)
    actual = reference.clone()
    actual[-1] = 2
    difference = tensor_difference(actual, reference)
    assert difference["numel"] == 1_048_577
    assert difference["max_absolute_difference"] == difference["difference_l2"] == 2
    assert difference["cosine"] is None
    with pytest.raises(ValueError, match="equally shaped"):
        tensor_difference(actual[:-1], reference)
