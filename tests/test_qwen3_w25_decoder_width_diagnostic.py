"""CPU contracts for the local KL width observer, not a GPU performance claim."""
import math

import pytest
import torch

from experiments.qwen3_w25.decoder_width_diagnostic import (
    decoder_snapshot, parameter_changes, project_selected_bf16_, summarize_losses, width_activity,
)
from experiments.qwen3_w25.widen_decoder import widen_packed_decoder_h128_to_h256_
from test_qwen3_w25_expand_decoder import _layer


def test_effective_token_weighted_loss_is_distinct_from_equal_batch_mean():
    result = summarize_losses([{"loss": 1., "denominator": 3.}, {"loss": 4., "denominator": 9.}])
    assert result["equal_batch_loss"] == 2.5
    assert result["loss_denominator_weighted"] == 3.25
    assert result["loss_denominator_sum"] == 12
    for invalid in ([], [{"loss": math.nan, "denominator": 1.}], [{"loss": 1., "denominator": 0.}]):
        with pytest.raises(ValueError):
            summarize_losses(invalid)


def test_fp32_changes_bf16_survival_and_projection_are_separate():
    selected = {"one": torch.nn.Parameter(torch.tensor([1., 1.], dtype=torch.float32))}
    other = torch.nn.Parameter(torch.tensor([2.], dtype=torch.bfloat16), requires_grad=False)
    original_object = selected["one"]
    before = decoder_snapshot(selected)
    with torch.no_grad():
        selected["one"].add_(torch.tensor([1e-5, .02]))
    fp32 = decoder_snapshot(selected)
    assert parameter_changes(before, fp32)["changed_elements"] == 2
    projected = project_selected_bf16_(selected)
    assert selected["one"] is original_object
    assert torch.equal(projected["one"], fp32["one"].bfloat16())
    assert parameter_changes({name: p.bfloat16() for name, p in before.items()}, projected)["changed_elements"] == 1
    assert fp32["one"].dtype == torch.float32 and other.dtype == torch.bfloat16 and other.item() == 2
    with pytest.raises(ValueError, match="shape/dtype"):
        parameter_changes(before, projected)


def test_real_packed_width_observer_reports_new_half_and_split_after_update():
    layer = _layer(trainable=True)
    widen_packed_decoder_h128_to_h256_(layer)
    decoder = layer._parallel_stage_decoder
    initial = width_activity(decoder)
    assert initial["new_output_half_nonzero"] == initial["hidden_copy_different_elements"] == 0
    probe = torch.randn(17, 2, 32)
    (decoder(torch.randint(0, 2, (17, 2, 32)).float()) * probe).square().mean().backward()
    gradient = width_activity(decoder, gradients=True)
    assert gradient["new_output_half_nonzero"] > 0 and gradient["hidden_copy_different_elements"] > 0
    torch.optim.AdamW(decoder.parameters(), lr=1e-5, weight_decay=0).step()
    after = width_activity(decoder)
    assert after["new_output_half_nonzero"] > 0 and after["hidden_copy_different_elements"] > 0
