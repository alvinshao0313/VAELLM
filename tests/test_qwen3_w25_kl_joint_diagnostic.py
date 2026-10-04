"""CPU persistent/grouped state restoration and fixed-state interaction arithmetic."""

import pytest
import torch

from experiments.qwen3_w25.kl_joint_diagnostic import CODE_NAMES, interaction_summary, restore_phase, snapshot
from test_qwen3_w25_expand_decoder import _layer


def _parts(layer):
    state = layer.state_dict()
    return (snapshot({k: v for k, v in state.items() if k.startswith("_parallel_stage_decoder.")}),
            snapshot({k: state[k] for k in CODE_NAMES}))


def _decoded(layer):
    with torch.no_grad():
        values = layer._get_parallel_stage_grouped_vq(dtype=torch.float32, device=torch.device("cpu"))
        return layer._parallel_stage_decoder(values).clone()


def test_crossing_rebuilds_real_grouped_codes_and_preserves_parameter_objects():
    layer = _layer()
    decoder0, codes0 = _parts(layer)
    parameters = dict(layer.named_parameters())
    output0 = _decoded(layer)
    decoder1, codes1 = snapshot(decoder0), snapshot(codes0)
    for value in decoder1.values():
        value.add_(0.25)
    codes1[CODE_NAMES[0]][0, 0, 0] ^= 1
    restore_phase(layer, decoder1, codes1)
    output1 = _decoded(layer)
    assert not torch.equal(output1, output0)
    layer._cached_weight = torch.zeros(64, 64)
    restore_phase(layer, decoder0, codes0)
    assert torch.equal(_decoded(layer), output0)
    assert layer._cached_weight is None
    assert all(dict(layer.named_parameters())[name] is param for name, param in parameters.items())
    actual_decoder, actual_codes = _parts(layer)
    assert all(torch.equal(actual_decoder[k], v) for k, v in decoder0.items())
    assert all(torch.equal(actual_codes[k], v) for k, v in codes0.items())
    restore_phase(layer, decoder1, codes1)
    assert torch.equal(_decoded(layer), output1)


def test_invalid_crossing_is_rejected_before_any_persistent_write():
    layer = _layer()
    before = snapshot(layer.state_dict())
    decoder, codes = _parts(layer)
    next(iter(decoder.values())).add_(1)
    codes[CODE_NAMES[1]] = codes[CODE_NAMES[1]][:1]
    with pytest.raises(ValueError, match="shape/dtype"):
        restore_phase(layer, decoder, codes)
    assert all(torch.equal(layer.state_dict()[k], v) for k, v in before.items())


def test_four_way_interaction_reports_benefits_and_nonadditivity():
    losses = dict(D0C0=1., D1C0=.8, D0C1=.9, D1C1=.6)
    result = interaction_summary({k: {"loss_denominator_weighted": v} for k, v in losses.items()})
    assert result["code_benefit_at_trained_decoder"] == pytest.approx(.2)
    assert result["decoder_benefit_at_trained_codes"] == pytest.approx(.3)
    assert result["interaction_loss"] == pytest.approx(-.1)
