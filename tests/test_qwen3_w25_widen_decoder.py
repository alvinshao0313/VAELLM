"""Native width initialization, gradient asymmetry and actual v6 round trip."""

import pytest
import torch
from torch import nn

from experiments.qwen3_w25.widen_decoder import widen_packed_decoder_h128_to_h256_
from test_qwen3_w25_expand_decoder import _exact, _layer, _snapshot
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, save_v6_full_checkpoint


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_width_initialization_preserves_native_function_and_other_state(dtype):
    layer = _layer(dtype)
    old = layer._parallel_stage_decoder
    state = _snapshot(layer)
    x = torch.randint(0, 2, (97, 2, 32)).to(dtype)
    with torch.no_grad():
        expected = old(x)
    rng = torch.get_rng_state().clone()
    report = widen_packed_decoder_h128_to_h256_(layer)
    new = layer._parallel_stage_decoder
    assert torch.equal(rng, torch.get_rng_state())
    assert (new.training, new.use_checkpoint, new._q_scale_fused) == (old.training, old.use_checkpoint, old._q_scale_fused)
    assert not any(p.requires_grad for p in new.parameters())
    assert layer.state_dict().keys() == state.keys()
    for name, value in state.items():
        if not name.startswith("_parallel_stage_decoder."):
            _exact(layer.state_dict()[name], value)
    _exact(new.linear_out.conv.bias, old.linear_out.conv.bias)
    assert report["added_parameters"] == 2 * 128 * (32 + 32 + 3)
    with torch.no_grad():
        actual = new(x)
    # Reduction widths differ. Use two storage ULPs plus a relative term, while
    # testing the mathematical identity separately in double precision below.
    epsilon = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, atol=2 * epsilon, rtol=2 * epsilon)
    with torch.no_grad():
        torch.testing.assert_close(new.double()(x.double()), old.double()(x.double()), atol=1e-12, rtol=1e-12)


def test_zero_output_half_can_learn_and_hidden_copies_split():
    layer = _layer(trainable=True)
    widen_packed_decoder_h128_to_h256_(layer)
    decoder = layer._parallel_stage_decoder
    x = torch.randint(0, 2, (29, 2, 32)).float()
    probe = torch.randn(29, 2, 32)
    (decoder(x) * probe).square().mean().backward()
    gradients = decoder.linear_in.conv.weight.grad.reshape(2, 256, 32, 1)
    assert torch.count_nonzero(gradients[:, :128]) > 0
    assert torch.count_nonzero(gradients[:, 128:]) > 0  # LayerNorm couples both copies.
    assert not torch.equal(gradients[:, :128], gradients[:, 128:])
    out_grad = decoder.linear_out.conv.weight.grad.reshape(2, 32, 256, 1)
    assert torch.count_nonzero(out_grad[:, :, 128:]) > 0
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-5)
    optimizer.step()
    weights = decoder.linear_in.conv.weight.detach().reshape(2, 256, 32, 1)
    assert not torch.equal(weights[:, :128], weights[:, 128:])
    assert torch.count_nonzero(decoder.linear_out.conv.weight.reshape(2, 32, 256, 1)[:, :, 128:]) > 0


def test_width_pack_unpack_and_v6_reload_keep_exact_state(tmp_path):
    layer = _layer(torch.bfloat16)
    widen_packed_decoder_h128_to_h256_(layer)
    state = _snapshot(layer)
    assert layer.unpack_parallel_stage_decoder_()
    for index in range(2):
        decoder = layer.get_stage_part_decoder(stage_idx=index, part_idx=0)
        assert decoder.hidden_dim == 256 and decoder.decoder_type == "asymmetric"
    assert layer.pack_parallel_stage_decoder_(trainable=False)
    for name, value in state.items():
        _exact(layer.state_dict()[name], value)
    model = nn.Module()
    model.down_proj = layer
    output = tmp_path / "widened"
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind="final_model",
        compressed_targets=("down_proj",), base_model_path="cpu-width-fixture", save_config=False)
    skeleton = nn.Module()
    skeleton.down_proj = nn.Linear(64, 64, bias=False).bfloat16()
    reloaded, meta, status = load_v6_full_checkpoint_into_model(skeleton, str(output),
        expected_kind="final_model", map_location="cpu", strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    assert meta["checkpoint_id"] == saved["checkpoint_id"]
    for name, value in model.state_dict().items():
        _exact(reloaded.state_dict()[name], value)
    decoder = reloaded.down_proj._parallel_stage_decoder
    assert decoder.hidden_dim == 256 and decoder.decoder_type == "asymmetric"
    budget = audit_state_dict(meta, reloaded.state_dict(), max_bpw=1024, max_lora_rank=0)
    assert budget["components"]["decoder"]["storage_bits"] == sum(
        value.numel() * value.element_size() * 8 for value in decoder.state_dict().values())


def test_repeated_width_expansion_rejected_without_mutation():
    layer = _layer()
    widen_packed_decoder_h128_to_h256_(layer)
    state = _snapshot(layer)
    with pytest.raises(ValueError, match="H128 symmetric"):
        widen_packed_decoder_h128_to_h256_(layer)
    for name, value in state.items():
        _exact(layer.state_dict()[name], value)
