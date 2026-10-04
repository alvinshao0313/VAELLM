"""CPU native mathematics and real v6 state; no packed CUDA-kernel claim."""

import pytest
import torch
from torch import nn

from experiments.qwen3_w25.expand_decoder import expand_packed_decoder_r0_to_r1_
from litebsq.autoencoder import Decoder
from litebsq.protected_channel_quant import encode_protected_channel_weight
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, save_v6_full_checkpoint


def _layer(dtype=torch.float32, *, trainable=False):
    torch.manual_seed(731)
    decoders = [Decoder(in_dim=32, out_dim=32, hidden_dim=128, num_res_blocks=0,
        norm_type="layer", decoder_type="symmetric", use_checkpoint=True).to(dtype)
        for _ in range(2)]
    for decoder in decoders:
        decoder._fuse_q_scale()
    qvalues, scales = encode_protected_channel_weight(torch.randn(32, 64), quant_format="int8")
    module = VAELinear(
        in_features=64, out_features=64, bias=None, codebook_dim=32, transpose=False,
        compressed_in_features=32, protected_input_indices=torch.arange(32),
        protected_input_qvalues=qvalues, protected_input_scales=scales,
        protected_channel_quant_format="int8",
        weight_rotation_spec=rotation_spec("two_sided", 16, 31, "q_proj"),
        stage_vq_weights=[torch.randint(0, 2, (64, 1, 32), dtype=torch.bool) for _ in range(2)],
        stage_decoders=decoders,
    )
    module.train(trainable)
    module.pack_parallel_stage_decoder_(trainable=trainable)
    return module


def _snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8),
                       expected.contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_identity_preserves_original_objects_state_and_rng(dtype):
    layer = _layer(dtype)
    decoder = layer._parallel_stage_decoder
    original = _snapshot(layer)
    parameters = dict(decoder.named_parameters())
    flags = (decoder.training, decoder.use_checkpoint, decoder._q_scale_fused)
    x = torch.randn(17, 2, 32).to(dtype)
    with torch.no_grad():
        expected = decoder(x)
    rng = torch.get_rng_state().clone()
    report = expand_packed_decoder_r0_to_r1_(layer)
    assert torch.equal(rng, torch.get_rng_state())
    assert layer._parallel_stage_decoder is decoder
    assert flags == (decoder.training, decoder.use_checkpoint, decoder._q_scale_fused)
    assert decoder.num_res_blocks == len(decoder.blocks) == 1
    assert not any(p.requires_grad for p in decoder.parameters())
    for name, parameter in parameters.items():
        assert dict(decoder.named_parameters())[name] is parameter
    actual = layer.state_dict()
    for name, value in original.items():
        _exact(actual[name], value)
    assert set(actual) - set(original) == set(report["added_tensor_keys"])
    assert len(report["added_tensor_keys"]) == 4
    assert report["added_parameters"] == 2 * (128 ** 2 + 3 * 128)
    assert report["added_tensor_bytes"] == report["added_parameters"] * torch.empty((), dtype=dtype).element_size()
    with torch.no_grad():
        _exact(decoder(x), expected)


def test_zero_output_block_has_linear_gradient_before_norm_gradient():
    layer = _layer(trainable=True)
    expand_packed_decoder_r0_to_r1_(layer)
    decoder = layer._parallel_stage_decoder
    block = decoder.blocks[0]
    x = torch.randn(19, 2, 32)
    probe = torch.randn(19, 2, 32)
    loss = (decoder(x) * probe).sum()
    loss.backward()
    for parameter in block.linear1.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    for parameter in block.norm1.parameters():
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
    before = block.linear1.conv.weight.detach().clone()
    torch.optim.SGD(decoder.parameters(), lr=1e-4).step()
    assert not torch.equal(before, block.linear1.conv.weight)


def test_pack_unpack_and_v6_reload_preserve_expanded_state(tmp_path):
    layer = _layer(torch.bfloat16)
    expand_packed_decoder_r0_to_r1_(layer)
    expanded = _snapshot(layer)
    flags = layer._parallel_stage_decoder._q_scale_fused
    assert layer.unpack_parallel_stage_decoder_()
    for stage in range(2):
        decoder = layer.get_stage_part_decoder(stage_idx=stage, part_idx=0)
        assert decoder.num_res_blocks == len(decoder.blocks) == 1
        assert decoder._q_scale_fused == flags
    assert layer.pack_parallel_stage_decoder_(trainable=False)
    for key, value in expanded.items():
        _exact(layer.state_dict()[key], value)
    assert layer.state_dict().keys() == expanded.keys()

    root = nn.Module()
    root.q_proj = layer
    root.norm = nn.LayerNorm(64).bfloat16()
    root.lm_head = nn.Linear(64, 8, bias=False).bfloat16()
    state = _snapshot(root)
    output = tmp_path / "expanded"
    saved = save_v6_full_checkpoint(root, str(output), checkpoint_kind="final_model",
        compressed_targets=("q_proj",), base_model_path="cpu-expansion-fixture", save_config=False)
    skeleton = nn.Module()
    skeleton.q_proj = nn.Linear(64, 64, bias=False)
    skeleton.norm = nn.LayerNorm(64).bfloat16()
    skeleton.lm_head = nn.Linear(64, 8, bias=False).bfloat16()
    loaded, meta, status = load_v6_full_checkpoint_into_model(skeleton, str(output),
        expected_kind="final_model", map_location="cpu", strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    assert meta["checkpoint_id"] == saved["checkpoint_id"]
    assert loaded.state_dict().keys() == state.keys()
    for key, value in state.items():
        _exact(loaded.state_dict()[key], value)
    assert all(spec["num_res_blocks"] == 1 for spec in meta["converted_modules"][0]["stage_decoders"])
    decoder = loaded.q_proj._parallel_stage_decoder
    assert decoder.num_res_blocks == len(decoder.blocks) == 1
    x = torch.randn(7, 2, 32).bfloat16()
    with torch.no_grad():
        _exact(decoder(x), layer._parallel_stage_decoder(x))
    # Tiny carrier has a deliberately small denominator; measure its real bytes,
    # without pretending it satisfies the full model's 2.5-bpw requirement.
    budget = audit_state_dict(meta, loaded.state_dict(), max_bpw=1024, max_lora_rank=0)
    assert budget["components"]["decoder"]["storage_bits"] == sum(
        value.numel() * value.element_size() * 8 for key, value in state.items()
        if key.startswith("q_proj._parallel_stage_decoder."))


def test_repeated_expansion_is_rejected_without_state_change():
    layer = _layer()
    expand_packed_decoder_r0_to_r1_(layer)
    before = _snapshot(layer)
    with pytest.raises(ValueError, match="requires R0"):
        expand_packed_decoder_r0_to_r1_(layer)
    assert layer.state_dict().keys() == before.keys()
    for key, value in before.items():
        _exact(layer.state_dict()[key], value)
