"""Real small packed modules: endpoint substitution, metadata and v6 storage."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.qwen3_w25.decoder_width_export import SELECTED_KEYS, apply_payload_, validate_payload, width_specs
from experiments.qwen3_w25.kl_gradient_diagnostic import SOURCE_ID, TARGETS
from experiments.qwen3_w25.widen_decoder import widen_packed_decoder_h128_to_h256_
from test_qwen3_w25_expand_decoder import _exact, _layer
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, load_v6_meta, save_v6_full_checkpoint


def _model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.gate_proj = _layer(torch.bfloat16)
    model.model.layers[0].mlp.down_proj = _layer(torch.bfloat16)
    model.register_buffer("frozen_marker", torch.tensor([3.], dtype=torch.bfloat16))
    return model


def _payload(source, width):
    donor = deepcopy(source)
    if width == 256:
        for name in TARGETS:
            widen_packed_decoder_h128_to_h256_(donor.get_submodule(name))
    fp32 = {name: value.float().clone() + .013 for name, value in donor.state_dict().items() if name in SELECTED_KEYS}
    return {"format": "qwen3_decoder_width_probe_v1", "source_checkpoint_id": SOURCE_ID,
            "hidden_dim": width, "targets": list(TARGETS), "steps": 256,
            "fp32": fp32, "bf16": {name: value.bfloat16() for name, value in fp32.items()}}


@pytest.mark.parametrize("width", [128, 256])
def test_saved_endpoint_preserves_frozen_bytes_and_real_v6_specs(tmp_path, width):
    model = _model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    save_v6_full_checkpoint(model, str(tmp_path / "source"), checkpoint_kind="final_model",
        compressed_targets=TARGETS, base_model_path="cpu-width-export-fixture", save_config=False)
    source_meta = load_v6_meta(str(tmp_path / "source"))
    payload = _payload(model, width)
    transformations = apply_payload_(model, payload)
    assert len(transformations) == (2 if width == 256 else 0)
    for name, value in model.state_dict().items():
        _exact(value, payload["bf16"][name] if name in SELECTED_KEYS else original[name])
    save_v6_full_checkpoint(model, str(tmp_path / "output"), checkpoint_kind="final_model",
        compressed_targets=TARGETS, base_model_path="cpu-width-export-fixture", save_config=False)
    saved = load_v6_meta(str(tmp_path / "output"))
    assert saved["converted_modules"] == width_specs(source_meta, width)
    skeleton = _model()
    for name in TARGETS:
        parent = skeleton.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], nn.Linear(64, 64, bias=False).bfloat16())
    restored, meta, status = load_v6_full_checkpoint_into_model(skeleton, str(tmp_path / "output"),
        expected_kind="final_model", map_location="cpu", strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    assert meta["converted_modules"] == saved["converted_modules"]
    for name, value in model.state_dict().items():
        _exact(restored.state_dict()[name], value)
    before = audit_state_dict(source_meta, original, max_bpw=1024, max_lora_rank=0)
    after = audit_state_dict(meta, restored.state_dict(), max_bpw=1024, max_lora_rank=0)
    added = sum((payload["bf16"][name].numel() - original[name].numel()) * 16 for name in SELECTED_KEYS)
    assert after["user_linear_bits"] - before["user_linear_bits"] == added
    assert before["outside_linear_bits"] == after["outside_linear_bits"]


def test_bad_cast_and_extra_backbone_payload_are_rejected():
    payload = _payload(_model(), 128)
    key = next(iter(SELECTED_KEYS))
    payload["bf16"][key].reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="bytes differ"):
        validate_payload(payload)
    payload = _payload(_model(), 128)
    payload["bf16"]["frozen_marker"] = torch.zeros(1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="twelve"):
        validate_payload(payload)


def test_bad_h256_shape_rejected_before_either_module_is_changed():
    model = _model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    payload = _payload(model, 256)
    key = TARGETS[1] + "._parallel_stage_decoder.linear_out.conv.weight"
    payload["fp32"][key] = payload["fp32"][key][:1].clone()
    payload["bf16"][key] = payload["fp32"][key].bfloat16()
    with pytest.raises(ValueError, match="real source geometry"):
        apply_payload_(model, payload)
    for name, value in model.state_dict().items():
        _exact(value, original[name])
