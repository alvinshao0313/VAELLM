"""Real small packed-v6 endpoint export: exact BF16 cast, codes, frozen tensors and reload."""
import hashlib
import json
from uuid import uuid4

import pytest
import torch
from torch import nn

from experiments.qwen3_w25.kl_gradient_diagnostic import SOURCE_ID
from experiments.qwen3_w25.kl_joint_export import (
    CODE_NAMES, DECODER_NAMES, SELECTED_KEYS, TARGET, apply_payload_, export_joint_endpoint, project_payload)
from test_qwen3_w25_expand_decoder import _exact, _layer
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import load_v6_full_checkpoint_into_model, load_v6_meta, save_v6_full_checkpoint


def _model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mlp = nn.Module()
    model.model.layers[0].mlp.down_proj = _layer(torch.bfloat16)
    model.register_buffer("frozen_marker", torch.tensor([3.], dtype=torch.bfloat16))
    return model


def _payload(model, joint, source_id=SOURCE_ID):
    state = model.get_submodule(TARGET).state_dict()
    codes = {name: state[name].clone() for name in CODE_NAMES}
    if joint:
        codes[CODE_NAMES[1]][0, 0, 0] ^= 1
    return {"format": "qwen3_kl_joint_bridge_v1", "source_checkpoint_id": source_id,
            "target": TARGET, "steps": 256, "joint": joint,
            "decoder_fp32": {name: state[name].float().clone() + .013 for name in DECODER_NAMES},
            "codes": codes}


@pytest.mark.parametrize("joint", [False, True])
def test_bf16_endpoint_keeps_other_bytes_specs_and_budget_through_v6_reload(tmp_path, joint):
    model = _model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    save_v6_full_checkpoint(model, str(tmp_path / "source"), checkpoint_kind="final_model",
        compressed_targets=(TARGET,), base_model_path="cpu-joint-export-fixture", save_config=False)
    source_meta = load_v6_meta(str(tmp_path / "source"))
    source_id = source_meta["checkpoint_id"]
    assert source_id != SOURCE_ID
    payload = _payload(model, joint, source_id)
    selected = project_payload(payload, expected_source_id=source_id)
    parameters = dict(model.named_parameters())
    model.get_submodule(TARGET)._cached_weight = torch.ones(64, 64)
    apply_payload_(model, payload, expected_source_id=source_id)
    assert model.get_submodule(TARGET)._cached_weight is None
    assert all(dict(model.named_parameters())[name] is p for name, p in parameters.items())
    for name, value in model.state_dict().items():
        _exact(value, selected[name] if name in SELECTED_KEYS else original[name])
    for name in DECODER_NAMES:
        _exact(model.state_dict()[TARGET + "." + name], payload["decoder_fp32"][name].bfloat16())
    save_v6_full_checkpoint(model, str(tmp_path / "output"), checkpoint_kind="final_model",
        compressed_targets=(TARGET,), base_model_path="cpu-joint-export-fixture", save_config=False)
    saved = load_v6_meta(str(tmp_path / "output"))
    assert saved["converted_modules"] == source_meta["converted_modules"]
    skeleton = _model()
    skeleton.model.layers[0].mlp.down_proj = nn.Linear(64, 64, bias=False).bfloat16()
    restored, meta, status = load_v6_full_checkpoint_into_model(skeleton, str(tmp_path / "output"),
        expected_kind="final_model", map_location="cpu", strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    for name, value in model.state_dict().items():
        _exact(restored.state_dict()[name], value)
    before = audit_state_dict(source_meta, original, max_bpw=1024, max_lora_rank=0)
    after = audit_state_dict(meta, restored.state_dict(), max_bpw=1024, max_lora_rank=0)
    for key in ("user_linear_bits", "packed_linear_bits", "original_linear_weights", "components", "outside_linear_bits"):
        assert before[key] == after[key]


def test_wrong_shape_is_rejected_before_any_write_and_extra_leaf_is_forbidden():
    model = _model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    payload = _payload(model, True)
    payload["codes"][CODE_NAMES[1]] = payload["codes"][CODE_NAMES[1]][:1].clone()
    with pytest.raises(ValueError, match="source geometry"):
        apply_payload_(model, payload, expected_source_id=SOURCE_ID)
    for name, value in original.items():
        _exact(model.state_dict()[name], value)
    payload = _payload(model, True)
    payload["decoder_fp32"]["other.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="six decoder"):
        project_payload(payload, expected_source_id=SOURCE_ID)


def test_payload_from_another_source_is_rejected_before_any_write():
    model = _model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    payload = _payload(model, True, str(uuid4()))
    with pytest.raises(ValueError, match="same-source"):
        apply_payload_(model, payload, expected_source_id=str(uuid4()))
    for name, value in original.items():
        _exact(model.state_dict()[name], value)


@pytest.mark.parametrize("mismatch", ["source", "payload"])
def test_export_rejects_report_source_payload_identity_mismatch(tmp_path, mismatch):
    source = tmp_path / "source"
    model = _model()
    save_v6_full_checkpoint(model, str(source), checkpoint_kind="final_model",
        compressed_targets=(TARGET,), base_model_path="cpu-joint-export-fixture", save_config=False)
    source_id = load_v6_meta(str(source))["checkpoint_id"]
    observation = tmp_path / "observation"
    observation.mkdir()
    payload_path = observation / "selected_state.pt"
    torch.save(_payload(model, True, str(uuid4())), payload_path)
    diagnostic = observation / "diagnostic.json"
    diagnostic.write_text(json.dumps({
        "status": "complete", "steps": 256, "target": TARGET,
        "source_checkpoint_id": str(uuid4()) if mismatch == "source" else source_id,
        "source_dir": str(source), "joint": True,
        "frozen_state": {"unchanged": True}, "frozen_cache_pointers_unchanged": True,
        "source_pins_data_unchanged": True, "runtime": {"frozen_tensor_count": 3415},
        "selected_payload": {"path": str(payload_path), "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest()},
    }))
    message = "identity differs from the endpoint report" if mismatch == "source" else "same-source"
    with pytest.raises(ValueError, match=message):
        export_joint_endpoint(diagnostic, tmp_path / "output")
    assert not (tmp_path / "output").exists()
