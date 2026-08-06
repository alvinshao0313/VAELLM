from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rotation.model_utils as model_utils


def test_known_qwen_path_keeps_existing_loader_branch():
    sentinel = object()
    with patch.object(model_utils, "get_qwen3", return_value=sentinel) as mock_qwen3, patch.object(
        model_utils, "get_auto_causal_lm"
    ) as mock_auto:
        out = model_utils.get_model("Qwen/Qwen3-8B", hf_token=None)
    assert out is sentinel
    mock_qwen3.assert_called_once_with("Qwen/Qwen3-8B", None)
    mock_auto.assert_not_called()


def test_unknown_model_path_uses_auto_causal_lm_fallback():
    sentinel = object()
    with patch.object(model_utils, "get_auto_causal_lm", return_value=sentinel) as mock_auto:
        out = model_utils.get_model("org/SomeNewCausalLM", hf_token="tok")
    assert out is sentinel
    mock_auto.assert_called_once_with("org/SomeNewCausalLM", "tok")


def test_auto_fallback_passes_dtype_and_low_cpu_mem_usage():
    captured = {}

    def fake_from_pretrained(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        model = MagicMock()
        model.config = SimpleNamespace(max_position_embeddings=4096, model_type="newlm")
        model.__class__.__name__ = "NewLMForCausalLM"
        return model

    with patch.object(
        model_utils.transformers.AutoModelForCausalLM,
        "from_pretrained",
        side_effect=fake_from_pretrained,
    ):
        model = model_utils.get_auto_causal_lm("org/NewLM", hf_token="abc")

    assert captured["name"] == "org/NewLM"
    assert captured["kwargs"]["torch_dtype"] == "auto"
    assert captured["kwargs"]["low_cpu_mem_usage"] is True
    assert captured["kwargs"]["trust_remote_code"] is False
    assert captured["kwargs"]["token"] == "abc"
    assert model.seqlen == 4096


def test_auto_fallback_sets_seqlen_from_config():
    model = MagicMock()
    model.config = SimpleNamespace(max_position_embeddings=8192, model_type="x")
    model.__class__.__name__ = "X"
    with patch.object(
        model_utils.transformers.AutoModelForCausalLM,
        "from_pretrained",
        return_value=model,
    ):
        out = model_utils.get_auto_causal_lm("org/X", hf_token=None)
    assert out.seqlen == 8192


def test_auto_fallback_defaults_seqlen_to_2048():
    model = MagicMock()
    model.config = SimpleNamespace(model_type="x")
    model.__class__.__name__ = "X"
    with patch.object(
        model_utils.transformers.AutoModelForCausalLM,
        "from_pretrained",
        return_value=model,
    ):
        out = model_utils.get_auto_causal_lm("org/X", hf_token="")
    assert out.seqlen == 2048


def test_auto_fallback_does_not_enable_trust_remote_code():
    captured = {}

    def fake_from_pretrained(name, **kwargs):
        captured["kwargs"] = kwargs
        model = MagicMock()
        model.config = SimpleNamespace(max_position_embeddings=0, model_type="x")
        model.__class__.__name__ = "X"
        return model

    with patch.object(
        model_utils.transformers.AutoModelForCausalLM,
        "from_pretrained",
        side_effect=fake_from_pretrained,
    ):
        model_utils.get_auto_causal_lm("org/X", hf_token=None)

    assert captured["kwargs"]["trust_remote_code"] is False
    assert "token" not in captured["kwargs"]
