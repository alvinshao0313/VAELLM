from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.trainer_callback import TrainerState

from compressed_e2e_fintuning.runtime_v6_pipeline import (
    _assert_finalization_probe_close,
    _build_finalization_probe_inputs,
    _load_completed_resume_state,
    _run_finalization_probe,
    _unwrap_model_for_finalization,
)


class _TinyTokenizer:
    def __call__(self, _text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is True
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.embed(input_ids)
        return SimpleNamespace(logits=self.head(hidden))


def test_finalization_probe_is_short_finite_and_exact_for_unchanged_model():
    torch.manual_seed(5)
    model = _TinyLM().eval()
    probe_inputs = _build_finalization_probe_inputs(_TinyTokenizer())
    assert tuple(probe_inputs["input_ids"].shape) == (1, 4)

    before, dtype_before = _run_finalization_probe(model, probe_inputs)
    after, dtype_after = _run_finalization_probe(model, probe_inputs)
    assert dtype_after == dtype_before

    stats = _assert_finalization_probe_close(
        before,
        after,
        output_dtype=dtype_before,
    )
    assert stats["max_abs"] == 0.0
    assert stats["relative_l2"] == 0.0


def test_finalization_probe_rejects_material_output_change():
    before = torch.zeros((1, 2, 3), dtype=torch.float32)
    after = before.clone()
    after[0, 0, 0] = 0.1
    with pytest.raises(AssertionError):
        _assert_finalization_probe_close(
            before,
            after,
            output_dtype=torch.float32,
        )


def test_bfloat16_fusion_rounding_tolerance_must_be_explicit():
    before = torch.zeros((1, 2, 3), dtype=torch.float32)
    after = before.clone()
    after[0, 0, 0] = 0.125

    with pytest.raises(AssertionError):
        _assert_finalization_probe_close(
            before,
            after,
            output_dtype=torch.bfloat16,
        )

    stats = _assert_finalization_probe_close(
        before,
        after,
        output_dtype=torch.bfloat16,
        ulp_multiplier=32.0,
    )
    assert stats["max_abs"] == pytest.approx(0.125)
    assert stats["atol"] == pytest.approx(0.25)


def test_finalization_unwrap_removes_accelerate_fp32_forward_wrapper():
    model = _TinyLM()

    class _Accelerator:
        def __init__(self):
            self.calls = []

        def unwrap_model(self, target, *, keep_fp32_wrapper):
            self.calls.append((target, keep_fp32_wrapper))
            return target

    accelerator = _Accelerator()
    trainer = SimpleNamespace(model=model, accelerator=accelerator)

    assert _unwrap_model_for_finalization(trainer) is model
    assert accelerator.calls == [(model, False)]


def test_completed_resume_state_prevents_extra_iterable_dataset_step(tmp_path):
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    state = TrainerState(global_step=20, max_steps=20)
    state.save_to_json(str(checkpoint / "trainer_state.json"))

    assert _load_completed_resume_state(str(checkpoint), max_steps=21) is None
    completed = _load_completed_resume_state(str(checkpoint), max_steps=20)
    assert completed is not None
    assert completed.global_step == 20
    with pytest.raises(ValueError, match="global_step exceeds"):
        _load_completed_resume_state(str(checkpoint), max_steps=19)
