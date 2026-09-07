from types import MethodType
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from accelerate.utils.operations import convert_outputs_to_fp32

from compressed_e2e_fintuning.mid_eval import (
    EvalAfterSaveCallback,
    temporary_disable_fp32_output_conversion,
)


class _TinyWrappedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.amp_hits = 0
        self.linear = nn.Linear(4, 4, bias=False)

    def _inner_amp_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.amp_hits += 1
        return self.linear(x).to(dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner_amp_forward(x)


def _wrap_like_accelerate_1_7_0(model: nn.Module) -> nn.Module:
    inner_amp_forward = model._inner_amp_forward.__func__
    model.forward = MethodType(inner_amp_forward, model)
    model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
    return model


def _assert_same_forward(actual, expected) -> None:
    if actual is expected:
        return
    assert actual.__func__ is expected.__func__
    assert actual.__self__ is expected.__self__


def test_temporary_disable_fp32_output_conversion_keeps_inner_amp_forward():
    model = _wrap_like_accelerate_1_7_0(_TinyWrappedModel())
    x = torch.ones(2, 4)
    saved_wrapped_forward = model.forward

    assert model(x).dtype == torch.float32
    amp_hits_before = int(model.amp_hits)

    with temporary_disable_fp32_output_conversion(model):
        out = model(x)
        assert out.dtype == torch.bfloat16
        assert model.amp_hits == amp_hits_before + 1

    assert model(x).dtype == torch.float32
    _assert_same_forward(model.forward, saved_wrapped_forward)


def test_temporary_disable_fp32_output_conversion_restores_on_exception():
    model = _wrap_like_accelerate_1_7_0(_TinyWrappedModel())
    x = torch.ones(2, 4)
    saved_wrapped_forward = model.forward
    assert model(x).dtype == torch.float32

    with pytest.raises(RuntimeError, match="boom"):
        with temporary_disable_fp32_output_conversion(model):
            assert model(x).dtype == torch.bfloat16
            raise RuntimeError("boom")

    assert model(x).dtype == torch.float32
    _assert_same_forward(model.forward, saved_wrapped_forward)


def test_resolve_eval_model_explicitly_keeps_fp32_wrapper():
    class FakeAccelerator:
        def __init__(self):
            self.calls = []

        def unwrap_model(self, model, *, keep_fp32_wrapper):
            self.calls.append((model, keep_fp32_wrapper))
            return model

    model = nn.Linear(2, 2)
    accelerator = FakeAccelerator()
    callback = EvalAfterSaveCallback(
        e2e_args=SimpleNamespace(),
        tokenizer=object(),
        base_model_path="/tmp/base",
        run_output_dir="/tmp/run",
        log=SimpleNamespace(),
        parallel_mode="layer_mp",
    )
    callback.bind_trainer(SimpleNamespace(model=model, accelerator=accelerator))

    resolved = callback._resolve_eval_model(None)

    assert resolved is model
    assert accelerator.calls == [(model, True)]
