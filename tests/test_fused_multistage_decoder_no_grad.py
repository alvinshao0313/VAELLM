import pytest
import torch

import litebsq.fused_multistage_decoder as fused_decoder


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _inputs(*, requires_grad: bool):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch, stages, in_dim, hidden_dim, out_dim = 2, 2, 4, 8, 4
    x = torch.randn(batch, stages, in_dim, device=device, dtype=dtype)
    w_in = torch.randn(stages, hidden_dim, in_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    b_in = torch.randn(stages, hidden_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    ln_w = torch.randn(stages, hidden_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    ln_b = torch.randn(stages, hidden_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    w_out = torch.randn(stages, out_dim, hidden_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    b_out = torch.randn(stages, out_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    return x, w_in, b_in, ln_w, ln_b, w_out, b_out


def test_fused_decode_no_grad_does_not_save_backward_aux(monkeypatch):
    calls = []

    def fake_forward(x, w_in, b_in, ln_w, ln_b, w_out, b_out, *, eps, save_aux):
        calls.append(bool(save_aux))
        y = torch.empty((x.shape[0], w_out.shape[1]), device=x.device, dtype=x.dtype)
        if save_aux:
            h_shape = (x.shape[0], x.shape[1], w_in.shape[1])
            h_pre = torch.empty(h_shape, device=x.device, dtype=x.dtype)
            h_act = torch.empty(h_shape, device=x.device, dtype=x.dtype)
            return y, h_pre, h_act
        return y, None, None

    monkeypatch.setattr(fused_decoder, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fused_decoder, "_fused_forward_triton", fake_forward)
    inputs = _inputs(requires_grad=True)

    with torch.no_grad():
        fused_decoder.fused_multistage_symmetric_decode(*inputs)

    assert calls == [False]


def test_fused_decode_grad_enabled_still_saves_backward_aux(monkeypatch):
    calls = []

    def fake_forward(x, w_in, b_in, ln_w, ln_b, w_out, b_out, *, eps, save_aux):
        calls.append(bool(save_aux))
        y = torch.empty((x.shape[0], w_out.shape[1]), device=x.device, dtype=x.dtype)
        h_shape = (x.shape[0], x.shape[1], w_in.shape[1])
        h_pre = torch.empty(h_shape, device=x.device, dtype=x.dtype)
        h_act = torch.empty(h_shape, device=x.device, dtype=x.dtype)
        return y, h_pre, h_act

    monkeypatch.setattr(fused_decoder, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fused_decoder, "_fused_forward_triton", fake_forward)
    inputs = _inputs(requires_grad=True)

    fused_decoder.fused_multistage_symmetric_decode(*inputs)

    assert calls == [True]
