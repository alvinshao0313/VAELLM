from types import SimpleNamespace

import torch
from torch import nn


class _TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.config = SimpleNamespace(use_cache=True)


def test_e2e_dp_teacher_uses_distributed_loader(monkeypatch):
    import compressed_e2e_fintuning.runtime as runtime

    teacher = _TinyTeacher()
    calls = []

    def fake_distributed_loader(model_path, hf_args, *, device, logger=None):
        calls.append((model_path, hf_args, str(device), logger))
        return teacher

    def fail_regular_loader(*_args, **_kwargs):
        raise AssertionError("DP GPU teacher must not use the per-rank checkpoint loader")

    monkeypatch.setattr(runtime, "load_frozen_base_reference_model_distributed_from_hf_args", fake_distributed_loader)
    monkeypatch.setattr(runtime, "get_model", fail_regular_loader)

    args = SimpleNamespace(
        loss_type="kl_top_100",
        hidden_loss_weight=0.0,
        pre_mlp_hidden_loss_weight=0.0,
        teacher_model_path="teacher-path",
        teacher_model_offload="none",
    )
    hf_args = SimpleNamespace(access_token=None)
    log = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    loaded, source = runtime._load_teacher(
        args=args,
        hf_args=hf_args,
        base_model_path="base-path",
        log=log,
        device=torch.device("cuda:0"),
    )

    assert loaded is teacher
    assert source == "external_teacher"
    assert calls == [("teacher-path", hf_args, "cuda:0", log)]


def test_cat_gpu_teacher_uses_distributed_loader(monkeypatch):
    import train_utils.distill_teacher as distill_teacher

    teacher = _TinyTeacher()
    calls = []

    def fake_distributed_loader(model_path, *, access_token, device, dtype, logger=None):
        calls.append((model_path, access_token, str(device), dtype, logger))
        teacher.requires_grad_(False)
        teacher.eval()
        teacher.config.use_cache = False
        return teacher

    def fail_regular_loader(*_args, **_kwargs):
        raise AssertionError("GPU teacher must not use the per-rank checkpoint loader")

    monkeypatch.setattr(distill_teacher, "load_frozen_base_reference_model_distributed", fake_distributed_loader)
    monkeypatch.setattr(distill_teacher, "load_frozen_base_reference_model", fail_regular_loader)

    runtime = distill_teacher.DistillTeacherRuntime(
        model_path="teacher-path",
        access_token=None,
        forward_device="cuda:0",
        dtype=torch.bfloat16,
        model_offload="none",
        logger=None,
    )

    loaded = runtime.get_or_load()

    assert loaded is teacher
    assert calls == [("teacher-path", None, "cuda:0", torch.bfloat16, None)]
    assert loaded.training is False
    assert loaded.config.use_cache is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
