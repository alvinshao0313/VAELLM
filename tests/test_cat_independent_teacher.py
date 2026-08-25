from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train_utils.cat_after_category_distill import _build_distill_stage_meta
from train_utils.distill_teacher import (
    DistillTeacherRuntime,
    resolve_distill_teacher_dtype,
    resolve_distill_teacher_required,
)
from train_utils.lora_training import CustomSFTTrainer


class TrackingTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.linear = nn.Linear(2, 2)
        self.to_calls = []

    def to(self, *args, **kwargs):
        if args:
            self.to_calls.append(str(args[0]))
        elif "device" in kwargs:
            self.to_calls.append(str(kwargs["device"]))
        elif "dtype" in kwargs:
            self.to_calls.append(str(kwargs["dtype"]))
        return super().to(*args, **kwargs)


def _frozen_eval_teacher() -> TrackingTeacher:
    model = TrackingTeacher()
    model.requires_grad_(False)
    model.eval()
    return model


def test_distill_teacher_runtime_is_lazy_and_reuses_loaded_model(monkeypatch):
    calls = []
    teacher = _frozen_eval_teacher()

    def fake_loader(model_path, *, access_token, device, dtype):
        calls.append((model_path, access_token, str(device), dtype))
        return teacher

    monkeypatch.setattr("train_utils.distill_teacher.load_frozen_base_reference_model", fake_loader)

    runtime = DistillTeacherRuntime(
        model_path="base",
        access_token="token",
        forward_device="cpu",
        dtype=torch.bfloat16,
        model_offload="none",
        logger=None,
    )

    assert runtime.is_loaded is False
    assert calls == []

    first = runtime.get_or_load()
    second = runtime.prepare_for_forward()
    runtime.finish_forward()

    assert first is teacher
    assert second is teacher
    assert runtime.is_loaded is True
    assert calls == [("base", "token", "cpu", torch.bfloat16)]
    assert teacher.to_calls == []
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_distill_teacher_runtime_cpu_offload_moves_same_object(monkeypatch):
    calls = []
    teacher = _frozen_eval_teacher()

    def fake_loader(model_path, *, access_token, device, dtype):
        calls.append((model_path, access_token, str(device), dtype))
        return teacher

    monkeypatch.setattr("train_utils.distill_teacher.load_frozen_base_reference_model", fake_loader)

    runtime = DistillTeacherRuntime(
        model_path="base",
        access_token=None,
        forward_device="cpu",
        dtype=torch.float32,
        model_offload="cpu",
        logger=None,
    )

    prepared = runtime.prepare_for_forward()
    first_id = id(prepared)
    runtime.finish_forward()
    prepared_again = runtime.prepare_for_forward()

    assert id(prepared_again) == first_id
    assert calls == [("base", None, "cpu", torch.float32)]
    assert teacher.to_calls == ["cpu", "cpu", "cpu"]


def test_resolve_distill_teacher_dtype_prefers_training_precision_flags():
    student = nn.Linear(2, 2).to(dtype=torch.float64)

    assert resolve_distill_teacher_dtype(SimpleNamespace(bf16=True, fp16=True), student) is torch.bfloat16
    assert resolve_distill_teacher_dtype(SimpleNamespace(bf16=False, fp16=True), student) is torch.float16
    assert resolve_distill_teacher_dtype(SimpleNamespace(bf16=False, fp16=False), student) is torch.float64


def test_resolve_distill_teacher_dtype_defaults_to_float32_without_floating_params():
    class IntegerOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.ones(2, dtype=torch.int64), requires_grad=False)

    assert (
        resolve_distill_teacher_dtype(SimpleNamespace(bf16=False, fp16=False), IntegerOnly())
        is torch.float32
    )


@pytest.mark.parametrize(
    "loss_type,hidden,pre_mlp,expected",
    [
        ("sft", 0.0, 0.0, False),
        ("origin", 0.0, 0.0, False),
        ("none", 0.0, 0.0, False),
        ("sft", 0.1, 0.0, True),
        ("origin", 0.0, 0.01, True),
        ("kl", 0.0, 0.0, True),
        ("kl_top_100", 0.0, 0.0, True),
        ("kd", 0.0, 0.0, True),
        ("eakld", 0.0, 0.0, True),
    ],
)
def test_resolve_distill_teacher_required_matrix(loss_type, hidden, pre_mlp, expected):
    assert (
        resolve_distill_teacher_required(
            loss_type=loss_type,
            hidden_loss_weight=hidden,
            pre_mlp_hidden_loss_weight=pre_mlp,
        )
        is expected
    )


def _build_trainer_for_teacher_required(monkeypatch, *, loss_type, hidden, pre_mlp, teacher_runtime):
    monkeypatch.setattr("train_utils.lora_training.SFTTrainer.__init__", lambda self, *args, **kwargs: None)
    return CustomSFTTrainer(
        loss_type=loss_type,
        hidden_loss_weight=hidden,
        pre_mlp_hidden_loss_weight=pre_mlp,
        eakld_confidence_k=16,
        teacher_runtime=teacher_runtime,
    )


@pytest.mark.parametrize(
    "loss_type,hidden,pre_mlp",
    [
        ("sft", 0.0, 0.0),
        ("origin", 0.0, 0.0),
        ("kl", 0.0, 0.0),
        ("sft", 0.1, 0.0),
    ],
)
def test_distill_stage_meta_teacher_required_matches_trainer(monkeypatch, loss_type, hidden, pre_mlp):
    teacher_needed = resolve_distill_teacher_required(
        loss_type=loss_type,
        hidden_loss_weight=hidden,
        pre_mlp_hidden_loss_weight=pre_mlp,
    )
    trainer = _build_trainer_for_teacher_required(
        monkeypatch,
        loss_type=loss_type,
        hidden=hidden,
        pre_mlp=pre_mlp,
        teacher_runtime=(object() if teacher_needed else None),
    )
    cfg = SimpleNamespace(
        lr=1e-4,
        weight_decay=0.0,
        loss_type=loss_type,
        hidden_loss_weight=hidden,
        pre_mlp_hidden_loss_weight=pre_mlp,
    )

    meta = _build_distill_stage_meta(
        mode="remaining_lora",
        category="q_proj",
        did_train=False,
        newly_compressed_target_count=0,
        remaining_lora_target_count=1,
        decoder_target_count=0,
        cfg=cfg,
        training_args=SimpleNamespace(distill_teacher_model_offload="none"),
    )

    assert trainer.teacher_required is teacher_needed
    assert meta["teacher_required"] is trainer.teacher_required


def test_custom_sft_trainer_rejects_missing_runtime_when_teacher_required(monkeypatch):
    monkeypatch.setattr("train_utils.lora_training.SFTTrainer.__init__", lambda self, *args, **kwargs: None)

    with pytest.raises(ValueError, match="teacher_runtime"):
        CustomSFTTrainer(loss_type="kl", teacher_runtime=None)


def test_pure_sft_compute_loss_does_not_load_lazy_teacher(monkeypatch):
    monkeypatch.setattr("train_utils.lora_training.SFTTrainer.__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        "train_utils.lora_training.SFTTrainer.compute_loss",
        lambda self, model, inputs, return_outputs=False, **_kwargs: torch.tensor(0.0),
    )

    def fail_loader(*_args, **_kwargs):
        raise AssertionError("pure SFT must not materialize teacher")

    monkeypatch.setattr("train_utils.distill_teacher.load_frozen_base_reference_model", fail_loader)
    runtime = DistillTeacherRuntime(
        model_path="base",
        access_token=None,
        forward_device="cpu",
        dtype=torch.float32,
        model_offload="none",
        logger=None,
    )
    trainer = CustomSFTTrainer(
        loss_type="sft",
        hidden_loss_weight=0.0,
        pre_mlp_hidden_loss_weight=0.0,
        teacher_runtime=runtime,
    )
    trainer.args = SimpleNamespace(bf16=False, fp16=False)
    model = nn.Linear(2, 2)
    model.eval()

    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "labels": torch.ones(1, 1, dtype=torch.long),
        },
    )

    assert torch.equal(loss, torch.tensor(0.0))
    assert runtime.is_loaded is False
