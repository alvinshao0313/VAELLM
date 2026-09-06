from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from compressed_e2e_fintuning.trainer import (
    VAEDecoderE2ETrainer,
    _resolve_e2e_pre_mlp_capture_modules,
)
from train_utils.lora_training import (
    _capture_pre_mlp_hiddens_from_modules,
    _compute_named_pre_mlp_hidden_alignment_loss,
)
from train_utils.train_args import TrainingArguments


class _TinyBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)
        self.last_output = None

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        hidden_states = self.post_attention_layernorm(hidden_states)
        output = hidden_states + 0.1 * torch.tanh(self.proj(hidden_states))
        self.last_output = output
        return (output,)


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_TinyBlock(hidden_size) for _ in range(num_layers)]
        )


class _TinyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        role: str,
        events: list[str],
        vocab_size: int = 17,
        hidden_size: int = 8,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.role = str(role)
        self.events = events
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = _TinyBackbone(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(use_cache=False, vocab_size=vocab_size)
        self.output_hidden_states_calls: list[bool] = []
        self.to_calls: list[str] = []
        self.last_logits_ref = None
        self.teacher_for_lifetime_check = None

    def to(self, *args, **kwargs):
        requested = kwargs.get("device", args[0] if args else None)
        self.to_calls.append(str(requested))
        return super().to(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_kwargs,
    ):
        del attention_mask
        self.events.append(f"{self.role}_forward")
        self.output_hidden_states_calls.append(bool(output_hidden_states))

        if self.role == "student" and self.teacher_for_lifetime_check is not None:
            teacher_logits_ref = self.teacher_for_lifetime_check.last_logits_ref
            if teacher_logits_ref is not None:
                gc.collect()
                assert teacher_logits_ref() is None

        hidden = self.embed_tokens(input_ids)
        hidden_states = [hidden] if output_hidden_states else None
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
            if hidden_states is not None:
                hidden_states.append(hidden)
        logits = self.lm_head(hidden)
        if self.role == "teacher":
            self.last_logits_ref = weakref.ref(logits)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(tuple(hidden_states) if hidden_states is not None else None),
        )


def _build_trainer(
    tmp_path,
    *,
    loss_type: str,
    hidden_loss_weight: float,
    pre_mlp_hidden_loss_weight: float = 0.0,
    hidden_layer_weighting: str = "adaptive_top_2",
    prompt_kd_weight: float = 0.0,
    teacher_output_offload: str = "cpu",
    teacher_model_offload: str = "none",
    top_k: int = 100,
):
    events: list[str] = []
    teacher = _TinyCausalLM(role="teacher", events=events)
    student = _TinyCausalLM(role="student", events=events)
    student.teacher_for_lifetime_check = teacher
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
        use_cpu=True,
    )
    trainer = VAEDecoderE2ETrainer(
        model=student,
        args=args,
        loss_type=loss_type,
        top_k=top_k,
        teacher_model=teacher,
        hidden_loss_weight=hidden_loss_weight,
        pre_mlp_hidden_loss_weight=pre_mlp_hidden_loss_weight,
        hidden_layer_weighting=hidden_layer_weighting,
        prompt_kd_weight=prompt_kd_weight,
        teacher_output_offload=teacher_output_offload,
        teacher_model_offload=teacher_model_offload,
        teacher_output_pin_memory=False,
        teacher_output_chunk_tokens=2,
    )
    trainer._teacher_device = torch.device("cpu")
    teacher.to_calls.clear()
    return trainer, student, teacher, events


def _inputs() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _assert_student_has_finite_trainable_grad(student) -> None:
    finite_grads = [
        parameter.grad
        for parameter in student.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert finite_grads
    assert all(torch.isfinite(grad).all() for grad in finite_grads)


@pytest.fixture(autouse=True)
def _patch_tiny_get_layers(monkeypatch):
    monkeypatch.setattr(
        "compressed_e2e_fintuning.teacher_targets.get_layers",
        lambda model: model.model.layers,
    )


def test_cpu_kl_teacher_before_student_and_backward(tmp_path):
    trainer, student, teacher, events = _build_trainer(
        tmp_path,
        loss_type="kl",
        hidden_loss_weight=0.0,
    )
    del teacher
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)
    assert events == ["teacher_forward", "student_forward"]
    assert trainer._active_teacher_targets is not None
    assert trainer._active_teacher_targets.logits_cpu is not None
    assert trainer._active_teacher_targets.logits_cpu.device.type == "cpu"
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()
    assert trainer._active_teacher_targets is None


def test_teacher_to_calls_have_no_cpu_target(tmp_path):
    trainer, student, teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl",
        hidden_loss_weight=0.1,
    )
    inputs = _inputs()
    teacher.to_calls.clear()
    loss = trainer.compute_loss(student, inputs)
    assert all("cpu" not in call.lower() for call in teacher.to_calls)
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()

    teacher.to_calls.clear()
    step_loss = trainer.training_step(student, inputs)
    assert torch.is_tensor(step_loss)
    assert trainer._active_teacher_targets is None
    assert all("cpu" not in call.lower() for call in teacher.to_calls)


def test_teacher_parameter_data_ptr_unchanged(tmp_path):
    trainer, student, teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl",
        hidden_loss_weight=0.1,
    )
    inputs = _inputs()
    ptrs_before = [p.data_ptr() for p in teacher.parameters()]
    loss = trainer.compute_loss(student, inputs)
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()
    ptrs_after = [p.data_ptr() for p in teacher.parameters()]
    assert ptrs_before == ptrs_after

    ptrs_before = [p.data_ptr() for p in teacher.parameters()]
    trainer.training_step(student, inputs)
    ptrs_after = [p.data_ptr() for p in teacher.parameters()]
    assert ptrs_before == ptrs_after
    assert trainer._active_teacher_targets is None


def test_cpu_kl_adaptive_top_2_hidden_collectors(tmp_path):
    trainer, student, teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl",
        hidden_loss_weight=0.1,
        hidden_layer_weighting="adaptive_top_2",
    )
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)
    assert teacher.output_hidden_states_calls == [False]
    assert student.output_hidden_states_calls == [False]
    stats = trainer._last_teacher_target_stats
    assert int(stats["hidden_layer_count"]) == 2
    assert len(stats["hidden_layer_indices"]) == 2
    assert int(stats["num_hidden_layers"]) == 4
    targets = trainer._active_teacher_targets
    assert targets is not None
    assert len(targets.hidden_layer_indices) == 2
    assert len(targets.hidden_cpu_by_layer) == 2
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_cpu_sft_hidden_zero_skips_teacher(tmp_path):
    trainer, student, _teacher, events = _build_trainer(
        tmp_path,
        loss_type="sft",
        hidden_loss_weight=0.0,
    )
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)
    assert events == ["student_forward"]
    assert trainer._active_teacher_targets is None
    assert trainer._last_teacher_target_stats == {}
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_cpu_sft_hidden_positive_teacher_first_no_logits_cache(tmp_path):
    trainer, student, _teacher, events = _build_trainer(
        tmp_path,
        loss_type="sft",
        hidden_loss_weight=0.1,
    )
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)
    assert events == ["teacher_forward", "student_forward"]
    assert trainer._last_teacher_target_stats.get("logits_device") == "none"
    targets = trainer._active_teacher_targets
    assert targets is not None
    assert targets.logits_cpu is None
    assert len(targets.hidden_cpu_by_layer) == 2
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_cpu_pre_mlp_loss_matches_cat_helper(tmp_path):
    trainer, student, teacher, events = _build_trainer(
        tmp_path,
        loss_type="kl_top",
        top_k=1000,
        hidden_loss_weight=0.0,
        pre_mlp_hidden_loss_weight=0.25,
        hidden_layer_weighting="adaptive_top_2",
        teacher_output_offload="cpu",
    )
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)

    assert events == ["teacher_forward", "student_forward"]
    assert "pre_mlp_hidden_loss" in trainer._last_loss_parts
    targets = trainer._active_teacher_targets
    assert targets is not None
    assert len(targets.pre_mlp_hidden_cpu_by_name) == 4
    assert targets.pre_mlp_reference_hidden_cpu is not None

    student.teacher_for_lifetime_check = None
    teacher_inputs = dict(inputs)
    teacher_inputs.pop("labels")
    teacher_modules = _resolve_e2e_pre_mlp_capture_modules(teacher)
    student_modules = _resolve_e2e_pre_mlp_capture_modules(student)
    with torch.no_grad():
        with _capture_pre_mlp_hiddens_from_modules(teacher_modules) as teacher_capture:
            teacher_outputs = teacher(
                **teacher_inputs,
                output_hidden_states=True,
            )
        with _capture_pre_mlp_hiddens_from_modules(student_modules) as student_capture:
            student(**teacher_inputs, output_hidden_states=False)

    expected = _compute_named_pre_mlp_hidden_alignment_loss(
        teacher_by_name={
            name: tensor.detach().clone().cpu()
            for name, tensor in teacher_capture.items()
        },
        student_by_name=dict(student_capture),
        attention_mask=inputs["attention_mask"],
        layer_weighting="adaptive_top_2",
        teacher_reference_hidden=teacher_outputs.hidden_states[0].detach().clone().cpu(),
        teacher_targets_on_cpu=True,
    )
    assert trainer._last_loss_parts["pre_mlp_hidden_loss"] == pytest.approx(
        float(expected.item()),
        rel=1e-6,
        abs=1e-7,
    )

    try:
        loss.backward()
        _assert_student_has_finite_trainable_grad(student)
    finally:
        trainer._release_active_teacher_targets()


def test_teacher_model_offload_cpu_returns_teacher_to_cpu(tmp_path):
    trainer, student, teacher, events = _build_trainer(
        tmp_path,
        loss_type="kl_top",
        top_k=1000,
        hidden_loss_weight=0.0,
        teacher_output_offload="cpu",
        teacher_model_offload="cpu",
    )
    loss = trainer.compute_loss(student, _inputs())
    assert events == ["teacher_forward", "student_forward"]
    assert teacher.to_calls
    assert teacher.to_calls[-1] == "cpu"
    assert trainer._teacher_device == torch.device("cpu")
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_cpu_kl_top_with_top_k_1000_teacher_before_student_and_backward(tmp_path):
    trainer, student, teacher, events = _build_trainer(
        tmp_path,
        loss_type="kl_top",
        top_k=1000,
        hidden_loss_weight=0.0,
    )
    del teacher
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)
    assert events == ["teacher_forward", "student_forward"]
    assert trainer._active_teacher_targets is not None
    assert trainer._active_teacher_targets.logits_cpu is not None
    assert trainer._active_teacher_targets.logits_cpu.device.type == "cpu"
    assert torch.isfinite(loss)
    try:
        loss.backward()
        _assert_student_has_finite_trainable_grad(student)
    finally:
        trainer._release_active_teacher_targets()
    assert trainer._active_teacher_targets is None


def test_return_outputs_true_returns_student_outputs(tmp_path):
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl",
        hidden_loss_weight=0.0,
    )
    inputs = _inputs()
    loss, outputs = trainer.compute_loss(student, inputs, return_outputs=True)
    assert outputs is not None
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape[-1] == student.config.vocab_size
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_training_step_clears_active_teacher_targets(tmp_path):
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl",
        hidden_loss_weight=0.1,
    )
    inputs = _inputs()
    loss = trainer.training_step(student, inputs)
    assert torch.is_tensor(loss)
    assert trainer._active_teacher_targets is None


def test_nograd_compute_loss_clears_targets(tmp_path):
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl",
        hidden_loss_weight=0.1,
    )
    inputs = _inputs()
    student.eval()
    with torch.no_grad():
        loss = trainer.compute_loss(student, inputs)
    assert torch.is_tensor(loss)
    assert trainer._active_teacher_targets is None




def _inputs_with_prompt_prefix() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def _install_counting_copy_helper(monkeypatch):
    copy_calls = {"count": 0}
    original_copy = (
        __import__(
            "compressed_e2e_fintuning.trainer",
            fromlist=["copy_detached_tensor_to_cpu"],
        ).copy_detached_tensor_to_cpu
    )

    def counting_copy(tensor, *, pin_memory):
        copy_calls["count"] += 1
        return original_copy(tensor, pin_memory=pin_memory)

    monkeypatch.setattr(
        "compressed_e2e_fintuning.trainer.copy_detached_tensor_to_cpu",
        counting_copy,
    )
    return copy_calls




CPU_OFFLOAD_DENSE_DISTILL_LOSS_TYPES = (
    "kl",
    "kl_top",
    "kd",
    "kd_top",
)


@pytest.mark.parametrize("loss_type", CPU_OFFLOAD_DENSE_DISTILL_LOSS_TYPES)
def test_cpu_offload_all_dense_distillation_losses_backward(tmp_path, loss_type):
    trainer, student, _teacher, events = _build_trainer(
        tmp_path,
        loss_type=loss_type,
        top_k=7 if loss_type.endswith("_top") else 100,
        hidden_loss_weight=0.0,
        prompt_kd_weight=0.03,
        teacher_output_offload="cpu",
    )
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)
    assert events == ["teacher_forward", "student_forward"]
    assert torch.isfinite(loss)
    try:
        loss.backward()
        _assert_student_has_finite_trainable_grad(student)
    finally:
        trainer._release_active_teacher_targets()
    assert trainer._active_teacher_targets is None


def test_cpu_kl_top_with_top_k_1000_copies_logits_once(tmp_path, monkeypatch):
    copy_calls = _install_counting_copy_helper(monkeypatch)
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl_top",
        top_k=1000,
        hidden_loss_weight=0.0,
        prompt_kd_weight=0.03,
    )
    inputs = _inputs_with_prompt_prefix()
    loss = trainer.compute_loss(student, inputs)
    target = trainer._active_teacher_targets
    assert target is not None
    assert target.logits_cpu is not None
    assert copy_calls["count"] == 1
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_cpu_kl_top_with_top_k_1000_adaptive_top_2_hidden_collectors(tmp_path):
    trainer, student, teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl_top",
        top_k=1000,
        hidden_loss_weight=0.1,
        hidden_layer_weighting="adaptive_top_2",
        prompt_kd_weight=0.03,
    )
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)
    assert teacher.output_hidden_states_calls == [False]
    assert student.output_hidden_states_calls == [False]
    stats = trainer._last_teacher_target_stats
    assert int(stats["hidden_layer_count"]) == 2
    assert len(stats["hidden_layer_indices"]) == 2
    assert int(stats["num_hidden_layers"]) == 4
    targets = trainer._active_teacher_targets
    assert targets is not None
    assert len(targets.hidden_layer_indices) == 2
    assert len(targets.hidden_cpu_by_layer) == 2
    assert targets.logits_cpu is not None
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_training_step_clears_active_teacher_targets_kl_top_with_top_k_1000(tmp_path):
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="kl_top",
        top_k=1000,
        hidden_loss_weight=0.1,
    )
    inputs = _inputs()
    loss = trainer.training_step(student, inputs)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)
    assert trainer._active_teacher_targets is None


def test_cpu_kl_top_with_top_k_1000_legacy_vs_cpu_trainer_parity(tmp_path):
    seed = 20260817

    def _seeded_trainer(output_dir, *, teacher_output_offload: str):
        torch.manual_seed(seed)
        return _build_trainer(
            output_dir,
            loss_type="kl_top",
            top_k=1000,
            hidden_loss_weight=0.0,
            prompt_kd_weight=0.03,
            teacher_output_offload=teacher_output_offload,
        )

    cpu_trainer, cpu_student, cpu_teacher, _cpu_events = _seeded_trainer(
        tmp_path / "cpu",
        teacher_output_offload="cpu",
    )
    none_trainer, none_student, none_teacher, _none_events = _seeded_trainer(
        tmp_path / "none",
        teacher_output_offload="none",
    )
    for left, right in zip(
        cpu_student.parameters(), none_student.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right)
    for left, right in zip(
        cpu_teacher.parameters(), none_teacher.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right)

    inputs = _inputs()
    try:
        cpu_loss = cpu_trainer.compute_loss(cpu_student, inputs)
        none_loss = none_trainer.compute_loss(none_student, inputs)
        torch.testing.assert_close(cpu_loss, none_loss, atol=2e-6, rtol=2e-5)
        cpu_loss.backward()
        none_loss.backward()
        assert cpu_student.lm_head.weight.grad is not None
        assert none_student.lm_head.weight.grad is not None
        torch.testing.assert_close(
            cpu_student.lm_head.weight.grad,
            none_student.lm_head.weight.grad,
            atol=3e-6,
            rtol=3e-5,
        )
    finally:
        cpu_trainer._release_active_teacher_targets()
