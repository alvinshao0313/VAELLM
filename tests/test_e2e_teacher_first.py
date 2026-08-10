from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
from train_utils.distill_losses import compute_teacher_entropy_mean_and_gamma
from train_utils.train_args import TrainingArguments


class _TinyBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)
        self.last_output = None

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
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
    hidden_layer_weighting: str = "adaptive_top_2",
    prompt_kd_weight: float = 0.0,
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
        teacher_model=teacher,
        hidden_loss_weight=hidden_loss_weight,
        hidden_layer_weighting=hidden_layer_weighting,
        prompt_kd_weight=prompt_kd_weight,
        eakld_confidence_k=16,
        teacher_output_offload="cpu",
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


@pytest.fixture(autouse=True)
def _patch_tiny_get_layers(monkeypatch):
    monkeypatch.setattr(
        "compressed_e2e_fintuning.teacher_targets.get_layers",
        lambda model: model.model.layers,
    )


def test_cpu_eakld_teacher_before_student_and_backward(tmp_path):
    trainer, student, teacher, events = _build_trainer(
        tmp_path,
        loss_type="eakld",
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
        loss_type="eakld",
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
        loss_type="eakld",
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


def test_cpu_eakld_adaptive_top_2_hidden_collectors(tmp_path):
    trainer, student, teacher, _events = _build_trainer(
        tmp_path,
        loss_type="eakld",
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
    assert targets.eakld_gamma_cpu is None
    assert len(targets.hidden_cpu_by_layer) == 2
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_unsupported_kl_raises_before_student_forward(tmp_path):
    trainer, student, _teacher, events = _build_trainer(
        tmp_path,
        loss_type="kl",
        hidden_loss_weight=0.0,
    )
    inputs = _inputs()
    with pytest.raises(
        ValueError,
        match=(
            "teacher_output_offload=cpu supports only sft/origin hidden alignment "
            "and EAKLD-family losses."
        ),
    ):
        trainer.compute_loss(student, inputs)
    assert events == []
    assert trainer._active_teacher_targets is None


def test_return_outputs_true_returns_student_outputs(tmp_path):
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="eakld",
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
        loss_type="eakld",
        hidden_loss_weight=0.1,
    )
    inputs = _inputs()
    loss = trainer.training_step(student, inputs)
    assert torch.is_tensor(loss)
    assert trainer._active_teacher_targets is None


def test_nograd_compute_loss_clears_targets(tmp_path):
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="eakld",
        hidden_loss_weight=0.1,
    )
    inputs = _inputs()
    student.eval()
    with torch.no_grad():
        loss = trainer.compute_loss(student, inputs)
    assert torch.is_tensor(loss)
    assert trainer._active_teacher_targets is None


def test_cpu_eakld_teacher_targets_include_entropy_scalars(tmp_path, monkeypatch):
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
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="eakld",
        hidden_loss_weight=0.0,
    )
    inputs = _inputs()
    loss = trainer.compute_loss(student, inputs)
    target = trainer._active_teacher_targets
    assert target is not None
    assert target.teacher_entropy_mean_cpu.device.type == "cpu"
    assert target.teacher_entropy_mean_cpu.ndim == 0
    assert target.teacher_entropy_mean_cpu.dtype == torch.float32
    assert target.teacher_valid_token_count_cpu.ndim == 0
    assert target.teacher_valid_token_count_cpu.device.type == "cpu"
    assert copy_calls["count"] == 1
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_eakld_telemetry_weighted_accumulator_and_reset(tmp_path):
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="eakld",
        hidden_loss_weight=0.0,
    )
    del student, _teacher, _events

    batch_a = {
        "teacher_entropy_mean": torch.tensor(2.0),
        "gamma_reverse": torch.tensor(0.0),
        "lambda_forward": torch.tensor(1.0),
        "forward_kl": torch.tensor(0.4),
        "reverse_kl": torch.tensor(0.1),
        "eakld_total": torch.tensor(0.4),
        "valid_tokens": torch.tensor(10.0),
    }
    batch_b = {
        "teacher_entropy_mean": torch.tensor(0.5),
        "gamma_reverse": torch.tensor(1.0),
        "lambda_forward": torch.tensor(0.0),
        "forward_kl": torch.tensor(0.2),
        "reverse_kl": torch.tensor(0.8),
        "eakld_total": torch.tensor(0.8),
        "valid_tokens": torch.tensor(30.0),
    }
    trainer._record_eakld_telemetry(batch_a)
    trainer._record_eakld_telemetry(batch_b)

    flushed = {}
    trainer.log(flushed)
    assert flushed["eakld/gamma_reverse_mean"] == pytest.approx(0.75)
    assert flushed["eakld/gamma_reverse_zero_fraction"] == pytest.approx(0.25)
    assert flushed["eakld/gamma_reverse_one_fraction"] == pytest.approx(0.75)
    assert trainer._eakld_telemetry_weight == pytest.approx(0.0)
    assert trainer._eakld_telemetry_weighted_sums == {}
    assert trainer._eakld_gamma_zero_weight == pytest.approx(0.0)
    assert trainer._eakld_gamma_one_weight == pytest.approx(0.0)

    second = {}
    trainer.log(second)
    assert "eakld/gamma_reverse_mean" not in second


def _inputs_with_prompt_prefix() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def _install_counting_entropy_helper(monkeypatch):
    calls: list[dict] = []

    def counting_helper(teacher_logits, mask, *, confidence_k):
        calls.append(
            {
                "logits_id": id(teacher_logits),
                "mask": (mask.detach().clone() if torch.is_tensor(mask) else mask),
                "confidence_k": int(confidence_k),
            }
        )
        return compute_teacher_entropy_mean_and_gamma(
            teacher_logits, mask, confidence_k=int(confidence_k)
        )

    monkeypatch.setattr(
        "compressed_e2e_fintuning.trainer.compute_teacher_entropy_mean_and_gamma",
        counting_helper,
    )
    return calls


def test_cpu_teacher_targets_zero_prompt_weight_response_only(tmp_path, monkeypatch):
    calls = _install_counting_entropy_helper(monkeypatch)
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="eakld",
        hidden_loss_weight=0.0,
        prompt_kd_weight=0.0,
    )
    inputs = _inputs_with_prompt_prefix()
    loss = trainer.compute_loss(student, inputs)
    target = trainer._active_teacher_targets
    assert target is not None
    # Response scalars populated.
    assert target.eakld_gamma_cpu is not None
    assert target.teacher_entropy_mean_cpu is not None
    assert target.teacher_valid_token_count_cpu is not None
    assert target.teacher_entropy_mean_cpu.device.type == "cpu"
    assert target.teacher_entropy_mean_cpu.dtype == torch.float32
    assert target.teacher_valid_token_count_cpu.ndim == 0
    # Prompt scalars absent.
    assert target.eakld_prompt_gamma_cpu is None
    assert target.teacher_prompt_entropy_mean_cpu is None
    assert target.teacher_prompt_valid_token_count_cpu is None
    # Entropy/gamma helper called exactly once (response region only).
    assert len(calls) == 1
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()


def test_cpu_teacher_targets_positive_prompt_weight_both_regions(tmp_path, monkeypatch):
    calls = _install_counting_entropy_helper(monkeypatch)
    trainer, student, _teacher, _events = _build_trainer(
        tmp_path,
        loss_type="eakld",
        hidden_loss_weight=0.0,
        prompt_kd_weight=0.5,
    )
    inputs = _inputs_with_prompt_prefix()
    loss = trainer.compute_loss(student, inputs)
    target = trainer._active_teacher_targets
    assert target is not None
    # Both scalar sets populated.
    assert target.eakld_gamma_cpu is not None
    assert target.teacher_entropy_mean_cpu is not None
    assert target.teacher_valid_token_count_cpu is not None
    assert target.eakld_prompt_gamma_cpu is not None
    assert target.teacher_prompt_entropy_mean_cpu is not None
    assert target.teacher_prompt_valid_token_count_cpu is not None
    for scalar in (
        target.eakld_prompt_gamma_cpu,
        target.teacher_prompt_entropy_mean_cpu,
        target.teacher_prompt_valid_token_count_cpu,
    ):
        assert scalar.device.type == "cpu"
        assert scalar.dtype == torch.float32
        assert scalar.ndim == 0
    # Helper called exactly twice with distinct binary masks.
    assert len(calls) == 2
    mask_a = calls[0]["mask"]
    mask_b = calls[1]["mask"]
    assert torch.is_tensor(mask_a) and torch.is_tensor(mask_b)
    assert not torch.equal(mask_a, mask_b)
    # Each valid count equals its region-mask sum.
    assert float(target.teacher_valid_token_count_cpu) == pytest.approx(
        float(mask_a.sum())
    )
    assert float(target.teacher_prompt_valid_token_count_cpu) == pytest.approx(
        float(mask_b.sum())
    )
    # Both calls reused the same single CPU copy of teacher logits.
    assert calls[0]["logits_id"] == calls[1]["logits_id"]
    try:
        loss.backward()
    finally:
        trainer._release_active_teacher_targets()
