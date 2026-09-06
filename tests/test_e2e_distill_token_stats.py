"""Task 8: E2EDistillTokenStatsCallback + VAEDecoderE2ETrainer token-stats wiring.

Mirrors tests/test_lora_distill_token_stats_callback.py for the E2E trainer:
- boundary / window_start_step / reduce-before-rank0 / resume semantics
- compute_loss updates telemetry exactly once before dense/CPU dispatch
- CPU-offload path produces the same totals as the dense path for identical batches
- teacher-target construction (_build_cpu_teacher_targets) adds zero extra counts
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from compressed_e2e_fintuning.trainer import (
    E2EDistillTokenStatsCallback,
    VAEDecoderE2ETrainer,
)
from train_utils.distill_token_stats import DistillTokenStatsAccumulator
from train_utils.train_args import TrainingArguments


# --- callback contract fakes (mirror LoRA test) ---


class _FakeAccelerator:
    def __init__(self, device, *, reduced=None):
        self.device = device
        self._reduced = reduced
        self.reduce_calls = []

    def reduce(self, tensor, reduction="sum"):
        self.reduce_calls.append((tensor.clone(), reduction))
        if self._reduced is not None:
            return self._reduced.to(device=tensor.device, dtype=tensor.dtype)
        return tensor.clone()


class _FakeArgs:
    def __init__(self, logging_steps):
        self.logging_steps = logging_steps


class _FakeState:
    def __init__(self, global_step, logging_steps, is_world_process_zero=True):
        self.global_step = global_step
        self.logging_steps = logging_steps
        self.is_world_process_zero = is_world_process_zero


class _FakeControl:
    pass


class _FakeTrainer:
    def __init__(self, accelerator, *, stats=None):
        self.accelerator = accelerator
        self.distill_token_stats = stats if stats is not None else DistillTokenStatsAccumulator()


def _make_logger_with_file_handler(tmp_path):
    logger = logging.getLogger(f"e2e_token_stats_test_{id(tmp_path)}")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    log_path = os.path.join(str(tmp_path), "token_stats.log")
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, log_path


def _read_log(log_path):
    with open(log_path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle.readlines()]


def _step_labels():
    # one sample: 3 prompt tokens, 3 response tokens
    labels = torch.tensor([[-100, -100, -100, 1, 2, 3]])
    attention = torch.ones_like(labels)
    return labels, attention


def _run_steps(callback, args, state, control, *, start, end, update_each_step=True):
    for step in range(start, end + 1):
        state.global_step = step
        if update_each_step:
            labels, attention = _step_labels()
            callback._trainer.distill_token_stats.update(labels, attention)
        callback.on_step_end(args, state, control)


def test_window_one_to_ten_no_consume_until_step_ten(tmp_path):
    accelerator = _FakeAccelerator(torch.device("cpu"))
    trainer = _FakeTrainer(accelerator)
    logger, log_path = _make_logger_with_file_handler(tmp_path)
    callback = E2EDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=10)
    state = _FakeState(global_step=0, logging_steps=10, is_world_process_zero=True)
    control = _FakeControl()

    _run_steps(callback, args, state, control, start=1, end=9)

    assert len(accelerator.reduce_calls) == 0
    assert _read_log(log_path) == []
    # accumulator not reset at step 1 special loss-log boundary
    assert trainer.distill_token_stats._accumulator is not None

    state.global_step = 10
    labels, attention = _step_labels()
    trainer.distill_token_stats.update(labels, attention)
    callback.on_step_end(args, state, control)

    assert len(accelerator.reduce_calls) == 1
    lines = _read_log(log_path)
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("E2E token stats:")
    assert "step=10" in line
    assert "window_optimizer_steps=10" in line
    assert "avg_prompt_tokens=3.0000" in line
    assert "avg_response_tokens=3.0000" in line
    assert "global_samples=10" in line


def test_cadence_resolution_uses_state_logging_steps_not_raw_args(tmp_path):
    accelerator = _FakeAccelerator(torch.device("cpu"))
    trainer = _FakeTrainer(accelerator)
    logger, log_path = _make_logger_with_file_handler(tmp_path)
    callback = E2EDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=0.1)
    state = _FakeState(global_step=0, logging_steps=10, is_world_process_zero=True)
    control = _FakeControl()

    _run_steps(callback, args, state, control, start=1, end=9)
    assert len(accelerator.reduce_calls) == 0

    state.global_step = 10
    labels, attention = _step_labels()
    trainer.distill_token_stats.update(labels, attention)
    callback.on_step_end(args, state, control)

    assert len(accelerator.reduce_calls) == 1
    lines = _read_log(log_path)
    assert len(lines) == 1
    assert "window_optimizer_steps=10" in lines[0]


def test_second_window_reports_only_steps_eleven_to_twenty(tmp_path):
    accelerator = _FakeAccelerator(torch.device("cpu"))
    trainer = _FakeTrainer(accelerator)
    logger, log_path = _make_logger_with_file_handler(tmp_path)
    callback = E2EDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=10)
    state = _FakeState(global_step=0, logging_steps=10, is_world_process_zero=True)
    control = _FakeControl()

    _run_steps(callback, args, state, control, start=1, end=10)
    first_lines = _read_log(log_path)
    assert len(first_lines) == 1
    assert "window_optimizer_steps=10" in first_lines[0]
    assert "global_samples=10" in first_lines[0]

    _run_steps(callback, args, state, control, start=11, end=20)
    all_lines = _read_log(log_path)
    assert len(all_lines) == 2
    second_line = all_lines[1]
    assert "step=20" in second_line
    assert "window_optimizer_steps=10" in second_line
    assert "global_samples=10" in second_line


def test_resume_from_non_boundary_reports_partial_first_window(tmp_path):
    accelerator = _FakeAccelerator(torch.device("cpu"))
    trainer = _FakeTrainer(accelerator)
    logger, log_path = _make_logger_with_file_handler(tmp_path)
    callback = E2EDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=10)
    state = _FakeState(global_step=8, logging_steps=10, is_world_process_zero=True)
    control = _FakeControl()

    _run_steps(callback, args, state, control, start=8, end=9)
    assert len(accelerator.reduce_calls) == 0

    state.global_step = 10
    labels, attention = _step_labels()
    trainer.distill_token_stats.update(labels, attention)
    callback.on_step_end(args, state, control)

    assert len(accelerator.reduce_calls) == 1
    lines = _read_log(log_path)
    assert len(lines) == 1
    assert "window_optimizer_steps=3" in lines[0]
    assert "global_samples=3" in lines[0]


def test_nonzero_rank_invokes_consume_but_does_not_write(tmp_path):
    accelerator = _FakeAccelerator(torch.device("cpu"))
    trainer = _FakeTrainer(accelerator)
    logger, log_path = _make_logger_with_file_handler(tmp_path)
    callback = E2EDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=10)
    state = _FakeState(global_step=0, logging_steps=10, is_world_process_zero=False)
    control = _FakeControl()

    _run_steps(callback, args, state, control, start=1, end=10)

    assert len(accelerator.reduce_calls) == 1
    assert _read_log(log_path) == []


# --- compute_loss wiring through real VAEDecoderE2ETrainer ---


class _FakeOutput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


class _TinyBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        return (hidden_states + 0.1 * torch.tanh(self.proj(hidden_states)),)


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyBlock(hidden_size) for _ in range(num_layers)])


class _TinyCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 17, hidden_size: int = 8, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = _TinyBackbone(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(use_cache=False, vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, **_kwargs):
        del attention_mask
        hidden = self.embed_tokens(input_ids)
        hidden_states = [hidden] if output_hidden_states else None
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
            if hidden_states is not None:
                hidden_states.append(hidden)
        logits = self.lm_head(hidden)
        return _FakeOutput(logits=logits, hidden_states=None)


@pytest.fixture(autouse=True)
def _patch_tiny_get_layers(monkeypatch):
    monkeypatch.setattr(
        "compressed_e2e_fintuning.teacher_targets.get_layers",
        lambda model: model.model.layers,
    )


def _build_e2e_trainer(
    tmp_path,
    *,
    teacher_output_offload,
    prompt_kd_weight=0.0,
    loss_type="kl",
):
    teacher = _TinyCausalLM()
    student = _TinyCausalLM()
    with torch.no_grad():
        student.lm_head.weight.add_(0.05)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
        use_cpu=True,
        logging_steps=1,
    )
    trainer = VAEDecoderE2ETrainer(
        model=student,
        args=args,
        loss_type=loss_type,
        teacher_model=teacher,
        distill_temperature=1.0,
        hidden_loss_weight=0.0,
        prompt_kd_weight=float(prompt_kd_weight),
        hidden_layer_weighting="uniform",
        teacher_output_offload=teacher_output_offload,
        teacher_output_pin_memory=False,
        teacher_output_chunk_tokens=8,
    )
    trainer._teacher_device = torch.device("cpu")
    return trainer, student, teacher


def _e2e_inputs(*, seq_len: int = 8):
    assert seq_len <= 128
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def test_compute_loss_dense_updates_token_stats_once(tmp_path):
    trainer, student, _teacher = _build_e2e_trainer(
        tmp_path, teacher_output_offload="none"
    )
    inputs = _e2e_inputs(seq_len=8)
    student.train()
    trainer.compute_loss(student, inputs)

    stats = trainer.distill_token_stats.consume_global(
        _FakeAccelerator(torch.device("cpu"))
    )
    assert stats is not None
    # 2 prompt tokens (-100), 6 response tokens, 1 sample
    assert stats.global_samples == 1
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(2.0)
    assert stats.avg_response_tokens_per_sample == pytest.approx(6.0)


def test_compute_loss_cpu_offload_matches_dense_totals(tmp_path):
    dense_trainer, dense_student, _dense_teacher = _build_e2e_trainer(
        tmp_path, teacher_output_offload="none"
    )
    cpu_trainer, cpu_student, _cpu_teacher = _build_e2e_trainer(
        tmp_path, teacher_output_offload="cpu"
    )
    inputs = _e2e_inputs(seq_len=8)

    dense_student.train()
    dense_trainer.compute_loss(dense_student, inputs)
    cpu_student.train()
    cpu_trainer.compute_loss(cpu_student, inputs)

    accelerator = _FakeAccelerator(torch.device("cpu"))
    dense_stats = dense_trainer.distill_token_stats.consume_global(accelerator)
    cpu_stats = cpu_trainer.distill_token_stats.consume_global(accelerator)

    assert dense_stats is not None
    assert cpu_stats is not None
    assert cpu_stats.global_samples == dense_stats.global_samples
    assert cpu_stats.avg_prompt_tokens_per_sample == pytest.approx(
        dense_stats.avg_prompt_tokens_per_sample
    )
    assert cpu_stats.avg_response_tokens_per_sample == pytest.approx(
        dense_stats.avg_response_tokens_per_sample
    )


def test_compute_loss_teacher_target_construction_adds_zero_extra(tmp_path, monkeypatch):
    """_build_cpu_teacher_targets must not double-count token telemetry."""
    trainer, student, _teacher = _build_e2e_trainer(
        tmp_path, teacher_output_offload="cpu"
    )
    inputs = _e2e_inputs(seq_len=8)

    update_calls = {"n": 0}
    original_update = trainer.distill_token_stats.update

    def counting_update(*args, **kwargs):
        update_calls["n"] += 1
        return original_update(*args, **kwargs)

    monkeypatch.setattr(trainer.distill_token_stats, "update", counting_update)

    student.train()
    trainer.compute_loss(student, inputs)

    # Exactly one update from the top-level compute_loss; teacher-target
    # construction must not call update.
    assert update_calls["n"] == 1


def test_compute_loss_does_not_update_when_model_not_training(tmp_path):
    trainer, student, _teacher = _build_e2e_trainer(
        tmp_path, teacher_output_offload="none"
    )
    inputs = _e2e_inputs(seq_len=8)
    student.eval()
    trainer.compute_loss(student, inputs)

    assert trainer.distill_token_stats._accumulator is None


def test_dense_smoke_cadence_ten_one_line_at_step_ten(tmp_path):
    """With logging cadence 10, dense path emits one token line at step 10
    covering steps 1-10 and no token line at the special step-1 loss log."""
    trainer, student, _teacher = _build_e2e_trainer(
        tmp_path, teacher_output_offload="none"
    )
    trainer.args.logging_steps = 10
    trainer.args.logging_first_step = True

    logger, log_path = _make_logger_with_file_handler(tmp_path)
    callback = E2EDistillTokenStatsCallback(trainer=trainer, logger=logger)
    args = _FakeArgs(logging_steps=10)
    state = _FakeState(global_step=0, logging_steps=10, is_world_process_zero=True)
    control = _FakeControl()

    inputs = _e2e_inputs(seq_len=8)
    student.train()
    for step in range(1, 11):
        state.global_step = step
        trainer.compute_loss(student, inputs)
        # Simulate the HF loss-log event: at step 1 (logging_first_step) the
        # E2ETrainerLogCallback writes a loss line, but the token-stats
        # callback must NOT emit at step 1.
        callback.on_step_end(args, state, control)

    lines = _read_log(log_path)
    assert len(lines) == 1
