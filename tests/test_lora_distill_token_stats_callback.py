"""Task 7: _LoraDistillTokenStatsCallback + CustomSFTTrainer token-stats wiring."""

from __future__ import annotations

import logging
import copy
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train_utils.distill_token_stats import DistillTokenStatsAccumulator
from train_utils.lora_training import CustomSFTTrainer
from train_utils.lora_utils import _LoraDistillTokenStatsCallback


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
    logger = logging.getLogger(f"lora_token_stats_test_{id(tmp_path)}")
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
    callback = _LoraDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=10)
    state = _FakeState(global_step=0, logging_steps=10, is_world_process_zero=True)
    control = _FakeControl()

    _run_steps(callback, args, state, control, start=1, end=9)

    assert len(accelerator.reduce_calls) == 0
    assert _read_log(log_path) == []
    # accumulator not reset at step 1 special log boundary
    assert trainer.distill_token_stats._accumulator is not None

    state.global_step = 10
    labels, attention = _step_labels()
    trainer.distill_token_stats.update(labels, attention)
    callback.on_step_end(args, state, control)

    assert len(accelerator.reduce_calls) == 1
    lines = _read_log(log_path)
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("LoRA token stats:")
    assert "step=10" in line
    assert "window_optimizer_steps=10" in line
    assert "avg_prompt_tokens=3.0000" in line
    assert "avg_response_tokens=3.0000" in line
    assert "global_samples=10" in line


def test_cadence_resolution_uses_state_logging_steps_not_raw_args(tmp_path):
    accelerator = _FakeAccelerator(torch.device("cpu"))
    trainer = _FakeTrainer(accelerator)
    logger, log_path = _make_logger_with_file_handler(tmp_path)
    callback = _LoraDistillTokenStatsCallback(trainer=trainer, logger=logger)

    # raw args.logging_steps is a non-integer ratio, but state.logging_steps=10
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
    callback = _LoraDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=10)
    state = _FakeState(global_step=0, logging_steps=10, is_world_process_zero=True)
    control = _FakeControl()

    # first window: steps 1-10
    _run_steps(callback, args, state, control, start=1, end=10)
    first_lines = _read_log(log_path)
    assert len(first_lines) == 1
    assert "window_optimizer_steps=10" in first_lines[0]
    assert "global_samples=10" in first_lines[0]

    # second window: steps 11-20
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
    callback = _LoraDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=10)
    # resumed from step 7; first observed on_step_end is step 8
    state = _FakeState(global_step=8, logging_steps=10, is_world_process_zero=True)
    control = _FakeControl()

    # steps 8, 9: no boundary
    _run_steps(callback, args, state, control, start=8, end=9)
    assert len(accelerator.reduce_calls) == 0

    # step 10: boundary; window covers steps 8-10 = 3 optimizer steps
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
    callback = _LoraDistillTokenStatsCallback(trainer=trainer, logger=logger)

    args = _FakeArgs(logging_steps=10)
    state = _FakeState(global_step=0, logging_steps=10, is_world_process_zero=False)
    control = _FakeControl()

    _run_steps(callback, args, state, control, start=1, end=10)

    # nonzero rank still participated in the collective
    assert len(accelerator.reduce_calls) == 1
    # but wrote no line
    assert _read_log(log_path) == []


# --- compute_loss update wiring (gradient-accumulation factor 2) ---


class _FakeOutput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


class _KlFakeCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 11, hidden_size: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(use_cache=False, vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, **_kwargs):
        del attention_mask
        hidden = self.embed_tokens(input_ids)
        logits = self.lm_head(hidden)
        loss = logits.float().pow(2).mean()
        return _FakeOutput(loss=loss, logits=logits, hidden_states=None)


class _StaticTeacherRuntime:
    model_offload = "none"

    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.model.eval()

    def prepare_for_forward(self):
        return self.model

    def finish_forward(self):
        return None


def _build_kl_trainer(model: _KlFakeCausalLM) -> CustomSFTTrainer:
    trainer = CustomSFTTrainer.__new__(CustomSFTTrainer)
    trainer.args = SimpleNamespace(bf16=False, fp16=False)
    trainer.loss_type = "kl"
    trainer.temperature = 1.0
    trainer.loss_alpha = 0.5
    trainer.hidden_loss_weight = 0.0
    trainer.pre_mlp_hidden_loss_weight = 0.0
    trainer.prompt_kd_weight = 0.0
    trainer.hidden_alignment_layer_weighting = "uniform"
    trainer.teacher_logits_cpu_staging = False
    trainer.selective_student_topk = False
    trainer.selective_student_topk_chunk_rows = 32
    trainer.selective_teacher_topk_chunk_tokens = 8
    trainer.distill_hif4_act_controller = None
    trainer.teacher_runtime = _StaticTeacherRuntime(model)
    trainer.teacher_required = True
    trainer.accelerator = None
    trainer.distill_token_stats = DistillTokenStatsAccumulator()
    return trainer


def _grad_accum_micro_batch_a():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 3, 4, 5, 6]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def _grad_accum_micro_batch_b():
    input_ids = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 9, 10]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def test_gradient_accumulation_factor_two_includes_both_micro_batches(tmp_path):
    model = _KlFakeCausalLM()
    trainer = _build_kl_trainer(model)

    # simulate 10 optimizer steps with grad accum 2: two compute_loss calls per step
    for _ in range(10):
        model.train()
        trainer.compute_loss(model, _grad_accum_micro_batch_a())
        model.train()
        trainer.compute_loss(model, _grad_accum_micro_batch_b())

    # consume: should reflect all 20 micro-batches
    accelerator = _FakeAccelerator(torch.device("cpu"))
    stats = trainer.distill_token_stats.consume_global(accelerator)

    assert stats is not None
    assert stats.global_samples == 20
    # batch a: 2 prompt + 4 response (1 sample); batch b: 2 prompt + 2 response (1 sample)
    # per optimizer step (2 micro-batches): prompt=4, response=6, samples=2
    # over 10 steps: prompt=40, response=60, samples=20
    # weighted avg prompt = 40/20 = 2.0; avg response = 60/20 = 3.0
    assert stats.avg_prompt_tokens_per_sample == pytest.approx(2.0)
    assert stats.avg_response_tokens_per_sample == pytest.approx(3.0)


def test_compute_loss_does_not_update_when_model_not_training(tmp_path):
    model = _KlFakeCausalLM()
    trainer = _build_kl_trainer(model)

    model.eval()
    trainer.compute_loss(model, _grad_accum_micro_batch_a())

    # no update should have happened
    assert trainer.distill_token_stats._accumulator is None
