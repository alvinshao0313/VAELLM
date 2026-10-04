"""Exercise real Trainer stopping/saving while retaining the formal LR budget."""

from copy import deepcopy
import json
import math

import pytest
import torch
from torch import nn
from transformers import Trainer, TrainerCallback

from compressed_e2e_fintuning.args import parse_args
from experiments.qwen3_w25.train_reproduction_stage import (
    FinishStepCallback, validate_configuration, validate_schedule,
)
from train_utils.train_args import TrainingArguments


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, input_values, labels):
        predictions = self.linear(input_values).squeeze(-1)
        return {"loss": (predictions.float() - labels.float()).square().mean(), "logits": predictions}


class TrainingObservations(TrainerCallback):
    def __init__(self):
        self.events = []
        self.rates = {}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in (20, 400, 800, 1200):
            self.rates[state.global_step] = kwargs["lr_scheduler"].get_last_lr()[0]
        if state.global_step % 400 == 0:
            self.events.append(("update", state.global_step))
        return control

    def on_save(self, args, state, control, **kwargs):
        self.events.append(("save", state.global_step))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.events.append(("end", state.global_step))
        return control


def test_real_trainer_saves_1200_with_the_2000_step_cosine_schedule(tmp_path):
    torch.set_num_threads(1)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], disable_tqdm=True,
        max_steps=2000, per_device_train_batch_size=2, learning_rate=1e-3,
        warmup_ratio=0.01, lr_scheduler_type="cosine", save_strategy="steps", save_steps=400,
        save_total_limit=1, logging_strategy="no", eval_strategy="no", save_safetensors=False)
    finish, observations = FinishStepCallback(1200), TrainingObservations()
    model = LinearRegression()
    before = model.linear.weight.detach().clone()
    dataset = [{"input_values": torch.tensor([float(i), 1.0]), "labels": torch.tensor(float(2 * i + 1))}
               for i in range(8)]
    trainer = Trainer(model=model, args=args, train_dataset=dataset, callbacks=[finish, observations])
    trained = trainer.train()
    assert trained.global_step == trainer.state.global_step == finish.global_step == finish.saved_step == 1200
    assert trainer.args.max_steps == trainer.state.max_steps == 2000
    assert args.get_warmup_steps(args.max_steps) == 20
    assert not torch.equal(before, model.linear.weight)
    for step, rate in observations.rates.items():
        expected = 1e-3 * 0.5 * (1 + math.cos(math.pi * (step - 20) / (2000 - 20)))
        assert rate == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert observations.rates[1200] > 0
    assert observations.events == [("update", 400), ("save", 400), ("update", 800), ("save", 800),
                                   ("update", 1200), ("save", 1200), ("end", 1200)]
    assert [p.name for p in tmp_path.glob("checkpoint-*")] == ["checkpoint-1200"]
    saved = json.loads((tmp_path / "checkpoint-1200/trainer_state.json").read_text())
    assert saved["global_step"] == 1200 and saved["max_steps"] == 2000


def _configuration(tmp_path, mode="decoder"):
    argv = ["--student_checkpoint_dir", str(tmp_path / "source"), "--run_root_dir", str(tmp_path),
        "--dataset_mix", "vaellm_eval_task=1", "--dataset_task", "sft",
        "--train_mode", mode, "--distill_fp32_components", "decoder", "--steps", "2000",
        "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.01", "--bf16", "true",
        "--use_cpu", "true", "--report_to", "none", "--save_strategy", "steps", "--save_steps", "400",
        "--save_total_limit", "1", "--eval_strategy", "no", "--eval_after_save", "false", "--eval_tasks", "",
        "--parallel_mode", "dp", "--offload_mode", "none", "--norm_train_mode", "none",
        "--lm_head_train_mode", "none", "--residual_lora_mode", "none"]
    if mode == "decoder_sparse_bit":
        argv += ["--target_layers", "0", "--target_modules", "down_proj", "--bit_active_ratio", "1",
                 "--bit_proxy_coordinates", "decoder_sensitivity", "--bit_lr", "4e-5"]
    cfg, _hf, args = parse_args(argv)
    return cfg, args


@pytest.mark.parametrize("mode,finish", [("decoder", 2000), ("decoder_sparse_bit", 1200)])
def test_canonical_route_configuration_and_invalid_selections(tmp_path, mode, finish):
    cfg, args = _configuration(tmp_path, mode)
    validate_configuration(cfg, args, finish)
    for key, value in (("resume_from_checkpoint", "old-step"), ("stop_after_step", 400),
                       ("train_mode", "lora"), ("target_layers", (1,)), ("target_modules", ("gate_proj",))):
        bad = deepcopy(cfg)
        setattr(bad, key, value)
        with pytest.raises(ValueError):
            validate_configuration(bad, args, finish)
    for owner, key, value in (("aux", "norm_train_mode", "all"), ("aux", "lm_head_train_mode", "linear"),
                              ("aux", "residual_lora_mode", "all"), ("opt", "distill_fp32_components", ()),
                              ("runtime", "parallel_mode", "layer_mp"), ("runtime", "offload_mode", "streaming")):
        bad = deepcopy(cfg)
        setattr(getattr(bad, owner), key, value)
        with pytest.raises(ValueError):
            validate_configuration(bad, args, finish)
    if mode == "decoder_sparse_bit":
        for key, value in (("bit_active_ratio", 0.05), ("bit_proxy_coordinates", "unit")):
            bad = deepcopy(cfg)
            setattr(bad, key, value)
            with pytest.raises(ValueError):
                validate_configuration(bad, args, finish)


def test_rejects_changed_schedule_or_non_save_endpoints(tmp_path):
    _cfg, args = _configuration(tmp_path)
    for finish in (0, -400, 1201, 2400, True):
        with pytest.raises(ValueError):
            validate_schedule(args, finish)
    for key, value in (("max_steps", 1200), ("warmup_ratio", 0.02), ("lr_scheduler_type", "linear"),
                       ("save_steps", 200), ("save_total_limit", 2), ("save_strategy", "no")):
        bad = deepcopy(args)
        setattr(bad, key, value)
        with pytest.raises(ValueError):
            validate_schedule(bad, 1200)
