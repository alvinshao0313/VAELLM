"""Shared model-level trainer logging and distributed runtime helpers."""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import List, Sequence

import torch
from torch import nn

try:
    from transformers import AutoTokenizer, TrainerCallback, TrainingArguments
    from transformers.trainer_callback import ProgressCallback
except ImportError:
    AutoTokenizer = None
    ProgressCallback = None
    TrainerCallback = None
    TrainingArguments = None

from train_utils.lora_data import ensure_distill_dataset_stack_available
from train_utils.lora_training import ensure_lora_training_stack_available


class _LoraTrainerLogCallback(TrainerCallback if TrainerCallback is not None else object):
    def __init__(self, *, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not bool(getattr(state, "is_world_process_zero", True)) or not logs:
            return
        values = dict(logs)
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            decoder_lrs = []
            main_lrs = []
            for group in getattr(optimizer, "param_groups", []):
                if not list(group.get("params", [])):
                    continue
                lr = float(group.get("lr", 0.0))
                group_name = str(group.get("group_name") or "")
                if group_name == "decoder":
                    decoder_lrs.append(lr)
                elif group_name == "lora" or group_name.startswith("nondecoder"):
                    main_lrs.append(lr)
            if len(set(main_lrs)) > 1 or len(set(decoder_lrs)) > 1:
                raise ValueError("model-level optimizer groups in one family must share the same lr.")
            if main_lrs:
                values["lr_lora"] = main_lrs[0]
            if decoder_lrs:
                values["lr_decoder"] = decoder_lrs[0]
        values.pop("total_flos", None)
        ordered_keys = (
            "loss", "train_loss", "eval_loss", "learning_rate", "lr_lora",
            "lr_decoder", "grad_norm", "epoch",
        )
        parts = []
        for key in ordered_keys:
            if key in values:
                parts.append(f"{key}={values.pop(key)}")
        parts.extend(f"{key}={values[key]}" for key in sorted(values))
        if parts:
            _log_lora_trainer_message_to_file_handlers(
                self.logger,
                "LoRA train: step=%s %s",
                str(getattr(state, "global_step", "unknown")),
                " ".join(parts),
            )


class _QuietProgressCallback(ProgressCallback if ProgressCallback is not None else object):
    def on_log(self, args, state, control, logs=None, **kwargs):
        return


class _LoraDistillTokenStatsCallback(TrainerCallback if TrainerCallback is not None else object):
    def __init__(self, *, trainer, logger):
        self._trainer = trainer
        self._logger = logger
        self.window_start_step = None

    def on_step_end(self, args, state, control, **kwargs):
        logging_steps = getattr(state, "logging_steps", None)
        if not isinstance(logging_steps, int) or logging_steps <= 0:
            raise ValueError(f"state.logging_steps must be a positive integer, got {logging_steps!r}.")
        global_step = int(getattr(state, "global_step", 0))
        if self.window_start_step is None:
            self.window_start_step = global_step
        if global_step <= 0 or global_step % logging_steps != 0:
            return
        stats = self._trainer.distill_token_stats.consume_global(self._trainer.accelerator)
        window_optimizer_steps = global_step - self.window_start_step + 1
        self.window_start_step = global_step + 1
        if stats is None or not bool(getattr(state, "is_world_process_zero", True)):
            return
        _log_lora_trainer_message_to_file_handlers(
            self._logger,
            "LoRA token stats: step=%s window_optimizer_steps=%d avg_prompt_tokens=%.4f avg_response_tokens=%.4f global_samples=%d",
            str(global_step),
            int(window_optimizer_steps),
            float(stats.avg_prompt_tokens_per_sample),
            float(stats.avg_response_tokens_per_sample),
            int(stats.global_samples),
        )


def _log_lora_trainer_message_to_file_handlers(logger, message: str, *args) -> None:
    record = logger.makeRecord(
        logger.name, logging.INFO, fn="", lno=0, msg=message, args=args, exc_info=None
    )
    for handler in list(getattr(logger, "handlers", [])):
        if isinstance(handler, logging.FileHandler) and record.levelno >= handler.level:
            handler.handle(record)


def _replace_progress_log_callback(trainer):
    if ProgressCallback is None:
        return trainer
    callbacks = getattr(getattr(trainer, "callback_handler", None), "callbacks", None)
    if not isinstance(callbacks, list):
        return trainer
    for idx, callback in enumerate(callbacks):
        if isinstance(callback, ProgressCallback) and not isinstance(callback, _QuietProgressCallback):
            callbacks[idx] = _QuietProgressCallback()
    return trainer


def _ensure_lora_stack_available() -> None:
    ensure_lora_training_stack_available()
    ensure_distill_dataset_stack_available()
    if AutoTokenizer is None or TrainingArguments is None:
        raise ImportError("未安装 transformers。请先安装：pip install transformers")


def distill_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distill_distributed() -> bool:
    return distill_world_size() > 1


def resolve_distill_train_device(fallback: str) -> str:
    device = str(fallback).strip()
    if not is_distill_distributed():
        return device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device.startswith("cuda") and torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return device


def is_distill_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()) == 0
    return int(os.environ.get("RANK", "0")) == 0


def distill_distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _resolve_distill_process_group_timeout_sec() -> int:
    raw = str(os.environ.get("DISTILL_NCCL_TIMEOUT_SEC", "10800")).strip()
    try:
        timeout_sec = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"DISTILL_NCCL_TIMEOUT_SEC must be an integer number of seconds, got {raw!r}."
        ) from exc
    if timeout_sec <= 0:
        raise ValueError(f"DISTILL_NCCL_TIMEOUT_SEC must be > 0, got {timeout_sec}.")
    return timeout_sec


def _apply_distill_process_group_timeout(timeout: timedelta) -> None:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    import torch.distributed.distributed_c10d as c10d

    c10d._set_pg_timeout(timeout, c10d._get_default_group())


def ensure_distill_process_group_initialized() -> None:
    if not is_distill_distributed():
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable but WORLD_SIZE > 1.")
    timeout = timedelta(seconds=_resolve_distill_process_group_timeout_sec())
    if torch.distributed.is_initialized():
        _apply_distill_process_group_timeout(timeout)
        return
    backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=backend, timeout=timeout)
    _apply_distill_process_group_timeout(timeout)


def distill_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def get_distill_local_device(*, fallback: str = "cuda") -> str:
    return resolve_distill_train_device(str(fallback))


def unwrap_distill_model(model: nn.Module) -> nn.Module:
    current = model
    while hasattr(current, "module"):
        inner = getattr(current, "module")
        if inner is current:
            break
        current = inner
    return current


def split_tasks_for_distill_rank(
    task_names: Sequence[str], *, rank: int, world_size: int
) -> List[str]:
    world = int(world_size)
    current_rank = int(rank)
    if world <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}.")
    if current_rank < 0 or current_rank >= world:
        raise ValueError(f"rank must be in [0, {world}), got {rank}.")
    return [str(name) for idx, name in enumerate(task_names) if idx % world == current_rank]


def _ensure_lora_tokenizer_ready(*, vae_args, model: nn.Module) -> None:
    tokenizer = getattr(vae_args, "_cached_lora_tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            vae_args.model_path,
            use_fast=True,
            token=getattr(vae_args, "access_token", None),
        )
        setattr(vae_args, "_cached_lora_tokenizer", tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


__all__ = [
    "_LoraDistillTokenStatsCallback",
    "_LoraTrainerLogCallback",
    "_ensure_lora_stack_available",
    "_ensure_lora_tokenizer_ready",
    "_replace_progress_log_callback",
    "_resolve_distill_process_group_timeout_sec",
    "distill_distributed_barrier",
    "distill_rank",
    "distill_world_size",
    "ensure_distill_process_group_initialized",
    "get_distill_local_device",
    "is_distill_distributed",
    "is_distill_main_process",
    "resolve_distill_train_device",
    "split_tasks_for_distill_rank",
    "unwrap_distill_model",
]
