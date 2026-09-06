"""Task-9 E2E CLI bridge.

The public E2E CLI is owned by :mod:`train_utils.config.cli`. This module only
parses HF/Trainer-only arguments that remain after the common parser and maps
resolved common optimization/data config into ``TrainingArguments``.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

from transformers import HfArgumentParser

from train_utils.config.cli import E2ECLIConfig, build_e2e_parser, parse_e2e_cli
from train_utils.train_args import HFArguments, TrainingArguments


_HF_DUPLICATE_TRAINING_FLAGS = {
    "--max_steps",
    "--per_device_train_batch_size",
}


def _collect_flags(argv: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for token in argv:
        text = str(token)
        if text.startswith("--"):
            out.add(text.split("=", 1)[0])
    return out


def build_parser():
    """Return the canonical common E2E parser for compatibility with tests/tools."""
    return build_e2e_parser()


def _apply_common_training_config(cfg: E2ECLIConfig, training_args: TrainingArguments) -> None:
    opt = cfg.opt
    data = cfg.data

    training_args.max_steps = int(opt.steps)
    training_args.per_device_train_batch_size = int(opt.batch_size)
    training_args.learning_rate = float(opt.learning_rate)
    training_args.weight_decay = float(opt.weight_decay)
    training_args.gradient_accumulation_steps = int(opt.gradient_accumulation_steps)
    training_args.max_grad_norm = float(opt.max_grad_norm)
    training_args.warmup_ratio = float(opt.warmup_ratio)
    training_args.lr_scheduler_type = str(opt.lr_scheduler_type)
    training_args.optim = str(opt.optim)
    training_args.gradient_checkpointing = bool(opt.gradient_checkpointing)
    training_args.gradient_checkpointing_kwargs = dict(opt.gradient_checkpointing_kwargs)
    training_args.logging_steps = int(opt.logging_steps)
    training_args.seed = int(data.seed)
    training_args.data_seed = int(data.data_seed)
    training_args.model_max_length = int(data.model_max_length)
    training_args.group_by_length = bool(data.group_by_length)


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[E2ECLIConfig, HFArguments, TrainingArguments]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    duplicate_hf = sorted(_collect_flags(raw_argv) & _HF_DUPLICATE_TRAINING_FLAGS)
    if duplicate_hf:
        parser = build_e2e_parser()
        parser.error(
            "Do not pass HF duplicate training controls; use common E2E flags instead: "
            + ",".join(duplicate_hf)
            + ". Use --steps and --batch_size."
        )

    cfg = parse_e2e_cli(raw_argv)
    remaining = list(cfg.remaining_argv)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        from train_utils.lora_utils import (
            _resolve_distill_process_group_timeout_sec,
            ensure_distill_process_group_initialized,
        )

        timeout_sec = _resolve_distill_process_group_timeout_sec()
        if "--ddp_timeout" not in remaining:
            remaining.extend(["--ddp_timeout", str(timeout_sec)])
        ensure_distill_process_group_initialized()

    hf_parser = HfArgumentParser((HFArguments, TrainingArguments))
    hf_args, training_args = hf_parser.parse_args_into_dataclasses(args=remaining)
    _apply_common_training_config(cfg, training_args)

    if world_size > 1:
        training_args.ddp_timeout = int(timeout_sec)
        from train_utils.lora_utils import ensure_distill_process_group_initialized

        ensure_distill_process_group_initialized()

    fsdp = getattr(training_args, "fsdp", "")
    if not (fsdp is None or fsdp == "" or fsdp == []):
        raise ValueError("compressed_e2e_fintuning does not support FSDP.")

    return cfg, hf_args, training_args


__all__ = ["build_parser", "parse_args"]
