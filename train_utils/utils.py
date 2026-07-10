# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging
import os
import random
import argparse
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as PT_FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def pt_fsdp_state_dict(model: torch.nn.Module):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with PT_FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(ctx, u):
        return hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return hadamard_transform(grad)


def llama_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, "groupsize should be greater than 1!"

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f"(Act.) Groupsiz = Down_proj Groupsize: {groupsize}")
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert (
        groupsize * group_num == model.config.hidden_size
    ), "Invalid groupsize for llama!"

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert (
        down_proj_groupsize * group_num == model.config.intermediate_size
    ), "Invalid groupsize for down_proj!"
    logging.info(
        f"(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}"
    )
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def configure_deterministic_mode(enabled: bool) -> None:
    if not bool(enabled):
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(message)s"
            else:
                self._style._fmt = "%(levelname)s: %(message)s"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def _has_file_handler(logger: logging.Logger, log_file: str) -> bool:
    log_file_abs = os.path.abspath(log_file)
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        try:
            if os.path.abspath(handler.baseFilename) == log_file_abs:
                return True
        except Exception:
            continue
    return False


def _add_file_handler_once(
    logger: logging.Logger,
    *,
    log_file: str,
    formatter: logging.Formatter,
    level: int,
) -> None:
    if _has_file_handler(logger, log_file):
        return
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def _is_rank_zero_for_logging() -> bool:
    rank = os.environ.get("RANK")
    if rank is not None and str(rank).strip() != "":
        return int(rank) == 0
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and str(local_rank).strip() != "":
        return int(local_rank) == 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()) == 0
    return True


# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name: Optional[str]) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    if not _is_rank_zero_for_logging():
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = os.environ.get("LOG_FILE")
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        _add_file_handler_once(
            logger,
            log_file=log_file,
            formatter=formatter,
            level=logging.INFO,
        )

    return logger


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


def split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def format_intra_parallel_desc(row_parts: int, col_parts: int) -> str:
    row_parts = int(row_parts)
    col_parts = int(col_parts)
    if col_parts == 1:
        return str(row_parts)
    return f"[{row_parts},{col_parts}]"


@dataclass(frozen=True)
class LinearRef:
    name: str
    module: nn.Linear
    category: str
    transpose: bool


def is_decoder_layer_projection(name: str, target_categories: Sequence[str]) -> bool:
    # Llama/Mistral/Qwen: "model.layers.{i}.<...>.<proj>"
    # OPT: "model.decoder.layers.{i}.<...>.<proj>"
    in_decoder_layers = (
        ".model.layers." in name
        or name.startswith("model.layers.")
        or ".model.decoder.layers." in name
        or name.startswith("model.decoder.layers.")
    )
    if not in_decoder_layers:
        return False
    return any(name.endswith(f".{category}") or name.endswith(category) for category in target_categories)


def collect_linears(
    model: nn.Module,
    transpose_modules: Sequence[str],
    *,
    only_decoder_projections: bool,
    target_categories: Sequence[str],
) -> List[LinearRef]:
    transpose_set = set(transpose_modules)
    target_set = set(target_categories)
    out: List[LinearRef] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        category = name.split(".")[-1]
        if category not in target_set:
            continue
        if only_decoder_projections:
            if not is_decoder_layer_projection(name, target_categories):
                continue
        out.append(
            LinearRef(
                name=name,
                module=module,
                category=category,
                transpose=(category in transpose_set),
            )
        )
    return out


_LAYER_IDX_PATTERNS = [
    re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\."),
    re.compile(r"(?:^|\.)(?:model\.)?decoder\.layers\.(\d+)\."),
]


def extract_layer_idx(name: str) -> Optional[int]:
    for pat in _LAYER_IDX_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    return None


def clone_namespace(ns, **overrides):
    data = dict(vars(ns))
    data.update(overrides)
    return argparse.Namespace(**data)


def format_namespace(ns: argparse.Namespace) -> str:
    return json.dumps(vars(ns), ensure_ascii=False, indent=2, sort_keys=True, default=str)
