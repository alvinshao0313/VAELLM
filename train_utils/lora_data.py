from typing import Dict, List, Optional

import torch

from e2e_common.lazy_datasets import (
    build_calibration_input_ids_lazy,
    dataset_length_or_none,
)
from train_utils.config.configs import DistillDataConfig
from train_utils.distill_data import build_distill_dataset

__all__ = [
    "build_calibration_input_ids",
    "dataset_length_or_none",
    "ensure_distill_dataset_stack_available",
    "prepare_distill_datasets",
]


def ensure_distill_dataset_stack_available() -> None:
    try:
        from datasets import Dataset  # noqa: F401
    except ImportError as exc:
        raise ImportError("未安装 datasets。请先安装：pip install datasets") from exc


def prepare_distill_datasets(
    dataset_name: str,
    *,
    task: str = "sft",
    seed: int,
    cache_dir: Optional[str] = None,
    tokenizer=None,
    max_seq_len: int = 2048,
    raw_dataset_cache=None,
):
    del cache_dir
    ensure_distill_dataset_stack_available()
    if tokenizer is None:
        raise ValueError("prepare_distill_datasets requires tokenizer for lazy EdgeRazor loading.")

    task_norm = str(task).strip().lower()
    if task_norm not in {"sft", "lm"}:
        raise ValueError(f"distill dataset task must be one of: sft | lm, got {task!r}.")

    # Shorthand like "openorca" is expanded by DistillDataConfig.validate()
    # via parse_dataset_mix_spec (openorca == openorca=1.0).
    cfg = DistillDataConfig(
        dataset_mix=str(dataset_name),
        dataset_task=task_norm,
        model_max_length=int(max_seq_len),
        seed=int(seed),
        data_seed=int(seed),
    )
    cfg.validate()
    bundle = build_distill_dataset(
        cfg,
        tokenizer,
        raw_dataset_cache=raw_dataset_cache,
    )
    source_stats = list(bundle.source_stats)
    for source_info in source_stats:
        source_info["is_iterable"] = bool(bundle.is_iterable)
        source_info["actual_rows"] = dataset_length_or_none(bundle.train_dataset)
    return (
        str(bundle.dataset_mix_spec or dataset_name),
        source_stats,
        bundle.train_dataset,
        None,
        None,
    )


def build_calibration_input_ids(
    dataset_name: str,
    *,
    tokenizer,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> List[torch.Tensor]:
    ensure_distill_dataset_stack_available()
    return build_calibration_input_ids_lazy(
        str(dataset_name),
        tokenizer=tokenizer,
        nsamples=int(nsamples),
        seqlen=int(seqlen),
        seed=int(seed),
    )
