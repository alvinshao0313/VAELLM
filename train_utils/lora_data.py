from typing import Dict, List, Optional, Tuple

import torch

from e2e_common.lazy_datasets import (
    build_calibration_input_ids_lazy,
    build_distill_lazy_dataset,
    dataset_length_or_none,
)

__all__ = [
    "build_calibration_input_ids",
    "build_distill_lazy_dataset",
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
    seed: int,
    cache_dir: Optional[str] = None,
    tokenizer=None,
    max_seq_len: int = 2048,
):
    del cache_dir
    ensure_distill_dataset_stack_available()
    if "=" not in str(dataset_name):
        raise ValueError(
            "--distill_dataset only accepts ratio-style dataset specs, for example "
            "'wiki=1.0', 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
        )
    if tokenizer is None:
        raise ValueError("prepare_distill_datasets requires tokenizer for lazy EdgeRazor loading.")

    dataset_mix_spec, source_stats, train_ds, is_iterable = build_distill_lazy_dataset(
        str(dataset_name),
        tokenizer=tokenizer,
        max_seq_len=int(max_seq_len),
        seed=int(seed),
    )
    for source_info in source_stats:
        source_info["is_iterable"] = bool(is_iterable)
        source_info["actual_rows"] = dataset_length_or_none(train_ds)
    return dataset_mix_spec, source_stats, train_ds, None, None


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
