"""Disjoint Alpaca training documents for local decoder reconstruction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from datasets import Dataset, config as datasets_config
import torch

from train_utils.distill_data import (
    FORMATTING_VERSION,
    build_distill_data_collator,
    encode_canonical_record,
)


DATASET_PATH = "vicgalle/alpaca-gpt4"
DATASET_REVISION = "f7e3ded725cb81e8e564e32feb12860f376f2b51"


def _sha256(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_calibration_batches(
    tokenizer,
    *,
    seed: int = 31,
    train_documents: int,
    heldout_documents: int,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    if train_documents < 1 or heldout_documents < 1:
        raise ValueError("Train and heldout document counts must both be positive.")
    if max_length < 2:
        raise ValueError("max_length must be at least 2.")
    source = (
        Path(datasets_config.HF_DATASETS_CACHE)
        / "vicgalle___alpaca-gpt4/default/0.0.0"
        / DATASET_REVISION
        / "alpaca-gpt4-train.arrow"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Required local Alpaca train cache is missing: {source}")
    raw = Dataset.from_file(str(source))
    order = list(range(len(raw)))
    random.Random(seed).shuffle(order)
    requested = train_documents + heldout_documents
    samples, records = [], []
    seen_groups, seen_tokens = set(), set()
    for row_index in order:
        record = dict(raw[row_index])
        # Different responses to the same instruction/input stay in one group.
        group_hash = _sha256([
            " ".join(record[key].split()) for key in ("instruction", "input")
        ])
        if group_hash in seen_groups:
            continue
        sample = encode_canonical_record(
            record,
            tokenizer,
            text_format="alpaca",
            text_field="text",
            task="lm",
            model_max_length=max_length,
        )
        if sample is None:
            continue
        token_hash = _sha256(sample["input_ids"].tolist())
        if token_hash in seen_tokens:
            continue
        seen_groups.add(group_hash)
        seen_tokens.add(token_hash)
        samples.append(sample)
        records.append({
            "row_index": row_index,
            "group_sha256": group_hash,
            "content_sha256": _sha256([
                record[key] for key in ("instruction", "input", "output")
            ]),
            "token_sha256": token_hash,
            "valid_tokens": int(sample["attention_mask"].sum().item()),
        })
        if len(samples) == requested:
            break
    if len(samples) != requested:
        raise ValueError(
            f"Requested {requested} disjoint documents, found only {len(samples)}."
        )
    collator = build_distill_data_collator(
        tokenizer, model_max_length=max_length, dynamic_padding=True,
    )
    batches, splits = [], {}
    for name, start, stop in (
        ("train", 0, train_documents),
        ("heldout", train_documents, requested),
    ):
        batch = collator(samples[start:stop])
        batches.append({key: batch[key] for key in ("input_ids", "attention_mask")})
        splits[name] = {
            "documents": records[start:stop],
            "valid_tokens": int(batch["attention_mask"].sum().item()),
        }
    manifest = {
        "dataset_path": DATASET_PATH,
        "revision": DATASET_REVISION,
        "split": "train",
        "source_arrow": str(source),
        "source_rows": len(raw),
        "seed": seed,
        "max_length": max_length,
        "formatting_version": FORMATTING_VERSION,
        "dataset_task": "lm",
        "grouping": "whitespace-normalized instruction/input; exact token sequence deduplication",
        "train": splits["train"],
        "heldout": splits["heldout"],
    }
    return batches[0], batches[1], manifest
