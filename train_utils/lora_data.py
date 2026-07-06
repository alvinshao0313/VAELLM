import math
import os
from typing import Dict, List

import torch

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
except ImportError:
    Dataset = None
    DatasetDict = None
    concatenate_datasets = None
    load_dataset = None


def ensure_distill_dataset_stack_available() -> None:
    if load_dataset is None or DatasetDict is None or concatenate_datasets is None:
        raise ImportError("未安装 datasets。请先安装：pip install datasets")


def _resolve_distill_dataset_num_proc() -> int:
    raw_num_proc = os.environ.get("CAT_DISTILL_DATASET_NUM_PROC", "16")
    num_proc = int(raw_num_proc)
    if num_proc < 1:
        raise ValueError(f"CAT_DISTILL_DATASET_NUM_PROC must be >= 1, got {raw_num_proc}.")
    return num_proc


def _split_lora_mix_targets(total_rows: int, weights: List[float]) -> List[int]:
    if int(total_rows) < 1:
        raise ValueError(f"--distill_nsamples must be >= 1, got {total_rows}.")
    if not weights:
        raise ValueError("--distill_dataset cannot be empty.")
    targets: List[int] = []
    allocated = 0
    for idx, weight in enumerate(weights):
        if idx == len(weights) - 1:
            rows = int(total_rows) - int(allocated)
        else:
            rows = int(math.floor(float(total_rows) * float(weight)))
            allocated += int(rows)
        if rows < 1:
            raise ValueError(
                "--distill_nsamples is too small for the requested --distill_dataset mix; "
                f"source index {idx} got target_rows={rows}."
            )
        targets.append(int(rows))
    return targets


def _split_calibration_mix_targets(total_blocks: int, weights: List[float]) -> List[int]:
    if int(total_blocks) < 0:
        raise ValueError(f"--activation_calib_nsamples must be >= 0, got {total_blocks}.")
    if int(total_blocks) == 0:
        return [0 for _weight in weights]
    if not weights:
        raise ValueError("--activation_calib_dataset cannot be empty.")
    targets: List[int] = []
    allocated = 0
    for idx, weight in enumerate(weights):
        if idx == len(weights) - 1:
            blocks = int(total_blocks) - int(allocated)
        else:
            blocks = int(math.floor(float(total_blocks) * float(weight)))
            allocated += int(blocks)
        if blocks < 1:
            raise ValueError(
                "--activation_calib_nsamples is too small for the requested --activation_calib_dataset mix; "
                f"source index {idx} got target_blocks={blocks}."
            )
        targets.append(int(blocks))
    return targets


def _prepare_lora_mix_source(
    *,
    alias: str,
    target_rows: int,
    seed: int,
):
    from e2e_common.data import (
        DATASET_MIX_SOURCE_PRESETS,
        _load_preset_raw_datasets,
        _prepare_text_dataset as _prepare_e2e_text_dataset,
    )

    preset = DATASET_MIX_SOURCE_PRESETS[str(alias)]
    train_raw, _eval_raw = _load_preset_raw_datasets(preset)
    raw_rows = int(len(train_raw))
    chunk_size = 4096
    num_proc = _resolve_distill_dataset_num_proc()
    shuffled_raw = train_raw.shuffle(seed=int(seed))
    text_chunks = []
    processed_raw_rows = 0
    collected_text_rows = 0

    for start in range(0, raw_rows, chunk_size):
        stop = min(start + chunk_size, raw_rows)
        chunk = shuffled_raw.select(range(start, stop))
        processed_raw_rows += int(len(chunk))
        text_chunk = _prepare_e2e_text_dataset(
            chunk,
            text_field=str(preset.text_field),
            text_format=str(preset.text_format),
            num_proc=int(num_proc),
        )
        if len(text_chunk) > 0:
            text_chunks.append(text_chunk)
            collected_text_rows += int(len(text_chunk))
        if collected_text_rows >= int(target_rows):
            break

    if collected_text_rows < int(target_rows):
        raise ValueError(
            f"LoRA dataset mix source '{alias}' has only {collected_text_rows} usable text rows, "
            f"but target_rows={int(target_rows)}."
        )

    train_text = text_chunks[0] if len(text_chunks) == 1 else concatenate_datasets(text_chunks)
    selected = train_text.shuffle(seed=int(seed)).select(range(int(target_rows)))
    return selected, {
        "alias": str(alias),
        "path": str(preset.path),
        "config": None if preset.config is None else str(preset.config),
        "train_split": str(preset.train_split),
        "raw_rows": int(raw_rows),
        "text_rows": int(collected_text_rows),
        "target_rows": int(target_rows),
        "actual_rows": int(len(selected)),
        "processed_raw_rows": int(processed_raw_rows),
        "limited_preprocessing": bool(processed_raw_rows < raw_rows),
        "sampling_policy": "shuffled_raw_streaming_text",
    }


def _prepare_lora_mixed_dataset(
    dataset_mix_spec: str,
    *,
    nsamples: int,
    seed: int,
):
    from e2e_common.data import normalize_dataset_mix_spec

    sources, weights, normalized_spec = normalize_dataset_mix_spec(dataset_mix_spec)
    targets = _split_lora_mix_targets(int(nsamples), [float(weight) for weight in weights])

    train_datasets = []
    source_stats: List[Dict[str, object]] = []
    for idx, (alias, weight, target_rows) in enumerate(zip(sources, weights, targets)):
        source_ds, source_info = _prepare_lora_mix_source(
            alias=str(alias),
            target_rows=int(target_rows),
            seed=int(seed) + int(idx),
        )
        source_info["weight"] = float(weight)
        train_datasets.append(source_ds)
        source_stats.append(source_info)

    train_ds = concatenate_datasets(train_datasets)
    return str(normalized_spec), source_stats, train_ds


def _iter_calibration_texts_for_source(
    *,
    alias: str,
    seed: int,
):
    from e2e_common.data import (
        DATASET_MIX_SOURCE_PRESETS,
        _load_preset_raw_datasets,
        _prepare_text_dataset as _prepare_e2e_text_dataset,
    )

    preset = DATASET_MIX_SOURCE_PRESETS[str(alias)]
    train_raw, _eval_raw = _load_preset_raw_datasets(preset)
    raw_rows = int(len(train_raw))
    chunk_size = 4096
    num_proc = _resolve_distill_dataset_num_proc()
    shuffled_raw = train_raw.shuffle(seed=int(seed))
    yielded_text = False

    for start in range(0, raw_rows, chunk_size):
        stop = min(start + chunk_size, raw_rows)
        chunk = shuffled_raw.select(range(start, stop))
        text_chunk = _prepare_e2e_text_dataset(
            chunk,
            text_field=str(preset.text_field),
            text_format=str(preset.text_format),
            num_proc=int(num_proc),
        )
        if len(text_chunk) < 1:
            continue
        for record in text_chunk.shuffle(seed=int(seed)):
            text = record.get("text")
            if text is None:
                continue
            text = str(text).strip()
            if text:
                yielded_text = True
                yield text

    if not yielded_text:
        raise ValueError(f"Calibration dataset mix source '{alias}' has no usable text rows.")


def _build_calibration_blocks_for_source(
    *,
    alias: str,
    tokenizer,
    target_blocks: int,
    block_size: int,
    seed: int,
) -> List[torch.Tensor]:
    blocks: List[torch.Tensor] = []
    token_buffer: List[int] = []
    for text in _iter_calibration_texts_for_source(alias=str(alias), seed=int(seed)):
        encoded = tokenizer(
            text + "\n\n",
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError(f"Tokenizer output for calibration dataset mix source '{alias}' is missing input_ids.")
        token_buffer.extend(int(token) for token in input_ids)
        while len(token_buffer) >= int(block_size) and len(blocks) < int(target_blocks):
            blocks.append(torch.tensor(token_buffer[:int(block_size)], dtype=torch.long).unsqueeze(0))
            del token_buffer[:int(block_size)]
        if len(blocks) >= int(target_blocks):
            break

    if len(blocks) != int(target_blocks):
        raise ValueError(
            f"Calibration dataset mix source '{alias}' does not contain enough tokens to build "
            f"{int(target_blocks)} blocks of length {int(block_size)}. Built only {len(blocks)} blocks."
        )
    return blocks


def build_calibration_input_ids(
    dataset_name: str,
    *,
    tokenizer,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> List[torch.Tensor]:
    ensure_distill_dataset_stack_available()
    target_blocks = int(nsamples)
    block_size = int(seqlen)
    dataset_mix_spec = str(dataset_name or "").strip()
    if not dataset_mix_spec:
        raise ValueError("--activation_calib_dataset must be set when dynamic calibration is enabled.")
    if "=" not in dataset_mix_spec:
        raise ValueError(
            "--activation_calib_dataset only accepts ratio-style dataset specs, for example "
            "'wiki=1.0', 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
        )
    from e2e_common.data import normalize_dataset_mix_spec

    sources, weights, _normalized_spec = normalize_dataset_mix_spec(dataset_mix_spec)
    if target_blocks < 0:
        raise ValueError(f"--activation_calib_nsamples must be >= 0, got {target_blocks}.")
    if block_size <= 0:
        raise ValueError(f"--activation_calib_seqlen must be > 0, got {block_size}.")
    if target_blocks == 0:
        return []

    per_source_targets = _split_calibration_mix_targets(target_blocks, [float(weight) for weight in weights])
    blocks: List[torch.Tensor] = []
    for idx, (alias, source_target_blocks) in enumerate(zip(sources, per_source_targets)):
        if int(source_target_blocks) == 0:
            continue
        blocks.extend(
            _build_calibration_blocks_for_source(
                alias=str(alias),
                tokenizer=tokenizer,
                target_blocks=int(source_target_blocks),
                block_size=int(block_size),
                seed=int(seed) + int(idx),
            )
        )

    if len(blocks) != target_blocks:
        raise ValueError(
            f"Calibration dataset mix does not contain enough tokens to build "
            f"{target_blocks} blocks of length {block_size}. Built only {len(blocks)} blocks."
        )
    return blocks


def prepare_distill_datasets(
    dataset_name: str,
    *,
    nsamples: int,
    seed: int,
):
    ensure_distill_dataset_stack_available()
    if "=" not in str(dataset_name):
        raise ValueError(
            "--distill_dataset only accepts ratio-style dataset specs, for example "
            "'wiki=1.0', 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
        )
    dataset_mix_spec, source_stats, train_ds = _prepare_lora_mixed_dataset(
        str(dataset_name),
        nsamples=int(nsamples),
        seed=int(seed),
    )
    return dataset_mix_spec, source_stats, train_ds, None, None
