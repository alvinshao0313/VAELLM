import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
except ImportError:
    Dataset = None
    DatasetDict = None
    concatenate_datasets = None
    load_dataset = None
    load_from_disk = None


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
        _prepare_edgerazor_messages_dataset,
        _prepare_text_dataset as _prepare_e2e_text_dataset,
    )

    preset = DATASET_MIX_SOURCE_PRESETS[str(alias)]
    train_raw, _eval_raw = _load_preset_raw_datasets(preset)
    raw_rows = int(len(train_raw))
    chunk_size = 4096
    num_proc = _resolve_distill_dataset_num_proc()
    shuffled_raw = train_raw.shuffle(seed=int(seed))
    text_format = str(preset.text_format)
    if text_format == "edgerazor_messages":
        message_chunks = []
        processed_raw_rows = 0
        collected_message_rows = 0
        for start in range(0, raw_rows, chunk_size):
            stop = min(start + chunk_size, raw_rows)
            chunk = shuffled_raw.select(range(start, stop))
            processed_raw_rows += int(len(chunk))
            message_chunk = _prepare_edgerazor_messages_dataset(chunk, num_proc=int(num_proc))
            if len(message_chunk) > 0:
                message_chunks.append(message_chunk)
                collected_message_rows += int(len(message_chunk))
            if collected_message_rows >= int(target_rows):
                break
        if collected_message_rows < int(target_rows):
            raise ValueError(
                f"LoRA dataset mix source '{alias}' has only {collected_message_rows} usable message rows, "
                f"but target_rows={int(target_rows)}."
            )
        train_messages = message_chunks[0] if len(message_chunks) == 1 else concatenate_datasets(message_chunks)
        selected = train_messages.shuffle(seed=int(seed)).select(range(int(target_rows)))
        return selected, {
            "alias": str(alias),
            "path": str(preset.path),
            "config": None if preset.config is None else str(preset.config),
            "train_split": str(preset.train_split),
            "raw_rows": int(raw_rows),
            "text_rows": int(collected_message_rows),
            "target_rows": int(target_rows),
            "actual_rows": int(len(selected)),
            "processed_raw_rows": int(processed_raw_rows),
            "limited_preprocessing": bool(processed_raw_rows < raw_rows),
            "sampling_policy": "shuffled_raw_streaming_messages",
        }

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
    tokenizer=None,
):
    from e2e_common.chat_template_utils import render_messages
    from e2e_common.data import (
        DATASET_MIX_SOURCE_PRESETS,
        _load_preset_raw_datasets,
        _prepare_edgerazor_messages_dataset,
        _prepare_text_dataset as _prepare_e2e_text_dataset,
    )

    preset = DATASET_MIX_SOURCE_PRESETS[str(alias)]
    train_raw, _eval_raw = _load_preset_raw_datasets(preset)
    raw_rows = int(len(train_raw))
    chunk_size = 4096
    num_proc = _resolve_distill_dataset_num_proc()
    shuffled_raw = train_raw.shuffle(seed=int(seed))
    yielded_text = False
    text_format = str(preset.text_format)

    for start in range(0, raw_rows, chunk_size):
        stop = min(start + chunk_size, raw_rows)
        chunk = shuffled_raw.select(range(start, stop))
        if text_format == "edgerazor_messages":
            if tokenizer is None:
                raise ValueError(
                    f"Calibration dataset mix source '{alias}' uses edgerazor_messages and requires tokenizer."
                )
            message_chunk = _prepare_edgerazor_messages_dataset(chunk, num_proc=int(num_proc))
            if len(message_chunk) < 1:
                continue
            for record in message_chunk.shuffle(seed=int(seed)):
                messages = record.get("messages")
                if not isinstance(messages, list) or len(messages) < 1:
                    continue
                text = render_messages(messages, tokenizer).strip()
                if text:
                    yielded_text = True
                    yield text
            continue

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
    for text in _iter_calibration_texts_for_source(alias=str(alias), seed=int(seed), tokenizer=tokenizer):
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


def _distill_dataset_cache_path(cache_dir: str, dataset_name: str, nsamples: int, seed: int) -> str:
    key = f"{dataset_name}|{int(nsamples)}|{int(seed)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(str(cache_dir), f"distill_train_cache_{digest}")


def _distill_dataset_cache_ready(cache_path: str) -> bool:
    return (
        os.path.isdir(cache_path)
        and os.path.isfile(os.path.join(cache_path, "source_stats.json"))
        and os.path.isfile(os.path.join(cache_path, "dataset_mix_spec.txt"))
        and os.path.isdir(os.path.join(cache_path, "dataset"))
    )


def _save_distill_dataset_cache(
    cache_path: str,
    dataset_mix_spec: str,
    source_stats: List[Dict[str, object]],
    train_ds,
) -> None:
    dataset_dir = os.path.join(cache_path, "dataset")
    os.makedirs(cache_path, exist_ok=True)
    train_ds.save_to_disk(dataset_dir)
    with open(os.path.join(cache_path, "source_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(source_stats, handle, ensure_ascii=False)
    with open(os.path.join(cache_path, "dataset_mix_spec.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(dataset_mix_spec))


def _load_distill_dataset_cache(cache_path: str) -> Tuple[str, List[Dict[str, object]], object]:
    if load_from_disk is None:
        raise ImportError("未安装 datasets。请先安装：pip install datasets")
    with open(os.path.join(cache_path, "dataset_mix_spec.txt"), "r", encoding="utf-8") as handle:
        dataset_mix_spec = handle.read()
    with open(os.path.join(cache_path, "source_stats.json"), "r", encoding="utf-8") as handle:
        source_stats = json.load(handle)
    train_ds = load_from_disk(os.path.join(cache_path, "dataset"))
    return str(dataset_mix_spec), list(source_stats), train_ds


def prepare_distill_datasets(
    dataset_name: str,
    *,
    nsamples: int,
    seed: int,
    cache_dir: Optional[str] = None,
):
    ensure_distill_dataset_stack_available()
    if "=" not in str(dataset_name):
        raise ValueError(
            "--distill_dataset only accepts ratio-style dataset specs, for example "
            "'wiki=1.0', 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
        )

    cache_path = None
    if cache_dir and str(cache_dir).strip():
        cache_path = _distill_dataset_cache_path(str(cache_dir), str(dataset_name), int(nsamples), int(seed))

    if cache_path is not None:
        from train_utils.lora_utils import (
            distill_distributed_barrier,
            ensure_distill_process_group_initialized,
            is_distill_distributed,
            is_distill_main_process,
        )

        if is_distill_distributed():
            ensure_distill_process_group_initialized()
            if is_distill_main_process() and not _distill_dataset_cache_ready(cache_path):
                dataset_mix_spec, source_stats, train_ds = _prepare_lora_mixed_dataset(
                    str(dataset_name),
                    nsamples=int(nsamples),
                    seed=int(seed),
                )
                _save_distill_dataset_cache(cache_path, dataset_mix_spec, source_stats, train_ds)
            distill_distributed_barrier()
            if not _distill_dataset_cache_ready(cache_path):
                raise RuntimeError(f"Distill dataset cache is missing after rank-0 preparation: {cache_path}")
            dataset_mix_spec, source_stats, train_ds = _load_distill_dataset_cache(cache_path)
            return dataset_mix_spec, source_stats, train_ds, None, None

        if _distill_dataset_cache_ready(cache_path):
            dataset_mix_spec, source_stats, train_ds = _load_distill_dataset_cache(cache_path)
            return dataset_mix_spec, source_stats, train_ds, None, None

    dataset_mix_spec, source_stats, train_ds = _prepare_lora_mixed_dataset(
        str(dataset_name),
        nsamples=int(nsamples),
        seed=int(seed),
    )
    if cache_path is not None:
        _save_distill_dataset_cache(cache_path, dataset_mix_spec, source_stats, train_ds)
    return dataset_mix_spec, source_stats, train_ds, None, None
