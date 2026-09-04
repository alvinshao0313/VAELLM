"""
EdgeRazor-style lazy dataset loading for VAELLM.

Raw records are loaded via HuggingFace ``load_dataset``; tokenization and label
masking happen in ``__getitem__`` / ``__iter__`` (no bulk ``dataset.map``).
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

try:
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset
except ImportError:
    HFDataset = None
    HFIterableDataset = None

from e2e_common.data import (
    DATASET_MIX_SOURCE_PRESETS,
    DatasetMixSourcePreset,
    _load_preset_raw_datasets,
    _normalize_edgerazor_messages,
    _record_to_text,
    normalize_dataset_mix_spec,
)

IGNORE_ID = -100
_INDEX_SCHEDULER_CHOICE_BATCH_SIZE = 1000

RawDataset = Union["HFDataset", "HFIterableDataset"]


def default_dataloader_num_workers() -> int:
    return 16


def build_edgerazor_data_collator(
    tokenizer,
    *,
    max_seq_len: int,
    dynamic_padding: bool = False,
):
    from transformers import DataCollatorForSeq2Seq

    max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}.")

    if bool(dynamic_padding):
        if max_seq_len % 8 != 0:
            raise ValueError(
                "dynamic padding requires max_seq_len to be divisible by 8, "
                f"got {max_seq_len}."
            )
        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_ID,
            return_tensors="pt",
        )

    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_seq_len,
        label_pad_token_id=IGNORE_ID,
        return_tensors="pt",
    )


def encode_edgerazor_messages_record(
    record: Dict[str, object],
    tokenizer,
    *,
    max_seq_len: int,
    add_system_prompt: bool = False,
) -> Dict[str, torch.Tensor]:
    """EdgeRazor ReasoningDataset.__getitem__ encoding (with VAELLM message normalize)."""
    messages = _normalize_edgerazor_messages(record)
    if messages is None:
        raise ValueError("Record has no valid edgerazor messages.")

    conversation = list(messages)
    if add_system_prompt:
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful and harmless AI assistant.",
            }
        ] + conversation

    full_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [IGNORE_ID] * len(input_ids)

    temp_messages: List[Dict[str, str]] = []
    for msg in conversation:
        temp_messages.append(msg)
        if msg.get("role") != "assistant":
            continue

        current_text = tokenizer.apply_chat_template(
            temp_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        current_ids = tokenizer.encode(current_text, add_special_tokens=False)

        prev_text = tokenizer.apply_chat_template(
            temp_messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prev_ids = tokenizer.encode(prev_text, add_special_tokens=False)

        start_pos = len(prev_ids)
        end_pos = len(current_ids)
        for pos in range(start_pos, min(end_pos, len(labels))):
            labels[pos] = input_ids[pos]

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is required for lazy message encoding.")

    if len(input_ids) > int(max_seq_len) - 1:
        input_ids = input_ids[: int(max_seq_len) - 1]
        labels = labels[: int(max_seq_len) - 1]

    input_ids.append(int(eos_token_id))
    labels.append(int(eos_token_id))
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def encode_text_lm_record(
    record: Dict[str, object],
    tokenizer,
    *,
    max_seq_len: int,
    text_field: str,
    text_format: str,
) -> Dict[str, torch.Tensor]:
    text = _record_to_text(record, text_field=str(text_field), text_format=str(text_format))
    if text is None or not str(text).strip():
        raise ValueError("Record has no usable text for LM lazy encoding.")

    encoded = tokenizer(
        str(text),
        add_special_tokens=False,
        return_attention_mask=True,
        return_token_type_ids=False,
        truncation=True,
        max_length=int(max_seq_len),
    )
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    attention_mask = [int(value) for value in encoded.get("attention_mask", [1] * len(input_ids))]
    labels = list(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class ReasoningDataset(Dataset):
    """EdgeRazor ReasoningDataset with self.dataset assignment fix."""

    def __init__(
        self,
        dataset,
        tokenizer,
        *,
        max_seq_len: int = 4096,
        add_system_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.add_system_prompt = bool(add_system_prompt)
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return int(len(self.dataset))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[int(index)]
        return encode_edgerazor_messages_record(
            item,
            self.tokenizer,
            max_seq_len=self.max_seq_len,
            add_system_prompt=self.add_system_prompt,
        )


class LazyTextLMDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        *,
        max_seq_len: int,
        text_field: str,
        text_format: str,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.text_field = str(text_field)
        self.text_format = str(text_format)

    def __len__(self) -> int:
        return int(len(self.dataset))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return encode_text_lm_record(
            self.dataset[int(index)],
            self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_field=self.text_field,
            text_format=self.text_format,
        )


class LazySFTDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        *,
        max_seq_len: int,
        text_format: str,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.text_format = str(text_format)

    def __len__(self) -> int:
        return int(len(self.dataset))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        from e2e_common.data import _encode_sft_segments, _record_to_sft_segments

        record = self.dataset[int(index)]
        segments = _record_to_sft_segments(record, text_format=self.text_format)
        if segments is None:
            raise ValueError(f"SFT record is not usable for format {self.text_format!r}.")
        input_ids, attention_mask, labels = _encode_sft_segments(
            segments,
            self.tokenizer,
            block_size=self.max_seq_len,
        )
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            attention_mask = attention_mask[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _iter_raw_records_for_worker(raw_dataset):
    worker_info = get_worker_info()

    if hasattr(raw_dataset, "iter_worker"):
        if worker_info is None:
            yield from raw_dataset.iter_worker(worker_id=0, num_workers=1)
        else:
            yield from raw_dataset.iter_worker(
                worker_id=int(worker_info.id),
                num_workers=int(worker_info.num_workers),
            )
        return

    if worker_info is None:
        yield from raw_dataset
        return

    worker_id = int(worker_info.id)
    num_workers = int(worker_info.num_workers)

    raw_len = None
    if hasattr(raw_dataset, "__getitem__"):
        try:
            raw_len = int(len(raw_dataset))
        except (TypeError, NotImplementedError):
            raw_len = None

    if raw_len is not None:
        for index in range(worker_id, raw_len, num_workers):
            yield raw_dataset[index]
        return

    for index, record in enumerate(raw_dataset):
        if index % num_workers == worker_id:
            yield record


class _LazyMixedTextLMIterableDataset(IterableDataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        *,
        max_seq_len: int,
        presets: Sequence[DatasetMixSourcePreset],
    ) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.presets = list(presets)
        raw_source_count = len(getattr(raw_dataset, "raw_datasets", ()))
        if raw_source_count != len(self.presets):
            raise ValueError(
                "LM mix presets must align with _IndexedMixedRawStream.raw_datasets. "
                f"Got {len(self.presets)} presets and {raw_source_count} raw sources."
            )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        for source_idx, record in self.raw_dataset.iter_worker_with_source(
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            preset = self.presets[int(source_idx)]
            try:
                yield encode_text_lm_record(
                    dict(record),
                    self.tokenizer,
                    max_seq_len=self.max_seq_len,
                    text_field=str(preset.text_field),
                    text_format=str(preset.text_format),
                )
            except (ValueError, KeyError):
                continue


class _LazyPresetIterableDataset(IterableDataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        *,
        max_seq_len: int,
        task: str,
        preset: DatasetMixSourcePreset,
        seed: int,
        add_system_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.task = str(task).strip().lower()
        self.preset = preset
        self.seed = int(seed)
        self.add_system_prompt = bool(add_system_prompt)

    def _encode_record(self, record: Dict[str, object]) -> Dict[str, torch.Tensor]:
        preset = self.preset
        if self.task in {"messages", "sft"}:
            if str(preset.text_format) == "edgerazor_messages":
                return encode_edgerazor_messages_record(
                    record,
                    self.tokenizer,
                    max_seq_len=self.max_seq_len,
                    add_system_prompt=self.add_system_prompt,
                )
            from e2e_common.data import _record_to_sft_segments, _encode_sft_segments

            segments = _record_to_sft_segments(record, text_format=str(preset.text_format))
            if segments is None:
                raise ValueError(f"SFT record is not usable for format {preset.text_format!r}.")
            input_ids, attention_mask, labels = _encode_sft_segments(
                segments,
                self.tokenizer,
                block_size=self.max_seq_len,
            )
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                attention_mask = attention_mask[: self.max_seq_len]
                labels = labels[: self.max_seq_len]
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        if self.task == "lm":
            return encode_text_lm_record(
                record,
                self.tokenizer,
                max_seq_len=self.max_seq_len,
                text_field=str(preset.text_field),
                text_format=str(preset.text_format),
            )
        raise ValueError(f"Unsupported lazy dataset task: {self.task!r}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for record in _iter_raw_records_for_worker(self.raw_dataset):
            try:
                yield self._encode_record(dict(record))
            except (ValueError, KeyError):
                continue


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(float(weight) for weight in weights))
    if total <= 0.0:
        raise ValueError("Dataset mix weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def _build_permutation_indices(length: int, *, seed: int) -> np.ndarray:
    length = int(length)
    if length < 1:
        raise ValueError(f"Permutation length must be >= 1, got {length}.")

    if length <= int(np.iinfo(np.int32).max):
        dtype = np.int32
    else:
        dtype = np.int64

    indices = np.arange(length, dtype=dtype)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    indices.flags.writeable = False
    return indices


class _PermutedRawDatasetView(Dataset):
    def __init__(self, raw_dataset, permutation: np.ndarray) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset
        self.permutation = permutation
        if int(len(self.raw_dataset)) != int(len(self.permutation)):
            raise ValueError(
                f"Raw dataset length {len(self.raw_dataset)} does not match "
                f"permutation length {len(self.permutation)}."
            )

    def __len__(self) -> int:
        return int(len(self.permutation))

    def __getitem__(self, index: int):
        raw_index = int(self.permutation[int(index)])
        return self.raw_dataset[raw_index]


def _iter_random_source_indices(
    *,
    num_sources: int,
    probabilities: Sequence[float],
    seed: int,
) -> Iterator[int]:
    num_sources = int(num_sources)
    if num_sources < 1:
        raise ValueError("num_sources must be >= 1.")
    if len(probabilities) != num_sources:
        raise ValueError(
            f"probabilities length {len(probabilities)} != num_sources {num_sources}."
        )

    probs = np.asarray([float(value) for value in probabilities], dtype=np.float64)
    rng = np.random.default_rng(int(seed))

    while True:
        sampled = rng.choice(
            num_sources,
            size=_INDEX_SCHEDULER_CHOICE_BATCH_SIZE,
            p=probs,
        )
        for source_idx in sampled:
            yield int(source_idx)


class _IndexedMixedRawStream:
    def __init__(
        self,
        raw_datasets: Sequence[object],
        permutations: Sequence[np.ndarray],
        probabilities: Sequence[float],
        *,
        seed: int,
    ) -> None:
        self.raw_datasets = tuple(raw_datasets)
        self.permutations = tuple(permutations)
        self.probabilities = tuple(float(value) for value in probabilities)
        self.seed = int(seed)

        source_count = len(self.raw_datasets)
        if source_count < 2:
            raise ValueError("_IndexedMixedRawStream requires at least two sources.")
        if len(self.permutations) != source_count:
            raise ValueError("permutations/source count mismatch.")
        if len(self.probabilities) != source_count:
            raise ValueError("probabilities/source count mismatch.")
        for raw_dataset, permutation in zip(self.raw_datasets, self.permutations):
            if int(len(raw_dataset)) != int(len(permutation)):
                raise ValueError("raw dataset/permutation length mismatch.")

    def iter_worker_with_source(self, *, worker_id: int, num_workers: int):
        worker_id = int(worker_id)
        num_workers = int(num_workers)
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}.")
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(
                f"worker_id must be in [0, {num_workers}), got {worker_id}."
            )

        source_count = len(self.raw_datasets)
        cursors = [0 for _ in range(source_count)]
        exhausted_once = [False for _ in range(source_count)]
        exhausted_count = 0
        global_position = 0

        source_indices = _iter_random_source_indices(
            num_sources=source_count,
            probabilities=self.probabilities,
            seed=self.seed,
        )

        while exhausted_count < source_count:
            source_idx = next(source_indices)
            permutation = self.permutations[source_idx]
            cursor = int(cursors[source_idx])
            raw_index = int(permutation[cursor])

            cursor += 1
            if cursor >= int(len(permutation)):
                cursor = 0
                if not exhausted_once[source_idx]:
                    exhausted_once[source_idx] = True
                    exhausted_count += 1
            cursors[source_idx] = cursor

            if global_position % num_workers == worker_id:
                yield source_idx, self.raw_datasets[source_idx][raw_index]

            global_position += 1

    def iter_worker(self, *, worker_id: int, num_workers: int):
        for _source_idx, record in self.iter_worker_with_source(
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            yield record

    def __iter__(self):
        yield from self.iter_worker(worker_id=0, num_workers=1)


def _build_source_stats(
    alias: str,
    preset: DatasetMixSourcePreset,
    raw_rows: int,
    weight: float,
) -> Dict[str, object]:
    return {
        "alias": str(alias),
        "weight": float(weight),
        "path": str(preset.path),
        "config": None if preset.config is None else str(preset.config),
        "train_split": str(preset.train_split),
        "raw_rows": int(raw_rows),
        "text_rows": int(raw_rows),
        "target_rows": int(raw_rows),
        "actual_rows": int(raw_rows),
        "processed_raw_rows": int(raw_rows),
        "limited_preprocessing": False,
        "sampling_policy": "lazy_streaming",
    }


def _raw_dataset_cache_key(preset: DatasetMixSourcePreset) -> tuple:
    return (
        str(preset.alias),
        str(preset.path),
        None if preset.config is None else str(preset.config),
        str(preset.train_split),
    )


def _load_indexed_raw_mix(
    dataset_mix_spec: str,
    *,
    seed: int,
    raw_dataset_cache: Optional[Dict[tuple, RawDataset]] = None,
) -> Tuple[str, List[Dict[str, object]], RawDataset, List[DatasetMixSourcePreset], str]:
    sources, weights, normalized_spec = normalize_dataset_mix_spec(dataset_mix_spec)
    normalized_weights = _normalize_weights(weights)

    raw_datasets: List[RawDataset] = []
    permutations: List[np.ndarray] = []
    source_stats: List[Dict[str, object]] = []
    presets: List[DatasetMixSourcePreset] = []

    for alias, weight in zip(sources, normalized_weights):
        preset = DATASET_MIX_SOURCE_PRESETS[str(alias)]
        cache_key = _raw_dataset_cache_key(preset)

        if raw_dataset_cache is not None and cache_key in raw_dataset_cache:
            train_raw = raw_dataset_cache[cache_key]
        else:
            train_raw, _eval_raw = _load_preset_raw_datasets(preset)
            if raw_dataset_cache is not None:
                raw_dataset_cache[cache_key] = train_raw

        if isinstance(train_raw, HFIterableDataset):
            raise TypeError(
                f"Indexed lazy scheduler requires a map-style dataset for source '{alias}', "
                f"got {type(train_raw).__name__}."
            )
        if not hasattr(train_raw, "__len__") or not hasattr(train_raw, "__getitem__"):
            raise TypeError(
                f"Indexed lazy scheduler requires __len__ and __getitem__ for source '{alias}'."
            )

        raw_rows = int(len(train_raw))
        if raw_rows < 1:
            raise ValueError(f"Dataset mix source '{alias}' is empty.")

        permutation = _build_permutation_indices(raw_rows, seed=int(seed))

        raw_datasets.append(train_raw)
        permutations.append(permutation)
        source_stats.append(
            _build_source_stats(str(alias), preset, raw_rows, float(weight))
        )
        presets.append(preset)
        source_stats[-1]["weight"] = float(weight)

    if len(raw_datasets) == 1:
        view = _PermutedRawDatasetView(raw_datasets[0], permutations[0])
        return str(normalized_spec), source_stats, view, presets, str(sources[0])

    mixed = _IndexedMixedRawStream(
        raw_datasets,
        permutations,
        normalized_weights,
        seed=int(seed),
    )
    return str(normalized_spec), source_stats, mixed, presets, "mix"


def build_mixed_lazy_dataset(
    dataset_mix_spec: str,
    *,
    task: str,
    tokenizer,
    max_seq_len: int,
    seed: int,
    add_system_prompt: bool = False,
    raw_dataset_cache: Optional[Dict[tuple, RawDataset]] = None,
) -> Tuple[str, List[Dict[str, object]], Union[Dataset, IterableDataset], bool]:
    normalized_spec, source_stats, raw_dataset, presets, mix_kind = _load_indexed_raw_mix(
        dataset_mix_spec,
        seed=int(seed),
        raw_dataset_cache=raw_dataset_cache,
    )
    task_norm = str(task).strip().lower()
    is_iterable = mix_kind == "mix" or isinstance(raw_dataset, HFIterableDataset)

    if not is_iterable:
        preset = presets[0]
        if task_norm in {"messages", "sft"} and str(preset.text_format) == "edgerazor_messages":
            dataset: Union[Dataset, IterableDataset] = ReasoningDataset(
                raw_dataset,
                tokenizer,
                max_seq_len=int(max_seq_len),
                add_system_prompt=bool(add_system_prompt),
            )
        elif task_norm in {"messages", "sft"}:
            dataset = LazySFTDataset(
                raw_dataset,
                tokenizer,
                max_seq_len=int(max_seq_len),
                text_format=str(preset.text_format),
            )
        elif task_norm == "lm":
            dataset = LazyTextLMDataset(
                raw_dataset,
                tokenizer,
                max_seq_len=int(max_seq_len),
                text_field=str(preset.text_field),
                text_format=str(preset.text_format),
            )
        else:
            raise ValueError(f"Unsupported lazy dataset task: {task_norm!r}")
        return normalized_spec, source_stats, dataset, is_iterable

    if task_norm == "lm" and mix_kind == "mix":
        dataset = _LazyMixedTextLMIterableDataset(
            raw_dataset,
            tokenizer,
            max_seq_len=int(max_seq_len),
            presets=presets,
        )
        return normalized_spec, source_stats, dataset, True

    if len(set(str(preset.text_format) for preset in presets)) != 1:
        raise ValueError(
            "Weighted lazy SFT/messages mix with multiple text_format values is not supported "
            "in one iterable dataset. Use a single-format mix or one source."
        )

    preset = presets[0]
    dataset = _LazyPresetIterableDataset(
        raw_dataset,
        tokenizer,
        max_seq_len=int(max_seq_len),
        task=task_norm,
        preset=preset,
        seed=int(seed),
        add_system_prompt=bool(add_system_prompt),
    )
    return normalized_spec, source_stats, dataset, True


def build_distill_lazy_dataset(
    dataset_mix_spec: str,
    *,
    tokenizer,
    max_seq_len: int,
    seed: int,
    raw_dataset_cache: Optional[Dict[tuple, RawDataset]] = None,
) -> Tuple[str, List[Dict[str, object]], Union[Dataset, IterableDataset], bool]:
    return build_mixed_lazy_dataset(
        dataset_mix_spec,
        task="messages",
        tokenizer=tokenizer,
        max_seq_len=int(max_seq_len),
        seed=int(seed),
        add_system_prompt=False,
        raw_dataset_cache=raw_dataset_cache,
    )


def build_single_file_lazy_dataset(
    *,
    train_file: str,
    task: str,
    tokenizer,
    max_seq_len: int,
    text_field: str = "text",
    text_format: str = "auto",
    max_train_samples: Optional[int] = None,
) -> Union[Dataset, IterableDataset]:
    from datasets import DatasetDict
    from e2e_common import data as data_module

    lower = str(train_file).strip().lower()
    if lower.endswith((".json", ".jsonl")):
        raw = data_module.load_dataset("json", data_files={"train": str(train_file)})
    elif lower.endswith(".txt"):
        raw = data_module.load_dataset("text", data_files={"train": str(train_file)})
    else:
        raise ValueError(f"Unsupported local dataset file extension: {train_file}")

    if not isinstance(raw, DatasetDict):
        raise RuntimeError(f"Expected DatasetDict from local loader, got {type(raw)}")
    train_raw = raw["train"]
    if max_train_samples is not None and int(max_train_samples) < len(train_raw):
        train_raw = train_raw.select(range(int(max_train_samples)))

    task_norm = str(task).strip().lower()
    if task_norm == "lm":
        return LazyTextLMDataset(
            train_raw,
            tokenizer,
            max_seq_len=int(max_seq_len),
            text_field=str(text_field),
            text_format=str(text_format),
        )
    raise ValueError(f"build_single_file_lazy_dataset only supports task=lm, got {task_norm!r}")


class LazyCalibrationTextStream:
    def __init__(
        self,
        dataset_mix_spec: str,
        *,
        tokenizer,
        seed: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.seed = int(seed)
        normalized_spec, _source_stats, raw_dataset, presets, _mix_kind = _load_indexed_raw_mix(
            dataset_mix_spec,
            seed=int(seed),
        )
        self.dataset_mix_spec = str(normalized_spec)
        self.raw_dataset = raw_dataset
        if len(set(str(preset.text_format) for preset in presets)) != 1:
            raise ValueError("Calibration lazy stream requires a single text_format across mix sources.")
        self.text_format = str(presets[0].text_format)
        self.text_field = str(presets[0].text_field)

    def iter_texts(self) -> Iterator[str]:
        from e2e_common.chat_template_utils import render_messages

        for record in self.raw_dataset:
            record_dict = dict(record)
            if self.text_format == "edgerazor_messages":
                messages = _normalize_edgerazor_messages(record_dict)
                if messages is None:
                    continue
                text = render_messages(messages, self.tokenizer).strip()
            else:
                text_value = _record_to_text(
                    record_dict,
                    text_field=self.text_field,
                    text_format=self.text_format,
                )
                text = "" if text_value is None else str(text_value).strip()
            if text:
                yield text


def build_calibration_input_ids_lazy(
    dataset_name: str,
    *,
    tokenizer,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> List[torch.Tensor]:
    dataset_mix_spec = str(dataset_name or "").strip()
    if not dataset_mix_spec:
        raise ValueError("--activation_calib_dataset must be set when dynamic calibration is enabled.")
    if "=" not in dataset_mix_spec:
        raise ValueError(
            "--activation_calib_dataset only accepts ratio-style dataset specs, for example "
            "'wiki=1.0', 'openorca=1.0' or 'openorca=0.5,fineweb_edu=0.5'."
        )

    target_blocks = int(nsamples)
    block_size = int(seqlen)
    if target_blocks < 0:
        raise ValueError(f"--activation_calib_nsamples must be >= 0, got {target_blocks}.")
    if block_size <= 0:
        raise ValueError(f"--activation_calib_seqlen must be > 0, got {block_size}.")
    if target_blocks == 0:
        return []

    stream = LazyCalibrationTextStream(
        dataset_mix_spec,
        tokenizer=tokenizer,
        seed=int(seed),
    )
    blocks: List[torch.Tensor] = []
    token_buffer: List[int] = []
    for text in stream.iter_texts():
        encoded = tokenizer(
            text + "\n\n",
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("Tokenizer output for calibration lazy stream is missing input_ids.")
        token_buffer.extend(int(token) for token in input_ids)
        while len(token_buffer) >= block_size and len(blocks) < target_blocks:
            blocks.append(torch.tensor(token_buffer[:block_size], dtype=torch.long).unsqueeze(0))
            del token_buffer[:block_size]
        if len(blocks) >= target_blocks:
            break

    if len(blocks) != target_blocks:
        raise ValueError(
            f"Calibration dataset mix does not contain enough tokens to build "
            f"{target_blocks} blocks of length {block_size}. Built only {len(blocks)} blocks."
        )
    return blocks


def dataset_length_or_none(dataset) -> Optional[int]:
    try:
        return int(len(dataset))
    except (TypeError, NotImplementedError):
        return None


def is_iterable_training_dataset(dataset) -> bool:
    if isinstance(dataset, IterableDataset):
        return True
    return dataset_length_or_none(dataset) is None
