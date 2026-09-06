"""Unified model-level distill/SFT/LM dataset entry for CAT and E2E."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, IterableDataset

from e2e_common.data import (
    DatasetMixSourcePreset,
    _normalize_edgerazor_messages,
    _record_to_sft_segments,
    _record_to_text,
)
from train_utils.config.configs import DistillDataConfig

IGNORE_ID = -100
FORMATTING_VERSION = "vaellm_distill_canonical_v1"

_NATIVE_MESSAGE_FORMATS = frozenset({"edgerazor_messages", "longalign_chat"})
_INSTRUCTION_SFT_FORMATS = frozenset({"openorca", "alpaca", "race_mcqa", "sciq_qa"})


@dataclass(frozen=True)
class DistillDatasetBundle:
    train_dataset: Union[Dataset, IterableDataset]
    eval_dataset: Optional[object]
    dataset_mix_spec: Optional[str]
    source_stats: List[Dict[str, object]]
    is_iterable: bool
    cache_key: Tuple[object, ...]
    group_by_length: bool


def build_distill_data_collator(
    tokenizer,
    *,
    model_max_length: int,
    dynamic_padding: bool = True,
):
    from transformers import DataCollatorForSeq2Seq

    max_len = int(model_max_length)
    if max_len < 2:
        raise ValueError(f"model_max_length must be >= 2, got {model_max_length}.")
    if bool(dynamic_padding):
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
        max_length=max_len,
        label_pad_token_id=IGNORE_ID,
        return_tensors="pt",
    )


def tokenizer_identity(tokenizer) -> Tuple[str, str]:
    name = str(
        getattr(tokenizer, "name_or_path", None)
        or getattr(tokenizer, "name", None)
        or type(tokenizer).__name__
    )
    revision = "unknown"
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict) and init_kwargs.get("revision") is not None:
        revision = str(init_kwargs.get("revision"))
    elif getattr(tokenizer, "revision", None) is not None:
        revision = str(getattr(tokenizer, "revision"))
    return name, revision


def distill_dataset_cache_key(data_config: DistillDataConfig, tokenizer) -> Tuple[object, ...]:
    cfg = data_config
    if cfg.dataset_mix:
        source_identity: object = str(cfg.dataset_mix)
    elif cfg.train_file:
        source_identity = ("train_file", str(cfg.train_file), str(cfg.text_field))
    else:
        source_identity = None
    tok_name, tok_revision = tokenizer_identity(tokenizer)
    return (
        source_identity,
        str(cfg.dataset_task),
        int(cfg.model_max_length),
        int(cfg.data_seed),
        tok_name,
        tok_revision,
        FORMATTING_VERSION,
    )


def _to_tensor_sample(
    input_ids: Sequence[int],
    labels: Sequence[int],
) -> Dict[str, torch.Tensor]:
    ids = [int(v) for v in input_ids]
    labs = [int(v) for v in labels]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(ids), dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
    }


def _normalize_terminal_eos_and_truncate(
    input_ids: Sequence[int],
    response_mask: Sequence[bool],
    *,
    eos_token_id: int,
    model_max_length: int,
    task: str,
) -> Optional[Tuple[List[int], List[bool]]]:
    max_len = int(model_max_length)
    if max_len < 2:
        raise ValueError(f"model_max_length must be >= 2, got {model_max_length}.")
    if len(input_ids) != len(response_mask):
        raise ValueError("input_ids and response_mask length mismatch.")

    ids = [int(v) for v in input_ids]
    mask = [bool(v) for v in response_mask]
    eos_id = int(eos_token_id)

    has_terminal_eos = bool(ids) and ids[-1] == eos_id
    if has_terminal_eos:
        payload = ids[:-1]
        payload_mask = mask[:-1]
    else:
        payload = ids
        payload_mask = mask

    if len(payload) >= max_len:
        keep = max_len - 1
        payload = payload[:keep]
        payload_mask = payload_mask[:keep]
    out_ids = list(payload) + [eos_id]
    out_mask = list(payload_mask) + [True]

    if str(task).strip().lower() == "sft" and not any(payload_mask):
        return None
    return out_ids, out_mask


def _labels_from_mask(input_ids: Sequence[int], response_mask: Sequence[bool], *, task: str) -> List[int]:
    task_norm = str(task).strip().lower()
    if task_norm == "lm":
        return [int(v) for v in input_ids]
    if task_norm != "sft":
        raise ValueError(f"Unsupported dataset task: {task!r}.")
    return [
        int(token_id) if bool(trainable) else IGNORE_ID
        for token_id, trainable in zip(input_ids, response_mask)
    ]


def _tokenize_text(tokenizer, text: str, *, return_offsets: bool = False):
    kwargs = {
        "add_special_tokens": False,
        "return_attention_mask": False,
        "return_token_type_ids": False,
    }
    if return_offsets:
        kwargs["return_offsets_mapping"] = True
    return tokenizer(str(text), **kwargs)


def _encode_ids_only(tokenizer, text: str) -> List[int]:
    encoded = _tokenize_text(tokenizer, text, return_offsets=False)
    return [int(v) for v in encoded["input_ids"]]


def _try_assistant_mask_chat_template(
    tokenizer,
    messages: Sequence[Dict[str, str]],
) -> Optional[Tuple[List[int], List[bool]]]:
    try:
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except (TypeError, AttributeError):
        # Missing method or unsupported assistant-mask kwargs -> prefix fallback.
        return None
    if not isinstance(encoded, dict):
        return None
    input_ids = encoded.get("input_ids")
    mask = encoded.get("assistant_masks")
    if mask is None:
        mask = encoded.get("assistant_mask")
    if input_ids is None or mask is None:
        return None
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if isinstance(mask, list) and mask and isinstance(mask[0], list):
        mask = mask[0]
    ids = [int(v) for v in input_ids]
    response_mask = [bool(v) for v in mask]
    if len(response_mask) != len(ids):
        return None
    return ids, response_mask


def _encode_messages_prefix_boundary(
    tokenizer,
    messages: Sequence[Dict[str, str]],
) -> Tuple[List[int], List[bool]]:
    from e2e_common.chat_template_utils import render_messages

    full_text = render_messages(messages, tokenizer, add_generation_prompt=False)
    full_ids = _encode_ids_only(tokenizer, full_text)
    response_mask = [False] * len(full_ids)

    prefix_messages: List[Dict[str, str]] = []
    for message in messages:
        prefix_messages.append(dict(message))
        if str(message.get("role")) != "assistant":
            continue
        current_text = render_messages(prefix_messages, tokenizer, add_generation_prompt=False)
        current_ids = _encode_ids_only(tokenizer, current_text)
        if current_ids != full_ids[: len(current_ids)]:
            chatml_mask = _encode_chatml_response_mask(tokenizer, messages, full_text, full_ids)
            if chatml_mask is None:
                raise ValueError(
                    "Conversation prefix token sequence is not a strict prefix of the canonical full sequence."
                )
            return full_ids, chatml_mask
        prev_text = render_messages(prefix_messages[:-1], tokenizer, add_generation_prompt=True)
        prev_ids = _encode_ids_only(tokenizer, prev_text)
        if prev_ids != full_ids[: len(prev_ids)]:
            raise ValueError(
                "Conversation generation-prompt prefix is not a strict prefix of the canonical full sequence."
            )
        for pos in range(len(prev_ids), len(current_ids)):
            response_mask[pos] = True
    return full_ids, response_mask


def _encode_chatml_response_mask(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    full_text: str,
    full_ids: Sequence[int],
) -> Optional[List[bool]]:
    """Build exact response spans for context-sensitive ChatML templates.

    Qwen3 renders an assistant differently when it is the final turn, so a
    rendered multi-turn prefix is not necessarily a prefix of the complete
    conversation. The canonical ChatML role/end markers remain explicit in
    the complete rendered text; use their tokenizer offsets instead of
    assigning labels from a differently rendered prefix.
    """

    end_marker = "<|im_end|>\n"
    cursor = 0
    response_ranges: List[Tuple[int, int]] = []
    for message in messages:
        role = str(message.get("role", ""))
        role_marker = f"<|im_start|>{role}\n"
        role_start = full_text.find(role_marker, cursor)
        if role_start < 0:
            return None
        content_start = role_start + len(role_marker)
        turn_end = full_text.find(end_marker, content_start)
        if turn_end < 0:
            return None
        cursor = turn_end + len(end_marker)
        if role == "assistant":
            response_ranges.append((content_start, cursor))

    if not response_ranges:
        return None
    try:
        encoded = _tokenize_text(tokenizer, full_text, return_offsets=True)
    except TypeError:
        return None
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    offset_ids = [int(value) for value in encoded["input_ids"]]
    if offset_ids != [int(value) for value in full_ids] or len(offsets) != len(offset_ids):
        raise ValueError("ChatML offset tokenization does not match canonical input ids.")

    response_mask: List[bool] = []
    for raw_start, raw_end in offsets:
        start = int(raw_start)
        end = int(raw_end)
        response_mask.append(
            end > start
            and any(end > range_start and start < range_end for range_start, range_end in response_ranges)
        )
    return response_mask


def _normalize_longalign_messages(record: Dict[str, object]) -> Optional[List[Dict[str, str]]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    normalized: List[Dict[str, str]] = []
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            return None
        if role in {"user", "human"}:
            role = "user"
        elif role in {"assistant", "gpt"}:
            role = "assistant"
            has_assistant = True
        elif role == "system":
            role = "system"
        else:
            return None
        normalized.append({"role": role, "content": content})
    if not has_assistant:
        return None
    return normalized


def _encode_native_messages(
    record: Dict[str, object],
    tokenizer,
    *,
    text_format: str,
) -> Optional[Tuple[List[int], List[bool]]]:
    fmt = str(text_format).strip().lower()
    if fmt == "edgerazor_messages":
        messages = _normalize_edgerazor_messages(record)
    elif fmt == "longalign_chat":
        messages = _normalize_longalign_messages(record)
    else:
        raise ValueError(f"Unsupported native message format: {text_format!r}")
    if messages is None:
        return None
    masked = _try_assistant_mask_chat_template(tokenizer, messages)
    if masked is not None:
        return masked
    return _encode_messages_prefix_boundary(tokenizer, messages)


def _instruction_prompt_and_response(
    record: Dict[str, object],
    *,
    text_format: str,
) -> Optional[Tuple[str, str]]:
    segments = _record_to_sft_segments(record, text_format=text_format)
    if segments is None:
        return None
    prompt_parts: List[str] = []
    response_parts: List[str] = []
    for text, trainable in segments:
        if bool(trainable):
            response_parts.append(str(text))
        else:
            prompt_parts.append(str(text))
    prompt = "".join(prompt_parts)
    response = "".join(response_parts)
    if not response:
        return None
    return prompt, response


def _response_mask_from_offsets(
    offsets: Sequence[Tuple[int, int]],
    response_start: int,
    response_end: int,
) -> List[bool]:
    mask: List[bool] = []
    for start, end in offsets:
        if int(end) <= int(start):
            mask.append(False)
            continue
        mask.append(not (int(end) <= int(response_start) or int(start) >= int(response_end)))
    return mask


def _encode_instruction_qa(
    record: Dict[str, object],
    tokenizer,
    *,
    text_format: str,
) -> Optional[Tuple[List[int], List[bool]]]:
    rendered = _instruction_prompt_and_response(record, text_format=text_format)
    if rendered is None:
        return None
    prompt, response = rendered
    full_text = f"{prompt}{response}"
    try:
        encoded = _tokenize_text(tokenizer, full_text, return_offsets=True)
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise TypeError("missing offset_mapping")
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        input_ids = [int(v) for v in encoded["input_ids"]]
        response_start = len(prompt)
        response_end = len(full_text)
        response_mask = _response_mask_from_offsets(offsets, response_start, response_end)
        if len(response_mask) != len(input_ids):
            raise ValueError("offset_mapping length mismatch.")
        return input_ids, response_mask
    except TypeError:
        prompt_ids = _encode_ids_only(tokenizer, prompt)
        full_ids = _encode_ids_only(tokenizer, full_text)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Prompt token ids are not a strict prefix of full-sequence ids; "
                "cannot safely build SFT response mask without offset mapping."
            )
        response_mask = [False] * len(prompt_ids) + [True] * (len(full_ids) - len(prompt_ids))
        return full_ids, response_mask


def _encode_plain_lm_text(
    record: Dict[str, object],
    tokenizer,
    *,
    text_field: str,
    text_format: str,
) -> Optional[Tuple[List[int], List[bool]]]:
    text = _record_to_text(record, text_field=str(text_field), text_format=str(text_format))
    if text is None or not str(text).strip():
        return None
    input_ids = _encode_ids_only(tokenizer, str(text))
    return input_ids, [True] * len(input_ids)


def encode_canonical_record(
    record: Dict[str, object],
    tokenizer,
    *,
    text_format: str,
    text_field: str,
    task: str,
    model_max_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    task_norm = str(task).strip().lower()
    if task_norm not in {"lm", "sft"}:
        raise ValueError(f"Unsupported dataset task: {task!r}.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is required for distill encoding.")

    fmt = str(text_format).strip().lower()
    if fmt in _NATIVE_MESSAGE_FORMATS:
        encoded = _encode_native_messages(record, tokenizer, text_format=fmt)
    elif fmt in _INSTRUCTION_SFT_FORMATS:
        encoded = _encode_instruction_qa(record, tokenizer, text_format=fmt)
    elif task_norm == "lm":
        encoded = _encode_plain_lm_text(
            record,
            tokenizer,
            text_field=text_field,
            text_format=fmt,
        )
    else:
        raise ValueError(
            f"SFT task does not support text_format={text_format!r}. "
            f"Supported: {sorted(_NATIVE_MESSAGE_FORMATS | _INSTRUCTION_SFT_FORMATS)}."
        )
    if encoded is None:
        return None
    input_ids, response_mask = encoded
    normalized = _normalize_terminal_eos_and_truncate(
        input_ids,
        response_mask,
        eos_token_id=int(eos_token_id),
        model_max_length=int(model_max_length),
        task=task_norm,
    )
    if normalized is None:
        return None
    final_ids, final_mask = normalized
    labels = _labels_from_mask(final_ids, final_mask, task=task_norm)
    return _to_tensor_sample(final_ids, labels)


def _validate_presets_for_task(presets: Sequence[DatasetMixSourcePreset], task: str) -> None:
    task_norm = str(task).strip().lower()
    for preset in presets:
        if task_norm == "lm" and not bool(preset.supports_lm):
            raise ValueError(
                f"dataset source {preset.alias!r} does not support dataset_task=lm "
                "(supports_lm=False)."
            )
        if task_norm == "sft" and not bool(preset.supports_sft):
            raise ValueError(
                f"dataset source {preset.alias!r} does not support dataset_task=sft "
                "(supports_sft=False)."
            )


class _CanonicalMapDataset(Dataset):
    """Map-style lazy encoding: tokenize only on __getitem__, never at init.

    Invalid records that encode to None raise at access time. Callers that must
    skip invalid SFT records should use the lazy iterable wrappers instead.
    """

    def __init__(
        self,
        raw_dataset,
        tokenizer,
        *,
        task: str,
        preset: DatasetMixSourcePreset,
        model_max_length: int,
    ) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.task = str(task)
        self.preset = preset
        self.model_max_length = int(model_max_length)

    def __len__(self) -> int:
        return int(len(self.raw_dataset))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        raw_index = int(index)
        sample = encode_canonical_record(
            dict(self.raw_dataset[raw_index]),
            self.tokenizer,
            text_format=str(self.preset.text_format),
            text_field=str(self.preset.text_field),
            task=self.task,
            model_max_length=self.model_max_length,
        )
        if sample is None:
            raise RuntimeError(
                f"Canonical encode returned None for map-style index {raw_index}. "
                "Use a lazy iterable dataset when invalid SFT records must be skipped."
            )
        return sample


class _CanonicalMixedIterableDataset(IterableDataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        *,
        task: str,
        presets: Sequence[DatasetMixSourcePreset],
        model_max_length: int,
    ) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.task = str(task)
        self.presets = list(presets)
        self.model_max_length = int(model_max_length)
        raw_source_count = len(getattr(raw_dataset, "raw_datasets", ()))
        if raw_source_count != len(self.presets):
            raise ValueError(
                "Mix presets must align with indexed raw sources. "
                f"Got {len(self.presets)} presets and {raw_source_count} raw sources."
            )

    def __iter__(self):
        from torch.utils.data import get_worker_info

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
            sample = encode_canonical_record(
                dict(record),
                self.tokenizer,
                text_format=str(preset.text_format),
                text_field=str(preset.text_field),
                task=self.task,
                model_max_length=self.model_max_length,
            )
            # Only renderer/normalizer-returned None is skippable; structural
            # errors from encode_canonical_record must propagate.
            if sample is not None:
                yield sample


class _LazySinglePresetIterable(IterableDataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        *,
        task: str,
        preset: DatasetMixSourcePreset,
        model_max_length: int,
    ) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.task = str(task)
        self.preset = preset
        self.model_max_length = int(model_max_length)

    def __iter__(self):
        from e2e_common.lazy_datasets import _iter_raw_records_for_worker

        for record in _iter_raw_records_for_worker(self.raw_dataset):
            sample = encode_canonical_record(
                dict(record),
                self.tokenizer,
                text_format=str(self.preset.text_format),
                text_field=str(self.preset.text_field),
                task=self.task,
                model_max_length=self.model_max_length,
            )
            if sample is not None:
                yield sample


def build_distill_dataset(
    data_config: DistillDataConfig,
    tokenizer,
    *,
    raw_dataset_cache: Optional[Dict[tuple, object]] = None,
) -> DistillDatasetBundle:
    cfg = data_config
    cfg.validate()
    from e2e_common.lazy_datasets import (
        _load_indexed_raw_mix,
        is_iterable_training_dataset,
    )

    cache_key = distill_dataset_cache_key(cfg, tokenizer)
    group_by_length = bool(cfg.group_by_length)

    if cfg.train_file:
        if str(cfg.dataset_task) != "lm":
            raise ValueError("--train_file currently supports dataset_task=lm only.")
        from datasets import DatasetDict, load_dataset
        from e2e_common.data import _resolve_local_dataset_loader

        loader = _resolve_local_dataset_loader(str(cfg.train_file))
        raw = load_dataset(loader, data_files={"train": str(cfg.train_file)})
        if not isinstance(raw, DatasetDict):
            raise RuntimeError(f"Expected DatasetDict from local loader, got {type(raw)}")
        train_raw = raw["train"]
        pseudo_preset = DatasetMixSourcePreset(
            alias="train_file",
            path=str(cfg.train_file),
            config=None,
            train_split="train",
            eval_split=None,
            text_field=str(cfg.text_field),
            text_format="text",
            supports_lm=True,
            supports_sft=False,
        )
        train_dataset = _CanonicalMapDataset(
            train_raw,
            tokenizer,
            task="lm",
            preset=pseudo_preset,
            model_max_length=int(cfg.model_max_length),
        )
        is_iterable = bool(is_iterable_training_dataset(train_dataset))
        if is_iterable:
            group_by_length = False
        return DistillDatasetBundle(
            train_dataset=train_dataset,
            eval_dataset=None,
            dataset_mix_spec=None,
            source_stats=[
                {
                    "alias": "train_file",
                    "weight": 1.0,
                    "path": str(cfg.train_file),
                    "is_iterable": bool(is_iterable),
                }
            ],
            is_iterable=bool(is_iterable),
            cache_key=cache_key,
            group_by_length=bool(group_by_length),
        )

    if not cfg.dataset_mix:
        raise ValueError("build_distill_dataset requires dataset_mix or train_file.")

    from e2e_common.data import normalize_dataset_mix_spec
    from e2e_common import data as data_module

    sources, _weights, _normalized = normalize_dataset_mix_spec(str(cfg.dataset_mix))
    presets_preview = [data_module.DATASET_MIX_SOURCE_PRESETS[str(alias)] for alias in sources]
    _validate_presets_for_task(presets_preview, str(cfg.dataset_task))

    normalized_spec, source_stats, raw_dataset, presets, mix_kind = _load_indexed_raw_mix(
        str(cfg.dataset_mix),
        seed=int(cfg.data_seed),
        raw_dataset_cache=raw_dataset_cache,
    )
    _validate_presets_for_task(presets, str(cfg.dataset_task))
    task_norm = str(cfg.dataset_task).strip().lower()
    raw_is_iterable = mix_kind == "mix" or is_iterable_training_dataset(raw_dataset)
    # SFT may need to skip invalid records (encode -> None). Prefer lazy
    # iterable and disable group_by_length instead of eager valid-index scans.
    use_lazy_iterable = bool(raw_is_iterable) or task_norm == "sft"
    if mix_kind == "mix":
        train_dataset = _CanonicalMixedIterableDataset(
            raw_dataset,
            tokenizer,
            task=task_norm,
            presets=presets,
            model_max_length=int(cfg.model_max_length),
        )
        group_by_length = False
        is_iterable = True
    elif use_lazy_iterable:
        train_dataset = _LazySinglePresetIterable(
            raw_dataset,
            tokenizer,
            task=task_norm,
            preset=presets[0],
            model_max_length=int(cfg.model_max_length),
        )
        group_by_length = False
        is_iterable = True
    else:
        train_dataset = _CanonicalMapDataset(
            raw_dataset,
            tokenizer,
            task=task_norm,
            preset=presets[0],
            model_max_length=int(cfg.model_max_length),
        )
        is_iterable = False

    for source_info in source_stats:
        source_info["is_iterable"] = bool(is_iterable)

    return DistillDatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=None,
        dataset_mix_spec=str(normalized_spec),
        source_stats=source_stats,
        is_iterable=bool(is_iterable),
        cache_key=cache_key,
        group_by_length=bool(group_by_length),
    )
