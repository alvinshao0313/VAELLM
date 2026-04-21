import math
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset
from transformers import AutoTokenizer


@dataclass(frozen=True)
class DatasetMixSourcePreset:
    alias: str
    path: str
    config: Optional[str]
    train_split: str
    eval_split: Optional[str]
    text_field: str
    text_format: str


DATASET_MIX_SOURCE_PRESETS: Dict[str, DatasetMixSourcePreset] = {
    "openorca": DatasetMixSourcePreset(
        alias="openorca",
        path="Open-Orca/OpenOrca",
        config=None,
        train_split="train",
        eval_split="validation",
        text_field="text",
        text_format="openorca",
    ),
    "fineweb_edu": DatasetMixSourcePreset(
        alias="fineweb_edu",
        path="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        train_split="train",
        eval_split="validation",
        text_field="text",
        text_format="text",
    ),
    "redpajama": DatasetMixSourcePreset(
        alias="redpajama",
        path="ZengXiangyu/RedPajama-Data-1T-Sample",
        config=None,
        train_split="train",
        eval_split="validation",
        text_field="text",
        text_format="text",
    ),
    "alpaca": DatasetMixSourcePreset(
        alias="alpaca",
        path="vicgalle/alpaca-gpt4",
        config=None,
        train_split="train",
        eval_split="validation",
        text_field="text",
        text_format="alpaca",
    ),
    "longalpaca": DatasetMixSourcePreset(
        alias="longalpaca",
        path="Yukang/LongAlpaca-12k",
        config=None,
        train_split="train",
        eval_split=None,
        text_field="text",
        text_format="alpaca",
    ),
    "longalign": DatasetMixSourcePreset(
        alias="longalign",
        path="zai-org/LongAlign-10k",
        config=None,
        train_split="train",
        eval_split=None,
        text_field="messages",
        text_format="longalign_chat",
    ),
    "race": DatasetMixSourcePreset(
        alias="race",
        path="ehovy/race",
        config="all",
        train_split="train",
        eval_split="validation",
        text_field="article",
        text_format="race_mcqa",
    ),
    "sciq": DatasetMixSourcePreset(
        alias="sciq",
        path="allenai/sciq",
        config=None,
        train_split="train",
        eval_split="validation",
        text_field="support",
        text_format="sciq_qa",
    ),
}


def build_tokenizer(model_path: str, access_token: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=access_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_dataset_mix_spec(sources: Sequence[str], weights: Sequence[float]) -> str:
    if len(sources) != len(weights):
        raise ValueError(f"dataset mix sources length {len(sources)} != weights {len(weights)}")
    parts = []
    for alias, weight in zip(sources, weights):
        normalized_alias = str(alias).strip().lower()
        normalized_weight = float(weight)
        parts.append(f"{normalized_alias}={normalized_weight:.12g}")
    return ",".join(parts)


def normalize_dataset_mix_spec(spec_text: Optional[str]) -> Tuple[List[str], List[float], str]:
    raw = str(spec_text or "").strip()
    if not raw:
        raise ValueError("--dataset_mix cannot be empty.")

    sources: List[str] = []
    raw_weights: List[float] = []
    seen = set()
    for item in raw.split(","):
        token = str(item).strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid --dataset_mix token '{token}'. Expected alias=weight."
            )
        alias_text, weight_text = token.split("=", 1)
        alias = str(alias_text).strip().lower()
        if not alias:
            raise ValueError(f"Invalid --dataset_mix token '{token}': empty alias.")
        if alias not in DATASET_MIX_SOURCE_PRESETS:
            raise ValueError(
                f"Unsupported --dataset_mix alias '{alias}'. Supported: {sorted(DATASET_MIX_SOURCE_PRESETS)}."
            )
        if alias in seen:
            raise ValueError(f"Duplicate --dataset_mix alias '{alias}'.")
        try:
            weight = float(str(weight_text).strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid --dataset_mix weight '{weight_text}' for alias '{alias}'."
            ) from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"--dataset_mix weight for alias '{alias}' must be > 0.")
        seen.add(alias)
        sources.append(alias)
        raw_weights.append(weight)

    if not sources:
        raise ValueError("--dataset_mix cannot be empty.")
    weight_sum = float(sum(raw_weights))
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("--dataset_mix total weight must be > 0.")
    weights = [float(weight / weight_sum) for weight in raw_weights]
    return sources, weights, format_dataset_mix_spec(sources, weights)


def _stringify_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_plain_text(record: Dict[str, object], *, text_field: str) -> Optional[str]:
    text = _stringify_text(record.get(text_field))
    return text or None


def _format_openorca_record(record: Dict[str, object], *, text_field: str) -> Optional[str]:
    raw_text = _format_plain_text(record, text_field=text_field)
    if raw_text:
        return raw_text

    question = _stringify_text(record.get("question"))
    response = _stringify_text(record.get("response"))
    system_prompt = _stringify_text(record.get("system_prompt"))
    if not question or not response:
        return None
    if system_prompt:
        return (
            f"### System:\n{system_prompt}\n\n"
            f"### User:\n{question}\n\n"
            f"### Assistant:\n{response}"
        )
    return f"### User:\n{question}\n\n### Assistant:\n{response}"


def _format_alpaca_record(record: Dict[str, object]) -> Optional[str]:
    instruction = _stringify_text(record.get("instruction"))
    input_text = _stringify_text(record.get("input"))
    output_text = _stringify_text(record.get("output"))
    if not instruction and not output_text:
        return None
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"


def _format_longalign_chat_record(record: Dict[str, object]) -> Optional[str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    parts: List[str] = []
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = _stringify_text(message.get("role")).lower()
        content = _stringify_text(message.get("content"))
        if not role or not content:
            return None
        if role in {"user", "human"}:
            header = "User"
        elif role in {"assistant", "gpt"}:
            header = "Assistant"
            has_assistant = True
        elif role == "system":
            header = "System"
        else:
            return None
        parts.append(f"### {header}:\n{content}")

    if not has_assistant:
        return None
    return "\n\n".join(parts)


def _normalize_choice_options(raw_options: object) -> List[str]:
    if raw_options is None:
        return []
    if isinstance(raw_options, dict):
        ordered_items = sorted(raw_options.items(), key=lambda item: str(item[0]))
        return [_stringify_text(value) for _key, value in ordered_items if _stringify_text(value)]
    if isinstance(raw_options, (list, tuple)):
        return [_stringify_text(item) for item in raw_options if _stringify_text(item)]
    return []


def _resolve_choice_index(answer: object, num_options: int) -> Optional[int]:
    text = _stringify_text(answer)
    if not text or num_options < 1:
        return None
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < num_options:
            return idx
        if 1 <= idx <= num_options:
            return idx - 1
    upper = text.upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        idx = ord(upper) - ord("A")
        if 0 <= idx < num_options:
            return idx
    return None


def _format_race_record(record: Dict[str, object]) -> Optional[str]:
    article = _stringify_text(record.get("article"))
    question = _stringify_text(record.get("question"))
    options = _normalize_choice_options(record.get("options"))
    answer_idx = _resolve_choice_index(record.get("answer"), len(options))
    if not article or not question or not options or answer_idx is None:
        return None

    option_lines = []
    for idx, option in enumerate(options):
        option_lines.append(f"{chr(ord('A') + idx)}. {option}")
    answer_text = options[answer_idx]
    return (
        f"### Passage:\n{article}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Options:\n" + "\n".join(option_lines) + f"\n\n"
        f"### Response:\n{answer_text}"
    )


def _format_sciq_record(record: Dict[str, object]) -> Optional[str]:
    support = _stringify_text(record.get("support"))
    question = _stringify_text(record.get("question"))
    correct_answer = _stringify_text(record.get("correct_answer"))
    if not question or not correct_answer:
        return None
    if support:
        return (
            f"### Support:\n{support}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Response:\n{correct_answer}"
        )
    return f"### Question:\n{question}\n\n### Response:\n{correct_answer}"


def _record_to_text(
    record: Dict[str, object],
    *,
    text_field: str,
    text_format: str = "auto",
) -> Optional[str]:
    normalized_text_format = str(text_format).strip().lower()
    if normalized_text_format == "text":
        return _format_plain_text(record, text_field=text_field)
    if normalized_text_format == "openorca":
        return _format_openorca_record(record, text_field=text_field)
    if normalized_text_format == "alpaca":
        return _format_alpaca_record(record)
    if normalized_text_format == "longalign_chat":
        return _format_longalign_chat_record(record)
    if normalized_text_format == "race_mcqa":
        return _format_race_record(record)
    if normalized_text_format == "sciq_qa":
        return _format_sciq_record(record)
    if normalized_text_format != "auto":
        raise ValueError(f"Unsupported dataset text format: {text_format}")

    raw_text = _format_plain_text(record, text_field=text_field)
    if raw_text is not None:
        return raw_text
    openorca_text = _format_openorca_record(record, text_field=text_field)
    if openorca_text is not None:
        return openorca_text
    return _format_alpaca_record(record)


def _resolve_local_dataset_loader(train_file: str) -> str:
    lower = str(train_file).strip().lower()
    if lower.endswith(".json") or lower.endswith(".jsonl"):
        return "json"
    if lower.endswith(".txt"):
        return "text"
    raise ValueError(f"Unsupported local dataset file extension: {train_file}")


def _load_hf_dataset_splits(
    *,
    path: str,
    config: Optional[str],
    train_split: str,
    eval_split: Optional[str],
) -> Tuple[Dataset, Optional[Dataset]]:
    dataset = load_dataset(
        path=str(path),
        name=None if config is None else str(config),
    )
    if isinstance(dataset, DatasetDict):
        if str(train_split) not in dataset:
            raise ValueError(f"Missing train split '{train_split}' in dataset {path}.")
        train_ds = dataset[str(train_split)]
        eval_ds = dataset.get(str(eval_split)) if eval_split else None
    else:
        train_ds = dataset
        eval_ds = None
    return train_ds, eval_ds


def _load_raw_datasets(args) -> Tuple[Dataset, Optional[Dataset]]:
    if args.dataset_name:
        return _load_hf_dataset_splits(
            path=str(args.dataset_name),
            config=None if args.dataset_config_name is None else str(args.dataset_config_name),
            train_split=str(args.train_split),
            eval_split=str(args.eval_split),
        )

    loader = _resolve_local_dataset_loader(str(args.train_file))
    data_files = {"train": str(args.train_file)}
    if args.eval_file:
        data_files["validation"] = str(args.eval_file)
    dataset = load_dataset(loader, data_files=data_files)
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError(f"Expected DatasetDict from local loader, got {type(dataset)}")
    return dataset["train"], dataset.get("validation")


def _apply_sample_limit(dataset: Optional[Dataset], max_samples: Optional[int]) -> Optional[Dataset]:
    if dataset is None or max_samples is None:
        return dataset
    if int(max_samples) >= len(dataset):
        return dataset
    return dataset.select(range(int(max_samples)))


def _prepare_text_dataset(dataset: Dataset, *, text_field: str, text_format: str = "auto") -> Dataset:
    prepared = dataset.map(
        lambda rec: {"text": _record_to_text(rec, text_field=text_field, text_format=text_format)},
        remove_columns=list(dataset.column_names),
    )
    prepared = prepared.filter(lambda rec: rec["text"] is not None and len(str(rec["text"]).strip()) > 0)
    return prepared


def _group_texts(examples: Dict[str, Sequence[Sequence[int]]], *, block_size: int) -> Dict[str, Sequence[Sequence[int]]]:
    concatenated = {key: list(chain.from_iterable(examples[key])) for key in examples.keys()}
    total_length = len(concatenated.get("input_ids", []))
    if total_length < int(block_size):
        return {key: [] for key in list(examples.keys()) + ["labels"]}
    total_length = (total_length // int(block_size)) * int(block_size)

    result = {
        key: [
            values[i: i + int(block_size)]
            for i in range(0, total_length, int(block_size))
        ]
        for key, values in concatenated.items()
    }
    result["labels"] = [list(seq) for seq in result["input_ids"]]
    return result


def _set_torch_columns(dataset: Dataset) -> Dataset:
    columns = [name for name in ("input_ids", "attention_mask", "labels") if name in dataset.column_names]
    dataset.set_format(type="torch", columns=columns)
    return dataset


def _tokenize_and_pack(dataset: Dataset, tokenizer, *, block_size: int) -> Dataset:
    tokenized = dataset.map(
        lambda rec: tokenizer(rec["text"]),
        batched=True,
        remove_columns=list(dataset.column_names),
    )
    packed = tokenized.map(
        lambda rec: _group_texts(rec, block_size=int(block_size)),
        batched=True,
    )
    return _set_torch_columns(packed)


def _resolve_training_world_size(training_args) -> int:
    world_size = getattr(training_args, "world_size", None)
    if world_size is None:
        world_size = 1
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"training world_size must be >= 1, got {world_size}")
    return world_size


def _compute_mix_target_examples(training_args) -> Tuple[int, int]:
    max_steps = int(getattr(training_args, "max_steps", -1))
    if max_steps <= 0:
        raise ValueError("--dataset_mix requires TrainingArguments.max_steps > 0.")
    grad_acc = int(getattr(training_args, "gradient_accumulation_steps", 1))
    if grad_acc < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1 for --dataset_mix.")
    per_device_batch = int(getattr(training_args, "per_device_train_batch_size", 0))
    if per_device_batch < 1:
        raise ValueError("per_device_train_batch_size must be >= 1 for --dataset_mix.")
    world_size = _resolve_training_world_size(training_args)
    required_examples = int(max_steps * grad_acc * per_device_batch * world_size)
    target_examples = int(math.ceil(float(required_examples) * 1.10))
    return required_examples, target_examples


def _split_target_rows(total_rows: int, weights: Sequence[float]) -> List[int]:
    if total_rows < 1:
        raise ValueError(f"total_rows must be >= 1, got {total_rows}")
    if len(weights) < 1:
        raise ValueError("weights cannot be empty.")
    targets: List[int] = []
    allocated = 0
    for idx, weight in enumerate(weights):
        if idx == len(weights) - 1:
            rows = int(total_rows - allocated)
        else:
            rows = int(math.floor(float(total_rows) * float(weight)))
            allocated += rows
        targets.append(rows)
    return targets


def _resize_packed_dataset(dataset: Dataset, *, target_rows: int, seed: int) -> Tuple[Dataset, float]:
    current_rows = len(dataset)
    if current_rows < 1:
        raise ValueError("cannot resize an empty packed dataset.")
    if target_rows < 1:
        raise ValueError(f"target_rows must be >= 1, got {target_rows}")

    shuffled = dataset.shuffle(seed=int(seed))
    if current_rows >= target_rows:
        resized = shuffled.select(range(int(target_rows)))
        return _set_torch_columns(resized), 1.0

    full_repeats, remainder = divmod(int(target_rows), int(current_rows))
    indices = list(range(int(current_rows))) * int(full_repeats)
    if remainder > 0:
        indices.extend(range(int(remainder)))
    resized = shuffled.select(indices)
    return _set_torch_columns(resized), float(target_rows) / float(current_rows)


def _load_preset_raw_datasets(preset: DatasetMixSourcePreset) -> Tuple[Dataset, Optional[Dataset]]:
    return _load_hf_dataset_splits(
        path=preset.path,
        config=preset.config,
        train_split=preset.train_split,
        eval_split=preset.eval_split,
    )


def _build_mixed_datasets(args, training_args, tokenizer):
    block_size = int(training_args.model_max_length)
    sources = list(getattr(args, "dataset_mix_sources", []) or [])
    weights = list(getattr(args, "dataset_mix_weights", []) or [])
    if not sources or not weights or len(sources) != len(weights):
        raise ValueError("Invalid dataset mix configuration. Run argument validation first.")

    required_examples, target_mixed_examples = _compute_mix_target_examples(training_args)
    per_source_targets = _split_target_rows(target_mixed_examples, weights)
    seed = int(getattr(training_args, "seed", 0))

    train_datasets: List[Dataset] = []
    eval_datasets: List[Dataset] = []
    source_stats: List[Dict[str, object]] = []
    for idx, (alias, weight, target_rows) in enumerate(zip(sources, weights, per_source_targets)):
        preset = DATASET_MIX_SOURCE_PRESETS[str(alias)]
        train_raw, eval_raw = _load_preset_raw_datasets(preset)
        train_text = _prepare_text_dataset(
            train_raw,
            text_field=str(preset.text_field),
            text_format=str(preset.text_format),
        )
        eval_text = None
        if eval_raw is not None:
            eval_text = _prepare_text_dataset(
                eval_raw,
                text_field=str(preset.text_field),
                text_format=str(preset.text_format),
            )

        train_packed = _tokenize_and_pack(train_text, tokenizer, block_size=block_size)
        packed_rows = len(train_packed)
        if packed_rows < 1:
            raise ValueError(
                f"Packed training dataset for mix source '{alias}' is empty. "
                "Increase source text volume or lower --model_max_length."
            )
        resized_train, repeat_factor = _resize_packed_dataset(
            train_packed,
            target_rows=int(target_rows),
            seed=int(seed + idx),
        )
        train_datasets.append(resized_train)

        eval_packed_rows = 0
        if eval_text is not None:
            eval_packed = _tokenize_and_pack(eval_text, tokenizer, block_size=block_size)
            eval_packed_rows = len(eval_packed)
            if eval_packed_rows > 0:
                eval_datasets.append(eval_packed)

        source_stats.append(
            {
                "alias": str(alias),
                "weight": float(weight),
                "raw_rows": int(len(train_raw)),
                "text_rows": int(len(train_text)),
                "packed_rows": int(packed_rows),
                "target_rows": int(target_rows),
                "repeat_factor": float(repeat_factor),
                "eval_packed_rows": int(eval_packed_rows),
            }
        )

    train_dataset = interleave_datasets(
        train_datasets,
        probabilities=[float(weight) for weight in weights],
        seed=int(seed),
        stopping_strategy="first_exhausted",
    )
    train_dataset = _set_torch_columns(train_dataset)

    eval_dataset = None
    if eval_datasets:
        eval_dataset = concatenate_datasets(eval_datasets)
        eval_dataset = _set_torch_columns(eval_dataset)

    return train_dataset, eval_dataset, {
        "dataset_mode": "mix",
        "block_size": int(block_size),
        "dataset_mix_spec": str(args.dataset_mix_spec),
        "dataset_mix_sources": list(sources),
        "dataset_mix_weights": [float(weight) for weight in weights],
        "dataset_mix_target_examples": int(target_mixed_examples),
        "required_train_examples": int(required_examples),
        "source_stats": source_stats,
    }


def build_datasets(args, training_args, tokenizer):
    if getattr(args, "dataset_mix_spec", None):
        return _build_mixed_datasets(args, training_args, tokenizer)

    block_size = int(training_args.model_max_length)

    train_raw, eval_raw = _load_raw_datasets(args)
    train_raw = _apply_sample_limit(train_raw, args.max_train_samples)
    eval_raw = _apply_sample_limit(eval_raw, args.max_eval_samples)

    train_text = _prepare_text_dataset(train_raw, text_field=str(args.text_field))
    eval_text = _prepare_text_dataset(eval_raw, text_field=str(args.text_field)) if eval_raw is not None else None

    train_dataset = _tokenize_and_pack(train_text, tokenizer, block_size=block_size)
    eval_dataset = _tokenize_and_pack(eval_text, tokenizer, block_size=block_size) if eval_text is not None else None
    return train_dataset, eval_dataset, {
        "dataset_mode": "single",
        "block_size": int(block_size),
        "source_stats": [],
    }
