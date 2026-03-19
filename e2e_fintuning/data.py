from itertools import chain
from typing import Dict, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def build_tokenizer(model_path: str, access_token: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=access_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _record_to_text(record: Dict[str, object], text_field: str) -> Optional[str]:
    raw_text = record.get(text_field)
    if raw_text is not None:
        text = str(raw_text).strip()
        if text:
            return text

    instruction = str(record.get("instruction", "") or "").strip()
    input_text = str(record.get("input", "") or "").strip()
    output_text = str(record.get("output", "") or "").strip()
    if not instruction and not output_text:
        return None
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"


def _resolve_local_dataset_loader(train_file: str) -> str:
    lower = str(train_file).strip().lower()
    if lower.endswith(".json") or lower.endswith(".jsonl"):
        return "json"
    if lower.endswith(".txt"):
        return "text"
    raise ValueError(f"Unsupported local dataset file extension: {train_file}")


def _load_raw_datasets(args) -> Tuple[Dataset, Optional[Dataset]]:
    if args.dataset_name:
        dataset = load_dataset(
            path=str(args.dataset_name),
            name=str(args.dataset_config_name) if args.dataset_config_name else None,
        )
        if isinstance(dataset, DatasetDict):
            train_split = str(args.train_split)
            if train_split not in dataset:
                raise ValueError(f"Missing train split '{train_split}' in dataset {args.dataset_name}.")
            train_ds = dataset[train_split]
            eval_ds = dataset.get(str(args.eval_split))
        else:
            train_ds = dataset
            eval_ds = None
        return train_ds, eval_ds

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


def _prepare_text_dataset(dataset: Dataset, *, text_field: str) -> Dataset:
    prepared = dataset.map(
        lambda rec: {"text": _record_to_text(rec, text_field=text_field)},
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
    columns = [name for name in ("input_ids", "attention_mask", "labels") if name in packed.column_names]
    packed.set_format(type="torch", columns=columns)
    return packed


def build_datasets(args, training_args, tokenizer):
    block_size = int(args.packing_block_size or min(int(training_args.model_max_length), 2048))
    block_size = min(block_size, int(training_args.model_max_length))

    train_raw, eval_raw = _load_raw_datasets(args)
    train_raw = _apply_sample_limit(train_raw, args.max_train_samples)
    eval_raw = _apply_sample_limit(eval_raw, args.max_eval_samples)

    train_text = _prepare_text_dataset(train_raw, text_field=str(args.text_field))
    eval_text = _prepare_text_dataset(eval_raw, text_field=str(args.text_field)) if eval_raw is not None else None

    train_dataset = _tokenize_and_pack(train_text, tokenizer, block_size=block_size)
    eval_dataset = _tokenize_and_pack(eval_text, tokenizer, block_size=block_size) if eval_text is not None else None
    return train_dataset, eval_dataset, {"block_size": int(block_size)}
