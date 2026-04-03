from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError:
    Dataset = None
    DatasetDict = None
    load_dataset = None


@dataclass(frozen=True)
class LoraDatasetSpec:
    path: str
    config: Optional[str]
    train_split: str
    eval_splits: Tuple[str, ...]
    text_format: str


LORA_DATASET_SPECS = {
    "wiki": LoraDatasetSpec(
        path="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        train_split="train",
        eval_splits=("validation", "test"),
        text_format="plain_text",
    ),
    "fineweb_edu": LoraDatasetSpec(
        path="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        train_split="train",
        eval_splits=("validation", "test"),
        text_format="plain_text",
    ),
    "openorca": LoraDatasetSpec(
        path="Open-Orca/OpenOrca",
        config=None,
        train_split="train",
        eval_splits=("validation", "test"),
        text_format="openorca",
    ),
    "redpajama": LoraDatasetSpec(
        path="ZengXiangyu/RedPajama-Data-1T-Sample",
        config=None,
        train_split="train",
        eval_splits=("validation", "test"),
        text_format="plain_text",
    ),
    "alpaca": LoraDatasetSpec(
        path="vicgalle/alpaca-gpt4",
        config=None,
        train_split="train",
        eval_splits=("validation", "test"),
        text_format="alpaca",
    ),
}


def ensure_lora_dataset_stack_available() -> None:
    if load_dataset is None or DatasetDict is None:
        raise ImportError("未安装 datasets。请先安装：pip install datasets")


def _get_nonempty_record_text(record: dict, field_name: str) -> Optional[str]:
    value = record.get(field_name)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _record_to_text(record: dict, *, text_format: str) -> Optional[str]:
    if text_format == "plain_text":
        return _get_nonempty_record_text(record, "text")

    if text_format == "openorca":
        text = _get_nonempty_record_text(record, "text")
        if text is not None:
            return text
        question = _get_nonempty_record_text(record, "question")
        response = _get_nonempty_record_text(record, "response")
        system_prompt = _get_nonempty_record_text(record, "system_prompt")
        if question is None or response is None:
            return None
        if system_prompt is not None:
            return (
                f"### System:\n{system_prompt}\n\n"
                f"### User:\n{question}\n\n"
                f"### Assistant:\n{response}"
            )
        return f"### User:\n{question}\n\n### Assistant:\n{response}"

    if text_format == "alpaca":
        text = _get_nonempty_record_text(record, "text")
        if text is not None:
            return text
        instruction = _get_nonempty_record_text(record, "instruction")
        output = _get_nonempty_record_text(record, "output")
        input_text = _get_nonempty_record_text(record, "input")
        if instruction is None or output is None:
            return None
        if input_text is not None:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    raise ValueError(f"Unsupported LoRA dataset text format: {text_format}")


def _prepare_text_dataset(dataset, *, text_format: str):
    dataset = dataset.map(lambda rec: {"text": _record_to_text(rec, text_format=text_format)})
    dataset = dataset.filter(lambda rec: rec["text"] is not None and len(rec["text"]) > 0)
    return dataset


def prepare_lora_datasets(
    dataset_name: str,
    *,
    nsamples: int,
    seed: int,
):
    ensure_lora_dataset_stack_available()

    dataset_key = str(dataset_name).strip().lower()
    spec = LORA_DATASET_SPECS.get(dataset_key)
    if spec is None:
        raise ValueError(
            "Unsupported --lora_dataset: "
            f"{dataset_name}. Supported values: {', '.join(LORA_DATASET_SPECS.keys())}."
        )

    load_kwargs = {"path": str(spec.path)}
    if spec.config is not None:
        load_kwargs["name"] = str(spec.config)
    dataset_dict = load_dataset(**load_kwargs)
    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(f"Expected DatasetDict from load_dataset, got {type(dataset_dict)}")

    if spec.train_split not in dataset_dict:
        raise ValueError(f"LoRA dataset {dataset_key} is missing train split '{spec.train_split}'.")
    train_ds = dataset_dict[spec.train_split]
    if int(nsamples) > 0:
        train_ds = train_ds.shuffle(seed=int(seed)).select(range(min(int(nsamples), len(train_ds))))
    train_ds = _prepare_text_dataset(train_ds, text_format=str(spec.text_format))

    eval_ds = None
    eval_split = None
    for candidate in spec.eval_splits:
        if candidate in dataset_dict:
            eval_split = str(candidate)
            eval_ds = _prepare_text_dataset(dataset_dict[candidate], text_format=str(spec.text_format))
            if len(eval_ds) == 0:
                eval_ds = None
            break

    return dataset_key, spec, train_ds, eval_ds, eval_split
