#!/usr/bin/env python3
"""Generate VAELLM eval-aligned downstream task jsonl for EdgeRazor-style training."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset, concatenate_datasets, load_dataset
from jinja2 import Template
from tqdm import tqdm

from e2e_common.data import _MMLU_NO_TRAIN_SUBJECTS, record_to_mcqa_example

LONGBENCH_SUBSETS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
)


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _winogrande_doc_to_text(doc: Dict[str, Any]) -> Optional[str]:
    if not doc.get("answer") or doc["answer"] == "":
        return None
    idx = doc["sentence"].index("_")
    answer_to_num = {"1": 0, "2": 1}
    target_idx = answer_to_num[doc["answer"]]
    option = doc["option1"] if target_idx == 0 else doc["option2"]
    return doc["sentence"][:idx].rstrip() + " " + option


def _winogrande_doc_to_target(doc: Dict[str, Any]) -> Optional[int]:
    if not doc.get("answer") or doc["answer"] == "":
        return None
    return {"1": 0, "2": 1}[doc["answer"]]


def _winogrande_doc_to_choice(doc: Dict[str, Any]) -> List[str]:
    idx = doc["sentence"].index("_")
    remaining = doc["sentence"][idx + 1 :].lstrip()
    return [remaining, remaining]


def _render_template(template_str: Any, doc: Dict[str, Any]) -> Any:
    if callable(template_str):
        return template_str(doc)
    if isinstance(template_str, list):
        return template_str
    try:
        return Template(str(template_str)).render(**doc)
    except Exception:
        if isinstance(template_str, str) and "{{" in template_str and "}}" in template_str:
            expr = template_str.split("{{", 1)[1].split("}}", 1)[0].strip()
            try:
                return eval(expr, {"__builtins__": {}}, doc)
            except Exception:
                return template_str
        return template_str


def _process_mcqa_document(doc: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, List[Dict[str, str]]]]:
    question_text = _render_template(config["doc_to_text"], doc)
    if question_text is None:
        return None

    if callable(config["doc_to_choice"]):
        choices = config["doc_to_choice"](doc)
    else:
        choices_result = _render_template(config["doc_to_choice"], doc)
        if isinstance(choices_result, str) and choices_result.startswith("["):
            try:
                choices = eval(choices_result)
            except Exception:
                choices = choices_result
        else:
            choices = choices_result

    if callable(config["doc_to_target"]):
        target_idx = config["doc_to_target"](doc)
    else:
        doc_target = config["doc_to_target"]
        if isinstance(doc_target, str) and "{{" not in doc_target and doc_target in doc:
            target_idx = doc[doc_target]
        else:
            target_result = _render_template(doc_target, doc)
            if isinstance(target_result, int):
                target_idx = target_result
            elif isinstance(target_result, str) and target_result.isdigit():
                target_idx = int(target_result)
            else:
                try:
                    target_idx = eval(target_result) if isinstance(target_result, str) else target_result
                except Exception:
                    target_idx = target_result

    if target_idx is None or target_idx == "":
        return None

    if isinstance(choices, list) and len(choices) == 2 and set(choices) == {"no", "yes"}:
        try:
            answer_text = choices[int(target_idx)]
        except Exception:
            answer_text = str(target_idx)
    elif isinstance(choices, list) and isinstance(target_idx, int) and 0 <= target_idx < len(choices):
        answer_text = choices[target_idx]
    elif isinstance(choices, list):
        try:
            answer_text = choices[int(target_idx)]
        except Exception:
            answer_text = str(choices)
    else:
        answer_text = str(choices)

    return {
        "messages": [
            {"role": "user", "content": str(question_text)},
            {"role": "assistant", "content": " " + str(answer_text)},
        ]
    }


KEEP_TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "arc_e": {
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Easy",
        "training_split": "train",
        "doc_to_text": "Question: {{question}}\nAnswer:",
        "doc_to_target": "{{choices.label.index(answerKey)}}",
        "doc_to_choice": "{{choices.text}}",
    },
    "arc_c": {
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Challenge",
        "training_split": "train",
        "doc_to_text": "Question: {{question}}\nAnswer:",
        "doc_to_target": "{{choices.label.index(answerKey)}}",
        "doc_to_choice": "{{choices.text}}",
    },
    "boolq": {
        "dataset_path": "super_glue",
        "dataset_name": "boolq",
        "training_split": "train",
        "doc_to_text": "{{passage}}\nQuestion: {{question}}?\nAnswer:",
        "doc_to_target": "{{label}}",
        "doc_to_choice": ["no", "yes"],
    },
    "piqa": {
        "dataset_path": "piqa",
        "dataset_name": None,
        "training_split": "train",
        "doc_to_text": "Question: {{goal}}\nAnswer:",
        "doc_to_target": "{{label}}",
        "doc_to_choice": "{{[sol1, sol2]}}",
    },
    "winogrande": {
        "dataset_path": "winogrande",
        "dataset_name": "winogrande_xl",
        "training_split": "train",
        "doc_to_text": _winogrande_doc_to_text,
        "doc_to_target": _winogrande_doc_to_target,
        "doc_to_choice": _winogrande_doc_to_choice,
    },
    "openbookqa": {
        "dataset_path": "openbookqa",
        "dataset_name": "main",
        "training_split": "train",
        "doc_to_text": "{{question_stem}}",
        "doc_to_target": "{{choices.label.index(answerKey.lstrip())}}",
        "doc_to_choice": "{{choices.text}}",
    },
    "rte": {
        "dataset_path": "super_glue",
        "dataset_name": "rte",
        "training_split": "train",
        "doc_to_text": "{{sentence1}}\nQuestion: {{sentence2}} True or False?\nAnswer:",
        "doc_to_target": "{{label}}",
        "doc_to_choice": ["False", "True"],
    },
}


def _load_hf_dataset(path: str, name: Optional[str] = None, *, split: Optional[str] = None):
    # datasets>=2.14 对脚本式仓库需要显式信任；非交互环境不能等 stdin 确认。
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if split is not None:
        kwargs["split"] = split
    if name:
        return load_dataset(path, name, **kwargs)
    return load_dataset(path, **kwargs)


def _iter_task_config_records(
    task_name: str,
    config: Dict[str, Any],
    *,
    max_samples: Optional[int] = None,
) -> Iterable[Dict[str, List[Dict[str, str]]]]:
    dataset = _load_hf_dataset(config["dataset_path"], config["dataset_name"])

    split_name = str(config["training_split"])
    if split_name not in dataset:
        raise ValueError(f"Task {task_name} missing split {split_name!r}.")

    split_data = dataset[split_name]
    produced = 0
    for doc in split_data:
        processed = _process_mcqa_document(doc, config)
        if processed is None:
            continue
        yield processed
        produced += 1
        if max_samples is not None and produced >= int(max_samples):
            break


def _iter_mmlu_records(*, max_samples: Optional[int] = None) -> Iterable[Dict[str, List[Dict[str, str]]]]:
    produced = 0
    for subject in _MMLU_NO_TRAIN_SUBJECTS:
        dataset = _load_hf_dataset("hails/mmlu_no_train", str(subject))
        for split_name in ("dev", "validation"):
            if split_name not in dataset:
                continue
            for record in dataset[split_name]:
                if "subject" not in record:
                    record = dict(record)
                    record["subject"] = str(subject)
                example = record_to_mcqa_example(record, text_format="mmlu_mcqa")
                if example is None:
                    continue
                answer_letter = chr(ord("A") + int(example["answer_index"]))
                yield {
                    "messages": [
                        {"role": "user", "content": str(example["prompt"])},
                        {"role": "assistant", "content": f" {answer_letter}"},
                    ]
                }
                produced += 1
                if max_samples is not None and produced >= int(max_samples):
                    return


def _iter_longbench_records(*, max_samples: Optional[int] = None) -> Iterable[Dict[str, List[Dict[str, str]]]]:
    import zipfile

    from huggingface_hub import hf_hub_download

    # THUDM/LongBench 以 data.zip + 脚本仓库分发；datasets.load_dataset 容易长时间卡住。
    # 直接下载 zip 后按 subset 读取本地 jsonl。
    try:
        zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    except Exception as exc:
        print(f"Warning: skip LongBench (data.zip download failed): {exc}", flush=True)
        return

    produced = 0
    with zipfile.ZipFile(zip_path) as zf:
        available = set(zf.namelist())
        for subset in LONGBENCH_SUBSETS:
            member = f"data/{subset}.jsonl"
            if member not in available:
                print(f"Warning: skip LongBench subset {subset}: missing {member}", flush=True)
                continue
            try:
                with zf.open(member) as handle:
                    for raw in handle:
                        record = json.loads(raw)
                        user_text = str(record.get("input", "")).strip()
                        answers = record.get("answers")
                        if not user_text or not answers:
                            continue
                        answer_text = str(answers[0]).strip()
                        if not answer_text:
                            continue
                        yield {
                            "messages": [
                                {"role": "user", "content": user_text},
                                {"role": "assistant", "content": " " + answer_text},
                            ]
                        }
                        produced += 1
                        if max_samples is not None and produced >= int(max_samples):
                            return
            except Exception as exc:
                print(f"Warning: skip LongBench subset {subset}: {exc}", flush=True)
                continue


def _write_task_jsonl(
    *,
    output_path: Path,
    max_samples_per_task: Optional[int],
) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as handle:
        for task_name, config in KEEP_TASK_CONFIGS.items():
            count = 0
            iterator = _iter_task_config_records(
                task_name,
                config,
                max_samples=max_samples_per_task,
            )
            for record in tqdm(iterator, desc=f"task:{task_name}"):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            stats[task_name] = int(count)

        count = 0
        for record in tqdm(_iter_mmlu_records(max_samples=max_samples_per_task), desc="task:mmlu"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
        stats["mmlu"] = int(count)

        count = 0
        for record in tqdm(_iter_longbench_records(max_samples=max_samples_per_task), desc="task:longbench"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
        stats["longbench"] = int(count)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VAELLM eval-aligned downstream task jsonl.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "edgerazor_qwen3"),
        help="Directory for task_vaellm_eval_instruct.jsonl",
    )
    parser.add_argument(
        "--max_samples_per_task",
        type=int,
        default=None,
        help="Optional per-task cap for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_path = output_dir / "task_vaellm_eval_instruct.jsonl"
    stats = _write_task_jsonl(
        output_path=output_path,
        max_samples_per_task=args.max_samples_per_task,
    )
    total = sum(int(value) for value in stats.values())
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Wrote {total} samples -> {output_path} ({size_mb:.2f} MB)")
    for task_name, count in sorted(stats.items()):
        pct = (count / total * 100.0) if total > 0 else 0.0
        print(f"  {task_name}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
