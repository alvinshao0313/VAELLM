import math
import os
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import IntervalStrategy

from e2e_common.chat_template_utils import (
    infer_assistant_response_template,
    infer_user_instruction_template,
    render_messages,
)


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
    "wiki": DatasetMixSourcePreset(
        alias="wiki",
        path="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        train_split="train",
        eval_split="validation",
        text_field="text",
        text_format="text",
    ),
    "wikitext2": DatasetMixSourcePreset(
        alias="wikitext2",
        path="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        train_split="train",
        eval_split="validation",
        text_field="text",
        text_format="text",
    ),
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
    "mmlu": DatasetMixSourcePreset(
        alias="mmlu",
        path="hails/mmlu_no_train",
        config=None,
        train_split="dev+validation",
        eval_split=None,
        text_field="question",
        text_format="mmlu_mcqa",
    ),
    "arc": DatasetMixSourcePreset(
        alias="arc",
        path="allenai/ai2_arc",
        config="ARC-Challenge",
        train_split="train",
        eval_split="validation",
        text_field="question",
        text_format="arc_mcqa",
    ),
    "openbookqa": DatasetMixSourcePreset(
        alias="openbookqa",
        path="allenai/openbookqa",
        config="main",
        train_split="train",
        eval_split="validation",
        text_field="question_stem",
        text_format="openbookqa_mcqa",
    ),
    "edgerazor_ii_7m": DatasetMixSourcePreset(
        alias="edgerazor_ii_7m",
        path="data/edgerazor_qwen3/ii_7M_instruct.jsonl",
        config=None,
        train_split="train",
        eval_split=None,
        text_field="messages",
        text_format="edgerazor_messages",
    ),
    "edgerazor_ii_gen": DatasetMixSourcePreset(
        alias="edgerazor_ii_gen",
        path="data/edgerazor_qwen3/ii_gen_1.4M_instruct.jsonl",
        config=None,
        train_split="train",
        eval_split=None,
        text_field="messages",
        text_format="edgerazor_messages",
    ),
    "edgerazor_tulu": DatasetMixSourcePreset(
        alias="edgerazor_tulu",
        path="data/edgerazor_qwen3/tulu_0.6M_instruct.jsonl",
        config=None,
        train_split="train",
        eval_split=None,
        text_field="messages",
        text_format="edgerazor_messages",
    ),
    "edgerazor_am": DatasetMixSourcePreset(
        alias="edgerazor_am",
        path="data/edgerazor_qwen3/am_1.4M_instruct.jsonl",
        config=None,
        train_split="train",
        eval_split=None,
        text_field="messages",
        text_format="edgerazor_messages",
    ),
    "vaellm_eval_task": DatasetMixSourcePreset(
        alias="vaellm_eval_task",
        path="data/edgerazor_qwen3/task_vaellm_eval_instruct.jsonl",
        config=None,
        train_split="train",
        eval_split=None,
        text_field="messages",
        text_format="edgerazor_messages",
    ),
}


VAELLM_EDGERAZOR_DATA_DIR = "data/edgerazor_qwen3"
VAELLM_EDGERAZOR_DATASET_MIX = (
    "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,"
    "edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009"
)
VAELLM_EDGERAZOR_TOTAL_SAMPLES = 11_000_000
VAELLM_EDGERAZOR_SFT_ALIASES = {
    "edgerazor_ii_7m",
    "edgerazor_ii_gen",
    "edgerazor_tulu",
    "edgerazor_am",
    "vaellm_eval_task",
}

MCQA_DATASET_MIX_ALIASES = {"mmlu", "race", "sciq", "arc", "openbookqa"}
_MCQA_CONTINUATIONS = [" A", " B", " C", " D"]
_MMLU_NO_TRAIN_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


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


def _normalize_four_options(raw_options: object) -> Optional[List[str]]:
    options = _normalize_choice_options(raw_options)
    if len(options) != 4:
        return None
    return options


def _normalize_labeled_choice_dict(raw_choices: object) -> Optional[Tuple[List[str], List[str]]]:
    if not isinstance(raw_choices, dict):
        return None
    texts = raw_choices.get("text")
    labels = raw_choices.get("label")
    if not isinstance(texts, (list, tuple)) or not isinstance(labels, (list, tuple)):
        return None
    pairs = []
    for label, text in zip(labels, texts):
        norm_label = _stringify_text(label).upper()
        norm_text = _stringify_text(text)
        if not norm_label or not norm_text:
            continue
        pairs.append((norm_label, norm_text))
    if len(pairs) != 4:
        return None
    pairs.sort(key=lambda item: item[0])
    expected = ["A", "B", "C", "D"]
    sorted_labels = [label for label, _text in pairs]
    if sorted_labels != expected:
        return None
    return [text for _label, text in pairs], sorted_labels


def _stable_sciq_answer_index(question: str) -> int:
    return int(sum(ord(char) for char in str(question)) % 4)


def _build_mcqa_prompt(
    *,
    question: str,
    options: Sequence[str],
    subject: Optional[str] = None,
    passage: Optional[str] = None,
    support: Optional[str] = None,
) -> str:
    parts: List[str] = []
    if subject:
        parts.append(f"Subject: {subject}")
    if passage:
        parts.append(f"Passage: {passage}")
    if support:
        parts.append(f"Support: {support}")
    parts.append(f"Question: {question}")
    for idx, option in enumerate(options):
        parts.append(f"{chr(ord('A') + idx)}. {option}")
    parts.append("Answer:")
    return "\n".join(parts)


def record_to_mcqa_example(
    record: Dict[str, object],
    *,
    text_format: str,
) -> Optional[Dict[str, object]]:
    normalized_text_format = str(text_format).strip().lower()
    if normalized_text_format == "mmlu_mcqa":
        question = _stringify_text(record.get("question"))
        options = _normalize_four_options(record.get("choices"))
        answer_idx = _resolve_choice_index(record.get("answer"), 4)
        subject = _stringify_text(record.get("subject")).replace("_", " ")
        if not question or options is None or answer_idx is None:
            return None
        prompt = _build_mcqa_prompt(question=question, options=options, subject=subject or None)
        return {"prompt": prompt, "continuations": list(_MCQA_CONTINUATIONS), "answer_index": int(answer_idx)}

    if normalized_text_format == "race_mcqa":
        article = _stringify_text(record.get("article"))
        question = _stringify_text(record.get("question"))
        options = _normalize_four_options(record.get("options"))
        answer_idx = _resolve_choice_index(record.get("answer"), 4)
        if not article or not question or options is None or answer_idx is None:
            return None
        prompt = _build_mcqa_prompt(question=question, options=options, passage=article)
        return {"prompt": prompt, "continuations": list(_MCQA_CONTINUATIONS), "answer_index": int(answer_idx)}

    if normalized_text_format == "sciq_qa":
        support = _stringify_text(record.get("support"))
        question = _stringify_text(record.get("question"))
        correct_answer = _stringify_text(record.get("correct_answer"))
        distractors = [
            _stringify_text(record.get("distractor1")),
            _stringify_text(record.get("distractor2")),
            _stringify_text(record.get("distractor3")),
        ]
        if not question or not correct_answer or any(not item for item in distractors):
            return None
        answer_idx = _stable_sciq_answer_index(question)
        options = list(distractors)
        options.insert(answer_idx, correct_answer)
        prompt = _build_mcqa_prompt(question=question, options=options, support=support or None)
        return {"prompt": prompt, "continuations": list(_MCQA_CONTINUATIONS), "answer_index": int(answer_idx)}

    if normalized_text_format in {"arc_mcqa", "openbookqa_mcqa"}:
        question_key = "question_stem" if normalized_text_format == "openbookqa_mcqa" else "question"
        question = _stringify_text(record.get(question_key))
        normalized_choices = _normalize_labeled_choice_dict(record.get("choices"))
        if normalized_choices is None:
            return None
        options, labels = normalized_choices
        answer_text = _stringify_text(record.get("answerKey")).upper()
        if not question or answer_text not in labels:
            return None
        answer_idx = labels.index(answer_text)
        prompt = _build_mcqa_prompt(question=question, options=options)
        return {"prompt": prompt, "continuations": list(_MCQA_CONTINUATIONS), "answer_index": int(answer_idx)}

    raise ValueError(
        f"MCQA dataset_task does not support text_format={text_format!r}. "
        "Supported formats: mmlu_mcqa, race_mcqa, sciq_qa, arc_mcqa, openbookqa_mcqa."
    )


def encode_mcqa_example(
    example: Dict[str, object],
    tokenizer,
    *,
    block_size: int,
) -> Dict[str, torch.Tensor]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if eos_token_id is None:
        raise ValueError("MCQA dataset_task requires tokenizer.eos_token_id.")
    if pad_token_id is None:
        raise ValueError("MCQA dataset_task requires tokenizer.pad_token_id.")

    prompt = _stringify_text(example.get("prompt"))
    continuations = list(example.get("continuations") or [])
    answer_index = int(example.get("answer_index"))
    if not prompt or len(continuations) != 4 or not (0 <= answer_index < len(continuations)):
        raise ValueError("Invalid MCQA example.")

    prompt_ids = [
        int(token_id)
        for token_id in tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        ).get("input_ids", [])
    ]
    if not prompt_ids:
        raise ValueError("MCQA prompt tokenization produced no tokens.")

    out_input_ids: List[List[int]] = []
    out_attention_mask: List[List[int]] = []
    out_labels: List[List[int]] = []
    for continuation in continuations:
        continuation_ids = [
            int(token_id)
            for token_id in tokenizer(
                str(continuation),
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            ).get("input_ids", [])
        ]
        if not continuation_ids:
            raise ValueError("MCQA continuation tokenization produced no tokens.")
        input_ids = list(prompt_ids) + continuation_ids + [int(eos_token_id)]
        labels = [-100] * len(prompt_ids) + list(continuation_ids) + [-100]
        attention_mask = [1] * len(input_ids)
        if len(input_ids) > int(block_size):
            raise ValueError(
                f"MCQA sample length {len(input_ids)} exceeds --model_max_length={int(block_size)}. "
                "Increase --model_max_length or remove this over-length sample."
            )
        pad_len = int(block_size) - len(input_ids)
        out_input_ids.append(input_ids + [int(pad_token_id)] * pad_len)
        out_attention_mask.append(attention_mask + [0] * pad_len)
        out_labels.append(labels + [-100] * pad_len)

    return {
        "choice_input_ids": torch.tensor(out_input_ids, dtype=torch.long),
        "choice_attention_mask": torch.tensor(out_attention_mask, dtype=torch.long),
        "choice_labels": torch.tensor(out_labels, dtype=torch.long),
        "answer_index": torch.tensor(answer_index, dtype=torch.long),
    }


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


def _sft_openorca_segments(record: Dict[str, object]) -> Optional[List[Tuple[str, bool]]]:
    question = _stringify_text(record.get("question"))
    response = _stringify_text(record.get("response"))
    system_prompt = _stringify_text(record.get("system_prompt"))
    if not question or not response:
        return None

    prompt = ""
    if system_prompt:
        prompt += f"### System:\n{system_prompt}\n\n"
    prompt += f"### User:\n{question}\n\n### Assistant:\n"
    return [(prompt, False), (response, True)]


def _sft_alpaca_segments(record: Dict[str, object]) -> Optional[List[Tuple[str, bool]]]:
    instruction = _stringify_text(record.get("instruction"))
    input_text = _stringify_text(record.get("input"))
    output_text = _stringify_text(record.get("output"))
    if not instruction or not output_text:
        return None

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return [(prompt, False), (output_text, True)]


def _sft_longalign_chat_segments(record: Dict[str, object]) -> Optional[List[Tuple[str, bool]]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    segments: List[Tuple[str, bool]] = []
    has_assistant = False
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            return None
        role = _stringify_text(message.get("role")).lower()
        content = _stringify_text(message.get("content"))
        if not role or not content:
            return None
        prefix = "" if idx == 0 else "\n\n"
        if role in {"user", "human"}:
            segments.append((f"{prefix}### User:\n{content}", False))
        elif role in {"assistant", "gpt"}:
            segments.append((f"{prefix}### Assistant:\n", False))
            segments.append((content, True))
            has_assistant = True
        elif role == "system":
            segments.append((f"{prefix}### System:\n{content}", False))
        else:
            return None

    if not has_assistant:
        return None
    return segments


def _sft_race_segments(record: Dict[str, object]) -> Optional[List[Tuple[str, bool]]]:
    article = _stringify_text(record.get("article"))
    question = _stringify_text(record.get("question"))
    options = _normalize_choice_options(record.get("options"))
    answer_idx = _resolve_choice_index(record.get("answer"), len(options))
    if not article or not question or not options or answer_idx is None:
        return None

    option_lines = []
    for idx, option in enumerate(options):
        option_lines.append(f"{chr(ord('A') + idx)}. {option}")
    prompt = (
        f"### Passage:\n{article}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Options:\n" + "\n".join(option_lines) + "\n\n"
        f"### Response:\n"
    )
    return [(prompt, False), (options[answer_idx], True)]


def _resolve_dataset_path(path: str) -> str:
    path_str = str(path).strip()
    if not path_str.lower().endswith((".jsonl", ".json")):
        return path_str
    candidate = path_str
    if not os.path.isabs(candidate):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, candidate)
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"Local dataset file not found: {candidate}")
    return candidate


def _normalize_edgerazor_messages(record: Dict[str, object]) -> Optional[List[Dict[str, str]]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 1:
        return None
    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            return None
        normalized.append({"role": role, "content": content})
    return normalized


def _format_edgerazor_messages_record(record: Dict[str, object]) -> Optional[str]:
    messages = _normalize_edgerazor_messages(record)
    if messages is None:
        return None
    parts: List[str] = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        if role == "system":
            parts.append(f"### System:\n{content}")
        elif role == "user":
            parts.append(f"### User:\n{content}")
        else:
            parts.append(f"### Assistant:\n{content}")
    return "\n\n".join(parts)


def _encode_edgerazor_messages_sft(
    record: Dict[str, object],
    tokenizer,
    *,
    block_size: int,
) -> Tuple[List[int], List[int], List[int]]:
    messages = _normalize_edgerazor_messages(record)
    if messages is None:
        raise ValueError("edgerazor_messages SFT record is missing a valid messages field.")

    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    except ImportError as exc:
        raise ImportError("未安装 trl。edgerazor_messages SFT 需要 DataCollatorForCompletionOnlyLM。") from exc

    full_text = render_messages(messages, tokenizer, add_generation_prompt=False)
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = [int(token_id) for token_id in encoded.get("input_ids", [])]
    attention_mask = [1] * len(input_ids)
    collator = DataCollatorForCompletionOnlyLM(
        infer_assistant_response_template(tokenizer),
        instruction_template=infer_user_instruction_template(tokenizer),
        tokenizer=tokenizer,
        mlm=False,
    )
    batch = collator(
        [
            {
                "input_ids": list(input_ids),
                "attention_mask": list(attention_mask),
            }
        ]
    )
    labels = [int(value) for value in batch["labels"][0].tolist()]
    has_trainable_token = any(int(value) != -100 for value in labels)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        input_ids.append(int(eos_token_id))
        labels.append(int(eos_token_id))
        attention_mask.append(1)
        has_trainable_token = True

    if not has_trainable_token:
        raise ValueError("edgerazor_messages SFT sample has no trainable response tokens.")
    if len(input_ids) > int(block_size):
        raise ValueError(
            f"edgerazor_messages SFT sample length {len(input_ids)} exceeds --model_max_length={int(block_size)}. "
            "Increase --model_max_length or remove this over-length sample."
        )
    return input_ids, attention_mask, labels


def _sft_edgerazor_messages_segments(record: Dict[str, object]) -> Optional[List[Tuple[str, bool]]]:
    raise ValueError(
        "edgerazor_messages SFT encoding requires tokenizer.apply_chat_template. "
        "Use _encode_edgerazor_messages_sft instead of segment-based encoding."
    )


def _sft_sciq_segments(record: Dict[str, object]) -> Optional[List[Tuple[str, bool]]]:
    support = _stringify_text(record.get("support"))
    question = _stringify_text(record.get("question"))
    correct_answer = _stringify_text(record.get("correct_answer"))
    if not question or not correct_answer:
        return None

    if support:
        prompt = f"### Support:\n{support}\n\n### Question:\n{question}\n\n### Response:\n"
    else:
        prompt = f"### Question:\n{question}\n\n### Response:\n"
    return [(prompt, False), (correct_answer, True)]


def _record_to_sft_segments(
    record: Dict[str, object],
    *,
    text_format: str,
) -> Optional[List[Tuple[str, bool]]]:
    normalized_text_format = str(text_format).strip().lower()
    if normalized_text_format == "openorca":
        return _sft_openorca_segments(record)
    if normalized_text_format == "alpaca":
        return _sft_alpaca_segments(record)
    if normalized_text_format == "longalign_chat":
        return _sft_longalign_chat_segments(record)
    if normalized_text_format == "race_mcqa":
        return _sft_race_segments(record)
    if normalized_text_format == "sciq_qa":
        return _sft_sciq_segments(record)
    if normalized_text_format == "edgerazor_messages":
        return _sft_edgerazor_messages_segments(record)
    raise ValueError(
        f"SFT dataset_task does not support text_format={text_format!r}. "
        "Supported formats: openorca, alpaca, longalign_chat, race_mcqa, sciq_qa, edgerazor_messages."
    )


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
    if normalized_text_format == "edgerazor_messages":
        return _format_edgerazor_messages_record(record)
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


def _records_to_text_batch(
    examples: Dict[str, Sequence[object]],
    *,
    text_field: str,
    text_format: str = "auto",
) -> Dict[str, List[Optional[str]]]:
    if not examples:
        return {"text": []}
    keys = list(examples.keys())
    batch_size = len(examples[keys[0]])
    out: List[Optional[str]] = []
    for idx in range(batch_size):
        record = {key: examples[key][idx] for key in keys}
        out.append(_record_to_text(record, text_field=text_field, text_format=text_format))
    return {"text": out}


def _prepare_edgerazor_messages_dataset(
    dataset: Dataset,
    *,
    num_proc: int = 1,
) -> Dataset:
    def _records_to_messages_batch(examples: Dict[str, Sequence[object]]) -> Dict[str, List[Optional[List[Dict[str, str]]]]]:
        if not examples:
            return {"messages": []}
        keys = list(examples.keys())
        batch_size = len(examples[keys[0]])
        out: List[Optional[List[Dict[str, str]]]] = []
        for idx in range(batch_size):
            record = {key: examples[key][idx] for key in keys}
            out.append(_normalize_edgerazor_messages(record))
        return {"messages": out}

    prepared = dataset.map(
        _records_to_messages_batch,
        batched=True,
        remove_columns=list(dataset.column_names),
        num_proc=None if int(num_proc) == 1 else int(num_proc),
    )
    prepared = prepared.filter(lambda rec: rec["messages"] is not None)
    return prepared


def _prepare_text_dataset(
    dataset: Dataset,
    *,
    text_field: str,
    text_format: str = "auto",
    num_proc: int = 1,
) -> Dataset:
    prepared = dataset.map(
        lambda batch: _records_to_text_batch(batch, text_field=text_field, text_format=text_format),
        batched=True,
        remove_columns=list(dataset.column_names),
        num_proc=None if int(num_proc) == 1 else int(num_proc),
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
    columns = [
        name
        for name in (
            "input_ids",
            "attention_mask",
            "labels",
            "choice_input_ids",
            "choice_attention_mask",
            "choice_labels",
            "answer_index",
        )
        if name in dataset.column_names
    ]
    dataset.set_format(type="torch", columns=columns)
    return dataset


def _tokenize_and_pack(dataset: Dataset, tokenizer, *, block_size: int, num_proc: int = 1) -> Dataset:
    tokenized = dataset.map(
        lambda rec: tokenizer(rec["text"]),
        batched=True,
        remove_columns=list(dataset.column_names),
        num_proc=None if int(num_proc) == 1 else int(num_proc),
    )
    packed = tokenized.map(
        lambda rec: _group_texts(rec, block_size=int(block_size)),
        batched=True,
        num_proc=None if int(num_proc) == 1 else int(num_proc),
    )
    return _set_torch_columns(packed)


def _encode_sft_segments(
    segments: Sequence[Tuple[str, bool]],
    tokenizer,
    *,
    block_size: int,
) -> Tuple[List[int], List[int], List[int]]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("SFT dataset_task requires tokenizer.eos_token_id.")

    input_ids: List[int] = []
    labels: List[int] = []
    attention_mask: List[int] = []

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        input_ids.append(int(bos_token_id))
        labels.append(-100)
        attention_mask.append(1)

    has_trainable_token = False
    for text, trainable in segments:
        encoded = tokenizer(
            str(text),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        segment_ids = [int(token_id) for token_id in encoded.get("input_ids", [])]
        if not segment_ids:
            continue
        input_ids.extend(segment_ids)
        attention_mask.extend([1] * len(segment_ids))
        if bool(trainable):
            labels.extend(segment_ids)
            has_trainable_token = True
        else:
            labels.extend([-100] * len(segment_ids))

    input_ids.append(int(eos_token_id))
    labels.append(int(eos_token_id))
    attention_mask.append(1)
    has_trainable_token = True

    if not has_trainable_token:
        raise ValueError("SFT sample has no trainable response tokens.")
    if len(input_ids) > int(block_size):
        raise ValueError(
            f"SFT sample length {len(input_ids)} exceeds --model_max_length={int(block_size)}. "
            "Increase --model_max_length or remove this over-length sample."
        )
    return input_ids, attention_mask, labels


def _records_to_sft_blocks_batch(
    examples: Dict[str, Sequence[object]],
    *,
    text_format: str,
    tokenizer,
    block_size: int,
) -> Dict[str, List[List[int]]]:
    if not examples:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    keys = list(examples.keys())
    batch_size = len(examples[keys[0]])
    buffered_input_ids: List[int] = []
    buffered_attention_mask: List[int] = []
    buffered_labels: List[int] = []
    out_input_ids: List[List[int]] = []
    out_attention_mask: List[List[int]] = []
    out_labels: List[List[int]] = []
    block = int(block_size)

    def emit_full_blocks() -> None:
        while len(buffered_input_ids) >= block:
            out_input_ids.append(list(buffered_input_ids[:block]))
            out_attention_mask.append(list(buffered_attention_mask[:block]))
            out_labels.append(list(buffered_labels[:block]))
            del buffered_input_ids[:block]
            del buffered_attention_mask[:block]
            del buffered_labels[:block]

    for idx in range(batch_size):
        record = {key: examples[key][idx] for key in keys}
        if str(text_format).strip().lower() == "edgerazor_messages":
            try:
                input_ids, attention_mask, labels = _encode_edgerazor_messages_sft(
                    record,
                    tokenizer,
                    block_size=block,
                )
            except ValueError:
                continue
        else:
            segments = _record_to_sft_segments(record, text_format=text_format)
            if segments is None:
                continue
            input_ids, attention_mask, labels = _encode_sft_segments(
                segments,
                tokenizer,
                block_size=block,
            )
        buffered_input_ids.extend(input_ids)
        buffered_attention_mask.extend(attention_mask)
        buffered_labels.extend(labels)
        emit_full_blocks()

    if buffered_input_ids:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("SFT dataset_task requires tokenizer.pad_token_id.")
        pad_len = block - len(buffered_input_ids)
        out_input_ids.append(list(buffered_input_ids) + [int(pad_token_id)] * pad_len)
        out_attention_mask.append(list(buffered_attention_mask) + [0] * pad_len)
        out_labels.append(list(buffered_labels) + [-100] * pad_len)

    return {
        "input_ids": out_input_ids,
        "attention_mask": out_attention_mask,
        "labels": out_labels,
    }


def _tokenize_and_pack_sft(
    dataset: Dataset,
    tokenizer,
    *,
    block_size: int,
    text_format: str,
    num_proc: int = 1,
) -> Dataset:
    packed = dataset.map(
        lambda batch: _records_to_sft_blocks_batch(
            batch,
            text_format=str(text_format),
            tokenizer=tokenizer,
            block_size=int(block_size),
        ),
        batched=True,
        remove_columns=list(dataset.column_names),
        num_proc=None if int(num_proc) == 1 else int(num_proc),
    )
    return _set_torch_columns(packed)


def _records_to_mcqa_batch(
    examples: Dict[str, Sequence[object]],
    *,
    text_format: str,
    tokenizer,
    block_size: int,
) -> Dict[str, List[object]]:
    if not examples:
        return {
            "choice_input_ids": [],
            "choice_attention_mask": [],
            "choice_labels": [],
            "answer_index": [],
        }

    keys = list(examples.keys())
    batch_size = len(examples[keys[0]])
    out_input_ids: List[List[List[int]]] = []
    out_attention_mask: List[List[List[int]]] = []
    out_labels: List[List[List[int]]] = []
    out_answer_index: List[int] = []

    for idx in range(batch_size):
        record = {key: examples[key][idx] for key in keys}
        example = record_to_mcqa_example(record, text_format=text_format)
        if example is None:
            continue
        try:
            encoded = encode_mcqa_example(example, tokenizer, block_size=int(block_size))
        except ValueError:
            continue
        out_input_ids.append(encoded["choice_input_ids"].tolist())
        out_attention_mask.append(encoded["choice_attention_mask"].tolist())
        out_labels.append(encoded["choice_labels"].tolist())
        out_answer_index.append(int(encoded["answer_index"].item()))

    return {
        "choice_input_ids": out_input_ids,
        "choice_attention_mask": out_attention_mask,
        "choice_labels": out_labels,
        "answer_index": out_answer_index,
    }


def _tokenize_mcqa(
    dataset: Dataset,
    tokenizer,
    *,
    block_size: int,
    text_format: str,
    num_proc: int = 1,
) -> Dataset:
    tokenized = dataset.map(
        lambda batch: _records_to_mcqa_batch(
            batch,
            text_format=str(text_format),
            tokenizer=tokenizer,
            block_size=int(block_size),
        ),
        batched=True,
        remove_columns=list(dataset.column_names),
        num_proc=None if int(num_proc) == 1 else int(num_proc),
    )
    return _set_torch_columns(tokenized)


def _resolve_dataset_num_proc(args) -> int:
    raw_num_proc = getattr(args, "dataset_num_proc", 1)
    num_proc = int(raw_num_proc)
    if num_proc < 1:
        raise ValueError(f"dataset_num_proc must be >= 1, got {raw_num_proc}")
    return num_proc


def _should_prepare_eval_dataset(training_args) -> bool:
    eval_strategy = getattr(training_args, "eval_strategy", None)
    normalized = getattr(eval_strategy, "value", eval_strategy)
    if normalized == IntervalStrategy.NO or str(normalized).strip().lower() == "no":
        return False
    return True


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
    if total_rows < len(weights):
        raise ValueError(f"total_rows={total_rows} is smaller than source count={len(weights)}")

    remaining_rows = int(total_rows) - len(weights)
    targets: List[int] = []
    allocated = 0
    for idx, weight in enumerate(weights):
        if idx == len(weights) - 1:
            rows = int(remaining_rows - allocated)
        else:
            rows = int(math.floor(float(remaining_rows) * float(weight)))
            allocated += rows
        targets.append(int(rows) + 1)
    return targets


def _resize_packed_dataset(
    dataset: Dataset,
    *,
    target_rows: int,
    seed: int,
    shuffle: bool = True,
) -> Tuple[Dataset, float]:
    current_rows = len(dataset)
    if current_rows < 1:
        raise ValueError("cannot resize an empty packed dataset.")
    if target_rows < 1:
        raise ValueError(f"target_rows must be >= 1, got {target_rows}")

    shuffled = dataset.shuffle(seed=int(seed)) if bool(shuffle) else dataset
    if current_rows >= target_rows:
        resized = shuffled.select(range(int(target_rows)))
        return _set_torch_columns(resized), 1.0

    full_repeats, remainder = divmod(int(target_rows), int(current_rows))
    indices = list(range(int(current_rows))) * int(full_repeats)
    if remainder > 0:
        indices.extend(range(int(remainder)))
    resized = shuffled.select(indices)
    return _set_torch_columns(resized), float(target_rows) / float(current_rows)


def _sample_and_pack_source(
    raw_dataset: Dataset,
    *,
    target_rows: int,
    tokenizer,
    block_size: int,
    text_field: str,
    text_format: str,
    num_proc: int,
    seed: int,
) -> Tuple[Dataset, Dict[str, object]]:
    if int(target_rows) < 1:
        raise ValueError(f"target_rows must be >= 1, got {target_rows}")

    raw_rows = int(len(raw_dataset))
    chunk_size = max(4096, int(num_proc) * 256)
    shuffled_raw = raw_dataset.shuffle(seed=int(seed))
    packed_chunks: List[Dataset] = []
    processed_raw_rows = 0
    text_rows = 0
    collected_packed_rows = 0

    for start in range(0, raw_rows, chunk_size):
        stop = min(start + chunk_size, raw_rows)
        chunk = shuffled_raw.select(range(start, stop))
        processed_raw_rows += int(len(chunk))

        text_chunk = _prepare_text_dataset(
            chunk,
            text_field=str(text_field),
            text_format=str(text_format),
            num_proc=int(num_proc),
        )
        text_rows += int(len(text_chunk))

        packed_chunk = _tokenize_and_pack(
            text_chunk,
            tokenizer,
            block_size=int(block_size),
            num_proc=int(num_proc),
        )
        if len(packed_chunk) > 0:
            packed_chunks.append(packed_chunk)
            collected_packed_rows += int(len(packed_chunk))

        if collected_packed_rows >= int(target_rows):
            break

    if collected_packed_rows < 1:
        raise ValueError(
            "Packed training dataset for mix source is empty. "
            "Increase source text volume or lower --model_max_length."
        )

    collected = packed_chunks[0] if len(packed_chunks) == 1 else concatenate_datasets(packed_chunks)
    collected = _set_torch_columns(collected.shuffle(seed=int(seed)))
    resized, repeat_factor = _resize_packed_dataset(
        collected,
        target_rows=int(target_rows),
        seed=int(seed),
        shuffle=False,
    )
    return resized, {
        "raw_rows": int(raw_rows),
        "text_rows": int(text_rows),
        "packed_rows": int(collected_packed_rows),
        "target_rows": int(target_rows),
        "repeat_factor": float(repeat_factor),
        "processed_raw_rows": int(processed_raw_rows),
        "limited_preprocessing": bool(processed_raw_rows < raw_rows),
        "sampling_policy": "shuffled_raw_streaming_pack",
        "collected_packed_rows": int(collected_packed_rows),
    }


def _sample_and_pack_sft_source(
    raw_dataset: Dataset,
    *,
    target_rows: int,
    tokenizer,
    block_size: int,
    text_format: str,
    num_proc: int,
    seed: int,
) -> Tuple[Dataset, Dict[str, object]]:
    if int(target_rows) < 1:
        raise ValueError(f"target_rows must be >= 1, got {target_rows}")

    raw_rows = int(len(raw_dataset))
    chunk_size = max(4096, int(num_proc) * 256)
    shuffled_raw = raw_dataset.shuffle(seed=int(seed))
    packed_chunks: List[Dataset] = []
    processed_raw_rows = 0
    collected_packed_rows = 0

    for start in range(0, raw_rows, chunk_size):
        stop = min(start + chunk_size, raw_rows)
        chunk = shuffled_raw.select(range(start, stop))
        processed_raw_rows += int(len(chunk))

        packed_chunk = _tokenize_and_pack_sft(
            chunk,
            tokenizer,
            block_size=int(block_size),
            text_format=str(text_format),
            num_proc=int(num_proc),
        )
        if len(packed_chunk) > 0:
            packed_chunks.append(packed_chunk)
            collected_packed_rows += int(len(packed_chunk))

        if collected_packed_rows >= int(target_rows):
            break

    if collected_packed_rows < 1:
        raise ValueError(
            "Packed SFT training dataset for mix source is empty. "
            "Check the source schema and --dataset_task sft support."
        )

    collected = packed_chunks[0] if len(packed_chunks) == 1 else concatenate_datasets(packed_chunks)
    collected = _set_torch_columns(collected.shuffle(seed=int(seed)))
    resized, repeat_factor = _resize_packed_dataset(
        collected,
        target_rows=int(target_rows),
        seed=int(seed),
        shuffle=False,
    )
    return resized, {
        "raw_rows": int(raw_rows),
        "text_rows": int(processed_raw_rows),
        "packed_rows": int(collected_packed_rows),
        "target_rows": int(target_rows),
        "repeat_factor": float(repeat_factor),
        "processed_raw_rows": int(processed_raw_rows),
        "limited_preprocessing": bool(processed_raw_rows < raw_rows),
        "sampling_policy": "shuffled_raw_streaming_sft_pack",
        "collected_packed_rows": int(collected_packed_rows),
    }


def _sample_and_pack_mcqa_source(
    raw_dataset: Dataset,
    *,
    target_rows: int,
    tokenizer,
    block_size: int,
    text_format: str,
    num_proc: int,
    seed: int,
) -> Tuple[Dataset, Dict[str, object]]:
    if int(target_rows) < 1:
        raise ValueError(f"target_rows must be >= 1, got {target_rows}")

    raw_rows = int(len(raw_dataset))
    chunk_size = max(4096, int(num_proc) * 256)
    shuffled_raw = raw_dataset.shuffle(seed=int(seed))
    packed_chunks: List[Dataset] = []
    processed_raw_rows = 0
    collected_packed_rows = 0

    for start in range(0, raw_rows, chunk_size):
        stop = min(start + chunk_size, raw_rows)
        chunk = shuffled_raw.select(range(start, stop))
        processed_raw_rows += int(len(chunk))

        packed_chunk = _tokenize_mcqa(
            chunk,
            tokenizer,
            block_size=int(block_size),
            text_format=str(text_format),
            num_proc=int(num_proc),
        )
        if len(packed_chunk) > 0:
            packed_chunks.append(packed_chunk)
            collected_packed_rows += int(len(packed_chunk))

        if collected_packed_rows >= int(target_rows):
            break

    if collected_packed_rows < 1:
        raise ValueError(
            "Packed MCQA training dataset for mix source is empty. "
            "Check the source schema and --model_max_length."
        )

    collected = packed_chunks[0] if len(packed_chunks) == 1 else concatenate_datasets(packed_chunks)
    collected = _set_torch_columns(collected.shuffle(seed=int(seed)))
    resized, repeat_factor = _resize_packed_dataset(
        collected,
        target_rows=int(target_rows),
        seed=int(seed),
        shuffle=False,
    )
    return resized, {
        "raw_rows": int(raw_rows),
        "text_rows": int(processed_raw_rows),
        "packed_rows": int(collected_packed_rows),
        "target_rows": int(target_rows),
        "repeat_factor": float(repeat_factor),
        "processed_raw_rows": int(processed_raw_rows),
        "limited_preprocessing": bool(processed_raw_rows < raw_rows),
        "sampling_policy": "shuffled_raw_streaming_mcqa_pack",
        "collected_packed_rows": int(collected_packed_rows),
    }


def _load_preset_raw_datasets(preset: DatasetMixSourcePreset) -> Tuple[Dataset, Optional[Dataset]]:
    resolved_path = _resolve_dataset_path(str(preset.path))
    if resolved_path.lower().endswith((".jsonl", ".json")):
        dataset = load_dataset("json", data_files={"train": resolved_path})
        if isinstance(dataset, DatasetDict):
            if str(preset.train_split) not in dataset:
                raise ValueError(f"Missing train split '{preset.train_split}' in local dataset {resolved_path}.")
            return dataset[str(preset.train_split)], None
        return dataset, None
    if str(preset.alias) == "mmlu":
        train_parts: List[Dataset] = []
        for subject in _MMLU_NO_TRAIN_SUBJECTS:
            dataset = load_dataset(str(preset.path), name=str(subject))
            if not isinstance(dataset, DatasetDict):
                raise RuntimeError(f"Expected DatasetDict from MMLU subject {subject}, got {type(dataset)}")
            missing = [split for split in ("dev", "validation") if split not in dataset]
            if missing:
                raise ValueError(f"Missing MMLU splits for subject {subject}: {missing}")
            subject_train = concatenate_datasets([dataset["dev"], dataset["validation"]])
            if "subject" not in subject_train.column_names:
                subject_train = subject_train.add_column("subject", [str(subject)] * len(subject_train))
            train_parts.append(subject_train)
        if not train_parts:
            raise ValueError("MMLU subject list is empty.")
        return concatenate_datasets(train_parts), None
    return _load_hf_dataset_splits(
        path=preset.path,
        config=preset.config,
        train_split=preset.train_split,
        eval_split=preset.eval_split,
    )


def _build_mixed_datasets(args, training_args, tokenizer):
    block_size = int(training_args.model_max_length)
    dataset_num_proc = _resolve_dataset_num_proc(args)
    prepare_eval = _should_prepare_eval_dataset(training_args)
    dataset_task = str(getattr(args, "dataset_task", "lm")).strip().lower()
    sources = list(getattr(args, "dataset_mix_sources", []) or [])
    weights = list(getattr(args, "dataset_mix_weights", []) or [])
    if not sources or not weights or len(sources) != len(weights):
        raise ValueError("Invalid dataset mix configuration. Run argument validation first.")

    required_examples, target_mixed_examples = _compute_mix_target_examples(training_args)
    target_mixed_examples = max(int(target_mixed_examples), len(sources))
    per_source_targets = _split_target_rows(target_mixed_examples, weights)
    seed = int(getattr(training_args, "seed", 0))

    train_datasets: List[Dataset] = []
    eval_datasets: List[Dataset] = []
    source_stats: List[Dict[str, object]] = []
    for idx, (alias, weight, target_rows) in enumerate(zip(sources, weights, per_source_targets)):
        preset = DATASET_MIX_SOURCE_PRESETS[str(alias)]
        train_raw, eval_raw = _load_preset_raw_datasets(preset)
        source_seed = int(seed + idx)
        if dataset_task == "sft":
            resized_train, train_stats = _sample_and_pack_sft_source(
                train_raw,
                target_rows=int(target_rows),
                tokenizer=tokenizer,
                block_size=int(block_size),
                text_format=str(preset.text_format),
                num_proc=dataset_num_proc,
                seed=int(source_seed),
            )
        elif dataset_task == "mcqa":
            resized_train, train_stats = _sample_and_pack_mcqa_source(
                train_raw,
                target_rows=int(target_rows),
                tokenizer=tokenizer,
                block_size=int(block_size),
                text_format=str(preset.text_format),
                num_proc=dataset_num_proc,
                seed=int(source_seed),
            )
        elif dataset_task == "lm":
            resized_train, train_stats = _sample_and_pack_source(
                train_raw,
                target_rows=int(target_rows),
                tokenizer=tokenizer,
                block_size=int(block_size),
                text_field=str(preset.text_field),
                text_format=str(preset.text_format),
                num_proc=dataset_num_proc,
                seed=int(source_seed),
            )
        else:
            raise ValueError(f"Unsupported dataset_task={dataset_task!r}. Expected 'lm', 'sft', or 'mcqa'.")
        train_datasets.append(resized_train)

        eval_text = None
        if dataset_task == "lm" and prepare_eval and eval_raw is not None:
            eval_text = _prepare_text_dataset(
                eval_raw,
                text_field=str(preset.text_field),
                text_format=str(preset.text_format),
                num_proc=dataset_num_proc,
            )

        eval_packed_rows = 0
        if dataset_task == "sft" and prepare_eval and eval_raw is not None:
            eval_packed = _tokenize_and_pack_sft(
                eval_raw,
                tokenizer,
                block_size=block_size,
                text_format=str(preset.text_format),
                num_proc=dataset_num_proc,
            )
            eval_packed_rows = len(eval_packed)
            if eval_packed_rows > 0:
                eval_datasets.append(eval_packed)
        elif dataset_task == "mcqa" and prepare_eval and eval_raw is not None:
            eval_packed = _tokenize_mcqa(
                eval_raw,
                tokenizer,
                block_size=block_size,
                text_format=str(preset.text_format),
                num_proc=dataset_num_proc,
            )
            eval_packed_rows = len(eval_packed)
            if eval_packed_rows > 0:
                eval_datasets.append(eval_packed)
        elif eval_text is not None:
            eval_packed = _tokenize_and_pack(eval_text, tokenizer, block_size=block_size, num_proc=dataset_num_proc)
            eval_packed_rows = len(eval_packed)
            if eval_packed_rows > 0:
                eval_datasets.append(eval_packed)

        source_stat = {
            "alias": str(alias),
            "weight": float(weight),
            "eval_packed_rows": int(eval_packed_rows),
        }
        source_stat.update(train_stats)
        source_stats.append(source_stat)

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
        "dataset_task": str(dataset_task),
        "block_size": int(block_size),
        "dataset_mix_spec": str(args.dataset_mix_spec),
        "dataset_mix_sources": list(sources),
        "dataset_mix_weights": [float(weight) for weight in weights],
        "dataset_mix_target_examples": int(target_mixed_examples),
        "required_train_examples": int(required_examples),
        "source_stats": source_stats,
    }


def build_datasets(args, training_args, tokenizer):
    from e2e_common.lazy_datasets import (
        build_mixed_lazy_dataset,
        build_single_file_lazy_dataset,
        dataset_length_or_none,
        is_iterable_training_dataset,
    )

    block_size = int(training_args.model_max_length)
    if getattr(args, "dataset_mix_spec", None):
        dataset_task = str(getattr(args, "dataset_task", "lm")).strip().lower()
        normalized_spec, source_stats, train_dataset, train_is_iterable = build_mixed_lazy_dataset(
            str(args.dataset_mix_spec),
            task=dataset_task,
            tokenizer=tokenizer,
            max_seq_len=int(block_size),
            seed=int(getattr(training_args, "seed", 0)),
        )
        eval_dataset = None
        return train_dataset, eval_dataset, {
            "dataset_mode": "mix",
            "dataset_task": str(dataset_task),
            "block_size": int(block_size),
            "dataset_mix_spec": str(normalized_spec),
            "dataset_mix_sources": [str(item["alias"]) for item in source_stats],
            "dataset_mix_weights": [float(item["weight"]) for item in source_stats],
            "lazy_iterable": bool(train_is_iterable),
            "dataset_length": dataset_length_or_none(train_dataset),
            "source_stats": source_stats,
        }

    dataset_task = str(getattr(args, "dataset_task", "lm")).strip().lower()
    if dataset_task in {"sft", "mcqa"}:
        raise ValueError(f"--dataset_task {dataset_task} currently supports --dataset_mix only.")
    if dataset_task != "lm":
        raise ValueError(f"Unsupported dataset_task={dataset_task!r}. Expected 'lm', 'sft', or 'mcqa'.")

    train_dataset = build_single_file_lazy_dataset(
        train_file=str(args.train_file),
        task="lm",
        tokenizer=tokenizer,
        max_seq_len=int(block_size),
        text_field=str(args.text_field),
        text_format="auto",
        max_train_samples=getattr(args, "max_train_samples", None),
    )
    eval_dataset = None
    return train_dataset, eval_dataset, {
        "dataset_mode": "single",
        "dataset_task": str(dataset_task),
        "block_size": int(block_size),
        "lazy_iterable": is_iterable_training_dataset(train_dataset),
        "dataset_length": dataset_length_or_none(train_dataset),
        "source_stats": [],
    }
