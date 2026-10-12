from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import sha256

UPSTREAM = Path(__file__).resolve().parent / "upstream"
UPSTREAM_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
DATA_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
DATA_ZIP_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
DATA_ZIP_BYTES = 113932529
WINDOW = 32768
EOS_TOKEN_IDS = (151645, 151643)
PAD_TOKEN_ID = 151643
DATASETS = ("narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum",
            "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p")
BARE_TASKS = frozenset(("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"))
EXPECTED_COUNTS = {name: (150 if name == "multifieldqa_en" else
                          500 if name in ("lcc", "repobench-p") else 200) for name in DATASETS}
CATEGORIES = {
    "single_document_qa": DATASETS[:4],
    "multi_document_qa": DATASETS[4:8],
    "summarization": DATASETS[8:12],
    "few_shot_learning": DATASETS[12:16],
    "synthetic": DATASETS[16:19],
    "code_completion": DATASETS[19:],
}


def official_files():
    provenance = json.loads((UPSTREAM / "provenance.json").read_text())
    if provenance["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Wrong official LongBench source revision.")
    for name, record in provenance["files"].items():
        if sha256(UPSTREAM / name) != record["sha256"]:
            raise ValueError(f"Official source bytes changed: {name}")
    prompts = json.loads((UPSTREAM / "dataset2prompt.json").read_text())
    lengths = json.loads((UPSTREAM / "dataset2maxlen.json").read_text())
    if set(prompts) != set(DATASETS) or set(lengths) != set(DATASETS):
        raise ValueError("Expected all 21 official LongBench v1 tasks.")
    return prompts, lengths, provenance


def model_policy(model_path):
    """Read the local model's inference policy without loading model weights."""
    from transformers import AutoConfig, AutoTokenizer
    path = Path(model_path).resolve()
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type not in ("qwen3", "llama"):
        raise ValueError(f"Unsupported LongBench model type: {config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("A complete original fast tokenizer/chat template is required.")
    generation = json.loads((path / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else list(eos)
    if tokenizer.eos_token_id not in eos or config.max_position_embeddings < WINDOW:
        raise ValueError("Original EOS/context configuration does not cover this protocol.")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id  # Same missing-pad rule as the native training/eval tokenizer.
    if config.model_type == "qwen3" and (eos != list(EOS_TOKEN_IDS) or pad != PAD_TOKEN_ID
                                         or config.rope_scaling is not None):
        raise ValueError("The existing Qwen3 protocol requires its original EOS/pad/unscaled RoPE.")
    return {"model_type": config.model_type,
            "attn_implementation": "flash_attention_2" if config.model_type == "qwen3" else "sdpa",
            "rope_scaling": config.rope_scaling, "rope_theta": config.rope_theta,
            "bos_token_id": tokenizer.bos_token_id, "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "eos_token_id": eos, "pad_token_id": pad,
            "pad_policy": "original tokenizer pad, otherwise original tokenizer EOS"}


def protocol_record(model_path=None):
    _, lengths, _ = official_files()
    record = {"benchmark": "LongBench_v1", "datasets": list(DATASETS), "total_samples": 4750,
            "official_code_commit": UPSTREAM_COMMIT, "dataset_revision": DATA_REVISION,
            "total_window_tokens": WINDOW, "input_budget": "32768 minus each task max_new_tokens",
            "max_new_tokens": lengths, "batch_size": 1, "seed": 42, "do_sample": False,
            "num_beams": 1, "use_cache": True, "rope_scaling": None,
            "eos_token_id": list(EOS_TOKEN_IDS), "pad_token_id": PAD_TOKEN_ID,
            "chat": "official Qwen3 template, add_generation_prompt=True, enable_thinking=False",
            "bare_tasks": sorted(BARE_TASKS),
            "truncation": "middle deletion of FINAL token IDs, including chat; preserve both ends; no decode/reencode",
            "truncation_adaptation": "official retains prompt ends before chat; this Qwen3 adapter reserves template/output space exactly",
            "samsum_stop": "min_length=input_length+1; EOS=[tokenizer.eos_token_id, last token of newline]",
            "extra_fewshot": 0, "score": "official per-task metrics and max over reference answers"}
    if model_path is not None:
        policy = model_policy(model_path)
        if policy["model_type"] == "llama":
            record.update(model_type="llama", rope_scaling=policy["rope_scaling"],
                          rope_theta=policy["rope_theta"], bos_token_id=policy["bos_token_id"],
                          eos_token_id=policy["eos_token_id"], pad_token_id=policy["pad_token_id"],
                          pad_policy=policy["pad_policy"], attn_implementation="sdpa",
                          chat="original Llama chat template with its native defaults; add_generation_prompt=True",
                          truncation_adaptation="official retains prompt ends before chat; this Llama adapter reserves template/output space exactly")
    return record


def input_bytes(ids):
    return np.asarray(ids, dtype="<u4").tobytes()


def middle_truncate(ids, budget):
    if budget < 2:
        raise ValueError("At least two input tokens are required.")
    if len(ids) <= budget:
        return list(ids)
    left = budget // 2
    return list(ids[:left]) + list(ids[-(budget - left):])


def encode_example(tokenizer, dataset, record, prompts, lengths, *, protocol=None):
    prompt = prompts[dataset].format(**record)
    if dataset in BARE_TASKS:
        ids = tokenizer.encode(prompt, add_special_tokens=True)
    else:
        kwargs = {} if protocol is not None and protocol.get("model_type") == "llama" else {"enable_thinking": False}
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, **kwargs)
    if not ids:
        raise ValueError("Empty LongBench prompt.")
    max_new = int(lengths[dataset])
    final_ids = middle_truncate(ids, WINDOW - max_new)
    return final_ids, {
        "prompt_text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "untruncated_tokens": len(ids), "input_tokens": len(final_ids),
        "removed_tokens": len(ids) - len(final_ids), "max_new_tokens": max_new,
        "chat_wrapped": dataset not in BARE_TASKS,
        "input_ids_sha256": hashlib.sha256(input_bytes(final_ids)).hexdigest(),
    }


def generation_options(tokenizer, dataset, input_length, max_new_tokens, *, protocol=None):
    llama = protocol is not None and protocol.get("model_type") == "llama"
    eos = list(protocol["eos_token_id"]) if llama else list(EOS_TOKEN_IDS)
    pad = protocol["pad_token_id"] if llama else PAD_TOKEN_ID
    if not llama and (tokenizer.eos_token_id != EOS_TOKEN_IDS[0] or tokenizer.pad_token_id != PAD_TOKEN_ID):
        raise ValueError("Expected original Qwen3 EOS/pad identities.")
    if llama and (tokenizer.eos_token_id not in eos or tokenizer.pad_token_id not in (None, pad)
                  or tokenizer.bos_token_id != protocol["bos_token_id"]):
        raise ValueError("Llama tokenizer differs from the original model policy.")
    options = {"max_new_tokens": int(max_new_tokens), "num_beams": 1,
               "do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 50,
               "use_cache": True, "pad_token_id": pad,
               "eos_token_id": eos, "repetition_penalty": 1.0,
               "no_repeat_ngram_size": 0, "length_penalty": 1.0}
    if dataset == "samsum":
        newline = tokenizer.encode("\n", add_special_tokens=False)
        if not newline:
            raise ValueError("Tokenizer must encode a newline.")
        options.update(min_length=int(input_length) + 1,
                       eos_token_id=[int(tokenizer.eos_token_id), int(newline[-1])])
    return options
