"""Real local tokenizer/config compatibility; no model weights or GPU."""
import json
from pathlib import Path

from transformers import AutoTokenizer

from experiments.longbench_v1.protocol import (
    BARE_TASKS, WINDOW, encode_example, generation_options, model_policy,
    official_files, protocol_record,
)


ROOT = Path(__file__).resolve().parents[1]
QWEN = Path("/home/star/models/Qwen/Qwen3-8B")
LLAMA = Path("/home/star/models/meta-llama/Llama-3.1-8B-Instruct")
ARCHIVE = ROOT / "result/qwen3_8b_longbench_20261004/data/manifest.json"


def test_existing_qwen_protocol_and_real_record_token_hashes_are_unchanged():
    archive = json.loads(ARCHIVE.read_text())
    assert protocol_record() == protocol_record(QWEN) == archive["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, use_fast=True)
    prompts, lengths, _ = official_files()
    index = [json.loads(line) for line in Path(archive["index"]["path"]).read_text().splitlines()]
    for dataset in ("gov_report", "samsum", "lcc"):
        raw = json.loads((ARCHIVE.parent / "raw" / (dataset + ".jsonl")).read_text().splitlines()[0])
        ids, report = encode_example(tokenizer, dataset, raw, prompts, lengths, protocol=protocol_record(QWEN))
        original = next(row for row in index if row["dataset"] == dataset and row["row_index"] == 0)
        assert report["input_ids_sha256"] == original["input_ids_sha256"]
        assert generation_options(tokenizer, dataset, len(ids), lengths[dataset]) == generation_options(
            tokenizer, dataset, len(ids), lengths[dataset], protocol=archive["protocol"])


def test_llama_original_template_bos_eos_rope_and_final_token_budget():
    tokenizer = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True, use_fast=True)
    original = json.loads((LLAMA / "config.json").read_text())
    policy, protocol = model_policy(LLAMA), protocol_record(LLAMA)
    assert policy["model_type"] == "llama" and policy["attn_implementation"] == "sdpa"
    assert policy["rope_scaling"] == original["rope_scaling"] and policy["rope_theta"] == original["rope_theta"]
    assert protocol["bos_token_id"] == 128000 and protocol["eos_token_id"] == [128001, 128008, 128009]
    assert tokenizer.pad_token_id is None and protocol["pad_token_id"] == tokenizer.eos_token_id == 128009
    prompts, lengths, _ = official_files()
    # A real official long record exercises native chat and actual middle truncation.
    rows = [json.loads(line) for line in (ARCHIVE.parent / "raw/gov_report.jsonl").read_text().splitlines()]
    raw = max(rows, key=lambda item: len(item["context"]))
    ids, metadata = encode_example(tokenizer, "gov_report", raw, prompts, lengths, protocol=protocol)
    full = tokenizer.apply_chat_template([{"role": "user", "content": prompts["gov_report"].format(**raw)}],
                                         tokenize=True, add_generation_prompt=True)
    budget = WINDOW - 512
    assert len(full) > budget and ids == full[:budget // 2] + full[-(budget - budget // 2):]
    assert ids[0] == tokenizer.bos_token_id and len(ids) == budget
    assert metadata["removed_tokens"] == len(full) - budget
    assert tokenizer.decode(ids[-4:]) == "<|start_header_id|>assistant<|end_header_id|>\n\n"
    for dataset in BARE_TASKS:
        raw = json.loads((ARCHIVE.parent / "raw" / (dataset + ".jsonl")).read_text().splitlines()[0])
        ids, report = encode_example(tokenizer, dataset, raw, prompts, lengths, protocol=protocol)
        direct = tokenizer.encode(prompts[dataset].format(**raw), add_special_tokens=True)
        assert ids[0] == tokenizer.bos_token_id and report["chat_wrapped"] is False
        assert report["untruncated_tokens"] == len(direct)
        options = generation_options(tokenizer, dataset, len(ids), lengths[dataset], protocol=protocol)
        assert options["pad_token_id"] == 128009 and not options["do_sample"] and options["use_cache"]
        assert options["eos_token_id"] == ([128009, tokenizer.encode("\n", add_special_tokens=False)[-1]]
                                           if dataset == "samsum" else [128001, 128008, 128009])
