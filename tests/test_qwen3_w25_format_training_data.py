"""Real-tokenizer CPU contracts for the isolated paired source archive."""
import copy
import json

import pytest
import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.format_training_data import (
    RejectedPair, _check_batch, _index_source, _select_pairs, encode_paired_record,
)
from train_utils.distill_data import build_distill_data_collator, encode_canonical_record


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("/home/star/models/Qwen/Qwen3-8B", use_fast=True,
                                         trust_remote_code=True, local_files_only=True)


def record(user="Question: choose a letter. Answer:", answer="C"):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


def test_real_canonical_labels_and_bare_answer_overlap_are_distinct(tokenizer):
    for answer in ("C", "2).", "genes."):
        raw = record("  Question: choose a letter. Answer: \n", "\n" + answer + " ")
        pair, metadata = encode_paired_record(raw, tokenizer)
        canonical = encode_canonical_record(raw, tokenizer, text_format="edgerazor_messages", text_field="text",
                                            task="sft", model_max_length=1024)
        assert pair["chatml"] == {k: canonical[k].tolist() for k in ("input_ids", "labels")}
        bare = pair["bare_space"]
        assert tokenizer.decode(bare["input_ids"], clean_up_tokenization_spaces=False) == "Question: choose a letter. Answer: " + answer
        assert [i for i, label in enumerate(bare["labels"]) if label != -100] == metadata["bare_space"]["answer_positions"]
        assert metadata["bare_space"]["prefix_exact"] and metadata["bare_space"]["separator_added_bytes"] == 1
        assert sum(v != -100 for v in pair["chatml"]["labels"]) > len(metadata["chatml"]["answer_positions"])


def test_shared_selection_replays_real_rows_and_skips_duplicates_without_truncation(tokenizer, tmp_path):
    rows = [record(f"Question {i}: choose. Answer:") for i in range(4)]
    rows += [rows[0], record("  Question 0: choose. Answer:\n", " C "),
             record("Long " * 1300), {"messages": [{"role": "system", "content": "System"}] + rows[1]["messages"]}]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    offsets, _ = _index_source(source)
    first = _select_pairs(source, offsets, tokenizer, count=4)
    second = _select_pairs(source, offsets, tokenizer, count=4)
    assert first == second
    identities, _, selection = first
    assert len({s["content_pair_sha256"] for s in identities}) == 4
    assert selection["skip_counts"] and len(selection["skipped"]) + 4 == selection["visited_rows"]
    for sample in identities:
        actual = rows[sample["raw_line_number"] - 1]
        assert encode_paired_record(actual, tokenizer)[1] == sample["formats"]
    with pytest.raises(RejectedPair, match="exceeds_1024"):
        encode_paired_record(rows[-2], tokenizer)


def test_dynamic_batch_masks_reject_answer_or_padding_corruption(tokenizer):
    pairs = [encode_paired_record(record(f"Question {'long ' * i}{i}. Answer:"), tokenizer) for i in range(4)]
    samples = [{"formats": meta} for _, meta in pairs]
    collator = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)
    batch = dict(collator([pair["bare_space"] for pair, _ in pairs]))
    _check_batch(batch, samples, "bare_space", pad_id=tokenizer.pad_token_id)
    broken = copy.deepcopy(batch)
    broken["labels"][0, samples[0]["formats"]["bare_space"]["answer_positions"][0]] = -100
    with pytest.raises(ValueError, match="answer token"):
        _check_batch(broken, samples, "bare_space", pad_id=tokenizer.pad_token_id)
    broken = copy.deepcopy(batch)
    padding = int(broken["attention_mask"][0].sum())
    assert padding < broken["input_ids"].shape[1]
    broken["labels"][0, padding] = broken["input_ids"][0, padding]
    with pytest.raises(ValueError, match="Padding IDs/labels"):
        _check_batch(broken, samples, "bare_space", pad_id=tokenizer.pad_token_id)
    assert all(t.device.type == "cpu" and t.dtype == torch.int64 for t in batch.values())
