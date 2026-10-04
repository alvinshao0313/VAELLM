"""CPU boundaries and causal CE only; no model or substitute teacher is loaded."""
import pytest
import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.teacher_format_diagnostic import answer_observations, prepare_bare, recover_single_turn
from train_utils.distill_data import encode_canonical_record


def test_real_tokenizer_roundtrip_and_explicit_space_preserve_content_and_reject_extra_roles():
    tok = AutoTokenizer.from_pretrained("/home/star/models/Qwen/Qwen3-8B", use_fast=True, trust_remote_code=True)
    messages = [{"role": "user", "content": "Question: which letter?\nAnswer: "}, {"role": "assistant", "content": "C"}]
    item = encode_canonical_record({"messages": messages}, tok, text_format="edgerazor_messages", text_field="text", task="sft", model_max_length=1024)
    parsed = recover_single_turn(tok, item["input_ids"].tolist(), item["labels"].tolist())
    # The production message normalizer strips raw content before rendering.
    assert parsed["user"] == messages[0]["content"].strip() and parsed["answer"] == "C"
    user_with_trailing_space = parsed["user"] + " "
    for sep in ("", " "):
        bare = prepare_bare(tok, user_with_trailing_space, parsed["answer"], separator=sep)
        assert tok.decode(bare["input_ids"], clean_up_tokenization_spaces=False) == user_with_trailing_space + sep + "C"
        assert bare["user_trailing_whitespace_moved_chars"] == 1 and bare["separator_added_bytes"] == len(sep)
        assert bare["context_ids"] + bare["continuation_ids"] == bare["input_ids"]
    multi = encode_canonical_record({"messages": [{"role": "system", "content": "Answer briefly."}] + messages}, tok,
                                    text_format="edgerazor_messages", text_field="text", task="sft", model_max_length=1024)
    with pytest.raises(ValueError, match="single user/assistant"):
        recover_single_turn(tok, multi["input_ids"].tolist(), multi["labels"].tolist())
    with pytest.raises(ValueError, match="separator"):
        prepare_bare(tok, "Question", "C", separator="\n")
    with pytest.raises(ValueError, match="token prefix"):
        prepare_bare(tok, "Answer:", "C", separator="")


def test_answer_ce_and_topfive_have_one_shift_and_include_mixed_boundary_token():
    logits = torch.arange(24).float().reshape(4, 6) / 7
    ids = torch.tensor([1, 2, 3, 4])
    prepared = {"answer_positions": [2, 3], "answer_tokens": [
        {"target_position": 2, "offset": [5, 7], "includes_user_suffix": False, "includes_added_separator": True},
        {"target_position": 3, "offset": [7, 8], "includes_user_suffix": False, "includes_added_separator": False}]}
    result = answer_observations(logits, ids, prepared)
    expected = -torch.log_softmax(logits[[1, 2]], dim=-1)[[0, 1], [3, 4]]
    assert result["answer_nll_sum"] == pytest.approx(float(expected.sum()), abs=1e-6)
    assert result["first_answer_token_ce"] == pytest.approx(float(expected[0]), abs=1e-6)
    assert result["answer_tokens"][0]["prediction_position"] == 1
    assert result["answer_tokens"][0]["includes_added_separator"]
    assert result["answer_tokens"][0]["top5_probabilities"] == pytest.approx(torch.softmax(logits[1], -1)[[5,4,3,2,1]].tolist(), abs=1e-6)
