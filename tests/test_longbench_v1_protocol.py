"""CPU properties for the isolated, pinned LongBench v1 adapter."""
import hashlib

import pytest
from transformers import AutoTokenizer, GenerationConfig

from experiments.longbench_v1.protocol import (
    BARE_TASKS, DATASETS, EOS_TOKEN_IDS, EXPECTED_COUNTS, WINDOW, encode_example,
    generation_options, input_bytes, official_files,
)
from experiments.longbench_v1.scoring import check_prediction, score_example


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("/home/star/models/Qwen/Qwen3-8B", use_fast=True, local_files_only=True)


def test_real_qwen_final_tokens_preserve_both_ends_and_generation_budget(tokenizer):
    prompts, lengths, _ = official_files()
    assert sum(EXPECTED_COUNTS.values()) == 4750
    assert len(DATASETS) == 21
    record = {"context": " Long context evidence." * 11000, "input": "What is the evidence?"}
    ids, audit = encode_example(tokenizer, "gov_report", record, prompts, lengths)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts["gov_report"].format(**record)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    budget = WINDOW - 512
    assert len(full) > WINDOW
    assert len(ids) == budget and ids == full[:budget // 2] + full[-(budget - budget // 2):]
    assert audit["input_ids_sha256"] == hashlib.sha256(input_bytes(ids)).hexdigest()
    assert audit["removed_tokens"] == len(full) - budget
    assert tokenizer.decode(ids[-8:]).endswith("<think>\n\n</think>\n\n")


def test_official_bare_tasks_and_explicit_shared_stopping(tokenizer):
    prompts, lengths, _ = official_files()
    record = {"context": "甲乙丙\nAnswer: example", "input": "Question: test\nAnswer:"}
    for dataset in DATASETS:
        ids, audit = encode_example(tokenizer, dataset, record, prompts, lengths)
        assert audit["chat_wrapped"] == (dataset not in BARE_TASKS)
        if dataset in BARE_TASKS:
            assert ids == tokenizer.encode(prompts[dataset].format(**record), add_special_tokens=True)
        options = generation_options(tokenizer, dataset, len(ids), lengths[dataset])
        generation = GenerationConfig(**options)
        assert generation.do_sample is False and generation.num_beams == 1 and generation.use_cache is True
        assert generation.pad_token_id == 151643 and generation.repetition_penalty == 1.0
        if dataset == "samsum":
            assert generation.eos_token_id == [151645, tokenizer.encode("\n", add_special_tokens=False)[-1]]
            assert generation.min_length == len(ids) + 1
        else:
            assert generation.eos_token_id == list(EOS_TOKEN_IDS)
            assert "min_length" not in options


def test_official_metrics_reference_max_first_line_and_diagnostic_rejection():
    official_files()
    assert score_example("qasper", "The blue whale.", ["wrong", "blue whale"], None) == 1.0
    assert score_example("triviaqa", "\nParis\nLondon", ["Paris"], None) == 1.0
    assert score_example("passage_count", "3 4", ["3"], None) == 0.5
    assert score_example("passage_retrieval_zh", "段落3", ["段落3"], None) == 1.0
    assert score_example("trec", "location human", ["location"], ["location", "human"]) == 0.5
    assert score_example("lcc", "\n# comment\nreturn x", ["return x"], None) == 1.0
    # rouge 1.0.1 keeps its 1e-8 denominator epsilon, even for identical text.
    assert score_example("gov_report", "one two three", ["one two three"], None) == pytest.approx(1.0, abs=1e-8, rel=0)
    source = {"dataset": "qasper", "row_index": 0, "_id": "cpu-test", "answers": ["blue whale"],
              "all_classes": None, "length": 2000, "input_ids_sha256": "same-input"}
    prediction = {**source, "pred": "blue whale", "config_sha256": "same-config", "diagnostic": False}
    check_prediction(prediction, source, "same-config")
    with pytest.raises(ValueError, match="Diagnostic"):
        check_prediction({**prediction, "diagnostic": True}, source, "same-config")
    with pytest.raises(ValueError, match="mismatch"):
        check_prediction({**prediction, "input_ids_sha256": "different-input"}, source, "same-config")


def test_real_qwen_generation_mixin_does_not_inherit_sampling_defaults():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, eos_token_id=7))
    model.generation_config = GenerationConfig(do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                               eos_token_id=[7, 1], pad_token_id=1)
    options = dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=50,
                   eos_token_id=[7, 1], pad_token_id=1, max_new_tokens=512, use_cache=True, num_beams=1)
    inherited, _ = model._prepare_generation_config(GenerationConfig(**options))
    assert inherited.do_sample is True  # Actual 4.51 regression reproduced, no mock model.
    effective, _ = model._prepare_generation_config(GenerationConfig(**options), use_model_defaults=False)
    assert all(getattr(effective, key) == value for key, value in options.items())
    assert next(model.parameters()).device.type == "cpu"
