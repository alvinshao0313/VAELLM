import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import IterableDataset

from e2e_common.data import DATASET_MIX_SOURCE_PRESETS
from train_utils.config.configs import DistillDataConfig
from train_utils.distill_data import (
    FORMATTING_VERSION,
    build_distill_data_collator,
    build_distill_dataset,
    distill_dataset_cache_key,
    encode_canonical_record,
)


class WordTokenizer:
    """Deterministic word tokenizer with optional chat-template + offset mapping."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1
    bos_token_id = None
    chat_template = "chat"
    padding_side = "right"

    def __init__(self, *, support_assistant_mask: bool = False, support_offset_mapping: bool = True):
        self.support_assistant_mask = bool(support_assistant_mask)
        self.support_offset_mapping = bool(support_offset_mapping)
        self.name_or_path = "dummy-word-tokenizer"
        self.init_kwargs = {"revision": "main"}

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **_kwargs,
    ):
        parts = []
        ids = []
        mask = []
        for message in messages:
            role = str(message["role"])
            content = str(message["content"])
            parts.append(f"<|{role}|> {content}")
            role_ids = self._encode(f"<|{role}|>")
            content_ids = self._encode(content)
            ids.extend(role_ids)
            ids.extend(content_ids)
            mask.extend([False] * len(role_ids))
            mask.extend([role == "assistant"] * len(content_ids))
        if add_generation_prompt:
            parts.append("<|assistant|>")
            prompt_ids = self._encode("<|assistant|>")
            ids.extend(prompt_ids)
            mask.extend([False] * len(prompt_ids))
        text = "\n".join(parts)
        if not tokenize:
            return text
        if return_assistant_tokens_mask and not self.support_assistant_mask:
            raise TypeError("assistant mask unsupported")
        payload = {"input_ids": ids}
        if return_assistant_tokens_mask:
            payload["assistant_masks"] = mask
        if return_dict:
            return payload
        return ids

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._encode(text)

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None, **_kwargs):
        del padding
        if isinstance(encoded_inputs, dict):
            items = [encoded_inputs]
            single = True
        else:
            items = list(encoded_inputs)
            single = False
        max_len = max(len(item["input_ids"]) for item in items)
        if max_length is not None:
            max_len = max(max_len, int(max_length))
        if pad_to_multiple_of:
            rem = max_len % int(pad_to_multiple_of)
            if rem:
                max_len += int(pad_to_multiple_of) - rem
        padded = []
        for item in items:
            ids = list(item["input_ids"])
            mask = list(item.get("attention_mask", [1] * len(ids)))
            pad_n = max_len - len(ids)
            ids = ids + [self.pad_token_id] * pad_n
            mask = mask + [0] * pad_n
            out = {"input_ids": ids, "attention_mask": mask}
            if "labels" in item:
                labels = list(item["labels"]) + [-100] * pad_n
                out["labels"] = labels
            padded.append(out)
        if return_tensors == "pt":
            import torch

            batch = {
                key: torch.tensor([row[key] for row in padded], dtype=torch.long)
                for key in padded[0]
            }
            return batch
        if single:
            return padded[0]
        return padded

    def __call__(self, text, **kwargs):
        ids = self._encode(text)
        out = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if kwargs.get("return_offsets_mapping") and self.support_offset_mapping:
            out["offset_mapping"] = self._offsets(str(text))
        if not self.support_offset_mapping and kwargs.get("return_offsets_mapping"):
            raise TypeError("offset mapping unsupported")
        return out

    @staticmethod
    def _encode(text: str):
        tokens = [tok for tok in str(text).split() if tok]
        return [sum(ord(ch) for ch in tok) % 997 + 2 for tok in tokens] or [2]

    @staticmethod
    def _offsets(text: str):
        spans = []
        idx = 0
        for tok in [t for t in text.split() if t]:
            start = text.find(tok, idx)
            end = start + len(tok)
            spans.append((start, end))
            idx = end
        return spans or [(0, 0)]


class ContextSensitiveChatMLTokenizer:
    """Character tokenizer reproducing Qwen3's final-assistant rendering change."""

    eos_token_id = 200000
    chat_template = "context-sensitive-chatml"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_assistant_tokens_mask=False,
        **_kwargs,
    ):
        if return_assistant_tokens_mask:
            raise TypeError("assistant mask unsupported")
        parts = []
        for index, message in enumerate(messages):
            role = str(message["role"])
            content = str(message["content"])
            if role == "assistant" and index == len(messages) - 1:
                content = f"<think>\n\n</think>\n\n{content}"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        if tokenize:
            return self(text)["input_ids"]
        return text

    def __call__(self, text, **kwargs):
        value = str(text)
        out = {"input_ids": [ord(char) + 2 for char in value]}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = [(index, index + 1) for index in range(len(value))]
        return out


def _openorca_record():
    return {
        "question": "what is two plus two",
        "response": "four is correct",
        "system_prompt": "be brief",
    }


def _alpaca_record():
    return {
        "instruction": "summarize this",
        "input": "long document here",
        "output": "short summary",
    }


def _sciq_record():
    return {
        "support": "water freezes at zero",
        "question": "what temperature does water freeze",
        "correct_answer": "zero celsius",
    }


def _race_record():
    return {
        "article": "cats sleep often",
        "question": "what do cats do",
        "options": ["run", "sleep", "fly", "swim"],
        "answer": "B",
    }


def _edgerazor_record():
    return {
        "messages": [
            {"role": "user", "content": "hello there friend"},
            {"role": "assistant", "content": "hi friend reply"},
        ]
    }


def _longalign_record():
    return {
        "messages": [
            {"role": "human", "content": "tell me a fact"},
            {"role": "gpt", "content": "earth orbits sun"},
        ]
    }


@pytest.mark.parametrize(
    ("text_format", "record"),
    [
        ("openorca", _openorca_record()),
        ("alpaca", _alpaca_record()),
        ("sciq_qa", _sciq_record()),
        ("race_mcqa", _race_record()),
        ("edgerazor_messages", _edgerazor_record()),
        ("longalign_chat", _longalign_record()),
    ],
)
def test_lm_and_sft_share_identical_input_ids(text_format, record):
    tokenizer = WordTokenizer(support_assistant_mask=True, support_offset_mapping=True)
    lm = encode_canonical_record(
        record,
        tokenizer,
        text_format=text_format,
        text_field="text",
        task="lm",
        model_max_length=64,
    )
    sft = encode_canonical_record(
        record,
        tokenizer,
        text_format=text_format,
        text_field="text",
        task="sft",
        model_max_length=64,
    )
    assert lm is not None and sft is not None
    assert lm["input_ids"].tolist() == sft["input_ids"].tolist()
    assert any(int(v) != -100 for v in sft["labels"].tolist())
    assert lm["labels"].tolist() == lm["input_ids"].tolist()


def test_model_max_length_one_rejected_by_config():
    with pytest.raises(ValueError, match="model_max_length"):
        DistillDataConfig(dataset_mix="openorca=1.0", model_max_length=1).validate()


def test_model_max_length_two_keeps_eos_and_one_payload_token():
    tokenizer = WordTokenizer()
    sample = encode_canonical_record(
        {"text": "alpha beta gamma"},
        tokenizer,
        text_format="text",
        text_field="text",
        task="lm",
        model_max_length=2,
    )
    assert sample is not None
    assert len(sample["input_ids"]) == 2
    assert int(sample["input_ids"][-1].item()) == int(tokenizer.eos_token_id)
    assert sample["labels"].tolist() == sample["input_ids"].tolist()


def test_sft_skips_when_response_fully_truncated_except_terminal_eos():
    tokenizer = WordTokenizer(support_offset_mapping=True)
    record = {
        "instruction": " ".join(f"word{i}" for i in range(20)),
        "input": "",
        "output": "answer token",
    }
    sample = encode_canonical_record(
        record,
        tokenizer,
        text_format="alpaca",
        text_field="text",
        task="sft",
        model_max_length=4,
    )
    assert sample is None


def test_heterogeneous_sft_mix_is_source_aware(tmp_path):
    openorca = tmp_path / "openorca.jsonl"
    alpaca = tmp_path / "alpaca.jsonl"
    openorca.write_text(
        json.dumps(_openorca_record()) + "\n",
        encoding="utf-8",
    )
    alpaca.write_text(
        json.dumps(
            {
                "instruction": "say hi",
                "input": "",
                "output": "hello there",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    from e2e_common import data as data_module
    from e2e_common.data import DatasetMixSourcePreset

    presets = {
        "tmp_openorca": DatasetMixSourcePreset(
            alias="tmp_openorca",
            path=str(openorca),
            config=None,
            train_split="train",
            eval_split=None,
            text_field="text",
            text_format="openorca",
            supports_lm=True,
            supports_sft=True,
        ),
        "tmp_alpaca": DatasetMixSourcePreset(
            alias="tmp_alpaca",
            path=str(alpaca),
            config=None,
            train_split="train",
            eval_split=None,
            text_field="text",
            text_format="alpaca",
            supports_lm=True,
            supports_sft=True,
        ),
    }
    original = dict(data_module.DATASET_MIX_SOURCE_PRESETS)
    data_module.DATASET_MIX_SOURCE_PRESETS.clear()
    data_module.DATASET_MIX_SOURCE_PRESETS.update(presets)
    try:
        cfg = DistillDataConfig(
            dataset_mix="tmp_openorca=1.0,tmp_alpaca=1.0",
            dataset_task="sft",
            model_max_length=64,
            seed=3,
            data_seed=3,
        )
        cfg.validate()
        bundle = build_distill_dataset(cfg, WordTokenizer(support_offset_mapping=True))
        assert isinstance(bundle.train_dataset, IterableDataset)
        rows = []
        for idx, row in enumerate(bundle.train_dataset):
            rows.append(row)
            if idx >= 3:
                break
        assert rows
        assert all(set(row.keys()) == {"input_ids", "attention_mask", "labels"} for row in rows)
    finally:
        data_module.DATASET_MIX_SOURCE_PRESETS.clear()
        data_module.DATASET_MIX_SOURCE_PRESETS.update(original)


def test_lm_only_source_rejects_sft_task():
    cfg = DistillDataConfig(dataset_mix="wiki=1.0", dataset_task="sft", model_max_length=32)
    cfg.validate()
    with pytest.raises(ValueError, match="supports_sft"):
        build_distill_dataset(cfg, WordTokenizer())


def test_cache_key_includes_required_fields_excludes_dynamic_padding():
    tokenizer = WordTokenizer()
    cfg_a = DistillDataConfig(
        dataset_mix="openorca=1.0",
        dataset_task="sft",
        model_max_length=32,
        data_seed=9,
        dynamic_padding=True,
    )
    cfg_b = DistillDataConfig(
        dataset_mix="openorca=1.0",
        dataset_task="sft",
        model_max_length=32,
        data_seed=9,
        dynamic_padding=False,
    )
    cfg_a.validate()
    cfg_b.validate()
    key_a = distill_dataset_cache_key(cfg_a, tokenizer)
    key_b = distill_dataset_cache_key(cfg_b, tokenizer)
    assert key_a == key_b
    assert FORMATTING_VERSION in key_a
    assert "sft" in key_a
    assert 32 in key_a
    assert 9 in key_a


def test_build_distill_data_collator_defaults_to_dynamic_seq2seq():
    from transformers import DataCollatorForSeq2Seq

    tokenizer = WordTokenizer()
    collator = build_distill_data_collator(tokenizer, model_max_length=32, dynamic_padding=True)
    assert isinstance(collator, DataCollatorForSeq2Seq)
    assert collator.padding == "longest"
    assert collator.pad_to_multiple_of == 8
    assert collator.label_pad_token_id == -100

    fixed = build_distill_data_collator(tokenizer, model_max_length=32, dynamic_padding=False)
    assert fixed.padding == "max_length"
    assert fixed.max_length == 32


def test_lora_prepare_distill_datasets_uses_public_builder():
    from train_utils.lora_data import prepare_distill_datasets

    tokenizer = WordTokenizer(support_assistant_mask=True, support_offset_mapping=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "openorca.jsonl"
        path.write_text(json.dumps(_openorca_record()) + "\n", encoding="utf-8")
        from e2e_common import data as data_module
        from e2e_common.data import DatasetMixSourcePreset

        presets = {
            "tmp_one": DatasetMixSourcePreset(
                alias="tmp_one",
                path=str(path),
                config=None,
                train_split="train",
                eval_split=None,
                text_field="text",
                text_format="openorca",
                supports_lm=True,
                supports_sft=True,
            )
        }
        original = dict(data_module.DATASET_MIX_SOURCE_PRESETS)
        data_module.DATASET_MIX_SOURCE_PRESETS.clear()
        data_module.DATASET_MIX_SOURCE_PRESETS.update(presets)
        try:
            mix, stats, train_ds, eval_ds, _ = prepare_distill_datasets(
                "tmp_one=1.0",
                task="sft",
                seed=1,
                tokenizer=tokenizer,
                max_seq_len=64,
            )
            assert "tmp_one=1.0" in mix or mix.startswith("tmp_one=")
            assert stats
            assert eval_ds is None
            assert isinstance(train_ds, IterableDataset)
            row = next(iter(train_ds))
            assert set(row.keys()) == {"input_ids", "attention_mask", "labels"}
        finally:
            data_module.DATASET_MIX_SOURCE_PRESETS.clear()
            data_module.DATASET_MIX_SOURCE_PRESETS.update(original)


def test_prepare_distill_datasets_openorca_shorthand_equals_weighted():
    from train_utils.lora_data import prepare_distill_datasets

    tokenizer = WordTokenizer(support_offset_mapping=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "openorca.jsonl"
        path.write_text(json.dumps(_openorca_record()) + "\n", encoding="utf-8")
        from e2e_common import data as data_module
        from e2e_common.data import DatasetMixSourcePreset

        presets = {
            "openorca": DatasetMixSourcePreset(
                alias="openorca",
                path=str(path),
                config=None,
                train_split="train",
                eval_split=None,
                text_field="text",
                text_format="openorca",
                supports_lm=True,
                supports_sft=True,
            )
        }
        original = dict(data_module.DATASET_MIX_SOURCE_PRESETS)
        data_module.DATASET_MIX_SOURCE_PRESETS.clear()
        data_module.DATASET_MIX_SOURCE_PRESETS.update(presets)
        try:
            mix_short, stats_short, ds_short, _, _ = prepare_distill_datasets(
                "openorca",
                task="sft",
                seed=11,
                tokenizer=tokenizer,
                max_seq_len=64,
            )
            mix_full, stats_full, ds_full, _, _ = prepare_distill_datasets(
                "openorca=1.0",
                task="sft",
                seed=11,
                tokenizer=tokenizer,
                max_seq_len=64,
            )
            assert mix_short == mix_full
            assert stats_short[0]["alias"] == stats_full[0]["alias"]
            assert isinstance(ds_short, IterableDataset)
            assert isinstance(ds_full, IterableDataset)
            row_short = next(iter(ds_short))
            row_full = next(iter(ds_full))
            assert row_short["input_ids"].tolist() == row_full["input_ids"].tolist()
            assert row_short["labels"].tolist() == row_full["labels"].tolist()
        finally:
            data_module.DATASET_MIX_SOURCE_PRESETS.clear()
            data_module.DATASET_MIX_SOURCE_PRESETS.update(original)


def test_assistant_mask_matches_prefix_fallback_response_labels():
    record = _edgerazor_record()
    masked = encode_canonical_record(
        record,
        WordTokenizer(support_assistant_mask=True, support_offset_mapping=True),
        text_format="edgerazor_messages",
        text_field="text",
        task="sft",
        model_max_length=64,
    )
    fallback = encode_canonical_record(
        record,
        WordTokenizer(support_assistant_mask=False, support_offset_mapping=True),
        text_format="edgerazor_messages",
        text_field="text",
        task="sft",
        model_max_length=64,
    )
    assert masked is not None and fallback is not None
    assert masked["input_ids"].tolist() == fallback["input_ids"].tolist()
    assert masked["labels"].tolist() == fallback["labels"].tolist()


def test_context_sensitive_chatml_multiturn_builds_exact_response_mask():
    tokenizer = ContextSensitiveChatMLTokenizer()
    record = {
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]
    }

    sample = encode_canonical_record(
        record,
        tokenizer,
        text_format="edgerazor_messages",
        text_field="messages",
        task="sft",
        model_max_length=1000,
    )

    assert sample is not None
    full_text = tokenizer.apply_chat_template(record["messages"], add_generation_prompt=False)
    labels = sample["labels"].tolist()
    assert sample["input_ids"][:-1].tolist() == tokenizer(full_text)["input_ids"]
    for content in ("first answer", "second answer"):
        start = full_text.index(content)
        assert all(value != -100 for value in labels[start : start + len(content)])
    for content in ("first question", "second question"):
        start = full_text.index(content)
        assert all(value == -100 for value in labels[start : start + len(content)])
    first_header = full_text.index("<|im_start|>assistant\n")
    assert all(value == -100 for value in labels[first_header : first_header + 22])


class _CountingTokenizer(WordTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0
        self.apply_chat_count = 0

    def __call__(self, text, **kwargs):
        self.call_count += 1
        return super().__call__(text, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        self.apply_chat_count += 1
        return super().apply_chat_template(messages, **kwargs)


def test_map_style_lm_train_file_tokenizes_only_on_access(tmp_path):
    train_file = tmp_path / "lm.jsonl"
    train_file.write_text(
        json.dumps({"text": "alpha beta gamma"})
        + "\n"
        + json.dumps({"text": "delta epsilon"})
        + "\n",
        encoding="utf-8",
    )
    tokenizer = _CountingTokenizer(support_offset_mapping=True)
    cfg = DistillDataConfig(
        train_file=str(train_file),
        dataset_task="lm",
        text_field="text",
        model_max_length=32,
        seed=1,
        data_seed=1,
    )
    cfg.validate()
    bundle = build_distill_dataset(cfg, tokenizer)
    assert tokenizer.call_count == 0
    assert tokenizer.apply_chat_count == 0
    _ = bundle.train_dataset[0]
    assert tokenizer.call_count == 1
    _ = bundle.train_dataset[1]
    assert tokenizer.call_count == 2
    _ = bundle.train_dataset[0]
    assert tokenizer.call_count == 3


def test_single_source_sft_is_lazy_iterable_without_eager_tokenize(tmp_path):
    path = tmp_path / "openorca.jsonl"
    path.write_text(
        "\n".join(json.dumps(_openorca_record()) for _ in range(3)) + "\n",
        encoding="utf-8",
    )
    from e2e_common import data as data_module
    from e2e_common.data import DatasetMixSourcePreset

    presets = {
        "tmp_sft": DatasetMixSourcePreset(
            alias="tmp_sft",
            path=str(path),
            config=None,
            train_split="train",
            eval_split=None,
            text_field="text",
            text_format="openorca",
            supports_lm=True,
            supports_sft=True,
        )
    }
    original = dict(data_module.DATASET_MIX_SOURCE_PRESETS)
    data_module.DATASET_MIX_SOURCE_PRESETS.clear()
    data_module.DATASET_MIX_SOURCE_PRESETS.update(presets)
    try:
        tokenizer = _CountingTokenizer(support_offset_mapping=True)
        cfg = DistillDataConfig(
            dataset_mix="tmp_sft=1.0",
            dataset_task="sft",
            model_max_length=64,
            seed=2,
            data_seed=2,
            group_by_length=True,
        )
        cfg.validate()
        bundle = build_distill_dataset(cfg, tokenizer)
        assert isinstance(bundle.train_dataset, IterableDataset)
        assert bundle.is_iterable is True
        assert bundle.group_by_length is False
        assert tokenizer.call_count == 0
        rows = list(bundle.train_dataset)
        assert len(rows) == 3
        assert tokenizer.call_count == 3
    finally:
        data_module.DATASET_MIX_SOURCE_PRESETS.clear()
        data_module.DATASET_MIX_SOURCE_PRESETS.update(original)


def test_iterable_propagates_structural_value_error(tmp_path, monkeypatch):
    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps(_edgerazor_record()) + "\n", encoding="utf-8")
    from e2e_common import data as data_module
    from e2e_common.chat_template_utils import render_messages as real_render
    from e2e_common.data import DatasetMixSourcePreset

    presets = {
        "tmp_chat": DatasetMixSourcePreset(
            alias="tmp_chat",
            path=str(path),
            config=None,
            train_split="train",
            eval_split=None,
            text_field="text",
            text_format="edgerazor_messages",
            supports_lm=True,
            supports_sft=True,
        )
    }
    original = dict(data_module.DATASET_MIX_SOURCE_PRESETS)
    data_module.DATASET_MIX_SOURCE_PRESETS.clear()
    data_module.DATASET_MIX_SOURCE_PRESETS.update(presets)

    call_state = {"n": 0}

    def flaky_render(messages, tokenizer, add_generation_prompt=False):
        call_state["n"] += 1
        text = real_render(messages, tokenizer, add_generation_prompt=add_generation_prompt)
        # Break a later prefix render so prefix tokens are not a strict prefix.
        if call_state["n"] > 1 and not add_generation_prompt:
            return text + " EXTRA"
        return text

    try:
        tokenizer = WordTokenizer(support_assistant_mask=False, support_offset_mapping=True)
        cfg = DistillDataConfig(
            dataset_mix="tmp_chat=1.0",
            dataset_task="sft",
            model_max_length=64,
            seed=1,
            data_seed=1,
        )
        cfg.validate()
        bundle = build_distill_dataset(cfg, tokenizer)
        monkeypatch.setattr(
            "e2e_common.chat_template_utils.render_messages",
            flaky_render,
        )
        with pytest.raises(ValueError, match="strict prefix"):
            list(bundle.train_dataset)
    finally:
        data_module.DATASET_MIX_SOURCE_PRESETS.clear()
        data_module.DATASET_MIX_SOURCE_PRESETS.update(original)


def test_supports_flags_match_plan_conservative_matrix():
    assert DATASET_MIX_SOURCE_PRESETS["wiki"].supports_lm is True
    assert DATASET_MIX_SOURCE_PRESETS["wiki"].supports_sft is False
    assert DATASET_MIX_SOURCE_PRESETS["fineweb_edu"].supports_sft is False
    assert DATASET_MIX_SOURCE_PRESETS["openorca"].supports_sft is True
    assert DATASET_MIX_SOURCE_PRESETS["alpaca"].supports_sft is True
    assert DATASET_MIX_SOURCE_PRESETS["longalign"].supports_sft is True
    assert DATASET_MIX_SOURCE_PRESETS["edgerazor_ii_7m"].supports_sft is True
    assert DATASET_MIX_SOURCE_PRESETS["race"].supports_sft is True
    assert DATASET_MIX_SOURCE_PRESETS["sciq"].supports_sft is True
    assert DATASET_MIX_SOURCE_PRESETS["mmlu"].supports_sft is False
