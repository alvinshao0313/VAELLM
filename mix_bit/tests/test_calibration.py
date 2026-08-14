from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mix_bit.calibration import (
    build_causal_kl_mask,
    prepare_calibration_dataset,
    resolve_record_schema,
)
from mix_bit.model_inventory import ModelInventory, TargetLinearSpec
from mix_bit.schema import (
    CalibrationConfig,
    CandidateMode,
    CandidateSpaceConfig,
    CandidateTrainingSpec,
    CategorySpec,
    MixBitRunConfig,
    ModelProfile,
    ResolvedRunConfig,
    TrainingRecipeConfig,
)


class FakeTokenizer:
    def __init__(
        self,
        *,
        vocab_size: int = 32,
        pad_token_id: int | None = 0,
        eos_token_id: int = 0,
        name_or_path: str = "fake-tokenizer",
        chat_prefix: list[int] | None = None,
        chat_template: str | None = None,
        vocab_seed: int = 0,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = None
        self.unk_token_id = None
        self.name_or_path = name_or_path
        self.padding_side = "left"
        self.truncation_side = "right"
        self.model_max_length = 1024
        self.chat_template = chat_template
        self.init_kwargs = {"name_or_path": name_or_path, "vocab_size": vocab_size}
        self._chat_prefix = chat_prefix if chat_prefix is not None else [7, 8]
        self.mix_bit_pad_token_normalized_from_eos = False
        self._vocab_seed = vocab_seed
        self._vocab = {f"tok_{i + vocab_seed}": i for i in range(vocab_size)}
        self.special_tokens_map: dict = {}

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return {}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        ids = list(self._chat_prefix)
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            ids.append(1 if role == "user" else 2)
            ids.extend((ord(ch) % (self.vocab_size - 3)) + 3 for ch in content)
        return ids

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        return_attention_mask: bool = True,
    ):
        assert padding in (False, "do_not_pad")
        ids = [11]
        if add_special_tokens:
            ids.append(12)
        ids.extend((ord(ch) % (self.vocab_size - 3)) + 3 for ch in text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attention_mask = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": attention_mask}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(rec, ensure_ascii=False) for rec in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inventory(*, fingerprint: str = "inv-fp-1") -> ModelInventory:
    target = TargetLinearSpec(
        module_name="model.layers.0.self_attn.q_proj",
        category="q_proj",
        module_suffix="q_proj",
        block_index=0,
        in_features=4,
        out_features=4,
        has_bias=False,
        param_count=16,
        transpose=True,
    )
    return ModelInventory(
        model_id="toy",
        model_path="toy-model",
        transformers_model_type="toy",
        resolved_model_class="ToyLM",
        adapter_name="generic_decoder",
        model_profile_sha256="profile-sha",
        category_order=("q_proj",),
        block_count=1,
        targets=(target,),
        total_target_parameters=16,
        fingerprint_sha256=fingerprint,
    )


def _resolved(
    tmp_path: Path,
    *,
    source_jsonl: Path,
    max_samples: int = 3,
    max_length: int = 64,
    seed: int = 31,
    input_format: str = "auto",
) -> ResolvedRunConfig:
    profile = ModelProfile(
        model_id="toy",
        model_path="toy-model",
        adapter="generic_decoder",
        only_decoder_projections=True,
        candidate_training=CandidateTrainingSpec(
            linear_group_size="all",
            allow_tail_group=True,
        ),
        layer_index_patterns=(r"(?:^|\.)layers\.(\d+)\.",),
        categories=(CategorySpec("q_proj", "q_proj", True),),
        regression_expectations={},
    )
    space = CandidateSpaceConfig(
        candidate_space_id="toy_space",
        target_average_bit=2.0,
        baseline_mode="b32d32s2",
        modes=(
            CandidateMode(
                name="b32d32s2",
                nominal_bit=2.0,
                codebook_bits=32,
                codebook_dim=32,
                residual_stages=2,
            ),
        ),
    )
    run_root = tmp_path / "run"
    return ResolvedRunConfig(
        config=MixBitRunConfig(
            run_id="toy_run",
            model_profile=profile,
            candidate_space=space,
            training_recipe=TrainingRecipeConfig(recipe_id="toy_recipe", values={}),
            calibration=CalibrationConfig(
                source_jsonl=str(source_jsonl.resolve()),
                input_format=input_format,  # type: ignore[arg-type]
                max_samples=max_samples,
                max_length=max_length,
                seed=seed,
                label_mode="all_nonpad",
            ),
        ),
        run_config_path=str(tmp_path / "run.json"),
        run_config_sha256="run-sha",
        model_profile_path=str(tmp_path / "profile.json"),
        model_profile_sha256="profile-sha",
        candidate_space_path=str(tmp_path / "space.json"),
        candidate_space_sha256="space-sha",
        training_recipe_path=str(tmp_path / "recipe.json"),
        training_recipe_sha256="recipe-sha",
        canonical_model_root=str(tmp_path / "model_root"),
        canonical_run_root=str(run_root),
    )


def test_auto_schema_accepts_messages_records():
    record = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    assert resolve_record_schema(record) == "messages"


def test_auto_schema_accepts_text_records():
    assert resolve_record_schema({"text": "plain text sample"}) == "text"


def test_auto_schema_accepts_prompt_response_records():
    assert resolve_record_schema({"prompt": "q", "response": "a"}) == "prompt_response"
    assert resolve_record_schema({"instruction": "q", "output": "a"}) == "prompt_response"


def test_auto_schema_rejects_ambiguous_or_unknown_records():
    with pytest.raises(ValueError, match="ambiguous|unknown|unsupported"):
        resolve_record_schema({"messages": [{"role": "user", "content": "x"}], "text": "y"})
    with pytest.raises(ValueError, match="ambiguous|unknown|unsupported"):
        resolve_record_schema({"foo": 1})
    with pytest.raises(ValueError, match="ambiguous|unknown|unsupported|empty"):
        resolve_record_schema({"text": ""})
    with pytest.raises(ValueError, match="role|content|malformed"):
        resolve_record_schema({"messages": [{"role": "user"}]})


def test_dataset_order_is_deterministic_for_seed_31(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    records = [{"text": f"sample-{i}"} for i in range(20)]
    _write_jsonl(source, records)
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=5, seed=31)
    inventory = _inventory()
    tokenizer = FakeTokenizer()

    examples_a, _ = prepare_calibration_dataset(
        resolved,
        inventory,
        tokenizer=tokenizer,
        output_dir=tmp_path / "cal_a",
        overwrite=True,
        seqlen=128,
    )
    examples_b, _ = prepare_calibration_dataset(
        resolved,
        inventory,
        tokenizer=FakeTokenizer(),
        output_dir=tmp_path / "cal_b",
        overwrite=True,
        seqlen=128,
    )

    valid_line_indices = list(range(20))
    rng = random.Random(31)
    rng.shuffle(valid_line_indices)
    expected_ids = valid_line_indices[:5]

    assert [ex.sample_id for ex in examples_a] == expected_ids
    assert [ex.sample_id for ex in examples_b] == expected_ids


def test_dataset_keeps_exactly_max_samples(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"text": f"s{i}"} for i in range(50)])
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=7, seed=31)
    examples, manifest = prepare_calibration_dataset(
        resolved,
        _inventory(),
        tokenizer=FakeTokenizer(),
        output_dir=tmp_path / "cal",
        overwrite=True,
        seqlen=128,
    )
    assert len(examples) == 7
    assert manifest.sample_count == 7
    assert len(manifest.selected_source_line_ids) == 7


def test_sample_id_is_stable_source_line_index(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    # blank line + valid records; blank must not shift source indices
    raw_lines = [
        json.dumps({"text": "a"}),
        "",
        json.dumps({"text": "b"}),
        json.dumps({"text": "c"}),
    ]
    source.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=3, seed=0)
    examples, _ = prepare_calibration_dataset(
        resolved,
        _inventory(),
        tokenizer=FakeTokenizer(),
        output_dir=tmp_path / "cal",
        overwrite=True,
        seqlen=128,
    )
    assert {ex.sample_id for ex in examples} == {0, 2, 3}
    for ex in examples:
        assert isinstance(ex.sample_id, int)


def test_tokenizer_hash_change_invalidates_resume(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"text": f"s{i}"} for i in range(10)])
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=3, seed=31)
    out = tmp_path / "cal"
    prepare_calibration_dataset(
        resolved,
        _inventory(),
        tokenizer=FakeTokenizer(vocab_seed=0),
        output_dir=out,
        overwrite=True,
        seqlen=128,
    )
    with pytest.raises(ValueError, match="tokenizer|resume|overwrite"):
        prepare_calibration_dataset(
            resolved,
            _inventory(),
            tokenizer=FakeTokenizer(vocab_seed=1),
            output_dir=out,
            overwrite=False,
            seqlen=128,
        )


def test_causal_mask_drops_last_logit_position():
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    mask = build_causal_kl_mask(attention_mask, labels=None)
    assert mask.shape == (1, 3)
    assert torch.equal(mask, torch.tensor([[True, True, True]]))


def test_right_padding_transition_is_masked():
    # positions: tok tok tok pad
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    mask = build_causal_kl_mask(attention_mask, labels=None)
    # causal pairs (0->1), (1->2), (2->3); last pair includes pad -> False
    assert torch.equal(mask, torch.tensor([[True, True, False]]))


def test_labels_minus_100_mask_prompt_tokens():
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 5, 6]], dtype=torch.long)
    mask = build_causal_kl_mask(attention_mask, labels)
    # pairs predict positions 1,2,3 of labels → -100, 5, 6
    assert torch.equal(mask, torch.tensor([[False, True, True]]))


def test_each_retained_sample_has_at_least_one_valid_token(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"text": "hello"}, {"text": "world"}])
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=2, seed=31, max_length=64)
    examples, _ = prepare_calibration_dataset(
        resolved,
        _inventory(),
        tokenizer=FakeTokenizer(),
        output_dir=tmp_path / "cal",
        overwrite=True,
        seqlen=128,
    )
    for ex in examples:
        batched_attn = ex.attention_mask.unsqueeze(0)
        batched_labels = None if ex.labels is None else ex.labels.unsqueeze(0)
        mask = build_causal_kl_mask(batched_attn, batched_labels)
        assert int(mask.sum().item()) >= 1


def test_max_length_above_model_seqlen_fails(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"text": "hello"}, {"text": "world"}])
    # max_length 64 > model.seqlen 32 (production loaders set model.seqlen, not config max_pos)
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=2, seed=31, max_length=64)
    model = SimpleNamespace(seqlen=32)
    with pytest.raises(ValueError, match="max_length.*exceeds model seqlen"):
        prepare_calibration_dataset(
            resolved,
            _inventory(),
            tokenizer=FakeTokenizer(),
            output_dir=tmp_path / "cal",
            overwrite=True,
            model=model,
        )


# --- Tokenizer fingerprint v2 tests (Task 9 Step 1 & 2) ---

from mix_bit.calibration import (
    TOKENIZER_FINGERPRINT_VERSION,
    build_tokenizer_fingerprint_payload,
    compute_tokenizer_config_sha256,
)


def test_tokenizer_fingerprint_v2_same_content_different_path_keeps_hash():
    tok_a = FakeTokenizer(name_or_path="path-A", vocab_seed=5)
    tok_b = FakeTokenizer(name_or_path="path-B", vocab_seed=5)
    payload_a = build_tokenizer_fingerprint_payload(tok_a)
    payload_b = build_tokenizer_fingerprint_payload(tok_b)
    assert payload_a["reported_name_or_path"] == "path-A"
    assert payload_b["reported_name_or_path"] == "path-B"
    assert compute_tokenizer_config_sha256(tok_a) == compute_tokenizer_config_sha256(tok_b)


def test_tokenizer_fingerprint_v2_vocab_content_change_changes_hash():
    tok_a = FakeTokenizer(vocab_seed=0)
    tok_b = FakeTokenizer(vocab_seed=1)
    # Same vocab_size, different content.
    assert tok_a.vocab_size == tok_b.vocab_size
    assert compute_tokenizer_config_sha256(tok_a) != compute_tokenizer_config_sha256(tok_b)


def test_tokenizer_fingerprint_v2_chat_template_change_changes_hash():
    tok_a = FakeTokenizer(chat_template=None)
    tok_b = FakeTokenizer(chat_template="{% for m in messages %}{{ m.content }}{% endfor %}")
    assert compute_tokenizer_config_sha256(tok_a) != compute_tokenizer_config_sha256(tok_b)


def test_tokenizer_fingerprint_v2_added_vocab_change_changes_hash():
    tok_a = FakeTokenizer()

    class _WithAdded(FakeTokenizer):
        def get_added_vocab(self):
            return {"<extra>": 32}

    tok_b = _WithAdded()
    assert compute_tokenizer_config_sha256(tok_a) != compute_tokenizer_config_sha256(tok_b)


def test_tokenizer_fingerprint_v2_unsupported_object_does_not_leak_address():
    class _Weird:
        pass

    class _WithWeird(FakeTokenizer):
        def get_added_vocab(self):
            return {"<weird>": _Weird()}

    tok = _WithWeird()
    payload = build_tokenizer_fingerprint_payload(tok)
    serialized = json.dumps(payload["content"]["added_vocab"], sort_keys=True)
    assert "unsupported_type" in serialized
    # Two instances must produce identical payload (no address leak).
    payload_a = build_tokenizer_fingerprint_payload(_WithWeird())
    payload_b = build_tokenizer_fingerprint_payload(_WithWeird())
    assert payload_a["content"] == payload_b["content"]


def test_tokenizer_fingerprint_v2_core_change_changes_hash():
    tok_a = FakeTokenizer(vocab_seed=0)

    class _OtherVocab(FakeTokenizer):
        def get_vocab(self):
            return {f"other_{i}": i for i in range(self.vocab_size)}

    tok_b = _OtherVocab(vocab_seed=0)
    assert tok_a.vocab_size == tok_b.vocab_size
    assert compute_tokenizer_config_sha256(tok_a) != compute_tokenizer_config_sha256(tok_b)


class _FakeBackend:
    """Mimics HF `backend_tokenizer.to_str()` returning a stable JSON string."""

    def __init__(self, payload: str):
        self._payload = payload

    def to_str(self) -> str:
        return self._payload


class _BackendTokenizer(FakeTokenizer):
    """FakeTokenizer that exposes `backend_tokenizer.to_str()` (production fast path)."""

    def __init__(self, *, backend_payload: str, **kwargs):
        super().__init__(**kwargs)
        self.backend_tokenizer = _FakeBackend(backend_payload)


def test_tokenizer_fingerprint_v2_backend_tokenizer_json_path_is_used_and_stable():
    backend_a = '{"model":{"type":"BPE","vocab":{"a":0,"b":1}}}'
    backend_b = '{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2}}}'
    tok_a = _BackendTokenizer(backend_payload=backend_a)
    tok_b = _BackendTokenizer(backend_payload=backend_a)
    payload_a = build_tokenizer_fingerprint_payload(tok_a)
    payload_b = build_tokenizer_fingerprint_payload(tok_b)
    assert payload_a["content"]["core_kind"] == "backend_tokenizer_json"
    assert payload_b["content"]["core_kind"] == "backend_tokenizer_json"
    # identical backend JSON -> identical hash
    assert compute_tokenizer_config_sha256(tok_a) == compute_tokenizer_config_sha256(tok_b)
    # different backend JSON -> different hash
    tok_c = _BackendTokenizer(backend_payload=backend_b)
    assert compute_tokenizer_config_sha256(tok_a) != compute_tokenizer_config_sha256(tok_c)


def test_tokenizer_fingerprint_v2_backend_path_wins_over_get_vocab():
    backend_payload = '{"model":{"type":"BPE","vocab":{"a":0}}}'
    # Two tokenizers with the SAME backend JSON but DIFFERENT get_vocab content:
    # the backend path must win, so hashes are equal.
    tok_a = _BackendTokenizer(backend_payload=backend_payload, vocab_seed=0)
    tok_b = _BackendTokenizer(backend_payload=backend_payload, vocab_seed=99)
    assert tok_a.get_vocab() != tok_b.get_vocab()
    payload_a = build_tokenizer_fingerprint_payload(tok_a)
    payload_b = build_tokenizer_fingerprint_payload(tok_b)
    assert payload_a["content"]["core_kind"] == "backend_tokenizer_json"
    assert payload_b["content"]["core_kind"] == "backend_tokenizer_json"
    assert compute_tokenizer_config_sha256(tok_a) == compute_tokenizer_config_sha256(tok_b)


def test_tokenizer_fingerprint_v2_empty_extra_special_tokens_matches_missing():
    """HF save_pretrained injects extra_special_tokens={} into tokenizer_config.json.

    Hub configs often omit the key; after local reload it appears as {}. That must
    not change the content fingerprint. Non-empty mappings remain content.
    """
    tok_missing = FakeTokenizer()
    tok_empty = FakeTokenizer()
    tok_empty.init_kwargs = dict(tok_missing.init_kwargs)
    tok_empty.init_kwargs["extra_special_tokens"] = {}
    assert "extra_special_tokens" not in tok_missing.init_kwargs
    assert compute_tokenizer_config_sha256(tok_missing) == compute_tokenizer_config_sha256(
        tok_empty
    )
    payload_empty = build_tokenizer_fingerprint_payload(tok_empty)
    assert "extra_special_tokens" not in payload_empty["content"]["stable_init_kwargs"]

    tok_nonempty = FakeTokenizer()
    tok_nonempty.init_kwargs = dict(tok_missing.init_kwargs)
    tok_nonempty.init_kwargs["extra_special_tokens"] = {"image": "<image>"}
    assert compute_tokenizer_config_sha256(tok_missing) != compute_tokenizer_config_sha256(
        tok_nonempty
    )


def test_calibration_manifest_records_tokenizer_fingerprint_v2(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"text": f"s{i}"} for i in range(10)])
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=3, seed=31)
    out = tmp_path / "cal"
    _, manifest = prepare_calibration_dataset(
        resolved,
        _inventory(),
        tokenizer=FakeTokenizer(),
        output_dir=out,
        overwrite=True,
        seqlen=128,
    )
    assert manifest.tokenizer_fingerprint_version == TOKENIZER_FINGERPRINT_VERSION
    manifest_path = out / "dataset_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["tokenizer_fingerprint_version"] == TOKENIZER_FINGERPRINT_VERSION


def test_calibration_resume_rejects_legacy_tokenizer_fingerprint(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"text": f"s{i}"} for i in range(10)])
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=3, seed=31)
    out = tmp_path / "cal"
    prepare_calibration_dataset(
        resolved,
        _inventory(),
        tokenizer=FakeTokenizer(),
        output_dir=out,
        overwrite=True,
        seqlen=128,
    )
    manifest_path = out / "dataset_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    del raw["tokenizer_fingerprint_version"]
    manifest_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy|fingerprint_version|overwrite"):
        prepare_calibration_dataset(
            resolved,
            _inventory(),
            tokenizer=FakeTokenizer(),
            output_dir=out,
            overwrite=False,
            seqlen=128,
        )


def test_calibration_resume_rejects_same_vocab_size_changed_core(tmp_path: Path):
    source = tmp_path / "src.jsonl"
    _write_jsonl(source, [{"text": f"s{i}"} for i in range(10)])
    resolved = _resolved(tmp_path, source_jsonl=source, max_samples=3, seed=31)
    out = tmp_path / "cal"
    prepare_calibration_dataset(
        resolved,
        _inventory(),
        tokenizer=FakeTokenizer(vocab_seed=0),
        output_dir=out,
        overwrite=True,
        seqlen=128,
    )
    with pytest.raises(ValueError, match="tokenizer_config_sha256|overwrite"):
        prepare_calibration_dataset(
            resolved,
            _inventory(),
            tokenizer=FakeTokenizer(vocab_seed=7),
            output_dir=out,
            overwrite=False,
            seqlen=128,
        )
