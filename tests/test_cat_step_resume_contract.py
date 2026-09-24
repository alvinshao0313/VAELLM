from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from train_utils.cat_step_resume_v6 import (
    build_cat_step_immutable_resume_contract,
    build_distill_dataset_identity,
    model_identity,
    prune_completed_cat_round_roots,
    validate_cat_step_immutable_resume_contract,
)

from train_utils.config.configs import AuxTrainableConfig


class _FakeDataset:
    def __init__(self, fingerprint: str, length: int = 3, *, raw_dataset=None) -> None:
        self._fingerprint = str(fingerprint)
        self._length = int(length)
        self.raw_dataset = raw_dataset

    def __len__(self) -> int:
        return self._length


class _TinyTeacher(nn.Module):
    def __init__(self, *, commit_hash: str | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.config = SimpleNamespace(
            architectures=["TinyTeacher"],
            model_type="tiny",
            hidden_size=2,
            num_hidden_layers=1,
            vocab_size=8,
            _commit_hash=commit_hash,
        )


def _bundle(*, source_path: str, fingerprint: str):
    return SimpleNamespace(
        dataset_mix_spec=None,
        cache_key=("train_file", source_path),
        source_stats=[{"alias": "train_file", "path": source_path, "weight": 1.0}],
        train_dataset=_FakeDataset(fingerprint),
    )


def test_dataset_identity_changes_when_same_local_path_is_replaced(tmp_path: Path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text":"aaaa"}\n', encoding="utf-8")
    first = build_distill_dataset_identity(_bundle(source_path=str(data_path), fingerprint="same-fp"))

    data_path.write_text('{"text":"bbbb"}\n', encoding="utf-8")
    second = build_distill_dataset_identity(_bundle(source_path=str(data_path), fingerprint="same-fp"))

    assert first != second
    first_manifest = first["source_stats"][0]["local_manifest"]
    second_manifest = second["source_stats"][0]["local_manifest"]
    assert first_manifest["root"] == second_manifest["root"] == str(data_path.resolve())
    assert first_manifest["entries"] != second_manifest["entries"]


def test_dataset_identity_tracks_hf_style_fingerprint_without_local_path():
    first = build_distill_dataset_identity(_bundle(source_path="org/remote-dataset", fingerprint="fp-a"))
    second = build_distill_dataset_identity(_bundle(source_path="org/remote-dataset", fingerprint="fp-b"))
    assert first != second
    assert first["train_dataset"]["fingerprint"] == "fp-a"
    assert second["train_dataset"]["fingerprint"] == "fp-b"


def test_teacher_identity_changes_when_same_local_checkpoint_is_replaced(tmp_path: Path):
    model_dir = tmp_path / "teacher"
    model_dir.mkdir()
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"aaaa")
    teacher = _TinyTeacher()
    first = model_identity(teacher, str(model_dir))

    weight_path.write_bytes(b"bbbb")
    second = model_identity(teacher, str(model_dir))

    assert first != second
    assert first["model_path"] == second["model_path"] == str(model_dir)
    assert first["local_manifest"] != second["local_manifest"]


def test_teacher_identity_carries_hf_commit_hash():
    identity = model_identity(_TinyTeacher(commit_hash="abc123"), "org/teacher")
    assert identity["revision_hint"] == "abc123"
    assert identity["local_manifest"] is None


def test_completed_round_retention_preserves_unlimited_and_prunes_oldest(tmp_path: Path):
    rounds = tmp_path / "training_rounds"
    for name in ("0000_q_proj", "0001_k_proj", "0002_v_proj"):
        root = rounds / name
        root.mkdir(parents=True)
        (root / "round_base.marker").write_text(name, encoding="utf-8")

    assert prune_completed_cat_round_roots(str(tmp_path), save_total_limit=None) == ()
    assert all((rounds / name).is_dir() for name in ("0000_q_proj", "0001_k_proj", "0002_v_proj"))

    removed = prune_completed_cat_round_roots(str(tmp_path), save_total_limit=2)
    assert removed == (str((rounds / "0000_q_proj").resolve()),)
    assert not (rounds / "0000_q_proj").exists()
    assert (rounds / "0001_k_proj").is_dir()
    assert (rounds / "0002_v_proj").is_dir()


def _resume_contract_for_aux(aux):
    stage = SimpleNamespace(
        mode="remaining_lora",
        config=SimpleNamespace(
            data={},
            loss={},
            opt=SimpleNamespace(logging_steps=1, distill_fp32_components=()),
            aux=aux,
            runtime=SimpleNamespace(parallel_mode="dp", layer_device_map="auto"),
        ),
    )
    return build_cat_step_immutable_resume_contract(
        stage=stage,
        trainer_args=SimpleNamespace(),
        tokenizer=SimpleNamespace(name_or_path="tiny-tokenizer"),
        round_base_checkpoint_id="tiny-round-base",
        active_category="down_proj",
        round_base_meta={},
        lora_target_names=(),
        decoder_target_names=(),
        teacher_identity=None,
        dataset_identity={},
        lora_config=None,
    )


def test_disabled_residual_lora_keeps_legacy_cat_resume_contract():
    legacy = _resume_contract_for_aux(
        SimpleNamespace(norm_train_mode="none", norm_lr=None, lm_head_train_mode="none", lm_head_lr=None)
    )
    current = _resume_contract_for_aux(AuxTrainableConfig())
    validate_cat_step_immutable_resume_contract(legacy, current)
    assert current["aux"] == legacy["aux"]
    inactive = _resume_contract_for_aux(
        AuxTrainableConfig(
            residual_lora_mode="none",
            residual_lora_rank=4,
            residual_lora_alpha=8,
            residual_lora_dropout=0.1,
            residual_lora_lr=3e-5,
        )
    )
    validate_cat_step_immutable_resume_contract(legacy, inactive)


def test_enabled_residual_lora_is_recorded_in_cat_resume_contract():
    contract = _resume_contract_for_aux(
        AuxTrainableConfig(residual_lora_mode="additive", residual_lora_lr=3e-5)
    )
    assert contract["aux"]["residual_lora_mode"] == "additive"
    assert contract["aux"]["residual_lora_rank"] == 8
    assert contract["aux"]["residual_lora_alpha"] == 16
    assert contract["aux"]["residual_lora_dropout"] == 0
    assert contract["aux"]["residual_lora_lr"] == 3e-5


@pytest.mark.parametrize(
    "change",
    [
        {"residual_lora_mode": "none"},
        {"residual_lora_mode": "replace"},
        {"residual_lora_rank": 4},
        {"residual_lora_alpha": 8},
        {"residual_lora_dropout": 0.1},
        {"residual_lora_lr": 7e-5},
    ],
)
def test_cat_resume_rejects_residual_lora_contract_changes(change):
    aux = AuxTrainableConfig(residual_lora_mode="additive", residual_lora_lr=3e-5)
    saved = _resume_contract_for_aux(aux)
    current = _resume_contract_for_aux(replace(aux, **change))
    with pytest.raises(ValueError, match="immutable"):
        validate_cat_step_immutable_resume_contract(saved, current)
