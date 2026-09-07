import argparse
import json
from dataclasses import dataclass

import pytest

import train_utils.cat_residual_from_base as residual_from_base
from compressed_e2e_fintuning.runtime_v6 import (
    _build_e2e_runtime_parameter_payload,
    _save_normalized_e2e_runtime_snapshot,
)
from train_utils.cat_train_runtime import (
    build_cat_runtime_parameter_payload,
    save_normalized_cat_train_snapshot,
)
from train_utils.runtime_snapshot import to_runtime_jsonable


@dataclass
class _NestedConfig:
    dataset_mix: str
    loss_type: str
    learning_rate: float
    access_token: str = "dummy"


@dataclass
class _CanonicalConfig:
    data: _NestedConfig
    train_mode: str


def test_runtime_jsonable_omits_credentials_recursively():
    payload = {
        "access_token": "dummy",
        "nested": {
            "api_key": "dummy",
            "hub_token": "dummy",
            "pad_token_id": 151643,
        },
    }

    normalized = to_runtime_jsonable(payload)

    assert "access_token" not in normalized
    assert "api_key" not in normalized["nested"]
    assert "hub_token" not in normalized["nested"]
    assert normalized["nested"]["pad_token_id"] == 151643


def test_cat_snapshot_keeps_canonical_config_and_drops_adapter_helpers(tmp_path):
    canonical = _CanonicalConfig(
        data=_NestedConfig(
            dataset_mix="edgerazor_ii_7m=1",
            loss_type="kl_top",
            learning_rate=1e-4,
        ),
        train_mode="remaining_lora_prefix_decoder",
    )
    cat_args = argparse.Namespace(
        _common_cat_config=canonical,
        resolve_after_category_config=lambda *_args, **_kwargs: None,
        output_dir=str(tmp_path),
        seed=31,
    )
    vae_args = argparse.Namespace(model_path="Qwen/Qwen3-8B", access_token="dummy")
    training_args = argparse.Namespace(bf16=True, save_steps=500)
    resolved = {
        "q_proj": argparse.Namespace(
            category="q_proj",
            codebook_bits=32,
            codebook_dim=32,
            residual_stages=2,
        )
    }

    payload = build_cat_runtime_parameter_payload(
        cat_args=cat_args,
        vae_args=vae_args,
        training_args=training_args,
        resolved_category_cfgs=resolved,
    )

    assert payload["canonical_config"]["data"]["dataset_mix"] == "edgerazor_ii_7m=1"
    assert payload["canonical_config"]["data"]["loss_type"] == "kl_top"
    assert payload["canonical_config"]["data"]["learning_rate"] == 1e-4
    assert "access_token" not in payload["canonical_config"]["data"]
    assert "_common_cat_config" not in payload["cat_args"]
    assert "resolve_after_category_config" not in payload["cat_args"]
    assert "access_token" not in payload["vae_args"]
    assert payload["resolved_category_runtime"]["q_proj"]["codebook_bits"] == 32

    snapshot_path = save_normalized_cat_train_snapshot(
        run_output_dir=str(tmp_path),
        cat_args=cat_args,
        vae_args=vae_args,
        training_args=training_args,
        resolved_category_cfgs=resolved,
    )
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["canonical_config"] == payload["canonical_config"]


def test_e2e_snapshot_contains_canonical_and_resolved_runtime_without_credentials(tmp_path):
    cfg = _CanonicalConfig(
        data=_NestedConfig(
            dataset_mix="edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133",
            loss_type="kl_top",
            learning_rate=1e-5,
        ),
        train_mode="decoder_sparse_bit",
    )
    hf_args = argparse.Namespace(access_token="dummy")
    training_args = argparse.Namespace(
        output_dir=str(tmp_path / "trainer_state"),
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        max_steps=5000,
    )

    payload = _build_e2e_runtime_parameter_payload(
        cfg=cfg,
        hf_args=hf_args,
        training_args=training_args,
        run_output_dir=str(tmp_path),
    )

    assert payload["canonical_config"]["train_mode"] == "decoder_sparse_bit"
    assert payload["canonical_config"]["data"]["dataset_mix"].startswith("edgerazor_ii_7m")
    assert payload["training_args"]["max_steps"] == 5000
    assert "access_token" not in payload["hf_args"]
    assert payload["resolved_runtime"]["run_output_dir"] == str(tmp_path.resolve())

    snapshot_path = _save_normalized_e2e_runtime_snapshot(
        cfg=cfg,
        hf_args=hf_args,
        training_args=training_args,
        run_output_dir=str(tmp_path),
    )
    assert snapshot_path.endswith("normalized_e2e_runtime_args.json")
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved == payload


def test_residual_from_base_writes_config_before_checkpoint_load(tmp_path, monkeypatch):
    output_root = tmp_path / "runs"
    args = argparse.Namespace(
        deterministic=False,
        seed=7,
        output_dir=str(output_root),
        overwrite=False,
        model_path="dummy/model",
        target_categories="q_proj",
        transpose_modules="q_proj",
        outlier_protect_mode="none",
        outlier_rank_metric="channel_weight_abs",
        eval_tasks="",
        eval_ppl=False,
        base_vae_checkpoint=str(tmp_path / "missing-checkpoint"),
        access_token="dummy",
    )

    def _stop_before_checkpoint_load(_path):
        raise RuntimeError("stop-after-snapshot")

    monkeypatch.setattr(residual_from_base, "resolve_v6_checkpoint_dir", _stop_before_checkpoint_load)

    with pytest.raises(RuntimeError, match="stop-after-snapshot"):
        residual_from_base.run_residual_from_base(args)

    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    config_path = run_dir / "config.json"
    log_path = run_dir / "residual_from_base.log"
    assert config_path.is_file()
    assert log_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["model_path"] == "dummy/model"
    assert config["target_categories"] == "q_proj"
    assert "access_token" not in config
    assert "Runtime parameters:" in log_path.read_text(encoding="utf-8")
