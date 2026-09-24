"""Real tiny VAEs exercise export failure and recovery without a teacher forward."""
import json

import pytest
import torch
from torch import nn

from experiments.linear_output.config import parser
from experiments.linear_output.final_state import (
    export_saved_state,
    export_trainers,
    finalize_training,
    load_final_training_state,
    save_final_training_state,
)
from experiments.linear_output.incremental import LinearTrainer


def _args(output):
    args = parser().parse_args([
        "--output_dir", str(output), "--device", "cpu", "--steps", "1",
        "--objective", "weight_mse", "--vae_autocast_dtype", "fp32",
        "--base_ch", "8", "--decoder_base_ch", "8",
        "--decoder_num_res_blocks", "0", "--vae_chunk_vectors", "64",
    ])
    return args


class ExportFailureTrainer(LinearTrainer):
    def export(self):
        raise RuntimeError("intentional export failure after training")


def test_final_state_survives_export_failure_and_exports_only_remaining(tmp_path):
    torch.manual_seed(38)
    # ModuleDict has no forward implementation; recovery can only read weights.
    source = nn.ModuleDict({name: nn.Linear(32, 2, bias=False)
                           for name in ("first", "second", "third")})
    args = _args(tmp_path)
    trainers = {}
    for name, linear in source.items():
        cls = ExportFailureTrainer if name == "second" else LinearTrainer
        trainer = cls(linear, name, args, tmp_path / "linears" / name)
        trainer.step(torch.randn(3, 32), 1)
        trainers[name] = trainer
    saved = save_final_training_state(trainers, args, tmp_path)
    manifest = {"modules": list(trainers), "config": vars(args), "records": []}
    with pytest.raises(RuntimeError, match="intentional export failure"):
        export_trainers(trainers.items(), manifest, tmp_path)

    failed = json.loads((tmp_path / "manifest.json").read_text())
    assert failed["status"] == "FAILED"
    assert failed["failed_module"] == "second"
    assert [record["module"] for record in failed["records"]] == ["first"]
    state = load_final_training_state(saved)
    assert state["modules"] == list(trainers)
    assert state["seed"] == args.seed and state["data_seed"] == args.data_seed
    for name, trainer in trainers.items():
        retained = state["trainers"][name]
        assert retained["update_count"] == 1
        assert retained["source_weight_sha256"] == trainer.source_weight_sha256
        assert retained["optimizer"]["state"]
        assert retained["scheduler"]
        assert "original" not in retained and "blocks" not in retained
        for key, value in trainer.vae.state_dict().items():
            assert state["trainers"][name]["vae"][key].device.type == "cpu"
            torch.testing.assert_close(retained["vae"][key], value)
    first_record = tmp_path / "linears" / "first" / "record.json"
    first_timestamp = first_record.stat().st_mtime_ns
    result = export_saved_state(source, state, tmp_path, "cpu")
    assert result["status"] == "COMPLETE"
    assert [record["module"] for record in result["records"]] == list(trainers)
    assert first_record.stat().st_mtime_ns == first_timestamp
    for record in result["records"]:
        assert record["initial_state_sha256"] == state["trainers"][record["module"]]["initial_state_sha256"]
    for name in trainers:
        assert len((tmp_path / "linears" / name / "training.jsonl").read_text().splitlines()) == 1
    assert not saved.with_suffix(".pt.tmp").exists()

    # A directory and success record must not hide damaged checkpoint contents.
    checkpoint_bytes = saved.read_bytes()
    (tmp_path / "linears" / "first" / "packed" / "checkpoint_meta.json").write_text("{ broken")
    with pytest.raises(ValueError):
        export_saved_state(source, state, tmp_path, "cpu")
    corrupt = json.loads((tmp_path / "manifest.json").read_text())
    assert corrupt["status"] == "FAILED" and corrupt["failed_module"] == "first"
    assert saved.read_bytes() == checkpoint_bytes


def test_incomplete_trainer_cannot_replace_final_checkpoint(tmp_path):
    args = _args(tmp_path)
    trainer = LinearTrainer(nn.Linear(32, 2, bias=False), "toy", args, tmp_path / "linears" / "toy")
    trainer.step(torch.randn(3, 32), 1)
    path = save_final_training_state({"toy": trainer}, args, tmp_path)
    original = path.read_bytes()
    trainer.update_count = 0
    with pytest.raises(ValueError, match="Incomplete training state"):
        save_final_training_state({"toy": trainer}, args, tmp_path)
    assert path.read_bytes() == original


def test_final_checkpoint_write_error_is_failed_not_training(tmp_path):
    args = _args(tmp_path)
    trainer = LinearTrainer(nn.Linear(32, 2, bias=False), "toy", args, tmp_path / "linears" / "toy")
    trainer.step(torch.randn(3, 32), 1)
    # A real filesystem write error, without replacing checkpoint I/O in tests.
    (tmp_path / "final_training_state.pt.tmp").mkdir()
    manifest = {"modules": ["toy"], "config": vars(args), "records": [], "status": "TRAINING"}
    with pytest.raises(IsADirectoryError):
        finalize_training({"toy": trainer}, args, manifest, tmp_path)
    failed = json.loads((tmp_path / "manifest.json").read_text())
    assert failed["status"] == "FAILED" and failed["failed_stage"] == "final_state_save"
    assert not (tmp_path / "linears" / "toy" / "packed").exists()


def test_export_reconstruction_error_names_actual_failed_module(tmp_path):
    args = _args(tmp_path)
    source = nn.ModuleDict({name: nn.Linear(32, 2, bias=False) for name in ("first", "second")})
    trainers = {}
    for name, linear in source.items():
        trainer = LinearTrainer(linear, name, args, tmp_path / "linears" / name)
        trainer.step(torch.randn(3, 32), 1)
        trainers[name] = trainer
    path = save_final_training_state(trainers, args, tmp_path)
    state = load_final_training_state(path)
    with torch.no_grad():
        source["second"].weight.add_(1)
    with pytest.raises(ValueError, match="source weight hash"):
        export_saved_state(source, state, tmp_path, "cpu")
    failed = json.loads((tmp_path / "manifest.json").read_text())
    assert failed["status"] == "FAILED" and failed["failed_module"] == "second"
    assert [record["module"] for record in failed["records"]] == ["first"]
