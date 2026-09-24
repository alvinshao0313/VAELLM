"""Preserve every trained VAE before deployment export can fail.

This is an end-of-training checkpoint, not a calibration-stream resume format.
Source Linear weights are verified by hash and loaded from the original model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .artifacts import dump_json, load_linear
from .incremental import LinearTrainer


FINAL_STATE_NAME = "final_training_state.pt"


def save_final_training_state(trainers, args, output: Path) -> Path:
    """Atomically save all trainer states before calling any trainer.export()."""
    names = list(trainers)
    states = {name: trainer.state_dict() for name, trainer in trainers.items()}
    for name, state in states.items():
        if state["update_count"] != args.steps:
            raise ValueError(f"Incomplete training state: {name}")
    checkpoint = {
        "format": "linear_output_final_training_state",
        "version": 1,
        "config": vars(args).copy(),
        "modules": names,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "trainers": states,
    }
    path = Path(output) / FINAL_STATE_NAME
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def load_final_training_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "linear_output_final_training_state" or checkpoint.get("version") != 1:
        raise ValueError(f"Unsupported final training checkpoint: {path}")
    names = checkpoint["modules"]
    if len(names) != len(set(names)) or set(names) != set(checkpoint["trainers"]):
        raise ValueError("Final training checkpoint has inconsistent module names")
    for name in names:
        state = checkpoint["trainers"][name]
        if state["name"] != name or state["update_count"] != checkpoint["config"]["steps"]:
            raise ValueError(f"Incomplete or inconsistent final trainer state: {name}")
    return checkpoint


def export_trainers(trainers, manifest: dict, output: Path) -> dict:
    """Persist each success and a clear failure status; never hide export errors."""
    manifest["status"] = "EXPORTING"
    manifest.pop("error", None)
    manifest.pop("failed_module", None)
    manifest.pop("failed_stage", None)
    dump_json(output / "manifest.json", manifest)
    completed = {record["module"] for record in manifest["records"]}
    current = None
    try:
        for current, trainer in trainers:
            if current in completed:
                continue
            manifest["active_module"] = current
            record = trainer.export()
            manifest["records"].append(record)
            completed.add(current)
            dump_json(output / "manifest.json", manifest)
        if completed != set(manifest["modules"]):
            raise RuntimeError("Export finished without all target Linear records")
    except Exception as error:
        manifest.update(status="FAILED", failed_module=manifest.get("active_module", current),
                        error=f"{type(error).__name__}: {error}")
        dump_json(output / "manifest.json", manifest)
        raise
    manifest["status"] = "COMPLETE"
    manifest.pop("active_module", None)
    dump_json(output / "manifest.json", manifest)
    return manifest


def finalize_training(trainers, args, manifest: dict, output: Path) -> dict:
    """Distinguish completed training, checkpoint saving, and deployment export."""
    manifest["status"] = "SAVING_FINAL_STATE"
    dump_json(output / "manifest.json", manifest)
    try:
        state_path = save_final_training_state(trainers, args, output)
    except Exception as error:
        manifest.update(status="FAILED", failed_stage="final_state_save",
                        error=f"{type(error).__name__}: {error}")
        dump_json(output / "manifest.json", manifest)
        raise
    manifest["final_training_state"] = str(state_path.resolve())
    return export_trainers(trainers.items(), manifest, output)


def export_saved_state(model, checkpoint: dict, output: Path, device: str) -> dict:
    """Export saved final VAEs without data, teacher forward, or optimizer steps."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(**checkpoint["config"])
    args.output_dir = str(output)
    args.device = device
    names = checkpoint["modules"]
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["modules"] != names or manifest.get("config") != checkpoint["config"]:
            raise ValueError("Output manifest does not belong to this saved training run")
    else:
        manifest = {
            "algorithm": "independent_linear_output_shared_teacher_sweep",
            "objective": args.objective,
            "modules": names,
            "steps": args.steps,
            "codebook_bits": args.codebook_bits,
            "codebook_dim": args.codebook_dim,
            "config": checkpoint["config"],
            "records": [],
        }
    completed = {record["module"] for record in manifest["records"]}
    manifest["status"] = "VALIDATING_EXPORTS"
    manifest.pop("active_module", None)
    dump_json(manifest_path, manifest)
    try:
        if len(completed) != len(manifest["records"]) or not completed <= set(names):
            raise ValueError("Output manifest has duplicate or unknown Linear records")
        for record in manifest["records"]:
            name = record["module"]
            manifest["active_module"] = name
            if record["source_weight_sha256"] != checkpoint["trainers"][name]["source_weight_sha256"]:
                raise ValueError(f"Saved export source hash mismatch: {name}")
            record_path = output / "linears" / name / "record.json"
            if not record_path.is_file():
                raise ValueError(f"Manifest claims success but record is absent: {name}")
            if json.loads(record_path.read_text(encoding="utf-8")) != record:
                raise ValueError(f"Persisted export record differs from manifest: {name}")
            # A directory alone is not evidence of a complete, loadable checkpoint.
            restored = load_linear(name, model.get_submodule(name), output / "linears" / name / "packed")
            del restored
    except Exception as error:
        manifest.update(status="FAILED", failed_module=manifest.get("active_module"),
                        error=f"{type(error).__name__}: {error}")
        dump_json(manifest_path, manifest)
        raise
    manifest.pop("active_module", None)

    def remaining_trainers():
        for name in names:
            if name in completed:
                continue
            manifest["active_module"] = name
            trainer = LinearTrainer(model.get_submodule(name), name, args, output / "linears" / name)
            trainer.load_state_dict(checkpoint["trainers"][name])
            trainer.initial_state_sha256 = checkpoint["trainers"][name]["initial_state_sha256"]
            yield name, trainer
            del trainer

    return export_trainers(remaining_trainers(), manifest, output)
