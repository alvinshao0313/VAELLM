"""Run the fixed two-model, two-matrix block32/full-rotation isolated screen."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch

from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.capture import FORMAT, tensor_record
from experiments.rotation_factor_probe.factors import load_factors
from experiments.rotation_factor_probe.metrics import measure, paired_comparison
from experiments.rotation_factor_probe.train import run_matrix
from train_utils.utils import configure_deterministic_mode, set_seed


def load_archive(directory: Path):
    manifest = json.loads((directory / "data_manifest.json").read_text())
    if manifest["status"] != "captured" or manifest["format"] != FORMAT:
        raise ValueError("A completed real-model capture is required.")
    if sha256(directory / "inputs.pt") != manifest["inputs_sha256"]:
        raise ValueError("Captured input archive changed.")
    archive = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True, mmap=True)
    for key in ("model_type", "model_path", "checkpoint_id", "document_lengths"):
        if archive[key] != manifest[key]:
            raise ValueError(f"Capture identity changed: {key}.")
    for name, bundle in archive["targets"].items():
        for key, value in bundle.items():
            if tensor_record(value) != manifest["tensors"][name][key]:
                raise ValueError(f"Captured tensor changed: {name}/{key}.")
    if tensor_record(archive["up_proj_weight"]) != manifest["up_proj_weight"]:
        raise ValueError("Captured original up-projection changed.")
    return archive, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--factor-source", type=Path, required=True)
    parser.add_argument("--native-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Screen output must be a new directory.")
    verification = json.loads(args.native_verification.read_text())
    if verification["status"] != "verified":
        raise ValueError("The native block32 equivalence verification has not passed.")
    check_hashes(verification["source_sha256"])
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one assigned CUDA GPU must be visible.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    factors = load_factors(args.factor_source)
    root = Path(__file__).resolve().parents[2]
    sources = {str(p): sha256(p) for folder in ("train_utils", "litebsq", "rotation",
        "experiments/rotation_factor_probe") for p in sorted((root / folder).rglob("*.py"))}
    inputs = {str(args.factor_source.resolve()): sha256(args.factor_source),
              str(args.native_verification.resolve()): sha256(args.native_verification)}
    for model in ("qwen", "llama"):
        for filename in ("data_manifest.json", "inputs.pt"):
            p = (args.data_dir / model / filename).resolve()
            inputs[str(p)] = sha256(p)
    record = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "device": torch.cuda.get_device_name(device), "started_unix": time.time(),
              "source_sha256": sources, "input_sha256": inputs, "steps_per_stage": 2000,
              "seed": 31, "arms": {}, "comparisons": {},
              "boundaries": ["Single L0 matrix VAE, not the formal parallel category entropy population.",
                  "Each stage uses a full 2000-step linear schedule, not a prefix of 10000 steps.",
                  "Both activation splits are measurement-only; pure weight MSE training.",
                  "Fixed original HF weights and checkpoint protection indices; no trained checkpoint export.",
                  "No downstream accuracy or whole-model bitrate acceptance is inferred."]}
    output.mkdir(parents=True)
    write_json(output / "screen.json", record)
    try:
        for model in ("qwen", "llama"):
            archive, manifest = load_archive(args.data_dir / model)
            for category in ("gate_proj", "down_proj"):
                name = "model.layers.0.mlp." + category
                pair = {}
                for full in (False, True):
                    arm = f"{model}/{category}/{'full' if full else 'block32'}"
                    check_hashes(sources)
                    torch.cuda.reset_peak_memory_stats(device)
                    logging.info("Starting %s", arm)
                    reconstruction, report = run_matrix(archive["targets"][name], name=name,
                        model_path=archive["model_path"], full=full, factors=factors,
                        steps=2000, seed=31, device=device)
                    report["metrics"] = measure(archive, name, reconstruction, device)
                    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                    report["checkpoint_indices_source"] = archive["checkpoint_id"]
                    report["capture_sha256"] = manifest["inputs_sha256"]
                    report["status"] = "completed"
                    record["arms"][arm] = report
                    pair["full" if full else "block32"] = report
                    write_json(output / "screen.json", record)
                    del reconstruction
                    gc.collect()
                    torch.cuda.empty_cache()
                for key in ("main_code_bits", "unprotected_module_bits", "protection_bits", "rotation_sign_bits"):
                    if pair["full"][key] != pair["block32"][key]:
                        raise ValueError(f"Matched rate changed: {model}/{category}/{key}.")
                record["comparisons"][f"{model}/{category}"] = paired_comparison(
                    pair["block32"]["metrics"], pair["full"]["metrics"])
                write_json(output / "screen.json", record)
            del archive
        check_hashes(sources)
        check_hashes(inputs)
        record.update(status="completed", finished_unix=time.time())
        write_json(output / "screen.json", record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        write_json(output / "screen.json", record)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
