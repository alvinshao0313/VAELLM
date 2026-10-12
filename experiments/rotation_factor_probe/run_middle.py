"""Run the fixed middle-layer counterpart of the existing L0 rotation screen."""
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
from experiments.rotation_factor_probe.factors import load_factors
from experiments.rotation_factor_probe.metrics import measure, paired_comparison
from experiments.rotation_factor_probe.run import load_archive
from experiments.rotation_factor_probe.train import run_matrix
from train_utils.utils import configure_deterministic_mode, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--factor-source", type=Path, required=True)
    parser.add_argument("--native-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Middle screen output must be a new directory.")
    verification = json.loads(args.native_verification.read_text())
    if verification["status"] != "verified":
        raise ValueError("The existing native block32 verification is required.")
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
    sources = {str(path): sha256(path) for folder in ("train_utils", "litebsq", "rotation",
        "experiments/rotation_factor_probe") for path in sorted((root / folder).rglob("*.py"))}
    inputs = {str(args.factor_source.resolve()): sha256(args.factor_source),
              str(args.native_verification.resolve()): sha256(args.native_verification)}
    for model in ("qwen", "llama"):
        for filename in ("data_manifest.json", "inputs.pt"):
            path = (args.data_dir / model / filename).resolve()
            inputs[str(path)] = sha256(path)
    record = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(device),
        "started_unix": time.time(), "source_sha256": sources, "input_sha256": inputs,
        "steps_per_stage": 2000, "seed": 31, "arms": {}, "comparisons": {},
        "boundaries": ["Qwen L18 and Llama L16: one preselected middle layer per model, num_hidden_layers // 2.",
            "The unchanged recipe parses --target_layers 0; this direct group API selects the actual layer through explicit L18/L16 LinearRef names, not that CLI selector.",
            "Exactly the L0 32+32 documents; original HF middle activations, no compressed-prefix rollout.",
            "Single matrix entropy population and full 2000-step schedule per stage; not formal category training.",
            "Both splits are measurement-only; fixed original HF weights and checkpoint protection indices.",
            "No downstream accuracy, model export or whole-model bitrate acceptance is inferred."]}
    output.mkdir(parents=True)
    write_json(output / "screen.json", record)
    try:
        for model, expected_layer in (("qwen", 18), ("llama", 16)):
            archive, manifest = load_archive(args.data_dir / model)
            layer = manifest["layer_index"]
            if layer != expected_layer or archive["layer_index"] != layer or layer != manifest["original_config"]["num_hidden_layers"] // 2:
                raise ValueError("The preselected middle layer changed.")
            targets = [f"model.layers.{layer}.mlp.{category}" for category in ("gate_proj", "down_proj")]
            if set(archive["targets"]) != set(targets) or manifest["targets"] != targets:
                raise ValueError("Middle capture target set changed.")
            check_hashes({manifest["reference_l0_manifest"]["path"]: manifest["reference_l0_manifest"]["sha256"]})
            for category, name in zip(("gate_proj", "down_proj"), targets, strict=True):
                pair = {}
                for full in (False, True):
                    arm = f"{model}/layer{layer}/{category}/{'full' if full else 'block32'}"
                    check_hashes(sources)
                    torch.cuda.reset_peak_memory_stats(device)
                    logging.info("Starting %s", arm)
                    reconstruction, report = run_matrix(archive["targets"][name], name=name,
                        model_path=archive["model_path"], full=full, factors=factors,
                        steps=2000, seed=31, device=device)
                    report["metrics"] = measure(archive, name, reconstruction, device)
                    report["metrics"]["scope"] = f"Original L{layer} inputs; gate-only block error holds same-layer up/down at original HF weights."
                    report.update(layer_index=layer, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                        checkpoint_indices_source=archive["checkpoint_id"], capture_sha256=manifest["inputs_sha256"],
                        reference_l0_manifest=manifest["reference_l0_manifest"], status="completed")
                    report["layer_dispatch"] = {"actual_module_name": name, "selection": "explicit LinearRef in direct train_group_vae_payload",
                        "recipe_target_layers": "0", "recipe_target_layers_used_for_selection": False}
                    record["arms"][arm] = report
                    pair["full" if full else "block32"] = report
                    write_json(output / "screen.json", record)
                    del reconstruction
                    gc.collect()
                    torch.cuda.empty_cache()
                for key in ("main_code_bits", "unprotected_module_bits", "protection_bits", "rotation_sign_bits"):
                    if pair["full"][key] != pair["block32"][key]:
                        raise ValueError(f"Matched rate changed: {model}/{category}/{key}.")
                record["comparisons"][f"{model}/layer{layer}/{category}"] = paired_comparison(
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
