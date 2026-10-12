"""Four paired, fixed-code decoder reconstructions; no whole-model checkpoint."""
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

from experiments.joint_reconstruction_probe.core import decode, error_stats, fit, rebuild, require, restore
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.metrics import measure
from experiments.rotation_factor_probe.run import load_archive
from experiments.rotation_factor_probe.train import run_matrix
from train_utils.utils import configure_deterministic_mode, set_seed


ROOT = Path(__file__).resolve().parents[2]


def source_pins():
    paths = {ROOT / "experiments/longbench_v1/common.py"}
    for folder in ("experiments/joint_reconstruction_probe", "experiments/rotation_factor_probe",
                   "train_utils", "litebsq", "rotation"):
        paths.update((ROOT / folder).rglob("*.py"))
    return {str(path): sha256(path) for path in sorted(paths)}


def comparison(before, after):
    result = {"weight_nmse_after_over_before": after["weight_nmse"] / before["weight_nmse"], "splits": {}}
    for split in ("train", "heldout"):
        result["splits"][split] = {}
        for metric in ("linear_output", "mlp_block_output"):
            a, b = before["splits"][split][metric], after["splits"][split][metric]
            pairs = list(zip(a["documents"], b["documents"], strict=True))
            require(all((x["index"], x["tokens"]) == (y["index"], y["tokens"]) for x, y in pairs),
                    "Paired activation document boundaries changed.")
            result["splits"][split][metric] = {"nmse_after_over_before": b["nmse"] / a["nmse"],
                "improved_documents": sum(y["nmse"] < x["nmse"] for x, y in pairs), "documents": len(pairs)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    preparation = json.loads(args.preparation.read_text())
    require(preparation["status"] == "cpu_ready", "CPU preparation has not passed.")
    check_hashes(preparation["source_sha256"])
    check_hashes(preparation["input_sha256"])
    output, data = Path(preparation["output_dir"]), Path(preparation["data_dir"])
    require(not output.exists(), "Result directory must be new.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one assigned visible CUDA device is required.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split(")", 1)[1].split()[19]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(device),
        "preparation_sha256": sha256(args.preparation), "source_sha256": preparation["source_sha256"],
        "input_sha256": preparation["input_sha256"], "started_unix": time.time(), "cases": {},
        "boundaries": ["Only original L0 gate/down matrices, fixed protection and block32 rotation.",
            "Baseline native 2000 steps per stage; joint arm has 500 additional decoder updates, not equal training budget.",
            "All activation documents are measurement only; neither loss nor endpoint selection consumes them.",
            "Fixed codes; no activation KD, LoRA, new encoder, or whole-model export.",
            "Local error changes do not establish downstream accuracy or whole-model bitrate acceptance."]}
    write_json(output / "report.json", report)
    try:
        for model in ("qwen", "llama"):
            archive, manifest = load_archive(data / model)
            for category in ("gate_proj", "down_proj"):
                key, name = f"{model}/{category}", "model.layers.0.mlp." + category
                check_hashes(preparation["source_sha256"])
                torch.cuda.reset_peak_memory_stats(device)
                logging.info("Starting %s native reconstruction", key)
                diagnostic = {}
                baseline, baseline_report = run_matrix(archive["targets"][name], name=name,
                    model_path=archive["model_path"], full=False, factors={}, steps=2000, seed=31,
                    device=device, diagnostic_out=diagnostic)
                layer = rebuild(diagnostic, category).to(device)
                rebuilt = restore(archive["targets"][name], name, decode(layer, deployment=True), device)
                require(torch.equal(rebuilt, baseline), "Rebuilt native deployment differs from actual baseline.")
                before = measure(archive, name, baseline, device)
                target = diagnostic["rotated_weight"]
                protection_before = {k: manifest["tensors"][name][k] for k in (
                    "protected_input_indices", "protected_input_qvalues", "protected_input_scales")}
                rotated, joint = fit(layer, target, lambda item: logging.info("%s joint %s", key, item))
                candidate = restore(archive["targets"][name], name, rotated, device)
                after = measure(archive, name, candidate, device)
                indices = archive["targets"][name]["protected_input_indices"]
                require(torch.equal(candidate[:, indices], baseline[:, indices]), "Protected INT8 columns changed.")
                # Keep only the small final decoder, not fixed-code or whole-matrix duplicates.
                state_path = output / f"{model}_{category}_decoder.pt"
                torch.save({k: v.detach().cpu() for k, v in layer._parallel_stage_decoder.state_dict().items()}, state_path)
                case = {"status": "completed", "native": baseline_report, "joint": joint,
                    "model_type": archive["model_type"], "protection_source_checkpoint_id": archive["checkpoint_id"],
                    "capture_sha256": manifest["inputs_sha256"], "rebuild_deployment_exact": True,
                    "protection_manifest": protection_before, "protection_columns_exact": True,
                    "before": before, "after": after, "comparison": comparison(before, after),
                    "original_coordinate_raw_mse_before": error_stats(baseline, archive["targets"][name]["reference_weight"]),
                    "original_coordinate_raw_mse_after": error_stats(candidate, archive["targets"][name]["reference_weight"]),
                    "final_decoder_file": str(state_path), "final_decoder_sha256": sha256(state_path),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                report["cases"][key] = case
                write_json(output / "report.json", report)
                del layer, baseline, rebuilt, rotated, candidate, diagnostic, target
                gc.collect()
                torch.cuda.empty_cache()
            del archive
        check_hashes(preparation["source_sha256"])
        check_hashes(preparation["input_sha256"])
        report.update(status="completed", finished_unix=time.time())
        write_json(output / "report.json", report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        write_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
