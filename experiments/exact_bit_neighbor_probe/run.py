"""Two fresh native gate payloads, one finite-neighbor pass, paired measurements."""
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

from experiments.exact_bit_neighbor_probe.core import probe
from experiments.joint_reconstruction_probe.core import error_stats, rebuild, require, restore
from experiments.joint_reconstruction_probe.run import comparison
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.metrics import measure
from experiments.rotation_factor_probe.run import load_archive
from train_utils.utils import configure_deterministic_mode, set_seed


NAME = "model.layers.0.mlp.gate_proj"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    preparation = json.loads(args.preparation.read_text())
    require(preparation["status"] == "cpu_ready", "A verified CPU preparation is required.")
    check_hashes(preparation["source_sha256"])
    check_hashes(preparation["input_sha256"])
    output = Path(preparation["output_root"]) / "run"
    require(not output.exists(), "Never overwrite an existing run.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One assigned GPU is required.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output.mkdir(parents=True)
    report = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split(")", 1)[1].split()[19]),
        "started_unix": time.time(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": torch.cuda.get_device_name(device), "preparation_sha256": sha256(args.preparation),
        "source_sha256": preparation["source_sha256"], "input_sha256": preparation["input_sha256"], "cases": {},
        "boundaries": ["Original native2k-per-stage payload; no prior decoder/bit updates reused.",
            "One pass, all 64 genuine single-bit neighbors per independent D32 block; decoder fixed.",
            "Original rotated FP32 weight target only; activation documents never select candidates.",
            "STE derivative is observed once; no optimizer, score update, or round advancement.",
            "Same-chunk objective monotonicity is distinct from fullpacked/fused/inverse-rotation BF16 errors.",
            "No whole-model checkpoint, task evaluation, downstream claim, or production algorithm integration."]}
    write_json(output / "report.json", report)
    try:
        for model in ("qwen", "llama"):
            check_hashes(preparation["source_sha256"])
            archive, manifest = load_archive(Path(preparation["data_dir"]) / model)
            info = preparation["models"][model]
            diagnostic = torch.load(info["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
            layer = rebuild(diagnostic, "gate_proj").to(device)
            torch.cuda.reset_peak_memory_stats(device)
            logging.info("Starting %s fixed decoder one-pass %d blocks", model, info["blocks"])
            left, right, selected, local = probe(layer, diagnostic["rotated_weight"],
                lambda item: logging.info("%s candidate progress %s", model, item))
            before = restore(archive["targets"][NAME], NAME, left, device)
            after = restore(archive["targets"][NAME], NAME, right, device)
            before_metrics, after_metrics = measure(archive, NAME, before, device), measure(archive, NAME, after, device)
            require(before_metrics == info["baseline"], "Original native deployment baseline changed.")
            indices = archive["targets"][NAME]["protected_input_indices"]
            require(torch.equal(before[:, indices], after[:, indices]), "Protected original INT8 columns changed.")
            # This compact choice vector plus the pinned original payload exactly reproduces every bank.
            selection_file = output / f"{model}_selected_choices.pt"
            torch.save(selected, selection_file)
            reloaded = torch.load(selection_file, weights_only=True, map_location="cpu")
            require(torch.equal(selected, reloaded), "Choice vector failed an exact save/reload.")
            for stage in range(2):
                original = diagnostic["stage_codes"][stage].clone()
                changed = (selected > stage * 32) & (selected <= (stage + 1) * 32)
                rows = torch.arange(len(selected))[changed]
                bits = selected[changed].long() - 1 - stage * 32
                original[rows, 0, bits // 8] ^= (1 << (bits % 8)).to(torch.uint8)
                require(torch.equal(original, layer.get_stage_part_vq_storage(stage).cpu()),
                        "Saved compact choices do not reconstruct the real persistent packed bank.")
            case = {"status": "completed", "model_type": archive["model_type"], "checkpoint_id": archive["checkpoint_id"],
                "capture_sha256": manifest["inputs_sha256"], "local": local,
                "before": before_metrics, "after": after_metrics, "comparison": comparison(before_metrics, after_metrics),
                "original_coordinate_before": error_stats(before, archive["targets"][NAME]["reference_weight"]),
                "original_coordinate_after": error_stats(after, archive["targets"][NAME]["reference_weight"]),
                "protected_columns_exact": True, "selection_file": str(selection_file),
                "selection_file_sha256": sha256(selection_file), "selection_tensor": tensor_record(selected),
                "compact_save_reload_and_bank_reconstruction_exact": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            report["cases"][model] = case
            write_json(output / "report.json", report)
            del layer, left, right, before, after, archive, diagnostic, selected, reloaded
            gc.collect()
            torch.cuda.empty_cache()
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
