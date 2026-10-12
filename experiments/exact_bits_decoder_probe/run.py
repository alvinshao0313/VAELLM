"""Fit two fixed exact64 assignments; reuse the completed decoder500 controls."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import torch

from experiments.exact_bits_decoder_probe.replay import replay
from experiments.full_weight_bits_probe.core import check_deadline, train
from experiments.joint_reconstruction_probe.core import rebuild, require, restore
from experiments.joint_reconstruction_probe.run import comparison
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.metrics import measure
from experiments.rotation_factor_probe.run import load_archive
from litebsq.protected_channel_quant import decode_protected_channel_weight
from train_utils.utils import configure_deterministic_mode, set_seed


NAME = "model.layers.0.mlp.gate_proj"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    preparation = json.loads(args.preparation.read_text())
    require(preparation["status"] == "cpu_ready", "CPU pairing is not ready.")
    check_hashes(preparation["source_sha256"])
    check_hashes(preparation["input_sha256"])
    output = Path(preparation["output_root"]) / "run"
    require(not output.exists(), "Use a new output; no overwrite or automatic retry.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One assigned GPU is required.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    control = json.loads(Path(preparation["full_weight_report"]).read_text())
    neighbor = json.loads(Path(preparation["neighbor_report"]).read_text())
    started = time.time()
    deadline = started + preparation["cost_estimate"]["wall_budget_seconds"]
    output.mkdir(parents=True)
    report = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "started_unix": started, "deadline_unix": deadline,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(device),
        "preparation_sha256": sha256(args.preparation), "source_sha256": preparation["source_sha256"],
        "input_sha256": preparation["input_sha256"], "cases": {},
        "boundaries": ["Only two new decoder500 arms; completed fullweight decoder500 controls are reused.",
            "Same original native2k decoder, original rotated FP32 target, and complete native BF16 decode.",
            "Initial banks replay the saved one-pass choices; no search, SparseBit optimizer, or later bit changes.",
            "One bit per selected D32 block, not one bit in the whole matrix.",
            "Equal decoder update budget, with additional prior exact64 search cost on the candidate side.",
            "Fixed 32+32 activation documents are measurement only, never gradients or endpoint selection.",
            "Local gate probe only; no whole-model export, downstream acceptance, or production integration."]}
    write_json(output / "report.json", report)
    try:
        for model in ("qwen", "llama"):
            check_deadline(deadline)
            info = preparation["cases"][model]
            c, n = control["cases"][model]["arms"]["decoder"], neighbor["cases"][model]
            archive, manifest = load_archive(Path(preparation["data_dir"]) / model)
            require(manifest["inputs_sha256"] == n["capture_sha256"], "Activation archive changed.")
            diagnostic = torch.load(info["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
            choices = torch.load(info["selection_file"], map_location="cpu", weights_only=True)
            layer, pairing = replay(diagnostic, choices, c, n)
            require(pairing == info["cpu_pairing"], "Prepared pairing changed.")
            layer = layer.to(device)
            logging.info("Starting %s fixed exact64 decoder500", model)
            rotated, observed, trace = train(layer, diagnostic["rotated_weight"], joint=False,
                steps=500, deadline=deadline, observer=lambda item: logging.info("%s %s", model, item))
            require(observed["initial_payload"] == n["local"]["final_payload"], "Training used a different initial state.")
            require(observed["settings"] == c["settings"], "Matched decoder recipe changed.")
            for actual, expected in ((observed["before_deployment"], n["local"]["path_metrics"]["deployment_after"]),
                                     (observed["before_train_decode"], n["local"]["path_metrics"]["full_packed_after"])):
                require(all(math.isclose(actual[k], expected[k], rel_tol=1e-6, abs_tol=1e-12) for k in actual),
                        "Initial same-path reconstruction differs materially from the saved exact64 endpoint.")
            candidate = restore(archive["targets"][NAME], NAME, rotated, device)
            indices = archive["targets"][NAME]["protected_input_indices"]
            # Restore uses the unchanged original INT8 payload for both endpoints.
            bundle = archive["targets"][NAME]
            protected = decode_protected_channel_weight(bundle["protected_input_qvalues"],
                bundle["protected_input_scales"], quant_format="int8", dtype=torch.bfloat16, device=device)
            require(torch.equal(candidate[:, indices], protected.T.cpu()), "Protected columns changed.")
            metrics = measure(archive, NAME, candidate, device)
            payload_file = output / f"{model}_exact64_decoder500_local_payload.pt"
            torch.save({name: value.detach().cpu() for name, value in layer.state_dict().items()}, payload_file)
            saved = torch.load(payload_file, map_location="cpu", weights_only=True, mmap=True)
            require({k: tensor_record(v) for k, v in saved.items()} == observed["final_payload"],
                    "Saved payload differs from the trained deployment state.")
            fresh = rebuild(diagnostic, "gate_proj")
            fresh.load_state_dict(saved, strict=True)
            fresh._build_parallel_stage_decode_plan()
            require({k: tensor_record(v) for k, v in fresh.state_dict().items()} == observed["final_payload"],
                    "Fresh strict reconstruction differs from the saved state.")
            require(observed["stage_final_net_xor"] == [0, 0]
                    and observed["stage_flip_events"] == [0, 0], "Decoder fitting changed fixed replayed codes.")
            report["cases"][model] = {"status": "completed", "model_type": archive["model_type"],
                "source_checkpoint_id": n["checkpoint_id"], "capture_sha256": manifest["inputs_sha256"],
                "pairing": pairing, "training": observed, "metrics": metrics,
                "versus_original_native2k": comparison(n["before"], metrics),
                "versus_exact64_before_fit": comparison(n["after"], metrics),
                "versus_matched_decoder500": comparison(c["metrics"], metrics),
                "control_reference": {"report": preparation["full_weight_report"], "arm": "decoder", "steps": 500},
                "protected_columns_exact": True, "local_payload_file": str(payload_file),
                "local_payload_sha256": sha256(payload_file), "saved_and_fresh_strict_state_exact": True}
            write_json(output / "report.json", report)
            del layer, diagnostic, choices, rotated, trace, candidate, archive, saved, fresh, protected
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
