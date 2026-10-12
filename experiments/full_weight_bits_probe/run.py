"""Two-model gate-only full-weight comparison with native SparseBit scores."""
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

from experiments.full_weight_bits_probe.core import check_deadline, train
from experiments.joint_reconstruction_probe.core import decode, rebuild, require, restore
from experiments.joint_reconstruction_probe.run import comparison
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.metrics import measure
from experiments.rotation_factor_probe.run import load_archive
from experiments.rotation_factor_probe.train import run_matrix
from train_utils.utils import configure_deterministic_mode, set_seed


NAME = "model.layers.0.mlp.gate_proj"


def paired_training(control, joint):
    require(control["initial_payload"] == joint["initial_payload"], "Paired initial payloads differ.")
    rows = []
    first_flip = joint["first_flip_step"]
    for a, b in zip(control["trajectory"], joint["trajectory"], strict=True):
        if first_flip is None or b["step"] <= first_flip:
            rows.append({"step": b["step"], "loss_equal": a["nmse_loss"] == b["nmse_loss"],
                "loss_abs_delta": abs(a["nmse_loss"] - b["nmse_loss"]),
                **b["decoder_gradient_vs_control"]})
    return {"initial_payload_exact": True, "first_flip_step": first_flip,
        "forward_steps_before_first_update_flip": rows,
        "all_preflip_losses_equal": all(x["loss_equal"] for x in rows),
        "all_preflip_decoder_gradients_equal": all(x["exact"] for x in rows),
        "max_preflip_loss_abs_delta": max(x["loss_abs_delta"] for x in rows),
        "max_preflip_decoder_gradient_relative_l2": max(x["relative_l2"] for x in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--mode", choices=("mechanism", "formal"), required=True)
    args = parser.parse_args()
    preparation = json.loads(args.preparation.read_text())
    require(preparation["status"] == "cpu_ready", "CPU preparation is not ready.")
    check_hashes(preparation["source_sha256"])
    check_hashes(preparation["input_sha256"])
    root, data = Path(preparation["output_root"]), Path(preparation["data_dir"])
    output = root / args.mode
    require(not output.exists(), "This phase must have a new directory; no overwrite or automatic retry.")
    short = args.mode == "mechanism"
    steps = 2 if short else 500
    started = time.time()
    if short:
        deadline = started + 3600
        mechanism = None
    else:
        proof = root / "mechanism/report.json"
        mechanism = json.loads(proof.read_text())
        require(mechanism["status"] == "verified" and (root / "mechanism.exit").read_text().strip() == "0",
                "The actual two-update full-matrix mechanism check must finish successfully first.")
        require(mechanism["preparation_sha256"] == sha256(args.preparation), "Mechanism preparation changed.")
        deadline = mechanism["deadline_unix"]
        require(time.time() + mechanism["projected_formal_seconds_with_margin"] < deadline,
                "Measured full-matrix cost does not fit the remaining 60-minute budget.")
    check_deadline(deadline)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one assigned CUDA GPU is required.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output.mkdir(parents=True)
    report = {"status": "running", "mode": args.mode, "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split(")", 1)[1].split()[19]),
        "started_unix": started, "deadline_unix": deadline, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "preparation_sha256": sha256(args.preparation), "source_sha256": preparation["source_sha256"],
        "input_sha256": preparation["input_sha256"], "cases": {},
        "boundaries": ["Complete original rotated FP32 gate matrix and native full-bank BF16 decode on every update.",
            "Same fresh native 2000-step-per-stage payload for decoder control and decoder+100%-bit arm.",
            "500-step linear decoder schedule also used during the 2-update mechanism check; short updates discarded.",
            "Measurement-only original 32+32 activation documents; no activation or evaluation data enters gradients.",
            "Full-batch is a changed optimization condition; no causal comparison with the preceding B8192 study.",
            "Isolated local probe; no whole-model export, full evaluation, or production algorithm integration."]}
    write_json(output / "report.json", report)
    try:
        for model in ("qwen", "llama"):
            check_deadline(deadline)
            check_hashes(preparation["source_sha256"])
            archive, manifest = load_archive(data / model)
            baseline_file = root / "mechanism" / f"{model}_initial_payload.pt"
            if short:
                require(not baseline_file.exists(), "Do not replace an initial payload.")
                diagnostic = {}
                baseline, native = run_matrix(archive["targets"][NAME], name=NAME,
                    model_path=archive["model_path"], full=False, factors={}, steps=2000, seed=31,
                    device=device, diagnostic_out=diagnostic)
                torch.save(diagnostic, baseline_file)
                diagnostic_sha = sha256(baseline_file)
                baseline_metrics = measure(archive, NAME, baseline, device)
            else:
                m = mechanism["cases"][model]
                require(sha256(baseline_file) == m["initial_payload_file_sha256"], "Fresh native initial payload changed.")
                diagnostic = torch.load(baseline_file, map_location="cpu", weights_only=True, mmap=True)
                diagnostic_sha, native = m["initial_payload_file_sha256"], m["native"]
                baseline_metrics = m["baseline"]
                baseline = None
            case = {"native": native, "initial_payload_file": str(baseline_file),
                "initial_payload_file_sha256": diagnostic_sha, "capture_sha256": manifest["inputs_sha256"],
                "model_type": archive["model_type"], "baseline": baseline_metrics, "arms": {}}
            control_trace = None
            for mode in ("decoder", "joint"):
                check_deadline(deadline)
                layer = rebuild(diagnostic, "gate_proj").to(device)
                before = restore(archive["targets"][NAME], NAME, decode(layer, deployment=True), device)
                if baseline is not None:
                    require(torch.equal(before, baseline), "Native full-matrix rebuild differs from the fresh baseline.")
                else:
                    require(measure(archive, NAME, before, device) == baseline_metrics,
                            "Persisted initial payload no longer produces the original baseline metrics.")
                logging.info("Starting %s/%s steps=%d, complete matrix=%s", model, mode, steps,
                             list(diagnostic["rotated_weight"].shape))
                rotated, observed, trace = train(layer, diagnostic["rotated_weight"], joint=mode == "joint",
                    steps=steps, deadline=deadline, observer=lambda item: logging.info("%s/%s %s", model, mode, item),
                    control_trace=control_trace)
                if mode == "decoder":
                    control_trace = trace
                candidate = restore(archive["targets"][NAME], NAME, rotated, device)
                indices = archive["targets"][NAME]["protected_input_indices"]
                require(torch.equal(candidate[:, indices], before[:, indices]), "The protected original INT8 columns changed.")
                observed["metrics"] = measure(archive, NAME, candidate, device)
                observed["protection_columns_exact"] = True
                observed["baseline_comparison"] = comparison(baseline_metrics, observed["metrics"])
                if not short:
                    state_file = output / f"{model}_{mode}_local_payload.pt"
                    torch.save({k: v.detach().cpu() for k, v in layer.state_dict().items()}, state_file)
                    reloaded = torch.load(state_file, map_location="cpu", weights_only=True, mmap=True)
                    require({k: tensor_record(v) for k, v in reloaded.items()} == observed["final_payload"],
                            "Saved local payload differs from the observed BF16/native packed state.")
                    observed["local_payload_file"] = str(state_file)
                    observed["local_payload_sha256"] = sha256(state_file)
                    observed["local_payload_reload_exact"] = True
                    del reloaded
                case["arms"][mode] = observed
                report["cases"][model] = case
                write_json(output / "report.json", report)
                del layer, rotated, candidate, before, trace
                gc.collect()
                torch.cuda.empty_cache()
            case["paired_training"] = paired_training(case["arms"]["decoder"], case["arms"]["joint"])
            case["joint_over_decoder"] = comparison(case["arms"]["decoder"]["metrics"], case["arms"]["joint"]["metrics"])
            case["status"] = "completed"
            require(sha256(baseline_file) == diagnostic_sha, "Training mutated the initial payload archive.")
            if short:
                pairing = case["paired_training"]
                require(pairing["max_preflip_loss_abs_delta"] <= 1e-7
                        and pairing["max_preflip_decoder_gradient_relative_l2"] <= 1e-5,
                        "Actual pre-flip control/joint loss or decoder gradients differ; investigate before formal work.")
            write_json(output / "report.json", report)
            del diagnostic, archive, baseline, control_trace
            gc.collect()
            torch.cuda.empty_cache()
        check_hashes(preparation["source_sha256"])
        check_hashes(preparation["input_sha256"])
        if short:
            # Keep first-step compilation/allocation overhead once, not 500 times.
            # The second update includes real full-batch backward and initialized Adam.
            arms = [arm for c in report["cases"].values() for arm in c["arms"].values()]
            projected = 180 + 1.5 * sum(500 * arm["trajectory"][1]["step_seconds"]
                + max(0, arm["trajectory"][0]["step_seconds"] - arm["trajectory"][1]["step_seconds"])
                for arm in arms)
            report["projected_formal_seconds_with_margin"] = projected
            report["projection_formula"] = "180s endpoint allowance + 1.5 * sum(500 * update2_time + positive(update1_time-update2_time))"
            report["fits_remaining_wall_clock_budget"] = time.time() + projected < deadline
            require(report["fits_remaining_wall_clock_budget"], "Measured execution does not fit the remaining 60-minute budget.")
        report.update(status="verified" if short else "completed", finished_unix=time.time())
        write_json(output / "report.json", report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        write_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
