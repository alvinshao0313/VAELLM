"""Fixed-code Qwen L18 decoder fitting against the actual gated MLP output."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch

from experiments.decoder_budget_probe.run import load_inputs
from experiments.gate_output_fit_probe.core import train
from experiments.joint_reconstruction_probe.core import rebuild, require, restore
from experiments.joint_reconstruction_probe.run import comparison
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.middle_exact_decoder_probe.data import measure_confirmation
from experiments.middle_exact_decoder_probe.run import confirmation_comparison, measure_dev
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.run import load_archive
from train_utils.utils import configure_deterministic_mode, set_seed


NAME = "model.layers.18.mlp.gate_proj"
ARMS = ("original_codes", "exact64_codes")


def read(path):
    return json.loads(Path(path).read_text())


def prepare(args):
    require(args.base_root is not None and args.output_root is not None, "Preparation requires both paths.")
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists(), "Use a new independent output directory.")
    budget = base / "decoder_budget_probe_r1"
    previous = read(budget / "preparation.json")
    report, review = read(budget / "run/report.json"), read(budget / "review.json")
    require(report["status"] == "completed" and review["status"] == "verified"
            and review["report_sha256"] == sha256(budget / "run/report.json")
            and (budget / "run.exit").read_text().strip() == "0", "The prior budget diagnostic must be completed and verified.")
    check_hashes(previous["source_sha256"])
    check_hashes(previous["input_sha256"])
    torch.set_num_threads(4)
    prior, initial, choices, selected = load_inputs(previous)
    archive, manifest = load_archive(Path(previous["archive_dir"]))
    require(archive["model_type"] == "qwen3" and archive["layer_index"] == 18
            and manifest["hidden_act"] == "silu" and prior["target_name"] == NAME,
            "This diagnostic is limited to the same Qwen L18 gated MLP.")
    original = rebuild(initial, "gate_proj")
    original_state, selected_state = original.state_dict(), selected.state_dict()
    require(all(torch.equal(value, selected_state[name]) for name, value in original_state.items()
                if name not in {"vq_weight", "vq_weight_s1"}), "Matched decoder initialization must be identical.")
    for split in ("train", "heldout"):
        lengths = archive["document_lengths"][split]
        inputs = archive["targets"][NAME][split + "_inputs"]
        require(len(lengths) == 32 and sum(lengths) == len(inputs) and inputs.dtype == torch.bfloat16,
                "The original training/development population changed.")
    cache = torch.load(previous["replay32_cache"], map_location="cpu", weights_only=True, mmap=True)
    capture = read(previous["capture_manifest"])
    require(cache["document_lengths"] == capture["document_lengths"]
            and tensor_record(cache["inputs"][NAME]) == capture["input_tensors"][NAME], "The previously viewed replay capture changed.")
    sources, inputs = dict(previous["source_sha256"]), dict(previous["input_sha256"])
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
    for path in (budget / "preparation.json", budget / "run/report.json", budget / "review.json", budget / "run.exit"):
        inputs[str(path)] = sha256(path)
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    record = {"status": "cpu_ready", "source_sha256": sources, "input_sha256": inputs,
        "output_root": str(output), "python": sys.executable, "torch": torch.__version__, "cuda_initialized": False,
        **{key: previous[key] for key in ("prior_report", "archive_dir", "capture_manifest", "replay32_cache",
            "cpu_pairing", "initial_payload", "selected_payload", "rotation", "protection")},
        "prior_weight_budget_report": str(budget / "run/report.json"),
        "protocol": {"target_name": NAME, "mechanism_updates": 2, "mechanism_arm": "original_codes",
            "steps_per_arm": 500, "schedule_steps": 500, "decoder_lr": 1e-5,
            "decoder_betas": [.9, .95], "decoder_eps": 1e-8, "decoder_weight_decay": 0,
            "initialization": "Fresh shared native2k BF16 decoder; candidate only replays previously saved exact64 choices.",
            "loss": "Actual BF16 gate/SiLU/up-product/down MSE divided by fixed original-HF MLP output mean square.",
            "population": "Every update uses all valid tokens of the original 32 training documents, no sampling.",
            "precision": "FP32 decoder masters/Adam, native BF16 packed decode/stage sum, differentiable block32 inverse rotation and fixed INT8 columns.",
            "evaluation": "Old heldout and previously viewed replay32 at fixed final endpoints, not fresh unseen tests.",
            "mechanism": "Two real updates validate the new differentiable output objective; neither resulting state nor optimizer initializes either 500-step arm.",
            "wall_budget_seconds": 900, "kill_grace_seconds": 20,
            "resource_estimate": "One A800, expected under 10GiB; real two-update peak/time recorded before paired fitting.",
            "boundary": "An isolated output-target diagnostic; no whole-model quality claim or production algorithm integration."},
        "experience_review": ["Full-weight decoder2000 did not remove the Qwen L18 gated-output reversal.",
            "Earlier complete 36-layer local output recovery improved most calibration/development blocks but worsened full downstream tasks.",
            "Reuse existing original activations only; no test answers or downstream evaluation examples enter this objective."],
        "command": [sys.executable, "-m", "experiments.gate_output_fit_probe.run", "--preparation", str(output / "preparation.json")]}
    output.mkdir(parents=True)
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": "cpu_ready", "preparation_sha256": sha256(output / "preparation.json"),
        "sources": len(sources), "inputs": len(inputs), "cuda_initialized": False}))


def run(path):
    prepared = read(path)
    require(prepared["status"] == "cpu_ready", "CPU preparation must pass first.")
    check_hashes(prepared["source_sha256"])
    check_hashes(prepared["input_sha256"])
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one assigned GPU is required.")
    output = Path(prepared["output_root"]) / "run"
    require(not output.exists(), "Never overwrite or retry an existing run.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    prior, initial, choices, selected = load_inputs(prepared)
    archive, manifest = load_archive(Path(prepared["archive_dir"]))
    cache = torch.load(prepared["replay32_cache"], map_location="cpu", weights_only=True, mmap=True)
    output.mkdir()
    started = time.time()
    deadline = started + prepared["protocol"]["wall_budget_seconds"]
    report = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "started_unix": started, "deadline_unix": deadline, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "preparation_sha256": sha256(path), "protocol": prepared["protocol"], "arms": {}}
    write_json(output / "report.json", report)
    try:
        layer = rebuild(initial, "gate_proj").to(device)
        logging.info("Starting two-update output-objective mechanism check")
        _, mechanism = train(layer, archive, NAME, steps=2, deadline=deadline,
            observer=lambda item: logging.info("mechanism %s", item))
        require(mechanism["initial_payload"] == prepared["initial_payload"], "Mechanism input changed.")
        report["mechanism"] = mechanism
        write_json(output / "report.json", report)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        for arm in ARMS:
            require(time.time() < deadline, "The total assigned wall-clock budget expired.")
            layer = rebuild(initial, "gate_proj")
            if arm == "exact64_codes":
                layer.load_state_dict(selected.state_dict(), strict=True)
                layer._build_parallel_stage_decode_plan()
            layer = layer.to(device)
            logging.info("Starting %s fresh output decoder500", arm)
            rotated, trained = train(layer, archive, NAME, steps=500, deadline=deadline,
                observer=lambda item: logging.info("%s %s", arm, item))
            expected = prepared["initial_payload"] if arm == "original_codes" else prepared["selected_payload"]
            require(trained["initial_payload"] == expected, "A formal arm reused a fitted decoder.")
            candidate = restore(archive["targets"][NAME], NAME, rotated, device)
            metrics = measure_dev(archive, NAME, candidate, device, 18)
            replay32 = measure_confirmation(archive, NAME, candidate, cache["inputs"][NAME], cache["document_lengths"], device)
            state_file = output / f"qwen_l18_{arm}_output_decoder500.pt"
            torch.save({key: value.detach().cpu() for key, value in layer.state_dict().items()}, state_file)
            fresh = rebuild(initial, "gate_proj")
            fresh.load_state_dict(torch.load(state_file, map_location="cpu", weights_only=True, mmap=True), strict=True)
            require({key: tensor_record(value) for key, value in fresh.state_dict().items()} == trained["final_payload"],
                    "Saved output-fit endpoint failed strict fresh state reconstruction.")
            old = prior["arms"]["decoder500" if arm == "original_codes" else "exact64_decoder500"]
            report["arms"][arm] = {"training": trained, "old_dev": metrics, "replay32": replay32,
                "versus_same_codes_weight500": {"old_dev": comparison(old["old_dev"], metrics),
                    "replay32": confirmation_comparison(old["confirmation"], replay32)},
                "state_file": str(state_file), "state_sha256": sha256(state_file), "saved_fresh_strict_state_exact": True,
                "original_archive_sha256": manifest["inputs_sha256"]}
            write_json(output / "report.json", report)
            del layer, rotated, candidate, fresh
            gc.collect()
            torch.cuda.empty_cache()
        a, b = (report["arms"][arm] for arm in ARMS)
        require(a["training"]["settings"] == b["training"]["settings"], "Matched output-objective settings differ.")
        report["comparisons"] = {"old_dev": comparison(a["old_dev"], b["old_dev"]),
            "replay32": confirmation_comparison(a["replay32"], b["replay32"])}
        check_hashes(prepared["source_sha256"])
        check_hashes(prepared["input_sha256"])
        report.update(status="completed", finished_unix=time.time())
        write_json(output / "report.json", report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        write_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preparation", type=Path)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    prepare(args) if args.prepare else run(args.preparation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
