"""Qwen L18 matched fixed-code fits with a fresh 2000-step linear horizon."""
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

from experiments.decoder_budget_probe.core import train
from experiments.exact_bits_decoder_probe.replay import replay
from experiments.joint_reconstruction_probe.core import rebuild, require, restore
from experiments.joint_reconstruction_probe.run import comparison
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.middle_exact_decoder_probe.data import measure_confirmation
from experiments.middle_exact_decoder_probe.run import confirmation_comparison, measure_dev
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.run import load_archive
from litebsq.protected_channel_quant import decode_protected_channel_weight
from train_utils.utils import configure_deterministic_mode, set_seed


NAME = "model.layers.18.mlp.gate_proj"
ARMS = ("original_codes", "exact64_codes")


def read(path):
    return json.loads(Path(path).read_text())


def load_inputs(prepared):
    prior = read(prepared["prior_report"])["cases"]["qwen"]["middle"]
    initial = torch.load(prior["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
    choices = torch.load(prior["choices_file"], map_location="cpu", weights_only=True, mmap=True)
    neighbor = {"local": prior["neighbor"], "selection_tensor": tensor_record(choices)}
    selected, pairing = replay(initial, choices, prior["arms"]["decoder500"]["training"], neighbor)
    require(pairing == prepared["cpu_pairing"], "Original initial decoder or saved exact64 choices changed.")
    return prior, initial, choices, selected


def prepare(args):
    require(args.base_root is not None and args.output_root is not None, "Preparation requires both paths.")
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not (output / "preparation.json").exists() and not (output / "run").exists(), "Never overwrite a prepared/run directory.")
    prior_dir = base / "middle_exact_decoder_probe"
    prior_path = prior_dir / "run/report.json"
    prior, reviewed, old_preparation = read(prior_path), read(prior_dir / "review.json"), read(prior_dir / "preparation.json")
    require(prior["status"] == "completed" and reviewed["status"] == "verified"
            and reviewed["report_sha256"] == sha256(prior_path)
            and (prior_dir / "run.exit").read_text().strip() == "0", "The previous matched 500-step result must be verified.")
    sources = dict(old_preparation["source_sha256"])
    check_hashes(sources)
    info, case = old_preparation["models"]["qwen"], prior["cases"]["qwen"]
    require(info["target_name"] == NAME and info["layer_index"] == 18, "Only the preselected Qwen L18 gate is authorized.")
    archive, manifest = load_archive(Path(info["middle_dir"]))
    require(archive["model_type"] == "qwen3" and archive["layer_index"] == 18
            and manifest["protection"][NAME]["spec"]["weight_rotation"] == info["middle_rotation"], "Original HF geometry/rotation changed.")
    inputs = dict(manifest["input_files_sha256"])
    middle = case["middle"]
    require(sha256(middle["initial_payload_file"]) == middle["initial_payload_sha256"]
            and sha256(middle["choices_file"]) == middle["choices_sha256"], "Native initial payload or saved one-pass choices changed.")
    torch.set_num_threads(4)
    initial = torch.load(middle["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
    choices = torch.load(middle["choices_file"], map_location="cpu", weights_only=True, mmap=True)
    neighbor = {"local": middle["neighbor"], "selection_tensor": tensor_record(choices)}
    selected, pairing = replay(initial, choices, middle["arms"]["decoder500"]["training"], neighbor)
    original = rebuild(initial, "gate_proj")
    require({k: tensor_record(v) for k, v in original.state_dict().items()} == middle["neighbor"]["initial_payload"],
            "The control must start from the unadapted native2k decoder.")
    capture_path = Path(case["confirmation_manifest"])
    capture = read(capture_path)
    cache_path = capture_path.parent / "inputs.pt"
    require(capture["status"] == "captured" and sha256(capture_path) == case["confirmation_manifest_sha256"]
            and sha256(cache_path) == capture["archive"]["sha256"], "Previously measured confirmation archive changed.")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    require(cache["model_type"] == "qwen3" and cache["document_lengths"] == capture["document_lengths"]
            and len(cache["document_lengths"]) == 32
            and tensor_record(cache["inputs"][NAME]) == capture["input_tensors"][NAME], "Replay32 activation identity changed.")
    for path in (prior_path, prior_dir / "review.json", prior_dir / "run.exit", prior_dir / "preparation.json",
                 Path(info["middle_dir"]) / "data_manifest.json", Path(info["middle_dir"]) / "inputs.pt",
                 Path(middle["initial_payload_file"]), Path(middle["choices_file"]), capture_path, cache_path):
        inputs[str(path)] = sha256(path)
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
    check_hashes(inputs)
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    record = {"status": "cpu_ready", "python": sys.executable, "torch": torch.__version__,
        "source_sha256": sources, "input_sha256": inputs, "cuda_initialized": False,
        "output_root": str(output), "prior_report": str(prior_path), "archive_dir": info["middle_dir"],
        "capture_manifest": str(capture_path), "replay32_cache": str(cache_path), "cpu_pairing": pairing,
        "initial_state_sha256": middle["initial_payload_sha256"], "choices_file_sha256": middle["choices_sha256"],
        "initial_payload": middle["neighbor"]["initial_payload"], "selected_payload": middle["neighbor"]["final_payload"],
        "rotation": info["middle_rotation"], "protection": manifest["protection"][NAME],
        "protocol": {"target_name": NAME, "steps": 2000, "schedule_steps": 2000,
            "decoder_lr": 1e-5, "betas": [.9, .95], "eps": 1e-8, "weight_decay": 0,
            "scheduler": "linear, warmup0; fresh optimizer from the common native2k decoder",
            "loss": "Full original rotated FP32 matrix MSE divided by its fixed mean(W^2)",
            "precision": "FP32 decoder masters and Adam moments, native BF16 packed decode and stage sum",
            "codes": "Original versus saved exact64 assignments, fully frozen in both arms",
            "budget_boundary": "2000-step horizon changes the earlier LR trajectory; not exact-resume of either old500 fit.",
            "documents": "Old train/heldout and already-viewed confirmation32 are endpoint-only replay sets; no fresh unseen-test claim.",
            "no_new_native_fit_or_search": True, "no_four_node_statistics": True,
            "estimated_seconds": [240, 300], "hard_wall_seconds": 600, "kill_grace_seconds": 20},
        "command": [sys.executable, "-m", "experiments.decoder_budget_probe.run", "--preparation", str(output / "preparation.json")]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": "cpu_ready", "preparation_sha256": sha256(output / "preparation.json"),
        "sources": len(sources), "inputs": len(inputs), "cpu_pairing": pairing}))


def run(path):
    prepared = read(path)
    require(prepared["status"] == "cpu_ready", "CPU preparation must pass first.")
    check_hashes(prepared["source_sha256"])
    check_hashes(prepared["input_sha256"])
    output = Path(prepared["output_root"]) / "run"
    require(not output.exists(), "Never replace or retry an existing run.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one assigned GPU is required.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    prior, initial, choices, selected = load_inputs(prepared)
    archive, manifest = load_archive(Path(prepared["archive_dir"]))
    cache = torch.load(prepared["replay32_cache"], map_location="cpu", weights_only=True, mmap=True)
    output.mkdir(parents=True)
    started = time.time()
    deadline = started + prepared["protocol"]["hard_wall_seconds"]
    report = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "started_unix": started, "deadline_unix": deadline, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "preparation_sha256": sha256(path), "source_sha256": prepared["source_sha256"], "input_sha256": prepared["input_sha256"],
        "protocol": prepared["protocol"], "arms": {},
        "boundaries": ["Qwen L18 gate only; no whole-model export, accuracy claim, or production integration.",
            "Both arms restart from the same unadapted native2k decoder with a fresh 2000-step schedule.",
            "Saved one-pass exact64 choices are replayed; no neighbor search or code updates.",
            "All document sets were seen previously and serve only fixed-endpoint measurement, never fitting or selection."]}
    write_json(output / "report.json", report)
    try:
        for arm in ARMS:
            require(time.time() < deadline, "The assigned total wall budget expired.")
            layer = rebuild(initial, "gate_proj")
            if arm == "exact64_codes":
                layer.load_state_dict(selected.state_dict(), strict=True)
                layer._build_parallel_stage_decode_plan()
            layer = layer.to(device)
            logging.info("Starting %s fresh decoder2000", arm)
            rotated, trained = train(layer, initial["rotated_weight"], deadline=deadline,
                observer=lambda item: logging.info("%s %s", arm, item))
            expected = prepared["initial_payload"] if arm == "original_codes" else prepared["selected_payload"]
            require(trained["initial_payload"] == expected, "The 2000-step run used a fitted or changed decoder.")
            candidate = restore(archive["targets"][NAME], NAME, rotated, device)
            bundle = archive["targets"][NAME]
            protected = decode_protected_channel_weight(bundle["protected_input_qvalues"], bundle["protected_input_scales"],
                quant_format="int8", dtype=torch.bfloat16, device=device)
            require(torch.equal(candidate[:, bundle["protected_input_indices"]], protected.T.cpu()), "Original INT8 protection changed.")
            metrics = measure_dev(archive, NAME, candidate, device, 18)
            replay32 = measure_confirmation(archive, NAME, candidate, cache["inputs"][NAME], cache["document_lengths"], device)
            replay32["scope"] = "Previously viewed confirmation32, now a fixed endpoint replay set; no new unseen-test claim."
            state_file = output / f"qwen_l18_{arm}_decoder2000.pt"
            torch.save({k: v.detach().cpu() for k, v in layer.state_dict().items()}, state_file)
            saved = torch.load(state_file, map_location="cpu", weights_only=True, mmap=True)
            fresh = rebuild(initial, "gate_proj")
            fresh.load_state_dict(saved, strict=True)
            require({k: tensor_record(v) for k, v in fresh.state_dict().items()} == trained["final_payload"],
                    "Saved state failed strict fresh reconstruction.")
            old_arm = prior["arms"]["decoder500" if arm == "original_codes" else "exact64_decoder500"]
            report["arms"][arm] = {"status": "completed", "training": trained, "old_dev": metrics, "replay32": replay32,
                "versus_same_codes_old500": {"old_dev": comparison(old_arm["old_dev"], metrics),
                    "replay32": confirmation_comparison(old_arm["confirmation"], replay32)},
                "state_file": str(state_file), "state_sha256": sha256(state_file), "saved_fresh_strict_state_exact": True,
                "protected_columns_exact": True, "rotation": prepared["rotation"], "original_archive_sha256": manifest["inputs_sha256"]}
            write_json(output / "report.json", report)
            del layer, rotated, candidate, protected, saved, fresh
            gc.collect()
            torch.cuda.empty_cache()
        left, right = (report["arms"][arm] for arm in ARMS)
        require(left["training"]["settings"] == right["training"]["settings"], "The matched 2000-step recipes differ.")
        report["comparisons"] = {"old_dev": comparison(left["old_dev"], right["old_dev"]),
            "replay32": confirmation_comparison(left["replay32"], right["replay32"])}
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
