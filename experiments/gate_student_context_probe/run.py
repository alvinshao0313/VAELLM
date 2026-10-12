"""Prepare and run one bounded real-student-context gate comparison."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch

from experiments.gate_student_context_probe.core import measure, ready, records, search_once, train
from experiments.gate_student_context_probe.data import capture, prepare_data
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


ARMS = ("original_codes", "exact64_codes")


def read(path):
    return json.loads(Path(path).read_text())


def merge(destination, incoming):
    require(all(k not in destination or destination[k] == v for k, v in incoming.items()), "Conflicting immutable source/input pins.")
    destination.update(incoming)


def cache_records(cache):
    return {"inputs": {s: tensor_record(x) for s, x in cache["inputs"].items()},
        **{k: tensor_record(cache[k]) for k in ("student_up_weight", "student_down_weight")}}


def paired(control, candidate):
    result = {"weight_nmse_ratio": candidate["original_coordinate_weight"]["nmse"] / control["original_coordinate_weight"]["nmse"]}
    for split in ("train", "heldout", "replay32"):
        a, b = control[split], candidate[split]
        require(a["squared_reference"] == b["squared_reference"] and len(a["documents"]) == len(b["documents"]) == 32,
                "Paired endpoint reference population changed.")
        for x, y in zip(a["documents"], b["documents"], strict=True):
            require(all(x[k] == y[k] for k in ("index", "tokens", "squared_reference")), "Per-document identity or denominator differs.")
        result[split] = {"nmse_ratio": b["nmse"] / a["nmse"], "nmse_delta": b["nmse"] - a["nmse"],
            "improved_documents": sum(y["squared_error"] < x["squared_error"] for x, y in zip(a["documents"], b["documents"], strict=True)),
            "worsened_documents": sum(y["squared_error"] > x["squared_error"] for x, y in zip(a["documents"], b["documents"], strict=True))}
    return result


def prepare(args):
    require(args.base_root is not None and args.output_root is not None, "Preparation requires both paths.")
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists(), "Never overwrite an existing preparation.")
    prior = base / "gate_output_fit_probe"
    previous, review = read(prior / "preparation.json"), read(prior / "review.json")
    require(review["status"] == "verified" and review["report_sha256"] == sha256(prior / "run/report.json")
            and (prior / "run.exit").read_text().strip() == "0", "The prior isolated output result must have a verified terminal.")
    check_hashes(previous["source_sha256"])
    check_hashes(previous["input_sha256"])
    torch.set_num_threads(4)
    data, batches, archive = prepare_data(base)
    sources, inputs = dict(previous["source_sha256"]), dict(previous["input_sha256"])
    merge(sources, data["source_sha256"])
    merge(inputs, data["input_sha256"])
    for path in Path(__file__).resolve().parent.glob("*.py"):
        ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
    for path in (prior / "preparation.json", prior / "review.json", prior / "run/report.json", prior / "run.exit"):
        inputs[str(path)] = sha256(path)
    require(not torch.cuda.is_initialized() and all(v.device.type == "cpu" for b in batches.values() for v in b.values()),
            "CPU preparation must not initialize CUDA or capture student activations.")
    record = {"status": "ready_for_capture", "source_sha256": sources, "input_sha256": inputs,
        "data": data, "base_root": str(base), "output_root": str(output), "python": sys.executable,
        "torch": torch.__version__, "cuda_initialized": False, "student_capture_completed": False,
        "cuda_mechanism_completed": False,
        "protocol": {"target": data["target"], "checkpoint_id": data["checkpoint_id"],
            "wall_budget_seconds": 1200, "kill_grace_seconds": 20, "visible_gpus": 1,
            "mechanism_updates": 2, "formal_updates_each": 500, "schedule_horizon": 500,
            "decoder_lr": 1e-5, "decoder_betas": [.9, .95], "eps": 1e-8, "weight_decay": 0, "warmup": 0,
            "search": "Once from f927 current decoder/current codes; current plus all64 single-bit neighbors per D32 vector; original HF rotated weight objective, no activation selection",
            "training": "Fresh same f927 decoder in both arms; fixed original or newly selected banks; actual compressed up/down fixed; all train32 student-prefix tokens every update",
            "target_semantics": "HF_MLP(x_student) conditional function, not HF complete forward or compensation for a different teacher prefix",
            "precision": "FP32 decoder/Adam masters; BF16 native full VAELinear decode already includes rotation/protection; BF16 MLP, FP32 MSE/fixed HF output mean square",
            "evaluation": "Original held32 and previously viewed replay32, fixed final endpoint only; no blind-test claim",
            "resource_estimate": "Prior two500 output arms took97s and about5.03GB peak; new full student load/capture is a separate higher peak, recorded before fitting. No automatic budget extension.",
            "boundary": "Single-layer isolated prototype; no production integration, model export, downstream score or nine-task acceptance."},
        "experience_review": "Old full36 student-prefix/teacher-next-hidden and teacher-input joint recovery had local gains but degraded full8. This uses a recovered f927 single gate, fixed actual context and same-input conditional teacher target; the old failure still forbids extrapolating local gains to downstream quality.",
        "command": [sys.executable, "-m", "experiments.gate_student_context_probe.run", "--preparation", str(output / "preparation.json")]}
    output.mkdir(parents=True)
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": record["status"], "preparation_sha256": sha256(output / "preparation.json"),
        "source_pins": len(sources), "input_pins": len(inputs), "tokens": {s: sum(v) for s, v in data["document_lengths"].items()}, "cuda_initialized": False}))


def fresh(initial, state, device):
    layer = deepcopy(initial)
    if state is not None:
        loaded = layer.load_state_dict(state, strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys, "Incomplete current-gate state reconstruction.")
    layer.to(device).eval().requires_grad_(False)
    ready(layer)
    return layer


def run(path):
    p = read(path)
    require(p["status"] == "ready_for_capture" and not p["student_capture_completed"] and not p["cuda_mechanism_completed"],
            "Use the original CPU preparation, not a resumed prototype.")
    check_hashes(p["source_sha256"]); check_hashes(p["input_sha256"]); check_stat(p["data"]["source_weight_stat"])
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One externally assigned GPU is required.")
    output = Path(p["output_root"]) / "run"
    require(not output.exists(), "Never overwrite or automatically retry a prototype.")
    output.mkdir()
    started = time.time()
    deadline = started + p["protocol"]["wall_budget_seconds"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    configure_deterministic_mode(True); set_seed(31); torch.set_num_threads(4)
    report = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "started_unix": started, "deadline_unix": deadline, "preparation_sha256": sha256(path),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "protocol": p["protocol"], "arms": {}}
    report_path = output / "report.json"
    write_json(report_path, report)
    try:
        data, batches, archive = prepare_data(p["base_root"])
        require(data == p["data"], "Runtime CPU data/source replay differs from the prepared contract.")
        initial, cache, manifest = capture(data, batches, archive, output / "student_capture", device)
        frozen_cache = cache_records(cache)
        report["capture"] = {"manifest_sha256": sha256(output / "student_capture/data_manifest.json"), "runtime": manifest["runtime"], "tensors": frozen_cache}
        require(records(initial) == data["expected_gate_state"], "Captured gate is not the original f927 gate.")
        layer = fresh(initial, None, device)
        report["before_original_codes"] = measure(layer, archive, cache, device)
        choices, report["search"] = search_once(layer, archive, deadline=deadline,
            observer=lambda row: logging.info("exact64 %s", row))
        require(report["search"]["initial_payload"] == data["expected_gate_state"], "Search reused an old decoder or old choices.")
        selected = {k: v.detach().cpu().clone() for k, v in layer.state_dict().items()}
        report["before_exact64_codes"] = measure(layer, archive, cache, device)
        torch.save(choices, output / "f927_l18_exact64_choices.pt")
        report["choices_sha256"] = sha256(output / "f927_l18_exact64_choices.pt")
        write_json(report_path, report)
        del layer, choices
        gc.collect(); torch.cuda.empty_cache()
        layer = fresh(initial, None, device)
        report["mechanism"] = train(layer, archive, cache, steps=2, deadline=deadline,
            observer=lambda row: logging.info("mechanism %s", row))
        require(report["mechanism"]["initial_payload"] == data["expected_gate_state"], "Mechanism must start from original f927.")
        estimate = 1000 * max(row["seconds"] for row in report["mechanism"]["trajectory"]) + 60
        report["resource_projection"] = {"two_arm_seconds_from_slower_mechanism_step_plus60": estimate,
            "remaining_seconds": deadline - time.time(), "peak_allocated_bytes": report["mechanism"]["peak_allocated_bytes"],
            "peak_reserved_bytes": report["mechanism"]["peak_reserved_bytes"]}
        write_json(report_path, report)
        require(estimate < deadline - time.time(), "Observed mechanism speed cannot fit the existing budget; stop without extending it.")
        del layer
        gc.collect(); torch.cuda.empty_cache()
        for arm in ARMS:
            layer = fresh(initial, selected if arm == "exact64_codes" else None, device)
            trained = train(layer, archive, cache, steps=500, deadline=deadline,
                observer=lambda row: logging.info("%s %s", arm, row))
            expected = report["search"]["final_payload"] if arm == "exact64_codes" else data["expected_gate_state"]
            require(trained["initial_payload"] == expected, "Formal fitting must discard mechanism updates and keep matched decoder initialization.")
            endpoint = measure(layer, archive, cache, device)
            payload = output / f"f927_l18_{arm}_output_decoder500.pt"
            torch.save({k: v.detach().cpu() for k, v in layer.state_dict().items()}, payload)
            saved = torch.load(payload, map_location="cpu", weights_only=True, mmap=True)
            rebuilt = fresh(initial, saved, device)
            require(records(rebuilt) == trained["final_payload"], "Saved gate failed fresh strict state reconstruction.")
            with torch.no_grad():
                require(torch.equal(rebuilt._decode_weight(torch.bfloat16), layer._decode_weight(torch.bfloat16)),
                        "Fresh strict saved gate changed the native full BF16 deployment weight.")
            report["arms"][arm] = {"training": trained, "endpoint": endpoint,
                "versus_same_codes_initial": paired(report["before_" + arm], endpoint),
                "versus_original_f927": paired(report["before_original_codes"], endpoint),
                "state_file": str(payload), "state_sha256": sha256(payload),
                "fresh_strict_state_and_native_decode_exact": True}
            write_json(report_path, report)
            del layer, rebuilt, saved
            gc.collect(); torch.cuda.empty_cache()
        a, b = (report["arms"][arm] for arm in ARMS)
        require(a["training"]["settings"] == b["training"]["settings"], "Paired fitting settings changed.")
        require(records(initial) == data["expected_gate_state"] and cache_records(cache) == frozen_cache,
                "Immutable source gate or captured student context changed.")
        report["between_arms"] = paired(a["endpoint"], b["endpoint"])
        check_hashes(p["source_sha256"]); check_hashes(p["input_sha256"]); check_stat(data["source_weight_stat"])
        report.update(status="completed", finished_unix=time.time(), source_and_context_unchanged=True)
        write_json(report_path, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", finished_unix=time.time())
        write_json(report_path, report)
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
