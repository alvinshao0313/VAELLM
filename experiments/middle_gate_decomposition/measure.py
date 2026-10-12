"""Locate the Qwen middle-gate error reversal in four existing endpoints."""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from experiments.gate_error_decomposition.measure import NODES, finalize_node, node_statistics
from experiments.joint_reconstruction_probe.core import decode, rebuild, require, restore
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.run import load_archive
from train_utils.utils import configure_deterministic_mode, set_seed


NAME = "model.layers.18.mlp.gate_proj"
ARMS = ("native2k", "exact64", "decoder500", "exact64_decoder500")


def read(path):
    return json.loads(Path(path).read_text())


def load_arm(initial, case, arm):
    layer = rebuild(initial, "gate_proj")
    if arm == "native2k":
        expected = case["neighbor"]["initial_payload"]
    elif arm == "exact64":
        choices = torch.load(case["choices_file"], map_location="cpu", weights_only=True)
        require(choices.dtype == torch.uint8 and choices.shape == (initial["rotated_weight"].numel() // 32,)
                and int(choices.max()) <= 64, "The saved complete neighbor choices changed.")
        for stage in range(2):
            selected = (choices > 32 * stage) & (choices <= 32 * (stage + 1))
            rows = torch.arange(len(choices))[selected]
            bits = choices[selected].long() - 1 - 32 * stage
            bank = layer.get_stage_part_vq_storage(stage)
            bank[rows, 0, bits // 8] ^= (1 << (bits % 8)).to(torch.uint8)
        expected = case["neighbor"]["final_payload"]
    else:
        info = case["arms"][arm]
        layer.load_state_dict(torch.load(info["state_file"], map_location="cpu", weights_only=True, mmap=True), strict=True)
        expected = info["training"]["final_payload"]
    require({k: tensor_record(v) for k, v in layer.state_dict().items()} == expected,
            f"The actual {arm} state differs from its completed report.")
    layer.eval().requires_grad_(False)
    layer._build_parallel_stage_decode_plan()
    return layer, expected


def old_metrics(case, arm, split):
    if split == "confirmation":
        return case["arms"][arm]["confirmation"] if arm in case["arms"] else None
    result = (case["baseline_old_dev"] if arm == "native2k" else
              case["exact_before_fit_old_dev"] if arm == "exact64" else case["arms"][arm]["old_dev"])
    return result["splits"][split]


def prepare(args):
    base, output = args.middle_root.resolve(), args.output_root.resolve()
    require(not output.exists(), "Use a new output directory.")
    prep, report, review = (read(base / path) for path in ("preparation.json", "run/report.json", "review.json"))
    require(report["status"] == "completed" and review["status"] == "verified"
            and review["report_sha256"] == sha256(base / "run/report.json")
            and (base / "run.exit").read_text().strip() == "0", "The source run must have a completed, verified result.")
    sources, inputs = dict(prep["source_sha256"]), dict(prep["input_sha256"])
    check_hashes(sources)
    check_hashes(inputs)
    case = report["cases"]["qwen"]["middle"]
    require(case["target_name"] == NAME and prep["models"]["qwen"]["layer_index"] == 18,
            "Only the previously selected Qwen L18 is measured.")
    manifest_path = Path(report["cases"]["qwen"]["confirmation_manifest"])
    manifest = read(manifest_path)
    files = [base / "preparation.json", base / "run/report.json", base / "review.json", manifest_path,
             Path(case["initial_payload_file"]), Path(case["choices_file"]), Path(manifest["archive"]["path"])]
    files.extend(Path(case["arms"][arm]["state_file"]) for arm in ARMS[2:])
    for path in files:
        inputs[str(path)] = sha256(path)
    require(sha256(case["initial_payload_file"]) == case["initial_payload_sha256"]
            and sha256(case["choices_file"]) == case["choices_sha256"]
            and sha256(manifest["archive"]["path"]) == manifest["archive"]["sha256"], "Existing saved input changed.")
    for arm in ARMS[2:]:
        require(sha256(case["arms"][arm]["state_file"]) == case["arms"][arm]["state_sha256"], "Fitted endpoint changed.")
    torch.set_num_threads(4)
    initial = torch.load(case["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
    for arm in ARMS:
        layer, _ = load_arm(initial, case, arm)
        del layer
    archive, _ = load_archive(Path(prep["models"]["qwen"]["middle_dir"]))
    require(archive["layer_index"] == 18 and len(archive["document_lengths"]["heldout"]) == 32,
            "Original middle archive changed.")
    cache = torch.load(manifest["archive"]["path"], map_location="cpu", weights_only=True, mmap=True)
    require(cache["document_lengths"] == manifest["document_lengths"]
            and tensor_record(cache["inputs"][NAME]) == manifest["input_tensors"][NAME], "Confirmation capture changed.")
    for path in (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "gate_error_decomposition/measure.py"):
        ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    output.mkdir(parents=True)
    value = {"status": "cpu_ready", "source_sha256": sources, "input_sha256": inputs,
        "middle_root": str(base), "output_root": str(output), "archive_dir": prep["models"]["qwen"]["middle_dir"],
        "confirmation_manifest": str(manifest_path), "cuda_initialized": False,
        "protocol": {"target": NAME, "arms": list(ARMS), "splits": ["heldout", "confirmation"],
            "documents_per_split": 32, "nodes": list(NODES), "wall_budget_seconds": 120,
            "arithmetic": "BF16 original-coordinate linear/SiLU/up-product/down, FP32 residual and FP64 statistics.",
            "scope": "Read-only existing states. Confirmation documents were already inspected; no new unseen-test claim."},
        "command": [sys.executable, "-m", "experiments.middle_gate_decomposition.measure", "--preparation", str(output / "preparation.json")]}
    write_json(output / "preparation.json", value)
    print(json.dumps({"status": "cpu_ready", "sha256": sha256(output / "preparation.json")}))


@torch.no_grad()
def measure(archive, cache, weights, case, device):
    ref = archive["targets"][NAME]["reference_weight"].to(device)
    up = archive["up_proj_weight"].to(device)
    down = archive["targets"][NAME.replace("gate_proj", "down_proj")]["reference_weight"].to(device)
    result = {}
    for split in ("heldout", "confirmation"):
        inputs = cache["inputs"][NAME] if split == "confirmation" else archive["targets"][NAME]["heldout_inputs"]
        lengths = cache["document_lengths"] if split == "confirmation" else archive["document_lengths"][split]
        docs = {arm: {node: [] for node in NODES} for arm in ARMS}
        moments = {arm: {node: {} for node in NODES} for arm in ARMS}
        offset = 0
        for index, length in enumerate(lengths):
            x = inputs[offset:offset + length].to(device)
            offset += length
            z, u = F.linear(x, ref), F.linear(x, up)
            a = F.silu(z)
            p = a * u
            y = F.linear(p, down)
            for arm in ARMS:
                cz = F.linear(x, weights[arm])
                ca = F.silu(cz)
                cp = ca * u
                cy = F.linear(cp, down)
                for node, pred, target in zip(NODES, (cz, ca, cp, cy), (z, a, p, y), strict=True):
                    stats, sums = node_statistics(pred, target, z if node != "down_output" else None)
                    stats["index"] = index
                    docs[arm][node].append(stats)
                    for key, value in sums.items():
                        if key not in moments[arm][node]:
                            moments[arm][node][key] = value.clone()
                        else:
                            moments[arm][node][key].add_(value)
                expected = old_metrics(case, arm, split)
                if expected is not None:
                    for node, metric in (("gate_linear", "linear_output"), ("down_output", "mlp_block_output")):
                        require(all(docs[arm][node][-1][key] == expected[metric]["documents"][index][key]
                                    for key in ("index", "tokens", "squared_error", "squared_reference", "nmse")),
                                f"Existing {split}/{arm}/{index}/{node} scalar result did not reproduce.")
        require(offset == len(inputs) and len(lengths) == 32, "Document boundary changed.")
        result[split] = {arm: {node: finalize_node(docs[arm][node], moments[arm][node]) for node in NODES} for arm in ARMS}
        logging.info("Finished %s, %d tokens", split, offset)
    return result


def run(path):
    prep = read(path)
    require(prep["status"] == "cpu_ready", "Preparation must pass first.")
    check_hashes(prep["source_sha256"])
    check_hashes(prep["input_sha256"])
    output = Path(prep["output_root"]) / "run"
    require(not output.exists() and torch.cuda.device_count() == 1, "One assigned GPU and new output are required.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output.mkdir()
    started = time.time()
    report = {"status": "running", "started_unix": started, "pid": os.getpid(),
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "preparation_sha256": sha256(path), "protocol": prep["protocol"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    write_json(output / "report.json", report)
    try:
        case = read(Path(prep["middle_root"]) / "run/report.json")["cases"]["qwen"]["middle"]
        initial = torch.load(case["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
        archive, _ = load_archive(Path(prep["archive_dir"]))
        manifest = read(prep["confirmation_manifest"])
        cache = torch.load(manifest["archive"]["path"], map_location="cpu", weights_only=True, mmap=True)
        weights = {}
        for arm in ARMS:
            layer, expected = load_arm(initial, case, arm)
            layer = layer.to(device)
            weights[arm] = restore(archive["targets"][NAME], NAME, decode(layer, deployment=True), device).to(device)
            require({k: tensor_record(v) for k, v in layer.state_dict().items()} == expected,
                    "Read-only native decode changed persistent state.")
            del layer
        indices = archive["targets"][NAME]["protected_input_indices"].to(device)
        require(all(torch.equal(weight[:, indices], weights["native2k"][:, indices]) for weight in weights.values()),
                "Protected columns changed between endpoints.")
        with torch.autocast("cuda", enabled=False):
            report["measurements"] = measure(archive, cache, weights, case, device)
        check_hashes(prep["source_sha256"])
        check_hashes(prep["input_sha256"])
        report.update(status="completed", elapsed_seconds=time.time() - started,
            old_endpoint_scalars_exact=True, frozen_state_exact=True, protected_columns_exact=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_unix"] = time.time()
        write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preparation", type=Path)
    parser.add_argument("--middle-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.middle_root is not None and args.output_root is not None, "Preparation requires both directories.")
        prepare(args)
    else:
        run(args.preparation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
