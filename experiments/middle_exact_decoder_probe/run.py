"""Fixed middle-layer replication and unseen-document endpoint confirmation."""
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

from experiments.exact_bit_neighbor_probe.core import probe
from experiments.full_weight_bits_probe.core import train
from experiments.joint_reconstruction_probe.core import decode, rebuild, require, restore
from experiments.joint_reconstruction_probe.run import comparison
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.middle_exact_decoder_probe.data import prepare_confirmation, capture_confirmation, measure_confirmation
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.metrics import measure
from experiments.rotation_factor_probe.run import load_archive
from experiments.rotation_factor_probe.train import run_matrix
from rotation.weight_preprocess import rotation_spec
from train_utils.utils import configure_deterministic_mode, set_seed


MODELS = (("qwen", 18), ("llama", 16))
L0_ARMS = ("native2k", "decoder500", "exact64_decoder500")
MIDDLE_ARMS = ("decoder500", "exact64_decoder500")


def read(path):
    return json.loads(Path(path).read_text())


def reports(base):
    paths = {"full": base / "full_weight_bits_probe/formal/report.json",
             "exact": base / "exact_bit_neighbor_probe/run/report.json",
             "fitted": base / "exact_bits_decoder_probe/run/report.json"}
    values = {}
    for key, path in paths.items():
        review = read(path.parents[1] / "review.json")
        value = read(path)
        require(value["status"] == "completed" and review["status"] == "verified"
                and review["report_sha256"] == sha256(path), f"Existing {key} result changed.")
        values[key] = value
    return paths, values


def l0_endpoints(base, prior, model):
    full, exact, fitted = (prior[key]["cases"][model] for key in ("full", "exact", "fitted"))
    require(full["baseline"] == exact["before"] and fitted["training"]["steps"] == 500
            and fitted["training"]["stage_final_net_xor"] == [0, 0]
            and fitted["training"]["initial_payload"] == exact["local"]["final_payload"],
            "L0 endpoints do not share the verified original native2k lineage.")
    initial_file = Path(full["initial_payload_file"])
    require(sha256(initial_file) == full["initial_payload_file_sha256"], "Original L0 payload changed.")
    endpoints = {
        "native2k": {"state_file": None, "state_sha256": None,
            "expected_state": exact["local"]["initial_payload"], "old_dev": exact["before"]},
        "decoder500": {"state_file": full["arms"]["decoder"]["local_payload_file"],
            "state_sha256": full["arms"]["decoder"]["local_payload_sha256"],
            "expected_state": full["arms"]["decoder"]["final_payload"],
            "old_dev": full["arms"]["decoder"]["metrics"]},
        "exact64_decoder500": {"state_file": fitted["local_payload_file"],
            "state_sha256": fitted["local_payload_sha256"], "expected_state": fitted["training"]["final_payload"],
            "old_dev": fitted["metrics"]}}
    return {"initial_file": str(initial_file), "initial_sha256": sha256(initial_file), "endpoints": endpoints}


def load_l0(initial, arm, info):
    layer = rebuild(initial, "gate_proj")
    if info["state_file"] is not None:
        require(sha256(info["state_file"]) == info["state_sha256"], "Existing L0 fitted payload changed.")
        layer.load_state_dict(torch.load(info["state_file"], map_location="cpu", weights_only=True, mmap=True), strict=True)
    require({k: tensor_record(v) for k, v in layer.state_dict().items()} == info["expected_state"],
            f"Actual {arm} state differs from its reviewed endpoint.")
    if arm == "decoder500":
        require(all(torch.equal(layer.get_stage_part_vq_storage(s), initial["stage_codes"][s]) for s in range(2)),
                "The original-code control changed codes.")
    layer.eval().requires_grad_(False)
    layer._build_parallel_stage_decode_plan()
    return layer


def prepare(args):
    require(args.base_root is not None and args.output_root is not None, "Preparation requires both directories.")
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not (output / "preparation.json").exists() and not (output / "run").exists(), "Never replace a prepared or active run.")
    reference = read(base / "exact_bits_decoder_probe/preparation.json")
    check_hashes(reference["source_sha256"])
    check_hashes(reference["input_sha256"])
    sources, inputs = dict(reference["source_sha256"]), dict(reference["input_sha256"])
    paths, prior = reports(base)
    for path in paths.values():
        inputs[str(path)] = sha256(path)
        inputs[str(path.parents[1] / "review.json")] = sha256(path.parents[1] / "review.json")
    screen_path = base / "rotation_factor_probe/middle_screen_s2000/screen.json"
    screen = read(screen_path)
    require(screen["status"] == "completed", "The existing middle screen must be complete.")
    inputs[str(screen_path)] = sha256(screen_path)
    models = {}
    torch.set_num_threads(4)
    for model, index in MODELS:
        l0_dir = base / "rotation_factor_probe/data" / model
        middle_dir = base / "rotation_factor_probe/middle_data" / model
        confirmed, batch = prepare_confirmation(l0_dir, middle_dir)
        sources.update(confirmed["source_sha256"])
        inputs.update(confirmed["input_sha256"])
        archive, manifest = load_archive(middle_dir)
        name = f"model.layers.{index}.mlp.gate_proj"
        spec = manifest["protection"][name]["spec"]
        require(archive["layer_index"] == index == manifest["original_config"]["num_hidden_layers"] // 2
                and name in archive["targets"] and spec["transpose"] is False and spec["parallel_parts"] == 1
                and spec["stage_codebook_dims"] == [32, 32] and spec["residual_stages"] == 2
                and all(v["logical_shape"][-1] == 32 for v in spec["stage_vq_weights"])
                and spec["weight_rotation"] == rotation_spec("two_sided", 32, 31, name),
                "Preselected middle gate geometry or actual-name rotation changed.")
        bundle = archive["targets"][name]
        require(bundle["reference_weight"].dtype == torch.bfloat16
                and bundle["protected_input_indices"].dtype == torch.int64
                and bundle["protected_input_indices"].numel() == bundle["protected_input_indices"].unique().numel() == 64,
                "Original middle HF/protection geometry changed.")
        l0 = l0_endpoints(base, prior, model)
        initial = torch.load(l0["initial_file"], map_location="cpu", weights_only=True, mmap=True)
        for arm in L0_ARMS:
            layer = load_l0(initial, arm, l0["endpoints"][arm])
            state_file = l0["endpoints"][arm]["state_file"]
            if state_file is not None:
                inputs[state_file] = sha256(state_file)
            del layer
        inputs[l0["initial_file"]] = l0["initial_sha256"]
        models[model] = {"layer_index": index, "target_name": name, "middle_dir": str(middle_dir),
            "l0_dir": str(l0_dir), "confirmation": confirmed, "l0": l0,
            "middle_original_weight": manifest["tensors"][name]["reference_weight"],
            "middle_rotation": spec["weight_rotation"], "middle_protection": manifest["protection"][name],
            "old_middle_screen_arm": screen["arms"][f"{model}/layer{index}/gate_proj/block32"]}
        del archive, initial, batch
        gc.collect()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    record = {"status": "cpu_ready", "python": sys.executable, "torch": torch.__version__,
        "source_sha256": sources, "input_sha256": inputs, "models": models, "base_root": str(base),
        "output_root": str(output), "cuda_initialized": False,
        "protocol": {"middle_layers": {m: i for m, i in MODELS}, "native_steps_per_stage": 2000,
            "native_seed": 31, "new_middle_initial_payloads": 2, "decoder_steps": 500,
            "decoder_recipe": "Unchanged full_weight_bits_probe.core.train(joint=False), complete matrix, FP32 masters/BF16 compute.",
            "neighbor_search": "Unchanged exact_bit_neighbor_probe.core.probe once per middle matrix, fixed initial decoder, 64 candidates/D32 block.",
            "l0": "Three existing fixed endpoints only; no L0 VAE, neighbor search, or optimizer.",
            "confirmation": "Predeclared new 32 documents, used only at fixed final endpoints; not previously used in this local selection, not a pretraining exclusion claim.",
            "comparison_budget": "Equal 500 decoder updates; candidate has extra one-pass search cost.",
            "wall_budget_seconds": 1200, "estimated_total_seconds": [360, 540]},
        "command": [sys.executable, "-m", "experiments.middle_exact_decoder_probe.run", "--preparation", str(output / "preparation.json")]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": "cpu_ready", "sha256": sha256(output / "preparation.json"),
        "sources": len(sources), "inputs": len(inputs), "models": list(models)}))


def confirmation_comparison(before, after):
    result = {}
    for metric in ("linear_output", "mlp_block_output"):
        left, right = before[metric], after[metric]
        pairs = list(zip(left["documents"], right["documents"], strict=True))
        require(len(pairs) == 32 and all((a["index"], a["tokens"], a["squared_reference"])
                == (b["index"], b["tokens"], b["squared_reference"]) for a, b in pairs),
                "Confirmation document or reference denominator changed between endpoints.")
        result[metric] = {"nmse_after_over_before": right["nmse"] / left["nmse"],
            "improved_documents": sum(b["squared_error"] < a["squared_error"] for a, b in pairs), "documents": 32}
    return result


def measure_dev(archive, name, weight, device, index):
    metrics = measure(archive, name, weight, device)
    metrics["scope"] = f"Original L{index} inputs; gate-only block error holds same-layer up/down at original HF weights."
    return metrics


def run(path):
    preparation = read(path)
    require(preparation["status"] == "cpu_ready", "CPU preparation must pass first.")
    check_hashes(preparation["source_sha256"])
    check_hashes(preparation["input_sha256"])
    output = Path(preparation["output_root"]) / "run"
    require(not output.exists(), "Use a new run; no overwrite or retry.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One assigned GPU is required.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output.mkdir(parents=True)
    started = time.time()
    deadline = started + preparation["protocol"]["wall_budget_seconds"]
    report = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "started_unix": started, "deadline_unix": deadline, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "preparation_sha256": sha256(path), "source_sha256": preparation["source_sha256"], "input_sha256": preparation["input_sha256"],
        "protocol": preparation["protocol"], "cases": {},
        "boundaries": ["Preselected Qwen L18 and Llama L16; gate only, single-matrix native entropy population.",
            "All optimization and search use the original rotated FP32 weight only, no activation/document objective.",
            "Old 64 dev documents are reported separately from the new fixed 32 confirmation documents.",
            "No whole-model checkpoint, full downstream evaluation, algorithm integration, or learning-rate/endpoint selection."]}
    write_json(output / "report.json", report)
    try:
        for model, index in MODELS:
            require(time.time() < deadline, "The total 1200-second budget expired.")
            info = preparation["models"][model]
            confirmation, batch = prepare_confirmation(info["l0_dir"], info["middle_dir"])
            require(confirmation == info["confirmation"], "Prepared confirmation data or model identity changed.")
            cache, manifest = capture_confirmation(confirmation, batch, output / "confirmation" / model, device)
            case = {"model_type": cache["model_type"], "layer_index": index,
                "confirmation_manifest": str(output / "confirmation" / model / "data_manifest.json"),
                "confirmation_manifest_sha256": sha256(output / "confirmation" / model / "data_manifest.json"),
                "l0": {}, "middle": {"arms": {}}}
            report["cases"][model] = case
            write_json(output / "report.json", report)
            l0_archive, _ = load_archive(Path(info["l0_dir"]))
            l0_initial = torch.load(info["l0"]["initial_file"], map_location="cpu", weights_only=True, mmap=True)
            l0_name = "model.layers.0.mlp.gate_proj"
            for arm in L0_ARMS:
                endpoint = info["l0"]["endpoints"][arm]
                layer = load_l0(l0_initial, arm, endpoint).to(device)
                weight = restore(l0_archive["targets"][l0_name], l0_name, decode(layer, deployment=True), device)
                result = measure_confirmation(l0_archive, l0_name, weight, cache["inputs"][l0_name], cache["document_lengths"], device)
                require({k: tensor_record(v) for k, v in layer.state_dict().items()} == endpoint["expected_state"],
                        "Read-only L0 endpoint measurement changed state.")
                case["l0"][arm] = {"confirmation": result, "old_dev_reference": endpoint["old_dev"],
                    "read_only_state_exact": True, "original_coordinate_weight": tensor_record(weight)}
                del layer, weight
            case["l0"]["comparisons"] = {"fitted_exact_vs_control": confirmation_comparison(
                case["l0"]["decoder500"]["confirmation"], case["l0"]["exact64_decoder500"]["confirmation"]),
                "fitted_exact_vs_native2k": confirmation_comparison(case["l0"]["native2k"]["confirmation"], case["l0"]["exact64_decoder500"]["confirmation"])}
            del l0_archive, l0_initial
            gc.collect()
            torch.cuda.empty_cache()
            archive, middle_manifest = load_archive(Path(info["middle_dir"]))
            name = info["target_name"]
            logging.info("Starting %s fresh middle gate native2k", model)
            initial = {}
            baseline, native = run_matrix(archive["targets"][name], name=name, model_path=archive["model_path"],
                full=False, factors={}, steps=2000, seed=31, device=device, diagnostic_out=initial)
            initial_file = output / f"{model}_middle_initial_payload.pt"
            torch.save(initial, initial_file)
            baseline_metrics = measure_dev(archive, name, baseline, device, index)
            case["middle"].update(native=native, initial_payload_file=str(initial_file), initial_payload_sha256=sha256(initial_file),
                capture_sha256=middle_manifest["inputs_sha256"], baseline_old_dev=baseline_metrics,
                rebuilt_vs_prior_native= comparison(info["old_middle_screen_arm"]["metrics"], baseline_metrics),
                rotation=info["middle_rotation"], target_name=name)
            # Build exact codes once from the untouched common native2k decoder.
            search_layer = rebuild(initial, "gate_proj").to(device)
            require(torch.equal(restore(archive["targets"][name], name, decode(search_layer, deployment=True), device), baseline),
                    "Native middle payload does not reconstruct its fresh baseline.")
            logging.info("Starting %s middle fixed-decoder exact64", model)
            _, exact_rotated, choices, neighbor = probe(search_layer, initial["rotated_weight"],
                lambda item: logging.info("%s exact64 %s", model, item))
            exact_banks = [search_layer.get_stage_part_vq_storage(s).detach().cpu().clone() for s in range(2)]
            exact_weight = restore(archive["targets"][name], name, exact_rotated, device)
            choices_file = output / f"{model}_middle_choices.pt"
            torch.save(choices, choices_file)
            case["middle"].update(neighbor=neighbor, choices_file=str(choices_file), choices_sha256=sha256(choices_file),
                exact_before_fit_old_dev=measure_dev(archive, name, exact_weight, device, index))
            del search_layer, exact_rotated, exact_weight
            gc.collect()
            torch.cuda.empty_cache()
            for arm in MIDDLE_ARMS:
                require(time.time() < deadline, "The total 1200-second budget expired.")
                layer = rebuild(initial, "gate_proj")
                if arm == "exact64_decoder500":
                    for stage in range(2):
                        layer.get_stage_part_vq_storage(stage).copy_(exact_banks[stage])
                    layer._build_parallel_stage_decode_plan()
                layer = layer.to(device)
                logging.info("Starting %s middle %s 500 steps", model, arm)
                rotated, trained, trace = train(layer, initial["rotated_weight"], joint=False, steps=500,
                    deadline=deadline, observer=lambda item: logging.info("%s/%s %s", model, arm, item))
                expected_initial = neighbor["initial_payload"] if arm == "decoder500" else neighbor["final_payload"]
                require(trained["initial_payload"] == expected_initial and trained["stage_final_net_xor"] == [0, 0],
                        "Matched fitting initial state or frozen assignment changed.")
                candidate = restore(archive["targets"][name], name, rotated, device)
                protected = archive["targets"][name]["protected_input_indices"]
                require(torch.equal(candidate[:, protected], baseline[:, protected]), "Original protected columns changed.")
                dev = measure_dev(archive, name, candidate, device, index)
                confirmed = measure_confirmation(archive, name, candidate, cache["inputs"][name], cache["document_lengths"], device)
                state_file = output / f"{model}_middle_{arm}_local_payload.pt"
                torch.save({k: v.detach().cpu() for k, v in layer.state_dict().items()}, state_file)
                saved = torch.load(state_file, map_location="cpu", weights_only=True, mmap=True)
                fresh = rebuild(initial, "gate_proj")
                fresh.load_state_dict(saved, strict=True)
                require({k: tensor_record(v) for k, v in fresh.state_dict().items()} == trained["final_payload"],
                        "Saved middle endpoint failed strict native state reconstruction.")
                case["middle"]["arms"][arm] = {"training": trained, "old_dev": dev, "confirmation": confirmed,
                    "state_file": str(state_file), "state_sha256": sha256(state_file), "strict_saved_state_exact": True,
                    "protected_columns_exact": True}
                write_json(output / "report.json", report)
                del layer, rotated, trained, trace, candidate, saved, fresh
                gc.collect()
                torch.cuda.empty_cache()
            a, b = (case["middle"]["arms"][arm] for arm in MIDDLE_ARMS)
            require(a["training"]["settings"] == b["training"]["settings"], "Matched decoder schedules or precision differ.")
            case["middle"]["comparisons"] = {"old_dev": comparison(a["old_dev"], b["old_dev"]),
                "confirmation": confirmation_comparison(a["confirmation"], b["confirmation"])}
            case["status"] = "completed"
            write_json(output / "report.json", report)
            require(sha256(initial_file) == case["middle"]["initial_payload_sha256"], "Training altered original native2k payload.")
            del cache, manifest, batch, archive, initial, exact_banks, baseline, choices
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
