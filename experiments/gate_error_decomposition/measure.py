"""Read-only four-node gate-error measurements from three existing payloads."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from experiments.joint_reconstruction_probe.core import decode, rebuild, require, restore
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.run import load_archive
from train_utils.utils import configure_deterministic_mode, set_seed


NAME = "model.layers.0.mlp.gate_proj"
ARMS = ("native2k", "exact64", "decoder500")
NODES = ("gate_linear", "silu", "up_product", "down_output")
SUMS = ("squared_error", "squared_reference", "squared_prediction", "error_reference_dot",
        "document_bias_sse", "document_centered_sse")


def existing_inputs(exact_root):
    preparation = json.loads((exact_root / "preparation.json").read_text())
    exact = json.loads((exact_root / "run/report.json").read_text())
    full_root = Path(preparation["models"]["qwen"]["initial_payload_file"]).parents[1]
    full = json.loads((full_root / "formal/report.json").read_text())
    for root, report_path in ((exact_root, "run/report.json"), (full_root, "formal/report.json")):
        review = json.loads((root / "review.json").read_text())
        report = json.loads((root / report_path).read_text())
        require(review["status"] == "verified" and report["status"] == "completed"
                and review["report_sha256"] == sha256(root / report_path), "Existing endpoint review changed.")
        exit_path = root / ("run.exit" if root == exact_root else "formal.exit")
        require(exit_path.read_text().strip() == "0", "Existing endpoint did not finish successfully.")
    return preparation, exact, full_root, full


def load_arm(initial, arm, exact_case, full_case):
    layer = rebuild(initial, "gate_proj")
    if arm == "exact64":
        choices = torch.load(exact_case["selection_file"], map_location="cpu", weights_only=True)
        require(choices.dtype == torch.uint8 and choices.shape == (initial["rotated_weight"].numel() // 32,)
                and int(choices.max()) <= 64, "Saved exact choices have changed geometry.")
        for stage in range(2):
            mask = (choices > stage * 32) & (choices <= (stage + 1) * 32)
            rows = torch.arange(len(choices))[mask]
            bit = choices[mask].long() - 1 - stage * 32
            bank = layer.get_stage_part_vq_storage(stage)
            bank[rows, 0, bit // 8] ^= (1 << (bit % 8)).to(torch.uint8)
        expected = exact_case["local"]["final_payload"]
    elif arm == "decoder500":
        state = torch.load(full_case["arms"]["decoder"]["local_payload_file"],
                           map_location="cpu", weights_only=True, mmap=True)
        layer.load_state_dict(state, strict=True)
        require(all(torch.equal(layer.get_stage_part_vq_storage(s), initial["stage_codes"][s])
                    for s in range(2)), "Decoder-only control did not freeze the original codes.")
        expected = full_case["arms"]["decoder"]["final_payload"]
    elif arm == "native2k":
        expected = exact_case["local"]["initial_payload"]
    else:
        raise ValueError(f"Unknown arm: {arm}")
    require({k: tensor_record(v) for k, v in layer.state_dict().items()} == expected,
            f"Real strict {arm} payload differs from its reviewed endpoint.")
    layer.eval().requires_grad_(False)
    layer._build_parallel_stage_decode_plan()
    return layer


def expected_metrics(arm, exact_case, full_case):
    return (exact_case["before"] if arm == "native2k" else exact_case["after"] if arm == "exact64"
            else full_case["arms"]["decoder"]["metrics"])


def prepare(args):
    require(args.exact_root is not None and args.output_root is not None, "Preparation requires both directories.")
    exact_root, output = args.exact_root.resolve(), args.output_root.resolve()
    require(not (output / "preparation.json").exists() and not (output / "run").exists(), "Never overwrite a previous result.")
    prior, exact, full_root, full = existing_inputs(exact_root)
    check_hashes(prior["source_sha256"])
    check_hashes(prior["input_sha256"])
    inputs = dict(prior["input_sha256"])
    for path in (exact_root / "preparation.json", exact_root / "review.json", exact_root / "run/report.json",
                 full_root / "formal/report.json"):
        inputs[str(path)] = sha256(path)
    models = {}
    for model in ("qwen", "llama"):
        ec, fc = exact["cases"][model], full["cases"][model]
        info = prior["models"][model]
        require(ec["before"] == fc["baseline"] and fc["arms"]["decoder"]["steps"] == 500
                and not fc["arms"]["decoder"]["joint"]
                and fc["initial_payload_file_sha256"] == info["initial_payload_sha256"]
                and fc["arms"]["decoder"]["initial_payload"] == ec["local"]["initial_payload"],
                "Positive control is not the same native2k/full-matrix decoder500 source.")
        for path, expected in ((ec["selection_file"], ec["selection_file_sha256"]),
                               (fc["arms"]["decoder"]["local_payload_file"], fc["arms"]["decoder"]["local_payload_sha256"])):
            require(sha256(path) == expected, "A saved measurement endpoint changed.")
            inputs[str(Path(path).resolve())] = expected
        initial = torch.load(info["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
        archive, manifest = load_archive(Path(prior["data_dir"]) / model)
        require(manifest["hidden_act"] == "silu" and manifest["runtime"]["actual_mlp_act_fn"] == "torch.nn.modules.activation.SiLU",
                "Reference activation identity changed.")
        for arm in ARMS:
            layer = load_arm(initial, arm, ec, fc)
            require(all(not p.requires_grad and p.dtype == torch.bfloat16 for p in layer.parameters()),
                    "Measurement must use frozen BF16 decoder storage.")
            expected = expected_metrics(arm, ec, fc)
            for split in ("train", "heldout"):
                for metric in ("linear_output", "mlp_block_output"):
                    a = ec["before"]["splits"][split][metric]["documents"]
                    b = expected["splits"][split][metric]["documents"]
                    require([(x["index"], x["tokens"], x["squared_reference"]) for x in a]
                            == [(x["index"], x["tokens"], x["squared_reference"]) for x in b],
                            "Paired document identity or fixed reference denominator differs.")
            del layer
        models[model] = {"model_type": archive["model_type"], "checkpoint_id": archive["checkpoint_id"],
            "initial_payload_file": info["initial_payload_file"], "initial_payload_sha256": info["initial_payload_sha256"],
            "strict_three_arm_identity_verified": True, "decoder500_original_codes_frozen": True,
            "all_reference_denominators_equal": True, "document_lengths": archive["document_lengths"],
            "gate_weight": manifest["tensors"][NAME]["reference_weight"], "up_weight": manifest["up_proj_weight"],
            "down_weight": manifest["tensors"][NAME.replace("gate_proj", "down_proj")]["reference_weight"]}
        del initial, archive
    source = Path(__file__).resolve()
    ast.parse(source.read_text())
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    record = {"status": "cpu_ready", "python": sys.executable, "torch": torch.__version__,
        "cuda_initialized": False, "models": models, "exact_root": str(exact_root), "full_root": str(full_root),
        "data_dir": prior["data_dir"], "output_root": str(output), "source_sha256": {**prior["source_sha256"], str(source): sha256(source)},
        "input_sha256": inputs, "arms": list(ARMS), "nodes": list(NODES),
        "protocol": "Original fixed 32+32 documents. BF16 linear/SiLU/product/down. FP32 residual then FP64 scalar/moment statistics. No optimization.",
        "statistics": "Per-document token-axis/channelwise error means and centered SSE, plus true whole-split token-weighted means; fixed reference z<0 versus z>=0 groups on gate-shaped nodes.",
        "command": [sys.executable, "-m", "experiments.gate_error_decomposition.measure", "--preparation", str(output / "preparation.json")]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": "cpu_ready", "preparation_sha256": sha256(output / "preparation.json"),
        "sources": len(record["source_sha256"]), "inputs": len(inputs), "models": list(models)}))


def ratio(a, b):
    return a / b if b > 0 else None


def node_statistics(prediction, reference, sign_reference):
    require(prediction.dtype == reference.dtype == torch.bfloat16 and prediction.shape == reference.shape
            and prediction.ndim == 2, "Node arithmetic or geometry changed.")
    delta = (prediction.float() - reference.float()).double()
    ref, pred = reference.double(), prediction.double()
    mu = delta.mean(dim=0)
    values = {"squared_error": delta.square().sum(), "squared_reference": ref.square().sum(),
        "squared_prediction": pred.square().sum(), "error_reference_dot": (delta * ref).sum(),
        "document_bias_sse": len(delta) * mu.square().sum(),
        "document_centered_sse": (delta - mu).square().sum()}
    require(all(bool(torch.isfinite(value)) for value in values.values()), "Nonfinite node statistic.")
    numbers = dict(zip(values, torch.stack(list(values.values())).cpu().tolist()))
    require(math.isclose(numbers["squared_error"], numbers["document_bias_sse"] + numbers["document_centered_sse"],
                         rel_tol=1e-11, abs_tol=1e-15), "Per-document bias/variance identity failed.")
    numbers.update(tokens=len(delta), channels=delta.shape[1], elements=delta.numel(),
        nmse=ratio(numbers["squared_error"], numbers["squared_reference"]),
        prediction_energy_over_reference=ratio(numbers["squared_prediction"], numbers["squared_reference"]),
        error_reference_dot_over_reference_energy=ratio(numbers["error_reference_dot"], numbers["squared_reference"]),
        document_bias_fraction=ratio(numbers["document_bias_sse"], numbers["squared_error"]))
    if sign_reference is not None:
        require(sign_reference.shape == reference.shape, "Fixed gate-sign grouping requires the same shape.")
        numbers["reference_gate_sign_groups"] = {}
        for name, mask in (("negative", sign_reference < 0), ("nonnegative", sign_reference >= 0)):
            error, signal = float(delta[mask].square().sum()), float(ref[mask].square().sum())
            numbers["reference_gate_sign_groups"][name] = {"elements": int(mask.sum()),
                "squared_error": error, "squared_reference": signal, "nmse": ratio(error, signal)}
    moments = {"error_sum": delta.sum(dim=0), "reference_sum": ref.sum(dim=0), "prediction_sum": pred.sum(dim=0)}
    return numbers, moments


def finalize_node(documents, moments):
    totals = {key: sum(doc[key] for doc in documents) for key in SUMS}
    tokens, channels = sum(doc["tokens"] for doc in documents), documents[0]["channels"]
    means = {key: value / tokens for key, value in moments.items()}
    pooled_bias = float(tokens * means["error_sum"].square().sum())
    centered = totals["squared_error"] - pooled_bias
    ref_centered = totals["squared_reference"] - float(tokens * means["reference_sum"].square().sum())
    covariance = totals["error_reference_dot"] - float(tokens * (means["error_sum"] * means["reference_sum"]).sum())
    require(centered >= 0 and ref_centered >= 0, "Invalid whole-split centered energy.")
    result = {**totals, "tokens": tokens, "channels": channels, "documents": documents,
        "nmse": ratio(totals["squared_error"], totals["squared_reference"]),
        "prediction_energy_over_reference": ratio(totals["squared_prediction"], totals["squared_reference"]),
        "error_reference_dot_over_reference_energy": ratio(totals["error_reference_dot"], totals["squared_reference"]),
        "error_reference_cosine": ratio(totals["error_reference_dot"], math.sqrt(totals["squared_error"] * totals["squared_reference"])),
        "document_bias_fraction": ratio(totals["document_bias_sse"], totals["squared_error"]),
        "pooled_token_channel_mean_error_sse": pooled_bias,
        "pooled_centered_error_sse": centered, "pooled_bias_fraction": ratio(pooled_bias, totals["squared_error"]),
        "pooled_mean_error_rms_over_channels": math.sqrt(pooled_bias / (tokens * channels)),
        "pooled_reference_centered_sse": ref_centered,
        "pooled_error_reference_centered_dot": covariance,
        "pooled_error_reference_centered_correlation": ratio(covariance, math.sqrt(centered * ref_centered))}
    if "reference_gate_sign_groups" in documents[0]:
        result["reference_gate_sign_groups"] = {}
        for name in ("negative", "nonnegative"):
            group = {key: sum(doc["reference_gate_sign_groups"][name][key] for doc in documents)
                     for key in ("elements", "squared_error", "squared_reference")}
            group["nmse"] = ratio(group["squared_error"], group["squared_reference"])
            result["reference_gate_sign_groups"][name] = group
    return result


@torch.no_grad()
def measure_model(archive, weights, ec, fc, device):
    reference = archive["targets"][NAME]["reference_weight"].to(device)
    up = archive["up_proj_weight"].to(device)
    down = archive["targets"][NAME.replace("gate_proj", "down_proj")]["reference_weight"].to(device)
    result = {arm: {"splits": {}} for arm in ARMS}
    with torch.autocast("cuda", enabled=False):
        for split in ("train", "heldout"):
            docs = {arm: {node: [] for node in NODES} for arm in ARMS}
            moments = {arm: {node: {} for node in NODES} for arm in ARMS}
            crossings = {arm: [] for arm in ARMS}
            inputs, offset = archive["targets"][NAME][split + "_inputs"], 0
            for index, length in enumerate(archive["document_lengths"][split]):
                x = inputs[offset:offset + length].to(device)
                offset += length
                z = F.linear(x, reference)
                activation = F.silu(z)
                u = F.linear(x, up)
                product = activation * u
                output = F.linear(product, down)
                dense = (z, activation, product, output)
                for arm in ARMS:
                    cz = F.linear(x, weights[arm])
                    ca = F.silu(cz)
                    cp = ca * u
                    cy = F.linear(cp, down)
                    crossings[arm].append({"index": index, "tokens": length, "elements": z.numel(),
                        "negative_to_nonnegative": int(((z < 0) & (cz >= 0)).sum()),
                        "nonnegative_to_negative": int(((z >= 0) & (cz < 0)).sum())})
                    for node, actual, expected in zip(NODES, (cz, ca, cp, cy), dense, strict=True):
                        stats, sums = node_statistics(actual, expected, z if node != "down_output" else None)
                        stats["index"] = index
                        docs[arm][node].append(stats)
                        for key, value in sums.items():
                            if key not in moments[arm][node]:
                                moments[arm][node][key] = value.clone()
                            else:
                                moments[arm][node][key].add_(value)
                    old = expected_metrics(arm, ec, fc)["splits"][split]
                    for node, metric in (("gate_linear", "linear_output"), ("down_output", "mlp_block_output")):
                        observed, prior = docs[arm][node][-1], old[metric]["documents"][index]
                        require(all(observed[key] == prior[key] for key in ("index", "tokens", "squared_error", "squared_reference", "nmse")),
                                f"Existing exact {arm}/{split}/{index}/{node} result did not reproduce.")
                if index in (0, 31):
                    logging.info("measured %s document %d / 32", split, index + 1)
            require(offset == len(inputs), "Document/token boundaries changed.")
            for arm in ARMS:
                result[arm]["splits"][split] = {"nodes": {node: finalize_node(docs[arm][node], moments[arm][node]) for node in NODES},
                    "gate_zero_crossings": crossings[arm]}
    comparisons = {}
    for arm in ARMS[1:]:
        comparisons[arm] = {}
        for split in ("train", "heldout"):
            comparisons[arm][split] = {}
            for node in NODES:
                before = result["native2k"]["splits"][split]["nodes"][node]
                after = result[arm]["splits"][split]["nodes"][node]
                comparisons[arm][split][node] = {"sse_after_over_before": after["squared_error"] / before["squared_error"],
                    "improved_documents": sum(b["squared_error"] < a["squared_error"] for a, b in zip(before["documents"], after["documents"], strict=True)),
                    "documents": 32}
    return {"arms": result, "comparisons": comparisons, "old_z_and_y_every_document_exact": True}


def run(path):
    preparation = json.loads(path.read_text())
    require(preparation["status"] == "cpu_ready", "CPU preparation is required.")
    check_hashes(preparation["source_sha256"])
    check_hashes(preparation["input_sha256"])
    prior, exact, _, full = existing_inputs(Path(preparation["exact_root"]))
    output = Path(preparation["output_root"]) / "run"
    require(not output.exists(), "Never overwrite a run.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One assigned CUDA GPU is required.")
    configure_deterministic_mode(True)
    set_seed(31)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output.mkdir(parents=True)
    report = {"status": "running", "pid": os.getpid(), "argv": sys.argv,
        "process_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split(")", 1)[1].split()[19]),
        "started_unix": time.time(), "preparation_sha256": sha256(path), "source_sha256": preparation["source_sha256"],
        "input_sha256": preparation["input_sha256"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cases": {},
        "boundaries": ["Read-only three existing endpoints; no training, gradient, code search, selection, or full-model export.",
            "The statistical token-axis mean error is not an omitted Linear bias or a proposed bias correction.",
            "Document-centered energy and pooled token-weighted centered energy are reported separately.",
            "BF16 node arithmetic matches the previous measurement. FP64 statistics do not redefine deployed arithmetic.",
            "Stage changes locate where local error worsens; they do not establish a downstream causal explanation."]}
    write_json(output / "report.json", report)
    try:
        for model in ("qwen", "llama"):
            torch.cuda.reset_peak_memory_stats(device)
            info, ec, fc = prior["models"][model], exact["cases"][model], full["cases"][model]
            initial = torch.load(info["initial_payload_file"], map_location="cpu", weights_only=True, mmap=True)
            archive, manifest = load_archive(Path(preparation["data_dir"]) / model)
            weights, identity = {}, {}
            for arm in ARMS:
                layer = load_arm(initial, arm, ec, fc).to(device)
                identity[arm] = {key: tensor_record(value) for key, value in layer.state_dict().items()}
                weights[arm] = restore(archive["targets"][NAME], NAME, decode(layer, deployment=True), device).to(device)
                require({key: tensor_record(value) for key, value in layer.state_dict().items()} == identity[arm], "Decode mutated a payload.")
                del layer
            protected = archive["targets"][NAME]["protected_input_indices"].to(device)
            require(all(torch.equal(weights[arm][:, protected], weights["native2k"][:, protected]) for arm in ARMS),
                    "The fixed protection columns differ between endpoints.")
            before_weight_hash = {arm: tensor_record(value) for arm, value in weights.items()}
            logging.info("Starting %s three frozen endpoints", model)
            observed = measure_model(archive, weights, ec, fc, device)
            require(before_weight_hash == {arm: tensor_record(value) for arm, value in weights.items()}, "Measurement changed a matrix.")
            report["cases"][model] = {"status": "completed", "model_type": archive["model_type"],
                "capture_sha256": manifest["inputs_sha256"], "decoded_weight_sha256": before_weight_hash,
                "persistent_payloads": identity, "protected_columns_exact": True, **observed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            write_json(output / "report.json", report)
            del archive, initial, weights
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
    parser.add_argument("--exact-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    prepare(args) if args.prepare else run(args.preparation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
