"""Paired reconstruction measurements on fixed original-model activations."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def squared_error(prediction: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    delta = prediction.float() - reference.float()
    return float(delta.double().square().sum()), float(reference.double().square().sum())


def aggregate(documents: list[dict]) -> dict:
    error = sum(item["squared_error"] for item in documents)
    signal = sum(item["squared_reference"] for item in documents)
    if signal <= 0:
        raise ValueError("A zero reference signal cannot define normalized error.")
    return {"nmse": error / signal, "squared_error": error, "squared_reference": signal,
            "document_mean_nmse": sum(item["nmse"] for item in documents) / len(documents),
            "documents": documents}


@torch.no_grad()
def measure(archive: dict, name: str, reconstruction: torch.Tensor, device: torch.device) -> dict:
    bundle = archive["targets"][name]
    reference = bundle["reference_weight"]
    if reconstruction.shape != reference.shape or reconstruction.dtype != torch.bfloat16:
        raise ValueError("Measurement requires the original-shape BF16 reconstruction.")
    # FP64 sums of FP32 differences preserve small errors without whole-matrix FP64 copies.
    weight_error, weight_signal = 0.0, 0.0
    for start in range(0, reference.shape[0], 256):
        error, signal = squared_error(reconstruction[start:start + 256], reference[start:start + 256])
        weight_error += error
        weight_signal += signal
    result = {"weight_nmse": weight_error / weight_signal, "splits": {},
              "arithmetic": "BF16 linear and SiLU outputs; FP32 differences with FP64 sums.",
              "scope": "Original L0 inputs; gate-only block error holds up/down at original HF weights."}
    original, candidate = reference.to(device), reconstruction.to(device)
    gate = name.endswith("gate_proj")
    if gate:
        up = archive["up_proj_weight"].to(device)
        down = archive["targets"][name.replace("gate_proj", "down_proj")]["reference_weight"].to(device)
    for split in ("train", "heldout"):
        inputs = bundle[split + "_inputs"]
        lengths = archive["document_lengths"][split]
        if sum(lengths) != inputs.shape[0] or len(lengths) != 32:
            raise ValueError("The fixed 32-document activation boundaries changed.")
        linear_documents, block_documents, offset = [], [], 0
        for index, length in enumerate(lengths):
            x = inputs[offset:offset + length].to(device)
            offset += length
            dense_output, compressed_output = F.linear(x, original), F.linear(x, candidate)
            error, signal = squared_error(compressed_output, dense_output)
            if signal <= 0:
                raise ValueError("A document has zero projection signal.")
            linear_documents.append({"index": index, "tokens": length, "squared_error": error,
                                     "squared_reference": signal, "nmse": error / signal})
            if gate:
                up_output = F.linear(x, up)
                dense_block = F.linear(F.silu(dense_output) * up_output, down)
                compressed_block = F.linear(F.silu(compressed_output) * up_output, down)
                error, signal = squared_error(compressed_block, dense_block)
                if signal <= 0:
                    raise ValueError("A document has zero MLP output signal.")
                block_documents.append({"index": index, "tokens": length, "squared_error": error,
                                        "squared_reference": signal, "nmse": error / signal})
        result["splits"][split] = {"tokens": offset, "linear_output": aggregate(linear_documents),
            "mlp_block_output": aggregate(block_documents if gate else linear_documents)}
    return result


def paired_comparison(block: dict, full: dict) -> dict:
    result = {"weight_nmse_full_over_block32": full["weight_nmse"] / block["weight_nmse"], "splits": {}}
    for split in ("train", "heldout"):
        result["splits"][split] = {}
        for metric in ("linear_output", "mlp_block_output"):
            left, right = block["splits"][split][metric], full["splits"][split][metric]
            pairs = list(zip(left["documents"], right["documents"], strict=True))
            if any((a["index"], a["tokens"]) != (b["index"], b["tokens"]) for a, b in pairs):
                raise ValueError("Paired document identity changed.")
            result["splits"][split][metric] = {
                "nmse_full_over_block32": right["nmse"] / left["nmse"],
                "full_better_documents": sum(b["nmse"] < a["nmse"] for a, b in pairs),
                "documents": len(pairs),
            }
    return result
