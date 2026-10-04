"""Compare three fixed local-decoder recipes on four real Qwen3 projections."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from experiments.linear_output.calibration import capture_inputs
from experiments.linear_output.objectives import linear_output_mse
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_decoder import optimize_local_decoder
from litebsq.vae_linear import VAELinear
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


BASE_MODEL = "/home/star/models/Qwen/Qwen3-8B"
TARGETS = (
    "model.layers.3.self_attn.v_proj",
    "model.layers.18.self_attn.o_proj",
    "model.layers.9.mlp.gate_proj",
    "model.layers.27.mlp.down_proj",
)
ARMS = (
    {"name": "lr6p25e6_steps64", "lr": 6.25e-6, "steps": 64},
    {"name": "lr1p25e5_steps64", "lr": 1.25e-5, "steps": 64},
    {"name": "lr1p25e5_steps192", "lr": 1.25e-5, "steps": 192},
)
CHUNK_SIZE = 256


def _write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@torch.no_grad()
def _native_document_metrics(layer, reference, inputs, lengths):
    if sum(lengths) != len(inputs) or any(length < 1 for length in lengths):
        raise ValueError("Captured token count does not match the document boundaries.")
    dtype = getattr(layer, "_decoder_compute_dtype", None) or inputs.dtype
    layer.clear_decoded_weight_cache()
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        error = layer._decode_weight(dtype=dtype).float() - reference.float()
        values, offset = [], 0
        for length in lengths:
            loss = linear_output_mse(error, inputs[offset:offset + length],
                                     chunk_size=CHUNK_SIZE, recompute=False)
            values.append(float(loss))
            offset += length
    layer.clear_decoded_weight_cache()
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Non-finite native document output MSE.")
    return {
        "token_mean_mse": sum(value * length for value, length in zip(values, lengths)) / sum(lengths),
        "document_mean_mse": sum(values) / len(values),
        "per_document_mse": values,
        "valid_tokens": sum(lengths),
    }


def _comparison(before, after):
    baseline = before["token_mean_mse"]
    if baseline <= 0:
        raise ValueError("A positive baseline MSE is required to compare relative improvements.")
    ratio = after["token_mean_mse"] / baseline
    return {
        "before": before,
        "after": after,
        "after_before_ratio": ratio,
        "improvement_percent": 100.0 * (1.0 - ratio),
        "improved_documents": sum(a < b for a, b in zip(after["per_document_mse"], before["per_document_mse"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.checkpoint_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "pilot.json"
    device = torch.device("cuda:0")
    started = time.perf_counter()
    report = {
        "purpose": "local_decoder_hyperparameter_pilot_not_downstream_accuracy",
        "status": "running",
        "configuration": {
            "checkpoint_dir": str(source), "output_dir": str(output), "base_model_path": BASE_MODEL,
            "targets": list(TARGETS), "arms": list(ARMS), "device": str(device), "seed": 31,
            "train_documents": 32, "selection_documents": 16, "confirmation_documents": 16,
            "max_length": 256, "chunk_size": CHUNK_SIZE, "teacher_microbatch_size": 1,
            "optimizer": "AdamW", "optimizer_dtype": "float32", "weight_decay": 0.0,
            "betas": [0.9, 0.999], "eps": 1e-8, "lr_schedule": "constant",
            "native_activation_dtype": "torch.bfloat16", "teacher_inputs": True,
            "selection_rule": "minimum equal-Linear mean of selection after/before token-MSE ratios; ties prefer fewer steps then lower LR",
            "confirmation_used_for_selection": False,
            "allow_tf32": False, "deterministic_algorithms": True,
        },
        "code_sha256": {
            relative: hashlib.sha256((Path(__file__).resolve().parents[2] / relative).read_bytes()).hexdigest()
            for relative in (
                "experiments/qwen3_w25/pilot_local_decoder.py",
                "experiments/qwen3_w25/local_decoder.py",
                "experiments/qwen3_w25/calibration.py",
                "experiments/linear_output/calibration.py",
                "experiments/linear_output/objectives.py",
            )
        },
        "targets": {},
    }
    _write_report(report_path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_batch, heldout_batch, manifest = build_calibration_batches(
            tokenizer, seed=31, train_documents=32, heldout_documents=32, max_length=256,
        )
        lengths = [row["valid_tokens"] for row in manifest["heldout"]["documents"]]
        selection_tokens = sum(lengths[:16])
        report["data"] = manifest
        report["data"]["heldout_roles"] = {
            "selection_document_indices": list(range(16)),
            "confirmation_document_indices": list(range(16, 32)),
        }
        phase_start = time.perf_counter()
        model, meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL,
                                                 map_location="cpu", strict=True)
        model.eval().requires_grad_(False)
        for name in TARGETS:
            if name not in meta["compressed_targets"] or not isinstance(model.get_submodule(name), VAELinear):
                raise ValueError(f"Pilot target is not compressed: {name}")
        report["source_checkpoint_id"] = meta["checkpoint_id"]
        report["checkpoint_load_seconds"] = time.perf_counter() - phase_start
        _write_report(report_path, report)

        print("Capturing teacher inputs once for all pilot arms", flush=True)
        phase_start = time.perf_counter()
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundles = {}
        for name in TARGETS:
            reference = teacher.get_submodule(name).weight.detach().cpu().clone()
            train_x = capture_inputs(teacher, name, train_batch, microbatch_size=1).cpu()
            heldout_x = capture_inputs(teacher, name, heldout_batch, microbatch_size=1).cpu()
            if len(train_x) != manifest["train"]["valid_tokens"] or len(heldout_x) != sum(lengths):
                raise RuntimeError(f"Teacher capture lost document token boundaries: {name}")
            if train_x.dtype != torch.bfloat16 or heldout_x.dtype != torch.bfloat16:
                raise ValueError("Pilot requires the existing BF16 teacher activation path.")
            bundles[name] = (reference, train_x, heldout_x)
            print(json.dumps({"captured": name, "train_tokens": len(train_x),
                              "selection_tokens": selection_tokens,
                              "confirmation_tokens": len(heldout_x) - selection_tokens}), flush=True)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        report["teacher_capture_seconds"] = time.perf_counter() - phase_start

        for name in TARGETS:
            layer = model.get_submodule(name).to(device)
            decoder = layer._modules["_parallel_stage_decoder"]
            initial = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
            reference_cpu, train_cpu, heldout_cpu = bundles.pop(name)
            reference = reference_cpu.to(device=device, dtype=torch.float32)
            train_x, heldout_x = train_cpu.to(device), heldout_cpu.to(device)
            selection_x, confirmation_x = heldout_x[:selection_tokens], heldout_x[selection_tokens:]
            baseline_selection = _native_document_metrics(layer, reference, selection_x, lengths[:16])
            baseline_confirmation = _native_document_metrics(layer, reference, confirmation_x, lengths[16:])
            target_report = {
                "source_module": {
                    "weight_shape": [layer.out_features, layer.in_features], "transpose": layer.transpose,
                    "stage_codebook_dims": list(layer.stage_codebook_dims),
                    "stage_codebook_bits": [layer.get_stage_part_vq_spec(stage_idx=stage, part_idx=0)["logical_shape"][-1]
                                            for stage in range(layer.residual_stages)],
                    "decoder_type": decoder.decoder_type, "decoder_hidden_dim": decoder.hidden_dim,
                    "decoder_num_res_blocks": decoder.num_res_blocks,
                    "decoder_norm_type": decoder.norm_type, "decoder_activation_type": decoder.activation_type,
                    "decoder_storage_dtype": str(next(decoder.parameters()).dtype),
                    "weight_rotation_block_size": layer.weight_rotation.block_size,
                },
                "arms": {},
            }
            report["targets"][name] = target_report
            try:
                for arm in ARMS:
                    decoder.load_state_dict(initial, strict=True)
                    layer.clear_decoded_weight_cache()
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    arm_start = time.perf_counter()

                    def progress(row):
                        if row["step"] % 10 == 0 or row["step"] == arm["steps"]:
                            print(json.dumps({"target": name, "arm": arm["name"], **row}), flush=True)

                    metrics = optimize_local_decoder(
                        layer, reference, train_x, selection_x, steps=arm["steps"],
                        lr=arm["lr"], chunk_size=CHUNK_SIZE, progress_callback=progress,
                    )
                    torch.cuda.synchronize(device)
                    optimization_seconds = time.perf_counter() - arm_start
                    selection_after = _native_document_metrics(layer, reference, selection_x, lengths[:16])
                    confirmation_after = _native_document_metrics(layer, reference, confirmation_x, lengths[16:])
                    torch.cuda.synchronize(device)
                    target_report["arms"][arm["name"]] = {
                        "local_metrics": metrics,
                        "native_selection": _comparison(baseline_selection, selection_after),
                        "native_confirmation": _comparison(baseline_confirmation, confirmation_after),
                        "optimization_seconds": optimization_seconds,
                        "elapsed_seconds": time.perf_counter() - arm_start,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    }
                    _write_report(report_path, report)
            finally:
                decoder.load_state_dict(initial, strict=True)
                layer.clear_decoded_weight_cache()
                layer.cpu()
                layer._build_parallel_stage_decode_plan()
            del reference, train_x, heldout_x, selection_x, confirmation_x, initial
            gc.collect()
            torch.cuda.empty_cache()

        scores = {}
        for arm in ARMS:
            ratios = [report["targets"][name]["arms"][arm["name"]]["native_selection"]["after_before_ratio"]
                      for name in TARGETS]
            scores[arm["name"]] = {"mean_selection_ratio": sum(ratios) / len(ratios),
                                    "worst_selection_ratio": max(ratios),
                                    "improved_targets": sum(value < 1.0 for value in ratios)}
        selected = min(ARMS, key=lambda arm: (scores[arm["name"]]["mean_selection_ratio"], arm["steps"], arm["lr"]))
        report.update(status="complete", selection_scores=scores, selected_arm=selected["name"],
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        print(json.dumps({"selected_arm": selected["name"], "selection_scores": scores,
                          "report": str(report_path)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        raise


if __name__ == "__main__":
    main()
