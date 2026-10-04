"""Measure the fixed block32 source through canonical Sparse Bit calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.qwen3_w25.local_bits import _digest
from litebsq.vae_linear import VAELinear
from sparse_bit_tuning.config import active_count
from sparse_bit_tuning.coordinates import calibrate_bank_scale
from sparse_bit_tuning.module import BankSpec
from train_utils.checkpoint_v6 import _atomic_write_json
from train_utils.config.configs import AuxTrainableConfig
from train_utils.distill_precision import configure_distill_precision
from train_utils.model_level_trainables import build_model_level_trainable_selection
from train_utils.v6_model_loader import load_v6_model_checkpoint


SOURCE_ID = "ab0280cf-3637-47e3-b771-074b74532a54"
SOURCE_STATE_SHA = "d6a5dfebb4eaeddfa7c7112d59f3601097e2a31ca91be01c0dd9319cb937b2a2"
SOURCE_META_SHA = "dd39c4b3163441434a3f7d5c98a86e4e8a5542eaa4b2d2c53ec3cbd3ea143907"
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
CODE_FILES = (
    "experiments/qwen3_w25/calibrate_decoder_sensitivity.py",
    "experiments/qwen3_w25/local_bits.py", "sparse_bit_tuning/coordinates.py",
    "sparse_bit_tuning/module.py", "sparse_bit_tuning/config.py",
    "litebsq/bit_sensitivity.py", "litebsq/bitpack.py", "litebsq/packed_bit_linear.py",
    "litebsq/autoencoder.py", "litebsq/parallel_layers.py", "litebsq/vae_linear.py",
    "train_utils/distill_precision.py", "train_utils/model_level_trainables.py",
    "train_utils/decoder_execution.py", "train_utils/v6_model_loader.py",
    "train_utils/checkpoint_v6.py", "rotation/model_utils.py",
)


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summary(values):
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)}


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    project = Path(__file__).resolve().parents[2]
    started = time.perf_counter()
    source_files = {path.name: file_sha(path) for path in sorted(source.iterdir()) if path.is_file()}
    if (source_files.get("pytorch_model.bin") != SOURCE_STATE_SHA
            or source_files.get("checkpoint_meta.json") != SOURCE_META_SHA):
        raise ValueError("The fixed block32 source file identity changed.")
    code_before = {name: file_sha(project / name) for name in CODE_FILES}
    meta = json.loads((source / "checkpoint_meta.json").read_text())
    if meta["checkpoint_id"] != SOURCE_ID or meta["converted_module_count"] != 252 or meta["lora_config"] is not None:
        raise ValueError("Expected the complete zero-LoRA block32 source.")
    for item in meta["converted_modules"]:
        if (item["residual_stages"] != 2 or item["parallel_parts"] != 1 or item["codebook_dim"] != 32
                or item["weight_rotation"]["block_size"] != 32
                or any(d["hidden_dim"] != 128 or d["num_res_blocks"] != 0
                       or d["decoder_type"] != "symmetric" for d in item["stage_decoders"])):
            raise ValueError(f"Unexpected fixed-source topology: {item['name']}")
    report = {"status": "running", "source_dir": str(source), "source_checkpoint_id": SOURCE_ID,
              "source_files_sha256_before": source_files, "code_sha256_before": code_before,
              "kernel_status": "existing packed_bit_linear kernel; proposed decoder kernel fix not applied",
              "method": "sparse_bit_tuning.coordinates.calibrate_bank_scale",
              "decoder_parameter_dtype": "torch.float32", "compute_dtype": "torch.bfloat16",
              "rows_per_bank": 256, "training": False, "teacher_created": False,
              "score_or_optimizer_created": False, "banks": []}
    model = None
    work_before = None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly the assigned GPU for canonical CUDA calibration.")
        device = torch.device("cuda:0")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(device)
        report["device_name"] = torch.cuda.get_device_name(device)
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True,
                                                        expected_kind="final_model")
        if loaded_meta["checkpoint_id"] != SOURCE_ID:
            raise ValueError("Loaded source UUID differs.")
        targets = [(name, module) for name, module in model.named_modules() if isinstance(module, VAELinear)]
        expected = {f"model.layers.{layer}.{name}" for layer in range(36) for name in PROJECTIONS}
        if {name for name, _ in targets} != expected:
            raise ValueError("Expected all 252 projections exactly once.")
        decoder_names = {name + "._parallel_stage_decoder." + key for name, module in targets
                         for key in module._parallel_stage_decoder.state_dict()}
        state = model.state_dict()
        source_keys = set(state)
        source_digest = _digest(state, set())
        nondecoder_digest = _digest(state, decoder_names)
        original_decoders = {name: state[name].clone() for name in decoder_names}
        if any(value.dtype != torch.bfloat16 for value in original_decoders.values()):
            raise ValueError("Expected BF16 decoder source values.")
        selection = build_model_level_trainable_selection(
            model, aux=AuxTrainableConfig(), compressed_modules=targets, dense_target_modules=(),
            rank=8, alpha=16.0, dropout=0.0, fp32_components=("decoder",),
            train_decoder=True, train_lora=False, decoder_execution_mode="decoder_sparse_bit", freeze=True)
        configure_distill_precision(selection, components=("decoder",),
                                    training_args=SimpleNamespace(bf16=True, fp16=False))
        state = model.state_dict()
        if set(state) != source_keys or _digest(state, decoder_names) != nondecoder_digest:
            raise RuntimeError("Precision preparation changed nondecoder persistent state.")
        if any(state[name].dtype != torch.float32 or not torch.equal(state[name], value.float())
               for name, value in original_decoders.items()):
            raise RuntimeError("BF16-to-FP32 decoder promotion was not exact.")
        report["precision_preparation"] = {"source_state_sha256": source_digest,
            "nondecoder_state_sha256": nondecoder_digest, "promoted_tensors": len(decoder_names),
            "promoted_parameters": sum(p.numel() for p in selection.decoder_parameters.values()),
            "decoder_values_exact": True, "nondecoder_bytes_unchanged": True}
        del original_decoders, state
        model.eval().requires_grad_(False)
        work_before = _digest(model.state_dict(), set())
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        report["working_state_sha256_before"] = work_before
        calibration_started = time.perf_counter()
        for name, module in targets:
            for stage in range(2):
                shape = tuple(module.get_stage_part_vq_spec(stage, 0)["logical_shape"])
                n_bits = math.prod(shape)
                key = f"{name}|stage={stage}|part=0"
                spec = BankSpec(key, name, stage, 0, shape, n_bits, active_count(n_bits, 0.01), device)
                spec.validate()
                scale = float(calibrate_bank_scale(module, spec, compute_dtype=torch.bfloat16))
                if not math.isfinite(scale) or scale <= 0:
                    raise RuntimeError(f"Invalid canonical scale: {key}")
                report["banks"].append({"key": key, "module": name, "category": name.rsplit(".", 1)[1],
                    "layer": int(name.split(".")[2]), "stage": stage, "part": 0,
                    "logical_shape": list(shape), "rows": min(256, shape[0]), "s": scale,
                    "initial_margin": scale / 2})
            if len(report["banks"]) % 14 == 0:
                print(json.dumps({"banks_complete": len(report["banks"]), "last_module": name,
                                  "elapsed_seconds": time.perf_counter() - calibration_started}), flush=True)
        torch.cuda.synchronize(device)
        report["calibration_seconds"] = time.perf_counter() - calibration_started
        if len(report["banks"]) != 504 or len({row["key"] for row in report["banks"]}) != 504:
            raise RuntimeError("Calibration did not cover exactly 504 unique banks.")
        if not torch.equal(torch.get_rng_state(), cpu_rng) or not torch.equal(torch.cuda.get_rng_state(device), cuda_rng):
            raise RuntimeError("Canonical calibration changed RNG state.")
        groups = {"all": [row["s"] for row in report["banks"]]}
        for category in sorted({row["category"] for row in report["banks"]}):
            for stage in range(2):
                groups[f"{category}/stage{stage}"] = [row["s"] for row in report["banks"]
                                                     if row["category"] == category and row["stage"] == stage]
        report["scale_summary"] = {name: summary(values) for name, values in groups.items()}
        report["constant_step_margin_reference"] = {str(round_steps): summary(
            [row["initial_margin"] / round_steps for row in report["banks"]]) for round_steps in (20, 128, 256)}
        report["reference_boundary"] = "s/(2R) is only a constant-direction coordinate displacement reference; it does not measure task gradients or establish an effective global LR/flip rate."
        report["rng_unchanged"] = True
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["source_files_sha256_after"] = {name: file_sha(source / name) for name in source_files}
        report["source_files_unchanged"] = report["source_files_sha256_after"] == source_files
        report["code_sha256_after"] = {name: file_sha(project / name) for name in CODE_FILES}
        report["code_unchanged"] = report["code_sha256_after"] == code_before
        if model is not None and work_before is not None:
            report["working_state_sha256_after"] = _digest(model.state_dict(), set())
            report["working_state_bytes_unchanged"] = report["working_state_sha256_after"] == work_before
        if torch.cuda.is_initialized():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["elapsed_seconds"] = time.perf_counter() - started
        checks = [report["source_files_unchanged"], report["code_unchanged"], report.get("working_state_bytes_unchanged", False)]
        if not all(checks):
            report["status"] = "failed"
        _atomic_write_json(report, str(output))
    if report["status"] != "complete":
        raise RuntimeError("Calibration persistence or code identity audit failed.")
    print(json.dumps({key: report[key] for key in ("status", "scale_summary", "calibration_seconds",
        "elapsed_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    run(args.checkpoint_dir, args.output_json)


if __name__ == "__main__":
    main()
