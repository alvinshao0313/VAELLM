"""Run the approved L27/down feedback on/off trial on the fixed 4b6 source."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.block_ldlq import (
    block_ldl, build_noncentered_hessian, prepare_protected_objective, quantize_reverse_blocks,
)
from experiments.qwen3_w25.block_ldlq_model import (
    CODE_KEYS, SOURCE_ID, NativeStageOracle, apply_codes, native_output_metrics, oracle_weight,
    read_codes, rotate_hessian, save_and_verify, validate_layer,
)
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_recovery import BASE_MODEL, capture_shard_inputs
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


TARGET = "model.layers.27.mlp.down_proj"


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _error_metrics(actual, expected):
    diff = actual.double() - expected.double()
    return {"max_absolute": float(diff.abs().max()),
            "relative_rms": float(diff.square().mean().sqrt() / expected.double().square().mean().sqrt())}


def _proxy(weight, reconstructed, hessian):
    error = weight - reconstructed.double()
    return float(((error @ hessian) * error).sum() / weight.shape[0])


@torch.no_grad()
def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the immutable source.")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "pilot.json"
    started = time.perf_counter()
    device = torch.device("cuda:0")
    root = Path(__file__).resolve().parents[2]
    files = ["experiments/qwen3_w25/" + name + ".py" for name in (
        "block_ldlq", "block_ldlq_model", "pilot_block_ldlq", "calibration", "local_recovery", "assemble_categories")]
    files += ["litebsq/vae_linear.py", "litebsq/fused_multistage_decoder.py", "litebsq/bitpack.py",
              "rotation/weight_preprocess.py", "train_utils/checkpoint_v6.py", "train_utils/v6_model_loader.py",
              "train_utils/base_reference.py", "train_utils/checkpoint_bpw.py", "train_utils/distill_data.py"]
    report = {"status": "running", "pid": os.getpid(), "source": str(source), "target": TARGET,
              "configuration": {"feedback_arms": [False, True], "seed": 31, "train_documents": 512,
                                "public_dev_documents": 64, "max_length": 256, "damping_ratio": 0.01,
                                "passes": 1, "candidates_per_vector": 49, "block_size": 16,
                                "hessian_dtype": "float64", "native_dtype": "bfloat16",
                                "public_dev_previously_used_for_selection": True, "external_lora": False},
              "code_sha256": {name: _sha(root / name) for name in files}, "arms": {}}
    _write(path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        model, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True, expected_kind="final_model")
        if meta["checkpoint_id"] != SOURCE_ID or len(meta["compressed_targets"]) != 252:
            raise ValueError("This pilot is bound to the approved complete 4b6 source.")
        model.eval().requires_grad_(False)
        layer = model.get_submodule(TARGET)
        validate_layer(layer)
        original = _load_state(source)
        report["source_checkpoint_id"] = SOURCE_ID
        report["source_meta_sha256"] = _sha(source / "checkpoint_meta.json")
        report["source_budget"] = audit_checkpoint(source, max_bpw=2.5, max_lora_rank=0)
        if not report["source_budget"]["budget_passed"]:
            raise ValueError("Source failed the bit budget.")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train, dev, manifest = build_calibration_batches(
            tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
        previous_path = root / "result/qwen3_8b_w25_20260930/local_bits_recipe_protected/pilot.json"
        previous_data = json.loads(previous_path.read_text(encoding="utf-8"))["data"]
        if manifest != previous_data:
            raise ValueError("Calibration samples/tokens differ from the approved existing 512/64 document set.")
        report["calibration_reference"] = {"path": str(previous_path), "sha256": _sha(previous_path),
                                           "manifest_exact": True}
        report["data"] = manifest
        _write(path, report)
        phase = time.perf_counter()
        print("Capturing original teacher L27/down inputs for 512 train + 64 public dev documents", flush=True)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=device, dtype=torch.bfloat16)
        bundle = capture_shard_inputs(teacher, [TARGET], train, dev)[TARGET]
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        report["capture_seconds"] = time.perf_counter() - phase
        print(json.dumps({"captured_train_tokens": len(bundle["train_inputs"]),
                          "captured_dev_tokens": len(bundle["heldout_inputs"])}), flush=True)
        reference = bundle["reference_weight"]
        layer.to(device)
        initial = read_codes(layer)
        oracle = NativeStageOracle(layer)
        q0 = oracle_weight(oracle, initial)
        native_q0 = layer._decode_compressed_weight(dtype=torch.bfloat16)
        report["initial_oracle_native_parity"] = _error_metrics(q0, native_q0)
        if report["initial_oracle_native_parity"]["relative_rms"] > 1e-6:
            raise RuntimeError("Stage-isolated production oracle differs materially from native joint decode.")
        del native_q0
        lengths = {split: [row["valid_tokens"] for row in manifest[split]["documents"]]
                   for split in ("train", "heldout")}
        report["baseline"] = {split: native_output_metrics(layer, reference, bundle[split + "_inputs"], lengths[split])
                              for split in lengths}
        _write(path, report)
        phase = time.perf_counter()
        hessian = build_noncentered_hessian(bundle["train_inputs"], device=device)
        p = layer.protected_input_indices
        mask = torch.ones(layer.in_features, device=device, dtype=torch.bool)
        mask[p] = False
        c = mask.nonzero().flatten()
        protected = layer._decode_protected_input_weight_rows(dtype=torch.bfloat16, device=device).T.double()
        objective = prepare_protected_objective(reference.to(device=device, dtype=torch.float64), protected, hessian, c, p)
        # Compare objective differences so the fixed completed-square constant
        # cancels, using the actual full input Hessian and protection payload.
        q_original = layer.weight_rotation(q0.double(), inverse=True)
        w_check = reference[:8].to(device=device, dtype=torch.float64)
        q_check = q_original[:8]
        original_costs, completed_costs = [], []
        for q in (q_check, q_check * 1.001):
            e = w_check.clone()
            e[:, c] -= q
            e[:, p] -= protected[:8]
            original_costs.append(((e @ hessian) * e).sum() + objective.damping * e[:, c].square().sum())
            ec = objective.target[:8] - q
            completed_costs.append(((ec @ objective.hessian) * ec).sum())
        delta_original = original_costs[1] - original_costs[0]
        delta_completed = completed_costs[1] - completed_costs[0]
        cross_relative = float((delta_original - delta_completed).abs() / delta_original.abs())
        report["protection_cross_term_relative_error"] = cross_relative
        if cross_relative > 1e-8:
            raise RuntimeError("Fixed INT8 error cross term was not preserved.")
        del q_original, w_check, q_check, e, ec, original_costs, completed_costs
        rotated = layer.weight_rotation(objective.target)
        rotated_hessian = rotate_hessian(objective.hessian, layer.weight_rotation)
        # Verify the actual full geometry using a dense, nonzero deterministic
        # residual; output rotation is orthogonal and must preserve total energy.
        error = objective.target - layer.weight_rotation(q0.double(), inverse=True)
        before = ((error @ objective.hessian) * error).sum()
        er = layer.weight_rotation(error)
        after = ((er @ rotated_hessian) * er).sum()
        relative = float((before - after).abs() / before.abs())
        report["rotation_objective_relative_error"] = relative
        if relative > 1e-10:
            raise RuntimeError("Rotated regularized objective is not equivalent in FP64.")
        del error, er, hessian, protected
        factors = block_ldl(rotated_hessian, block_size=16)
        probe = torch.randn((12032, 4), device=device, dtype=torch.float64)
        middle = factors.lower.T @ probe
        middle = torch.einsum("bij,bjk->bik", factors.diagonal, middle.reshape(752, 16, 4)).reshape(12032, 4)
        residual = factors.lower @ middle - rotated_hessian @ probe
        factor_error = float(residual.norm() / (rotated_hessian @ probe).norm())
        if factor_error > 1e-10:
            raise RuntimeError("Full-size block LDL factorization check failed.")
        report["hessian"] = {"seconds": time.perf_counter() - phase, "damping": objective.damping,
                             "shape": list(rotated_hessian.shape), "factor_probe_relative_error": factor_error,
                             "baseline_proxy": _proxy(rotated, q0, rotated_hessian)}
        del objective, probe, middle, residual
        _write(path, report)
        for feedback in (False, True):
            name = "feedback_on" if feedback else "feedback_off"
            phase = time.perf_counter()
            print(f"Starting {name}: all 752 reverse blocks, 49 candidates per vector", flush=True)
            result = quantize_reverse_blocks(
                rotated, factors, initial, oracle, feedback=feedback,
                progress=lambda value: print(json.dumps({"arm": name, **value}), flush=True))
            apply_codes(layer, result.codes)
            native_q = layer._decode_compressed_weight(dtype=torch.bfloat16)
            parity = _error_metrics(result.reconstructed_weight, native_q)
            if parity["relative_rms"] > 1e-6:
                raise RuntimeError("Selected oracle reconstruction differs from native decoder.")
            arm = {"statistics": result.statistics, "optimization_seconds": time.perf_counter() - phase,
                   "oracle_native_parity": parity, "proxy": _proxy(rotated, native_q, rotated_hessian),
                   "metrics": {split: native_output_metrics(layer, reference, bundle[split + "_inputs"], lengths[split])
                               for split in lengths}}
            del native_q
            layer.to("cpu")
            expected = dict(original)
            for key in CODE_KEYS:
                expected[TARGET + "." + key] = layer.state_dict()[key].detach().clone()
            _assert_state_equal(model.state_dict(), expected, "Only L27 two packed banks may change")
            arm["frozen_tensors_compared"] = len(expected) - 2
            provenance = {"source_checkpoint_id": SOURCE_ID, "target": TARGET, "feedback": feedback,
                          "configuration": report["configuration"], "code_sha256": report["code_sha256"],
                          "data_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}
            arm["export"] = save_and_verify(model, output / name / "final_model", meta, expected, provenance, tokenizer)
            if arm["export"]["bit_budget"]["user_linear_bits"] != report["source_budget"]["user_linear_bits"]:
                raise RuntimeError("Code-only update changed the persistent bit budget.")
            arm["dev_after_before_ratio"] = arm["metrics"]["heldout"]["token_mean_mse"] / report["baseline"]["heldout"]["token_mean_mse"]
            report["arms"][name] = arm
            _write(path, report)
            layer.to(device)
            apply_codes(layer, initial)
            del result, expected
            gc.collect()
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["code_sha256_after"] = {name: _sha(root / name) for name in files}
        if report["code_sha256_after"] != report["code_sha256"]:
            raise RuntimeError("Pinned pilot source changed during execution.")
        report["status"] = "verified"
        report["feedback_dev_improved"] = report["arms"]["feedback_on"]["dev_after_before_ratio"] < 1
        _write(path, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds"] = time.perf_counter() - started
        _write(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
