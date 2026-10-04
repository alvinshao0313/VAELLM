"""Two-shard packed-bit LR pilot on the fixed no-rotation INT8-protected base."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.bit_pilot import _json_digest, _restore, _trajectory, pilot_document_plan
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_bits import _digest, calibrate_local_bits_cpu, optimize_local_bits
from experiments.qwen3_w25.local_decoder import _validate_payload_profile
from experiments.qwen3_w25.local_recovery import TARGETS as ALL_TARGETS, PROJECTION_SHAPES, capture_shard_inputs
from experiments.qwen3_w25.pilot_local_decoder import BASE_MODEL, CHUNK_SIZE, TARGETS, _comparison, _native_document_metrics, _write_report
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from litebsq.vae_linear import VAELinear
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


SOURCE_CHECKPOINT_ID = "0db180e4-c633-4527-9630-36186b5accd5"
LR_DIVISORS = (64, 128)
STEPS = ROUND_STEPS = 128
CODE_FILES = (
    "experiments/qwen3_w25/bit_recipe_pilot.py", "experiments/qwen3_w25/pilot_bit_recipe.py",
    "experiments/qwen3_w25/bit_pilot.py", "experiments/qwen3_w25/local_bits.py",
    "experiments/qwen3_w25/local_decoder.py", "experiments/qwen3_w25/local_recovery.py",
    "experiments/qwen3_w25/pilot_local_decoder.py", "experiments/qwen3_w25/calibration.py",
    "experiments/linear_output/objectives.py", "litebsq/bit_sensitivity.py",
    "litebsq/vae_linear.py", "litebsq/packed_bit_linear.py", "litebsq/protected_channel_quant.py",
    "sparse_bit_tuning/config.py", "sparse_bit_tuning/manager.py", "sparse_bit_tuning/module.py",
    "sparse_bit_tuning/optimizer.py", "sparse_bit_tuning/coordinates.py",
    "sparse_bit_tuning/packed_ops.py", "sparse_bit_tuning/triton_kernels.py",
)


def shard_targets(index):
    if type(index) is not int or index not in (0, 1):
        raise ValueError("The fixed four-target pilot has shard indices 0 and 1.")
    return TARGETS[index::2]


def configuration():
    return {
        "protocol_version": 1, "targets": list(TARGETS), "lr_divisors": list(LR_DIVISORS),
        "num_shards": 2, "train_documents": 512, "heldout_documents": 64,
        "selection_documents": 32, "repeated_diagnostic_documents": 32,
        "seed": 31, "max_length": 256, "chunk_size": CHUNK_SIZE,
        "steps": STEPS, "round_steps": ROUND_STEPS, "early_stable_round_end": True,
        "source_profile": "no_rotation_int8_input", "optimizer": "adamw", "weight_decay": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8, "proxy_coordinates": "decoder_sensitivity",
        "active_ratio": 1.0, "streaming": False, "decoder_frozen": True,
        "decoder_compute_dtype": "torch.float32", "proxy_gradient_moment_dtype": "torch.float32",
        "native_activation_dtype": "torch.bfloat16", "teacher_microbatch_size": 1,
        "capture_passes_per_shard": 1, "lr_rule": "max(initial CPU decoder-sensitivity scales) / arm divisor",
        "selection_rule": "minimum equal-target mean selection after/before token-MSE ratio; ties prefer smaller LR",
        "repeated_diagnostic_used_for_selection": False,
        "heldout_observation_status": "Both halves were observed in earlier pilots; second half is repeated diagnostic, not new blind confirmation.",
        "checkpoint_written": False, "allow_tf32": False, "deterministic_algorithms": True,
    }


def validate_recipe_source(model, meta):
    if (meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID or meta["checkpoint_kind"] != "final_model"
            or getattr(model.config, "model_type", None) != "qwen3"
            or Path(meta["base_model_path"]).resolve() != Path(BASE_MODEL).resolve()
            or set(meta["compressed_targets"]) != set(ALL_TARGETS)
            or meta["pending_dense_targets"] or meta["skip_targets"] or meta.get("lora_config") is not None
            or meta.get("shared_protected_residual_decoders") or (meta.get("extra_meta") or {}).get("residual_lora")):
        raise ValueError("Expected the fixed complete 252-projection no-LoRA Qwen3-8B base.")
    for name in ALL_TARGETS:
        layer = model.get_submodule(name)
        suffix = ".".join(name.split(".")[3:])
        decoder = getattr(layer, "_parallel_stage_decoder", None)
        if (not isinstance(layer, VAELinear) or (layer.out_features, layer.in_features) != PROJECTION_SHAPES[suffix]
                or layer.residual_stages != 2 or layer.parallel_parts != 1
                or decoder is None or decoder.num_models != 2 or not layer.parallel_stage_decode
                or layer.original_weight is not None or layer.always_use_original or not layer.temporary
                or layer.trainable_decode or getattr(layer, "_sparse_bit_binding", None) is not None
                or layer.bias is not None or layer.has_protected_residual_vae()
                or _validate_payload_profile(layer) != "no_rotation_int8_input"):
            raise ValueError(f"Unsupported source projection: {name}")


def document_plan(train, heldout, manifest):
    original = pilot_document_plan(train, heldout, manifest)
    return {"train_lengths": original["train_lengths"], "heldout_lengths": original["heldout_lengths"],
            "train": original["arms"]["512"], "selection": original["selection"],
            "repeated_diagnostic": original["confirmation"], "manifest_sha256": original["manifest_sha256"]}


def _ratio(comparison, lengths):
    for metrics in (comparison["before"], comparison["after"]):
        values = metrics["per_document_mse"]
        if len(values) != len(lengths) or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Incomplete or invalid per-document MSE.")
        actual = sum(v * n for v, n in zip(values, lengths)) / sum(lengths)
        if metrics["valid_tokens"] != sum(lengths) or not math.isclose(actual, metrics["token_mean_mse"], rel_tol=1e-12):
            raise ValueError("Per-document and token-weighted MSE disagree.")
    actual = _comparison(comparison["before"], comparison["after"])["after_before_ratio"]
    if not math.isclose(actual, comparison["after_before_ratio"], rel_tol=1e-12):
        raise ValueError("Reported relative MSE disagrees with native endpoint metrics.")
    return actual


def select_recipe(target_results, heldout_lengths):
    if set(target_results) != set(TARGETS) or len(heldout_lengths) != 64:
        raise ValueError("Selection requires all four targets and 64 held-out document lengths.")
    if any(set(target["arms"]) != set(map(str, LR_DIVISORS)) for target in target_results.values()):
        raise ValueError("Each target must contain both LR arms.")
    scores = {}
    for divisor in LR_DIVISORS:
        ratios = {name: _ratio(target_results[name]["arms"][str(divisor)]["native_selection"], heldout_lengths[:32])
                  for name in TARGETS}
        scores[str(divisor)] = {"mean_selection_ratio": sum(ratios.values()) / 4,
                               "per_target_ratios": ratios, "improved_targets": sum(v < 1 for v in ratios.values())}
    selected = min(LR_DIVISORS, key=lambda d: (scores[str(d)]["mean_selection_ratio"], -d))
    diagnostic = {name: _ratio(target_results[name]["arms"][str(selected)]["native_repeated_diagnostic"], heldout_lengths[32:])
                  for name in TARGETS}
    return {"selected_lr_divisor": selected, "selection_scores": scores,
            "selected_repeated_diagnostic": {"mean_ratio": sum(diagnostic.values()) / 4,
                "per_target_ratios": diagnostic, "used_for_selection": False, "new_blind_confirmation": False}}


def run_bit_recipe_pilot(checkpoint_dir, output_dir, *, shard_index):
    targets = shard_targets(shard_index)
    source, output = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    meta, audit = _validate_checkpoint(source)
    if meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID or not audit["budget_passed"]:
        raise ValueError("Expected the fixed no-rotation INT8 source within the actual bit budget.")
    output.mkdir(parents=True, exist_ok=False)
    path, device, started = output / "pilot.json", torch.device("cuda:0"), time.perf_counter()
    repo = Path(__file__).resolve().parents[2]
    report = {"status": "running", "purpose": "protected_base_bit_lr_pilot_not_downstream_accuracy",
              "configuration": configuration(), "shard_index": shard_index, "target_names": list(targets),
              "source_checkpoint_dir": str(source), "source_checkpoint_id": meta["checkpoint_id"],
              "source_bit_budget": audit,
              "source_files_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                      for name in ("checkpoint_meta.json", "config.json")},
              "code_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in CODE_FILES},
              "targets": {}}
    _write_report(path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_batch, heldout_batch, manifest = build_calibration_batches(
            tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
        plan = document_plan(train_batch, heldout_batch, manifest)
        report.update(data=manifest, document_plan=plan)
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True)
        model.eval().requires_grad_(False)
        validate_recipe_source(model, loaded_meta)
        calibrations = {name: calibrate_local_bits_cpu(model.get_submodule(name)) for name in targets}
        phase = time.perf_counter()
        print(f"Capturing {targets} together over 512+64 documents", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundles = capture_shard_inputs(teacher, targets, train_batch, heldout_batch)
        torch.cuda.synchronize(device)
        report["teacher_capture_seconds"] = time.perf_counter() - phase
        report["teacher_capture_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["teacher_cache_cpu_bytes"] = sum(t.numel() * t.element_size() for item in bundles.values() for t in item.values())
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        _write_report(path, report)
        lengths, cut = plan["heldout_lengths"], plan["selection"]["valid_tokens"]
        for name in targets:
            bundle = bundles.pop(name)
            layer = model.get_submodule(name).to(device)
            initial = {key: value.detach().cpu().clone() for key, value in layer.state_dict().items()}
            initial_digest = _digest(initial, set())
            reference = bundle["reference_weight"].to(device=device, dtype=torch.float32)
            train_x, heldout_x = bundle["train_inputs"].to(device), bundle["heldout_inputs"].to(device)
            if (len(train_x) != manifest["train"]["valid_tokens"] or len(heldout_x) != manifest["heldout"]["valid_tokens"]
                    or train_x.dtype != torch.bfloat16 or heldout_x.dtype != torch.bfloat16):
                raise ValueError("Captured token inventory or dtype disagrees with the fixed data manifest.")
            selection_x, diagnostic_x = heldout_x[:cut], heldout_x[cut:]
            _restore(layer, initial, initial_digest)
            baseline_selection = _native_document_metrics(layer, reference, selection_x, lengths[:32])
            baseline_diagnostic = _native_document_metrics(layer, reference, diagnostic_x, lengths[32:])
            result = {"source_module_sha256": initial_digest, "cpu_coordinate_calibration": calibrations[name],
                      "source_module": {"weight_shape": [layer.out_features, layer.in_features], "transpose": layer.transpose,
                                        "protected_input_count": layer.protected_input_indices.numel()}, "arms": {}}
            report["targets"][name] = result
            try:
                for divisor in LR_DIVISORS:
                    _restore(layer, initial, initial_digest)
                    lr = max(stage["scale"] for stage in calibrations[name]["stages"]) / divisor
                    arm_start = time.perf_counter()

                    def progress(row):
                        if row["step"] % 10 == 0 or row["step"] == STEPS:
                            print(json.dumps({"target": name, "lr_divisor": divisor, **row}), flush=True)

                    metrics = optimize_local_bits(layer, reference, train_x, heldout_x, steps=STEPS, lr=lr,
                        round_steps=ROUND_STEPS, chunk_size=CHUNK_SIZE, seed=31, progress_callback=progress)
                    selection = _native_document_metrics(layer, reference, selection_x, lengths[:32])
                    diagnostic = _native_document_metrics(layer, reference, diagnostic_x, lengths[32:])
                    torch.cuda.synchronize(device)
                    _restore(layer, initial, initial_digest)
                    result["arms"][str(divisor)] = {"lr_divisor": divisor, "lr": lr, "steps": STEPS,
                        "restored_sha256_before_arm": initial_digest, "restored_sha256_after_arm": _digest(layer.state_dict(), set()),
                        "local_metrics": metrics, "trajectory": _trajectory(metrics),
                        "native_selection": _comparison(baseline_selection, selection),
                        "native_repeated_diagnostic": _comparison(baseline_diagnostic, diagnostic),
                        "elapsed_seconds": time.perf_counter() - arm_start}
                    _write_report(path, report)
            finally:
                _restore(layer, initial, initial_digest)
                result["final_restored_sha256"] = _digest(layer.state_dict(), set())
                layer.cpu()
                layer._build_parallel_stage_decode_plan()
                layer.clear_decoded_weight_cache()
            del bundle, reference, train_x, heldout_x, selection_x, diagnostic_x, initial
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        _write_report(path, report)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)}, elapsed_seconds=time.perf_counter() - started)
        _write_report(path, report)
        raise


def finalize_bit_recipe_pilot(shard_dirs, output_path):
    if len(shard_dirs) != 2:
        raise ValueError("Exactly two completed shards are required.")
    reports = [json.loads((Path(path) / "pilot.json").read_text(encoding="utf-8")) for path in shard_dirs]
    if {report["shard_index"] for report in reports} != {0, 1}:
        raise ValueError("Missing or duplicate shard index.")
    shared = ("configuration", "source_checkpoint_dir", "source_checkpoint_id", "source_files_sha256",
              "source_bit_budget", "code_sha256", "data", "document_plan")
    combined = {}
    for report in reports:
        if (report["status"] != "complete" or report["source_checkpoint_id"] != SOURCE_CHECKPOINT_ID
                or report["configuration"] != configuration() or not report["source_bit_budget"]["budget_passed"]
                or any(report[key] != reports[0][key] for key in shared)
                or set(report["code_sha256"]) != set(CODE_FILES)):
            raise ValueError("Incomplete or mismatched pilot source, recipe, data or code.")
        expected = shard_targets(report["shard_index"])
        if report["target_names"] != list(expected) or set(report["targets"]) != set(expected):
            raise ValueError("Target coverage differs from the fixed shard assignment.")
        plan, data = report["document_plan"], report["data"]
        if plan["manifest_sha256"] != _json_digest(data):
            raise ValueError("Data manifest digest mismatch.")
        for split, count in (("train", 512), ("heldout", 64)):
            lengths = [row["valid_tokens"] for row in data[split]["documents"]]
            if len(lengths) != count or lengths != plan[split + "_lengths"] or sum(lengths) != data[split]["valid_tokens"]:
                raise ValueError("Calibration document/token inventory is inconsistent.")
        for name, target in report["targets"].items():
            if target["final_restored_sha256"] != target["source_module_sha256"] or set(target["arms"]) != set(map(str, LR_DIVISORS)):
                raise ValueError("Missing LR arm or failed final full-state restoration.")
            for divisor, arm in target["arms"].items():
                expected_lr = max(row["scale"] for row in target["cpu_coordinate_calibration"]["stages"]) / int(divisor)
                metrics, evidence = arm["local_metrics"], arm["local_metrics"]["evidence"]
                settings = metrics["settings"]
                if (arm["lr_divisor"] != int(divisor) or arm["lr"] != expected_lr or arm["steps"] != STEPS
                        or settings["lr"] != expected_lr or settings["steps"] != STEPS or settings["round_steps"] != ROUND_STEPS
                        or settings["source_profile"] != "no_rotation_int8_input"
                        or settings["train_tokens"] != data["train"]["valid_tokens"]
                        or settings["heldout_tokens"] != data["heldout"]["valid_tokens"]
                        or arm["restored_sha256_before_arm"] != target["source_module_sha256"]
                        or arm["restored_sha256_after_arm"] != target["source_module_sha256"]
                        or evidence["frozen_sha256_before"] != evidence["frozen_sha256_after"]
                        or not evidence["non_code_bytes_unchanged"] or not evidence["packed_padding_unchanged"]
                        or evidence["persistent_storage_bits_before"] != evidence["persistent_storage_bits_after"]
                        or evidence["final_changed_bits"] <= 0 or len(metrics["training_loss"]) != STEPS
                        or [row["step"] for row in metrics["step_telemetry"]] != list(range(1, STEPS + 1))):
                    raise ValueError("Incomplete core evidence, altered recipe, or changed non-code payload.")
            combined[name] = target
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    merged = {key: reports[0][key] for key in shared}
    merged.update(status="complete", targets=combined, **select_recipe(combined, reports[0]["document_plan"]["heldout_lengths"]))
    _write_report(output, merged)
    return merged
