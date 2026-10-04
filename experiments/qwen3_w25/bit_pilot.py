"""Fixed-recipe packed-bit pilot varying only the number of training documents."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_bits import calibrate_local_bits_cpu, optimize_local_bits, _digest
from experiments.qwen3_w25.local_recovery import capture_shard_inputs, validate_recovery_source
from experiments.qwen3_w25.pilot_local_decoder import (
    BASE_MODEL, CHUNK_SIZE, TARGETS, _comparison, _native_document_metrics, _write_report,
)
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


SOURCE_CHECKPOINT_ID = "c24d4b96-01f6-43d7-8df4-8a9b15ca3469"
TRAIN_DOCUMENTS = (32, 128, 512)
STEPS = ROUND_STEPS = 128
CODE_FILES = (
    "experiments/qwen3_w25/bit_pilot.py", "experiments/qwen3_w25/pilot_local_bits.py",
    "experiments/qwen3_w25/local_bits.py", "experiments/qwen3_w25/local_decoder.py",
    "experiments/qwen3_w25/local_recovery.py", "experiments/qwen3_w25/pilot_local_decoder.py",
    "experiments/qwen3_w25/calibration.py", "experiments/linear_output/objectives.py",
    "litebsq/bit_sensitivity.py", "litebsq/vae_linear.py", "litebsq/packed_bit_linear.py",
    "sparse_bit_tuning/config.py", "sparse_bit_tuning/manager.py", "sparse_bit_tuning/module.py",
    "sparse_bit_tuning/optimizer.py", "sparse_bit_tuning/coordinates.py",
    "sparse_bit_tuning/packed_ops.py", "sparse_bit_tuning/triton_kernels.py",
)


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def pilot_document_plan(train_batch, heldout_batch, manifest):
    """Resolve exact document-prefix token boundaries from the single real manifest."""
    if manifest["seed"] != 31 or manifest["max_length"] != 256:
        raise ValueError("Bit pilot requires seed=31 and max_length=256.")
    lengths = {}
    for split, batch, count in (("train", train_batch, 512), ("heldout", heldout_batch, 64)):
        documents = manifest[split]["documents"]
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if len(documents) != count or ids.ndim != 2 or ids.shape != mask.shape or len(ids) != count:
            raise ValueError(f"Expected exactly {count} documents with matching token/mask shapes: {split}.")
        if ids.shape[1] > 256 or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"Invalid length or attention mask: {split}.")
        lengths[split] = [row["valid_tokens"] for row in documents]
        if (lengths[split] != mask.sum(dim=1).tolist() or any(n < 1 for n in lengths[split])
                or sum(lengths[split]) != manifest[split]["valid_tokens"]):
            raise ValueError(f"Document token boundaries disagree with the captured input contract: {split}.")
    documents = manifest["train"]["documents"] + manifest["heldout"]["documents"]
    for key in ("row_index", "group_sha256", "token_sha256"):
        if len({row[key] for row in documents}) != 576:
            raise ValueError(f"Duplicate document/group/token identity across the pilot: {key}.")
    return {
        "train_lengths": lengths["train"], "heldout_lengths": lengths["heldout"],
        "arms": {str(count): {"document_indices": list(range(count)),
                              "valid_tokens": sum(lengths["train"][:count]),
                              "documents_sha256": _json_digest(manifest["train"]["documents"][:count])}
                 for count in TRAIN_DOCUMENTS},
        "selection": {"heldout_document_indices": list(range(32)),
                      "valid_tokens": sum(lengths["heldout"][:32]),
                      "documents_sha256": _json_digest(manifest["heldout"]["documents"][:32])},
        "confirmation": {"heldout_document_indices": list(range(32, 64)),
                         "valid_tokens": sum(lengths["heldout"][32:]),
                         "documents_sha256": _json_digest(manifest["heldout"]["documents"][32:])},
        "manifest_sha256": _json_digest(manifest),
    }


def _trajectory(metrics):
    losses, telemetry = metrics["training_loss"], metrics["step_telemetry"]
    first_flip = next((row["step"] for row in telemetry if row["step_flip_count"] > 0), None)
    initial = losses[0]
    return {
        "initial_loss": initial, "max_loss": max(losses),
        "max_loss_over_initial": max(losses) / initial if initial > 0 else None,
        "first_flip_step": first_flip,
        "return_below_initial_step": next((index + 1 for index, loss in enumerate(losses)
                                            if first_flip is not None and index + 1 > first_flip and loss < initial), None),
        "loss_timing": "pre-update full-training-token MSE; step flip count describes the subsequent update",
    }


def _restore(layer, initial, expected_digest):
    layer.load_state_dict(initial, strict=True)
    layer._build_parallel_stage_decode_plan()
    layer.clear_decoded_weight_cache()
    if _digest(layer.state_dict(), set()) != expected_digest:
        raise RuntimeError("Pilot failed to restore the complete original module state.")


def run_bit_pilot(checkpoint_dir, output_dir):
    source, output = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    meta, source_audit = _validate_checkpoint(source)
    if meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID or not source_audit["budget_passed"]:
        raise ValueError("Pilot requires the fixed original down3 final checkpoint and its valid bit budget.")
    if Path(meta["base_model_path"]).resolve() != Path(BASE_MODEL).resolve():
        raise ValueError("Pilot source must reference the fixed Qwen3-8B teacher/tokenizer.")
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "pilot.json"
    device = torch.device("cuda:0")
    started = time.perf_counter()
    report = {
        "purpose": "local_bit_calibration_amount_pilot_not_downstream_accuracy", "status": "running",
        "source_checkpoint_id": meta["checkpoint_id"], "source_bit_budget": source_audit,
        "configuration": {
            "checkpoint_dir": str(source), "output_dir": str(output), "base_model_path": BASE_MODEL,
            "targets": list(TARGETS), "train_document_arms": list(TRAIN_DOCUMENTS),
            "maximum_train_documents": 512, "heldout_documents": 64,
            "selection_documents": 32, "confirmation_documents": 32,
            "seed": 31, "max_length": 256, "chunk_size": CHUNK_SIZE, "device": str(device),
            "steps": STEPS, "round_steps": ROUND_STEPS, "early_stable_round_end": True,
            "lr_rule": "per-target initial CPU max(decoder_sensitivity scale)/64, constant across arms",
            "optimizer": "adamw", "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8,
            "proxy_coordinates": "decoder_sensitivity", "active_ratio": 1.0, "streaming": False,
            "decoder_frozen": True, "decoder_compute_dtype": "torch.float32",
            "proxy_gradient_moment_dtype": "torch.float32", "native_activation_dtype": "torch.bfloat16",
            "teacher_microbatch_size": 1, "capture_passes": 1,
            "selection_rule": "minimum equal-target mean of selection after/before token-MSE ratios; ties prefer fewer training documents",
            "confirmation_used_for_selection": False, "checkpoint_written": False,
            "allow_tf32": False, "deterministic_algorithms": True,
        },
        "code_sha256": {relative: hashlib.sha256((Path(__file__).resolve().parents[2] / relative).read_bytes()).hexdigest()
                        for relative in CODE_FILES},
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
            tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
        plan = pilot_document_plan(train_batch, heldout_batch, manifest)
        report.update(data=manifest, document_plan=plan)
        phase_start = time.perf_counter()
        model, loaded_meta, _ = load_v6_model_checkpoint(
            str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True)
        if loaded_meta["checkpoint_id"] != meta["checkpoint_id"]:
            raise RuntimeError("Source identity changed during pilot initialization.")
        model.eval().requires_grad_(False)
        validate_recovery_source(model, loaded_meta)
        report["checkpoint_load_seconds"] = time.perf_counter() - phase_start
        calibrations = {name: calibrate_local_bits_cpu(model.get_submodule(name)) for name in TARGETS}
        _write_report(report_path, report)

        print("Capturing four targets together over the fixed 512+64 documents", flush=True)
        phase_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundles = capture_shard_inputs(teacher, TARGETS, train_batch, heldout_batch)
        torch.cuda.synchronize(device)
        report["teacher_capture_seconds"] = time.perf_counter() - phase_start
        report["teacher_capture_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["teacher_capture_peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["teacher_cache_cpu_bytes"] = sum(t.numel() * t.element_size() for item in bundles.values() for t in item.values())
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        _write_report(report_path, report)

        lengths = plan["heldout_lengths"]
        selection_tokens = plan["selection"]["valid_tokens"]
        for name in TARGETS:
            bundle = bundles.pop(name)
            train_cpu, heldout_cpu = bundle["train_inputs"], bundle["heldout_inputs"]
            if (len(train_cpu) != manifest["train"]["valid_tokens"]
                    or len(heldout_cpu) != manifest["heldout"]["valid_tokens"]
                    or train_cpu.dtype != torch.bfloat16 or heldout_cpu.dtype != torch.bfloat16):
                raise ValueError(f"Captured BF16 token inventory differs from the manifest: {name}.")
            layer = model.get_submodule(name).to(device)
            initial = {key: value.detach().cpu().clone() for key, value in layer.state_dict().items()}
            initial_digest = _digest(initial, set())
            _restore(layer, initial, initial_digest)
            reference = bundle["reference_weight"].to(device=device, dtype=torch.float32)
            train_x, heldout_x = train_cpu.to(device), heldout_cpu.to(device)
            selection_x, confirmation_x = heldout_x[:selection_tokens], heldout_x[selection_tokens:]
            baseline_selection = _native_document_metrics(layer, reference, selection_x, lengths[:32])
            baseline_confirmation = _native_document_metrics(layer, reference, confirmation_x, lengths[32:])
            cpu_calibration = calibrations[name]
            lr = max(stage["scale"] for stage in cpu_calibration["stages"]) / 64.0
            target_report = {
                "source_module_sha256": initial_digest,
                "source_module": {"weight_shape": [layer.out_features, layer.in_features], "transpose": layer.transpose,
                                  "stage_codebook_dims": list(layer.stage_codebook_dims),
                                  "stage_codebook_bits": [layer.get_stage_part_vq_spec(stage, 0)["logical_shape"][-1]
                                                          for stage in range(2)]},
                "cpu_coordinate_calibration": cpu_calibration, "lr": lr, "arms": {},
            }
            report["targets"][name] = target_report
            try:
                for count in TRAIN_DOCUMENTS:
                    _restore(layer, initial, initial_digest)
                    tokens = plan["arms"][str(count)]["valid_tokens"]
                    torch.cuda.synchronize(device)
                    arm_start = time.perf_counter()

                    def progress(row):
                        if row["step"] % 10 == 0 or row["step"] == STEPS:
                            print(json.dumps({"target": name, "train_documents": count, **row}), flush=True)

                    metrics = optimize_local_bits(
                        layer, reference, train_x[:tokens], heldout_x, steps=STEPS, lr=lr,
                        round_steps=ROUND_STEPS, chunk_size=CHUNK_SIZE, seed=31, progress_callback=progress)
                    torch.cuda.synchronize(device)
                    optimization_seconds = time.perf_counter() - arm_start
                    selection_after = _native_document_metrics(layer, reference, selection_x, lengths[:32])
                    confirmation_after = _native_document_metrics(layer, reference, confirmation_x, lengths[32:])
                    torch.cuda.synchronize(device)
                    target_report["arms"][str(count)] = {
                        "train_documents": count, "train_tokens": tokens,
                        "source_module_sha256": initial_digest, "source_state_restored_before_arm": True,
                        "local_metrics": metrics, "trajectory": _trajectory(metrics),
                        "native_selection": _comparison(baseline_selection, selection_after),
                        "native_confirmation": _comparison(baseline_confirmation, confirmation_after),
                        "optimization_seconds": optimization_seconds, "elapsed_seconds": time.perf_counter() - arm_start,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "bit_budget": {"user_linear_bpw": source_audit["user_linear_bpw"],
                                       "packed_linear_bpw": source_audit["packed_linear_bpw"],
                                       "basis": "source checkpoint actual audit plus core-verified unchanged persistent shape/dtype/storage; no new checkpoint"},
                    }
                    _write_report(report_path, report)
            finally:
                _restore(layer, initial, initial_digest)
                layer.cpu()
                layer._build_parallel_stage_decode_plan()
                layer.clear_decoded_weight_cache()
            del bundle, train_cpu, heldout_cpu, reference, train_x, heldout_x, selection_x, confirmation_x, initial
            gc.collect()
            torch.cuda.empty_cache()

        scores = {}
        for count in TRAIN_DOCUMENTS:
            ratios = [report["targets"][name]["arms"][str(count)]["native_selection"]["after_before_ratio"]
                      for name in TARGETS]
            scores[str(count)] = {"mean_selection_ratio": sum(ratios) / len(TARGETS),
                                  "worst_selection_ratio": max(ratios), "improved_targets": sum(value < 1 for value in ratios)}
        selected = min(TRAIN_DOCUMENTS, key=lambda count: (scores[str(count)]["mean_selection_ratio"], count))
        confirmation_ratios = [report["targets"][name]["arms"][str(selected)]["native_confirmation"]["after_before_ratio"]
                               for name in TARGETS]
        report.update(status="complete", selection_scores=scores, selected_train_documents=selected,
                      selected_confirmation={"mean_ratio": sum(confirmation_ratios) / len(TARGETS),
                                             "worst_ratio": max(confirmation_ratios),
                                             "improved_targets": sum(value < 1 for value in confirmation_ratios),
                                             "used_for_selection": False},
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        print(json.dumps({"selected_train_documents": selected, "selection_scores": scores,
                          "independent_confirmation": report["selected_confirmation"], "report": str(report_path)}), flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        raise
