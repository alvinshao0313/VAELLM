"""Two fixed L27/down trials diagnosing the completed bit pilot's late instability."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.bit_pilot import (
    CODE_FILES as PILOT_CODE_FILES, _digest, _restore, _trajectory, pilot_document_plan,
)
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_bit_recovery import load_pilot_selection, validate_bit_recovery_source
from experiments.qwen3_w25.local_bits import optimize_local_bits
from experiments.qwen3_w25.local_recovery import BASE_MODEL, capture_shard_inputs
from experiments.qwen3_w25.pilot_local_decoder import _comparison, _native_document_metrics, _write_report
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


TARGET = "model.layers.27.mlp.down_proj"
CONTROL = "original_lr_steps128"
NEW_ARMS = (
    {"name": "original_lr_steps96", "steps": 96, "lr_multiplier": 1.0},
    {"name": "half_lr_steps128", "steps": 128, "lr_multiplier": 0.5},
)
CODE_FILES = (*PILOT_CODE_FILES, "experiments/qwen3_w25/local_bit_recovery.py",
              "experiments/qwen3_w25/focused_bit_pilot.py", "experiments/qwen3_w25/pilot_focused_bits.py")


def run_focused_bit_pilot(checkpoint_dir, pilot_json, output_dir):
    source, pilot_path, output = map(lambda path: Path(path).resolve(), (checkpoint_dir, pilot_json, output_dir))
    if source == output or source in output.parents:
        raise ValueError("Focused pilot output must be outside the immutable source checkpoint.")
    meta, audit = _validate_checkpoint(source)
    pilot, pilot_sha, selected = load_pilot_selection(pilot_path, source, meta)
    if selected != 512 or not audit["budget_passed"]:
        raise ValueError("This focused experiment requires the completed 512-document selection and valid source budget.")
    original_target = pilot["targets"][TARGET]
    control = deepcopy(original_target["arms"]["512"])
    calibration = deepcopy(original_target["cpu_coordinate_calibration"])
    original_lr = max(stage["scale"] for stage in calibration["stages"]) / 64.0
    if (original_target["lr"] != original_lr or control["local_metrics"]["settings"]["lr"] != original_lr
            or control["source_module_sha256"] != original_target["source_module_sha256"]):
        raise ValueError("The reused control does not match its original source/calibrated learning rate.")
    arms = [{**arm, "lr": original_lr * arm["lr_multiplier"], "round_steps": 128} for arm in NEW_ARMS]
    output.mkdir(parents=True, exist_ok=False)
    report_path, device = output / "pilot.json", torch.device("cuda:0")
    repo = Path(__file__).resolve().parents[2]
    started = time.perf_counter()
    control.update(reused_from_completed_pilot=True, reused_pilot_sha256=pilot_sha)
    report = {
        "purpose": "focused_local_bit_down_diagnostic_not_downstream_accuracy", "status": "preparing",
        "source_checkpoint_id": meta["checkpoint_id"], "source_bit_budget": audit,
        "pilot_json": str(pilot_path), "pilot_sha256": pilot_sha,
        "configuration": {
            "checkpoint_dir": str(source), "output_dir": str(output), "base_model_path": BASE_MODEL,
            "target": TARGET, "train_documents": 512, "heldout_documents": 64,
            "selection_documents": 32, "confirmation_documents": 32, "seed": 31,
            "max_length": 256, "chunk_size": 256, "round_steps": 128,
            "new_arms": arms, "reused_control": {"name": CONTROL, "steps": 128, "lr": original_lr},
            "optimizer": "adamw", "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8,
            "proxy_coordinates": "decoder_sensitivity", "active_ratio": 1.0, "streaming": False,
            "decoder_frozen": True, "decoder_compute_dtype": "torch.float32",
            "native_activation_dtype": "torch.bfloat16", "early_stable_round_end": True,
            "teacher_microbatch_size": 1, "capture_passes": 1,
            "allow_tf32": False, "deterministic_algorithms": True, "checkpoint_written": False,
            "selection_rule": "minimum selection after/before token-MSE ratio; ties prefer fewer steps then lower LR",
            "confirmation_used_for_selection": False,
            "confirmation_reused_from_prior_pilot": True, "confirmation_is_new_blind_set": False,
            "confirmation_note": "same held-out documents as the completed pilot; reused diagnostic confirmation, not a newly untouched split",
        },
        "code_sha256": {relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest() for relative in CODE_FILES},
        "source_module_sha256": original_target["source_module_sha256"],
        "source_module": original_target["source_module"],
        "cpu_coordinate_calibration": calibration, "arms": {CONTROL: control},
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
        train_batch, heldout_batch, data = build_calibration_batches(
            tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
        plan = pilot_document_plan(train_batch, heldout_batch, data)
        if data != pilot["data"] or plan != pilot["document_plan"]:
            raise ValueError("Focused pilot calibration differs from the original 512+64 manifest.")
        report.update(data=data, document_plan=plan, manifest_sha256=plan["manifest_sha256"])
        model, loaded_meta, _ = load_v6_model_checkpoint(
            str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True)
        if loaded_meta["checkpoint_id"] != meta["checkpoint_id"]:
            raise RuntimeError("Source changed during focused pilot initialization.")
        model.eval().requires_grad_(False)
        validate_bit_recovery_source(model, loaded_meta)
        layer = model.get_submodule(TARGET)
        initial = {key: tensor.detach().cpu().clone() for key, tensor in layer.state_dict().items()}
        initial_digest = _digest(initial, set())
        if initial_digest != original_target["source_module_sha256"]:
            raise ValueError("Focused module bytes differ from the completed pilot's initial state.")
        _write_report(report_path, report)
        print("Capturing L27/down over the identical 512+64 documents", flush=True)
        phase_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundle = capture_shard_inputs(teacher, (TARGET,), train_batch, heldout_batch)[TARGET]
        torch.cuda.synchronize(device)
        report["teacher_capture_seconds"] = time.perf_counter() - phase_started
        report["teacher_capture_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["teacher_capture_peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["teacher_cache_cpu_bytes"] = sum(t.numel() * t.element_size() for t in bundle.values())
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        train_cpu, heldout_cpu = bundle["train_inputs"], bundle["heldout_inputs"]
        if (len(train_cpu) != data["train"]["valid_tokens"] or len(heldout_cpu) != data["heldout"]["valid_tokens"]
                or train_cpu.dtype != torch.bfloat16 or heldout_cpu.dtype != torch.bfloat16):
            raise ValueError("Focused capture token inventory/dtype differs from the manifest.")
        layer.to(device)
        reference = bundle["reference_weight"].to(device=device, dtype=torch.float32)
        train_x, heldout_x = train_cpu.to(device), heldout_cpu.to(device)
        lengths = plan["heldout_lengths"]
        selection_tokens = plan["selection"]["valid_tokens"]
        selection_x, confirmation_x = heldout_x[:selection_tokens], heldout_x[selection_tokens:]
        try:
            _restore(layer, initial, initial_digest)
            baseline_selection = _native_document_metrics(layer, reference, selection_x, lengths[:32])
            baseline_confirmation = _native_document_metrics(layer, reference, confirmation_x, lengths[32:])
            report["baseline_recheck"] = {
                "selection": _comparison(control["native_selection"]["before"], baseline_selection),
                "confirmation": _comparison(control["native_confirmation"]["before"], baseline_confirmation),
            }
            report["status"] = "running"
            _write_report(report_path, report)
            for arm in arms:
                _restore(layer, initial, initial_digest)
                torch.cuda.synchronize(device)
                arm_started = time.perf_counter()

                def progress(row):
                    if row["step"] % 10 == 0 or row["step"] == arm["steps"]:
                        print(json.dumps({"target": TARGET, "arm": arm["name"], **row}), flush=True)

                metrics = optimize_local_bits(
                    layer, reference, train_x, heldout_x, steps=arm["steps"], lr=arm["lr"],
                    round_steps=128, chunk_size=256, seed=31, progress_callback=progress)
                torch.cuda.synchronize(device)
                optimization_seconds = time.perf_counter() - arm_started
                selection_after = _native_document_metrics(layer, reference, selection_x, lengths[:32])
                confirmation_after = _native_document_metrics(layer, reference, confirmation_x, lengths[32:])
                torch.cuda.synchronize(device)
                report["arms"][arm["name"]] = {
                    "reused_from_completed_pilot": False, "train_documents": 512,
                    "train_tokens": len(train_x), "source_module_sha256": initial_digest,
                    "source_state_restored_before_arm": True, "local_metrics": metrics,
                    "trajectory": _trajectory(metrics),
                    "native_selection": _comparison(baseline_selection, selection_after),
                    "native_confirmation": _comparison(baseline_confirmation, confirmation_after),
                    "optimization_seconds": optimization_seconds, "elapsed_seconds": time.perf_counter() - arm_started,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "bit_budget": {"user_linear_bpw": audit["user_linear_bpw"],
                                   "packed_linear_bpw": audit["packed_linear_bpw"],
                                   "basis": "actual source audit plus core-verified unchanged persistent shape/dtype/storage; no checkpoint saved"},
                }
                _write_report(report_path, report)
        finally:
            _restore(layer, initial, initial_digest)
            layer.cpu()
            layer._build_parallel_stage_decode_plan()
            layer.clear_decoded_weight_cache()
        scores = {name: item["native_selection"]["after_before_ratio"] for name, item in report["arms"].items()}
        selected_arm = min(scores, key=lambda name: (
            scores[name], report["arms"][name]["local_metrics"]["settings"]["steps"],
            report["arms"][name]["local_metrics"]["settings"]["lr"]))
        report.update(status="complete", selection_scores=scores, selected_arm=selected_arm,
                      selected_confirmation={"after_before_ratio": report["arms"][selected_arm]["native_confirmation"]["after_before_ratio"],
                                             "used_for_selection": False, "reused_diagnostic_split": True},
                      source_state_restored_after_experiment=True, elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        print(json.dumps({"status": "complete", "selected_arm": selected_arm, "selection_scores": scores,
                          "selected_confirmation": report["selected_confirmation"], "report": str(report_path)}), flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        raise
