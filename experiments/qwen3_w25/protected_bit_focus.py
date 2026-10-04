"""Isolated MLP LR/budget diagnostic after the completed protected-base pilot."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.bit_pilot import _digest, _restore, _trajectory
from experiments.qwen3_w25.bit_recipe_pilot import (
    CODE_FILES as RECIPE_CODE_FILES, SOURCE_CHECKPOINT_ID, _ratio, configuration as pilot_configuration,
    document_plan, validate_recipe_source,
)
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_bits import optimize_local_bits
from experiments.qwen3_w25.local_recovery import BASE_MODEL, capture_shard_inputs
from experiments.qwen3_w25.pilot_local_decoder import _comparison, _native_document_metrics, _write_report
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


TARGETS = {"gate": "model.layers.9.mlp.gate_proj", "down": "model.layers.27.mlp.down_proj"}
CONTROL = "s128_steps128_round128"
STEPS = (128, 256)
ROUND_STEPS = 256
CODE_FILES = (*RECIPE_CODE_FILES, "experiments/qwen3_w25/protected_bit_focus.py",
              "experiments/qwen3_w25/pilot_protected_bit_focus.py")


def focus_configuration(target):
    if target not in TARGETS.values():
        raise ValueError("Focused targets are only L9/gate and L27/down.")
    return {"target": target, "lr_divisor": 256, "step_arms": list(STEPS), "round_steps": ROUND_STEPS,
            "train_documents": 512, "heldout_documents": 64, "selection_documents": 32,
            "repeated_diagnostic_documents": 32, "seed": 31, "max_length": 256,
            "early_stable_round_end": True, "stable_steps": 10, "checkpoint_written": False,
            "short_arm_termination": "128 updates then final_commit; no scheduled round reset at 128",
            "comparison_note": "New focused protocol changes LR, step budget and round_steps; the reused s128 control had round128.",
            "repeated_diagnostic_used_for_selection": False, "repeated_diagnostic_is_new_blind_set": False}


def prefix_comparison(short_metrics, long_metrics):
    first, second = short_metrics["training_loss"], long_metrics["training_loss"][:128]
    if len(first) != 128 or len(long_metrics["training_loss"]) != 256:
        raise ValueError("Prefix comparison requires complete 128/256-step trajectories.")
    difference = [abs(a - b) for a, b in zip(first, second)]
    left, right = short_metrics["step_telemetry"], long_metrics["step_telemetry"][:128]
    if len(left) != 128 or len(right) != 128:
        raise ValueError("Incomplete prefix optimizer telemetry.")
    return {"steps_compared": 128, "loss_exact": first == second, "max_absolute_loss_difference": max(difference),
            "max_relative_loss_difference": max(delta / max(abs(a), 1e-30) for delta, a in zip(difference, first)),
            "all_step_telemetry_exact": left == right,
            "step_flip_counts_exact": [row["step_flip_count"] for row in left] == [row["step_flip_count"] for row in right],
            "short_round_end_steps": [row["step"] for row in left if row["round_ended"]],
            "long_prefix_round_end_steps": [row["step"] for row in right if row["round_ended"]]}


def select_focus_recipe(arms, heldout_lengths):
    expected = {CONTROL, "s256_steps128_round256", "s256_steps256_round256"}
    if set(arms) != expected or len(heldout_lengths) != 64:
        raise ValueError("Selection requires the original control and both completed focused arms.")
    scores = {name: _ratio(arm["native_selection"], heldout_lengths[:32]) for name, arm in arms.items()}
    selected = min(scores, key=lambda name: (scores[name], arms[name]["local_metrics"]["settings"]["steps"],
                                             arms[name]["local_metrics"]["settings"]["lr"]))
    diagnostics = {name: _ratio(arm["native_repeated_diagnostic"], heldout_lengths[32:]) for name, arm in arms.items()}
    return {"selected_arm": selected, "selection_scores": scores, "repeated_diagnostic_ratios": diagnostics,
            "selected_repeated_diagnostic": {"after_before_ratio": diagnostics[selected],
                "used_for_selection": False, "new_blind_confirmation": False}}


def run_protected_bit_focus(checkpoint_dir, pilot_json, output_dir, *, target):
    config = focus_configuration(target)
    source, pilot_path, output = [Path(path).resolve() for path in (checkpoint_dir, pilot_json, output_dir)]
    meta, audit = _validate_checkpoint(source)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if (meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID or not audit["budget_passed"]
            or pilot["status"] != "complete" or pilot["source_checkpoint_id"] != SOURCE_CHECKPOINT_ID
            or Path(pilot["source_checkpoint_dir"]).resolve() != source
            or pilot["configuration"] != pilot_configuration() or pilot["source_bit_budget"] != audit):
        raise ValueError("Focused pilot requires the fixed source and completed protected-base LR pilot.")
    original = pilot["targets"][target]
    control = deepcopy(original["arms"]["128"])
    scales = original["cpu_coordinate_calibration"]
    lr = max(row["scale"] for row in scales["stages"]) / 256.0
    if control["lr"] != lr * 2 or control["local_metrics"]["settings"]["round_steps"] != 128:
        raise ValueError("Reused control does not match the original s128/128/round128 recipe.")
    repo = Path(__file__).resolve().parents[2]
    if any(hashlib.sha256((repo / name).read_bytes()).hexdigest() != value for name, value in pilot["code_sha256"].items()):
        raise ValueError("An implementation dependency changed since the completed control pilot.")
    output.mkdir(parents=True, exist_ok=False)
    path, device, started = output / "pilot.json", torch.device("cuda:0"), time.perf_counter()
    report = {"status": "preparing", "purpose": "protected_MLP_bit_lr_and_post_flip_budget_diagnostic_not_downstream_accuracy",
              "configuration": config, "source_checkpoint_dir": str(source), "source_checkpoint_id": meta["checkpoint_id"],
              "source_bit_budget": audit, "pilot_json": str(pilot_path), "pilot_sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
              "source_module_sha256": original["source_module_sha256"], "cpu_coordinate_calibration": scales,
              "code_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in CODE_FILES},
              "arms": {CONTROL: {**control, "reused_from_completed_pilot": True}}}
    _write_report(path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_batch, heldout_batch, data = build_calibration_batches(
            tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
        plan = document_plan(train_batch, heldout_batch, data)
        if data != pilot["data"] or plan != pilot["document_plan"]:
            raise ValueError("Focused documents or token boundaries differ from the reused control.")
        report.update(data=data, document_plan=plan)
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True)
        model.eval().requires_grad_(False)
        validate_recipe_source(model, loaded_meta)
        layer = model.get_submodule(target)
        initial = {name: tensor.detach().cpu().clone() for name, tensor in layer.state_dict().items()}
        initial_digest = _digest(initial, set())
        if initial_digest != original["source_module_sha256"]:
            raise ValueError("Focused target persistent bytes differ from the reused control's source.")
        phase = time.perf_counter()
        print(f"Capturing {target} over the identical 512+64 documents", flush=True)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundle = capture_shard_inputs(teacher, (target,), train_batch, heldout_batch)[target]
        torch.cuda.synchronize(device)
        report["teacher_capture_seconds"] = time.perf_counter() - phase
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        layer.to(device)
        reference = bundle["reference_weight"].to(device=device, dtype=torch.float32)
        train_x, heldout_x = bundle["train_inputs"].to(device), bundle["heldout_inputs"].to(device)
        if (len(train_x) != data["train"]["valid_tokens"] or len(heldout_x) != data["heldout"]["valid_tokens"]
                or train_x.dtype != torch.bfloat16 or heldout_x.dtype != torch.bfloat16):
            raise ValueError("Teacher capture dtype or token inventory differs from the control.")
        lengths, cut = plan["heldout_lengths"], plan["selection"]["valid_tokens"]
        selection_x, diagnostic_x = heldout_x[:cut], heldout_x[cut:]
        try:
            _restore(layer, initial, initial_digest)
            baseline_selection = _native_document_metrics(layer, reference, selection_x, lengths[:32])
            baseline_diagnostic = _native_document_metrics(layer, reference, diagnostic_x, lengths[32:])
            report["baseline_recheck"] = {
                "selection": _comparison(control["native_selection"]["before"], baseline_selection),
                "repeated_diagnostic": _comparison(control["native_repeated_diagnostic"]["before"], baseline_diagnostic)}
            report["status"] = "running"
            _write_report(path, report)
            for steps in STEPS:
                arm = f"s256_steps{steps}_round256"
                _restore(layer, initial, initial_digest)
                arm_start = time.perf_counter()

                def progress(row):
                    if row["step"] % 10 == 0 or row["step"] == steps:
                        print(json.dumps({"target": target, "arm": arm, **row}), flush=True)

                metrics = optimize_local_bits(layer, reference, train_x, heldout_x, steps=steps, lr=lr,
                    round_steps=ROUND_STEPS, chunk_size=256, seed=31, progress_callback=progress)
                after_selection = _native_document_metrics(layer, reference, selection_x, lengths[:32])
                after_diagnostic = _native_document_metrics(layer, reference, diagnostic_x, lengths[32:])
                torch.cuda.synchronize(device)
                _restore(layer, initial, initial_digest)
                report["arms"][arm] = {"reused_from_completed_pilot": False, "lr": lr, "steps": steps,
                    "round_steps": ROUND_STEPS, "local_metrics": metrics, "trajectory": _trajectory(metrics),
                    "native_selection": _comparison(baseline_selection, after_selection),
                    "native_repeated_diagnostic": _comparison(baseline_diagnostic, after_diagnostic),
                    "restored_sha256_before_arm": initial_digest, "restored_sha256_after_arm": _digest(layer.state_dict(), set()),
                    "elapsed_seconds": time.perf_counter() - arm_start}
                _write_report(path, report)
        finally:
            _restore(layer, initial, initial_digest)
            report["final_restored_sha256"] = _digest(layer.state_dict(), set())
            layer.cpu()
            layer._build_parallel_stage_decode_plan()
            layer.clear_decoded_weight_cache()
        report["prefix_comparison"] = prefix_comparison(
            report["arms"]["s256_steps128_round256"]["local_metrics"], report["arms"]["s256_steps256_round256"]["local_metrics"])
        report.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                      **select_focus_recipe(report["arms"], lengths))
        _write_report(path, report)
        print(json.dumps({"target": target, "selected_arm": report["selected_arm"],
                          "selection_scores": report["selection_scores"], "prefix_comparison": report["prefix_comparison"]}), flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)}, elapsed_seconds=time.perf_counter() - started)
        _write_report(path, report)
        raise
