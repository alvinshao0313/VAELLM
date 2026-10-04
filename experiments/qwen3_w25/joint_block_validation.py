"""Real block16 joint-code/decoder engineering validation on the immutable BEST."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.bit_recipe_pilot import SOURCE_CHECKPOINT_ID, document_plan, validate_recipe_source
from experiments.qwen3_w25.block_batching import batch_block_documents
from experiments.qwen3_w25.block_calibration import capture_block_batches
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_bits import _changed_bits, _code_banks, calibrate_local_bits_cpu
from experiments.qwen3_w25.local_block import PROJECTIONS, _tree_to
from experiments.qwen3_w25.local_decoder import _stage_norm, _storage_bits
from experiments.qwen3_w25.local_recovery import BASE_MODEL
from experiments.qwen3_w25.pilot_local_decoder import _write_report
from experiments.qwen3_w25.protected_bit_recovery import (
    CODE_FILES as RECOVERY_CODE_FILES, _file_sha256, load_recipe_selection,
    source_state_sha256, validate_pilot_source_model,
)
from experiments.qwen3_w25.sequential_recovery import CODE_FILES as SEQUENTIAL_CODE_FILES, select_calibration_prefix
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


BLOCK_INDEX = 16
PREFIX = f"model.layers.{BLOCK_INDEX}"
SOURCE_STATE_SHA256 = "b0505cacd73b5b0792f7908b9db61c583605f7e24c75951a98319d951231bf1c"
CODE_FILES = tuple(dict.fromkeys((*RECOVERY_CODE_FILES, *SEQUENTIAL_CODE_FILES,
    "experiments/qwen3_w25/joint_block.py", "experiments/qwen3_w25/joint_block_validation.py",
    "experiments/qwen3_w25/validate_joint_block.py")))


def configuration(*, train_documents=64):
    if type(train_documents) is not int or train_documents not in (64, 512):
        raise ValueError("train_documents must be exactly 64 or 512.")
    return {"block_index": BLOCK_INDEX, "seed": 31, "maximum_train_documents": 512,
            "train_documents": train_documents, "heldout_documents": 64, "max_length": 256, "microbatch_size": 8,
            "steps": 128, "decoder_lr": 1.25e-5, "teacher_microbatch_size": 1,
            "input_source": "original_teacher_block_inputs",
            "target_source": "original_teacher_block_outputs_before_next_layer",
            "bit_lr_rule": "per-projection initial CPU max sensitivity scale divided by family divisor",
            "attention": {"lr_divisor": 128, "round_steps": 128},
            "mlp": {"lr_divisor": 256, "round_steps": 256},
            "decoder_compute_dtype": "torch.float32", "projection_dtype": "torch.bfloat16",
            "heldout_used_for_selection": False, "cross_block_gradients": False,
            "allow_tf32": False, "deterministic_algorithms": True,
            "recipe_status": "starting configuration for new joint-path validation; not a validated joint-quality recipe",
            "comparison_boundary": "teacher-input single block; not directly comparable with rolling-student sequential recovery",
            "checkpoint_policy": "one diagnostic inference checkpoint; no optimizer or exact-resume state"}


def collect_state_inventory(block, *, prefix):
    """Record the actual seven decoder parameter sets and fourteen packed banks."""
    modules, allowed = {}, set()
    for name in PROJECTIONS:
        layer = block.get_submodule(name)
        decoder_keys = [name + "._parallel_stage_decoder." + key
                        for key, _ in layer._parallel_stage_decoder.named_parameters()]
        banks = [{"key": name + "." + key, "logical_shape": list(shape)} for key, shape in _code_banks(layer)]
        if not decoder_keys or len(banks) != 2:
            raise ValueError("Expected seven existing decoders and two real packed banks per projection.")
        modules[name] = {"decoder_keys": [prefix + "." + key for key in decoder_keys],
                         "stage_layout": [list(row) for row in layer._parallel_stage_layout],
                         "code_banks": [{**row, "key": prefix + "." + row["key"]} for row in banks]}
        allowed.update(modules[name]["decoder_keys"])
        allowed.update(row["key"] for row in modules[name]["code_banks"])
    return {"modules": modules, "allowed_state_keys": sorted(allowed)}


def audit_joint_state(source_state, actual_state, inventory):
    """Check all model tensors, then independently measure saved decoder/code updates."""
    if source_state.keys() != actual_state.keys():
        raise RuntimeError("Joint validation changed the complete source tensor inventory.")
    allowed = set(inventory["allowed_state_keys"])
    if not allowed <= source_state.keys() or set(inventory["modules"]) != set(PROJECTIONS):
        raise ValueError("Invalid seven-projection allowed-state inventory.")
    changed = []
    for name, before in source_state.items():
        after = actual_state[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(f"Joint validation changed tensor shape/dtype: {name}.")
        equal = torch.equal(before.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            after.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
        if not equal:
            if name not in allowed:
                raise RuntimeError(f"Joint validation changed a frozen source tensor: {name}.")
            if after.is_floating_point() and not bool(torch.isfinite(after).all()):
                raise FloatingPointError(f"Non-finite saved decoder tensor: {name}.")
            changed.append(name)
    per_module = {}
    for name, entry in inventory["modules"].items():
        deltas = _stage_norm([actual_state[key].float() - source_state[key].float()
                              for key in entry["decoder_keys"]], entry["stage_layout"])
        if not all(math.isfinite(value) and value > 0 for value in deltas):
            raise RuntimeError(f"Both decoder stages must retain an actual saved update: {name}.")
        flips = [_changed_bits(source_state[row["key"]], actual_state[row["key"]], row["logical_shape"])
                 for row in entry["code_banks"]]
        per_module[name] = {"stage_parameter_delta_l2_saved": deltas, "stage_changed_bits": flips}
    total_flips = sum(sum(row["stage_changed_bits"]) for row in per_module.values())
    return {"complete_source_tensors_compared": len(source_state), "changed_tensors": changed,
            "allowed_state_keys": sorted(allowed), "frozen_source_bytes_unchanged": True,
            "tensor_schema_unchanged": True, "packed_padding_unchanged": True,
            "total_changed_code_bits": total_flips, "modules": per_module}


def bit_configuration(calibrations):
    if set(calibrations) != set(PROJECTIONS):
        raise ValueError("CPU sensitivity calibration must cover exactly seven projections.")
    rates, rounds = {}, {}
    for name, calibration in calibrations.items():
        stages = calibration["stages"]
        if (calibration["method"] != "cpu_fp32_dense_decoder_estimate"
                or [row["stage"] for row in stages] != [0, 1]
                or any(not math.isfinite(row["scale"]) or row["scale"] <= 0 for row in stages)):
            raise ValueError(f"Invalid CPU sensitivity calibration: {name}.")
        divisor = 128 if name.startswith("self_attn.") else 256
        rates[name], rounds[name] = max(row["scale"] for row in stages) / divisor, divisor
    return rates, rounds


@torch.no_grad()
def native_outputs(block, batches, device):
    values = []
    for batch in batches:
        value = block(batch.hidden_states.to(device), **_tree_to(batch.forward_kwargs, device))[0]
        if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
            raise RuntimeError("Expected finite native BF16 block outputs.")
        values.append(value.detach().cpu())
    return values


def _release_block(block):
    block.cpu()
    for name in PROJECTIONS:
        layer = block.get_submodule(name)
        layer._build_parallel_stage_decode_plan()
        layer.clear_decoded_weight_cache()


def require_native_runtime(block):
    for name in PROJECTIONS:
        layer = block.get_submodule(name)
        if (layer.trainable_decode or getattr(layer, "_sparse_bit_binding", None) is not None
                or "forward" in layer.__dict__
                or getattr(layer, "_decoder_compute_dtype", None) not in (None, torch.bfloat16)
                or any(parameter.dtype != torch.bfloat16 or parameter.grad is not None or parameter.requires_grad
                       for parameter in layer._parallel_stage_decoder.parameters())):
            raise RuntimeError(f"Joint core did not restore the frozen native BF16 projection runtime: {name}.")


def validate_core_evidence(result, inventory, state_audit, bit_lrs, round_steps, plan, *, prefix=PREFIX):
    settings, evidence = result["settings"], result["evidence"]
    if (settings["steps"] != 128 or settings["decoder_lr"] != 1.25e-5
            or settings["bit_lrs"] != bit_lrs or settings["round_steps"] != round_steps
            or settings["train_tokens"] != plan["train_tokens"] or settings["heldout_tokens"] != plan["heldout_tokens"]):
        raise RuntimeError("Joint core settings differ from the explicit validation recipe/data.")
    expected_local = {name.removeprefix(prefix + ".") for name in inventory["allowed_state_keys"]}
    if set(evidence["allowed_state_keys"]) != expected_local or set(result["modules"]) != set(PROJECTIONS):
        raise RuntimeError("Joint core updated a different parameter/bank scope.")
    if len(result["training_loss"]) != 128 or any(not math.isfinite(value) or value < 0 for value in result["training_loss"]):
        raise RuntimeError("Incomplete/non-finite joint training-loss trajectory.")
    for name in PROJECTIONS:
        metrics, actual = result["modules"][name], state_audit["modules"][name]
        gradients = metrics["stage_proxy_gradient_l2_max"]
        if len(gradients) != 2 or any(not math.isfinite(value) or value < 0 for value in gradients):
            raise RuntimeError(f"Missing/non-finite two-bank gradient evidence: {name}.")
        decoder_gradients = metrics["stage_decoder_gradient_l2_max"]
        if len(decoder_gradients) != 2 or any(not math.isfinite(value) or value <= 0 for value in decoder_gradients):
            raise RuntimeError(f"Both decoder stages require finite nonzero gradient evidence: {name}.")
        first = metrics["first_step_proxy_gradients"]
        counts = [math.prod(row["logical_shape"]) for row in inventory["modules"][name]["code_banks"]]
        if ([row["stage"] for row in first] != [0, 1] or [row["numel"] for row in first] != counts
                or len(metrics["step_telemetry"]) != 128
                or [row["step"] for row in metrics["step_telemetry"]] != list(range(1, 129))
                or any(len(row["stage_proxy_gradient_l2"]) != 2
                       or any(not math.isfinite(value) or value < 0 for value in row["stage_proxy_gradient_l2"])
                       for row in metrics["step_telemetry"])):
            raise RuntimeError(f"Incomplete fourteen-bank trajectory evidence: {name}.")
        if metrics["stage_changed_bits"] != actual["stage_changed_bits"]:
            raise RuntimeError(f"Final packed changes differ from core evidence: {name}.")
        if any(not math.isclose(recorded, measured, rel_tol=1e-6, abs_tol=1e-10)
               for recorded, measured in zip(metrics["stage_parameter_delta_l2_saved"],
                                              actual["stage_parameter_delta_l2_saved"], strict=True)):
            raise RuntimeError(f"Saved two-stage decoder update differs from core evidence: {name}.")
    for key in ("tensor_schema_unchanged", "parameter_ids_unchanged", "packed_padding_unchanged",
                "coordinate_scales_fixed"):
        if evidence.get(key) is not True:
            raise RuntimeError(f"Failed joint-core invariant: {key}.")
    if evidence.get("has_committed_code_change") is not bool(state_audit["total_changed_code_bits"]):
        raise RuntimeError("Joint-core code-change flag differs from the actual packed state.")
    if (evidence["frozen_sha256_before"] != evidence["frozen_sha256_after"]
            or evidence["persistent_storage_bits_before"] != evidence["persistent_storage_bits_after"]):
        raise RuntimeError("Joint-core frozen state/storage proof failed.")
    for key in ("proxy_parameter_dtypes", "proxy_gradient_dtypes", "bit_optimizer_moment_dtypes",
                "decoder_optimizer_moment_dtypes"):
        if evidence[key] != ["torch.float32"]:
            raise RuntimeError(f"Joint core violated the FP32 optimization contract: {key}.")


def run_joint_block_validation(checkpoint_dir, pilot_json, output_dir, *, train_documents=64):
    config = configuration(train_documents=train_documents)
    # Keep this import lazy so CPU state-contract checks never instantiate the
    # new CUDA-only joint optimizer or present a CPU proxy surrogate as evidence.
    from experiments.qwen3_w25.joint_block import optimize_joint_block

    source, pilot_path, output = (Path(path).resolve() for path in (checkpoint_dir, pilot_json, output_dir))
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Validation output must be new and outside the immutable source checkpoint.")
    meta, source_audit = _validate_checkpoint(source)
    source_sha = source_state_sha256(source)
    if meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID or source_sha != SOURCE_STATE_SHA256:
        raise ValueError("This engineering validation requires the unchanged original BEST checkpoint.")
    pilot, pilot_sha, _ = load_recipe_selection(pilot_path, source, meta)
    if not source_audit["budget_passed"]:
        raise ValueError("Source violates the no-LoRA 2.5-bpw budget.")
    output.mkdir(parents=True)
    path, device, repo = output / "validation.json", torch.device("cuda:0"), Path(__file__).resolve().parents[2]
    report = {"status": "preparing", "purpose": "joint_block_engineering_validation_not_full_model_accuracy",
              "source_checkpoint_dir": str(source), "source_checkpoint_id": meta["checkpoint_id"],
              "source_state_sha256": source_sha, "source_bit_budget": source_audit,
              "pilot_json": str(pilot_path), "pilot_sha256": pilot_sha, "configuration": config,
              "invocation": {"python": sys.executable, "entrypoint": "experiments/qwen3_w25/validate_joint_block.py",
                  "checkpoint_dir": str(source), "pilot_json": str(pilot_path), "output_dir": str(output),
                  "train_documents": train_documents},
              "code_sha256": {name: _file_sha256(repo / name) for name in CODE_FILES}}
    _write_report(path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        full_train, heldout_batch, data = build_calibration_batches(tokenizer, seed=31, train_documents=512,
                                                                   heldout_documents=64, max_length=256)
        if data != pilot["data"] or document_plan(full_train, heldout_batch, data) != pilot["document_plan"]:
            raise ValueError("Joint validation changed the fixed 512+64 calibration identity.")
        train_batch, heldout_batch, plan = select_calibration_prefix(full_train, heldout_batch, data, train_documents)
        report.update(data=data, selected_document_plan=plan)
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL,
            map_location="cpu", strict=True, expected_kind="final_model")
        model.eval().requires_grad_(False)
        validate_recipe_source(model, loaded_meta)
        validate_pilot_source_model(model, pilot)
        block = model.get_submodule(PREFIX)
        inventory = collect_state_inventory(block, prefix=PREFIX)
        calibrations = {name: calibrate_local_bits_cpu(block.get_submodule(name)) for name in PROJECTIONS}
        bit_lrs, round_steps = bit_configuration(calibrations)
        report.update(state_inventory=inventory, cpu_coordinate_calibration=calibrations,
                      bit_lrs=bit_lrs, round_steps=round_steps)
        _write_report(path, report)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        train = capture_block_batches(teacher, BLOCK_INDEX, train_batch)
        heldout = capture_block_batches(teacher, BLOCK_INDEX, heldout_batch)
        torch.cuda.synchronize(device)
        report["teacher_capture"] = {"seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "attention_implementation": teacher.config._attn_implementation}
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        train_execution = batch_block_documents(train, microbatch_size=8)
        heldout_execution = batch_block_documents(heldout, microbatch_size=8)
        report["execution_batch_sizes"] = {"train": [row.hidden_states.shape[0] for row in train_execution],
                                            "heldout": [row.hidden_states.shape[0] for row in heldout_execution]}
        source_state = _load_state(source)
        _assert_state_equal(model.state_dict(), source_state, "Unmodified source before joint validation")
        storage_before = _storage_bits(model)
        block.to(device)
        report["status"] = "training"
        _write_report(path, report)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()

        def progress(row):
            print(json.dumps({**row, "elapsed_seconds": time.perf_counter() - started}), flush=True)

        try:
            result = optimize_joint_block(block, train_execution, heldout_execution, steps=128,
                decoder_lr=1.25e-5, bit_lrs=bit_lrs, round_steps=round_steps, seed=31, progress_callback=progress)
            torch.cuda.synchronize(device)
            report["training_resources"] = {"seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            require_native_runtime(block)
            expected_outputs = native_outputs(block, heldout, device)
        finally:
            _release_block(block)
        if _storage_bits(model) != storage_before:
            raise RuntimeError("Joint training changed the complete model persistent storage footprint.")
        state_audit = audit_joint_state(source_state, model.state_dict(), inventory)
        if state_audit["total_changed_code_bits"] <= 0:
            raise RuntimeError("No packed code change survived the joint engineering validation.")
        validate_core_evidence(result, inventory, state_audit, bit_lrs, round_steps, plan)
        report.update(local_metrics=result, in_memory_state_audit=state_audit, status="saving")
        _write_report(path, report)
        destination = output / "checkpoint"
        saved = save_v6_full_checkpoint(model, str(destination), checkpoint_kind="final_model",
            compressed_targets=meta["compressed_targets"], pending_dense_targets=(), skip_targets=(),
            train_mode="none", after_category_mode="none", lora_config=None, base_model_path=BASE_MODEL,
            tokenizer=tokenizer, extra_meta={"joint_block_validation": {
                "source_checkpoint_id": meta["checkpoint_id"], "configuration": config,
                "code_sha256": report["code_sha256"], "manifest_sha256": plan["manifest_sha256"]}})
        if saved["checkpoint_id"] == meta["checkpoint_id"]:
            raise RuntimeError("New joint inference checkpoint reused the source UUID.")
        serialized = _load_state(destination)
        serialized_compared = _assert_state_equal(model.state_dict(), serialized, "Serialized joint validation")
        saved_audit = audit_joint_state(source_state, serialized, inventory)
        if saved_audit != state_audit:
            raise RuntimeError("Serialized joint updates differ from in-memory evidence.")
        del block, model, train_execution, heldout_execution, train
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, reloaded_meta, _ = load_v6_model_checkpoint(str(destination), base_model_path=BASE_MODEL,
            map_location="cpu", strict=True, expected_kind="final_model")
        reloaded.eval().requires_grad_(False)
        compared = _assert_state_equal(reloaded.state_dict(), serialized, "Fresh strict joint validation load")
        for field in ("compressed_targets", "pending_dense_targets", "skip_targets", "converted_modules"):
            if reloaded_meta[field] != meta[field]:
                raise RuntimeError(f"Joint validation changed checkpoint metadata: {field}.")
        if (reloaded_meta["checkpoint_id"] != saved["checkpoint_id"]
                or reloaded_meta.get("immutable_resume_contract") is not None
                or (reloaded_meta.get("extra_meta") or {}).get("cat_runtime_state_file")
                or reloaded_meta.get("lora_config") is not None):
            raise RuntimeError("Joint inference checkpoint inherited resume/LoRA state or changed identity.")
        recovered = reloaded.get_submodule(PREFIX).to(device)
        try:
            require_native_runtime(recovered)
            actual_outputs = native_outputs(recovered, heldout, device)
            for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        finally:
            _release_block(recovered)
        audit = audit_checkpoint(destination, max_bpw=2.5, max_lora_rank=0)
        for field in ("original_linear_weights", "user_linear_bits", "packed_linear_bits",
                      "code_padding_bits", "full_model_tensor_bytes"):
            if audit[field] != source_audit[field]:
                raise RuntimeError(f"Joint checkpoint changed actual storage accounting: {field}.")
        if not audit["budget_passed"]:
            raise RuntimeError("Joint checkpoint failed the actual no-LoRA 2.5-bpw budget.")
        report.update(status="complete", checkpoint_dir=str(destination), checkpoint_id=saved["checkpoint_id"],
            serialized_state_audit=saved_audit, serialized_tensors_compared=serialized_compared,
            reloaded_tensors_compared=compared, native_heldout_outputs_compared=len(expected_outputs),
            independent_reload_output_exact=True, bit_budget=audit)
        _write_report(path, report)
        return report
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        _write_report(path, report)
        raise
