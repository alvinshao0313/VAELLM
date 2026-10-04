"""Ordered decoder-only block recovery on separate native teacher/student streams."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.bit_pilot import _json_digest, pilot_document_plan
from experiments.qwen3_w25.block_batching import batch_block_documents
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25 import local_block
from experiments.qwen3_w25.local_block import PROJECTIONS, optimize_local_block
from experiments.qwen3_w25.local_decoder import _validate_payload_profile
from experiments.qwen3_w25.local_recovery import BASE_MODEL, TARGETS, PROJECTION_SHAPES
from experiments.qwen3_w25.local_recovery_finalize import _save_final
from experiments.qwen3_w25.pilot_local_decoder import _write_report
from experiments.qwen3_w25.sequential_block_calibration import SequentialBlockCalibration
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint
from litebsq.vae_linear import VAELinear


INPUT_SOURCE = "caller_supplied_fixed_hidden_states"
CODE_FILES = (
    "experiments/qwen3_w25/sequential_recovery.py", "experiments/qwen3_w25/recover_sequential_blocks.py",
    "experiments/qwen3_w25/sequential_block_calibration.py", "experiments/qwen3_w25/block_batching.py",
    "experiments/qwen3_w25/local_block.py", "experiments/qwen3_w25/block_weight_graph.py",
    "experiments/qwen3_w25/block_calibration.py", "experiments/qwen3_w25/calibration.py",
    "experiments/qwen3_w25/local_decoder.py", "experiments/qwen3_w25/local_recovery.py",
    "experiments/qwen3_w25/local_recovery_finalize.py", "experiments/qwen3_w25/assemble_categories.py",
    "experiments/qwen3_w25/bit_pilot.py", "litebsq/vae_linear.py", "litebsq/autoencoder.py",
    "litebsq/parallel_layers.py", "litebsq/packed_bit_linear.py", "litebsq/fused_multistage_decoder.py",
    "train_utils/checkpoint_v6.py", "train_utils/v6_model_loader.py", "train_utils/checkpoint_bpw.py",
)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def recovery_configuration(*, train_documents, steps, lr):
    if type(train_documents) is not int or train_documents not in (64, 512):
        raise ValueError("train_documents must be 64 or 512.")
    if type(steps) is not int or steps not in (64, 192):
        raise ValueError("steps must be an explicitly selected 64 or 192.")
    if type(lr) not in (int, float) or lr != 1.25e-5:
        raise ValueError("This sequential experiment fixes lr=1.25e-5.")
    return {
        "train_documents": train_documents, "maximum_train_documents": 512, "heldout_documents": 64,
        "max_length": 256, "seed": 31, "steps": steps, "lr": lr, "microbatch_size": 8,
        "layer_order": list(range(36)), "objective": "block_output_mse", "optimizer": "AdamW",
        "optimizer_dtype": "float32", "decoder_compute_dtype": "torch.float32",
        "projection_dtype": "torch.bfloat16", "native_advance_dtype": "torch.bfloat16",
        "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8,
        "input_source": "rolling_native_student_hidden_states",
        "target_source": "independent_original_teacher_next_hidden_states_before_final_norm",
        "core_input_source": INPUT_SOURCE, "teacher_microbatch_size": 1,
        "student_advance_microbatch_size": 1, "heldout_used_for_selection": False,
        "cross_layer_gradients": False, "bias_added": False, "checkpoint_policy": "final_inference_only",
        "allow_tf32": False, "deterministic_algorithms": True,
    }


def _require_core_input_metadata():
    if getattr(local_block, "INPUT_SOURCE", None) != INPUT_SOURCE:
        raise RuntimeError("local_block must first report input_source=caller_supplied_fixed_hidden_states; legacy teacher-only metadata is invalid for sequential recovery.")


def validate_sequential_source(model, meta):
    if (getattr(model.config, "model_type", None) != "qwen3"
            or Path(meta["base_model_path"]).resolve() != Path(BASE_MODEL).resolve()
            or meta["checkpoint_kind"] != "final_model" or set(meta["compressed_targets"]) != set(TARGETS)
            or meta["pending_dense_targets"] or meta["skip_targets"] or meta.get("lora_config") is not None
            or meta.get("shared_protected_residual_decoders") or (meta.get("extra_meta") or {}).get("residual_lora")):
        raise ValueError("Expected a complete native Qwen3-8B source with no LoRA/shared residual state.")
    profiles = set()
    for name in TARGETS:
        module = model.get_submodule(name)
        suffix = ".".join(name.split(".")[3:])
        if (not isinstance(module, VAELinear)
                or (module.out_features, module.in_features) != PROJECTION_SHAPES[suffix]
                or module.residual_stages != 2 or module.parallel_parts != 1
                or not module.parallel_stage_decode or getattr(module, "_parallel_stage_decoder", None) is None
                or module._parallel_stage_decoder.num_models != 2):
            raise ValueError(f"Expected the native packed two-stage projection: {name}.")
        forbidden = ("low_rank_a", "low_rank_b", "original_weight", "bias", "protected_input_weight",
                     "protected_output_indices", "protected_output_weight", "protected_output_qvalues",
                     "protected_output_scales", "protected_residual_indices")
        if (any(getattr(module, key, None) is not None for key in forbidden)
                or module.always_use_original or not module.temporary or module.has_sparse_residual()
                or module.has_protected_residual_vae() or module.trainable_decode
                or getattr(module, "_sparse_bit_binding", None) is not None):
            raise ValueError(f"Unsupported mutable/LoRA/bias/dense/residual payload: {name}.")
        profiles.add(_validate_payload_profile(module))
    if len(profiles) != 1:
        raise ValueError("The source must use one supported rotation/protection profile consistently.")
    profile = profiles.pop()
    if profile not in getattr(local_block, "SUPPORTED_SOURCE_PROFILES", ()):
        raise RuntimeError(f"The installed local_block core does not declare support for {profile}.")
    return profile


def select_calibration_prefix(train_batch, heldout_batch, data, count):
    if type(count) is not int or count not in (64, 512):
        raise ValueError("Select exactly the first 64 or 512 training documents.")
    full_plan = pilot_document_plan(train_batch, heldout_batch, data)
    selected = {key: value[:count] for key, value in train_batch.items()}
    plan = {
        "manifest_sha256": full_plan["manifest_sha256"], "train_document_indices": list(range(count)),
        "train_documents_sha256": _json_digest(data["train"]["documents"][:count]),
        "train_tokens": sum(full_plan["train_lengths"][:count]),
        "heldout_document_indices": list(range(512, 576)),
        "heldout_documents_sha256": _json_digest(data["heldout"]["documents"]),
        "heldout_tokens": sum(full_plan["heldout_lengths"]),
        "train_lengths": full_plan["train_lengths"][:count], "heldout_lengths": full_plan["heldout_lengths"],
    }
    if int(selected["attention_mask"].sum()) != plan["train_tokens"]:
        raise ValueError("Selected training prefix token count is inconsistent.")
    return selected, heldout_batch, plan


def _initialize_streams(teacher, student, train_batch, heldout_batch, device):
    prefixes = [module for model in (teacher, student)
                for module in (model.model.embed_tokens, model.model.rotary_emb)]
    try:
        for module in prefixes:
            module.to(device)
        train_stream = SequentialBlockCalibration(teacher, student, train_batch)
        heldout_stream = SequentialBlockCalibration(teacher, student, heldout_batch)
        return train_stream, heldout_stream
    finally:
        for module in prefixes:
            module.cpu()


def _recover_blocks(teacher, student, train_stream, heldout_stream, *, device, steps, lr,
                    microbatch_size, on_block):
    """Execute real blocks in strict order; the caller validates production scope."""
    if len(teacher.model.layers) != len(student.model.layers):
        raise ValueError("Teacher and student block counts differ.")
    for index, (teacher_block, student_block) in enumerate(zip(teacher.model.layers, student.model.layers)):
        if train_stream.layer_index != index or heldout_stream.layer_index != index:
            raise RuntimeError("Sequential calibration streams are out of order.")
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        teacher_block.to(device)
        try:
            train_documents = train_stream.prepare(teacher_block)
            heldout_documents = heldout_stream.prepare(teacher_block)
        finally:
            teacher_block.cpu()
        train = batch_block_documents(train_documents, microbatch_size=microbatch_size)
        heldout = batch_block_documents(heldout_documents, microbatch_size=microbatch_size)
        batch_shapes = {
            "train": [int(batch.hidden_states.shape[0]) for batch in train],
            "heldout": [int(batch.hidden_states.shape[0]) for batch in heldout],
        }
        del train_documents, heldout_documents
        student_block.to(device)
        try:
            def progress(row):
                if row["step"] % 10 == 0 or row["step"] == steps:
                    print(json.dumps({"block_index": index, "block_elapsed_seconds": time.perf_counter() - started,
                                      **row}), flush=True)

            result = optimize_local_block(student_block, train, heldout, steps=steps, lr=lr,
                                          progress_callback=progress)
            if result["settings"]["input_source"] != INPUT_SOURCE:
                raise RuntimeError("Core returned teacher-only metadata for sequential student inputs.")
            # The core must restore frozen BF16 inference before the next layer.
            train_stream.advance(student_block)
            heldout_stream.advance(student_block)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            result.update(block_index=index, elapsed_seconds=time.perf_counter() - started,
                          execution_batch_sizes=batch_shapes, input_source="rolling_native_student_hidden_states",
                          target_source="independent_original_teacher_next_hidden_states_before_final_norm",
                          native_streams_advanced=True)
            on_block(index, result)
        finally:
            student_block.cpu()
            for projection in PROJECTIONS:
                module = student_block.get_submodule(projection)
                module._build_parallel_stage_decode_plan()
                module.clear_decoded_weight_cache()
        del train, heldout, result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not train_stream.complete or not heldout_stream.complete:
        raise RuntimeError("Both streams must finish every block.")


def _expected_decoder_state(source_state, final_state, targets):
    if set(source_state) != set(final_state):
        raise RuntimeError("Sequential recovery changed the registered tensor inventory.")
    prefixes = tuple(name + "._parallel_stage_decoder." for name in targets)
    expected, changed = {}, []
    for name, original in source_state.items():
        current = final_state[name]
        if original.shape != current.shape or original.dtype != current.dtype:
            raise RuntimeError(f"Sequential recovery changed tensor shape/dtype: {name}.")
        if name.startswith(prefixes):
            expected[name] = current.detach().cpu().clone()
            if not torch.equal(original.reshape(-1).contiguous().view(torch.uint8),
                               expected[name].reshape(-1).contiguous().view(torch.uint8)):
                changed.append(name)
        else:
            _assert_state_equal({name: current}, {name: original}, "Frozen sequential payload")
            expected[name] = original
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in changed):
            raise RuntimeError(f"No decoder update survived the original storage dtype: {prefix}")
    return expected, changed


def run_sequential_recovery(checkpoint_dir, output_dir, *, train_documents, steps, lr):
    configuration = recovery_configuration(train_documents=train_documents, steps=steps, lr=lr)
    _require_core_input_metadata()
    source, output = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents:
        raise ValueError("Recovery output must be outside the immutable source checkpoint.")
    meta, audit = _validate_checkpoint(source)
    if not audit["budget_passed"]:
        raise ValueError("Sequential recovery requires a complete budget-compliant source checkpoint.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "blocks").mkdir()
    report_path, device = output / "recovery.json", torch.device("cuda:0")
    repo = Path(__file__).resolve().parents[2]
    started = time.perf_counter()
    report = {
        "protocol_version": 1, "status": "preparing", "configuration": configuration,
        "configuration_sha256": _json_digest(configuration), "source_checkpoint_dir": str(source),
        "source_checkpoint_id": meta["checkpoint_id"], "source_bit_budget": audit,
        "source_files_sha256": {name: _sha(source / name) for name in
                                ("checkpoint_meta.json", "pytorch_model.bin", "config.json")},
        "code_sha256": {name: _sha(repo / name) for name in CODE_FILES}, "blocks": {},
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
        train, heldout, data = build_calibration_batches(
            tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
        train, heldout, plan = select_calibration_prefix(train, heldout, data, train_documents)
        report.update(data=data, document_plan=plan, manifest_sha256=plan["manifest_sha256"])
        student, loaded_meta, _ = load_v6_model_checkpoint(
            str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True, expected_kind="final_model")
        if loaded_meta["checkpoint_id"] != meta["checkpoint_id"]:
            raise RuntimeError("Source identity changed during loading.")
        student.eval().requires_grad_(False)
        report["source_profile"] = validate_sequential_source(student, loaded_meta)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device="cpu")
        train_stream, heldout_stream = _initialize_streams(teacher, student, train, heldout, device)
        report["initialization_passes"] = {"train": train_stream.initialization_passes,
                                           "heldout": heldout_stream.initialization_passes}
        report["status"] = "running"
        _write_report(report_path, report)

        def on_block(index, metrics):
            settings = metrics["settings"]
            if settings["train_tokens"] != plan["train_tokens"] or settings["heldout_tokens"] != plan["heldout_tokens"]:
                raise RuntimeError("Block optimization changed the fixed token inventory.")
            metrics.update(source_checkpoint_id=meta["checkpoint_id"], manifest_sha256=plan["manifest_sha256"],
                           configuration_sha256=report["configuration_sha256"])
            relative = Path("blocks") / f"block_{index:02d}.json"
            _write_report(output / relative, metrics)
            report["blocks"][str(index)] = {"report_file": str(relative), "report_sha256": _sha(output / relative),
                                            "before_native": metrics["before_native"], "after_cast": metrics["after_cast"]}
            _write_report(report_path, report)

        _recover_blocks(teacher, student, train_stream, heldout_stream, device=device, steps=steps, lr=lr,
                        microbatch_size=8, on_block=on_block)
        if set(report["blocks"]) != {str(index) for index in range(36)}:
            raise RuntimeError("Every block must complete before final saving.")
        del teacher, train_stream, heldout_stream
        gc.collect()
        torch.cuda.empty_cache()
        original_state = _load_state(source)
        expected, changed = _expected_decoder_state(original_state, student.state_dict(), TARGETS)
        provenance = {key: report[key] for key in (
            "protocol_version", "configuration", "configuration_sha256", "source_checkpoint_dir", "source_checkpoint_id",
            "source_files_sha256", "source_profile", "code_sha256", "data", "document_plan", "manifest_sha256", "blocks")}
        provenance.update(purpose="sequential local decoder recovery; inference only; no cross-layer gradients or resume state",
                          changed_decoder_tensor_count=len(changed))
        report["status"] = "saving"
        _write_report(report_path, report)
        finalized = _save_final(student, output / "final_model", meta, expected, provenance, tokenizer)
        for key in ("original_linear_weights", "user_linear_bits", "packed_linear_bits", "code_padding_bits", "full_model_tensor_bytes"):
            if finalized["bit_budget"][key] != audit[key]:
                raise RuntimeError(f"Sequential decoder recovery changed actual storage: {key}.")
        report.update(status="complete", final_checkpoint_dir=str(output / "final_model"),
                      final_checkpoint_id=finalized["checkpoint_id"], final_bit_budget=finalized["bit_budget"],
                      serialized_tensors_compared=finalized["serialized_tensors_compared"],
                      reloaded_tensors_compared=finalized["reloaded_tensors_compared"],
                      changed_decoder_tensors=changed, nondecoder_bytes_unchanged=True,
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        print(json.dumps({"status": "complete", "report": str(report_path),
                          "checkpoint": report["final_checkpoint_dir"], "checkpoint_id": finalized["checkpoint_id"]}), flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        _write_report(report_path, report)
        raise
