"""Two-shard, fixed-source block recovery pilot; metrics only, no model export."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.block_calibration import capture_block_batches
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_block import PROJECTIONS, _frozen_digest, _squared_error, optimize_local_block
from experiments.qwen3_w25.local_recovery import BASE_MODEL, validate_recovery_source
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


SOURCE_CHECKPOINT_ID = "c24d4b96-01f6-43d7-8df4-8a9b15ca3469"
BLOCKS = (3, 9, 18, 27)
STEPS = (64, 192)
LR = 1.25e-5
CODE_FILES = (
    "experiments/qwen3_w25/block_pilot.py", "experiments/qwen3_w25/pilot_local_block.py",
    "experiments/qwen3_w25/local_block.py", "experiments/qwen3_w25/block_weight_graph.py",
    "experiments/qwen3_w25/block_calibration.py", "experiments/qwen3_w25/calibration.py",
    "experiments/qwen3_w25/local_decoder.py", "experiments/qwen3_w25/local_recovery.py",
    "litebsq/vae_linear.py", "litebsq/autoencoder.py", "litebsq/parallel_layers.py",
    "litebsq/packed_bit_linear.py", "rotation/weight_preprocess.py",
)


def shard_blocks(shard_index):
    if type(shard_index) is not int or shard_index not in (0, 1):
        raise ValueError("shard_index must be 0 or 1; this pilot has exactly two shards.")
    return BLOCKS[shard_index::2]


def _config():
    return {"protocol_version": 1, "blocks": list(BLOCKS), "steps": list(STEPS), "lr": LR,
            "num_shards": 2, "seed": 31, "train_documents": 64, "heldout_documents": 64,
            "selection_documents": 32, "confirmation_documents": 32, "max_length": 256,
            "teacher_inputs": "fixed_original_teacher_hidden_states", "teacher_microbatch_size": 1,
            "objective": "block_output_mse", "gradient_path": "two_phase_fp32_weight_gradient_then_decoder_vjp",
            "optimizer": "AdamW", "optimizer_dtype": "float32", "weight_decay": 0.0,
            "betas": [0.9, 0.999], "eps": 1e-8, "projection_dtype": "torch.bfloat16",
            "selection_rule": "minimum equal-block mean of selection after/before token-MSE ratios; ties prefer fewer steps",
            "confirmation_used_for_selection": False, "checkpoint_written": False,
            "allow_tf32": False, "deterministic_algorithms": True}


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _document_plan(train, heldout, data):
    if data["seed"] != 31 or data["max_length"] != 256:
        raise ValueError("Pilot calibration requires seed31/max_length256.")
    lengths = {}
    for split, batch in (("train", train), ("heldout", heldout)):
        documents = data[split]["documents"]
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if (len(documents) != 64 or ids.ndim != 2 or ids.shape != mask.shape or len(ids) != 64
                or ids.shape[1] > 256 or not bool(((mask == 0) | (mask == 1)).all())):
            raise ValueError(f"Expected exactly 64 correctly masked calibration documents: {split}.")
        lengths[split] = [row["valid_tokens"] for row in documents]
        if (lengths[split] != mask.sum(dim=1).tolist() or any(n < 1 for n in lengths[split])
                or sum(lengths[split]) != data[split]["valid_tokens"]):
            raise ValueError(f"Calibration document token counts disagree: {split}.")
    all_documents = data["train"]["documents"] + data["heldout"]["documents"]
    for key in ("row_index", "group_sha256", "token_sha256"):
        if len({row[key] for row in all_documents}) != 128:
            raise ValueError(f"Calibration documents are not disjoint: {key}.")
    return {"train_lengths": lengths["train"], "heldout_lengths": lengths["heldout"],
            "selection_indices": list(range(32)), "confirmation_indices": list(range(32, 64)),
            "manifest_sha256": _json_hash(data)}


@torch.no_grad()
def _native_documents(block, batches, device):
    lengths, values = [], []
    for batch in batches:
        if batch.hidden_states.shape[0] != 1:
            raise ValueError("Pilot document metrics require one captured document per batch.")
        length = int(batch.valid_mask.sum())
        value = float(_squared_error(block, batch, device)) / (length * block.hidden_size)
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("Non-finite/negative native block document MSE.")
        lengths.append(length)
        values.append(value)
    return {"per_document_mse": values, "document_tokens": lengths, "valid_tokens": sum(lengths),
            "token_mean_mse": sum(value * length for value, length in zip(values, lengths)) / sum(lengths),
            "document_mean_mse": sum(values) / len(values)}


def _comparison(before, after):
    if before["token_mean_mse"] <= 0:
        raise ValueError("Relative selection requires a positive baseline block MSE.")
    return {"before": before, "after": after,
            "after_before_ratio": after["token_mean_mse"] / before["token_mean_mse"]}


def _ratio(comparison, expected_documents=32):
    for metrics in (comparison["before"], comparison["after"]):
        values, lengths = metrics["per_document_mse"], metrics["document_tokens"]
        if (len(values) != expected_documents or len(lengths) != expected_documents or any(type(n) is not int or n < 1 for n in lengths)
                or any(not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError(f"Expected {expected_documents} finite document MSEs and positive token counts.")
        mean = sum(v * n for v, n in zip(values, lengths)) / sum(lengths)
        if metrics["valid_tokens"] != sum(lengths) or not math.isclose(mean, metrics["token_mean_mse"], rel_tol=1e-12, abs_tol=0):
            raise ValueError("Reported token-weighted document MSE is inconsistent.")
    if comparison["before"]["document_tokens"] != comparison["after"]["document_tokens"]:
        raise ValueError("Before/after document identities have different token counts.")
    actual = _comparison(comparison["before"], comparison["after"])["after_before_ratio"]
    if not math.isclose(actual, comparison["after_before_ratio"], rel_tol=1e-12, abs_tol=0):
        raise ValueError("Reported before/after ratio is inconsistent.")
    return actual


def select_block_steps(block_results):
    """Choose solely by four equally weighted selection ratios, then report confirmation."""
    if set(block_results) != {str(index) for index in BLOCKS}:
        raise ValueError("Selection requires complete coverage of blocks 3, 9, 18 and 27.")
    for block in block_results.values():
        if set(block["arms"]) != {str(steps) for steps in STEPS}:
            raise ValueError("Every block requires both 64-step and 192-step arms.")
    scores = {}
    for steps in STEPS:
        ratios = [_ratio(block_results[str(index)]["arms"][str(steps)]["native_selection"]) for index in BLOCKS]
        scores[str(steps)] = {"mean_selection_ratio": sum(ratios) / 4, "per_block_ratios": dict(zip(map(str, BLOCKS), ratios))}
    selected = min(STEPS, key=lambda steps: (scores[str(steps)]["mean_selection_ratio"], steps))
    confirmation = [_ratio(block_results[str(index)]["arms"][str(selected)]["native_confirmation"]) for index in BLOCKS]
    return {"selected_steps": selected, "selection_scores": scores,
            "selected_confirmation": {"mean_ratio": sum(confirmation) / 4,
                                      "per_block_ratios": dict(zip(map(str, BLOCKS), confirmation)),
                                      "used_for_selection": False}}


def _restore(block, initial, expected_hash):
    block.load_state_dict(initial, strict=True)
    for name in PROJECTIONS:
        module = block.get_submodule(name)
        module._build_parallel_stage_decode_plan()
        module.clear_decoded_weight_cache()
    actual = _frozen_digest(block, set())
    if actual != expected_hash or any(p.grad is not None for p in block.parameters()):
        raise RuntimeError("Pilot failed to restore the full original block state and clear gradients.")
    return actual


def run_block_pilot(checkpoint_dir, output_dir, *, shard_index):
    indices = shard_blocks(shard_index)
    source, output = Path(checkpoint_dir).resolve(), Path(output_dir).resolve()
    meta, audit = _validate_checkpoint(source)
    if meta["checkpoint_id"] != SOURCE_CHECKPOINT_ID or not audit["budget_passed"]:
        raise ValueError("Block pilot requires the fixed original down3 final checkpoint.")
    if Path(meta["base_model_path"]).resolve() != Path(BASE_MODEL).resolve():
        raise ValueError("Source must reference the fixed original Qwen3-8B teacher.")
    output.mkdir(parents=True, exist_ok=False)
    path, device, started = output / "pilot.json", torch.device("cuda:0"), time.perf_counter()
    report = {"status": "running", "purpose": "block_recovery_step_pilot_not_downstream_accuracy",
              "configuration": _config(), "shard_index": shard_index, "block_indices": list(indices),
              "source_checkpoint_dir": str(source), "source_checkpoint_id": meta["checkpoint_id"],
              "source_files_sha256": {name: _file_hash(source / name) for name in ("checkpoint_meta.json", "config.json")},
              "source_bit_budget": audit, "device": str(device), "blocks": {},
              "code_sha256": {name: _file_hash(Path(__file__).resolve().parents[2] / name) for name in CODE_FILES}}
    _write(path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_batch, heldout_batch, data = build_calibration_batches(
            tokenizer, seed=31, train_documents=64, heldout_documents=64, max_length=256)
        plan = _document_plan(train_batch, heldout_batch, data)
        report.update(data=data, document_plan=plan)
        capture_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundles = {}
        for index in indices:
            bundles[index] = (capture_block_batches(teacher, index, train_batch),
                              capture_block_batches(teacher, index, heldout_batch))
        torch.cuda.synchronize(device)
        report["teacher_capture"] = {"seconds": time.perf_counter() - capture_start,
                                     "passes_over_128_documents": len(indices), "document_forwards": 128 * len(indices),
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True)
        if loaded_meta["checkpoint_id"] != meta["checkpoint_id"]:
            raise RuntimeError("Source checkpoint identity changed during loading.")
        model.eval().requires_grad_(False)
        validate_recovery_source(model, loaded_meta)
        _write(path, report)
        for index in indices:
            block = model.model.layers[index].to(device)
            train, heldout = bundles.pop(index)
            if ([int(batch.valid_mask.sum()) for batch in train] != plan["train_lengths"]
                    or [int(batch.valid_mask.sum()) for batch in heldout] != plan["heldout_lengths"]):
                raise ValueError("Captured block documents disagree with the calibration manifest.")
            initial = {name: value.detach().cpu().clone() for name, value in block.state_dict().items()}
            source_hash = _frozen_digest(block, set())
            block_result = {"source_block_sha256": source_hash, "arms": {}}
            report["blocks"][str(index)] = block_result
            try:
                for steps in STEPS:
                    restored_before = _restore(block, initial, source_hash)
                    torch.manual_seed(31)
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    arm_start = time.perf_counter()
                    baseline_train = _native_documents(block, train, device)
                    baseline_selection = _native_documents(block, heldout[:32], device)
                    baseline_confirmation = _native_documents(block, heldout[32:], device)
                    progress_rows = []

                    def progress(row):
                        row = {**row, "elapsed_seconds": time.perf_counter() - arm_start}
                        progress_rows.append(row)
                        if row["step"] % 16 == 0 or row["step"] == steps:
                            print(json.dumps({"block": index, "arm_steps": steps, **row}), flush=True)

                    metrics = optimize_local_block(block, train, heldout, steps=steps, lr=LR, progress_callback=progress)
                    train_after = _native_documents(block, train, device)
                    selection_after = _native_documents(block, heldout[:32], device)
                    confirmation_after = _native_documents(block, heldout[32:], device)
                    torch.cuda.synchronize(device)
                    arm_result = {"steps": steps, "lr": LR, "local_metrics": metrics, "step_telemetry": progress_rows,
                                  "native_train": _comparison(baseline_train, train_after),
                                  "native_selection": _comparison(baseline_selection, selection_after),
                                  "native_confirmation": _comparison(baseline_confirmation, confirmation_after),
                                  "elapsed_seconds": time.perf_counter() - arm_start,
                                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                                  "restored_sha256_before_arm": restored_before,
                                  "restored_sha256_after_arm": _restore(block, initial, source_hash)}
                    block_result["arms"][str(steps)] = arm_result
                    _write(path, report)
            finally:
                block_result["final_restored_sha256"] = _restore(block, initial, source_hash)
                block.cpu()
                for name in PROJECTIONS:
                    block.get_submodule(name)._build_parallel_stage_decode_plan()
                    block.get_submodule(name).clear_decoded_weight_cache()
            del initial, train, heldout
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        _write(path, report)
        return report
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                      elapsed_seconds=time.perf_counter() - started)
        _write(path, report)
        raise


def finalize_block_pilot(shard_dirs, output_path):
    if len(shard_dirs) != 2:
        raise ValueError("Exactly two completed block pilot shards are required.")
    reports = [json.loads((Path(directory) / "pilot.json").read_text(encoding="utf-8")) for directory in shard_dirs]
    if {report["shard_index"] for report in reports} != {0, 1}:
        raise ValueError("Duplicate or missing block pilot shard index.")
    shared = ("source_checkpoint_dir", "source_checkpoint_id", "source_files_sha256", "source_bit_budget",
              "configuration", "data", "document_plan", "code_sha256")
    combined = {}
    for report in reports:
        if report["status"] != "complete" or report["configuration"] != _config() or report["source_checkpoint_id"] != SOURCE_CHECKPOINT_ID:
            raise ValueError("Incomplete or wrong-protocol/source block pilot.")
        if any(report[key] != reports[0][key] for key in shared) or set(report["code_sha256"]) != set(CODE_FILES):
            raise ValueError("Pilot shards disagree on source, data, protocol or implementation.")
        expected = shard_blocks(report["shard_index"])
        if report["block_indices"] != list(expected) or set(report["blocks"]) != set(map(str, expected)):
            raise ValueError("Pilot shard has missing, duplicate or unexpected blocks.")
        if report["document_plan"]["manifest_sha256"] != _json_hash(report["data"]):
            raise ValueError("Pilot data manifest hash mismatch.")
        plan, data = report["document_plan"], report["data"]
        if (data["seed"] != 31 or data["max_length"] != 256 or plan["selection_indices"] != list(range(32))
                or plan["confirmation_indices"] != list(range(32, 64))):
            raise ValueError("Pilot calibration or held-out partition differs from the fixed protocol.")
        for split in ("train", "heldout"):
            lengths = [row["valid_tokens"] for row in data[split]["documents"]]
            if len(lengths) != 64 or lengths != plan[split + "_lengths"] or sum(lengths) != data[split]["valid_tokens"]:
                raise ValueError("Pilot calibration document counts disagree with the manifest.")
        documents = data["train"]["documents"] + data["heldout"]["documents"]
        if any(len({row[key] for row in documents}) != 128 for key in ("row_index", "group_sha256", "token_sha256")):
            raise ValueError("Pilot calibration document identities overlap.")
        for index, block in report["blocks"].items():
            if block["final_restored_sha256"] != block["source_block_sha256"] or set(block["arms"]) != set(map(str, STEPS)):
                raise ValueError("Block did not finish both arms and restore its full source state.")
            for steps, arm in block["arms"].items():
                if (arm["steps"] != int(steps) or arm["lr"] != LR
                        or arm["restored_sha256_before_arm"] != block["source_block_sha256"]
                        or arm["restored_sha256_after_arm"] != block["source_block_sha256"]):
                    raise ValueError("Pilot arm did not start/end at the exact source or used the wrong recipe.")
                metrics, evidence = arm["local_metrics"], arm["local_metrics"]["evidence"]
                if (metrics["settings"]["steps"] != int(steps) or metrics["settings"]["lr"] != LR
                        or metrics["settings"]["gradient_path"] != _config()["gradient_path"]
                        or metrics["settings"]["train_tokens"] != data["train"]["valid_tokens"]
                        or metrics["settings"]["heldout_tokens"] != data["heldout"]["valid_tokens"]
                        or evidence["frozen_sha256_before"] != evidence["frozen_sha256_after"]
                        or not evidence["parameter_ids_unchanged"] or not evidence["tensor_schema_unchanged"]
                        or evidence["persistent_storage_bits_before"] != evidence["persistent_storage_bits_after"]):
                    raise ValueError("Pilot arm is missing the fixed-payload/optimizer protocol evidence.")
                if len(metrics["training_loss"]) != int(steps) or [row["step"] for row in arm["step_telemetry"]] != list(range(1, int(steps) + 1)):
                    raise ValueError("Pilot arm has incomplete optimization telemetry.")
                _ratio(arm["native_train"], expected_documents=64)
                if arm["native_train"]["before"]["document_tokens"] != plan["train_lengths"]:
                    raise ValueError("Training document metrics disagree with the manifest.")
                for role, lengths in (("selection", report["document_plan"]["heldout_lengths"][:32]),
                                      ("confirmation", report["document_plan"]["heldout_lengths"][32:])):
                    comparison = arm["native_" + role]
                    _ratio(comparison)
                    if comparison["before"]["document_tokens"] != lengths:
                        raise ValueError("Selection/confirmation document split does not match the manifest.")
            combined[index] = block
    result = {"status": "complete", "purpose": "block_pilot_selection_not_downstream_accuracy",
              **{key: reports[0][key] for key in shared}, "shard_dirs": [str(Path(path).resolve()) for path in shard_dirs],
              "blocks": combined, **select_block_steps(combined)}
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write(destination, result)
    return result
