"""Validate seven-decoder reconstruction inside one frozen-input Qwen3 block."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from experiments.liftquant_recovery.recovery_runtime import tree_to
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.block_calibration import capture_block_batches
from experiments.qwen3_w25.block_batching import batch_block_documents
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_block import optimize_local_block
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


@torch.no_grad()
def _outputs(block, batches, device):
    return [block(batch.hidden_states.to(device), **tree_to(batch.forward_kwargs, device))[0].cpu()
            for batch in batches]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--base_model_path", default="/home/star/models/Qwen/Qwen3-8B")
    parser.add_argument("--block_index", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.25e-5)
    parser.add_argument("--train_documents", type=int, default=32)
    parser.add_argument("--heldout_documents", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--microbatch_size", type=int, default=1)
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    meta, before_audit = _validate_checkpoint(args.checkpoint_dir)
    if not before_audit["budget_passed"] or not 0 <= args.block_index < 36:
        raise ValueError("Expected a complete, budget-compliant source and block index in [0, 35].")
    if Path(args.base_model_path).resolve() != Path(meta["base_model_path"]).resolve():
        raise ValueError("Teacher/tokenizer path must match the original source model.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_batch, heldout_batch, data = build_calibration_batches(
        tokenizer, seed=args.seed, train_documents=args.train_documents,
        heldout_documents=args.heldout_documents, max_length=args.max_length)
    print("Capturing teacher block inputs, outputs and actual forward kwargs", flush=True)
    teacher = load_frozen_base_reference_model(args.base_model_path, access_token=None, device=args.device)
    train = capture_block_batches(teacher, args.block_index, train_batch)
    heldout = capture_block_batches(teacher, args.block_index, heldout_batch)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    model, _, _ = load_v6_model_checkpoint(
        str(args.checkpoint_dir), base_model_path=args.base_model_path, map_location="cpu", strict=True)
    model.eval().requires_grad_(False)
    train_execution = batch_block_documents(train, microbatch_size=args.microbatch_size)
    heldout_execution = batch_block_documents(heldout, microbatch_size=args.microbatch_size)
    prefix = f"model.layers.{args.block_index}"
    block = model.get_submodule(prefix).to(args.device)
    print("Optimizing seven existing decoders inside", prefix, flush=True)
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    progress = []

    def report_progress(row):
        row = {**row, "elapsed_seconds": time.perf_counter() - started}
        progress.append(row)
        print(json.dumps(row), flush=True)

    result = optimize_local_block(
        block, train_execution, heldout_execution, steps=args.steps, lr=args.lr,
        progress_callback=report_progress)
    torch.cuda.synchronize(args.device)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
    result["step_telemetry"] = progress
    result["execution_batch_sizes"] = {
        "train": [int(batch.hidden_states.shape[0]) for batch in train_execution],
        "heldout": [int(batch.hidden_states.shape[0]) for batch in heldout_execution],
    }
    expected_output = _outputs(block, heldout, args.device)
    for module in block.modules():
        if hasattr(module, "clear_decoded_weight_cache"):
            module.clear_decoded_weight_cache()
    model.cpu()
    destination = args.output_dir / "checkpoint"
    saved = save_v6_full_checkpoint(
        model, str(destination), checkpoint_kind="final_model",
        compressed_targets=meta["compressed_targets"], pending_dense_targets=(), skip_targets=(),
        train_mode="none", after_category_mode="none", lora_config=None,
        base_model_path=args.base_model_path, tokenizer=tokenizer,
        extra_meta={"local_block_validation": {"source_checkpoint_id": meta["checkpoint_id"],
                                              "block_index": args.block_index, "settings": result["settings"]}})
    if saved["checkpoint_id"] == meta["checkpoint_id"]:
        raise RuntimeError("New inference checkpoint reused the source identity.")
    del block, model
    gc.collect()
    torch.cuda.empty_cache()
    old = torch.load(args.checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    new = torch.load(destination / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    if old.keys() != new.keys():
        raise RuntimeError("Block recovery changed the serialized tensor inventory.")
    allowed = tuple(name + "._parallel_stage_decoder." for name in meta["compressed_targets"]
                    if name.startswith(prefix + "."))
    changed = []
    for name, before in old.items():
        after = new[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(f"Serialized shape/dtype changed: {name}")
        if not torch.equal(before.reshape(-1).contiguous().view(torch.uint8),
                           after.reshape(-1).contiguous().view(torch.uint8)):
            if not name.startswith(allowed):
                raise RuntimeError(f"Frozen serialized tensor changed: {name}")
            changed.append(name)
    if not changed:
        raise RuntimeError("No decoder update survived serialization.")
    reloaded, new_meta, _ = load_v6_model_checkpoint(
        str(destination), base_model_path=args.base_model_path, map_location="cpu", strict=True,
        expected_kind="final_model")
    compared = _assert_state_equal(reloaded.state_dict(), new, "Independent local block checkpoint reload")
    for field in ("compressed_targets", "pending_dense_targets", "skip_targets", "converted_modules"):
        if new_meta[field] != meta[field]:
            raise RuntimeError(f"Block recovery changed module metadata: {field}")
    if new_meta.get("immutable_resume_contract") is not None:
        raise RuntimeError("Local inference result inherited an exact-resume contract.")
    recovered = reloaded.get_submodule(prefix).to(args.device)
    actual_output = _outputs(recovered, heldout, args.device)
    for actual, expected in zip(actual_output, expected_output, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    audit = audit_checkpoint(destination, max_bpw=2.5, max_lora_rank=0)
    for field in ("user_linear_bits", "packed_linear_bits", "original_linear_weights"):
        if audit[field] != before_audit[field]:
            raise RuntimeError(f"Block recovery changed storage accounting: {field}")
    if not audit["budget_passed"]:
        raise RuntimeError("Saved block validation checkpoint failed the bit budget.")
    report = {"purpose": "local_block_path_validation_not_full_model_accuracy",
              "block_index": args.block_index, "source_checkpoint": str(args.checkpoint_dir),
              "source_checkpoint_id": meta["checkpoint_id"], "saved_checkpoint": str(destination),
              "saved_checkpoint_id": saved["checkpoint_id"], "changed_tensors": changed,
              "reloaded_tensors_compared": compared, "independent_reload_output_exact": True,
              "data": data, "local_metrics": result, "bit_budget": audit,
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}}
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "validated", "block_index": args.block_index,
                      "before": result["before_native"], "after": result["after_cast"],
                      "evidence": result["evidence"]}), flush=True)


if __name__ == "__main__":
    main()
