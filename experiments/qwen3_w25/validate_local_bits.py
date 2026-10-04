"""Calibrate and validate fixed-decoder packed-bit recovery on a real Qwen3 Linear."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from experiments.linear_output.calibration import capture_inputs
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_bits import calibrate_local_bits_cpu, optimize_local_bits
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.checkpoint_bpw import audit_checkpoint
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _saved_changes(source, destination, allowed):
    old = torch.load(source / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    new = torch.load(destination / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    if old.keys() != new.keys():
        raise RuntimeError("Packed-bit recovery changed the serialized tensor inventory.")
    changed = []
    for name, before in old.items():
        after = new[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(f"Serialized shape/dtype changed: {name}")
        if not torch.equal(before.reshape(-1).contiguous().view(torch.uint8),
                           after.reshape(-1).contiguous().view(torch.uint8)):
            if name not in allowed:
                raise RuntimeError(f"Frozen serialized tensor changed: {name}")
            changed.append(name)
    if not changed:
        raise RuntimeError("No packed-code update survived serialization.")
    return changed, new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("calibrate", "validate"):
        entry = sub.add_parser(name)
        entry.add_argument("--checkpoint_dir", type=Path, required=True)
        entry.add_argument("--base_model_path", default="/home/star/models/Qwen/Qwen3-8B")
        entry.add_argument("--target", required=True)
        entry.add_argument("--output_dir", type=Path, required=True)
        if name == "validate":
            entry.add_argument("--calibration_json", type=Path, required=True)
            entry.add_argument("--device", default="cuda:0")
            entry.add_argument("--steps", type=int, default=128)
            entry.add_argument("--round_steps", type=int, default=128)
            entry.add_argument("--lr", type=float, required=True)
            entry.add_argument("--train_documents", type=int, default=32)
            entry.add_argument("--heldout_documents", type=int, default=32)
            entry.add_argument("--max_length", type=int, default=256)
            entry.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    meta, before_audit = _validate_checkpoint(args.checkpoint_dir)
    if not before_audit["budget_passed"] or args.target not in meta["compressed_targets"]:
        raise ValueError("Expected a complete, budget-compliant no-LoRA source and compressed target.")
    if Path(args.base_model_path).resolve() != Path(meta["base_model_path"]).resolve():
        raise ValueError("Teacher/tokenizer path must match the source checkpoint's original model.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "calibrate":
        model, _, _ = load_v6_model_checkpoint(
            str(args.checkpoint_dir), base_model_path=args.base_model_path, map_location="cpu", strict=True)
        calibration = calibrate_local_bits_cpu(model.get_submodule(args.target))
        report = {"source_checkpoint": str(args.checkpoint_dir), "source_checkpoint_id": meta["checkpoint_id"],
                  "target": args.target, "calibration": calibration}
        _write(args.output_dir / "calibration.json", report)
        print(json.dumps({"target": args.target, "scales": [s["scale"] for s in calibration["stages"]]}), flush=True)
        return

    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    if (calibration["source_checkpoint_id"] != meta["checkpoint_id"]
            or calibration["target"] != args.target
            or Path(calibration["source_checkpoint"]).resolve() != args.checkpoint_dir):
        raise ValueError("CPU coordinate calibration belongs to a different source or target.")
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
    print("Capturing disjoint teacher calibration documents", flush=True)
    teacher = load_frozen_base_reference_model(args.base_model_path, access_token=None, device=args.device)
    reference = teacher.get_submodule(args.target).weight.detach().cpu().clone()
    train_inputs = capture_inputs(teacher, args.target, train_batch, microbatch_size=1)
    heldout_inputs = capture_inputs(teacher, args.target, heldout_batch, microbatch_size=1)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    model, _, _ = load_v6_model_checkpoint(
        str(args.checkpoint_dir), base_model_path=args.base_model_path, map_location="cpu", strict=True)
    model.eval().requires_grad_(False)
    layer = model.get_submodule(args.target).to(args.device)
    print("Optimizing only packed codes:", args.target, flush=True)
    result = optimize_local_bits(
        layer, reference, train_inputs, heldout_inputs, steps=args.steps, lr=args.lr,
        round_steps=args.round_steps, chunk_size=256, seed=args.seed,
        progress_callback=lambda row: print(json.dumps(row), flush=True))
    with torch.no_grad():
        expected_output = layer(heldout_inputs).detach().cpu()
    layer.clear_decoded_weight_cache()
    model.cpu()
    destination = args.output_dir / "checkpoint"
    saved = save_v6_full_checkpoint(
        model, str(destination), checkpoint_kind="final_model",
        compressed_targets=meta["compressed_targets"], pending_dense_targets=(), skip_targets=(),
        train_mode="none", after_category_mode="none", lora_config=None,
        base_model_path=args.base_model_path, tokenizer=tokenizer,
        extra_meta={"local_bit_validation": {"source_checkpoint_id": meta["checkpoint_id"],
                                            "target": args.target, "settings": result["settings"]}})
    if saved["checkpoint_id"] == meta["checkpoint_id"]:
        raise RuntimeError("New inference checkpoint reused the source identity.")
    del layer, model
    gc.collect()
    torch.cuda.empty_cache()
    allowed = {args.target + "." + key for key in result["evidence"]["code_state_keys"]}
    changed, serialized = _saved_changes(args.checkpoint_dir, destination, allowed)
    reloaded, reloaded_meta, _ = load_v6_model_checkpoint(
        str(destination), base_model_path=args.base_model_path, map_location="cpu", strict=True,
        expected_kind="final_model")
    compared = _assert_state_equal(reloaded.state_dict(), serialized, "Independent packed-bit checkpoint reload")
    if reloaded_meta.get("immutable_resume_contract") is not None:
        raise RuntimeError("Local inference result inherited an exact-resume contract.")
    for field in ("compressed_targets", "pending_dense_targets", "skip_targets", "converted_modules"):
        if reloaded_meta[field] != meta[field]:
            raise RuntimeError(f"Packed-bit recovery changed module metadata: {field}")
    recovered = reloaded.get_submodule(args.target).to(args.device)
    with torch.no_grad():
        actual_output = recovered(heldout_inputs).detach().cpu()
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    audit = audit_checkpoint(destination, max_bpw=2.5, max_lora_rank=0)
    for field in ("user_linear_bits", "packed_linear_bits", "original_linear_weights"):
        if audit[field] != before_audit[field]:
            raise RuntimeError(f"Packed-bit recovery changed storage accounting: {field}")
    if not audit["budget_passed"]:
        raise RuntimeError("Saved packed-bit validation checkpoint failed the bit budget.")
    report = {"purpose": "local_path_validation_not_full_model_accuracy", "target": args.target,
              "source_checkpoint": str(args.checkpoint_dir), "source_checkpoint_id": meta["checkpoint_id"],
              "saved_checkpoint": str(destination), "saved_checkpoint_id": saved["checkpoint_id"],
              "serialized_changed_tensors": changed, "reloaded_tensors_compared": compared,
              "independent_reload_output_exact": True, "coordinate_calibration": calibration,
              "data": data, "local_metrics": result, "bit_budget": audit,
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}}
    _write(args.output_dir / "validation.json", report)
    print(json.dumps({"status": "validated", "target": args.target, "before": result["before_native"],
                      "after": result["after_native"], "evidence": result["evidence"]}), flush=True)


if __name__ == "__main__":
    main()
