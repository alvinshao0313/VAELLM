"""Exercise fixed-code local decoder recovery on one real Qwen3 Linear."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from experiments.linear_output.calibration import capture_inputs
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_decoder import optimize_local_decoder
from litebsq.vae_linear import VAELinear
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.checkpoint_v6 import load_v6_cat_runtime_state, save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def _verify_payloads(source, destination, target):
    old = torch.load(source / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    new = torch.load(destination / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    if old.keys() != new.keys():
        raise RuntimeError("Local recovery changed the serialized tensor inventory.")
    changed = []
    for name, tensor in old.items():
        value = new[name]
        if tensor.shape != value.shape or tensor.dtype != value.dtype:
            raise RuntimeError(f"Local recovery changed tensor shape/dtype: {name}")
        if not torch.equal(tensor, value):
            if not name.startswith(target + "._parallel_stage_decoder."):
                raise RuntimeError(f"Frozen serialized tensor changed: {name}")
            changed.append(name)
    if not changed:
        raise RuntimeError("No decoder update survived the original storage dtype.")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--base_model_path", default="/home/star/models/Qwen/Qwen3-8B")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.25e-5)
    parser.add_argument("--train_documents", type=int, default=2)
    parser.add_argument("--heldout_documents", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
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
        heldout_documents=args.heldout_documents, max_length=args.max_length,
    )
    print("Capturing disjoint teacher calibration documents", flush=True)
    teacher = load_frozen_base_reference_model(args.base_model_path, access_token=None, device=args.device)
    reference_weight = teacher.get_submodule(args.target).weight.detach().cpu().clone()
    train_inputs = capture_inputs(teacher, args.target, train_batch, microbatch_size=1)
    heldout_inputs = capture_inputs(teacher, args.target, heldout_batch, microbatch_size=1)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    model, meta, _ = load_v6_model_checkpoint(
        str(args.checkpoint_dir), base_model_path=args.base_model_path, map_location="cpu", strict=True,
    )
    model.eval().requires_grad_(False)
    layer = model.get_submodule(args.target)
    if not isinstance(layer, VAELinear) or args.target not in meta["compressed_targets"]:
        raise ValueError("The target must be an already compressed VAELinear.")
    layer.to(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    print("Optimizing only", args.target, flush=True)
    result = optimize_local_decoder(
        layer, reference_weight, train_inputs, heldout_inputs,
        steps=args.steps, lr=args.lr, chunk_size=256,
        progress_callback=lambda row: print(json.dumps(row), flush=True),
    )
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    with torch.no_grad():
        expected_output = layer(heldout_inputs).detach().cpu()
    layer.clear_decoded_weight_cache()
    model.cpu()
    runtime = load_v6_cat_runtime_state(str(args.checkpoint_dir), map_location="cpu")
    fields = (
        "checkpoint_kind", "compressed_targets", "pending_dense_targets", "skip_targets",
        "legacy_original_only_sources", "train_mode", "after_category_mode", "norm_train_mode",
        "lm_head_train_mode", "lora_config", "resolved_learning_rates", "completed_categories",
        "compression_categories", "target_layers", "target_modules", "immutable_resume_contract",
        "finalized_status", "runtime_audit", "base_model_path", "extra_meta",
    )
    destination = args.output_dir / "checkpoint"
    saved = save_v6_full_checkpoint(
        model, str(destination), **{key: deepcopy(meta[key]) for key in fields if key in meta},
        cat_runtime_state=runtime, save_config=True,
    )
    if saved["checkpoint_id"] == meta["checkpoint_id"]:
        raise RuntimeError("Saving local recovery reused the source checkpoint identity.")
    del layer, model
    gc.collect()
    torch.cuda.empty_cache()
    print("Checking all saved tensors and strict independent reload", flush=True)
    result["changed_tensors"] = _verify_payloads(args.checkpoint_dir, destination, args.target)
    reloaded, reloaded_meta, _ = load_v6_model_checkpoint(
        str(destination), base_model_path=args.base_model_path,
        map_location="cpu", strict=True, expected_kind=meta["checkpoint_kind"],
    )
    for field in ("compressed_targets", "pending_dense_targets", "skip_targets"):
        if reloaded_meta[field] != meta[field]:
            raise RuntimeError(f"Reload changed the inventory: {field}")
    recovered = reloaded.get_submodule(args.target).to(args.device)
    with torch.no_grad():
        actual_output = recovered(heldout_inputs).detach().cpu()
    # Same serialized state and native execution path should produce the same outputs.
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    report = {
        "purpose": "local_path_validation_not_full_model_accuracy", "target": args.target,
        "source_checkpoint": str(args.checkpoint_dir), "source_checkpoint_id": meta["checkpoint_id"],
        "saved_checkpoint_id": saved["checkpoint_id"], "saved_checkpoint": str(destination),
        "independent_reload_output_exact": True, "data": data,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "local_metrics": result,
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
