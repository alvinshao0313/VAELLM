"""Stage B entry: existing native v6 checkpoint -> sequential block recovery."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.liftquant_recovery.block_train import train_block
from experiments.liftquant_recovery.recovery_data import calibration, REVISION
from experiments.liftquant_recovery.recovery_runtime import (
    audit_topology, block_output, clear_caches, first_inputs, frozen_digest,
    targets_by_block, teacher_outputs, tree_to, prime_packed_cache,
)
from experiments.liftquant_recovery.recovery_resume import (
    boundary_identity, checkpoint_fingerprint, load_boundary, mutable_names_by_block,
    restore_rng, save_boundary, teacher_fingerprint, training_fingerprint,
)
from experiments.liftquant_recovery.verify_recovery import check_ste
from rotation.model_utils import get_model
from train_utils.checkpoint_v6 import save_v6_full_checkpoint, refresh_vae_linear_runtime_after_state_load
from train_utils.v6_model_loader import load_v6_model_checkpoint


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="completed-block latest_boundary.pt; --output must still be a NEW directory")
    parser.add_argument("--blocks", default="all", help="all compressed blocks, or ascending comma-separated indices")
    parser.add_argument("--nsamples", type=int, default=4096)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--holdout", type=int, default=None, help="default nsamples//32, matching official Stage2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-lr", type=float, default=2e-5)
    parser.add_argument("--decoder-lr", type=float, default=1.25e-5)
    parser.add_argument("--smoke-rows", help="explicit RedPajama API-page JSON; smoke only")
    parser.add_argument("--redpajama-arrow-dir", help="original 11-shard RedPajama Arrow cache directory")
    parser.add_argument("--gpu-memory-gib", type=float, default=12)
    args = parser.parse_args()
    if min(args.nsamples, args.seqlen, args.batch_size, args.epochs, args.code_lr, args.decoder_lr) <= 0:
        parser.error("sample counts, lengths, epochs and learning rates must be positive")
    ntrain = args.nsamples - (args.holdout if args.holdout is not None else args.nsamples // 32)
    if not 0 < ntrain <= args.nsamples or ntrain % args.batch_size:
        parser.error("training sample count must be a positive batch multiple")
    if args.smoke_rows and (args.blocks == "all" or args.nsamples > 32 or args.seqlen > 128):
        parser.error("API-page data is only allowed for explicitly selected small smoke blocks")
    return args


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run(args):
    started = time.time()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for source_file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source_file, snapshot / source_file.name)
    source = Path(args.checkpoint).resolve()
    print("Hashing source model payload for strict restart identity", flush=True)
    fingerprint = checkpoint_fingerprint(source)
    code_fingerprint = training_fingerprint(Path(__file__).resolve().parents[2])
    source_meta = json.loads((source / "checkpoint_meta.json").read_text())
    for spec in source_meta["converted_modules"]:
        if spec["residual_stages"] != 1 or spec["parallel_parts"] != 1:
            raise ValueError("This entry is explicitly scoped to the supplied single-stage checkpoint.")
        if spec.get("weight_rotation") not in (None, "none"):
            raise ValueError("Rotated checkpoint outside current recovery scope.")
    if not torch.cuda.is_available():
        raise RuntimeError("Recovery requires the confirmed CUDA environment.")
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.gpu_memory_gib * 2**30 / total, 1.0), device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = dict(config=vars(args), source_checkpoint=str(source),
                    source_checkpoint_id=source_meta["checkpoint_id"],
                    source_fingerprint=fingerprint, training_fingerprint=code_fingerprint,
                    liftquant_commit=REVISION,
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    code_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in Path(__file__).parent.glob("*.py")},
                    torch_version=torch.__version__, device=torch.cuda.get_device_name(device),
                    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    torch_cuda=torch.version.cuda, upstream_alignment=1,
                    loss="whole block final hidden-state MSE, all unpadded tokens",
                    inference_decode="native packed-u8 BF16 cache prewarm (same arithmetic as training)",
                    proxy="p=s*(b-0.5), s=fixed native decoder single-bit RMS; round-clamp STE with 1/s gate; FP32",
                    bit_lr_mapping="min(code_lr, scaled proxy.std()/50); geometry and reachability recorded per Linear",
                    runtime_checks=check_ste(device))
    torch.manual_seed(args.seed)
    write_json(output / "manifest.json", manifest)
    print("CUDA all-bit STE numerical check PASS", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
    ids, data_meta = calibration(args, tokenizer)
    torch.save(ids, output / "calibration_ids.pt")
    write_json(output / "calibration.json", data_meta)
    print(f"Calibration ready: {tuple(ids.shape)}; {data_meta['scope']}", flush=True)
    student, meta, load_result = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(f"Checkpoint not strictly loaded: {load_result}")
    student.eval().requires_grad_(False)
    targets = targets_by_block(student, meta)
    selected = sorted(targets) if args.blocks == "all" else [int(v) for v in args.blocks.split(",")]
    if selected != sorted(set(selected)) or not selected or any(i not in targets for i in selected):
        raise ValueError("Select ascending unique blocks containing compressed targets.")
    allowed_ids = audit_topology(student, meta, selected, targets)
    allowed_names = {name for name, value in student.state_dict(keep_vars=True).items() if id(value) in allowed_ids}
    if not allowed_names:
        raise ValueError("Empty mutable checkpoint inventory.")
    before_frozen = frozen_digest(student, allowed_names)
    names_by_block = mutable_names_by_block(allowed_names, selected)
    print("Hashing FP teacher config and weight shards for restart identity", flush=True)
    teacher_payload = teacher_fingerprint(meta["base_model_path"])
    identity = boundary_identity(args, selected, fingerprint, code_fingerprint, ids,
                                 teacher_fingerprint=teacher_payload)
    completed, records, reload_inputs, reload_outputs = [], {}, {}, {}
    resume_rng = None
    if args.resume:
        resumed = load_boundary(args.resume, student, identity, names_by_block)
        refresh_vae_linear_runtime_after_state_load(student)
        completed, records = resumed["completed"], resumed["records"]
        reload_inputs, reload_outputs = resumed["reload_inputs"], resumed["reload_outputs"]
        resume_rng = resumed["rng"]
        del resumed
        print(f"Restored completed blocks {completed}; replaying FP teacher prefix", flush=True)
    manifest.update(selected_blocks=selected, compressed_count=len(meta["compressed_targets"]),
                    pending_dense_count=len(meta["pending_dense_targets"]),
                    mutable_state_names=sorted(allowed_names), frozen_sha256=before_frozen,
                    cross_block_decoder_sharing=False, teacher_fingerprint=teacher_payload,
                    resumed_completed_blocks=list(completed),
                    restart_policy="completed blocks only; fresh optimizer for the next block; FP teacher prefix replay")
    write_json(output / "manifest.json", manifest)
    if completed:
        write_json(output / "block_metrics.json", records)
    print(f"Native checkpoint loaded: {len(meta['compressed_targets'])} compressed; blocks {selected}", flush=True)
    teacher = get_model(meta["base_model_path"]).eval().requires_grad_(False)
    hidden, kwargs = first_inputs(teacher, ids, args.batch_size, device)
    boundary_path = output / "latest_boundary.pt"
    last_resumed_block = completed[-1] if completed else None
    for index in range(max(selected) + 1):
        fp_block = teacher.model.layers[index]
        target = teacher_outputs(fp_block, hidden, kwargs, args.batch_size, device)
        if index in completed:
            torch.testing.assert_close(hidden[:args.batch_size], reload_inputs[index], rtol=0, atol=0)
            restored_block = student.model.layers[index].to(device)
            try:
                prime_packed_cache(restored_block)
                with torch.no_grad():
                    restored_output = block_output(restored_block, hidden[:args.batch_size].to(device),
                                                   tree_to(kwargs, device)).cpu()
                torch.testing.assert_close(restored_output, reload_outputs[index], rtol=0, atol=0)
            finally:
                clear_caches(restored_block)
                restored_block.cpu()
                torch.cuda.empty_cache()
            print(f"Skipping completed block {index}; teacher input and native output match exactly", flush=True)
        elif index in selected:
            print(f"Recovering block {index}: {len(targets[index])} compressed linears together", flush=True)
            record, reference = train_block(
                student.model.layers[index], targets[index], hidden, target, kwargs, args, device,
            )
            records[str(index)] = record
            reload_inputs[index] = hidden[:args.batch_size].clone()
            reload_outputs[index] = reference
            completed.append(index)
            write_json(output / "block_metrics.json", records)
            boundary = save_boundary(boundary_path, student, identity, names_by_block, completed,
                                     records, reload_inputs, reload_outputs, device)
            print(f"Block {index} complete: MSE {record['before_mse']:.7g} -> {record['after_mse']:.7g}; "
                  f"atomic boundary {boundary['bytes']} bytes", flush=True)
        # align=1: advance exclusively through the untouched FP teacher.
        hidden = target
        if index == last_resumed_block:
            # Model loading and skipped teacher prefix may consume random values.
            # Restore exactly where the original completed-block snapshot was made.
            restore_rng(resume_rng, device)
            resume_rng = None
    del teacher, hidden, target, fp_block
    gc.collect()
    after_frozen = frozen_digest(student, allowed_names)
    if after_frozen != before_frozen:
        raise ValueError("Frozen weights or nonselected block payloads changed.")
    write_json(output / "freeze_check.json", dict(status="PASS", before=before_frozen, after=after_frozen))
    destination = output / "recovered_model"
    print(f"Saving native v6 checkpoint to {destination}", flush=True)
    save_v6_full_checkpoint(
        student, str(destination), checkpoint_kind="final_model",
        compressed_targets=meta["compressed_targets"], pending_dense_targets=meta["pending_dense_targets"],
        skip_targets=meta["skip_targets"], legacy_original_only_sources=meta.get("legacy_original_only_sources", []),
        train_mode="none", after_category_mode=None, norm_train_mode="none", lm_head_train_mode="none",
        completed_categories=meta.get("completed_categories", []),
        compression_categories=meta.get("compression_categories", []),
        target_layers=meta.get("target_layers"), target_modules=meta.get("target_modules"),
        base_model_path=meta["base_model_path"], tokenizer=tokenizer,
        extra_meta={"liftquant_recovery": {"source_checkpoint_id": meta["checkpoint_id"], "blocks": selected,
                                         "smoke": bool(args.smoke_rows), "manifest": "../manifest.json"}},
    )
    del targets, student
    gc.collect()
    restored, restored_meta, loaded = load_v6_model_checkpoint(str(destination), map_location="cpu", strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Strict B checkpoint reload failed.")
    restored.eval().requires_grad_(False)
    if restored_meta["compressed_targets"] != meta["compressed_targets"]:
        raise ValueError("Reload changed compression scope.")
    reload_report = {}
    gpu_kwargs = tree_to(kwargs, device)
    with torch.no_grad():
        for index in selected:
            block = restored.model.layers[index].to(device)
            prime_packed_cache(block)
            result = block_output(block, reload_inputs[index].to(device), gpu_kwargs).cpu()
            torch.testing.assert_close(result, reload_outputs[index], rtol=0, atol=0)
            reload_report[index] = {"max_abs": (result.float() - reload_outputs[index].float()).abs().max().item()}
            clear_caches(block)
            block.cpu()
            torch.cuda.empty_cache()
    if checkpoint_fingerprint(source) != fingerprint:
        raise ValueError("Source checkpoint payload or configuration changed during this run.")
    summary = dict(status="PASS", stage="B recovery smoke" if args.smoke_rows else "B recovery",
                   selected_blocks=selected, total_optimizer_steps=sum(r["optimizer_steps"] for r in records.values()),
                   frozen_state_unchanged=True, strict_native_reload=True,
                   reload=reload_report, checkpoint=str(destination), seconds=time.time() - started,
                   resumed_from=args.resume, resumed_completed_blocks=manifest["resumed_completed_blocks"],
                   latest_boundary=str(boundary_path) if boundary_path.exists() else args.resume,
                   downstream_evaluation="not run by recovery entry", final_experiment_run=not bool(args.smoke_rows))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(arguments())
