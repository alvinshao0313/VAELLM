"""Compare document microbatches 1/8 on the real down3 L18 block; no weights saved."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.block_batching import batch_block_documents
from experiments.qwen3_w25.block_calibration import capture_block_batches
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_block import PROJECTIONS, _frozen_digest, _tree_to, optimize_local_block
from experiments.qwen3_w25.local_decoder import _storage_bits
from experiments.qwen3_w25.local_recovery import BASE_MODEL, validate_recovery_source
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.v6_model_loader import load_v6_model_checkpoint


SOURCE_ID = "c24d4b96-01f6-43d7-8df4-8a9b15ca3469"
CODE_FILES = (
    "experiments/qwen3_w25/validate_block_batching.py", "experiments/qwen3_w25/block_batching.py",
    "experiments/qwen3_w25/local_block.py", "experiments/qwen3_w25/block_weight_graph.py",
    "experiments/qwen3_w25/block_calibration.py", "experiments/qwen3_w25/calibration.py",
    "litebsq/vae_linear.py", "litebsq/parallel_layers.py", "litebsq/packed_bit_linear.py",
)


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _comparison(actual, expected):
    # Accumulate norms in FP64 on CPU; no GPU tensor or model state survives here.
    a, b = actual.detach().cpu().double().reshape(-1), expected.detach().cpu().double().reshape(-1)
    na, nb, delta = float(a.norm()), float(b.norm()), a - b
    return {"numel": len(a), "reference_l2": nb, "candidate_l2": na,
            "difference_l2": float(delta.norm()), "relative_l2": float(delta.norm()) / nb if nb else None,
            "cosine": float(torch.dot(a, b)) / (na * nb) if na and nb else None,
            "max_abs_difference": float(delta.abs().max()), "different_values": int(torch.count_nonzero(delta))}


def _shapes(batches):
    return [{"hidden": list(batch.hidden_states.shape), "valid_tokens": int(batch.valid_mask.sum()),
             "attention_mask": None if batch.forward_kwargs["attention_mask"] is None
             else list(batch.forward_kwargs["attention_mask"].shape)} for batch in batches]


def _restore(block, initial, digest):
    block.load_state_dict(initial, strict=True)
    for name in PROJECTIONS:
        module = block.get_submodule(name)
        module._build_parallel_stage_decode_plan()
        module.clear_decoded_weight_cache()
    if _frozen_digest(block, set()) != digest or any(p.grad is not None for p in block.parameters()):
        raise RuntimeError("Failed to restore the entire original block before the arm.")


@torch.no_grad()
def _native_outputs(block, documents, device):
    outputs, squared_error, count = [], 0.0, 0
    for batch in documents:
        output = block(batch.hidden_states.to(device), **_tree_to(batch.forward_kwargs, device))[0]
        mask = batch.valid_mask.to(device=device, dtype=torch.bool)
        actual = output[mask]
        target = batch.target_hidden_states.to(device)[mask]
        squared_error += float((actual.float() - target.float()).square().sum())
        count += actual.numel()
        outputs.append(actual.detach().cpu())
    return torch.cat(outputs), squared_error / count


def _run_arm(block, train, heldout, *, microbatch_size, device):
    parameters = {name: value for name, value in block.named_parameters() if "._parallel_stage_decoder." in name}
    first_gradients, gradient_handles, forward_handles = {}, [], []
    pending_step, step_start = 1, None
    registered, progress = set(), []

    def decoder_pre_hook(prefix, decoder):
        def hook(_module, _args):
            nonlocal step_start
            if not torch.is_grad_enabled():
                return
            if prefix == PROJECTIONS[0] and step_start is None:
                torch.cuda.synchronize(device)
                step_start = time.perf_counter()
            if pending_step != 1 or prefix in registered:
                return
            registered.add(prefix)
            for local_name, parameter in decoder.named_parameters():
                name = prefix + "._parallel_stage_decoder." + local_name
                if not parameter.requires_grad:
                    raise RuntimeError("The production core did not enable a decoder gradient.")

                def capture(gradient, name=name):
                    if name in first_gradients:
                        raise RuntimeError("Decoder gradient arrived more than once in the first VJP.")
                    first_gradients[name] = gradient.detach().cpu().clone()

                gradient_handles.append(parameter.register_hook(capture))
        return hook

    def report_progress(row):
        nonlocal pending_step, step_start
        torch.cuda.synchronize(device)
        if step_start is None:
            raise RuntimeError("Did not observe the actual trainable decoder forward.")
        row = {**row, "step_seconds": time.perf_counter() - step_start}
        progress.append(row)
        print(json.dumps({"microbatch_size": microbatch_size, **row}), flush=True)
        if pending_step == 1:
            if set(first_gradients) != set(parameters):
                raise RuntimeError("First-step gradient capture missed decoder parameters.")
            for handle in gradient_handles:
                handle.remove()
            gradient_handles.clear()
        pending_step += 1
        step_start = time.perf_counter()

    for prefix in PROJECTIONS:
        decoder = block.get_submodule(prefix)._parallel_stage_decoder
        # Packed CUDA decoding bypasses Decoder.__call__, but executes norm_out.
        # Register on its real path before backward, without replacing forwards.
        forward_handles.append(decoder.norm_out.register_forward_pre_hook(decoder_pre_hook(prefix, decoder)))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        result = optimize_local_block(block, train, heldout, steps=3, lr=1.25e-5, progress_callback=report_progress)
    finally:
        for handle in [*forward_handles, *gradient_handles]:
            handle.remove()
    torch.cuda.synchronize(device)
    result.update(elapsed_seconds=time.perf_counter() - start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device), step_telemetry=progress,
                  timing_basis="step1 from q decoder norm_out through Adam update (includes gradient CPU capture); steps2/3 from previous callback end",
                  steady_step_seconds=sum(row["step_seconds"] for row in progress[1:]) / 2)
    return result, first_gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.checkpoint_dir.resolve(), args.output_dir.resolve()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Launch with CUBLAS_WORKSPACE_CONFIG=:4096:8 for the fixed deterministic protocol.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise ValueError("This validation is assigned exclusively to physical GPU 1.")
    meta, audit = _validate_checkpoint(source)
    if meta["checkpoint_id"] != SOURCE_ID or not audit["budget_passed"]:
        raise ValueError("Expected the fixed complete down3 source within the rank-zero bit budget.")
    output.mkdir(parents=True, exist_ok=False)
    device, path = torch.device("cuda:0"), output / "validation.json"
    repo = Path(__file__).resolve().parents[2]
    report = {"status": "preparing", "purpose": "native_fa2_block_execution_batch_equivalence_not_downstream_accuracy",
              "pid": os.getpid(), "physical_gpu": 1, "source_checkpoint_dir": str(source), "source_checkpoint_id": SOURCE_ID,
              "configuration": {"block_index": 18, "microbatch_sizes": [1, 8], "steps": 3, "lr": 1.25e-5,
                                "seed": 31, "train_documents": 64, "heldout_documents": 64, "max_length": 256,
                                "teacher_capture_passes": 1, "checkpoint_written": False,
                                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                "allow_tf32": False, "deterministic_algorithms": True},
              "code_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in CODE_FILES},
              "source_bit_budget": audit, "arms": {}}
    _write(path, report)
    try:
        torch.manual_seed(31)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        raw_train, raw_heldout, data = build_calibration_batches(
            tokenizer, seed=31, train_documents=64, heldout_documents=64, max_length=256)
        report["data"] = data
        _write(path, report)
        print(json.dumps({"status": "capturing", "pid": os.getpid(), "physical_gpu": 1}), flush=True)
        start = time.perf_counter()
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=device)
        if teacher.config._attn_implementation != "flash_attention_2":
            raise RuntimeError("The real teacher must use native FlashAttention2.")
        train = capture_block_batches(teacher, 18, raw_train)
        heldout = capture_block_batches(teacher, 18, raw_heldout)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        report["teacher_capture_seconds"] = time.perf_counter() - start
        model, loaded_meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL, map_location="cpu", strict=True)
        if loaded_meta["checkpoint_id"] != SOURCE_ID:
            raise RuntimeError("Source identity changed during load.")
        model.eval().requires_grad_(False)
        validate_recovery_source(model, loaded_meta)
        block = model.model.layers[18].to(device)
        if block.self_attn.config._attn_implementation != "flash_attention_2":
            raise RuntimeError("The real student block must use native FlashAttention2.")
        initial = {name: value.detach().cpu().clone() for name, value in block.state_dict().items()}
        source_digest, storage = _frozen_digest(block, set()), _storage_bits(block)
        report.update(source_block_sha256=source_digest, block_storage_bits=storage, status="running")
        _write(path, report)
        gradients, saved, outputs = {}, {}, {}
        for size in (1, 8):
            _restore(block, initial, source_digest)
            torch.manual_seed(31)
            batched_train = batch_block_documents(train, microbatch_size=size)
            batched_heldout = batch_block_documents(heldout, microbatch_size=size)
            print(json.dumps({"status": "optimizing", "microbatch_size": size,
                              "train_shapes": _shapes(batched_train), "heldout_shapes": _shapes(batched_heldout)}), flush=True)
            metrics, gradients[size] = _run_arm(block, batched_train, batched_heldout, microbatch_size=size, device=device)
            outputs[size], common_mse = _native_outputs(block, heldout, device)
            saved[size] = {name: value.detach().cpu().clone() for name, value in block.named_parameters()
                           if "._parallel_stage_decoder." in name}
            if _storage_bits(block) != storage:
                raise RuntimeError("The execution batch changed block persistent storage.")
            report["arms"][str(size)] = {"local_metrics": metrics, "common_single_document_heldout_mse": common_mse,
                                        "source_restored_sha256": source_digest,
                                        "train_shapes": _shapes(batched_train), "heldout_shapes": _shapes(batched_heldout)}
            _write(path, report)
        comparisons = {}
        for prefix in PROJECTIONS:
            names = [name for name in gradients[1] if name.startswith(prefix + ".")]
            g1, g8 = (torch.cat([gradients[size][name].reshape(-1) for name in names]) for size in (1, 8))
            u1, u8 = (torch.cat([(saved[size][name].float() - initial[name].float()).reshape(-1) for name in names]) for size in (1, 8))
            comparisons[prefix] = {"first_step_gradient": _comparison(g8, g1), "saved_decoder_update": _comparison(u8, u1)}
        mse1, mse8 = (report["arms"][str(size)]["common_single_document_heldout_mse"] for size in (1, 8))
        frozen = [report["arms"][str(size)]["local_metrics"]["evidence"]["frozen_sha256_after"] for size in (1, 8)]
        if frozen[0] != frozen[1]:
            raise RuntimeError("The arms changed different non-decoder payloads.")
        report.update(status="complete", comparisons=comparisons, output_comparison=_comparison(outputs[8], outputs[1]),
                      common_native_heldout_mse_relative_difference=abs(mse8 - mse1) / mse1,
                      steady_step_speedup=report["arms"]["1"]["local_metrics"]["steady_step_seconds"] / report["arms"]["8"]["local_metrics"]["steady_step_seconds"],
                      frozen_payload_identical=True, persistent_storage_identical=True)
        _restore(block, initial, source_digest)
        report["final_source_restored"] = True
        _write(path, report)
        print(json.dumps({"status": "complete", "report": str(path),
                          "heldout_mse_relative_difference": report["common_native_heldout_mse_relative_difference"],
                          "steady_step_speedup": report["steady_step_speedup"]}), flush=True)
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        _write(path, report)
        raise


if __name__ == "__main__":
    main()
