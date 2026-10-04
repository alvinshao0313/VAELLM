"""Independent fixed-code decoder recovery workers for complete Qwen3-8B v6 models."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_decoder import optimize_local_decoder
from litebsq.vae_linear import VAELinear
from train_utils.base_reference import load_frozen_base_reference_model
from train_utils.checkpoint_v6 import resolve_v6_checkpoint_dir
from train_utils.v6_model_loader import load_v6_model_checkpoint


PROTOCOL_VERSION = 1
BASE_MODEL = "/home/star/models/Qwen/Qwen3-8B"
PROJECTION_SHAPES = {
    "self_attn.q_proj": (4096, 4096), "self_attn.k_proj": (1024, 4096),
    "self_attn.v_proj": (1024, 4096), "self_attn.o_proj": (4096, 4096),
    "mlp.gate_proj": (12288, 4096), "mlp.up_proj": (12288, 4096),
    "mlp.down_proj": (4096, 12288),
}
TARGETS = tuple(f"model.layers.{layer}.{suffix}" for layer in range(36) for suffix in PROJECTION_SHAPES)
CODE_FILES = (
    "experiments/qwen3_w25/local_recovery.py",
    "experiments/qwen3_w25/local_decoder.py",
    "experiments/qwen3_w25/calibration.py",
    "experiments/linear_output/objectives.py",
)
DECODER_PREFIX = "_parallel_stage_decoder."


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shard_targets(shard_index: int, num_shards: int) -> tuple[str, ...]:
    if type(num_shards) is not int or not 1 <= num_shards <= 36:
        raise ValueError("num_shards must be an integer in [1, 36].")
    if type(shard_index) is not int or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be an integer in [0, num_shards).")
    return tuple(name for name in TARGETS if int(name.split(".")[2]) % num_shards == shard_index)


def recovery_config(*, steps: int, lr: float, num_shards: int) -> dict:
    shard_targets(0, num_shards)
    if type(steps) is not int or steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps must be a positive integer and lr must be finite and positive.")
    return {
        "steps": steps, "lr": lr, "seed": 31, "train_documents": 32, "heldout_documents": 32,
        "max_length": 256, "chunk_size": 256, "num_shards": num_shards,
        "objective": "full_linear_output_mse", "optimizer": "AdamW", "optimizer_dtype": "float32",
        "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8,
    }


def validate_recovery_source(model, meta):
    if (getattr(model.config, "model_type", None) != "qwen3"
            or Path(meta.get("base_model_path") or "").resolve() != Path(BASE_MODEL).resolve()):
        raise ValueError("Source must reference the fixed original Qwen3-8B teacher.")
    if meta["checkpoint_kind"] != "final_model" or set(meta["compressed_targets"]) != set(TARGETS):
        raise ValueError("Recovery requires a complete Qwen3-8B final_model with all 252 projections.")
    if meta["pending_dense_targets"] or meta["skip_targets"] or meta.get("lora_config") is not None:
        raise ValueError("Pending/skipped projections and LoRA are outside this recovery protocol.")
    if meta.get("shared_protected_residual_decoders") or (meta.get("extra_meta") or {}).get("residual_lora"):
        raise ValueError("Protected shared decoders and residual LoRA are outside this protocol.")
    if any("lora" in key.lower() or "low_rank_a" in key or "low_rank_b" in key for key in model.state_dict()):
        raise ValueError("Source contains a LoRA tensor.")
    for name in TARGETS:
        layer = model.get_submodule(name)
        suffix = ".".join(name.split(".")[3:])
        if not isinstance(layer, VAELinear) or (layer.out_features, layer.in_features) != PROJECTION_SHAPES[suffix]:
            raise ValueError(f"Expected original Qwen3-8B projection shape: {name}.")
        decoder = layer._modules.get("_parallel_stage_decoder")
        if (decoder is None or not layer.parallel_stage_decode or layer.residual_stages != 2
                or layer.parallel_parts != 1 or decoder.num_models != 2):
            raise ValueError(f"Expected an existing packed two-stage decoder: {name}.")
        if layer.weight_rotation is None or layer.weight_rotation.block_size != 0:
            raise ValueError(f"Expected full two-sided rotation: {name}.")
        if layer.original_weight is not None or layer.always_use_original or not layer.temporary:
            raise ValueError(f"Original-weight runtime is forbidden: {name}.")
        forbidden = ("low_rank_a", "low_rank_b", "protected_input_indices", "protected_output_indices",
                     "protected_residual_indices", "_sparse_bit_binding")
        if any(getattr(layer, key, None) is not None for key in forbidden) or layer.has_sparse_residual():
            raise ValueError(f"Only fixed-code unprotected Linear recovery is supported: {name}.")
        if layer.trainable_decode or any(isinstance(child, torch.nn.modules.batchnorm._BatchNorm) for child in layer.modules()):
            raise ValueError(f"Active training state or BatchNorm is forbidden: {name}.")


class _ShardInputsCaptured(Exception):
    """Stop a teacher document after every requested Linear has received its input."""


@torch.no_grad()
def capture_shard_inputs(teacher, targets, train_batch, heldout_batch) -> dict:
    """Capture every target in one forward per document, preserving valid-token order.

    Each target's CPU tensor is preallocated, avoiding a second full-size copy
    when concatenating document chunks. References are original teacher weights.
    """
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Teacher targets must be nonempty and unique.")
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("Teacher must be frozen and in eval mode.")
    device = next(teacher.parameters()).device
    batches = {"train": train_batch, "heldout": heldout_batch}
    sizes = {}
    for split, batch in batches.items():
        if batch["input_ids"].shape != batch["attention_mask"].shape or batch["input_ids"].ndim != 2:
            raise ValueError(f"Invalid teacher batch shape: {split}.")
        if not torch.all(batch["attention_mask"].sum(dim=1) > 0):
            raise ValueError(f"Every {split} document must contain valid tokens.")
        sizes[split] = int(batch["attention_mask"].sum())
    layers = {name: teacher.get_submodule(name) for name in targets}
    bundles = {name: {"reference_weight": layer.weight.detach().cpu().clone()} for name, layer in layers.items()}
    handles, seen = [], set()
    current_mask = None
    current_split = ""
    offset = 0

    def make_hook(name):
        def hook(_module, inputs):
            if name in seen:
                raise RuntimeError(f"Teacher executed a requested Linear twice in one document: {name}.")
            x = inputs[0]
            if x.ndim != 3 or x.shape[:2] != current_mask.shape or x.shape[-1] != layers[name].in_features:
                raise ValueError(f"Unexpected teacher input shape for {name}: {tuple(x.shape)}.")
            values = x.detach()[current_mask].contiguous().cpu()
            if not torch.isfinite(values).all():
                raise FloatingPointError(f"Non-finite teacher inputs: {name}.")
            key = current_split + "_inputs"
            if key not in bundles[name]:
                bundles[name][key] = torch.empty((sizes[current_split], values.shape[1]), dtype=values.dtype)
            cache = bundles[name][key]
            if cache.dtype != values.dtype:
                raise ValueError(f"Teacher activation dtype changed across documents: {name}.")
            cache[offset:offset + len(values)].copy_(values)
            seen.add(name)
            if len(seen) == len(targets):
                raise _ShardInputsCaptured()
        return hook

    try:
        for name, layer in layers.items():
            handles.append(layer.register_forward_pre_hook(make_hook(name)))
        for split, batch in batches.items():
            current_split, offset = split, 0
            for index in range(len(batch["input_ids"])):
                seen = set()
                ids = batch["input_ids"][index:index + 1].to(device)
                mask = batch["attention_mask"][index:index + 1].to(device)
                current_mask = mask.bool()
                try:
                    teacher(input_ids=ids, attention_mask=mask, use_cache=False)
                except _ShardInputsCaptured:
                    if seen != set(targets):
                        raise RuntimeError("Teacher capture stopped before all targets were visited.")
                else:
                    raise RuntimeError(f"Teacher did not execute all targets; missing={sorted(set(targets) - seen)}")
                offset += int(current_mask.sum())
            if offset != sizes[split]:
                raise RuntimeError(f"Captured token total differs for {split}.")
    finally:
        for handle in handles:
            handle.remove()
    return bundles


def run_local_recovery_worker(source_dir, output_dir, *, shard_index: int, num_shards: int,
                              steps: int, lr: float) -> dict:
    targets = shard_targets(shard_index, num_shards)
    config = recovery_config(steps=steps, lr=lr, num_shards=num_shards)
    source = Path(resolve_v6_checkpoint_dir(str(source_dir))).resolve()
    output = Path(output_dir).resolve()
    if output.is_relative_to(source):
        raise ValueError("Worker output must be outside the source checkpoint.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "patches").mkdir()
    (output / "reports").mkdir()
    repo = Path(__file__).resolve().parents[2]
    device = torch.device("cuda:0")
    started = time.perf_counter()
    report = {
        "protocol_version": PROTOCOL_VERSION, "status": "preparing",
        "source_checkpoint_dir": str(source), "source_checkpoint_id": None,
        "config": config, "shard_index": shard_index, "targets": list(targets),
        "code_sha256": {relative: _sha256(repo / relative) for relative in CODE_FILES},
        "modules": {},
    }
    report_path = output / "worker.json"
    _write_json(report_path, report)
    try:
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_batch, heldout_batch, data = build_calibration_batches(
            tokenizer, seed=config["seed"], train_documents=config["train_documents"],
            heldout_documents=config["heldout_documents"], max_length=config["max_length"],
        )
        report["data"] = data
        model, meta, _ = load_v6_model_checkpoint(str(source), base_model_path=BASE_MODEL,
                                                 map_location="cpu", strict=True, expected_kind="final_model")
        model.eval().requires_grad_(False)
        validate_recovery_source(model, meta)
        report["source_checkpoint_id"] = meta["checkpoint_id"]
        print(f"Capturing {len(targets)} projections in one teacher forward per document", flush=True)
        phase_start = time.perf_counter()
        teacher = load_frozen_base_reference_model(BASE_MODEL, access_token=None, device=str(device))
        bundles = capture_shard_inputs(teacher, targets, train_batch, heldout_batch)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        report["teacher_capture_seconds"] = time.perf_counter() - phase_start
        report["cached_cpu_tensor_bytes"] = sum(t.numel() * t.element_size() for item in bundles.values() for t in item.values())
        report["status"] = "running"
        _write_json(report_path, report)

        for name in targets:
            bundle = bundles.pop(name)
            train_x, heldout_x = bundle["train_inputs"], bundle["heldout_inputs"]
            if len(train_x) != data["train"]["valid_tokens"] or len(heldout_x) != data["heldout"]["valid_tokens"]:
                raise RuntimeError(f"Captured token count differs from the fixed manifest: {name}.")
            if train_x.dtype != torch.bfloat16 or heldout_x.dtype != torch.bfloat16:
                raise ValueError("Worker requires the existing BF16 teacher activation path.")
            layer = model.get_submodule(name).to(device)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            module_start = time.perf_counter()

            def progress(row):
                if row["step"] % 10 == 0 or row["step"] == steps:
                    print(json.dumps({"target": name, **row}), flush=True)

            metrics = optimize_local_decoder(
                layer, bundle["reference_weight"], train_x, heldout_x, steps=steps, lr=lr,
                chunk_size=config["chunk_size"], progress_callback=progress,
            )
            torch.cuda.synchronize(device)
            metrics.update(target=name, protocol_version=PROTOCOL_VERSION,
                           source_checkpoint_id=meta["checkpoint_id"], source_checkpoint_dir=str(source),
                           elapsed_seconds=time.perf_counter() - module_start,
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
            patch = {key: tensor.detach().cpu().clone() for key, tensor in layer.state_dict().items()
                     if key.startswith(DECODER_PREFIX)}
            if not patch:
                raise RuntimeError(f"No registered decoder tensors available for patch: {name}.")
            patch_file, module_file = Path("patches") / (name + ".pt"), Path("reports") / (name + ".json")
            torch.save(patch, output / patch_file)
            _write_json(output / module_file, metrics)
            report["modules"][name] = {"patch_file": str(patch_file), "report_file": str(module_file),
                                       "patch_sha256": _sha256(output / patch_file)}
            _write_json(report_path, report)
            layer.clear_decoded_weight_cache()
            layer.cpu()
            layer._build_parallel_stage_decode_plan()
            del patch, bundle, train_x, heldout_x
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        _write_json(report_path, report)
        return report
    except Exception as exc:
        report.update(status="failed", elapsed_seconds=time.perf_counter() - started,
                      error={"type": type(exc).__name__, "message": str(exc)})
        _write_json(report_path, report)
        raise


__all__ = ["PROTOCOL_VERSION", "TARGETS", "shard_targets", "recovery_config",
           "validate_recovery_source", "capture_shard_inputs", "run_local_recovery_worker"]
