"""Capture fixed original-model L0 inputs for the isolated rotation comparison."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_recovery import capture_shard_inputs
from litebsq.protected_channel_quant import encode_protected_channel_weight


FORMAT = "rotation_factor_probe_inputs_v1"
ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj")
COUNTS = dict(zip(TARGETS, (64, 256)))
PROTOCOL = {"seed": 137, "train_documents": 32, "heldout_documents": 32, "max_length": 256,
            "input_use": "Paired measurement only; neither split enters weight-VAE gradients or selects iterations.",
            "selection": "Existing deterministic Alpaca helper; no answer, prediction or loss selection."}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def _source_identity(model_type, tokenizer):
    modules = ["rotation.model_utils", "experiments.qwen3_w25.calibration",
               "experiments.qwen3_w25.local_recovery", "train_utils.distill_data", "e2e_common.data",
               "litebsq.protected_channel_quant", "experiments.longbench_v1.common",
               f"transformers.models.{model_type}.modeling_{model_type}",
               "transformers.modeling_rope_utils", "transformers.modeling_utils",
               "transformers.tokenization_utils_base", "transformers.tokenization_utils_fast",
               "transformers.data.data_collator", "datasets.arrow_dataset",
               "transformers.integrations.sdpa_attention" if model_type == "llama"
               else "transformers.modeling_flash_attention_utils"]
    paths = {Path(__file__).resolve(), Path(inspect.getfile(type(tokenizer))).resolve()}
    paths.update(Path(importlib.import_module(name).__file__).resolve() for name in modules)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "datasets", "safetensors", "accelerate")}
    if model_type == "qwen3":
        import flash_attn
        versions["flash-attn"] = importlib.metadata.version("flash-attn")
        paths.add(Path(flash_attn.__file__).resolve())
        paths.add(Path(importlib.import_module("flash_attn.flash_attn_interface").__file__).resolve())
    return {str(path): sha256(path) for path in sorted(paths)}, versions


def prepare_inputs(model_path, checkpoint_dir):
    """CPU-only identity, protection-index and document preparation; no model load."""
    model_path, checkpoint = Path(model_path).resolve(), Path(checkpoint_dir).resolve()
    require(model_path.is_dir() and checkpoint.is_dir(), "Model and checkpoint must be existing local directories.")
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    require(config.model_type in ("qwen3", "llama"), f"Unsupported model_type: {config.model_type}.")
    require(config.hidden_act == "silu", "This MLP error comparison requires the original SiLU model.")
    meta_path = checkpoint / "checkpoint_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    require(meta["checkpoint_kind"] == "final_model" and bool(meta.get("checkpoint_id"))
            and Path(meta["base_model_path"]).resolve() == model_path,
            "Protection source must be a final checkpoint of the requested original model.")
    checkpoint_config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    for key in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers"):
        require(getattr(checkpoint_config, key) == getattr(config, key), f"Checkpoint model geometry differs: {key}.")
    state_path = checkpoint / meta["state_dict_file"]
    require(state_path.resolve().parent == checkpoint, "Checkpoint state path must remain inside its directory.")
    state_stat = file_stat(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    specs = {item["name"]: item for item in meta["converted_modules"]}
    indices, protection = {}, {}
    for name in TARGETS:
        spec = specs[name]
        count = COUNTS[name]
        in_features = config.hidden_size if name.endswith("gate_proj") else config.intermediate_size
        out_features = config.intermediate_size if name.endswith("gate_proj") else config.hidden_size
        require((spec["out_features"], spec["in_features"]) == (out_features, in_features)
                and spec["compressed_in_features"] == in_features - count
                and spec["compressed_out_features"] == out_features
                and spec["codebook_dim"] == 32 and count % 32 == 0
                and spec["protected_channel_quant_format"] == "int8"
                and spec.get("protected_output_indices") is None,
                f"Unexpected input-protection geometry: {name}.")
        key = name + ".protected_input_indices"
        value = state[key]
        require(value.dtype == torch.int64 and value.device.type == "cpu" and tuple(value.shape) == (count,)
                and len(value.unique()) == count and int(value.min()) >= 0 and int(value.max()) < in_features,
                f"Invalid persisted protection indices: {name}.")
        require(torch.equal(value, value.sort().values), f"Native protection order must be ascending: {name}.")
        indices[name] = value.clone().contiguous()
        protection[name] = {"state_key": key, "indices": tensor_record(value), "spec": spec,
                            "payload_source": "INT8 re-encoding of original HF columns; compressed checkpoint supplies indices only."}
    del state
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True,
                                             trust_remote_code=False)
    original_pad = tokenizer.pad_token_id
    if original_pad is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train, heldout, data = build_calibration_batches(
        tokenizer, seed=137, train_documents=32, heldout_documents=32, max_length=256)
    lengths, boundaries, tokens = {}, {}, {}
    for split, batch in (("train", train), ("heldout", heldout)):
        lengths[split] = [int(value) for value in batch["attention_mask"].sum(dim=1).tolist()]
        require(len(lengths[split]) == 32 and all(0 < value <= 256 for value in lengths[split]),
                f"Invalid fixed document lengths: {split}.")
        offset, boundaries[split] = 0, []
        for document, length in zip(data[split]["documents"], lengths[split]):
            require(document["valid_tokens"] == length, "Manifest and actual token mask lengths differ.")
            boundaries[split].append({"row_index": document["row_index"], "start": offset, "stop": offset + length})
            offset += length
        tokens[split] = {key: tensor_record(value) for key, value in batch.items()}
    hf_files = {model_path / name for name in (
        "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt") if (model_path / name).is_file()}
    hf_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weights = [file_stat(model_path / name) for name in sorted(set(hf_index["weight_map"].values()))]
    source_hashes, versions = _source_identity(config.model_type, tokenizer)
    record = {"format": FORMAT, "status": "cpu_ready", "protocol": PROTOCOL,
        "model_path": str(model_path), "checkpoint_dir": str(checkpoint), "checkpoint_id": meta["checkpoint_id"],
        "model_type": config.model_type, "original_config": config.to_dict(), "hidden_act": config.hidden_act,
        "hf_commit_hash": getattr(config, "_commit_hash", None),
        "hf_weight_files": weights, "checkpoint_weight_file": state_stat,
        "input_files_sha256": {str(path): sha256(path) for path in sorted(hf_files | {meta_path, checkpoint / "config.json", Path(data["source_arrow"])})},
        "source_sha256": source_hashes, "library_versions": versions, "python": sys.version,
        "tokenizer": {"class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
            "is_fast": tokenizer.is_fast, "original_pad_token_id": original_pad,
            "pad_token_id": tokenizer.pad_token_id, "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id, "padding_side": tokenizer.padding_side,
            "add_bos_token": getattr(tokenizer, "add_bos_token", None), "model_max_length": tokenizer.model_max_length,
            "chat_template_applied": False},
        "data": data, "token_batches": tokens, "document_lengths": lengths,
        "document_boundaries": boundaries, "protection": protection}
    return config, train, heldout, indices, record


@torch.no_grad()
def run_capture(model_path, checkpoint_dir, output_dir):
    output = Path(output_dir).resolve()
    require(not output.exists(), "Capture output must be a new directory.")
    require(not any(output.is_relative_to(Path(path).resolve()) for path in (model_path, checkpoint_dir)),
            "Output must be outside immutable model and checkpoint directories.")
    config, train, heldout, indices, record = prepare_inputs(model_path, checkpoint_dir)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Capture requires exactly one externally assigned visible CUDA GPU.")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "data_manifest.json"
    record.update(status="capturing", pid=os.getpid(), started_at_unix=time.time())
    write_json(manifest_path, record)
    try:
        from rotation.model_utils import get_llama, get_qwen3
        loader = get_qwen3 if config.model_type == "qwen3" else get_llama
        torch.manual_seed(137)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        teacher = loader(str(Path(model_path).resolve()), None)
        teacher.eval().requires_grad_(False).to("cuda:0")
        teacher.config.use_cache = False
        expected_backend = "flash_attention_2" if config.model_type == "qwen3" else "sdpa"
        require(teacher.config.model_type == config.model_type
                and teacher.config._attn_implementation == expected_backend
                and teacher.config.rope_theta == config.rope_theta
                and teacher.config.rope_scaling == config.rope_scaling,
                "Original model backend or RoPE configuration changed.")
        require({parameter.dtype for parameter in teacher.parameters()} == {torch.bfloat16},
                "Original model parameters must remain BF16 without a whole-model dtype cast.")
        rope = {name: tensor_record(value) for name, value in teacher.named_buffers() if "inv_freq" in name}
        require(rope and all(item["dtype"] == "torch.float32" for item in rope.values()),
                "Original FP32 RoPE buffers must remain unrounded.")
        mlp = teacher.get_submodule("model.layers.0.mlp")
        require(isinstance(mlp.act_fn, torch.nn.SiLU), "Actual original MLP activation is not SiLU.")
        require(all(teacher.get_submodule(name).bias is None for name in (*TARGETS, "model.layers.0.mlp.up_proj")),
                "This fixed MLP reconstruction expects the original bias-free projections.")
        up_weight = mlp.up_proj.weight.detach().cpu().clone()
        bundles = capture_shard_inputs(teacher, TARGETS, train, heldout)
        for name, bundle in bundles.items():
            weight = bundle["reference_weight"]
            require(weight.dtype == torch.bfloat16, f"Reference dtype changed: {name}.")
            for split in ("train", "heldout"):
                value = bundle[split + "_inputs"]
                require(value.dtype == torch.bfloat16 and value.shape == (sum(record["document_lengths"][split]), weight.shape[1]),
                        f"Captured dtype/geometry differs: {name}/{split}.")
            qvalues, scales = encode_protected_channel_weight(
                weight.index_select(1, indices[name]).T.contiguous(), quant_format="int8")
            bundle.update(protected_input_indices=indices[name], protected_input_qvalues=qvalues,
                          protected_input_scales=scales)
        archive = {"format": FORMAT, "model_type": config.model_type, "model_path": record["model_path"],
                   "checkpoint_id": record["checkpoint_id"], "targets": bundles,
                   "up_proj_weight": up_weight, "document_lengths": record["document_lengths"]}
        record["runtime"] = {"loader": "rotation.model_utils." + loader.__name__, "backend": expected_backend,
            "whole_model_dtype_cast": False, "parameter_dtype": "torch.bfloat16", "rope_buffers": rope,
            "use_cache": False, "frozen": True, "eval": True,
            "actual_mlp_act_fn": type(mlp.act_fn).__module__ + "." + type(mlp.act_fn).__name__,
            "cuda_device": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf32": False, "deterministic_algorithms": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        record["tensors"] = {name: {key: tensor_record(value) for key, value in bundle.items()}
                             for name, bundle in bundles.items()}
        record["up_proj_weight"] = tensor_record(up_weight)
        check_hashes(record["source_sha256"])
        check_hashes(record["input_files_sha256"])
        for stat in record["hf_weight_files"] + [record["checkpoint_weight_file"]]:
            check_stat(stat)
        torch.save(archive, output / "inputs.pt")
        record["archive"] = {**file_stat(output / "inputs.pt"), "sha256": sha256(output / "inputs.pt")}
        record.update(status="captured", inputs_sha256=record["archive"]["sha256"], finished_at_unix=time.time())
        write_json(manifest_path, record)
        return record
    except BaseException as exc:
        record.update(status="failed", error=type(exc).__name__ + ": " + str(exc), finished_at_unix=time.time())
        write_json(manifest_path, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_capture(args.model_path, args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
