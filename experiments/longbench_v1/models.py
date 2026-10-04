from __future__ import annotations

import importlib.metadata
import json
import logging
from pathlib import Path

import torch

from .common import check_hashes, check_stat, file_stat, sha256, source_pins
from .data import load_archive, tokenizer_pins
from .protocol import protocol_record
from .protocol import EOS_TOKEN_IDS, PAD_TOKEN_ID


def make_config(manifest_path, *, kind, hf_path, checkpoint=None, mode="full"):
    if kind not in ("dense", "compressed") or mode not in ("full", "validation"):
        raise ValueError("Unknown model kind or evaluation mode.")
    manifest_path, hf_path = Path(manifest_path).resolve(), Path(hf_path).resolve()
    manifest, _ = load_archive(manifest_path)
    if str(hf_path) != manifest["tokenizer"]["path"]:
        raise ValueError("Both models must use the shared archive's original HF tokenizer.")
    weights = json.loads((hf_path / "model.safetensors.index.json").read_text())["weight_map"]
    generation = json.loads((hf_path / "generation_config.json").read_text())
    if generation["eos_token_id"] != list(EOS_TOKEN_IDS) or generation["pad_token_id"] != PAD_TOKEN_ID:
        raise ValueError("Pinned original HF generation EOS/pad identities changed.")
    pins = tokenizer_pins(hf_path)
    pins[str(hf_path / "model.safetensors.index.json")] = sha256(hf_path / "model.safetensors.index.json")
    model = {"kind": kind, "hf_path": str(hf_path), "hf_metadata_sha256": pins,
             "hf_weight_stats": [file_stat(hf_path / name) for name in sorted(set(weights.values()))]}
    if kind == "compressed":
        path = Path(checkpoint).resolve()
        meta_path = path / "checkpoint_meta.json"
        meta = json.loads(meta_path.read_text())
        if (meta["format"] != "vaellm_model_checkpoint_v6" or meta["checkpoint_kind"] != "final_model"
                or meta["converted_module_count"] != 252 or Path(meta["base_model_path"]).resolve() != hf_path):
            raise ValueError("A complete 252-projection v6 Qwen3 final model is required.")
        model.update(checkpoint=str(path), checkpoint_id=meta["checkpoint_id"],
                     metadata_sha256=sha256(meta_path), weight_stat=file_stat(path / meta["state_dict_file"]))
    elif checkpoint is not None:
        raise ValueError("Dense evaluation must not receive a compressed checkpoint.")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "datasets", "flash-attn", "rouge", "jieba", "fuzzywuzzy")}
    return {"schema": "longbench_v1_run_v1", "mode": mode, "protocol": protocol_record(),
            "archive": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "model": model, "source_sha256": source_pins(), "package_versions": versions,
            "prewarm_group_size": 8, "device": "cuda:0"}


def check_config(config):
    if config["schema"] != "longbench_v1_run_v1" or config["protocol"] != protocol_record():
        raise ValueError("Unexpected LongBench run protocol.")
    if sha256(config["archive"]["path"]) != config["archive"]["sha256"]:
        raise ValueError("Shared archive manifest changed.")
    check_hashes(config["source_sha256"])
    model = config["model"]
    check_hashes(model["hf_metadata_sha256"])
    for stat in model["hf_weight_stats"]:
        check_stat(stat)
    if model["kind"] == "compressed":
        if sha256(Path(model["checkpoint"]) / "checkpoint_meta.json") != model["metadata_sha256"]:
            raise ValueError("Compressed checkpoint metadata changed.")
        check_stat(model["weight_stat"])
    if config["prewarm_group_size"] != 8 or config["device"] != "cuda:0":
        raise ValueError("Use one assigned visible GPU and existing group8 prewarm.")
    for name, expected in config["package_versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Installed dependency version changed: {name}")


def load_model(config, logger):
    record = config["model"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one assigned, visible GPU is required.")
    torch.cuda.reset_peak_memory_stats()
    if record["kind"] == "compressed":
        from train_utils.v6_model_loader import load_v6_model_checkpoint
        from train_utils.checkpoint_bpw import audit_state_dict
        from tools.cat_eval import _prepare_model_for_eval
        model, meta, loaded = load_v6_model_checkpoint(
            record["checkpoint"], base_model_path=record["hf_path"], map_location="cpu", strict=True)
        if loaded.missing_keys or loaded.unexpected_keys or meta["checkpoint_id"] != record["checkpoint_id"]:
            raise ValueError("Compressed model did not load fresh and strict.")
        budget = audit_state_dict(meta, model.state_dict(), max_bpw=2.5, max_lora_rank=0)
        if budget["contract_violations"]:
            raise ValueError(f"Compressed bit budget audit failed: {budget['contract_violations']}")
        model.eval().requires_grad_(False)
        warmup = _prepare_model_for_eval(model, config["device"], 8, logger, {})
        audit = {"strict_missing": [], "strict_unexpected": [], "checkpoint_id": meta["checkpoint_id"],
                 "persistent_tensors": len(model.state_dict()), "bit_budget": budget, "warmup": warmup}
    else:
        from rotation.model_utils import get_qwen3
        model = get_qwen3(record["hf_path"], None)
        model.eval().requires_grad_(False)
        model.to(config["device"])
        audit = {"loader": "rotation.model_utils.get_qwen3; device move only; no whole-model dtype cast"}
    if model.config.rope_scaling is not None or model.config._attn_implementation != "flash_attention_2":
        raise ValueError("Expected original unscaled RoPE and FlashAttention2.")
    if model.get_input_embeddings().weight.dtype != torch.bfloat16:
        raise ValueError("BF16 model activations are required.")
    # Original FP32 RoPE buffers must not be rounded by an extra model.bfloat16().
    audit.update(parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                 rope_buffers={name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                               for name, value in model.named_buffers() if "inv_freq" in name},
                 attn_implementation=model.config._attn_implementation,
                 rope_scaling=model.config.rope_scaling,
                 load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 load_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return model, audit
