"""Capture the preselected middle layer on exactly the original L0 documents."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.qwen3_w25.local_recovery import capture_shard_inputs
from experiments.rotation_factor_probe.capture import FORMAT, prepare_inputs, require, tensor_record
from litebsq.protected_channel_quant import encode_protected_channel_weight
from train_utils.utils import configure_deterministic_mode, set_seed


def prepare_middle_inputs(reference_manifest):
    """Replay the existing CPU preparation and bind its exact documents to L0."""
    reference_manifest = Path(reference_manifest).resolve()
    reference = json.loads(reference_manifest.read_text(encoding="utf-8"))
    require(reference["format"] == FORMAT and reference["status"] == "captured",
            "A completed L0 capture is required as the document and model reference.")
    check_hashes(reference["source_sha256"])
    check_hashes(reference["input_files_sha256"])
    for item in reference["hf_weight_files"] + [reference["checkpoint_weight_file"]]:
        check_stat(item)
    config, train, heldout, _l0_indices, record = prepare_inputs(
        reference["model_path"], reference["checkpoint_dir"])
    matched = ("model_type", "model_path", "checkpoint_dir", "checkpoint_id", "original_config",
               "protocol", "tokenizer", "data", "token_batches", "document_lengths", "document_boundaries")
    for key in matched:
        # AutoConfig.id2label has integer keys in memory and string keys in JSON.
        require(json.loads(json.dumps(record[key])) == reference[key],
                f"Middle-layer preparation changed the L0 reference: {key}.")
    layer_index = config.num_hidden_layers // 2
    targets = tuple(f"model.layers.{layer_index}.mlp.{category}" for category in ("gate_proj", "down_proj"))
    checkpoint = Path(record["checkpoint_dir"])
    meta = json.loads((checkpoint / "checkpoint_meta.json").read_text(encoding="utf-8"))
    state = torch.load(checkpoint / meta["state_dict_file"], map_location="cpu", weights_only=True, mmap=True)
    specs = {item["name"]: item for item in meta["converted_modules"]}
    indices, protection = {}, {}
    for name, count in zip(targets, (64, 256), strict=True):
        gate = name.endswith("gate_proj")
        n_in = config.hidden_size if gate else config.intermediate_size
        n_out = config.intermediate_size if gate else config.hidden_size
        spec = specs[name]
        require((spec["out_features"], spec["in_features"]) == (n_out, n_in)
                and spec["compressed_out_features"] == n_out
                and spec["compressed_in_features"] == n_in - count
                and spec["codebook_dim"] == 32 and spec["residual_stages"] == 2
                and spec["transpose"] == (not gate)
                and spec["protected_channel_quant_format"] == "int8"
                and spec.get("protected_output_indices") is None,
                f"Middle-layer source geometry changed: {name}.")
        key = name + ".protected_input_indices"
        value = state[key]
        require(value.dtype == torch.int64 and tuple(value.shape) == (count,)
                and value.unique().numel() == count and int(value.min()) >= 0 and int(value.max()) < n_in
                and torch.equal(value, value.sort().values), f"Invalid fixed middle-layer indices: {name}.")
        indices[name] = value.clone().contiguous()
        protection[name] = {"state_key": key, "indices": tensor_record(value), "spec": spec,
            "payload_source": "INT8 re-encoding of original HF columns; compressed checkpoint supplies indices only."}
    del state
    record.update(layer_index=layer_index, targets=list(targets), protection=protection,
        layer_selection="num_hidden_layers // 2, selected before middle-layer measurements",
        reference_l0_manifest={"path": str(reference_manifest), "sha256": sha256(reference_manifest)},
        l0_document_contract_exact=list(matched))
    record["source_sha256"][str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    utility_path = Path(__file__).resolve().parents[2] / "train_utils/utils.py"
    record["source_sha256"][str(utility_path)] = sha256(utility_path)
    record["input_files_sha256"][str(reference_manifest)] = sha256(reference_manifest)
    return config, train, heldout, indices, record


@torch.no_grad()
def run_capture(reference_manifest, output_dir):
    output = Path(output_dir).resolve()
    config, train, heldout, indices, record = prepare_middle_inputs(reference_manifest)
    require(not output.exists() and not any(output.is_relative_to(Path(record[key]).resolve())
            for key in ("model_path", "checkpoint_dir")), "Capture output must be a new independent directory.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Middle capture requires exactly one externally assigned visible CUDA GPU.")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "data_manifest.json"
    record.update(status="capturing", pid=os.getpid(), started_at_unix=time.time())
    write_json(path, record)
    try:
        from rotation.model_utils import get_llama, get_qwen3
        loader = get_qwen3 if config.model_type == "qwen3" else get_llama
        configure_deterministic_mode(True)
        set_seed(137)
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        teacher = loader(record["model_path"], None).eval().requires_grad_(False).to("cuda:0")
        teacher.config.use_cache = False
        backend = "flash_attention_2" if config.model_type == "qwen3" else "sdpa"
        require(teacher.config.model_type == config.model_type and teacher.config._attn_implementation == backend
                and teacher.config.rope_theta == config.rope_theta and teacher.config.rope_scaling == config.rope_scaling,
                "Original model backend or RoPE changed.")
        require({value.dtype for value in teacher.parameters()} == {torch.bfloat16},
                "Original model parameters must remain BF16 without a whole-model dtype cast.")
        rope = {name: tensor_record(value) for name, value in teacher.named_buffers() if "inv_freq" in name}
        require(rope and all(item["dtype"] == "torch.float32" for item in rope.values()),
                "Original FP32 RoPE buffers must remain unrounded.")
        layer_index, targets = record["layer_index"], record["targets"]
        mlp = teacher.get_submodule(f"model.layers.{layer_index}.mlp")
        require(isinstance(mlp.act_fn, torch.nn.SiLU) and all(
            teacher.get_submodule(name).bias is None for name in (*targets, f"model.layers.{layer_index}.mlp.up_proj")),
            "This comparison requires original SiLU and bias-free MLP projections.")
        up_weight = mlp.up_proj.weight.detach().cpu().clone()
        bundles = capture_shard_inputs(teacher, targets, train, heldout)
        for name, bundle in bundles.items():
            weight = bundle["reference_weight"]
            require(weight.dtype == torch.bfloat16, "Original projection dtype changed.")
            for split in ("train", "heldout"):
                x = bundle[split + "_inputs"]
                require(x.dtype == torch.bfloat16 and tuple(x.shape) == (
                    sum(record["document_lengths"][split]), weight.shape[1]), "Captured activation geometry changed.")
            qvalues, scales = encode_protected_channel_weight(
                weight.index_select(1, indices[name]).T.contiguous(), quant_format="int8")
            bundle.update(protected_input_indices=indices[name], protected_input_qvalues=qvalues,
                          protected_input_scales=scales)
        archive = {"format": FORMAT, "model_type": config.model_type, "model_path": record["model_path"],
                   "checkpoint_id": record["checkpoint_id"], "targets": bundles, "layer_index": layer_index,
                   "up_proj_weight": up_weight, "document_lengths": record["document_lengths"]}
        record["runtime"] = {"loader": "rotation.model_utils." + loader.__name__, "backend": backend,
            "whole_model_dtype_cast": False, "parameter_dtype": "torch.bfloat16", "rope_buffers": rope,
            "use_cache": False, "frozen": True, "eval": True, "actual_mlp_act_fn": type(mlp.act_fn).__module__ + "." + type(mlp.act_fn).__name__,
            "cuda_device": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf32": False, "deterministic_algorithms": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        record["tensors"] = {name: {key: tensor_record(value) for key, value in bundle.items()}
                             for name, bundle in bundles.items()}
        record["up_proj_weight"] = tensor_record(up_weight)
        check_hashes(record["source_sha256"])
        check_hashes(record["input_files_sha256"])
        for item in record["hf_weight_files"] + [record["checkpoint_weight_file"]]:
            check_stat(item)
        torch.save(archive, output / "inputs.pt")
        record["archive"] = {**file_stat(output / "inputs.pt"), "sha256": sha256(output / "inputs.pt")}
        record.update(status="captured", inputs_sha256=record["archive"]["sha256"], finished_at_unix=time.time())
        write_json(path, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at_unix=time.time())
        write_json(path, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_capture(args.reference_manifest, args.output_dir)


if __name__ == "__main__":
    main()
