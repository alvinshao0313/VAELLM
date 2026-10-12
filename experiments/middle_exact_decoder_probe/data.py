"""Fixed unseen confirmation documents and original-model gate activations."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

from datasets import Dataset
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.qwen3_w25.calibration import _sha256, build_calibration_batches
from experiments.qwen3_w25.local_recovery import capture_shard_inputs
from experiments.rotation_factor_probe.capture import require, tensor_record
from experiments.rotation_factor_probe.metrics import aggregate, squared_error
from experiments.rotation_factor_probe.run import load_archive
from train_utils.distill_data import _instruction_prompt_and_response
from train_utils.utils import configure_deterministic_mode, set_seed


FORMAT = "middle_exact_decoder_confirmation_v1"
RAW_IDENTITY_SHA256 = "a9a0af2efa51100ce39aa889d11bc04229781d850866dad66e64a7a20deb3950"
CONFIRMATION_ROWS = [48475, 16796, 48723, 863, 6414, 11776, 33467, 31884,
    1789, 13646, 28326, 34834, 49572, 26600, 37953, 48608, 5765, 2425,
    17652, 14504, 840, 3367, 49838, 50629, 25864, 11051, 44493, 49134,
    19180, 50972, 35909, 42031]


def _merge_pins(destination, incoming):
    require(all(key not in destination or destination[key] == value for key, value in incoming.items()),
            "The existing captures pin conflicting file versions.")
    destination.update(incoming)


def prepare_confirmation(l0_dir, middle_dir):
    """Return a JSON-ready identity record and the native CPU confirmation batch."""
    directories = [Path(l0_dir).resolve(), Path(middle_dir).resolve()]
    records, references, sources, inputs = [], {}, {}, {}
    for directory in directories:
        archive, manifest = load_archive(directory)
        check_hashes(manifest["source_sha256"])
        check_hashes(manifest["input_files_sha256"])
        _merge_pins(sources, manifest["source_sha256"])
        _merge_pins(inputs, manifest["input_files_sha256"])
        for name in ("data_manifest.json", "inputs.pt"):
            inputs[str(directory / name)] = sha256(directory / name)
        records.append(manifest)
        gate_names = [name for name in archive["targets"] if name.endswith("gate_proj")]
        require(len(gate_names) == 1, "Each reference archive must have exactly one gate target.")
        gate = gate_names[0]
        references[gate] = {"directory": str(directory),
            "gate": tensor_record(archive["targets"][gate]["reference_weight"]),
            "up": tensor_record(archive["up_proj_weight"]),
            "down": tensor_record(archive["targets"][gate.replace("gate_proj", "down_proj")]["reference_weight"])}
        del archive
    original, middle = records
    require(all(original[key] == middle[key] for key in
                ("model_path", "model_type", "checkpoint_id", "data", "tokenizer", "original_config")),
            "The old L0 and middle captures do not share one model and the same 64 documents.")
    config = AutoConfig.from_pretrained(original["model_path"], local_files_only=True, trust_remote_code=False)
    require(config.model_type in ("qwen3", "llama") and config.hidden_act == "silu",
            "Only the original Qwen/Llama SiLU models are supported.")
    targets = ["model.layers.0.mlp.gate_proj", f"model.layers.{config.num_hidden_layers // 2}.mlp.gate_proj"]
    require(list(references) == targets and middle["layer_index"] == config.num_hidden_layers // 2,
            "Require the preselected L0 and middle targets, in that order.")
    tokenizer = AutoTokenizer.from_pretrained(original["model_path"], use_fast=True,
        local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prefix, batch, data = build_calibration_batches(tokenizer, seed=137,
        train_documents=64, heldout_documents=32, max_length=256)
    old_documents = original["data"]["train"]["documents"] + original["data"]["heldout"]["documents"]
    require(data["train"]["documents"] == old_documents, "The accepted first 64 documents changed.")
    for start, split in ((0, "train"), (32, "heldout")):
        require(all(tensor_record(value[start:start + 32]) == original["token_batches"][split][key]
                    for key, value in prefix.items()), "The old native 32+32 token batches do not replay exactly.")
    documents = data["heldout"]["documents"]
    require([item["row_index"] for item in documents] == CONFIRMATION_ROWS, "Fixed confirmation row order changed.")
    identity = [{key: item[key] for key in ("row_index", "content_sha256", "group_sha256")}
                for item in documents]
    require(_sha256(identity) == RAW_IDENTITY_SHA256, "Model-independent confirmation identities changed.")
    for key in ("row_index", "content_sha256", "group_sha256", "token_sha256"):
        previous, current = ({item[key] for item in group} for group in (old_documents, documents))
        require(len(previous) == 64 and len(current) == 32 and not previous & current,
                f"Confirmation overlaps or contains duplicates: {key}.")
    raw = Dataset.from_file(data["source_arrow"])
    raw_text_hashes, rendered_hashes = [], []
    for item in old_documents + documents:
        row = dict(raw[item["row_index"]])
        require(_sha256([row[key] for key in ("instruction", "input", "output")]) == item["content_sha256"]
                and _sha256([" ".join(row[key].split()) for key in ("instruction", "input")]) == item["group_sha256"],
                "Source Arrow no longer reproduces document content/group hashes.")
        prompt, response = _instruction_prompt_and_response(row, text_format="alpaca")
        raw_text_hashes.append(hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest())
        rendered_hashes.append(hashlib.sha256((prompt + response).encode("utf-8")).hexdigest())
    require(len(set(raw_text_hashes)) == len(set(rendered_hashes)) == 96,
            "Raw or actually rendered confirmation text overlaps the old 64 documents.")
    lengths = [int(value) for value in batch["attention_mask"].sum(dim=1).tolist()]
    require(lengths == [item["valid_tokens"] for item in documents]
            and len(lengths) == 32 and all(0 < value <= 256 for value in lengths), "Native confirmation lengths changed.")
    root = Path(__file__).resolve().parents[2]
    for path in (Path(__file__).resolve(), root / "experiments/rotation_factor_probe/metrics.py",
                 root / "experiments/rotation_factor_probe/run.py", root / "train_utils/utils.py"):
        sources[str(path)] = sha256(path)
    for stat in original["hf_weight_files"]:
        check_stat(stat)
    require(all(value.device.type == "cpu" for value in batch.values()), "Confirmation batches must remain on CPU.")
    record = {"format": FORMAT, "status": "cpu_ready", "model_path": original["model_path"],
        "model_type": config.model_type, "checkpoint_id": original["checkpoint_id"], "targets": targets,
        "source_sha256": sources, "input_sha256": inputs, "hf_weight_files": original["hf_weight_files"],
        "reference_weights": references, "tokenizer": original["tokenizer"],
        "rope_theta": config.rope_theta, "rope_scaling": config.rope_scaling,
        "protocol": {"seed": 137, "prefix_documents_skipped": 64, "confirmation_documents": 32,
            "max_length": 256, "task": "lm", "formatting_version": data["formatting_version"],
            "usage": "Endpoint measurement only; no code/decoder fitting or endpoint selection.",
            "independence_boundary": "Disjoint from this local study's prior 64; no pretraining or earlier calibration exclusion claim."},
        "data_source": {key: data[key] for key in ("dataset_path", "revision", "split", "source_arrow", "source_rows")},
        "documents": [{**item, "raw_text_utf8_sha256": raw_text_hashes[64 + i],
                       "rendered_utf8_sha256": rendered_hashes[64 + i]} for i, item in enumerate(documents)],
        "document_lengths": lengths, "valid_tokens": sum(lengths),
        "token_batches": {key: tensor_record(value) for key, value in batch.items()},
        "raw_identity_sha256": RAW_IDENTITY_SHA256,
        "raw_text_hash_list_sha256": _sha256(raw_text_hashes[64:]),
        "rendered_hash_list_sha256": _sha256(rendered_hashes[64:]),
        "prefix_metadata_and_token_batches_exact": True, "new_old_all_identity_sets_disjoint": True}
    return record, batch


@torch.no_grad()
def capture_confirmation(record, batch, output_dir, device):
    """Capture both gate inputs in one original-model forward per new document."""
    output, device = Path(output_dir).resolve(), torch.device(device)
    require(record["status"] == "cpu_ready" and device.type == "cuda"
            and torch.cuda.device_count() == 1, "One assigned GPU and CPU-ready confirmation are required.")
    require(not output.exists() and not output.is_relative_to(Path(record["model_path"]).resolve())
            and not any(output.is_relative_to(Path(value["directory"])) for value in record["reference_weights"].values()),
            "Capture output must be new and outside every original archive/model directory.")
    check_hashes(record["source_sha256"])
    check_hashes(record["input_sha256"])
    require({key: tensor_record(value) for key, value in batch.items()} == record["token_batches"],
            "Confirmation token batch differs from preparation.")
    for stat in record["hf_weight_files"]:
        check_stat(stat)
    from rotation.model_utils import get_llama, get_qwen3
    loader = get_qwen3 if record["model_type"] == "qwen3" else get_llama
    configure_deterministic_mode(True)
    set_seed(137)
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True)
    manifest = {**record, "status": "capturing", "started_unix": time.time()}
    write_json(output / "data_manifest.json", manifest)
    teacher = None
    try:
        teacher = loader(record["model_path"], None).eval().requires_grad_(False).to(device)
        teacher.config.use_cache = False
        backend = "flash_attention_2" if record["model_type"] == "qwen3" else "sdpa"
        require(teacher.config.model_type == record["model_type"] and teacher.config._attn_implementation == backend
                and teacher.config.rope_theta == record["rope_theta"] and teacher.config.rope_scaling == record["rope_scaling"],
                "Native model, attention backend, or RoPE configuration changed.")
        require({parameter.dtype for parameter in teacher.parameters()} == {torch.bfloat16},
                "Original model parameters must be BF16 without a whole-model dtype cast.")
        rope = {name: tensor_record(value) for name, value in teacher.named_buffers() if "inv_freq" in name}
        require(rope and all(value["dtype"] == "torch.float32" for value in rope.values()), "FP32 RoPE buffers changed.")
        for target, reference in record["reference_weights"].items():
            mlp = teacher.get_submodule(target.rsplit(".", 1)[0])
            require(isinstance(mlp.act_fn, torch.nn.SiLU), "Native MLP activation must be SiLU.")
            for category in ("gate", "up", "down"):
                layer = getattr(mlp, category + "_proj")
                require(layer.bias is None and tensor_record(layer.weight) == reference[category],
                        f"Original HF {target}/{category} differs from the prior archive.")
        del mlp, layer
        first = {key: value[:16] for key, value in batch.items()}
        second = {key: value[16:] for key, value in batch.items()}
        bundles = capture_shard_inputs(teacher, record["targets"], first, second)
        captured = {}
        for target, bundle in bundles.items():
            require(tensor_record(bundle["reference_weight"]) == record["reference_weights"][target]["gate"],
                    "Capture target weight differs from the original reference.")
            captured[target] = torch.cat((bundle["train_inputs"], bundle["heldout_inputs"]), dim=0)
            require(captured[target].dtype == torch.bfloat16 and captured[target].shape ==
                    (sum(record["document_lengths"]), bundle["reference_weight"].shape[1]),
                    "Confirmation input dtype, token order, or geometry changed.")
        cache = {"format": FORMAT, "model_type": record["model_type"], "inputs": captured,
                 "document_lengths": record["document_lengths"]}
        manifest["runtime"] = {"loader": "rotation.model_utils." + loader.__name__, "backend": backend,
            "whole_model_dtype_cast": False, "parameter_dtype": "torch.bfloat16", "rope_buffers": rope,
            "frozen": True, "eval": True, "use_cache": False, "same_forward_two_targets": True,
            "actual_documents": 32, "internal_capture_split_sizes": [16, 16],
            "device": torch.cuda.get_device_name(device), "tf32": False, "deterministic_algorithms": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        manifest["input_tensors"] = {name: tensor_record(value) for name, value in captured.items()}
        check_hashes(record["source_sha256"])
        check_hashes(record["input_sha256"])
        for stat in record["hf_weight_files"]:
            check_stat(stat)
        torch.save(cache, output / "inputs.pt")
        manifest.update(status="captured", finished_unix=time.time(),
            archive={**file_stat(output / "inputs.pt"), "sha256": sha256(output / "inputs.pt")})
        write_json(output / "data_manifest.json", manifest)
        return cache, manifest
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        write_json(output / "data_manifest.json", manifest)
        raise
    finally:
        del teacher
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def measure_confirmation(original_archive, target_name, weight, inputs, lengths, device):
    """One real 32-document endpoint measurement with the original BF16 MLP."""
    require(target_name.endswith("gate_proj") and len(lengths) == 32 and all(value > 0 for value in lengths)
            and sum(lengths) == len(inputs), "Require one complete, ordered confirmation set of 32 documents.")
    reference = original_archive["targets"][target_name]["reference_weight"]
    require(weight.shape == reference.shape and weight.dtype == inputs.dtype == torch.bfloat16
            and inputs.shape[1] == reference.shape[1], "Confirmation weights/inputs must preserve native BF16 geometry.")
    original, candidate = reference.to(device), weight.to(device)
    up = original_archive["up_proj_weight"].to(device)
    down = original_archive["targets"][target_name.replace("gate_proj", "down_proj")]["reference_weight"].to(device)
    linear_documents, block_documents, offset = [], [], 0
    for index, length in enumerate(lengths):
        x = inputs[offset:offset + length].to(device)
        offset += length
        dense_output, compressed_output = F.linear(x, original), F.linear(x, candidate)
        error, signal = squared_error(compressed_output, dense_output)
        require(signal > 0, "A confirmation document has zero linear reference energy.")
        linear_documents.append({"index": index, "tokens": length, "squared_error": error,
                                 "squared_reference": signal, "nmse": error / signal})
        up_output = F.linear(x, up)
        dense_block = F.linear(F.silu(dense_output) * up_output, down)
        compressed_block = F.linear(F.silu(compressed_output) * up_output, down)
        error, signal = squared_error(compressed_block, dense_block)
        require(signal > 0, "A confirmation document has zero MLP reference energy.")
        block_documents.append({"index": index, "tokens": length, "squared_error": error,
                                "squared_reference": signal, "nmse": error / signal})
    return {"tokens": offset, "linear_output": aggregate(linear_documents),
            "mlp_block_output": aggregate(block_documents),
            "arithmetic": "BF16 Linear/SiLU/product/down; FP32 differences and FP64 squared sums.",
            "scope": "One fixed unseen 32-document endpoint; original up/down and no parameter selection."}
