"""Export native L0 or L0-3 down decoder/SparseBit checkpoints independently.

Only subset selection and orchestration differ from the existing selected-step
export. Continuous-state restore, CPU exact-sidecar projection, finalization,
BF16 conversion and v6 serialization all use the unchanged production helpers.
The complete deployed model still contains all 252 compressed projections.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from experiments.qwen3_w25.assemble_categories import EXPECTED, _require
from experiments.qwen3_w25.export_e2e_selected import restore_decoder_state
from experiments.qwen3_w25.audit_e2e_decoder import persistent_inventory
from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref


REPO = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "experiments/qwen3_w25/e2e_subset_export.py", "experiments/qwen3_w25/e2e_subset_audit.py",
    "experiments/qwen3_w25/export_e2e_selected.py", "experiments/qwen3_w25/audit_e2e_decoder.py",
    "experiments/qwen3_w25/cpu_bit_export.py",
    "compressed_e2e_fintuning/runtime_v6.py", "compressed_e2e_fintuning/runtime_v6_pipeline.py",
    "compressed_e2e_fintuning/mid_eval.py", "compressed_e2e_fintuning/v6_runtime_state.py",
    "train_utils/model_level_trainables.py", "train_utils/distill_precision.py",
    "train_utils/checkpoint_v6.py", "train_utils/v6_model_loader.py",
    "litebsq/vae_linear.py", "litebsq/fused_multistage_decoder.py",
    "sparse_bit_tuning/exact_checkpoint.py", "sparse_bit_tuning/manager.py",
    "sparse_bit_tuning/optimizer.py",
)


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}.")
    return value


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def selected_names(target_layers, target_modules):
    """The two authorized subsets are explicit; never silently expand selection."""
    layers = tuple(target_layers) if not isinstance(target_layers, str) else ()
    modules = tuple(target_modules) if not isinstance(target_modules, str) else (target_modules,)
    _require(layers in ((0,), (0, 1, 2, 3)) and all(type(i) is int for i in layers),
             "Subset export supports target_layers=0 or 0-3 only.")
    _require(modules == ("down_proj",), "Subset export supports down_proj only.")
    return tuple(f"model.layers.{layer}.mlp.down_proj" for layer in layers)


def validate_configuration(meta, state, snapshot):
    """Compare saved facts; never rewrite a schedule to reach finalization."""
    contract = meta["immutable_resume_contract"]
    cfg, training = snapshot["canonical_config"], snapshot["training_args"]
    _require(meta["train_mode"] == contract["train_mode"] == "decoder_sparse_bit",
             "Subset export requires native decoder_sparse_bit training.")
    _require(meta["lora_config"] is None and contract["lora"] is None, "LoRA is outside this export topology.")
    for key in ("norm_train_mode", "lm_head_train_mode"):
        _require(meta[key] == contract["aux"][key] == "none", f"Expected {key}=none.")
    _require(contract["aux"].get("residual_lora_mode", "none") == cfg["aux"]["residual_lora_mode"] == "none", "Residual LoRA is forbidden.")
    _require(contract["runtime"]["parallel_mode"] in ("dp", "layer_mp")
             and contract["runtime"]["offload_mode"] == "none"
             and not contract["runtime"]["distill_hif4_act"], "Only single-process DP or layer_mp, without offload or HiFloat4, is supported.")
    _require(contract["distributed"]["world_size"] == 1, "This recipe uses single-process training/export.")
    _require(contract["precision"]["bf16"] and not contract["precision"]["fp16"], "BF16 compute/export is required.")
    _require(contract["optimization"].get("distill_fp32_components") == ["decoder"], "Expected FP32 decoder parameters only.")
    _require(contract["sparse_bit"].get("proxy_coordinates") == "decoder_sensitivity"
             and contract["sparse_bit"]["active_ratio"] == 1.0,
             "Subset export requires all bits and decoder_sensitivity coordinates.")
    _require(type(state["global_step"]) is int and 0 < state["global_step"] <= state["max_steps"], "Invalid selected optimizer step.")
    _require(state["max_steps"] == training["max_steps"] == contract["optimization"]["steps"], "Original total steps disagree; do not edit them to export.")
    selected_names(meta["target_layers"], meta["target_modules"])
    _require(meta["target_layers"] == contract["target_layers"] == cfg["target_layers"],
             "Subset target layers disagree across metadata, contract and snapshot.")
    _require(meta["target_modules"] == contract["target_modules"] == cfg["target_modules"],
             "Subset target modules disagree across metadata, contract and snapshot.")
    _require(training["seed"] == contract["data"]["seed"], "Snapshot training seed disagrees.")

    # The normalized snapshot precedes dataset construction, which resolves the
    # worker count. Preserve that saved runtime fact and teacher/tokenizer identity;
    # all user-configurable mathematical fields below must match the original run.
    expected = deepcopy(contract)
    for key in ("train_mode", "data", "loss", "runtime"):
        expected[key] = deepcopy(cfg[key])
    expected["input_checkpoint_id"] = meta["round_base_checkpoint_id"]
    expected["optimization"] = {key: value for key, value in cfg["opt"].items() if key != "logging_steps"}
    expected["aux"] = {key: value for key, value in cfg["aux"].items() if not key.startswith("residual_lora_")}
    expected["precision"] = {key: training[key] for key in contract["precision"]}
    expected["sparse_bit"] = {key: cfg["bit_" + key] for key in ("active_ratio", "optimizer", "weight_decay", "round_steps")}
    expected["sparse_bit"]["bit_lr"] = cfg["bit_lr"]
    coordinates = cfg.get("bit_proxy_coordinates", "unit")
    _require(coordinates in ("unit", "decoder_sensitivity"), "Unsupported snapshot bit coordinates.")
    if meta["train_mode"] == "decoder_sparse_bit" and coordinates != "unit":
        expected["sparse_bit"]["proxy_coordinates"] = coordinates
    for key in contract["evaluation_execution"]:
        value = cfg["runtime"]["evaluation"][key] if key == "eval_after_save" else training[key]
        # The snapshot serializes str-enums as their values; the production
        # contract uses str(enum).lower() after TrainingArguments normalization.
        if key == "eval_strategy":
            value = str(IntervalStrategy(value)).lower()
        elif key == "save_strategy":
            value = str(SaveStrategy(value)).lower()
        expected["evaluation_execution"][key] = value
    validate_e2e_immutable_resume_contract(contract, expected)
    return contract


def inspect_step(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    meta = load_v6_training_step_meta(str(checkpoint))
    source, base_meta = resolve_training_step_round_base_ref(str(checkpoint), meta)
    _require(checkpoint.parent.name == "trainer_state", "Select a checkpoint inside its original run/trainer_state directory.")
    snapshot_path = checkpoint.parent.parent / "normalized_e2e_runtime_args.json"
    snapshot, state = _read_json(snapshot_path), _read_json(checkpoint / "trainer_state.json")
    validate_configuration(meta, state, snapshot)
    _require(set(meta["compressed_targets"]) == set(base_meta["compressed_targets"]) == set(EXPECTED)
             and not meta["pending_dense_targets"] and not meta["skip_targets"], "The selected step must cover all 252 projections.")
    _require(meta["base_model_path"] == base_meta["base_model_path"], "Step/base model identities disagree.")
    sparse_path = checkpoint / "sparse_bit_tuning"
    if meta["train_mode"] == "decoder_sparse_bit":
        _require((sparse_path / "exact_state.pt").is_file(), "Missing SparseBit exact sidecar.")
    else:
        _require(not sparse_path.exists(), "Decoder-only checkpoint unexpectedly contains SparseBit state.")
    return meta, state, snapshot, Path(source), base_meta


def export_checkpoint(checkpoint, output_dir, *, device="cuda:0"):
    from compressed_e2e_fintuning.mid_eval import temporary_inference_decode_mode
    from experiments.qwen3_w25.cpu_bit_export import restore_packed_bits_cpu
    from compressed_e2e_fintuning.runtime_v6 import _assert_final_runtime_clean, _build_v6_tokenizer, _finalize_decoders, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from experiments.qwen3_w25.summarize_results import _validate_checkpoint
    from litebsq.vae_linear import clear_model_vae_linear_cache
    from train_utils.checkpoint_v6 import _atomic_write_json, save_v6_full_checkpoint
    from train_utils.config.targets import collect_e2e_compressed_targets
    from train_utils.distill_precision import prepare_model_export
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    checkpoint, output = Path(checkpoint).resolve(), Path(output_dir).resolve()
    _require(not output.exists(), "Export output must be a new directory.")
    _require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Export requires one process.")
    meta, state, snapshot, source, base_meta = inspect_step(checkpoint)
    contract, training = meta["immutable_resume_contract"], SimpleNamespace(**snapshot["training_args"])
    device = torch.device(device)
    _require(torch.cuda.is_available() and device.type == "cuda", "Real finalization probes require an available CUDA GPU.")
    started = time.perf_counter()
    configure_e2e_determinism(bool(training.full_determinism))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_e2e_seed(training.seed)
    if training.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(training.tf32)
        torch.backends.cudnn.allow_tf32 = bool(training.tf32)
    source_meta, source_budget = _validate_checkpoint(source)
    _require(source_budget["budget_passed"], "Source fails the rank-zero 2.5-bpw budget.")
    files = [checkpoint / name for name in ("checkpoint_meta.json", "trainer_state.json", "training_model_state.pt")]
    files += [checkpoint.parent.parent / "normalized_e2e_runtime_args.json", source / "pytorch_model.bin", source / "checkpoint_meta.json"]
    if meta["train_mode"] == "decoder_sparse_bit":
        files.append(checkpoint / "sparse_bit_tuning" / "exact_state.pt")
    audit = {"runtime": "experiments.qwen3_w25.e2e_subset_export", "source_checkpoint": str(checkpoint),
        "source_checkpoint_id": meta["checkpoint_id"], "round_base_dir": str(source),
        "round_base_checkpoint_id": source_meta["checkpoint_id"], "source_global_step": state["global_step"],
        "source_max_steps": state["max_steps"], "train_mode": meta["train_mode"], "optimizer_steps_executed": 0,
        "trainer_or_main_optimizer_constructed": False, "teacher_or_training_data_loaded": False,
        "source_files_sha256": {str(path): _sha256(path) for path in files},
        "code_sha256": {name: _sha256(REPO / name) for name in CODE_FILES}}
    model, loaded_meta, _ = load_v6_model_checkpoint(str(source), strict=True, map_location="cpu")
    _require(loaded_meta["checkpoint_id"] == meta["round_base_checkpoint_id"], "Loaded round-base UUID changed.")
    inventory, _banks = persistent_inventory(model)
    selected = collect_e2e_compressed_targets(model, target_layers=tuple(meta["target_layers"]),
                                             target_modules=tuple(meta["target_modules"]), num_layers=36)
    names = selected_names(meta["target_layers"], meta["target_modules"])
    _require(tuple(name for name, _ in selected) == names and set(names) <= inventory.keys(),
             "Selected runtime topology disagrees with the authorized subset.")
    _require(all(len(inventory[name]) == 6 and len(_banks[name]) == 2 for name in names),
             "Subset recipe requires six persistent decoder tensors and two code banks per projection.")
    audit.update(selected_targets=list(names), selected_projection_count=len(names),
                 deployed_projection_count=len(inventory), selected_decoder_tensor_count=6 * len(names),
                 selected_code_bank_count=2 * len(names))
    audit["restored_decoder_tensors"] = restore_decoder_state(model, selected, checkpoint=checkpoint, snapshot=snapshot)
    _require(audit["restored_decoder_tensors"] == 6 * len(names), "Mutable state escaped selected decoders.")
    tokenizer = _build_v6_tokenizer(str(base_meta["base_model_path"]), SimpleNamespace(**snapshot["hf_args"]))
    _sync_model_padding_config(model, tokenizer)
    model.config.use_cache = False
    if meta["train_mode"] == "decoder_sparse_bit":
        audit["sparse_bit_restore"] = restore_packed_bits_cpu(
            selected, checkpoint=checkpoint, contract=contract)
    model.eval().to(device)
    probe_inputs = _build_finalization_probe_inputs(tokenizer)
    training_probe, dtype = _run_finalization_probe(model, probe_inputs)
    with temporary_inference_decode_mode(model, cache_decoded_weight=True):
        deployment_probe, deployment_dtype = _run_finalization_probe(model, probe_inputs)
    _require(deployment_dtype == dtype, "Deployment reference changed output dtype.")
    delta = deployment_probe - training_probe
    audit["decoder_compute_path_difference"] = {"source_path": "training_decode", "reference_path": "deployment_decode",
        "output_dtype": str(dtype), "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / training_probe.norm().clamp_min(1e-12))}
    _require(_finalize_decoders(selected) == len(names), "Not all selected decoders were finalized.")
    _assert_final_runtime_clean(model)
    finalized, finalized_dtype = _run_finalization_probe(model, probe_inputs)
    _require(finalized_dtype == dtype, "Finalization changed output dtype.")
    audit["core_structural_finalization_forward_parity"] = _assert_finalization_probe_close(
        deployment_probe, finalized, output_dtype=dtype, label="Selected-step deployment structural parity")
    audit["export_dtype"] = str(prepare_model_export(model, training))
    exported, exported_dtype = _run_finalization_probe(model, probe_inputs)
    _require(exported_dtype == dtype, "BF16 export changed output dtype.")
    delta = exported - finalized
    audit["export_precision_probe"] = {"output_dtype": str(exported_dtype), "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / finalized.norm().clamp_min(1e-12))}
    clear_model_vae_linear_cache(model)
    _assert_final_runtime_clean(model)
    model.requires_grad_(False).to("cpu")
    result = save_v6_full_checkpoint(model, str(output), checkpoint_kind="final_model",
        **{key: tuple(base_meta.get(key) or ()) for key in ("compressed_targets", "pending_dense_targets", "skip_targets",
             "legacy_original_only_sources", "completed_categories", "compression_categories")},
        train_mode=meta["train_mode"], norm_train_mode="none", lm_head_train_mode="none", lora_config=None,
        resolved_learning_rates=meta["resolved_learning_rates"], target_layers=meta["target_layers"], target_modules=meta["target_modules"],
        immutable_resume_contract=contract, base_model_path=base_meta["base_model_path"], tokenizer=tokenizer, save_config=True,
        finalized_status={"sparse_bit_committed": meta["train_mode"] == "decoder_sparse_bit", "decoder_finalized": True,
                          "lora_finalized": False, "aux_finalized": True, "runtime_clean": True, "inference_forward_parity": True},
        runtime_audit=audit)
    _require(result["checkpoint_id"] not in (meta["checkpoint_id"], base_meta["checkpoint_id"]), "Export reused an input UUID.")
    record = {"status": "exported", "checkpoint_id": result["checkpoint_id"], "output_dir": str(output),
              "elapsed_seconds": time.perf_counter() - started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "fresh_strict_state_audit_pending": True,
              "full_downstream_evaluation_pending": True, **audit}
    _atomic_write_json(record, str(output / "export_record.json"))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = export_checkpoint(args.checkpoint_dir, args.output_dir, device=args.device)
    print(json.dumps({key: result[key] for key in ("status", "checkpoint_id", "source_global_step", "source_max_steps", "output_dir")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
