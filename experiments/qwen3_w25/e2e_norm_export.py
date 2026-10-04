"""Export decoder plus the existing final norm without replaying training.

Isolated from active decoder/head trials; native finalization and BF16 storage.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from experiments.qwen3_w25.assemble_categories import CATEGORIES, EXPECTED, _assert_state_equal, _require
from experiments.qwen3_w25.audit_e2e_decoder import persistent_inventory
from experiments.qwen3_w25.export_e2e_selected import _read_json, _sha256
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref


NORM_KEY = "model.norm.weight"
REPO = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "experiments/qwen3_w25/e2e_norm_export.py", "experiments/qwen3_w25/e2e_norm_audit.py",
    "experiments/qwen3_w25/export_e2e_selected.py", "experiments/qwen3_w25/audit_e2e_decoder.py",
    "compressed_e2e_fintuning/runtime_v6.py", "compressed_e2e_fintuning/runtime_v6_pipeline.py",
    "compressed_e2e_fintuning/mid_eval.py", "compressed_e2e_fintuning/v6_runtime_state.py",
    "train_utils/model_level_trainables.py", "train_utils/distill_precision.py",
    "train_utils/checkpoint_v6.py", "train_utils/v6_model_loader.py",
    "litebsq/vae_linear.py", "litebsq/fused_multistage_decoder.py",
)


def validate_configuration(meta, state, snapshot):
    """Preserve the actual schedule, topology, precision and immutable contract."""
    contract = meta["immutable_resume_contract"]
    cfg, training = snapshot["canonical_config"], snapshot["training_args"]
    _require(meta["train_mode"] == contract["train_mode"] == cfg["train_mode"] == "decoder",
             "Final-norm export requires decoder training with frozen hard codes.")
    _require(meta["lora_config"] is None and contract["lora"] is None, "LoRA is forbidden.")
    for key, value in (("norm_train_mode", "final"), ("lm_head_train_mode", "none")):
        _require(meta[key] == contract["aux"][key] == cfg["aux"][key] == value, f"Expected {key}={value}.")
    _require(contract["aux"].get("residual_lora_mode", "none") == cfg["aux"]["residual_lora_mode"] == "none",
             "Residual LoRA is forbidden.")
    _require(contract["runtime"]["parallel_mode"] == "dp" and contract["runtime"]["offload_mode"] == "none"
             and not contract["runtime"]["distill_hif4_act"], "Expected DP without student offload or HiFloat4.")
    _require(contract["distributed"]["world_size"] == 1, "Expected single-process training/export.")
    _require(contract["precision"]["bf16"] and not contract["precision"]["fp16"], "BF16 compute/export is required.")
    _require(contract["optimization"]["distill_fp32_components"] == ["decoder", "norm"],
             "Expected FP32 decoder and final-norm parameters only.")
    _require(type(state["global_step"]) is int and 0 < state["global_step"] <= state["max_steps"],
             "Invalid selected optimizer step.")
    _require(state["max_steps"] == training["max_steps"] == contract["optimization"]["steps"],
             "Original total steps disagree; do not edit them to export.")
    _require(meta["target_layers"] == contract["target_layers"] == list(range(36)), "All 36 layers are required.")
    _require(meta["target_modules"] == contract["target_modules"] and set(meta["target_modules"]) == set(CATEGORIES),
             "All seven projection categories are required.")
    _require(cfg["target_layers"] == "all" or cfg["target_layers"] == meta["target_layers"], "Target layers disagree.")
    _require(cfg["target_modules"] == "all" or set(cfg["target_modules"]) == set(CATEGORIES), "Target modules disagree.")
    _require(training["seed"] == contract["data"]["seed"], "Training seed disagrees.")

    # The snapshot precedes data-worker resolution. Keep the saved worker and
    # teacher identity while validating every user-configurable mathematical field.
    expected = deepcopy(contract)
    for key in ("train_mode", "data", "loss", "runtime"):
        expected[key] = deepcopy(cfg[key])
    expected["input_checkpoint_id"] = meta["round_base_checkpoint_id"]
    expected["optimization"] = {key: value for key, value in cfg["opt"].items() if key != "logging_steps"}
    expected["aux"] = {key: value for key, value in cfg["aux"].items() if not key.startswith("residual_lora_")}
    expected["precision"] = {key: training[key] for key in contract["precision"]}
    expected["sparse_bit"] = {key: cfg["bit_" + key] for key in ("active_ratio", "optimizer", "weight_decay", "round_steps")}
    expected["sparse_bit"]["bit_lr"] = cfg["bit_lr"]
    for key in contract["evaluation_execution"]:
        value = cfg["runtime"]["evaluation"][key] if key == "eval_after_save" else training[key]
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
    _require(checkpoint.parent.name == "trainer_state", "Select a checkpoint in its original trainer_state directory.")
    snapshot = _read_json(checkpoint.parent.parent / "normalized_e2e_runtime_args.json")
    state = _read_json(checkpoint / "trainer_state.json")
    validate_configuration(meta, state, snapshot)
    _require(set(meta["compressed_targets"]) == set(base_meta["compressed_targets"]) == set(EXPECTED)
             and not meta["pending_dense_targets"] and not meta["skip_targets"], "Expected all 252 projections.")
    _require(meta["base_model_path"] == base_meta["base_model_path"], "Step/base model identities disagree.")
    _require(not base_meta.get("post_norm_head_linear"), "Round base must have an ordinary fused head.")
    _require(not (checkpoint / "sparse_bit_tuning").exists(), "Frozen-code trial contains SparseBit state.")
    return meta, state, snapshot, Path(source), base_meta


def restore_norm_state(model, selected, *, checkpoint, snapshot):
    """Restore the real decoder/final-norm component inventory, without an optimizer."""
    from compressed_e2e_fintuning.runtime_v6_pipeline import _enable_internal_decode_runtime
    from compressed_e2e_fintuning.v6_runtime_state import collect_e2e_mutable_state, restore_e2e_mutable_state
    from train_utils.checkpoint_v6 import load_v6_training_model_state
    from train_utils.config.configs import AuxTrainableConfig
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_trainables import build_model_level_trainable_selection

    cfg = snapshot["canonical_config"]
    _enable_internal_decode_runtime(selected, decoder_enabled=True, sparse_enabled=False,
                                   vae_decoder_checkpoint=cfg["runtime"]["vae_decoder_checkpoint"])
    selection = build_model_level_trainable_selection(
        model, aux=AuxTrainableConfig(**cfg["aux"]), compressed_modules=selected, dense_target_modules=(),
        rank=cfg["lora"]["rank"], alpha=cfg["lora"]["alpha"], dropout=cfg["lora"]["dropout"],
        fp32_components=cfg["opt"]["distill_fp32_components"], train_decoder=True, train_lora=False,
        decoder_execution_mode="trainable_decoder", freeze=True)
    _require(selection.peft_model is model and len(selection.norm_parameters) == 1
             and not selection.lm_head_parameters and not selection.lora_parameters
             and not selection.residual_lora_parameters, "Unexpected mutable component topology.")
    configure_distill_precision(selection, components=cfg["opt"]["distill_fp32_components"],
                               training_args=SimpleNamespace(**snapshot["training_args"]))
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location="cpu")
    _require({row["component_class"] for row in manifest} == {"decoder", "norm"},
             "Only decoder and final-norm mutable state may be restored.")
    _require({row["name"] for row in manifest if row["component_class"] == "norm"} == {NORM_KEY},
             "Only model.norm.weight may be selected as a norm.")
    _require(all(tensor.dtype == torch.float32 for tensor in mutable.values()), "Mutable decoder/norm state must be FP32.")
    restore_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected,
                             checkpoint_state=mutable, checkpoint_manifest=manifest)
    restored, classes, _manifest = collect_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected)
    compared = _assert_state_equal(restored, mutable, "Selected FP32 decoder and final norm mutable restore")
    return {"tensors_compared": compared,
            "component_counts": {label: sum(value == label for value in classes.values()) for label in ("decoder", "norm")}}


def export_checkpoint(checkpoint, output_dir, *, device="cuda:0"):
    from compressed_e2e_fintuning.mid_eval import temporary_inference_decode_mode
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
    training = SimpleNamespace(**snapshot["training_args"])
    device = torch.device(device)
    _require(torch.cuda.is_available() and device.type == "cuda", "Finalization probes require CUDA.")
    started = time.perf_counter()
    configure_e2e_determinism(bool(training.full_determinism))
    torch.cuda.set_device(device)
    set_e2e_seed(training.seed)
    if training.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(training.tf32)
        torch.backends.cudnn.allow_tf32 = bool(training.tf32)
    source_meta, source_budget = _validate_checkpoint(source)
    _require(source_budget["budget_passed"], "Source fails the rank-zero 2.5-bpw budget.")
    files = [checkpoint / name for name in ("checkpoint_meta.json", "trainer_state.json", "training_model_state.pt")]
    files += [checkpoint.parent.parent / "normalized_e2e_runtime_args.json", source / "pytorch_model.bin", source / "checkpoint_meta.json"]
    audit = {"runtime": "experiments.qwen3_w25.e2e_norm_export", "source_checkpoint": str(checkpoint),
        "source_checkpoint_id": meta["checkpoint_id"], "round_base_dir": str(source),
        "round_base_checkpoint_id": source_meta["checkpoint_id"], "source_global_step": state["global_step"],
        "source_max_steps": state["max_steps"], "train_mode": "decoder", "norm_train_mode": "final", "lm_head_train_mode": "none",
        "optimizer_steps_executed": 0, "trainer_or_main_optimizer_constructed": False,
        "teacher_or_training_data_loaded": False, "source_files_sha256": {str(path): _sha256(path) for path in files},
        "code_sha256": {name: _sha256(REPO / name) for name in CODE_FILES}}
    model, loaded_meta, _ = load_v6_model_checkpoint(str(source), strict=True, map_location="cpu")
    _require(loaded_meta["checkpoint_id"] == meta["round_base_checkpoint_id"], "Round-base UUID changed.")
    _require(type(model.lm_head) is torch.nn.Linear and model.lm_head.bias is None
             and tuple(model.lm_head.weight.shape) == (151936, 4096)
             and not model.config.tie_word_embeddings, "Expected the untied original Qwen3-8B head.")
    inventory, _banks = persistent_inventory(model)
    selected = collect_e2e_compressed_targets(model, target_layers=tuple(meta["target_layers"]),
                                             target_modules=tuple(meta["target_modules"]), num_layers=36)
    _require({name for name, _ in selected} == set(inventory), "Selected topology does not cover all 252 decoders.")
    audit["mutable_restore"] = restore_norm_state(model, selected, checkpoint=checkpoint, snapshot=snapshot)
    tokenizer = _build_v6_tokenizer(str(base_meta["base_model_path"]), SimpleNamespace(**snapshot["hf_args"]))
    _sync_model_padding_config(model, tokenizer)
    model.config.use_cache = False
    model.eval().to(device)
    probe_inputs = _build_finalization_probe_inputs(tokenizer)
    training_probe, dtype = _run_finalization_probe(model, probe_inputs)
    with temporary_inference_decode_mode(model, cache_decoded_weight=True):
        deployment_probe, deployment_dtype = _run_finalization_probe(model, probe_inputs)
    _require(deployment_dtype == dtype, "Deployment reference changed output dtype.")
    delta = deployment_probe - training_probe
    audit["decoder_compute_path_difference"] = {"output_dtype": str(dtype), "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / training_probe.norm().clamp_min(1e-12))}
    _require(_finalize_decoders(selected) == 252, "Not all selected decoders were finalized.")
    _assert_final_runtime_clean(model)
    core, core_dtype = _run_finalization_probe(model, probe_inputs)
    _require(core_dtype == dtype, "Core finalization changed output dtype.")
    audit["core_structural_finalization_forward_parity"] = _assert_finalization_probe_close(
        deployment_probe, core, output_dtype=dtype, label="Selected-step deployment structural parity")
    audit["structural_finalization_forward_parity"] = audit["core_structural_finalization_forward_parity"]
    restored_norm = model.model.norm.weight.detach().clone()
    audit["export_dtype"] = str(prepare_model_export(model, training))
    exported, exported_dtype = _run_finalization_probe(model, probe_inputs)
    _require(exported_dtype == dtype, "BF16 export changed output dtype.")
    _assert_state_equal({NORM_KEY: model.model.norm.weight},
                        {NORM_KEY: restored_norm.bfloat16()}, "Exact native final-norm BF16 cast")
    delta = exported - core
    audit["export_precision_probe"] = {"output_dtype": str(exported_dtype), "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / core.norm().clamp_min(1e-12))}
    audit["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    clear_model_vae_linear_cache(model)
    _assert_final_runtime_clean(model)
    model.requires_grad_(False).to("cpu")
    result = save_v6_full_checkpoint(model, str(output), checkpoint_kind="final_model",
        **{key: tuple(base_meta.get(key) or ()) for key in ("compressed_targets", "pending_dense_targets", "skip_targets",
             "legacy_original_only_sources", "completed_categories", "compression_categories")},
        train_mode="decoder", norm_train_mode="final", lm_head_train_mode="none", lora_config=None,
        resolved_learning_rates=meta["resolved_learning_rates"], target_layers=meta["target_layers"], target_modules=meta["target_modules"],
        immutable_resume_contract=meta["immutable_resume_contract"], base_model_path=base_meta["base_model_path"],
        tokenizer=tokenizer, save_config=True,
        finalized_status={"sparse_bit_committed": False, "decoder_finalized": True,
                          "lora_finalized": False, "aux_finalized": True, "runtime_clean": True, "inference_forward_parity": True},
        runtime_audit=audit)
    _require(result["checkpoint_id"] not in (meta["checkpoint_id"], base_meta["checkpoint_id"]), "Export reused an input UUID.")
    record = {"status": "exported", "checkpoint_id": result["checkpoint_id"], "output_dir": str(output),
              "elapsed_seconds": time.perf_counter() - started, "fresh_strict_state_audit_pending": True,
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
    export_checkpoint(args.checkpoint_dir, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
