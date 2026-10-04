"""Independently finalize one saved all-projection rank4 LoRA step, without training."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.w25_eval.lora_state import ALPHA, RANK, inspect_step, project_lora_state, lora_storage


REPO = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "experiments/w25_eval/lora_state.py", "experiments/w25_eval/lora_export.py",
    "experiments/w25_eval/lora_audit.py", "experiments/w25_eval/model_inventory.py", "e2e_common/full_lora.py",
    "compressed_e2e_fintuning/v6_runtime_state.py", "compressed_e2e_fintuning/runtime_v6.py",
    "compressed_e2e_fintuning/runtime_v6_pipeline.py", "train_utils/model_level_trainables.py",
    "train_utils/distill_precision.py", "train_utils/checkpoint_v6.py",
    "train_utils/v6_model_loader.py", "train_utils/checkpoint_bpw.py", "litebsq/vae_linear.py",
)


def restore_lora_model(model, selected, checkpoint, meta, snapshot, architecture):
    from compressed_e2e_fintuning.runtime_v6_pipeline import _enable_internal_decode_runtime
    from compressed_e2e_fintuning.v6_runtime_state import restore_e2e_mutable_state
    from e2e_common.full_lora import collect_exact_peft_lora_config
    from train_utils.checkpoint_v6 import load_v6_training_model_state
    from train_utils.config.configs import AuxTrainableConfig
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_trainables import build_model_level_trainable_selection

    cfg = snapshot["canonical_config"]
    _enable_internal_decode_runtime(selected, decoder_enabled=False, sparse_enabled=False,
                                    vae_decoder_checkpoint=cfg["runtime"]["vae_decoder_checkpoint"])
    selection = build_model_level_trainable_selection(
        model, aux=AuxTrainableConfig(**cfg["aux"]), compressed_modules=selected,
        dense_target_modules=(), rank=RANK, alpha=ALPHA, dropout=0.0, rank_explicit=True,
        fp32_components=("lora",), train_decoder=False, train_lora=True, freeze=True)
    model = selection.peft_model or model
    configure_distill_precision(selection, components=("lora",), training_args=SimpleNamespace(bf16=True, fp16=False))
    actual = collect_exact_peft_lora_config(model, default_rank=RANK, alpha=ALPHA, dropout=0.0)
    _require(actual == meta["lora_config"], "Rebuilt adapter topology differs from saved state.")
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location="cpu")
    projected = project_lora_state(mutable, manifest, architecture["projection_shapes"])
    del projected
    restore_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected,
                             checkpoint_state=mutable, checkpoint_manifest=manifest)
    del mutable, manifest
    return model


def export_checkpoint(checkpoint, output_dir, *, device="cuda:0"):
    from compressed_e2e_fintuning.runtime_v6 import _assert_final_runtime_clean, _build_v6_tokenizer, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from e2e_common.full_lora import finalize_model_level_lora
    from experiments.e2e_0920_search.export_selected import _compressed_base_digest
    from litebsq.vae_linear import clear_model_vae_linear_cache
    from train_utils.checkpoint_v6 import _atomic_write_json, save_v6_full_checkpoint
    from train_utils.config.targets import collect_e2e_compressed_targets
    from train_utils.distill_precision import prepare_model_export
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    checkpoint, output = Path(checkpoint).resolve(), Path(output_dir).resolve()
    _require(not output.exists() and int(os.environ.get("WORLD_SIZE", "1")) == 1,
             "Use a new output and one export process.")
    meta, state, snapshot, source, base_meta, architecture = inspect_step(checkpoint)
    shapes = architecture["projection_shapes"]
    device = torch.device(device)
    _require(torch.cuda.is_available() and device.type == "cuda", "Actual finalization probes require CUDA.")
    training = SimpleNamespace(**snapshot["training_args"])
    configure_e2e_determinism(bool(training.full_determinism))
    set_e2e_seed(int(training.seed))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    if training.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(training.tf32)
        torch.backends.cudnn.allow_tf32 = bool(training.tf32)
    started = time.perf_counter()
    files = [checkpoint / name for name in ("checkpoint_meta.json", "trainer_state.json", "training_model_state.pt")]
    files += [checkpoint.parent.parent / "normalized_e2e_runtime_args.json", source / "checkpoint_meta.json", source / "pytorch_model.bin"]
    record = {"runtime": "experiments.w25_eval.lora_export", "source_checkpoint": str(checkpoint),
        "source_checkpoint_id": meta["checkpoint_id"], "round_base_dir": str(source),
        "round_base_checkpoint_id": base_meta["checkpoint_id"], "source_global_step": state["global_step"],
        "source_max_steps": state["max_steps"], "restored_lora_tensors": lora_storage(shapes)["tensors"], "architecture": architecture,
        "optimizer_steps_executed": 0, "trainer_or_main_optimizer_constructed": False,
        "teacher_or_training_data_loaded": False, "decoder_finalization_performed": False,
        "head_fusion_performed": False, "source_files_sha256": {str(p): _sha256(p) for p in files},
        "code_sha256": {p: _sha256(REPO / p) for p in CODE_FILES}}
    model, loaded, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True)
    _require(loaded["checkpoint_id"] == base_meta["checkpoint_id"], "Source changed during load.")
    selected = collect_e2e_compressed_targets(model, target_layers=tuple(range(architecture["num_hidden_layers"])),
        target_modules=tuple(meta["target_modules"]), num_layers=architecture["num_hidden_layers"])
    _require({name for name, _ in selected} == set(shapes), "Incomplete actual LoRA targets.")
    fingerprint = _compressed_base_digest(selected)
    model = restore_lora_model(model, selected, checkpoint, meta, snapshot, architecture)
    _require(_compressed_base_digest(selected) == fingerprint, "Restore changed the frozen compressed base.")
    tokenizer = _build_v6_tokenizer(base_meta["base_model_path"], SimpleNamespace(**snapshot["hf_args"]))
    _sync_model_padding_config(model, tokenizer)
    model.config.use_cache = False
    model.eval().to(device)
    inputs = _build_finalization_probe_inputs(tokenizer)
    before, dtype = _run_finalization_probe(model, inputs)
    model = finalize_model_level_lora(model, compressed_proxy_names=[name for name, _ in selected])
    core, core_dtype = _run_finalization_probe(model, inputs)
    _require(dtype == core_dtype, "LoRA finalization changed output precision.")
    record["core_structural_finalization_forward_parity"] = _assert_finalization_probe_close(
        before, core, output_dtype=dtype, label="LoRA proxy to native factor finalization")
    clear_model_vae_linear_cache(model)
    _assert_final_runtime_clean(model)
    record["export_dtype"] = str(prepare_model_export(model, training))
    exported, exported_dtype = _run_finalization_probe(model, inputs)
    difference = exported - core
    record["export_precision_probe"] = {"output_dtype": str(exported_dtype),
        "max_abs": float(difference.abs().max()), "relative_l2": float(difference.norm() / core.norm().clamp_min(1e-12))}
    clear_model_vae_linear_cache(model)
    model.requires_grad_(False).to("cpu")
    _require(_compressed_base_digest(selected) == fingerprint, "Finalization changed the frozen compressed base.")
    result = save_v6_full_checkpoint(model, str(output), checkpoint_kind="final_model",
        **{key: tuple(base_meta.get(key) or ()) for key in ("compressed_targets", "pending_dense_targets", "skip_targets",
            "legacy_original_only_sources", "completed_categories", "compression_categories")},
        train_mode="lora", norm_train_mode="none", lm_head_train_mode="none", lora_config=None,
        resolved_learning_rates=meta["resolved_learning_rates"], target_layers=meta["target_layers"],
        target_modules=meta["target_modules"], immutable_resume_contract=meta["immutable_resume_contract"],
        finalized_status={"sparse_bit_committed": False, "decoder_finalized": False, "lora_finalized": True,
            "aux_finalized": True, "runtime_clean": True, "inference_forward_parity": True},
        runtime_audit=record, base_model_path=base_meta["base_model_path"], tokenizer=tokenizer, save_config=True)
    _require(all(_sha256(p) == record["source_files_sha256"][str(p)] for p in files), "Export mutated an input file.")
    record.update(status="exported", checkpoint_id=result["checkpoint_id"], output_dir=str(output),
        elapsed_seconds=time.perf_counter() - started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device), fresh_strict_state_audit_pending=True,
        full_downstream_evaluation_pending=True)
    _atomic_write_json(record, str(output / "export_record.json"))
    logging.getLogger(__name__).info("Exported LoRA step %d; fresh audit/evaluation remain.", state["global_step"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint_dir, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
