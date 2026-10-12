"""Independent native decoder/full-head restore and BF16 v6 export."""
from __future__ import annotations

import argparse
import logging
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.e2e_subset_export import CODE_FILES as QWEN_CODE_FILES, _sha256
from experiments.general_lm_head_recovery.state import restore_union_state, HEAD_KEY, assert_head_cast
from experiments.general_lm_head_recovery.state import checkpoint_inventory, inspect_step, persistent_inventory, selected_layers, selected_modules, selected_names

REPO = Path(__file__).resolve().parents[2]
CODE_FILES = QWEN_CODE_FILES + (
    'experiments/w25_eval/model_inventory.py', 'experiments/w25_eval/acceptance.py',
    'experiments/w25_eval/subset_state.py', 'experiments/w25_eval/subset_export.py',
    'experiments/w25_eval/subset_audit.py', 'rotation/model_utils.py',
    'experiments/w25_eval/norm_state.py', 'experiments/qwen3_w25/e2e_fullhead_export.py',
    'experiments/qwen3_w25/e2e_aux_export.py', 'experiments/general_lm_head_recovery/state.py',
    'experiments/general_lm_head_recovery/export.py', 'experiments/general_lm_head_recovery/audit.py',
)


def export_checkpoint(checkpoint, output_dir, *, device="cuda:0"):
    from compressed_e2e_fintuning.mid_eval import temporary_inference_decode_mode
    from compressed_e2e_fintuning.runtime_v6 import _assert_final_runtime_clean, _build_v6_tokenizer, _finalize_decoders, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from litebsq.vae_linear import clear_model_vae_linear_cache
    from train_utils.checkpoint_v6 import _atomic_write_json, save_v6_full_checkpoint
    from train_utils.config.targets import collect_e2e_compressed_targets
    from train_utils.distill_precision import prepare_model_export
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    checkpoint, output = Path(checkpoint).resolve(), Path(output_dir).resolve()
    _require(not output.exists(), "Export output must be a new directory.")
    _require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Export requires one process.")
    meta, state, snapshot, source, base_meta, architecture = inspect_step(checkpoint)
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
    source_meta, source_budget, _ = checkpoint_inventory(source, architecture)
    _require(source_budget["budget_passed"], "Source fails the rank-zero 2.5-bpw budget.")
    files = [checkpoint / name for name in ("checkpoint_meta.json", "trainer_state.json", "training_model_state.pt")]
    files += [checkpoint.parent.parent / "normalized_e2e_runtime_args.json", source / "pytorch_model.bin", source / "checkpoint_meta.json"]
    audit = {"runtime": "experiments.general_lm_head_recovery.export", "source_checkpoint": str(checkpoint),
        "source_checkpoint_id": meta["checkpoint_id"], "round_base_dir": str(source),
        "round_base_checkpoint_id": source_meta["checkpoint_id"], "source_global_step": state["global_step"],
        "source_max_steps": state["max_steps"], "architecture": architecture, "train_mode": meta["train_mode"], "optimizer_steps_executed": 0,
        "trainer_or_main_optimizer_constructed": False, "teacher_or_training_data_loaded": False,
        "source_files_sha256": {str(path): _sha256(path) for path in files},
        "code_sha256": {name: _sha256(REPO / name) for name in CODE_FILES}}
    model, loaded_meta, _ = load_v6_model_checkpoint(str(source), strict=True, map_location="cpu")
    _require(loaded_meta["checkpoint_id"] == meta["round_base_checkpoint_id"], "Loaded round-base UUID changed.")
    inventory, _banks = persistent_inventory(model, architecture)
    selected = collect_e2e_compressed_targets(model, target_layers=selected_layers(meta["target_layers"], architecture),
                                             target_modules=selected_modules(meta["target_modules"], architecture), num_layers=architecture["num_hidden_layers"])
    names = selected_names(meta["target_layers"], meta["target_modules"], architecture)
    _require(tuple(name for name, _ in selected) == names and set(names) <= inventory.keys(),
             "Selected runtime topology disagrees with the authorized subset.")
    _require(all(len(inventory[name]) == 6 and len(_banks[name]) == 2 for name in names),
             "Subset recipe requires six persistent decoder tensors and two code banks per projection.")
    audit.update(selected_targets=list(names), selected_projection_count=len(names),
                 deployed_projection_count=len(inventory), selected_decoder_tensor_count=6 * len(names),
                 selected_projection_code_bank_count=2 * len(names),
                 selected_code_bank_count=2 * len(names) if meta["train_mode"] == "decoder_sparse_bit" else 0)
    audit['mutable_restore'] = restore_union_state(model, selected, checkpoint=checkpoint, snapshot=snapshot, architecture=architecture)
    audit['restored_decoder_tensors'] = audit['mutable_restore']['component_counts']['decoder']
    audit['head_shape'] = list(model.lm_head.weight.shape)
    audit['mutable_component_counts'] = audit['mutable_restore']['component_counts']
    tokenizer = _build_v6_tokenizer(str(base_meta["base_model_path"]), SimpleNamespace(**snapshot["hf_args"]))
    _sync_model_padding_config(model, tokenizer)
    model.config.use_cache = False
    audit["sparse_bit_restore"] = None
    audit["code_state_reference"] = "unchanged_original_round_base"
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
    selected_head = model.lm_head.weight.detach().cpu().clone()
    audit["export_dtype"] = str(prepare_model_export(model, training))
    assert_head_cast(model.lm_head.weight, selected_head)
    del selected_head
    audit['head_matches_selected_fp32_cast_exactly'] = True
    audit['head_fusion_performed'] = False
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
        train_mode=meta["train_mode"], norm_train_mode="none", lm_head_train_mode="full", lora_config=None,
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
