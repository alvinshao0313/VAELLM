"""Audit a full deployment while only selected down decoders/codes may change."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import time

import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.audit_e2e_decoder import (
    _decoder_storage, _inference_spec, _validate_export_storage, persistent_inventory,
)
from experiments.qwen3_w25.local_bits import _changed_bits


def audit_subset_state(original, actual, inventory, code_banks, *, selected_names,
                       training_decoder_state, packed_snapshot):
    """Check selected FP32 casts and current hard codes, then the exact complement.

    Storage is counted over every decoder, as required by the complete bit budget.
    A real FP32 update is required; its BF16 projection may legitimately be unchanged.
    """
    selected = tuple(selected_names)
    _require(bool(selected) and len(set(selected)) == len(selected)
             and set(selected) <= inventory.keys() and inventory.keys() == code_banks.keys(),
             "Invalid selected decoder/code inventory.")
    _require(original.keys() == actual.keys(), "Complete serialized tensor inventory changed.")
    decoder_keys = {key for keys in inventory.values() for key in keys}
    code_keys = {key for banks in code_banks.values() for key, _ in banks}
    selected_decoder = {key for name in selected for key in inventory[name]}
    selected_codes = {key for name in selected for key, _ in code_banks[name]}
    _require(not decoder_keys & code_keys and decoder_keys | code_keys <= original.keys(),
             "Missing or overlapping decoder/code tensors.")
    _require(all(len(inventory[name]) == 6 and len(code_banks[name]) == 2 for name in selected),
             "Expected six packed decoder tensors and two banks per selected projection.")
    _require(set(training_decoder_state) == selected_decoder and set(packed_snapshot) == selected_codes,
             "Saved mutable decoder/current-code set differs from the selected subset.")
    for key, before in original.items():
        after = actual[key]
        _require(before.device.type == after.device.type == "cpu" and before.shape == after.shape,
                 f"Serialized tensor shape/device changed: {key}.")
        _require(before.dtype == after.dtype, f"Source/export storage dtype changed: {key}.")
        if key in decoder_keys:
            _require(before.dtype == torch.bfloat16 and bool(torch.isfinite(before).all())
                     and bool(torch.isfinite(after).all()), f"Expected finite BF16 deployment decoder: {key}.")
    allowed = selected_decoder | selected_codes
    frozen = {key: value for key, value in original.items() if key not in allowed}
    frozen_count = _assert_state_equal({key: actual[key] for key in frozen}, frozen,
                                      "All tensors outside the selected down decoder/code subset")
    fp32_changes, bf16_changes, per_module = {}, {}, {}
    for name in selected:
        for key in inventory[name]:
            value = training_decoder_state[key]
            _require(value.device.type == "cpu" and value.dtype == torch.float32
                     and value.shape == original[key].shape and bool(torch.isfinite(value).all()),
                     f"Expected the finite selected FP32 decoder tensor: {key}.")
            projected = value.bfloat16()
            _require(bool(torch.isfinite(projected).all()), f"BF16 projection overflowed: {key}.")
            _assert_state_equal({key: actual[key]}, {key: projected}, "Exact selected FP32-to-BF16 projection")
            fp32_changes[key] = int(torch.count_nonzero(value != original[key].float()))
            bf16_changes[key] = int(torch.count_nonzero(actual[key] != original[key]))
        per_module[name] = {
            "decoder_tensor_count": len(inventory[name]),
            "changed_fp32_elements": sum(fp32_changes[key] for key in inventory[name]),
            "changed_bf16_elements": sum(bf16_changes[key] for key in inventory[name]),
        }
    _require(sum(fp32_changes.values()) > 0, "No actual selected FP32 decoder update was saved.")
    _assert_state_equal({key: actual[key] for key in selected_codes}, packed_snapshot,
                        "Selected persistent hard codes versus the saved current-round snapshot")
    stage_changes = {}
    for name, banks in code_banks.items():
        # Frozen bank equality above also verifies every padding bit in those banks.
        stage_changes[name] = [
            _changed_bits(original[key], actual[key], shape) if key in selected_codes else 0
            for key, shape in banks]
    source_storage, target_storage = (_decoder_storage(state, decoder_keys) for state in (original, actual))
    _require(source_storage == target_storage, "Complete decoder storage accounting changed.")
    changed_bits = sum(sum(values) for values in stage_changes.values())
    return {
        "decoder_tensor_count": len(decoder_keys), "selected_decoder_tensor_count": len(selected_decoder),
        "selected_module_count": len(selected), "selected_modules": list(selected),
        "selected_decoder_fp32_changed_elements": sum(fp32_changes.values()),
        "selected_decoder_fp32_changed_elements_by_key": fp32_changes,
        "selected_decoder_matches_fp32_cast_exactly": True,
        "changed_decoder_tensor_count": sum(value > 0 for value in bf16_changes.values()),
        "changed_decoder_elements": sum(bf16_changes.values()),
        "changed_decoder_module_count": sum(row["changed_bf16_elements"] > 0 for row in per_module.values()),
        "selected_decoder_update_survived_bf16": sum(bf16_changes.values()) > 0,
        "decoder_modules": per_module, "decoder_update_reference": "original_BF16_source",
        "decoder_source_dtypes": ["torch.bfloat16"], "decoder_target_dtype": "torch.bfloat16",
        "decoder_dtype_converted_tensors": 0,
        "decoder_storage_source": source_storage, "decoder_storage_target": target_storage,
        "decoder_storage_source_bits": source_storage["storage_bits"],
        "decoder_storage_target_bits": target_storage["storage_bits"],
        "decoder_storage_saved_bits": 0, "decoder_storage_saved_bytes": 0,
        "code_bank_count": len(code_keys), "selected_code_bank_count": len(selected_codes),
        "frozen_code_bank_count": len(code_keys - selected_codes), "changed_code_bits": changed_bits,
        "stage_changed_bits": stage_changes, "has_committed_code_change": changed_bits > 0,
        "selected_codes_match_current_snapshot_exactly": True, "packed_padding_unchanged": True,
        "packed_codes_unchanged": changed_bits == 0, "frozen_tensors_compared": frozen_count,
        "frozen_bytes_unchanged": True, "unselected_decoder_bytes_unchanged": True,
        "unselected_code_bytes_unchanged": True, "non_decoder_non_code_bytes_unchanged": True,
        "tensor_schema_unchanged": True, "actual_tensor_storage_unchanged": True,
    }


def audit_e2e_subset(source_dir, checkpoint_dir, output, *, training_checkpoint_dir):
    from experiments.qwen3_w25.cpu_bit_export import restore_packed_bits_cpu
    from experiments.qwen3_w25.e2e_subset_export import inspect_step, selected_names
    from experiments.qwen3_w25.export_e2e_selected import _sha256
    from experiments.qwen3_w25.summarize_results import _validate_checkpoint
    from train_utils.checkpoint_v6 import _atomic_write_json, load_v6_training_model_state
    from train_utils.config.targets import collect_e2e_compressed_targets
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    source, checkpoint, output, training = (Path(path).resolve() for path in
        (source_dir, checkpoint_dir, output, training_checkpoint_dir))
    _require(source != checkpoint and not output.exists(), "Use distinct source/final and a new audit output.")
    report = {"status": "failed", "source_dir": str(source), "checkpoint_dir": str(checkpoint),
              "training_checkpoint_dir": str(training),
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "CPU full deployment/state/storage and saved selected decoder/current-code audit; no forward or exact-resume"}
    started = time.perf_counter()
    try:
        source_meta, source_budget = _validate_checkpoint(source)
        final_meta, final_budget = _validate_checkpoint(checkpoint)
        step_meta, step_state, _snapshot, step_source, _base = inspect_step(training)
        names = selected_names(step_meta["target_layers"], step_meta["target_modules"])
        _require(step_source.resolve() == source and step_meta["round_base_checkpoint_id"] == source_meta["checkpoint_id"],
                 "Saved training state references another source.")
        _require(final_meta["immutable_resume_contract"] == step_meta["immutable_resume_contract"]
                 and final_meta["target_layers"] == step_meta["target_layers"]
                 and final_meta["target_modules"] == step_meta["target_modules"], "Final/selected contracts differ.")
        _require(final_meta["checkpoint_id"] not in (source_meta["checkpoint_id"], step_meta["checkpoint_id"]),
                 "Final reused an input checkpoint UUID.")
        _require(Path(source_meta["base_model_path"]).resolve() == Path(final_meta["base_model_path"]).resolve(),
                 "Final references another base model.")
        _require(final_meta["train_mode"] == "decoder_sparse_bit" and final_meta["norm_train_mode"] == "none"
                 and final_meta["lm_head_train_mode"] == "none" and final_meta["lora_config"] is None
                 and not final_meta.get("post_norm_head_linear"), "Unexpected final trainable branch.")
        _require({item["name"]: _inference_spec(item) for item in source_meta["converted_modules"]}
                 == {item["name"]: _inference_spec(item) for item in final_meta["converted_modules"]},
                 "Deployment decoder/rotation/protection specification changed.")
        original, actual = _load_state(source), _load_state(checkpoint)
        _require(len(original) == len(actual) == 3423, "Expected the complete3423 tensor inventory.")
        # Existing FA2-aware loading needs CUDA visible even for this CPU-only audit.
        model, loaded_source_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_source_meta["checkpoint_id"] == source_meta["checkpoint_id"], "Source identity changed.")
        inventory, banks = persistent_inventory(model)
        report["source_reloaded_tensors_compared"] = _assert_state_equal(model.state_dict(), original, "Source strict load")
        selected = collect_e2e_compressed_targets(model, target_layers=tuple(step_meta["target_layers"]),
            target_modules=tuple(step_meta["target_modules"]), num_layers=36)
        _require({name for name, _ in selected} == set(names), "Actual selected modules disagree with metadata.")
        mutable, manifest = load_v6_training_model_state(str(training), map_location="cpu")
        expected = {key for name in names for key in inventory[name]}
        _require(len(manifest) == len(expected) == 6 * len(names)
                 and {row["name"]: row["component_class"] for row in manifest} == {key: "decoder" for key in expected}
                 and set(mutable) == expected, "Continuous mutable state is not exactly the selected decoder subset.")
        # Reuse the genuine current-sampler/sign/radius check; do not restore an optimizer or recalibrate scales.
        bit_restore = restore_packed_bits_cpu(selected, checkpoint=training, contract=step_meta["immutable_resume_contract"])
        model_state = model.state_dict()
        packed = {key: model_state[key].detach().clone() for name in names for key, _ in banks[name]}
        del model_state, selected, model
        gc.collect()
        report.update(audit_subset_state(original, actual, inventory, banks, selected_names=names,
            training_decoder_state=mutable, packed_snapshot=packed))
        _require(report["frozen_tensors_compared"] == 3423 - 8 * len(names)
                 and report["code_bank_count"] == 504 and report["decoder_tensor_count"] == 1512,
                 "Incomplete frozen/full-model inventory.")
        _validate_export_storage(source_budget, final_budget, report)
        reloaded, loaded_meta, _ = load_v6_model_checkpoint(str(checkpoint), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_meta["checkpoint_id"] == final_meta["checkpoint_id"], "Fresh final identity changed.")
        _require(type(reloaded.lm_head) is torch.nn.Linear and not reloaded.config.tie_word_embeddings,
                 "Final head differs from the original untied Linear topology.")
        _require(persistent_inventory(reloaded) == (inventory, banks), "Fresh final decoder/code inventory changed.")
        compared = _assert_state_equal(reloaded.state_dict(), actual, "Fresh strict load versus every saved tensor")
        report.update(status="verified", source_checkpoint_id=source_meta["checkpoint_id"],
            checkpoint_id=final_meta["checkpoint_id"], selected_checkpoint_id=step_meta["checkpoint_id"],
            selected_global_step=step_state["global_step"], selected_training_state_sha256=_sha256(training / "training_model_state.pt"),
            selected_exact_sidecar_sha256=_sha256(training / "sparse_bit_tuning/exact_state.pt"),
            sparse_bit_restore=bit_restore, source_bit_budget=source_budget, bit_budget=final_budget,
            projection_count=len(inventory), serialized_tensors_compared=len(actual), reloaded_tensors_compared=compared,
            fresh_strict_reload_state_exact=True, rank_zero=True, native_decoder_dtype="torch.bfloat16",
            inference_specs_unchanged_except_activation_checkpointing=True)
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _atomic_write_json(report, str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_dir", "checkpoint_dir", "output", "training_checkpoint_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit_e2e_subset(args.source_dir, args.checkpoint_dir, args.output,
                     training_checkpoint_dir=args.training_checkpoint_dir)


if __name__ == "__main__":
    main()
