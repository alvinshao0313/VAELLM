"""Audit decoder/final-norm finals against the actual selected FP32 state."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import time

import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.audit_e2e_decoder import (
    _audit_decoder_export_state, _inference_spec, _validate_export_storage, persistent_inventory,
)
from experiments.qwen3_w25.e2e_norm_export import NORM_KEY, inspect_step
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.checkpoint_v6 import _atomic_write_json, load_v6_training_model_state
from train_utils.model_level_trainables import iter_named_backbone_norm_modules
from train_utils.v6_model_loader import load_v6_model_checkpoint


def audit_norm_state(original, actual, inventory, code_banks, selected_norm):
    """A real FP32 update may round away in BF16; report both independently."""
    _require(original.keys() == actual.keys() and NORM_KEY in original, "Serialized tensor inventory changed.")
    before, after = original[NORM_KEY], actual[NORM_KEY]
    _require(before.ndim == after.ndim == selected_norm.ndim == 1
             and before.shape == after.shape == selected_norm.shape, "Final-norm shape changed.")
    _require(before.dtype == after.dtype == torch.bfloat16 and selected_norm.dtype == torch.float32,
             "Final norm requires BF16 source/export and a selected FP32 training tensor.")
    _require(before.untyped_storage().nbytes() == after.untyped_storage().nbytes() == before.numel() * 2,
             "Final-norm storage changed or contains excess backing storage.")
    _require(all(bool(torch.isfinite(tensor).all()) for tensor in (before, after, selected_norm)),
             "Non-finite final norm.")
    _assert_state_equal({NORM_KEY: after}, {NORM_KEY: selected_norm.bfloat16()},
                        "Exported final norm versus exact native selected FP32-to-BF16 cast")
    changed_fp32 = int(torch.count_nonzero(selected_norm != before.float()))
    _require(changed_fp32 > 0, "No actual FP32 final-norm update in selected training state.")
    changed_bf16 = int(torch.count_nonzero(after != before))
    decoder_keys = {key for keys in inventory.values() for key in keys}
    _require(NORM_KEY not in decoder_keys and all(NORM_KEY != key for rows in code_banks.values() for key, _ in rows),
             "Final norm overlaps the decoder/code inventory.")
    # Audit the complete complement; head, all other norms, rotations and codes
    # remain byte-exact. The final norm is checked independently against its save.
    report = _audit_decoder_export_state(
        {key: value for key, value in original.items() if key != NORM_KEY},
        {key: value for key, value in actual.items() if key != NORM_KEY},
        inventory, code_banks, train_mode="decoder")
    report.update(norm_tensor_count=1, changed_norm_fp32_elements=changed_fp32,
                  changed_norm_bf16_elements=changed_bf16, norm_update_survived_bf16=changed_bf16 > 0,
                  norm_fp32_delta_l2=float((selected_norm.double() - before.double()).norm()),
                  norm_bf16_delta_l2=float((after.double() - before.double()).norm()),
                  norm_storage_bytes=after.untyped_storage().nbytes(), norm_shape=list(after.shape),
                  norm_dtype="torch.bfloat16", norm_matches_selected_fp32_cast_exactly=True,
                  norm_selected_fp32_sha256=hashlib.sha256(selected_norm.contiguous().numpy().tobytes()).hexdigest(),
                  non_decoder_bytes_unchanged=changed_bf16 == 0,
                  non_decoder_non_code_bytes_unchanged=changed_bf16 == 0,
                  non_decoder_non_final_norm_bytes_unchanged=True, original_head_bytes_unchanged=True)
    return report


def audit_checkpoint(source_dir, checkpoint_dir, output, *, training_checkpoint_dir):
    source, checkpoint, output, selected = (Path(path).resolve() for path in
        (source_dir, checkpoint_dir, output, training_checkpoint_dir))
    _require(source != checkpoint and not output.exists(), "Use distinct source/final and a new audit output.")
    report = {"status": "failed", "source_dir": str(source), "checkpoint_dir": str(checkpoint),
              "training_checkpoint_dir": str(selected),
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "CPU complete tensor/storage audit and selected FP32 final-norm projection"}
    started = time.perf_counter()
    try:
        source_meta, source_budget = _validate_checkpoint(source)
        final_meta, final_budget = _validate_checkpoint(checkpoint)
        step_meta, step_state, _snapshot, step_source, _base = inspect_step(selected)
        _require(step_source == source and step_meta["round_base_checkpoint_id"] == source_meta["checkpoint_id"],
                 "Selected training state references a different source.")
        _require(final_meta["immutable_resume_contract"] == step_meta["immutable_resume_contract"],
                 "Selected training state/final immutable contracts differ.")
        report.update(source_checkpoint_id=source_meta["checkpoint_id"], checkpoint_id=final_meta["checkpoint_id"],
                      selected_checkpoint_id=step_meta["checkpoint_id"], selected_global_step=step_state["global_step"],
                      selected_training_state_sha256=_sha256(selected / "training_model_state.pt"),
                      source_bit_budget=source_budget, bit_budget=final_budget)
        _require(source_meta["checkpoint_id"] != final_meta["checkpoint_id"], "Final reused the source UUID.")
        _require(Path(source_meta["base_model_path"]).resolve() == Path(final_meta["base_model_path"]).resolve(),
                 "Final references a different base model.")
        _require(not source_meta.get("post_norm_head_linear") and not final_meta.get("post_norm_head_linear"),
                 "Deployment contains a post-norm head branch.")
        _require(final_meta["train_mode"] == "decoder" and final_meta["norm_train_mode"] == "final"
                 and final_meta["lm_head_train_mode"] == "none", "Final metadata does not identify the requested trial.")
        _require({item["name"]: _inference_spec(item, export_decoder_dtype=True) for item in source_meta["converted_modules"]}
                 == {item["name"]: _inference_spec(item) for item in final_meta["converted_modules"]},
                 "Decoder/rotation/module inference specification changed.")
        original, actual = _load_state(source), _load_state(checkpoint)
        _require(tuple(original[NORM_KEY].shape) == (4096,), "Expected the original Qwen3-8B final norm.")
        source_model, loaded_source_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_source_meta["checkpoint_id"] == source_meta["checkpoint_id"], "Source UUID changed during load.")
        inventory, banks = persistent_inventory(source_model)
        other_norm_keys = {name + "." + key for name, module in iter_named_backbone_norm_modules(source_model)
                           for key in module.state_dict() if name + "." + key != NORM_KEY}
        _require(len(other_norm_keys) == 144, "Expected all 144 frozen backbone norms.")
        report["source_reloaded_tensors_compared"] = _assert_state_equal(source_model.state_dict(), original, "Source strict load")
        del source_model
        gc.collect()
        mutable, manifest = load_v6_training_model_state(str(selected), map_location="cpu")
        expected_classes = {key: "decoder" for keys in inventory.values() for key in keys}
        expected_classes[NORM_KEY] = "norm"
        _require({row["name"]: row["component_class"] for row in manifest} == expected_classes
                 and set(mutable) == set(expected_classes) and len(mutable) == 1513,
                 "Selected mutable state must contain exactly 1512 decoders and model.norm.weight.")
        _require(all(tensor.dtype == torch.float32 for tensor in mutable.values()), "Selected mutable state must be FP32.")
        report.update(audit_norm_state(original, actual, inventory, banks, mutable[NORM_KEY]))
        report["frozen_backbone_norm_tensors_compared"] = _assert_state_equal(
            {key: original[key] for key in other_norm_keys}, {key: actual[key] for key in other_norm_keys},
            "All other backbone norms frozen")
        _require(report["frozen_tensors_compared"] == 1910 and report["code_bank_count"] == 504,
                 "This source must retain 1910 frozen tensors, including 504 unchanged code banks.")
        _validate_export_storage(source_budget, final_budget, report)
        reloaded, loaded_meta, _ = load_v6_model_checkpoint(str(checkpoint), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_meta["checkpoint_id"] == final_meta["checkpoint_id"], "Final UUID changed during fresh load.")
        _require(type(reloaded.lm_head) is torch.nn.Linear and not reloaded.config.tie_word_embeddings,
                 "Final head is not the original untied Linear topology.")
        _require(persistent_inventory(reloaded) == (inventory, banks), "Fresh final changed decoder/code inventory.")
        compared = _assert_state_equal(reloaded.state_dict(), actual, "Fresh strict load versus every saved tensor")
        report.update(status="verified", projection_count=len(inventory), serialized_tensors_compared=len(actual),
                      reloaded_tensors_compared=compared, fresh_strict_reload_state_exact=True, rank_zero=True,
                      inference_specs_unchanged_except_activation_checkpointing_and_decoder_dtype=True)
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
    audit_checkpoint(args.source_dir, args.checkpoint_dir, args.output,
                     training_checkpoint_dir=args.training_checkpoint_dir)


if __name__ == "__main__":
    main()
