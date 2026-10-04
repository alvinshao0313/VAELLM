"""Audit the selected native projection recipe against the real Qwen/Llama inventory.

Only the old live Qwen audit's architecture boundary is isolated here. Its pure
state/packed-code/BF16 projection and storage comparisons are imported unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import time

import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec, _validate_export_storage
from experiments.qwen3_w25.e2e_subset_audit import audit_subset_state
from experiments.w25_eval.subset_state import checkpoint_inventory, inspect_step, persistent_inventory, selected_layers, selected_modules, selected_names


def audit_subset_training_state(original, actual, inventory, banks, *, names,
                                mutable, packed_snapshot, train_mode):
    """Reuse the numeric subset audit, with original codes as the decoder-only reference."""
    _require(train_mode in ('decoder', 'decoder_sparse_bit'), 'Unsupported subset training mode.')
    sparse = train_mode == 'decoder_sparse_bit'
    selected_codes = {key for name in names for key, _ in banks[name]}
    if not sparse:
        _require(packed_snapshot is None, 'Decoder-only audit must not supply a mutable-code snapshot.')
        packed_snapshot = {key: original[key] for key in selected_codes}
    report = audit_subset_state(original, actual, inventory, banks, selected_names=names,
                               training_decoder_state=mutable, packed_snapshot=packed_snapshot)
    report['train_mode'] = train_mode
    report['selected_projection_code_bank_count'] = len(selected_codes)
    if not sparse:
        _require(report['changed_code_bits'] == 0, 'Decoder-only export changed hard codes.')
        # The reused audit compares the disjoint frozen complement and these two
        # banks against their original bytes, so all non-decoder tensors are covered.
        report['frozen_tensors_compared'] += len(selected_codes)
        report['frozen_code_bank_count'] = report['code_bank_count']
        report['selected_code_bank_count'] = 0
        report['selected_codes_match_current_snapshot_exactly'] = None
        report['selected_codes_match_frozen_source_exactly'] = True
        report['code_state_reference'] = 'unchanged_original_round_base'
    return report


def audit_e2e_subset(source_dir, checkpoint_dir, output, *, training_checkpoint_dir):
    from experiments.qwen3_w25.cpu_bit_export import restore_packed_bits_cpu
    from experiments.qwen3_w25.export_e2e_selected import _sha256
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
        source_meta, source_budget, architecture = checkpoint_inventory(source)
        report["architecture"] = architecture
        final_meta, final_budget, _ = checkpoint_inventory(checkpoint, architecture)
        step_meta, step_state, _snapshot, step_source, _base, step_architecture = inspect_step(training)
        _require(step_architecture == architecture, "Training/source architecture identity changed.")
        names = selected_names(step_meta["target_layers"], step_meta["target_modules"], architecture)
        _require(step_source.resolve() == source and step_meta["round_base_checkpoint_id"] == source_meta["checkpoint_id"],
                 "Saved training state references another source.")
        _require(final_meta["immutable_resume_contract"] == step_meta["immutable_resume_contract"]
                 and final_meta["target_layers"] == step_meta["target_layers"]
                 and final_meta["target_modules"] == step_meta["target_modules"], "Final/selected contracts differ.")
        _require(final_meta["checkpoint_id"] not in (source_meta["checkpoint_id"], step_meta["checkpoint_id"]),
                 "Final reused an input checkpoint UUID.")
        _require(Path(source_meta["base_model_path"]).resolve() == Path(final_meta["base_model_path"]).resolve(),
                 "Final references another base model.")
        _require(final_meta["train_mode"] == step_meta["train_mode"]
                 and final_meta["train_mode"] in ("decoder", "decoder_sparse_bit")
                 and final_meta["norm_train_mode"] == "none"
                 and final_meta["lm_head_train_mode"] == "none" and final_meta["lora_config"] is None
                 and not final_meta.get("post_norm_head_linear"), "Unexpected final trainable branch.")
        _require({item["name"]: _inference_spec(item) for item in source_meta["converted_modules"]}
                 == {item["name"]: _inference_spec(item) for item in final_meta["converted_modules"]},
                 "Deployment decoder/rotation/protection specification changed.")
        original, actual = _load_state(source), _load_state(checkpoint)
        _require(original.keys() == actual.keys(), "Complete source/export serialized tensor inventory differs.")
        # Existing FA2-aware loading needs CUDA visible even for this CPU-only audit.
        model, loaded_source_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_source_meta["checkpoint_id"] == source_meta["checkpoint_id"], "Source identity changed.")
        inventory, banks = persistent_inventory(model, architecture)
        report["source_reloaded_tensors_compared"] = _assert_state_equal(model.state_dict(), original, "Source strict load")
        selected = collect_e2e_compressed_targets(model, target_layers=selected_layers(step_meta["target_layers"], architecture),
            target_modules=selected_modules(step_meta["target_modules"], architecture), num_layers=architecture["num_hidden_layers"])
        _require({name for name, _ in selected} == set(names), "Actual selected modules disagree with metadata.")
        mutable, manifest = load_v6_training_model_state(str(training), map_location="cpu")
        expected = {key for name in names for key in inventory[name]}
        _require(len(manifest) == len(expected) == 6 * len(names)
                 and {row["name"]: row["component_class"] for row in manifest} == {key: "decoder" for key in expected}
                 and set(mutable) == expected, "Continuous mutable state is not exactly the selected decoder subset.")
        # Reuse the genuine current-sampler/sign/radius check; do not restore an optimizer or recalibrate scales.
        sparse = step_meta["train_mode"] == "decoder_sparse_bit"
        bit_restore, packed = None, None
        if sparse:
            bit_restore = restore_packed_bits_cpu(selected, checkpoint=training, contract=step_meta["immutable_resume_contract"])
            model_state = model.state_dict()
            packed = {key: model_state[key].detach().clone() for name in names for key, _ in banks[name]}
            del model_state
        del selected, model
        gc.collect()
        report.update(audit_subset_training_state(original, actual, inventory, banks, names=names,
            mutable=mutable, packed_snapshot=packed, train_mode=step_meta["train_mode"]))
        _require(report["frozen_tensors_compared"] == len(original) - (8 if sparse else 6) * len(names)
                 and report["code_bank_count"] == 2 * len(inventory) and report["decoder_tensor_count"] == 6 * len(inventory),
                 "Incomplete frozen/full-model inventory.")
        _validate_export_storage(source_budget, final_budget, report)
        reloaded, loaded_meta, _ = load_v6_model_checkpoint(str(checkpoint), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_meta["checkpoint_id"] == final_meta["checkpoint_id"], "Fresh final identity changed.")
        _require(type(reloaded.lm_head) is torch.nn.Linear and not reloaded.config.tie_word_embeddings,
                 "Final head differs from the original untied Linear topology.")
        _require(persistent_inventory(reloaded, architecture) == (inventory, banks), "Fresh final decoder/code inventory changed.")
        compared = _assert_state_equal(reloaded.state_dict(), actual, "Fresh strict load versus every saved tensor")
        report.update(status="verified", source_checkpoint_id=source_meta["checkpoint_id"],
            checkpoint_id=final_meta["checkpoint_id"], selected_checkpoint_id=step_meta["checkpoint_id"],
            selected_global_step=step_state["global_step"], selected_training_state_sha256=_sha256(training / "training_model_state.pt"),
            selected_exact_sidecar_sha256=_sha256(training / "sparse_bit_tuning/exact_state.pt") if sparse else None,
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
