"""Audit decoder and fused linear-head finals, with all hard codes frozen."""

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
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from train_utils.checkpoint_v6 import _atomic_write_json
from train_utils.v6_model_loader import load_v6_model_checkpoint


HEAD_KEY = "lm_head.weight"


def audit_aux_state(original, actual, inventory, code_banks):
    """Audit disjoint head and decoder inventories without exempting other tensors."""
    _require(original.keys() == actual.keys() and HEAD_KEY in original, "Serialized tensor inventory changed.")
    before, after = original[HEAD_KEY], actual[HEAD_KEY]
    _require(before.ndim == after.ndim == 2 and before.shape == after.shape,
             "Fused head shape changed.")
    _require(before.dtype == after.dtype == torch.bfloat16, "The source and fused head must both be BF16.")
    _require(before.untyped_storage().nbytes() == after.untyped_storage().nbytes() == before.numel() * 2,
             "Fused head storage changed or contains excess backing storage.")
    decoder_keys = {key for keys in inventory.values() for key in keys}
    _require(HEAD_KEY not in decoder_keys and all(HEAD_KEY != key for rows in code_banks.values() for key, _ in rows),
             "Head overlaps the decoder/code inventory.")
    # The existing decoder audit sees the complete complement of the head. It
    # still checks every protected tensor, rotation, norm, embedding and code byte.
    report = _audit_decoder_export_state(
        {key: value for key, value in original.items() if key != HEAD_KEY},
        {key: value for key, value in actual.items() if key != HEAD_KEY},
        inventory, code_banks, train_mode="decoder")
    changed, square_delta = 0, 0.0
    for row in range(0, before.shape[0], 1024):
        left, right = before[row:row + 1024], after[row:row + 1024]
        _require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), "Non-finite fused head.")
        changed += int(torch.count_nonzero(left != right))
        square_delta += float((right.double() - left.double()).square().sum())
    _require(changed > 0, "No numerical head update survived fusion and BF16 export.")
    report.update(head_tensor_count=1, changed_head_elements=changed, head_delta_l2=square_delta ** 0.5,
                  head_storage_bytes=after.untyped_storage().nbytes(), head_shape=list(after.shape),
                  head_dtype="torch.bfloat16", non_decoder_bytes_unchanged=False,
                  non_decoder_non_code_bytes_unchanged=False, non_decoder_non_head_bytes_unchanged=True,
                  head_fused_into_original_weight=True)
    return report


def audit_checkpoint(source_dir, checkpoint_dir, output):
    source, checkpoint, output = (Path(path).resolve() for path in (source_dir, checkpoint_dir, output))
    _require(source != checkpoint and not output.exists(), "Use distinct source/final and a new audit output.")
    report = {"status": "failed", "source_dir": str(source), "checkpoint_dir": str(checkpoint),
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "CPU complete tensor/storage audit; forward fusion evidence is in export_record.json"}
    started = time.perf_counter()
    try:
        source_meta, source_budget = _validate_checkpoint(source)
        final_meta, final_budget = _validate_checkpoint(checkpoint)
        report.update(source_checkpoint_id=source_meta["checkpoint_id"], checkpoint_id=final_meta["checkpoint_id"],
                      source_bit_budget=source_budget, bit_budget=final_budget)
        _require(source_meta["checkpoint_id"] != final_meta["checkpoint_id"], "Final reused the source UUID.")
        _require(Path(source_meta["base_model_path"]).resolve() == Path(final_meta["base_model_path"]).resolve(),
                 "Final references a different base model.")
        _require(not source_meta.get("post_norm_head_linear") and not final_meta.get("post_norm_head_linear"),
                 "Deployment contains a post-norm head branch.")
        _require(final_meta["train_mode"] == "decoder" and final_meta["norm_train_mode"] == "none"
                 and final_meta["lm_head_train_mode"] == "linear", "Final metadata does not identify the requested trial.")
        _require({item["name"]: _inference_spec(item, export_decoder_dtype=True) for item in source_meta["converted_modules"]}
                 == {item["name"]: _inference_spec(item) for item in final_meta["converted_modules"]},
                 "Decoder/rotation/module inference specification changed.")
        original, actual = _load_state(source), _load_state(checkpoint)
        _require(tuple(original[HEAD_KEY].shape) == (151936, 4096), "Expected the original Qwen3-8B head shape.")
        source_model, loaded_source_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_source_meta["checkpoint_id"] == source_meta["checkpoint_id"], "Source UUID changed during load.")
        inventory, banks = persistent_inventory(source_model)
        report["source_reloaded_tensors_compared"] = _assert_state_equal(source_model.state_dict(), original, "Source strict load")
        del source_model
        gc.collect()
        report.update(audit_aux_state(original, actual, inventory, banks))
        _require(report["frozen_tensors_compared"] == 1910 and report["code_bank_count"] == 504,
                 "This source must retain 1910 frozen tensors and 504 unchanged code banks.")
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
    for name in ("source_dir", "checkpoint_dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit_checkpoint(args.source_dir, args.checkpoint_dir, args.output)


if __name__ == "__main__":
    main()
