"""Audit decoder or decoder-plus-packed-code E2E finals on CPU, without forward."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from experiments.qwen3_w25.assemble_categories import EXPECTED, _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.local_bits import _changed_bits, _code_banks
from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import _atomic_write_json
from train_utils.v6_model_loader import load_v6_model_checkpoint


def persistent_inventory(model):
    modules = {name: module for name, module in model.named_modules() if isinstance(module, VAELinear)}
    _require(set(modules) == set(EXPECTED), "Expected exactly all 252 Qwen3-8B VAELinear projections.")
    inventory, code_banks = {}, {}
    for name, module in modules.items():
        packed = getattr(module, "_parallel_stage_decoder", None)
        _require(isinstance(packed, nn.Module) and module.parallel_stage_decode
                 and module.residual_stages == 2 and module.parallel_parts == 1,
                 f"Expected the existing packed decoder: {name}.")
        keys = [name + "._parallel_stage_decoder." + key for key in packed.state_dict()]
        _require(bool(keys), f"Missing persistent decoder tensors: {name}.")
        inventory[name] = sorted(keys)
        code_banks[name] = [(name + "." + key, shape) for key, shape in _code_banks(module)]
    return inventory, code_banks


def _inference_spec(value, *, export_decoder_dtype=False, _decoder_spec=False):
    # Only the base decoder storage dtype and activation checkpointing may change
    # at export; rotation, packed-code and protected-channel metadata stay exact.
    if isinstance(value, dict):
        return {key: ("bfloat16" if export_decoder_dtype and _decoder_spec
                      and key == "param_dtype" and item == "float32" else
                      _inference_spec(item, export_decoder_dtype=export_decoder_dtype,
                                      _decoder_spec=_decoder_spec or key in ("decoders", "stage_decoders")))
                for key, item in value.items() if key != "use_checkpoint"}
    if isinstance(value, list):
        return [_inference_spec(item, export_decoder_dtype=export_decoder_dtype,
                                _decoder_spec=_decoder_spec) for item in value]
    return value


def _decoder_storage(state, keys):
    storages = {}
    for key in keys:
        tensor = state[key]
        storage = tensor.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes()) if storage.nbytes() else ("empty", key)
        storages[identity] = storage.nbytes() * 8
    bits = sum(storages.values())
    return {"budget_bits": bits, "storage_bits": bits, "unique_storages": len(storages),
            "dtypes": sorted({str(state[key].dtype).removeprefix("torch.") for key in keys})}


def _audit_decoder_export_state(original, actual, inventory, code_banks, *, train_mode):
    _require(train_mode in ("decoder", "decoder_sparse_bit"), "Invalid decoder audit train mode.")
    _require(original.keys() == actual.keys(), "E2E final changed the complete serialized tensor inventory.")
    decoder_keys = {key for keys in inventory.values() for key in keys}
    code_keys = {key for rows in code_banks.values() for key, _shape in rows}
    allowed = decoder_keys | (code_keys if train_mode == "decoder_sparse_bit" else set())
    _require(decoder_keys | code_keys <= original.keys(), "Actual decoder/code inventory is missing from serialized source state.")
    for key in original:
        _require(original[key].shape == actual[key].shape, f"E2E final changed tensor shape: {key}.")
        if key in decoder_keys:
            _require(original[key].dtype in (torch.float32, torch.bfloat16)
                     and actual[key].dtype == torch.bfloat16,
                     f"Expected FP32/BF16 source decoder and BF16 exported decoder: {key}.")
        else:
            _require(original[key].dtype == actual[key].dtype, f"E2E final changed non-decoder tensor dtype: {key}.")
    frozen = {key: value for key, value in original.items() if key not in allowed}
    frozen_count = _assert_state_equal({key: actual[key] for key in frozen}, frozen,
                                      "Every tensor outside the requested decoder/code update inventory")
    per_module, changed_keys, total_changed_elements = {}, [], 0
    for name, keys in inventory.items():
        changed, elements, square_delta = [], 0, 0.0
        for key in keys:
            before, after = original[key], actual[key]
            _require(bool(torch.isfinite(before).all()) and bool(torch.isfinite(after).all()),
                     f"Non-finite decoder tensor: {key}.")
            reference = before.to(dtype=torch.bfloat16)
            same = torch.equal(reference.contiguous().reshape(-1).view(torch.uint8),
                               after.contiguous().reshape(-1).view(torch.uint8))
            if not same:
                changed.append(key)
                elements += int(torch.count_nonzero(after != reference))
                square_delta += float((after.double() - reference.double()).square().sum())
        per_module[name] = {"decoder_tensor_count": len(keys), "changed_tensor_count": len(changed),
                            "changed_elements": elements, "delta_l2": square_delta ** 0.5,
                            "source_dtypes": sorted({str(original[key].dtype) for key in keys}),
                            "target_dtype": "torch.bfloat16"}
        changed_keys.extend(changed)
        total_changed_elements += elements
    _require(total_changed_elements > 0, "No actual numerical decoder update survived BF16 export.")
    stage_changed_bits = {name: [
        _changed_bits(original[key], actual[key], shape) if train_mode == "decoder_sparse_bit" else 0
        for key, shape in rows] for name, rows in code_banks.items()}
    source_storage, target_storage = (_decoder_storage(state, decoder_keys) for state in (original, actual))
    saved_bits = source_storage["storage_bits"] - target_storage["storage_bits"]
    converted_storage = _decoder_storage(original, {key for key in decoder_keys if original[key].dtype == torch.float32})
    _require(source_storage["unique_storages"] == target_storage["unique_storages"],
             "E2E final changed decoder storage sharing.")
    _require(saved_bits == converted_storage["storage_bits"] // 2,
             "Decoder storage decrease does not match FP32-to-BF16 conversion.")
    converted = sum(original[key].dtype != actual[key].dtype for key in decoder_keys)
    changed_code_bits = sum(sum(values) for values in stage_changed_bits.values())
    return {"decoder_tensor_count": len(decoder_keys), "frozen_tensors_compared": frozen_count,
            "changed_decoder_tensor_count": len(changed_keys), "changed_decoder_elements": total_changed_elements,
            "changed_decoder_module_count": sum(row["changed_elements"] > 0 for row in per_module.values()),
            "changed_decoder_keys": changed_keys, "decoder_modules": per_module,
            "decoder_update_reference": "source_decoder_cast_to_bfloat16",
            "decoder_source_dtypes": sorted({str(original[key].dtype) for key in decoder_keys}),
            "decoder_target_dtype": "torch.bfloat16", "decoder_dtype_converted_tensors": converted,
            "decoder_storage_source": source_storage, "decoder_storage_target": target_storage,
            "decoder_storage_source_bits": source_storage["storage_bits"],
            "decoder_storage_target_bits": target_storage["storage_bits"],
            "decoder_storage_saved_bits": saved_bits, "decoder_storage_saved_bytes": saved_bits // 8,
            "code_bank_count": len(code_keys), "changed_code_bits": changed_code_bits,
            "stage_changed_bits": stage_changed_bits, "has_committed_code_change": changed_code_bits > 0,
            "packed_padding_unchanged": True, "packed_codes_unchanged": changed_code_bits == 0,
            "non_decoder_bytes_unchanged": changed_code_bits == 0,
            "non_decoder_non_code_bytes_unchanged": True, "frozen_bytes_unchanged": True,
            "tensor_schema_unchanged": converted == 0,
            "tensor_schema_unchanged_except_decoder_dtype": True,
            "actual_tensor_storage_unchanged": saved_bits == 0}


def _validate_export_storage(source_budget, final_budget, state_report):
    _require(source_budget["budget_passed"] and final_budget["budget_passed"], "Source/final violates rank-zero 2.5 bpw.")
    for key in ("original_linear_weights", "code_padding_bits", "outside_linear_bits"):
        _require(source_budget[key] == final_budget[key], f"E2E final changed frozen tensor storage: {key}.")
    for budget, label in ((source_budget, "source"), (final_budget, "target")):
        _require(budget["components"]["decoder"] == state_report["decoder_storage_" + label],
                 f"Decoder {label} component disagrees with independently counted tensor storage.")
    _require({key: value for key, value in source_budget["components"].items() if key != "decoder"}
             == {key: value for key, value in final_budget["components"].items() if key != "decoder"},
             "E2E final changed non-decoder storage components.")
    saved_bits = state_report["decoder_storage_saved_bits"]
    for key in ("user_linear_bits", "packed_linear_bits"):
        _require(source_budget[key] - final_budget[key] == saved_bits,
                 f"E2E {key} decrease does not match decoder conversion.")
    _require(source_budget["full_model_tensor_bytes"] - final_budget["full_model_tensor_bytes"] == saved_bits // 8,
             "E2E full-model storage decrease does not match decoder conversion.")


def audit_e2e_decoder(source_dir, checkpoint_dir, output, *, train_mode="decoder"):
    source, checkpoint, output = (Path(path).resolve() for path in (source_dir, checkpoint_dir, output))
    _require(source != checkpoint, "Source and recovered checkpoint must be distinct.")
    _require(not output.exists(), "Audit output must be a new file.")
    _require(train_mode in ("decoder", "decoder_sparse_bit"), "Select decoder or decoder_sparse_bit auditing.")
    report = {"status": "failed", "source_dir": str(source), "checkpoint_dir": str(checkpoint),
        "train_mode": train_mode, "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "CPU tensor/state/storage audit; no forward, accuracy evaluation or exact-resume test"}
    started = time.perf_counter()
    try:
        # _validate_checkpoint reuses audit_checkpoint(max_bpw=2.5, max_lora_rank=0).
        source_meta, source_budget = _validate_checkpoint(source)
        final_meta, final_budget = _validate_checkpoint(checkpoint)
        report.update(source_checkpoint_id=source_meta["checkpoint_id"], checkpoint_id=final_meta["checkpoint_id"],
                      source_bit_budget=source_budget, bit_budget=final_budget)
        _require(source_meta["checkpoint_id"] != final_meta["checkpoint_id"], "E2E final reused the source UUID.")
        _require(Path(source_meta["base_model_path"]).resolve() == Path(final_meta["base_model_path"]).resolve(),
                 "E2E final references a different base model.")
        source_specs = {item["name"]: _inference_spec(item, export_decoder_dtype=True) for item in source_meta["converted_modules"]}
        final_specs = {item["name"]: _inference_spec(item) for item in final_meta["converted_modules"]}
        _require(source_specs == final_specs, "E2E final changed the decoder/rotation/module inference specification.")
        original, actual = _load_state(source), _load_state(checkpoint)
        _require(original.keys() == actual.keys(), "E2E final changed the complete serialized tensor inventory.")

        # Keep CUDA visible for the existing FA2-aware loader, but map both
        # independent model instances to CPU and never execute their forward.
        source_model, loaded_source_meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_source_meta["checkpoint_id"] == source_meta["checkpoint_id"], "Source UUID changed during load.")
        inventory, code_banks = persistent_inventory(source_model)
        report["source_reloaded_tensors_compared"] = _assert_state_equal(
            source_model.state_dict(), original, "Independent source strict load")
        del source_model
        gc.collect()
        report.update(_audit_decoder_export_state(original, actual, inventory, code_banks, train_mode=train_mode))
        _validate_export_storage(source_budget, final_budget, report)

        reloaded, loaded_meta, _ = load_v6_model_checkpoint(str(checkpoint), map_location="cpu",
            strict=True, expected_kind="final_model")
        _require(loaded_meta["checkpoint_id"] == final_meta["checkpoint_id"], "Final UUID changed during fresh load.")
        _require(persistent_inventory(reloaded) == (inventory, code_banks),
                 "Fresh final changed the actual decoder/code inventory.")
        compared = _assert_state_equal(reloaded.state_dict(), actual, "Independent final strict load versus every saved tensor")
        del reloaded
        gc.collect()
        report.update(status="verified", projection_count=len(inventory),
            serialized_tensors_compared=len(actual), reloaded_tensors_compared=compared,
            rank_zero=True, native_decoder_dtype="torch.bfloat16", fresh_strict_reload_state_exact=True,
            inference_specs_unchanged_except_activation_checkpointing=report["decoder_dtype_converted_tensors"] == 0,
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
    parser.add_argument("--train_mode", choices=("decoder", "decoder_sparse_bit"), default="decoder")
    args = parser.parse_args()
    report = audit_e2e_decoder(args.source_dir, args.checkpoint_dir, args.output, train_mode=args.train_mode)
    print(json.dumps({key: report[key] for key in ("status", "source_checkpoint_id", "checkpoint_id",
        "changed_decoder_tensor_count", "changed_decoder_module_count", "changed_code_bits", "fresh_strict_reload_state_exact")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
