"""Audit the whole frozen base, selected LoRA factors and actual deployment bytes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import time

import torch

from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from experiments.qwen3_w25.export_e2e_selected import _read_json, _sha256
from experiments.w25_eval.lora_state import audit_lora_state, inspect_step, validate_source, lora_storage, saved_lora_selection
from experiments.w25_eval.model_inventory import model_inventory
from experiments.qwen3_w25.local_bits import _code_banks
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import _atomic_write_json, load_v6_training_model_state, validate_v6_meta
from train_utils.v6_model_loader import load_v6_model_checkpoint


def load_state(directory):
    # The old pure-VAE helper rejects low_rank_* by design; native LoRA needs the
    # same genuine mmap load without that experiment-specific prohibition.
    value = torch.load(Path(directory) / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    _require(isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()),
             "Expected the complete native tensor state dictionary.")
    return value


def validate_deployment(meta, state, architecture=None):
    architecture = architecture or model_inventory(meta["base_model_path"])
    shapes = architecture["projection_shapes"]
    validate_v6_meta(meta, expected_kind="final_model")
    selected_shapes, lora = saved_lora_selection(meta, architecture)
    rank = lora["rank"]
    _require(Path(meta["base_model_path"]).resolve() == Path(architecture["base_model_path"]), "Wrong model architecture.")
    _require(meta["train_mode"] == "lora" and meta["norm_train_mode"] == meta["lm_head_train_mode"] == "none"
             and meta["lora_config"] is None and not meta.get("post_norm_head_linear")
             and (meta.get("extra_meta") or {}).get("residual_lora") is None, "Unexpected deployed auxiliary topology.")
    _require(set(meta["compressed_targets"]) == set(shapes) and not meta["pending_dense_targets"]
             and not meta["skip_targets"] and len(meta["converted_modules"]) == len(shapes)
             and {s["name"] for s in meta["converted_modules"]} == set(shapes), "Incomplete compressed model.")
    expected_factors = {name + "." + key for name in selected_shapes for key in ("low_rank_a", "low_rank_b")}
    _require({key for key in state if key.endswith((".low_rank_a", ".low_rank_b"))} == expected_factors,
             "Missing or extra deployment LoRA factors.")
    for spec in meta["converted_modules"]:
        name = spec["name"]
        _require(name in shapes and [spec["out_features"], spec["in_features"]] == list(shapes[name])
                 and not spec.get("has_original_weight") and not spec.get("always_use_original"), "Invalid projection.")
        out, inp = shapes[name]
        if name not in selected_shapes:
            _require(spec.get("low_rank_a") is None and spec.get("low_rank_b") is None,
                     "An unselected projection acquired a LoRA branch.")
            continue
        for key, shape in (("low_rank_a", [out, rank]), ("low_rank_b", [rank, inp])):
            tensor = state[name + "." + key]
            _require(spec.get(key) is not None and spec[key]["shape"] == shape
                     and list(tensor.shape) == shape and tensor.dtype == torch.bfloat16,
                     "Native factors must match the saved rank in BF16.")
    budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=rank)
    _require(budget["budget_passed"] and budget["original_linear_weights"] == architecture["original_linear_weights"]
             and budget["components"]["lora"]["storage_bits"] == lora_storage(selected_shapes, rank)["storage_bits"]
             and not any(budget["components"].get(k, {}).get("storage_bits", 0)
                         for k in ("dense_weights", "residual_lora")), "Invalid actual selected-LoRA deployment budget.")
    return budget


def audit_checkpoint(source_dir, checkpoint_dir, output, *, training_checkpoint_dir):
    source, final, output, selected = map(lambda p: Path(p).resolve(),
        (source_dir, checkpoint_dir, output, training_checkpoint_dir))
    _require(source != final and not output.exists(), "Use distinct source/final and a new audit report.")
    started = time.perf_counter()
    report = {"status": "failed", "source_dir": str(source), "checkpoint_dir": str(final),
              "training_checkpoint_dir": str(selected), "audit_script_sha256": _sha256(Path(__file__)),
              "scope": "CPU full state, exact native factor projection and fresh strict reload; no forward or optimizer."}
    try:
        step_meta, step_state, _snapshot, actual_source, source_meta, architecture = inspect_step(selected)
        shapes = architecture["projection_shapes"]
        selected_shapes, lora = saved_lora_selection(step_meta, architecture)
        storage = lora_storage(selected_shapes, lora["rank"])
        lora_bits = storage["storage_bits"]
        _require(actual_source.resolve() == source, "Wrong selected step source.")
        validate_source(source_meta, architecture)
        final_meta = _read_json(final / "checkpoint_meta.json")
        original, actual = load_state(source), load_state(final)
        _require(len(actual) == len(original) + storage["tensors"], "Wrong full native state count.")
        frozen_norm_count = sum("norm" in key and not any(key.startswith(n + ".") for n in shapes) for key in original)
        source_budget = audit_state_dict(source_meta, original, max_bpw=2.5, max_lora_rank=0)
        final_budget = validate_deployment(final_meta, actual, architecture)
        _require(saved_lora_selection(final_meta, architecture) == (selected_shapes, lora),
                 "Deployment LoRA selection differs from the actual native step.")
        _require(final_meta["base_model_path"] == source_meta["base_model_path"], "Base model changed.")
        stripped = deepcopy(final_meta["converted_modules"])
        for spec in stripped:
            spec["low_rank_a"] = spec["low_rank_b"] = None
        _require(_inference_spec(stripped) == _inference_spec(source_meta["converted_modules"]),
                 "A non-LoRA projection specification changed.")
        for key in ("shared_protected_residual_decoders", "post_norm_head_linear"):
            _require(final_meta.get(key) == source_meta.get(key), f"Source topology changed: {key}.")
        mutable, manifest = load_v6_training_model_state(str(selected), map_location="cpu")
        report.update(audit_lora_state(original, actual, mutable, manifest, selected_shapes,
                                      rank=lora["rank"], alpha=lora["alpha"]))
        _require(final_budget["user_linear_bits"] - source_budget["user_linear_bits"] == lora_bits
                 and final_budget["full_model_tensor_bytes"] - source_budget["full_model_tensor_bytes"] == lora_bits // 8,
                 "Measured extra storage is not exactly the selected LoRA factors.")
        for component, before in source_budget["components"].items():
            if component != "lora":
                _require(final_budget["components"].get(component) == before, f"Frozen component storage changed: {component}.")
        del mutable, manifest, original
        gc.collect()
        reloaded, loaded_meta, _ = load_v6_model_checkpoint(str(final), map_location="cpu", strict=True, expected_kind="final_model")
        _require(loaded_meta["checkpoint_id"] == final_meta["checkpoint_id"], "Fresh UUID differs.")
        modules = {name: module for name, module in reloaded.named_modules() if isinstance(module, VAELinear)}
        _require(set(modules) == set(shapes), "Fresh projection inventory differs.")
        decoder_keys = {name + "." + key for name, module in modules.items() for key in module.state_dict()
                        if "decoder" in key}
        code_keys = {name + "." + key for name, module in modules.items() for key, _shape in _code_banks(module)}
        _require(decoder_keys | code_keys <= actual.keys(), "Frozen decoder/code inventory missing from state.")
        compared = _assert_state_equal(reloaded.state_dict(), actual, "Every fresh strict deployment tensor")
        report.update(status="verified", source_checkpoint_id=source_meta["checkpoint_id"], checkpoint_id=final_meta["checkpoint_id"],
            selected_checkpoint_id=step_meta["checkpoint_id"], selected_global_step=step_state["global_step"],
            selected_training_state_sha256=_sha256(selected / "training_model_state.pt"),
            checkpoint_meta_sha256=_sha256(final / "checkpoint_meta.json"), source_bit_budget=source_budget,
            bit_budget=final_budget, serialized_tensors_compared=len(actual), reloaded_tensors_compared=compared,
            fresh_strict_reload_state_exact=True, lora_rank=lora["rank"], lora_alpha=lora["alpha"],
            selected_target_count=len(selected_shapes), lora_config=lora, projection_count=len(shapes), architecture=architecture,
            frozen_decoder_tensors=len(decoder_keys), frozen_code_banks=len(code_keys), frozen_norm_tensors=frozen_norm_count,
            head_and_embedding_frozen=True, native_lora_dtype="torch.bfloat16")
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
