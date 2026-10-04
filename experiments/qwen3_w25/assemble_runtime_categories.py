"""Assemble matching v6 sources whose saved inference runtime fields differ."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

from experiments.qwen3_w25.assemble_categories import (
    EXPECTED, _assert_state_equal, _expected_state, _load_state, _parse_categories,
    _replace_modules, _require, _save_and_verify,
)
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import load_v6_meta, resolve_v6_checkpoint_dir, validate_v6_meta
from train_utils.v6_model_loader import load_v6_model_checkpoint


def _validate_runtime_configs(base_config, donor_config, tokenizer_pad_id):
    """Permit only validated training runtime fields in otherwise equal configs."""
    _require(type(tokenizer_pad_id) is int and tokenizer_pad_id >= 0,
             "Original tokenizer must define an integer pad_token_id.")
    configs = [dict(base_config), dict(donor_config)]
    runtime = {}
    for label, config in zip(("base", "donor"), configs):
        cache, pad = config.get("use_cache"), config.get("pad_token_id")
        _require(type(cache) is bool, f"{label}: use_cache must be bool.")
        _require(pad is None or (type(pad) is int and pad == tokenizer_pad_id),
                 f"{label}: pad_token_id must be None or the original tokenizer pad id.")
        runtime[label] = {"use_cache": cache, "pad_token_id": pad}
        for key in ("_name_or_path", "transformers_version", "use_cache", "pad_token_id"):
            config.pop(key, None)
    _require(configs[0] == configs[1] and configs[0].get("model_type") == "qwen3",
             "Source model configurations differ outside the permitted runtime fields.")
    runtime["differences"] = {
        key: {label: runtime[label][key] for label in ("base", "donor")}
        for key in ("use_cache", "pad_token_id")
        if runtime["base"][key] != runtime["donor"][key]
    }
    runtime["original_tokenizer_pad_token_id"] = tokenizer_pad_id
    runtime["load_and_save_config_source"] = "shared original HF model via canonical v6 loader"
    return runtime


def _validate_sources(base_dir, donor_dir, categories):
    metas = [validate_v6_meta(load_v6_meta(str(path)), expected_kind="final_model")
             for path in (base_dir, donor_dir)]
    originals = []
    specifications = []
    for label, meta in zip(("base", "donor"), metas):
        _require(set(meta["compressed_targets"]) == set(EXPECTED)
                 and not meta["pending_dense_targets"] and not meta["skip_targets"],
                 f"{label}: expected all 252 Qwen3-8B projections without pending/skip targets.")
        _require(meta.get("base_model_path"), f"{label}: missing original model path.")
        originals.append(Path(meta["base_model_path"]).resolve())
        _require(meta.get("lora_config") is None
                 and (meta.get("extra_meta") or {}).get("residual_lora") is None,
                 f"{label}: LoRA is forbidden.")
        _require(not meta.get("shared_protected_residual_decoders"),
                 f"{label}: shared decoder registries cannot be transplanted.")
        specs = {spec["name"]: spec for spec in meta["converted_modules"]}
        _require(set(specs) == set(EXPECTED), f"{label}: converted specs do not match targets.")
        for name, spec in specs.items():
            _require((spec["out_features"], spec["in_features"]) == EXPECTED[name],
                     f"{label}: wrong Qwen3-8B projection shape: {name}.")
            _require(not spec.get("has_original_weight") and not spec.get("always_use_original"),
                     f"{label}: original dense payload enabled/present: {name}.")
            _require(spec.get("low_rank_a") is None and spec.get("low_rank_b") is None,
                     f"{label}: LoRA is forbidden: {name}.")
            _require(not spec.get("protected_residual_shared_decoder_refs"),
                     f"{label}: shared decoder reference: {name}.")
        specifications.append(specs)
    _require(specifications[0] == specifications[1],
             "Runtime-field assembly requires exactly matching projection specifications.")
    _require(originals[0] == originals[1] and originals[0].is_dir(),
             "Sources must reference the same existing original model directory.")
    tokenizer = AutoTokenizer.from_pretrained(str(originals[0]), use_fast=True, local_files_only=True)
    configs = [json.loads((path / "config.json").read_text(encoding="utf-8"))
               for path in (base_dir, donor_dir)]
    runtime = _validate_runtime_configs(*configs, tokenizer.pad_token_id)
    names = sorted(name for name in EXPECTED if name.rsplit(".", 1)[1] in categories)
    return *metas, names, originals[0], tokenizer, runtime


def assemble(base_dir, donor_dir, categories, output_dir):
    base_dir, donor_dir = [Path(resolve_v6_checkpoint_dir(str(path))) for path in (base_dir, donor_dir)]
    output_dir = Path(output_dir).resolve()
    _require(not output_dir.exists(), f"Output already exists: {output_dir}.")
    _require(not output_dir.is_relative_to(base_dir) and not output_dir.is_relative_to(donor_dir),
             "Output must be outside both source checkpoints.")
    categories = _parse_categories(categories)
    base_meta, donor_meta, names, original, tokenizer, runtime = _validate_sources(
        base_dir, donor_dir, categories)
    base_state, donor_state = _load_state(base_dir), _load_state(donor_dir)
    expected = _expected_state(base_state, donor_state, names, EXPECTED)
    donor_specs = {spec["name"]: spec for spec in donor_meta["converted_modules"]}
    assembled_meta = dict(base_meta)
    assembled_meta["converted_modules"] = [donor_specs[spec["name"]] if spec["name"] in names else spec
                                           for spec in base_meta["converted_modules"]]
    budget = audit_state_dict(assembled_meta, expected, max_bpw=2.5, max_lora_rank=0)
    _require(budget["budget_passed"], f"Selected combination exceeds the contract: {budget['contract_violations']}")
    provenance = {
        "base_checkpoint_dir": str(base_dir), "base_checkpoint_id": base_meta["checkpoint_id"],
        "donor_checkpoint_dir": str(donor_dir), "donor_checkpoint_id": donor_meta["checkpoint_id"],
        "base_meta_sha256": hashlib.sha256((base_dir / "checkpoint_meta.json").read_bytes()).hexdigest(),
        "donor_meta_sha256": hashlib.sha256((donor_dir / "checkpoint_meta.json").read_bytes()).hexdigest(),
        "original_model_path": str(original), "categories": categories, "replaced_targets": names,
        "source_config_runtime_fields": runtime,
        "purpose": "inference_only; no CAT exact-resume contract",
    }
    print(f"Verified source inventory; replacing {len(names)}/252 projections, expected {budget['user_linear_bpw']:.9f} bpw.", flush=True)
    base, _, _ = load_v6_model_checkpoint(str(base_dir), map_location="cpu", strict=True, expected_kind="final_model")
    donor, _, _ = load_v6_model_checkpoint(str(donor_dir), map_location="cpu", strict=True, expected_kind="final_model")
    _replace_modules(base, donor, names)
    _assert_state_equal(base.state_dict(), expected, "In-memory assembly")
    del donor
    gc.collect()
    return _save_and_verify(base, output_dir, assembled_meta, expected, provenance, tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--categories", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = assemble(args.base, args.donor, args.categories, args.output)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f"Assembly rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "checkpoint_id": report["checkpoint_id"],
                      "user_linear_bpw": report["bit_budget"]["user_linear_bpw"],
                      "report": str(args.output.resolve() / "assembly_report.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
