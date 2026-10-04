"""Assemble selected pure-VAE categories into a complete Qwen3-8B v6 final.

This creates an inference checkpoint, not a CAT exact-resume checkpoint.
All loading, comparison and budget accounting run on CPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import audit_checkpoint, audit_state_dict
from train_utils.checkpoint_v6 import (
    load_v6_meta, refresh_vae_linear_runtime_after_state_load,
    resolve_v6_checkpoint_dir, save_v6_full_checkpoint, validate_v6_meta,
)
from train_utils.v6_model_loader import load_v6_model_checkpoint


PROJECTION_SHAPES = {
    "self_attn.q_proj": (4096, 4096), "self_attn.k_proj": (1024, 4096),
    "self_attn.v_proj": (1024, 4096), "self_attn.o_proj": (4096, 4096),
    "mlp.gate_proj": (12288, 4096), "mlp.up_proj": (12288, 4096),
    "mlp.down_proj": (4096, 12288),
}
CATEGORIES = tuple(name.rsplit(".", 1)[1] for name in PROJECTION_SHAPES)
EXPECTED = {f"model.layers.{layer}.{suffix}": shape
            for layer in range(36) for suffix, shape in PROJECTION_SHAPES.items()}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _parse_categories(text):
    categories = [value.strip() for value in text.split(",")]
    _require(categories and len(categories) == len(set(categories))
             and set(categories) <= set(CATEGORIES),
             f"categories must be a nonempty, unique comma-separated subset of {CATEGORIES}.")
    return categories


def _validate_sources(base_dir, donor_dir, categories):
    metas = [validate_v6_meta(load_v6_meta(str(path))) for path in (base_dir, donor_dir)]
    base_meta, donor_meta = metas
    _require(base_meta["checkpoint_kind"] == "final_model", "Base must be a complete final_model.")
    _require(set(base_meta["compressed_targets"]) == set(EXPECTED)
             and not base_meta["pending_dense_targets"] and not base_meta["skip_targets"],
             "Base must cover exactly all 252 Qwen3-8B projections, with no pending/skip targets.")
    originals = []
    for label, meta in zip(("base", "donor"), metas):
        _require(meta["checkpoint_kind"] in {"final_model", "category_boundary"},
                 f"{label}: expected a stable full v6 checkpoint.")
        _require(meta.get("base_model_path"), f"{label}: missing original model path.")
        originals.append(Path(meta["base_model_path"]).resolve())
        _require(meta.get("lora_config") is None
                 and (meta.get("extra_meta") or {}).get("residual_lora") is None,
                 f"{label}: LoRA is forbidden.")
        _require(not meta.get("shared_protected_residual_decoders"),
                 f"{label}: shared decoder registries cannot be transplanted by this category tool.")
        specs = meta["converted_modules"]
        _require({spec["name"] for spec in specs} == set(meta["compressed_targets"]),
                 f"{label}: converted specs do not match compressed targets.")
        for spec in specs:
            name = spec["name"]
            _require(name in EXPECTED and (spec["out_features"], spec["in_features"]) == EXPECTED[name],
                     f"{label}: wrong Qwen3-8B projection name/shape: {name}.")
            _require(not spec.get("has_original_weight") and not spec.get("always_use_original"),
                     f"{label}: original dense payload enabled/present: {name}.")
            _require(spec.get("low_rank_a") is None and spec.get("low_rank_b") is None,
                     f"{label}: LoRA is forbidden: {name}.")
            _require(not spec.get("protected_residual_shared_decoder_refs"),
                     f"{label}: shared decoder reference: {name}.")
    _require(originals[0] == originals[1] and originals[0].is_dir(),
             "Sources must reference the same existing local original model directory.")
    configs = [json.loads((path / "config.json").read_text(encoding="utf-8"))
               for path in (base_dir, donor_dir)]
    for config in configs:
        config.pop("_name_or_path", None)
        config.pop("transformers_version", None)
    _require(configs[0] == configs[1] and configs[0].get("model_type") == "qwen3",
             "Sources have different model configurations or are not Qwen3.")
    names = sorted(name for name in EXPECTED if name.rsplit(".", 1)[1] in categories)
    _require(set(names) <= set(donor_meta["compressed_targets"]),
             "Donor does not contain every requested category in all 36 layers.")
    return base_meta, donor_meta, names, originals[0]


def _load_state(directory):
    state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    for key, tensor in state.items():
        _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                 and tensor.layout == torch.strided, f"Unsupported serialized tensor: {key}.")
        _require(not any(part in key.lower() for part in ("lora", "low_rank_a", "low_rank_b")),
                 f"LoRA tensor is forbidden: {key}.")
    return state


def _assert_state_equal(actual, expected, label):
    _require(set(actual) == set(expected),
             f"{label}: tensor keys differ; missing={sorted(set(expected) - set(actual))[:8]}, "
             f"extra={sorted(set(actual) - set(expected))[:8]}.")
    for key, source in expected.items():
        tensor = actual[key]
        _require(tensor.shape == source.shape and tensor.dtype == source.dtype,
                 f"{label}: shape/dtype differs: {key}.")
        # Byte comparison also preserves signed zero and packed payloads exactly.
        _require(torch.equal(tensor.detach().reshape(-1).contiguous().view(torch.uint8),
                             source.detach().reshape(-1).contiguous().view(torch.uint8)),
                 f"{label}: tensor bytes differ: {key}.")
    return len(expected)


def _expected_state(base_state, donor_state, names, all_projection_names):
    all_prefixes = tuple(name + "." for name in all_projection_names)
    outside = lambda state: {key: value for key, value in state.items() if not key.startswith(all_prefixes)}
    _assert_state_equal(outside(donor_state), outside(base_state), "Uncompressed backbone")
    prefixes = tuple(name + "." for name in names)
    expected = {key: value for key, value in base_state.items() if not key.startswith(prefixes)}
    selected = {key: value for key, value in donor_state.items() if key.startswith(prefixes)}
    for name in names:
        _require(any(key.startswith(name + ".") for key in selected), f"Donor tensor payload missing: {name}.")
    expected.update(selected)
    return expected


def _replace_modules(base, donor, names):
    for name in names:
        old, new = base.get_submodule(name), donor.get_submodule(name)
        _require(isinstance(old, VAELinear) and isinstance(new, VAELinear), f"Expected VAELinear: {name}.")
        _require((old.in_features, old.out_features) == (new.in_features, new.out_features),
                 f"Module shape mismatch: {name}.")
        set_module_by_name(base, name, new)
    refresh_vae_linear_runtime_after_state_load(base)
    base.eval()


def _save_and_verify(model, output_dir, assembled_meta, expected_state, provenance, tokenizer=None):
    saved = save_v6_full_checkpoint(
        model, str(output_dir), checkpoint_kind="final_model",
        compressed_targets=assembled_meta["compressed_targets"], pending_dense_targets=(), skip_targets=(),
        train_mode="none", after_category_mode="none", lora_config=None,
        base_model_path=assembled_meta["base_model_path"], tokenizer=tokenizer,
        extra_meta={"category_assembly": provenance},
    )
    report = {"status": "invalid", "checkpoint_dir": str(output_dir),
              "checkpoint_id": saved["checkpoint_id"], "sources": provenance}
    try:
        serialized = _load_state(output_dir)
        report["serialized_tensors_compared"] = _assert_state_equal(serialized, expected_state, "Saved checkpoint")
        reloaded, meta, _ = load_v6_model_checkpoint(
            str(output_dir), map_location="cpu", strict=True, expected_kind="final_model",
        )
        report["reloaded_tensors_compared"] = _assert_state_equal(
            reloaded.state_dict(), expected_state, "Independent v6 reload")
        _require(meta["checkpoint_id"] not in {provenance["base_checkpoint_id"], provenance["donor_checkpoint_id"]},
                 "Assembled checkpoint must receive a new identity.")
        _require(meta.get("immutable_resume_contract") is None
                 and not (meta.get("extra_meta") or {}).get("cat_runtime_state_file"),
                 "Assembled inference checkpoint must not inherit exact-resume state.")
        _require(set(meta["compressed_targets"]) == set(assembled_meta["compressed_targets"])
                 and not meta["pending_dense_targets"] and not meta["skip_targets"],
                 "Reloaded projection inventory differs from the complete base.")
        _require({spec["name"]: spec for spec in meta["converted_modules"]}
                 == {spec["name"]: spec for spec in assembled_meta["converted_modules"]},
                 "Reloaded module metadata differs from its source (including transpose/rotation).")
        del reloaded
        gc.collect()
        report["bit_budget"] = audit_checkpoint(output_dir, max_bpw=2.5, max_lora_rank=0)
        _require(report["bit_budget"]["budget_passed"],
                 f"Assembled bit budget failed: {report['bit_budget']['contract_violations']}")
        report["status"] = "verified"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "assembly_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def assemble(base_dir, donor_dir, categories, output_dir):
    base_dir, donor_dir = [Path(resolve_v6_checkpoint_dir(str(path))) for path in (base_dir, donor_dir)]
    output_dir = Path(output_dir).resolve()
    _require(not output_dir.exists(), f"Output already exists: {output_dir}.")
    _require(not output_dir.is_relative_to(base_dir) and not output_dir.is_relative_to(donor_dir),
             "Output must be outside both source checkpoints.")
    categories = _parse_categories(categories)
    base_meta, donor_meta, names, original = _validate_sources(base_dir, donor_dir, categories)
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
        "purpose": "inference_only; no CAT exact-resume contract",
    }
    print(f"Verified source inventory; replacing {len(names)}/252 projections, expected {budget['user_linear_bpw']:.9f} bpw.", flush=True)
    base, _, _ = load_v6_model_checkpoint(str(base_dir), map_location="cpu", strict=True, expected_kind="final_model")
    donor, _, _ = load_v6_model_checkpoint(str(donor_dir), map_location="cpu", strict=True)
    _replace_modules(base, donor, names)
    _assert_state_equal(base.state_dict(), expected, "In-memory assembly")
    del donor
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(str(original), use_fast=True, local_files_only=True)
    return _save_and_verify(base, output_dir, assembled_meta, expected, provenance, tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Complete Qwen3-8B v6 final_model.")
    parser.add_argument("--donor", type=Path, required=True, help="Stable v6 with every selected category compressed.")
    parser.add_argument("--categories", required=True, help="Explicit comma-separated projection categories, e.g. o_proj,down_proj.")
    parser.add_argument("--output", type=Path, required=True, help="New checkpoint directory outside both sources.")
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
