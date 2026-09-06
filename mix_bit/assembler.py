from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from mix_bit.calibration import (
    TOKENIZER_FINGERPRINT_VERSION,
    compute_tokenizer_config_sha256,
)
from mix_bit.checkpoint_pool import (
    CandidatePoolIndex,
    ModuleCandidate,
    load_compact_state_mmap,
)
from mix_bit.kl_metric import resolve_metric_contract
from mix_bit.model_adapter import get_model_adapter, normalize_tokenizer_for_mix_bit
from mix_bit.model_inventory import ModelInventory, inventory_from_targets
from mix_bit.module_swap import build_candidate_module, refresh_vae_runtime
from mix_bit.schema import ResolvedRunConfig, sha256_file
from mix_bit.solver import OBJECTIVE_MATCH_REL, bit_to_units
from mix_bit.state_fingerprint import (
    STATE_FINGERPRINT_FILENAME,
    STATE_FINGERPRINT_KIND,
    fingerprint_model_state,
    verify_saved_checkpoint_state,
    write_state_fingerprint_manifest,
)
from train_utils.checkpoint_v6 import (
    META_FILENAME,
    STATE_DICT_FILENAME,
    save_v6_full_checkpoint,
)
from transformers import AutoTokenizer


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _module_spec_sha256(spec: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(dict(spec)))


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    module: nn.Module = model
    for part in str(name).split("."):
        module = getattr(module, part)
    return module


def build_uniform_assignments(
    pool_index: CandidatePoolIndex,
    mode_name: str,
) -> dict[str, str]:
    """Assign every target Linear in the pool to ``mode_name``."""
    mode = str(mode_name)
    module_names = sorted({name for name, cand_mode in pool_index.candidates if cand_mode == mode})
    if not module_names:
        raise ValueError(f"No candidates found for mode {mode!r}")
    assignments: dict[str, str] = {}
    for module_name in module_names:
        key = (module_name, mode)
        if key not in pool_index.candidates:
            raise ValueError(f"Missing candidate for module={module_name!r} mode={mode!r}")
        assignments[module_name] = mode
    # Require full Linear coverage for this mode.
    all_modules = {name for name, _mode in pool_index.candidates}
    missing = sorted(all_modules - set(assignments))
    if missing:
        raise ValueError(
            f"Mode {mode!r} does not cover all target modules; missing={missing}"
        )
    if len(assignments) != pool_index.target_linear_count:
        raise ValueError(
            f"Uniform assignment count {len(assignments)} != "
            f"target_linear_count={pool_index.target_linear_count}"
        )
    return assignments


def _resolve_assignment_candidates(
    pool_index: CandidatePoolIndex,
    assignments: Mapping[str, str],
) -> list[ModuleCandidate]:
    if not assignments:
        raise ValueError("assignments must be non-empty")
    names = list(assignments)
    if len(names) != len(set(names)):
        raise ValueError("assignments contain duplicate module names")
    expected = {name for name, _mode in pool_index.candidates}
    assigned = set(names)
    missing = sorted(expected - assigned)
    extra = sorted(assigned - expected)
    if missing or extra:
        raise ValueError(
            f"assignments must cover exact target module set; missing={missing} unexpected={extra}"
        )
    selected: list[ModuleCandidate] = []
    for module_name, mode_name in assignments.items():
        key = (module_name, str(mode_name))
        try:
            selected.append(pool_index.candidates[key])
        except KeyError as exc:
            raise ValueError(
                f"Missing candidate for module={module_name!r} mode={mode_name!r}"
            ) from exc
    return selected


def _parameter_weighted_average_bit(
    candidates: list[ModuleCandidate],
) -> float:
    total = sum(int(c.param_count) for c in candidates)
    if total <= 0:
        raise ValueError("total target parameters must be positive")
    weighted = sum(int(c.param_count) * float(c.nominal_bit) for c in candidates)
    return float(weighted / total)


def _group_candidates_by_compact_artifact(
    candidates: list[ModuleCandidate],
) -> dict[str, list[ModuleCandidate]]:
    groups: dict[str, list[ModuleCandidate]] = defaultdict(list)
    for cand in candidates:
        groups[cand.source.compact_state_path].append(cand)
    return dict(groups)


def _extract_prefixed_module_state(
    compact_state: Mapping[str, torch.Tensor],
    module_name: str,
) -> dict[str, torch.Tensor]:
    prefix = f"{module_name}."
    prefixed = {key: value for key, value in compact_state.items() if key.startswith(prefix)}
    if not prefixed:
        raise ValueError(f"No compact state keys for selected module {module_name!r}")
    return prefixed


def _install_selected_modules_from_compact_state(
    model: nn.Module,
    compact_state: Mapping[str, torch.Tensor],
    candidates: list[ModuleCandidate],
    *,
    device: str,
) -> None:
    """Build selected modules via ``build_candidate_module`` and install into the backbone.

    Only selected prefixes are extracted from the mmap'd artifact; non-selected keys in the
    same compact state are ignored and never installed.
    """
    for cand in candidates:
        prefixed = _extract_prefixed_module_state(compact_state, cand.module_name)
        built = build_candidate_module(cand, prefixed, device=device)
        if not isinstance(built, VAELinear):
            raise TypeError(
                f"{cand.module_name}: expected VAELinear from build_candidate_module, "
                f"got {type(built)}"
            )
        set_module_by_name(model, cand.module_name, built)


def _require_no_original_weight(model: nn.Module, module_names: list[str]) -> None:
    for name in module_names:
        module = _get_module_by_name(model, name)
        if not isinstance(module, VAELinear):
            raise TypeError(f"{name}: expected VAELinear, got {type(module)}")
        if module.original_weight is not None:
            raise ValueError(f"{name}: original_weight must be None after assembly")
        if "original_weight" in module.state_dict():
            raise ValueError(f"{name}: original_weight leaked into state_dict")


def _clear_decoded_caches(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, VAELinear):
            module.clear_decoded_weight_cache()


def build_model_from_assignments(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    assignments: Mapping[str, str],
    device: str,
) -> nn.Module:
    """Load the original backbone and apply compact candidate modules in memory.

    Never writes ``model.state_dict()``, tokenizer files, config files, or a full checkpoint.
    """
    if pool_index.inventory_fingerprint != inventory.fingerprint_sha256:
        raise ValueError(
            "pool_index inventory fingerprint mismatch: "
            f"pool={pool_index.inventory_fingerprint!r} inventory={inventory.fingerprint_sha256!r}"
        )
    if pool_index.run_config_sha256 != resolved.run_config_sha256:
        raise ValueError(
            "pool_index run_config_sha256 mismatch: "
            f"pool={pool_index.run_config_sha256!r} resolved={resolved.run_config_sha256!r}"
        )

    selected = _resolve_assignment_candidates(pool_index, assignments)
    profile = resolved.config.model_profile
    adapter = get_model_adapter(profile.adapter)
    model = adapter.load_model(profile)

    live_targets = adapter.discover_target_linears(model, profile)
    live_inventory = inventory_from_targets(
        profile=profile,
        model=model,
        targets=live_targets,
        model_profile_sha256=resolved.model_profile_sha256,
    )
    if live_inventory.fingerprint_sha256 != inventory.fingerprint_sha256:
        raise ValueError(
            "Live inventory fingerprint mismatch: "
            f"live={live_inventory.fingerprint_sha256!r} "
            f"expected={inventory.fingerprint_sha256!r}"
        )

    for cand in selected:
        if bool(cand.module_spec.get("has_original_weight", False)):
            raise ValueError(
                f"{cand.module_name}: assembly rejects has_original_weight=true"
            )

    # Group by compact artifact: mmap each artifact once, then build/install only
    # the selected modules through Task-4 ``build_candidate_module`` verification.
    groups = _group_candidates_by_compact_artifact(selected)
    for _path, group in groups.items():
        source = group[0].source
        for cand in group:
            if cand.source.compact_state_path != source.compact_state_path:
                raise ValueError("internal error: mixed compact paths in one group")
            if cand.source.compact_state_sha256 != source.compact_state_sha256:
                raise ValueError(
                    f"compact_state_sha256 mismatch within group for {cand.module_name}"
                )
        compact_state = load_compact_state_mmap(source)
        _install_selected_modules_from_compact_state(
            model,
            compact_state,
            group,
            device=device,
        )

    module_names = [c.module_name for c in selected]
    _require_no_original_weight(model, module_names)
    refresh_vae_runtime(model)
    _clear_decoded_caches(model)
    model.eval()
    model.to(device)
    return model


def save_full_checkpoint_from_assignments(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    assignments: Mapping[str, str],
    output_dir: str,
    device: str,
    mix_bit_provenance: Mapping[str, Any],
    access_token: str | None = None,
) -> dict[str, Any]:
    """Build in memory, save a full standalone checkpoint, and verify reload equivalence.

    Reserved for Task 11 final output. Must not be used by baseline prep or cost search.
    """
    profile = resolved.config.model_profile
    adapter = get_model_adapter(profile.adapter)
    source_tokenizer = adapter.load_tokenizer(profile, access_token=access_token)
    source_fingerprint_sha = compute_tokenizer_config_sha256(source_tokenizer)
    source_reported_path = str(getattr(source_tokenizer, "name_or_path", ""))

    enriched_provenance: dict[str, Any] = dict(mix_bit_provenance)
    enriched_provenance["tokenizer_fingerprint_version"] = TOKENIZER_FINGERPRINT_VERSION
    enriched_provenance["tokenizer_fingerprint_sha256"] = source_fingerprint_sha
    enriched_provenance["source_tokenizer_reported_name_or_path"] = source_reported_path

    model = build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device=device,
    )
    try:
        compressed_targets = []
        for name, module in model.named_modules():
            if not isinstance(module, VAELinear):
                continue
            module.unload_original_linear()
            compressed_targets.append(str(name))
        save_paths = save_v6_full_checkpoint(
            model,
            output_dir,
            checkpoint_kind="final_model",
            compressed_targets=tuple(compressed_targets),
            pending_dense_targets=(),
            skip_targets=(),
            train_mode="none",
            lora_config=None,
            base_model_path=profile.model_path,
            tokenizer=source_tokenizer,
            save_config=True,
            extra_meta={"mix_bit": enriched_provenance},
        )
        # Fixed short-batch reference for Task-12 disk reload numerical check.
        reference_logits_path = write_reference_logits(model, output_dir)
        # Fingerprint the source model after the save function returned and its
        # temporary decoder pack/unpack context fully exited; this matches the old
        # capture point of the cloned reference state and ensures original weights
        # are unloaded and modules are back to normal runtime state.
        saved_source_fingerprint = fingerprint_model_state(model)
        converted_names = [
            name for name, module in model.named_modules() if isinstance(module, VAELinear)
        ]
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fingerprint_path = write_state_fingerprint_manifest(
        Path(output_dir) / STATE_FINGERPRINT_FILENAME,
        saved_source_fingerprint,
    )

    verify_saved_checkpoint_state(
        resolved=resolved,
        output_dir=output_dir,
        expected_fingerprint=saved_source_fingerprint,
        expected_converted_module_names=converted_names,
    )

    # Local-only reload of the saved tokenizer and fingerprint comparison.
    try:
        reload_tokenizer = AutoTokenizer.from_pretrained(
            output_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:  # noqa: BLE001 - surface any local reload failure
        raise ValueError(
            f"Failed to reload tokenizer from {output_dir} with local_files_only=True: {exc}; "
            "state/meta/tokenizer files retained for inspection"
        ) from exc
    normalize_tokenizer_for_mix_bit(reload_tokenizer, source_label=str(output_dir))
    reload_fingerprint_sha = compute_tokenizer_config_sha256(reload_tokenizer)
    if reload_fingerprint_sha != source_fingerprint_sha:
        raise ValueError(
            f"Tokenizer fingerprint mismatch after local reload from {output_dir}: "
            f"source={source_fingerprint_sha!r} reload={reload_fingerprint_sha!r}; "
            "state/meta/tokenizer files retained for inspection"
        )

    return {
        "output_dir": output_dir,
        "state_dict": save_paths["state_dict"],
        "meta": save_paths["meta"],
        "converted_module_count": len(converted_names),
        "reference_logits": reference_logits_path,
        "state_fingerprint": fingerprint_path,
        "tokenizer_fingerprint_sha256": source_fingerprint_sha,
        "tokenizer_reported_name_or_path": str(Path(output_dir).resolve()),
    }


def write_uniform_baseline_overlay(
    *,
    output_dir: str,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    assignments: Mapping[str, str],
    mode_name: str,
) -> str:
    """Write tensor-free ``baseline_overlay.json`` atomically. Returns the file path."""
    selected = _resolve_assignment_candidates(pool_index, assignments)
    if any(c.mode_name != mode_name for c in selected):
        bad = sorted({c.mode_name for c in selected if c.mode_name != mode_name})
        raise ValueError(f"Uniform overlay requires single mode {mode_name!r}, found {bad}")

    achieved = _parameter_weighted_average_bit(selected)
    target = float(resolved.config.candidate_space.target_average_bit)
    order = {t.module_name: idx for idx, t in enumerate(inventory.targets)}
    selected_sorted = sorted(selected, key=lambda c: order.get(c.module_name, c.module_name))

    modules_payload = []
    for cand in selected_sorted:
        modules_payload.append(
            {
                "module_name": cand.module_name,
                "category": cand.category,
                "module_suffix": cand.module_suffix,
                "block_index": cand.block_index,
                "mode_name": cand.mode_name,
                "nominal_bit": cand.nominal_bit,
                "in_features": cand.in_features,
                "out_features": cand.out_features,
                "has_bias": cand.has_bias,
                "param_count": cand.param_count,
                "compact_state_path": cand.source.compact_state_path,
                "compact_state_sha256": cand.source.compact_state_sha256,
                "candidate_meta_path": cand.source.candidate_meta_path,
                "candidate_meta_sha256": cand.source.candidate_meta_sha256,
                "module_spec_sha256": _module_spec_sha256(cand.module_spec),
            }
        )

    payload: dict[str, Any] = {
        "kind": "uniform_baseline_overlay",
        "model_id": inventory.model_id,
        "run_id": resolved.config.run_id,
        "base_model_path": resolved.config.model_profile.model_path,
        "mode": mode_name,
        "target_average_bit": target,
        "achieved_average_bit": achieved,
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "candidate_manifest_path": pool_index.manifest_path,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "modules": modules_payload,
    }
    payload["overlay_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / "baseline_overlay.json"
    _write_json_atomic(overlay_path, payload)
    return str(overlay_path.resolve())


def _assert_baseline_dir_has_no_full_model_files(baseline_dir: Path) -> None:
    banned_names = {
        STATE_DICT_FILENAME,
        "model.safetensors",
        "model.safetensors.index.json",
        "embedding.pt",
        "lm_head.pt",
        "module_state.pt",
        META_FILENAME,
    }
    for path in baseline_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in banned_names:
            raise ValueError(f"Baseline directory must not contain full-model file: {path}")
        if path.suffix in {".bin", ".pt", ".safetensors"} and path.name != "baseline_overlay.json":
            raise ValueError(f"Baseline directory must not contain tensor file: {path}")


def _resolve_audit_vocab_size(model: nn.Module) -> int:
    """Resolve vocab size for audit logits without assuming top-level ``embed_tokens``."""
    config = getattr(model, "config", None)
    if config is not None:
        vocab = getattr(config, "vocab_size", None)
        if vocab is not None:
            return int(vocab)

    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb):
        emb = get_emb()
        if emb is not None and hasattr(emb, "num_embeddings"):
            return int(emb.num_embeddings)

    raise ValueError(
        "Unable to determine model vocab size for assembly audit logits; "
        "expected config.vocab_size or get_input_embeddings().num_embeddings"
    )


REFERENCE_LOGITS_FILENAME = "reference_logits.pt"
REFERENCE_LOGITS_SEED = 0


def _deterministic_short_batch(
    model: nn.Module, *, seed: int = REFERENCE_LOGITS_SEED
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(input_ids_cpu, logits_cpu)`` for a fixed short deterministic batch."""
    torch.manual_seed(seed)
    vocab = _resolve_audit_vocab_size(model)
    device = next(model.parameters()).device
    input_ids = torch.randint(0, vocab, (2, 8), device=device)
    attention_mask = torch.ones_like(input_ids)
    was_training = bool(model.training)
    model.eval()
    use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if hasattr(model, "config") and use_cache is not None:
        model.config.use_cache = False
    try:
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return input_ids.detach().cpu(), logits.detach().cpu()
    finally:
        if hasattr(model, "config") and use_cache is not None:
            model.config.use_cache = use_cache
        if was_training:
            model.train()


def _deterministic_logits(model: nn.Module, *, seed: int = REFERENCE_LOGITS_SEED) -> torch.Tensor:
    _input_ids, logits = _deterministic_short_batch(model, seed=seed)
    return logits


def write_reference_logits(model: nn.Module, output_dir: str | Path, *, seed: int = REFERENCE_LOGITS_SEED) -> str:
    """Write ``reference_logits.pt`` beside a final checkpoint for save/reload validation."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_ids, logits = _deterministic_short_batch(model, seed=seed)
    payload = {
        "kind": "mix_bit_reference_logits",
        "seed": int(seed),
        "input_ids": input_ids,
        "logits": logits,
        "logits_shape": list(logits.shape),
    }
    path = out_dir / REFERENCE_LOGITS_FILENAME
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return str(path.resolve())


def prepare_uniform_baseline_overlay(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    device: str,
    skip_audit: bool = False,
) -> dict[str, Any]:
    """Write tensor-free baseline overlay (+ optional in-memory assembly audit)."""
    mode_name = str(resolved.config.candidate_space.baseline_mode)
    assignments = build_uniform_assignments(pool_index, mode_name)
    baseline_dir = Path(resolved.canonical_run_root) / "baseline" / mode_name
    baseline_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = write_uniform_baseline_overlay(
        output_dir=str(baseline_dir),
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        mode_name=mode_name,
    )

    audit: dict[str, Any] = {
        "kind": "uniform_baseline_assembly_audit",
        "mode": mode_name,
        "baseline_overlay_path": overlay_path,
        "baseline_overlay_sha256": sha256_file(overlay_path),
        "device": device,
        "skip_audit": bool(skip_audit),
    }

    if not skip_audit:
        model_a = build_model_from_assignments(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            assignments=assignments,
            device=device,
        )
        try:
            logits_a = _deterministic_logits(model_a)
        finally:
            del model_a
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_b = build_model_from_assignments(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            assignments=assignments,
            device=device,
        )
        try:
            logits_b = _deterministic_logits(model_b)
        finally:
            del model_b
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        abs_err = (logits_a - logits_b).abs()
        rel_err = abs_err / logits_a.abs().clamp_min(1e-12)
        torch.testing.assert_close(logits_a, logits_b, rtol=1e-4, atol=1e-4)
        audit.update(
            {
                "logits_shape": list(logits_a.shape),
                "max_abs_error": float(abs_err.max().item()),
                "max_rel_error": float(rel_err.max().item()),
                "rtol": 1e-4,
                "atol": 1e-4,
                "passed": True,
            }
        )

    audit_path = baseline_dir / "assembly_audit.json"
    _write_json_atomic(audit_path, audit)
    _assert_baseline_dir_has_no_full_model_files(baseline_dir)
    return {
        "baseline_dir": str(baseline_dir.resolve()),
        "baseline_overlay": overlay_path,
        "assembly_audit": str(audit_path.resolve()),
        "mode": mode_name,
        "assignment_count": len(assignments),
    }


def derive_mixed_model_dir(allocation_path: str | Path) -> Path:
    """Derive ``<run_root>/mixed_model/<kl-run>/<stem>/final_model`` from allocation JSON."""
    path = Path(allocation_path).resolve()
    stem = path.stem
    kl_run = path.parent.name
    allocation_dir = path.parent.parent
    if allocation_dir.name != "allocation":
        raise ValueError(
            f"Cannot derive mixed model dir from allocation path {path}: "
            "expected .../allocation/<kl_run>/<stem>.json"
        )
    run_root = allocation_dir.parent
    return run_root / "mixed_model" / kl_run / stem / "final_model"


def _read_allocation_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _validate_allocation_payload(
    payload: Mapping[str, Any],
    *,
    allocation_path: Path,
    allocation_sha256: str,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    allow_suboptimal: bool,
) -> dict[str, str]:
    """Validate allocation JSON; return module_name -> mode_name assignments."""
    if payload.get("kind") != "mix_bit_allocation":
        raise ValueError(
            f"Unexpected allocation kind={payload.get('kind')!r} in {allocation_path}"
        )
    if payload.get("model_id") != inventory.model_id:
        raise ValueError(
            f"allocation model_id mismatch: "
            f"allocation={payload.get('model_id')!r} inventory={inventory.model_id!r}"
        )
    if payload.get("run_id") != resolved.config.run_id:
        raise ValueError(
            f"allocation run_id mismatch: "
            f"allocation={payload.get('run_id')!r} resolved={resolved.config.run_id!r}"
        )

    hash_checks = {
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_sha256": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "candidate_space_sha256": resolved.candidate_space_sha256,
    }
    for key, expected in hash_checks.items():
        found = payload.get(key)
        if found != expected:
            raise ValueError(
                f"allocation provenance mismatch for {key}: "
                f"allocation={found!r} expected={expected!r}"
            )
    for key in ("cost_table_sha256", "cost_table_meta_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"allocation missing required hash field {key}")

    is_optimal = bool(payload.get("is_globally_optimal"))
    if not is_optimal and not allow_suboptimal:
        raise ValueError(
            "allocation is not globally optimal; pass allow_suboptimal=True to accept"
        )

    contract = resolve_metric_contract(
        kl_mode=str(payload.get("kl_mode")),
        teacher_topk=payload.get("teacher_topk"),
    )
    if payload.get("metric_name") != contract.metric_name:
        raise ValueError(
            f"allocation metric_name mismatch: "
            f"allocation={payload.get('metric_name')!r} expected={contract.metric_name!r}"
        )
    if payload.get("teacher_topk") != contract.teacher_topk:
        raise ValueError(
            f"allocation teacher_topk mismatch: "
            f"allocation={payload.get('teacher_topk')!r} expected={contract.teacher_topk!r}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("allocation entries must be a non-empty list")

    expected_modules = [t.module_name for t in inventory.targets]
    expected_set = set(expected_modules)
    seen: dict[str, str] = {}
    objective = 0.0
    used_units = 0
    total_params = 0
    weighted_bits = 0.0
    target_by_name = {t.module_name: t for t in inventory.targets}

    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("allocation entry must be an object")
        name = str(entry["module_name"])
        mode = str(entry["mode"])
        if name in seen:
            raise ValueError(f"duplicate allocation entry for module {name!r}")
        if name not in expected_set:
            raise ValueError(f"allocation entry module not in inventory: {name!r}")
        target = target_by_name[name]
        for field in ("category", "module_suffix", "block_index", "param_count"):
            found = entry.get(field)
            expected = getattr(target, field)
            if field in ("block_index", "param_count"):
                found = int(found)
                expected = int(expected)
            if found != expected:
                raise ValueError(
                    f"allocation entry inventory mismatch for {name}.{field}: "
                    f"entry={found!r} inventory={expected!r}"
                )
        key = (name, mode)
        if key not in pool_index.candidates:
            raise ValueError(f"Missing candidate for module={name!r} mode={mode!r}")
        cand = pool_index.candidates[key]
        if abs(float(entry["nominal_bit"]) - float(cand.nominal_bit)) > 1e-12:
            raise ValueError(
                f"allocation nominal_bit mismatch for {key}: "
                f"entry={entry['nominal_bit']} candidate={cand.nominal_bit}"
            )
        compact_sha = str(entry.get("compact_state_sha256") or "")
        if not compact_sha:
            raise ValueError(f"allocation entry missing compact_state_sha256 for {key}")
        if compact_sha != cand.source.compact_state_sha256:
            raise ValueError(
                f"allocation compact_state_sha256 mismatch for {key}: "
                f"entry={compact_sha!r} candidate={cand.source.compact_state_sha256!r}"
            )
        cost = float(entry["mean_delta_kl"])
        objective += cost
        used_units += int(target.param_count) * bit_to_units(float(cand.nominal_bit))
        total_params += int(target.param_count)
        weighted_bits += int(target.param_count) * float(cand.nominal_bit)
        seen[name] = mode

    missing = sorted(expected_set - set(seen))
    if missing:
        raise ValueError(f"allocation missing modules: {missing}")
    if len(seen) != len(expected_modules):
        raise ValueError(
            f"allocation coverage mismatch: got {len(seen)} expected {len(expected_modules)}"
        )

    target_bit = float(resolved.config.candidate_space.target_average_bit)
    if abs(float(payload["target_average_bit"]) - target_bit) > 1e-12:
        raise ValueError(
            f"allocation target_average_bit mismatch: "
            f"allocation={payload['target_average_bit']} resolved={target_bit}"
        )
    budget_units = bit_to_units(target_bit) * total_params
    if int(payload["budget_bit_units"]) != budget_units:
        raise ValueError(
            f"allocation budget_bit_units mismatch: "
            f"allocation={payload['budget_bit_units']} recomputed={budget_units}"
        )
    if int(payload["used_bit_units"]) != used_units:
        raise ValueError(
            f"allocation used_bit_units mismatch: "
            f"allocation={payload['used_bit_units']} recomputed={used_units}"
        )
    if used_units > budget_units:
        raise ValueError(
            f"allocation budget violation: used={used_units} > budget={budget_units}"
        )
    achieved = weighted_bits / float(total_params) if total_params else 0.0
    if abs(float(payload["achieved_average_bit"]) - achieved) > 1e-9:
        raise ValueError(
            f"allocation achieved_average_bit mismatch: "
            f"allocation={payload['achieved_average_bit']} recomputed={achieved}"
        )
    tol = OBJECTIVE_MATCH_REL * max(1.0, abs(objective))
    if abs(float(payload["objective_delta_kl"]) - objective) > tol:
        raise ValueError(
            f"allocation objective_delta_kl mismatch: "
            f"allocation={payload['objective_delta_kl']} recomputed={objective}"
        )
    if int(payload["total_target_parameters"]) != total_params:
        raise ValueError(
            f"allocation total_target_parameters mismatch: "
            f"allocation={payload['total_target_parameters']} recomputed={total_params}"
        )
    if payload.get("baseline_mode") != resolved.config.candidate_space.baseline_mode:
        raise ValueError(
            f"allocation baseline_mode mismatch: "
            f"allocation={payload.get('baseline_mode')!r} "
            f"resolved={resolved.config.candidate_space.baseline_mode!r}"
        )
    if pool_index.inventory_fingerprint != inventory.fingerprint_sha256:
        raise ValueError(
            "pool_index inventory fingerprint mismatch: "
            f"pool={pool_index.inventory_fingerprint!r} inventory={inventory.fingerprint_sha256!r}"
        )
    if pool_index.run_config_sha256 != resolved.run_config_sha256:
        raise ValueError(
            "pool_index run_config_sha256 mismatch: "
            f"pool={pool_index.run_config_sha256!r} resolved={resolved.run_config_sha256!r}"
        )
    # allocation file hash is recorded by the caller; ensure non-empty.
    if not allocation_sha256:
        raise ValueError("allocation_sha256 must be non-empty")

    return {name: seen[name] for name in expected_modules}


def _build_mix_bit_provenance(
    *,
    payload: Mapping[str, Any],
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    inventory_path: str,
    pool_index: CandidatePoolIndex,
    allocation_path: Path,
    allocation_sha256: str,
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    contract = resolve_metric_contract(
        kl_mode=str(payload["kl_mode"]),
        teacher_topk=payload.get("teacher_topk"),
    )
    teacher_topk = contract.teacher_topk  # None for exact_full_vocab

    order = {t.module_name: idx for idx, t in enumerate(inventory.targets)}
    assignment_entries = []
    compact_by_path: dict[str, dict[str, str]] = {}
    for module_name, mode_name in assignments.items():
        cand = pool_index.candidates[(module_name, mode_name)]
        assignment_entries.append(
            {
                "module_name": module_name,
                "category": cand.category,
                "module_suffix": cand.module_suffix,
                "block_index": cand.block_index,
                "mode_name": mode_name,
                "nominal_bit": cand.nominal_bit,
                "param_count": cand.param_count,
                "compact_state_path": cand.source.compact_state_path,
                "compact_state_sha256": cand.source.compact_state_sha256,
                "mean_delta_kl": next(
                    float(e["mean_delta_kl"])
                    for e in payload["entries"]
                    if e["module_name"] == module_name
                ),
            }
        )
        compact_by_path[cand.source.compact_state_path] = {
            "compact_state_path": cand.source.compact_state_path,
            "compact_state_sha256": cand.source.compact_state_sha256,
        }
    assignment_entries.sort(key=lambda e: order[e["module_name"]])
    compact_artifacts = sorted(
        compact_by_path.values(), key=lambda a: a["compact_state_path"]
    )

    return {
        "kind": "optimal_mixed_bit",
        "model_id": inventory.model_id,
        "run_id": resolved.config.run_id,
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "model_inventory_sha256": sha256_file(inventory_path),
        "candidate_manifest_path": pool_index.manifest_path,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "cost_table_sha256": payload["cost_table_sha256"],
        "cost_table_meta_sha256": payload["cost_table_meta_sha256"],
        "allocation_path": str(allocation_path.resolve()),
        "allocation_sha256": allocation_sha256,
        "baseline_mode": payload["baseline_mode"],
        "target_average_bit": float(payload["target_average_bit"]),
        "achieved_average_bit": float(payload["achieved_average_bit"]),
        "used_bit_units": int(payload["used_bit_units"]),
        "budget_bit_units": int(payload["budget_bit_units"]),
        "objective_delta_kl": float(payload["objective_delta_kl"]),
        "predicted_mixed_model_kl": payload.get("predicted_mixed_model_kl"),
        "is_globally_optimal": bool(payload.get("is_globally_optimal")),
        "allow_suboptimal": bool(payload.get("allow_suboptimal")),
        "kl_mode": contract.kl_mode,
        "metric_name": contract.metric_name,
        "teacher_topk": teacher_topk,
        "assignments": assignment_entries,
        "compact_artifacts": compact_artifacts,
    }


def _assert_checkpoint_has_no_original_or_dense_target_weight(
    output_dir: str,
    *,
    module_names: Sequence[str],
) -> None:
    meta_path = Path(output_dir) / META_FILENAME
    state_path = Path(output_dir) / STATE_DICT_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for spec in meta.get("converted_modules", []):
        if bool(spec.get("has_original_weight")):
            raise ValueError(
                f"{spec.get('name')}: checkpoint meta reports has_original_weight=true"
            )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    try:
        for name in module_names:
            if f"{name}.original_weight" in state:
                raise ValueError(f"{name}: original_weight present in saved state_dict")
            if f"{name}.weight" in state:
                raise ValueError(f"{name}: dense .weight present in saved state_dict")
            vq_keys = [
                key
                for key in state
                if key.startswith(f"{name}.")
                and key.split(".")[-1].startswith("vq_weight")
            ]
            if not vq_keys:
                raise ValueError(f"{name}: missing packed vq_weight buffers in checkpoint")
            for key in vq_keys:
                if state[key].dtype != torch.uint8:
                    raise ValueError(
                        f"{key}: expected uint8 packed codes, got dtype={state[key].dtype}"
                    )
    finally:
        del state


def _existing_mix_bit_provenance(output_dir: Path) -> dict[str, Any] | None:
    """Return existing ``extra_meta.mix_bit`` mapping, or None if missing/corrupt."""
    meta_path = output_dir / META_FILENAME
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, Mapping):
        return None
    extra = meta.get("extra_meta")
    if not isinstance(extra, Mapping):
        return None
    mix = extra.get("mix_bit")
    if not isinstance(mix, Mapping):
        return None
    return dict(mix)


def _output_has_checkpoint_artifacts(output_dir: Path) -> bool:
    return (output_dir / META_FILENAME).is_file() or (output_dir / STATE_DICT_FILENAME).is_file()


def _verify_existing_tokenizer_fingerprint(
    *,
    output_dir: Path,
    existing_mix: Mapping[str, Any],
) -> None:
    """Local-only reload tokenizer from final dir and compare fingerprint to meta."""
    expected_version = existing_mix.get("tokenizer_fingerprint_version")
    if expected_version != TOKENIZER_FINGERPRINT_VERSION:
        raise ValueError(
            f"Existing mixed model at {output_dir} has tokenizer_fingerprint_version="
            f"{expected_version!r}, expected {TOKENIZER_FINGERPRINT_VERSION}; "
            "pass overwrite=True to rebuild"
        )
    expected_sha = existing_mix.get("tokenizer_fingerprint_sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError(
            f"Existing mixed model at {output_dir} missing tokenizer_fingerprint_sha256; "
            "pass overwrite=True to rebuild"
        )
    try:
        reload_tokenizer = AutoTokenizer.from_pretrained(
            str(output_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Cannot local-only reload tokenizer from {output_dir}: {exc}; "
            "pass overwrite=True to rebuild"
        ) from exc
    normalize_tokenizer_for_mix_bit(reload_tokenizer, source_label=str(output_dir))
    reload_sha = compute_tokenizer_config_sha256(reload_tokenizer)
    if reload_sha != expected_sha:
        raise ValueError(
            f"Existing mixed model at {output_dir} tokenizer fingerprint mismatch: "
            f"meta={expected_sha!r} reload={reload_sha!r}; pass overwrite=True to rebuild"
        )


def assemble_optimal_mixed_checkpoint(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    inventory_path: str,
    pool_index: CandidatePoolIndex,
    allocation_path: str,
    device: str,
    allow_suboptimal: bool = False,
    overwrite: bool = False,
    output_dir: str | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    """Validate ``optimal_2bit.json`` and materialize the sole full mixed-bit checkpoint."""
    alloc_path = Path(allocation_path)
    if not alloc_path.is_file():
        raise FileNotFoundError(f"Missing allocation file: {alloc_path}")
    allocation_sha256 = sha256_file(alloc_path)
    payload = _read_allocation_json(alloc_path)
    assignments = _validate_allocation_payload(
        payload,
        allocation_path=alloc_path,
        allocation_sha256=allocation_sha256,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        allow_suboptimal=allow_suboptimal,
    )

    out_dir = Path(output_dir) if output_dir is not None else derive_mixed_model_dir(alloc_path)
    provenance = _build_mix_bit_provenance(
        payload=payload,
        resolved=resolved,
        inventory=inventory,
        inventory_path=inventory_path,
        pool_index=pool_index,
        allocation_path=alloc_path,
        allocation_sha256=allocation_sha256,
        assignments=assignments,
    )

    if _output_has_checkpoint_artifacts(out_dir):
        existing = _existing_mix_bit_provenance(out_dir)
        has_complete = (
            (out_dir / META_FILENAME).is_file() and (out_dir / STATE_DICT_FILENAME).is_file()
        )
        # Tokenizer fingerprint fields are verified separately by
        # _verify_existing_tokenizer_fingerprint; exclude them from the
        # base provenance equality check.
        _tokenizer_fp_fields = {
            "tokenizer_fingerprint_version",
            "tokenizer_fingerprint_sha256",
            "source_tokenizer_reported_name_or_path",
        }
        existing_base = (
            {k: v for k, v in existing.items() if k not in _tokenizer_fp_fields}
            if existing is not None
            else None
        )
        if existing_base is not None and existing_base == provenance and has_complete:
            ref_path = out_dir / REFERENCE_LOGITS_FILENAME
            fingerprint_path = out_dir / STATE_FINGERPRINT_FILENAME
            required_files = {
                STATE_DICT_FILENAME: out_dir / STATE_DICT_FILENAME,
                META_FILENAME: out_dir / META_FILENAME,
                REFERENCE_LOGITS_FILENAME: ref_path,
                STATE_FINGERPRINT_FILENAME: fingerprint_path,
            }
            missing_files = sorted(
                name for name, path in required_files.items() if not path.is_file()
            )
            if missing_files:
                raise ValueError(
                    f"Existing mixed model at {out_dir} is missing required files "
                    f"{missing_files}; pass overwrite=True to rebuild"
                )
            try:
                manifest = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Cannot read state fingerprint manifest at {fingerprint_path}: {exc}; "
                    "pass overwrite=True to rebuild"
                ) from exc
            if not isinstance(manifest, Mapping) or manifest.get("kind") != STATE_FINGERPRINT_KIND:
                raise ValueError(
                    f"State fingerprint manifest at {fingerprint_path} has invalid kind; "
                    "pass overwrite=True to rebuild"
                )
            expected_converted = [str(entry["module_name"]) for entry in payload["entries"]]
            try:
                verify_saved_checkpoint_state(
                    resolved=resolved,
                    output_dir=str(out_dir),
                    expected_fingerprint=manifest,
                    expected_converted_module_names=expected_converted,
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Existing mixed model at {out_dir} failed state verification: {exc}; "
                    "pass overwrite=True to rebuild"
                ) from exc
            _verify_existing_tokenizer_fingerprint(
                output_dir=out_dir,
                existing_mix=existing,
            )
            return {
                "output_dir": str(out_dir.resolve()),
                "state_dict": str((out_dir / STATE_DICT_FILENAME).resolve()),
                "meta": str((out_dir / META_FILENAME).resolve()),
                "converted_module_count": len(assignments),
                "assignment_count": len(assignments),
                "allocation_sha256": allocation_sha256,
                "skipped_identical": True,
                "reference_logits": str(ref_path.resolve()),
                "state_fingerprint": str(fingerprint_path.resolve()),
                "tokenizer_fingerprint_sha256": existing.get("tokenizer_fingerprint_sha256"),
                "tokenizer_reported_name_or_path": str(out_dir.resolve()),
            }
        if not overwrite:
            reason = (
                "missing or corrupt mix_bit provenance"
                if existing is None
                else "different or incomplete provenance"
            )
            raise ValueError(
                f"Refusing to overwrite existing mixed model at {out_dir} "
                f"({reason}); require identical mix_bit provenance or pass overwrite=True"
            )
        shutil.rmtree(out_dir)

    save_result = save_full_checkpoint_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        output_dir=str(out_dir),
        device=device,
        mix_bit_provenance=provenance,
        access_token=access_token,
    )
    _assert_checkpoint_has_no_original_or_dense_target_weight(
        str(out_dir),
        module_names=list(assignments),
    )
    save_result["skipped_identical"] = False
    save_result["allocation_sha256"] = allocation_sha256
    save_result["assignment_count"] = len(assignments)
    return save_result
