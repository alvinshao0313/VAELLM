from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from mix_bit.candidate_artifact import (
    CANDIDATE_META_FILENAME,
    COMPLETED_FILENAME,
    MODULE_STATE_FILENAME,
)
from mix_bit.candidate_contract import (
    validate_mode_payload,
    validate_module_spec_mode_contract,
)
from mix_bit.candidate_pool import candidate_pool_root, generate_candidate_trials
from mix_bit.model_inventory import ModelInventory, TargetLinearSpec, validate_inventory_for_run
from mix_bit.schema import CandidateMode, ResolvedRunConfig, sha256_file

FORBIDDEN_PAYLOAD_LEAVES = frozenset(
    {
        "original_weight",
        "low_rank_a",
        "low_rank_b",
    }
)
PROTECTED_SPARSE_SPEC_KEYS = (
    "protected_input_indices",
    "protected_input_weight",
    "protected_input_qvalues",
    "protected_input_scales",
    "protected_output_indices",
    "protected_output_weight",
    "protected_output_qvalues",
    "protected_output_scales",
    "protected_residual_indices",
    "protected_residual_stage_vq_weights",
    "protected_residual_stage_decoders",
    "low_rank_a",
    "low_rank_b",
    "sparse_residual_row_indices",
    "sparse_residual_col_indices",
    "sparse_residual_values",
    "sparse_residual_active_block_ids",
    "sparse_residual_block_ptr",
    "sparse_residual_local_indices",
    "sparse_residual_qvalues",
    "sparse_residual_scales",
    "sparse_residual_zero_points",
)


@dataclass(frozen=True)
class CheckpointSource:
    category: str
    module_suffix: str
    mode_name: str
    trial_root: str
    candidate_meta_path: str
    compact_state_path: str
    candidate_meta_sha256: str
    compact_state_sha256: str


@dataclass(frozen=True)
class ModuleCandidate:
    module_name: str
    category: str
    module_suffix: str
    block_index: int
    mode_name: str
    nominal_bit: float
    in_features: int
    out_features: int
    has_bias: bool
    param_count: int
    source: CheckpointSource
    module_spec: dict[str, Any]


@dataclass(frozen=True)
class CandidatePoolIndex:
    model_id: str
    run_id: str
    category_count: int
    target_linear_count: int
    mode_count: int
    expected_trial_count: int
    dense_module_mode_count: int
    inventory_fingerprint: str
    run_config_sha256: str
    model_profile_sha256: str
    candidate_space_sha256: str
    training_recipe_sha256: str
    candidates: dict[tuple[str, str], ModuleCandidate]
    sources: tuple[CheckpointSource, ...]
    manifest_path: str


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


def _targets_by_category(inventory: ModelInventory) -> dict[str, tuple[TargetLinearSpec, ...]]:
    grouped: dict[str, list[TargetLinearSpec]] = {}
    for target in inventory.targets:
        grouped.setdefault(target.category, []).append(target)
    return {key: tuple(value) for key, value in grouped.items()}


def _category_suffix_map(resolved: ResolvedRunConfig) -> dict[str, str]:
    return {cat.name: cat.module_suffix for cat in resolved.config.model_profile.categories}


def _reject_protected_or_sparse_payload(spec: Mapping[str, Any], *, module_name: str) -> None:
    for key in PROTECTED_SPARSE_SPEC_KEYS:
        value = spec.get(key)
        if value is None:
            continue
        if isinstance(value, (list, dict)) and len(value) == 0:
            continue
        raise ValueError(
            f"{module_name}: compact candidate rejects protected/sparse payload field {key!r}"
        )
    stages = int(spec.get("protected_residual_stages", 0) or 0)
    if stages > 0:
        raise ValueError(
            f"{module_name}: compact candidate rejects protected_residual_stages={stages}"
        )


def _reject_non_target_payload_keys(
    payload_summaries: Mapping[str, Any],
    expected_names: set[str],
    *,
    category_name: str,
    mode_name: str,
) -> None:
    for key in payload_summaries:
        if not any(key.startswith(f"{name}.") for name in expected_names):
            raise ValueError(
                f"Artifact {category_name}/{mode_name} payload key escapes target prefixes: {key}"
            )
        leaf = key.split(".")[-1]
        if leaf in FORBIDDEN_PAYLOAD_LEAVES or leaf.startswith("protected_") or leaf.startswith(
            "sparse_residual_"
        ):
            raise ValueError(
                f"Artifact {category_name}/{mode_name} rejects forbidden payload key: {key}"
            )
        if "cached" in leaf:
            raise ValueError(
                f"Artifact {category_name}/{mode_name} rejects cached payload key: {key}"
            )


def _require_meta_hashes(
    meta: Mapping[str, Any],
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    label: str,
) -> None:
    required = {
        "run_config_sha256": resolved.run_config_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
    }
    for key, expected in required.items():
        found = meta.get(key)
        if found is None:
            raise ValueError(f"{label}: candidate_meta missing required hash field {key}")
        if found != expected:
            raise ValueError(
                f"{label}: {key} mismatch: meta={found!r} expected={expected!r}"
                + (" (inventory fingerprint mismatch)" if key == "model_inventory_fingerprint" else "")
            )


def _validate_module_spec_against_target(
    spec: Mapping[str, Any],
    target: TargetLinearSpec,
    *,
    module_suffix: str,
) -> None:
    name = str(spec.get("name", ""))
    if name != target.module_name:
        raise ValueError(f"Module spec name {name!r} != inventory {target.module_name!r}")
    if target.module_suffix != module_suffix:
        raise ValueError(
            f"{name}: inventory module_suffix {target.module_suffix!r} != category suffix {module_suffix!r}"
        )
    if not (name == module_suffix or name.endswith("." + module_suffix)):
        raise ValueError(f"{name}: full module name does not end with suffix {module_suffix!r}")
    checks = {
        "in_features": int(target.in_features),
        "out_features": int(target.out_features),
        "has_bias": bool(target.has_bias),
        "transpose": bool(target.transpose),
    }
    for key, expected in checks.items():
        found = spec.get(key)
        if found is None:
            raise ValueError(f"{name}: module_spec missing {key}")
        if key in {"in_features", "out_features"}:
            found = int(found)
        elif key in {"has_bias", "transpose"}:
            found = bool(found)
        if found != expected:
            raise ValueError(
                f"{name}: shape/identity mismatch for {key}: spec={found!r} inventory={expected!r}"
            )
    if bool(spec.get("has_original_weight", False)):
        raise ValueError(f"{name}: compact candidate rejects has_original_weight=true")
    _reject_protected_or_sparse_payload(spec, module_name=name)


def _load_and_validate_artifact(
    *,
    trial_root: Path,
    category_name: str,
    module_suffix: str,
    mode: CandidateMode,
    expected_targets: tuple[TargetLinearSpec, ...],
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
) -> tuple[CheckpointSource, list[ModuleCandidate]]:
    label = f"{category_name}/{mode.name}"
    artifact_dir = trial_root / "artifact"
    completed_path = artifact_dir / COMPLETED_FILENAME
    meta_path = artifact_dir / CANDIDATE_META_FILENAME
    state_path = artifact_dir / MODULE_STATE_FILENAME
    for path in (completed_path, meta_path, state_path):
        if not path.is_file():
            raise ValueError(f"Missing compact artifact file for {label}: {path}")

    completed = _read_json(completed_path)
    meta = _read_json(meta_path)
    state_sha = sha256_file(state_path)
    meta_sha = sha256_file(meta_path)

    if completed.get("module_state_sha256") != state_sha:
        raise ValueError(
            f"{label}: completed.json module_state sha256 hash mismatch: "
            f"completed={completed.get('module_state_sha256')!r} file={state_sha!r}"
        )
    if completed.get("candidate_meta_sha256") != meta_sha:
        raise ValueError(
            f"{label}: completed.json candidate_meta sha256 hash mismatch: "
            f"completed={completed.get('candidate_meta_sha256')!r} file={meta_sha!r}"
        )
    if meta.get("module_state_sha256") != state_sha:
        raise ValueError(
            f"{label}: candidate_meta module_state_sha256 hash mismatch: "
            f"meta={meta.get('module_state_sha256')!r} file={state_sha!r}"
        )

    _require_meta_hashes(meta, resolved=resolved, inventory=inventory, label=label)

    if str(meta.get("category_name")) != category_name:
        raise ValueError(
            f"{label}: candidate_meta category_name {meta.get('category_name')!r} "
            f"!= expected {category_name!r}"
        )
    mode_raw = meta.get("mode")
    if not isinstance(mode_raw, dict):
        raise ValueError(f"{label}: candidate_meta mode must be an object, got {mode_raw!r}")
    validate_mode_payload(mode_raw, mode, label=f"{label}/candidate_meta.mode")

    expected_names = [t.module_name for t in expected_targets]
    expected_set = set(expected_names)
    meta_expected = meta.get("expected_module_names")
    if not isinstance(meta_expected, list):
        raise ValueError(f"{label}: candidate_meta.expected_module_names must be a list")
    meta_expected_names = [str(x) for x in meta_expected]
    if len(meta_expected_names) != len(set(meta_expected_names)):
        raise ValueError(f"{label}: duplicate names in expected_module_names: {meta_expected_names}")

    module_specs = meta.get("module_specs")
    if not isinstance(module_specs, list):
        raise ValueError(f"{label}: candidate_meta.module_specs must be a list")

    seen_names: set[str] = set()
    spec_by_name: dict[str, dict[str, Any]] = {}
    for raw_spec in module_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError(f"{label}: module_spec entries must be objects")
        name = str(raw_spec.get("name", ""))
        if not name:
            raise ValueError(f"{label}: module_spec missing name")
        if name in seen_names:
            raise ValueError(f"{label}: duplicate module_spec for {name!r}")
        seen_names.add(name)
        spec_by_name[name] = dict(raw_spec)

    if set(meta_expected_names) != expected_set:
        missing = sorted(expected_set - set(meta_expected_names))
        unexpected = sorted(set(meta_expected_names) - expected_set)
        raise ValueError(
            f"{label}: expected_module_names must match exact inventory target set; "
            f"missing={missing} unexpected={unexpected}"
        )
    missing_specs = sorted(expected_set - seen_names)
    unexpected_specs = sorted(seen_names - expected_set)
    if missing_specs or unexpected_specs:
        raise ValueError(
            f"{label}: module_specs must match exact inventory target set; "
            f"missing={missing_specs} unexpected={unexpected_specs}"
        )

    for target in expected_targets:
        if target.category != category_name:
            raise ValueError(
                f"{label}: inventory target {target.module_name} has wrong category "
                f"{target.category!r}"
            )
        _validate_module_spec_against_target(
            spec_by_name[target.module_name],
            target,
            module_suffix=module_suffix,
        )
        validate_module_spec_mode_contract(
            spec_by_name[target.module_name],
            mode,
            label=f"{label}/{target.module_name}",
        )

    payload_summaries = meta.get("payload_summaries")
    if not isinstance(payload_summaries, dict):
        raise ValueError(f"{label}: candidate_meta.payload_summaries must be an object")
    _reject_non_target_payload_keys(
        payload_summaries,
        expected_set,
        category_name=category_name,
        mode_name=mode.name,
    )

    source = CheckpointSource(
        category=category_name,
        module_suffix=module_suffix,
        mode_name=mode.name,
        trial_root=str(trial_root),
        candidate_meta_path=str(meta_path.resolve()),
        compact_state_path=str(state_path.resolve()),
        candidate_meta_sha256=meta_sha,
        compact_state_sha256=state_sha,
    )
    candidates: list[ModuleCandidate] = []
    for target in expected_targets:
        spec = spec_by_name[target.module_name]
        in_features = int(spec["in_features"])
        out_features = int(spec["out_features"])
        candidates.append(
            ModuleCandidate(
                module_name=target.module_name,
                category=target.category,
                module_suffix=target.module_suffix,
                block_index=int(target.block_index),
                mode_name=mode.name,
                nominal_bit=float(mode.nominal_bit),
                in_features=in_features,
                out_features=out_features,
                has_bias=bool(spec["has_bias"]),
                param_count=in_features * out_features,
                source=source,
                module_spec=spec,
            )
        )
    return source, candidates


def _candidate_manifest_payload(index: CandidatePoolIndex) -> dict[str, Any]:
    """Return the exact canonical manifest payload from a validated index."""
    return {
        "kind": "mix_bit_candidate_pool_manifest",
        "model_id": index.model_id,
        "run_id": index.run_id,
        "C": index.category_count,
        "L": index.target_linear_count,
        "R": index.mode_count,
        "expected_trial_count": index.expected_trial_count,
        "dense_module_mode_count": index.dense_module_mode_count,
        "run_config_sha256": index.run_config_sha256,
        "model_profile_sha256": index.model_profile_sha256,
        "candidate_space_sha256": index.candidate_space_sha256,
        "training_recipe_sha256": index.training_recipe_sha256,
        "model_inventory_fingerprint": index.inventory_fingerprint,
        "artifacts": [
            {
                "category": src.category,
                "module_suffix": src.module_suffix,
                "mode_name": src.mode_name,
                "trial_root": src.trial_root,
                "candidate_meta_path": src.candidate_meta_path,
                "candidate_meta_sha256": src.candidate_meta_sha256,
                "compact_state_path": src.compact_state_path,
                "compact_state_sha256": src.compact_state_sha256,
            }
            for src in index.sources
        ],
    }


def build_candidate_pool_index(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    *,
    output_root: str | None = None,
    write_manifest: bool = True,
) -> CandidatePoolIndex:
    """Validate artifacts; write candidate_manifest.json only when explicitly enabled."""
    validate_inventory_for_run(inventory, resolved)
    trials = generate_candidate_trials(resolved, inventory, output_root=output_root)
    pool_root = candidate_pool_root(resolved, output_root=output_root)
    suffix_by_category = _category_suffix_map(resolved)
    targets_by_cat = _targets_by_category(inventory)

    missing_pairs: list[str] = []
    for trial in trials:
        artifact_dir = Path(trial.trial_root) / "artifact"
        if not (
            (artifact_dir / COMPLETED_FILENAME).is_file()
            and (artifact_dir / CANDIDATE_META_FILENAME).is_file()
            and (artifact_dir / MODULE_STATE_FILENAME).is_file()
        ):
            missing_pairs.append(f"{trial.category_name}/{trial.mode.name}")
    if missing_pairs:
        preview = ", ".join(missing_pairs[:20])
        more = "" if len(missing_pairs) <= 20 else f" (+{len(missing_pairs) - 20} more)"
        raise ValueError(
            f"Missing compact artifacts for {len(missing_pairs)} logical-category/mode pairs: "
            f"{preview}{more}"
        )

    candidates: dict[tuple[str, str], ModuleCandidate] = {}
    sources: list[CheckpointSource] = []
    shape_anchor: dict[str, tuple[int, int, bool, bool, int, str]] = {}

    for trial in trials:
        expected = targets_by_cat.get(trial.category_name)
        if not expected:
            raise ValueError(f"No inventory targets for category {trial.category_name!r}")
        source, module_candidates = _load_and_validate_artifact(
            trial_root=Path(trial.trial_root),
            category_name=trial.category_name,
            module_suffix=suffix_by_category[trial.category_name],
            mode=trial.mode,
            expected_targets=expected,
            resolved=resolved,
            inventory=inventory,
        )
        sources.append(source)
        for cand in module_candidates:
            key = (cand.module_name, cand.mode_name)
            if key in candidates:
                raise ValueError(f"Duplicate candidate mapping for {key}")
            shape_key = cand.module_name
            shape_tuple = (
                cand.in_features,
                cand.out_features,
                cand.has_bias,
                bool(cand.module_spec.get("transpose")),
                cand.block_index,
                cand.module_suffix,
            )
            if shape_key in shape_anchor and shape_anchor[shape_key] != shape_tuple:
                raise ValueError(
                    f"shape mismatch across modes for {cand.module_name}: "
                    f"existing={shape_anchor[shape_key]} new={shape_tuple} mode={cand.mode_name}"
                )
            shape_anchor[shape_key] = shape_tuple
            # Also require inventory agreement (already checked per artifact, keep explicit).
            inv_target = next(t for t in inventory.targets if t.module_name == cand.module_name)
            if (
                cand.in_features != inv_target.in_features
                or cand.out_features != inv_target.out_features
                or cand.has_bias != inv_target.has_bias
                or bool(cand.module_spec.get("transpose")) != inv_target.transpose
                or cand.block_index != inv_target.block_index
                or cand.module_suffix != inv_target.module_suffix
            ):
                raise ValueError(
                    f"Candidate {cand.module_name}/{cand.mode_name} does not match inventory identity/shape"
                )
            candidates[key] = cand

    category_count = len(resolved.config.model_profile.categories)
    mode_count = len(resolved.config.candidate_space.modes)
    target_linear_count = len(inventory.targets)
    expected_trial_count = category_count * mode_count
    dense_module_mode_count = target_linear_count * mode_count
    if len(trials) != expected_trial_count:
        raise ValueError(
            f"Trial count {len(trials)} != C*R={expected_trial_count}"
        )
    if len(candidates) != dense_module_mode_count:
        raise ValueError(
            f"Dense coverage {len(candidates)} != L*R={dense_module_mode_count}"
        )
    if len(sources) != expected_trial_count:
        raise ValueError(f"Source count {len(sources)} != expected_trial_count={expected_trial_count}")

    manifest_path = pool_root / "candidate_manifest.json"
    index = CandidatePoolIndex(
        model_id=inventory.model_id,
        run_id=resolved.config.run_id,
        category_count=category_count,
        target_linear_count=target_linear_count,
        mode_count=mode_count,
        expected_trial_count=expected_trial_count,
        dense_module_mode_count=dense_module_mode_count,
        inventory_fingerprint=inventory.fingerprint_sha256,
        run_config_sha256=resolved.run_config_sha256,
        model_profile_sha256=resolved.model_profile_sha256,
        candidate_space_sha256=resolved.candidate_space_sha256,
        training_recipe_sha256=resolved.training_recipe_sha256,
        candidates=candidates,
        sources=tuple(sources),
        manifest_path=str(manifest_path.resolve()),
    )
    if write_manifest:
        _write_json_atomic(manifest_path, _candidate_manifest_payload(index))
    return index


def build_candidate_pool_index_from_manifest(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    manifest_path: str | Path,
) -> CandidatePoolIndex:
    """Read and validate an existing manifest without rewriting it.

    The supplied manifest path is authoritative: pool_root is its parent, the
    canonical builder is invoked with write_manifest=False, and the supplied
    JSON must equal the payload that would be regenerated from the artifacts.
    """
    manifest_path = Path(manifest_path).resolve()
    if manifest_path.name != "candidate_manifest.json":
        raise ValueError(
            f"Manifest file name must be exactly candidate_manifest.json, got {manifest_path.name!r}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing candidate manifest: {manifest_path}")

    raw_bytes = manifest_path.read_bytes()
    supplied_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    supplied = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    if supplied.get("kind") != "mix_bit_candidate_pool_manifest":
        raise ValueError(
            f"Manifest kind must be mix_bit_candidate_pool_manifest, got {supplied.get('kind')!r}"
        )

    pool_root = manifest_path.parent
    index = build_candidate_pool_index(
        resolved,
        inventory,
        output_root=str(pool_root),
        write_manifest=False,
    )
    if Path(index.manifest_path).resolve() != manifest_path:
        raise ValueError(
            f"Indexed manifest path {index.manifest_path!r} != supplied {str(manifest_path)!r}"
        )

    expected = _candidate_manifest_payload(index)
    if supplied != expected:
        raise ValueError(
            f"Supplied manifest payload does not match the validated candidate pool "
            f"(artifact order, absolute paths, or SHAs differ): {manifest_path}"
        )

    on_disk_sha256 = sha256_file(manifest_path)
    if on_disk_sha256 != supplied_sha256:
        raise ValueError(
            f"Manifest sha256 changed during validation: supplied={supplied_sha256} "
            f"on_disk={on_disk_sha256}"
        )
    return index


def load_compact_state_mmap(source: CheckpointSource) -> Mapping[str, torch.Tensor]:
    return torch.load(
        source.compact_state_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )


def extract_module_state(
    state_dict: Mapping[str, torch.Tensor],
    module_name: str,
) -> dict[str, torch.Tensor]:
    prefix = f"{module_name}."
    local = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not local:
        raise ValueError(f"extract_module_state: empty result for module {module_name!r}")
    return local
