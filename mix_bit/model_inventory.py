from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch import nn

from mix_bit.schema import ModelProfile, ResolvedRunConfig


@dataclass(frozen=True)
class TargetLinearSpec:
    module_name: str
    category: str
    module_suffix: str
    block_index: int
    in_features: int
    out_features: int
    has_bias: bool
    param_count: int
    transpose: bool


@dataclass(frozen=True)
class ModelInventory:
    model_id: str
    model_path: str
    transformers_model_type: str
    resolved_model_class: str
    adapter_name: str
    model_profile_sha256: str
    category_order: tuple[str, ...]
    block_count: int
    targets: tuple[TargetLinearSpec, ...]
    total_target_parameters: int
    fingerprint_sha256: str


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def inventory_payload_without_fingerprint(inventory: ModelInventory) -> dict[str, Any]:
    return {
        "adapter_name": inventory.adapter_name,
        "block_count": inventory.block_count,
        "category_order": list(inventory.category_order),
        "model_id": inventory.model_id,
        "model_path": inventory.model_path,
        "model_profile_sha256": inventory.model_profile_sha256,
        "resolved_model_class": inventory.resolved_model_class,
        "targets": [asdict(target) for target in inventory.targets],
        "total_target_parameters": inventory.total_target_parameters,
        "transformers_model_type": inventory.transformers_model_type,
    }


def compute_inventory_fingerprint(inventory: ModelInventory) -> str:
    payload = inventory_payload_without_fingerprint(inventory)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def with_fingerprint(inventory: ModelInventory) -> ModelInventory:
    fingerprint = compute_inventory_fingerprint(inventory)
    return ModelInventory(
        model_id=inventory.model_id,
        model_path=inventory.model_path,
        transformers_model_type=inventory.transformers_model_type,
        resolved_model_class=inventory.resolved_model_class,
        adapter_name=inventory.adapter_name,
        model_profile_sha256=inventory.model_profile_sha256,
        category_order=inventory.category_order,
        block_count=inventory.block_count,
        targets=inventory.targets,
        total_target_parameters=inventory.total_target_parameters,
        fingerprint_sha256=fingerprint,
    )


def _model_type_and_class(model: nn.Module) -> tuple[str, str]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        model_type = "unknown"
    return str(model_type), type(model).__name__


def inventory_from_targets(
    *,
    profile: ModelProfile,
    model: nn.Module,
    targets: tuple[TargetLinearSpec, ...],
    model_profile_sha256: str,
) -> ModelInventory:
    block_count = len({target.block_index for target in targets}) if targets else 0
    total = int(sum(target.param_count for target in targets))
    model_type, model_class = _model_type_and_class(model)
    inventory = ModelInventory(
        model_id=profile.model_id,
        model_path=profile.model_path,
        transformers_model_type=model_type,
        resolved_model_class=model_class,
        adapter_name=profile.adapter,
        model_profile_sha256=model_profile_sha256,
        category_order=tuple(cat.name for cat in profile.categories),
        block_count=block_count,
        targets=targets,
        total_target_parameters=total,
        fingerprint_sha256="",
    )
    return with_fingerprint(inventory)


def apply_regression_expectations(inventory: ModelInventory, profile: ModelProfile) -> None:
    expectations = profile.regression_expectations or {}
    if not expectations:
        return
    actual = {
        "block_count": inventory.block_count,
        "target_linear_count": len(inventory.targets),
        "category_count": len(inventory.category_order),
    }
    for key, expected in expectations.items():
        if key not in actual:
            raise ValueError(f"Unsupported regression expectation key: {key!r}")
        if actual[key] != expected:
            raise ValueError(
                f"Profile regression expectation failed for {key}: "
                f"expected={expected} actual={actual[key]}"
            )


def build_model_inventory(
    resolved: ResolvedRunConfig,
    *,
    access_token: str | None = None,
) -> ModelInventory:
    from mix_bit.model_adapter import get_model_adapter

    profile = resolved.config.model_profile
    adapter = get_model_adapter(profile.adapter)
    model = adapter.load_model(profile, access_token=access_token)
    try:
        targets = adapter.discover_target_linears(model, profile)
        inventory = inventory_from_targets(
            profile=profile,
            model=model,
            targets=targets,
            model_profile_sha256=resolved.model_profile_sha256,
        )
        apply_regression_expectations(inventory, profile)
        return inventory
    finally:
        del model


def inventory_to_dict(inventory: ModelInventory) -> dict[str, Any]:
    payload = inventory_payload_without_fingerprint(inventory)
    payload["fingerprint_sha256"] = inventory.fingerprint_sha256
    return payload


def write_model_inventory(inventory: ModelInventory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(inventory_to_dict(inventory), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def load_model_inventory(path: str) -> ModelInventory:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    targets_raw = raw.get("targets")
    if not isinstance(targets_raw, list):
        raise ValueError(f"Inventory {path} missing targets list")
    targets = tuple(
        TargetLinearSpec(
            module_name=str(item["module_name"]),
            category=str(item["category"]),
            module_suffix=str(item["module_suffix"]),
            block_index=int(item["block_index"]),
            in_features=int(item["in_features"]),
            out_features=int(item["out_features"]),
            has_bias=bool(item["has_bias"]),
            param_count=int(item["param_count"]),
            transpose=bool(item["transpose"]),
        )
        for item in targets_raw
    )
    inventory = ModelInventory(
        model_id=str(raw["model_id"]),
        model_path=str(raw["model_path"]),
        transformers_model_type=str(raw["transformers_model_type"]),
        resolved_model_class=str(raw["resolved_model_class"]),
        adapter_name=str(raw["adapter_name"]),
        model_profile_sha256=str(raw["model_profile_sha256"]),
        category_order=tuple(str(x) for x in raw["category_order"]),
        block_count=int(raw["block_count"]),
        targets=targets,
        total_target_parameters=int(raw["total_target_parameters"]),
        fingerprint_sha256=str(raw.get("fingerprint_sha256", "")),
    )
    expected = compute_inventory_fingerprint(inventory)
    if inventory.fingerprint_sha256 and inventory.fingerprint_sha256 != expected:
        raise ValueError(
            f"Inventory fingerprint mismatch in {path}: "
            f"stored={inventory.fingerprint_sha256} computed={expected}"
        )
    if not inventory.fingerprint_sha256:
        inventory = with_fingerprint(inventory)
    return inventory


def validate_inventory_for_run(inventory: ModelInventory, resolved: ResolvedRunConfig) -> None:
    profile = resolved.config.model_profile
    if inventory.model_id != profile.model_id:
        raise ValueError(
            f"Inventory model_id {inventory.model_id!r} != profile {profile.model_id!r}"
        )
    if inventory.model_path != profile.model_path:
        raise ValueError(
            f"Inventory model_path {inventory.model_path!r} != profile {profile.model_path!r}"
        )
    if inventory.adapter_name != profile.adapter:
        raise ValueError(
            f"Inventory adapter_name {inventory.adapter_name!r} != profile {profile.adapter!r}"
        )
    if inventory.model_profile_sha256 != resolved.model_profile_sha256:
        raise ValueError(
            "Inventory model_profile_sha256 does not match resolved run config "
            f"({inventory.model_profile_sha256} != {resolved.model_profile_sha256})"
        )
    expected_categories = tuple(cat.name for cat in profile.categories)
    if inventory.category_order != expected_categories:
        raise ValueError(
            f"Inventory category_order {inventory.category_order} != profile {expected_categories}"
        )


def maybe_write_inventory(
    inventory: ModelInventory,
    output_path: str,
    *,
    overwrite: bool = False,
) -> tuple[ModelInventory, bool]:
    path = Path(output_path)
    if path.is_file():
        existing = load_model_inventory(str(path))
        if existing.fingerprint_sha256 == inventory.fingerprint_sha256:
            return existing, False
        if not overwrite:
            raise ValueError(
                f"Refusing to overwrite inventory at {path} with different fingerprint: "
                f"old={existing.fingerprint_sha256} new={inventory.fingerprint_sha256}. "
                "Pass --overwrite to replace it."
            )
        print(
            f"Overwriting inventory fingerprint {existing.fingerprint_sha256} -> "
            f"{inventory.fingerprint_sha256}"
        )
    write_model_inventory(inventory, path)
    return inventory, True
