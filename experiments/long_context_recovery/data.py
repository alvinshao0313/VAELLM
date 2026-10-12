"""Process-local registration of unchanged existing long-context SFT presets."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
from pathlib import Path

from e2e_common import data as data_module
from train_utils.config.configs import DistillDataConfig

REPOSITORIES = {"longalign": "zai-org/LongAlign-10k", "longalpaca": "Yukang/LongAlpaca-12k"}


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    alias = manifest.get("alias")
    if (manifest.get("schema") != "long_context_sft_registration_v1"
            or manifest.get("status") != "verified"
            or alias not in REPOSITORIES
            or manifest.get("source", {}).get("repo") != REPOSITORIES[alias]
            or manifest.get("purpose") not in {"resource_diagnostic", "full_dataset"}):
        raise ValueError("Require an explicit verified original long-context dataset manifest.")
    if len(manifest["source"].get("revision", "")) != 40:
        raise ValueError("Dataset identity requires a pinned HF commit.")
    for item in (manifest["source"], manifest["output"]):
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Pinned long-context data changed.")
    if manifest["purpose"] == "resource_diagnostic" and not manifest.get("selection"):
        raise ValueError("A diagnostic subset must preserve its source selection record.")
    if (manifest["purpose"] == "full_dataset"
            and manifest["output"]["sha256"] != manifest["source"]["sha256"]):
        raise ValueError("The full-data preset must use the unchanged official file.")
    return manifest


@dataclass
class RegisteredDataConfig(DistillDataConfig):
    data_registration: dict = field(default_factory=dict)


def registered_config(config: DistillDataConfig, manifest_path: str | Path) -> RegisteredDataConfig:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path)
    alias = manifest["alias"]
    if config.dataset_mix != alias + "=1" or config.dataset_task != "sft" or config.train_file:
        raise ValueError("The registered data must use its original single SFT preset.")
    original = data_module.DATASET_MIX_SOURCE_PRESETS[alias]
    if original.path != REPOSITORIES[alias]:
        raise ValueError("The existing preset was already rebound.")
    return RegisteredDataConfig(
        **{item.name: getattr(config, item.name) for item in fields(DistillDataConfig)},
        data_registration={
            "kind": "isolated_long_context_sft_preset_path", "alias": alias,
            "original_path": original.path, "path": manifest["output"]["path"],
            "sha256": manifest["output"]["sha256"], "rows": manifest["output"]["rows"],
            "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "source": manifest["source"], "purpose": manifest["purpose"],
            "selection": manifest.get("selection"), "text_format": original.text_format,
            "sampling": "Unchanged native shuffle, worker sharding, canonical SFT mask and prefix truncation.",
        },
    )


@contextmanager
def registered_preset(config: RegisteredDataConfig):
    registration = config.data_registration
    alias = registration["alias"]
    old = data_module.DATASET_MIX_SOURCE_PRESETS[alias]
    if old.path != registration["original_path"]:
        raise ValueError("The preset was already rebound in this interpreter.")
    if sha256(registration["path"]) != registration["sha256"]:
        raise ValueError("Long-context data differs from the native configuration identity.")
    selected = replace(old, path=registration["path"])
    data_module.DATASET_MIX_SOURCE_PRESETS[alias] = selected
    try:
        yield selected
    finally:
        data_module.DATASET_MIX_SOURCE_PRESETS[alias] = old
