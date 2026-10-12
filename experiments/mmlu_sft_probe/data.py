"""Explicit process-local data registration for the isolated MMLU SFT probe."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
from pathlib import Path

from e2e_common import data as data_module
from train_utils.config.configs import DistillDataConfig


ALIAS = "vaellm_eval_task"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (value.get("status") != "verified" or value.get("rows") != 99842
            or value.get("source", {}).get("repo") != "cais/mmlu"
            or value["source"].get("config") != "auxiliary_train"
            or value["source"].get("split") != "train"
            or value.get("all_rows_matched") is not True
            or value.get("output_bytes_preserved") is not True):
        raise ValueError("Require the fully verified auxiliary_train SFT extraction manifest.")
    if sha256(value["output"]["path"]) != value["output"]["sha256"]:
        raise ValueError("Extracted MMLU SFT data changed.")
    return value


@dataclass
class RegisteredDataConfig(DistillDataConfig):
    # Dataclass serialization places this identity in both native snapshots and
    # immutable checkpoint contracts without changing the production schema.
    data_registration: dict = field(default_factory=dict)


def registered_config(config: DistillDataConfig, manifest_path: str | Path) -> RegisteredDataConfig:
    if config.dataset_mix != ALIAS + "=1" or config.dataset_task != "sft" or config.train_file:
        raise ValueError("This probe only rebinds the single task SFT preset.")
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path)
    original = data_module.DATASET_MIX_SOURCE_PRESETS[ALIAS]
    return RegisteredDataConfig(
        **{item.name: getattr(config, item.name) for item in fields(DistillDataConfig)},
        data_registration={
            "kind": "isolated_mmlu_sft_preset_path",
            "alias": ALIAS,
            "original_path": original.path,
            "path": manifest["output"]["path"],
            "sha256": manifest["output"]["sha256"],
            "rows": manifest["rows"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "source": manifest["source"],
            "sampling": "Unchanged native seed permutation, worker sharding and canonical SFT skip/truncate rules.",
        },
    )


@contextmanager
def registered_preset(config: RegisteredDataConfig):
    """Bind only this interpreter's existing preset; restore it even on failure."""
    registration = config.data_registration
    old = data_module.DATASET_MIX_SOURCE_PRESETS[ALIAS]
    if old.path != registration["original_path"]:
        raise ValueError("The preset was already rebound in this interpreter.")
    if sha256(registration["path"]) != registration["sha256"]:
        raise ValueError("MMLU SFT data differs from the native configuration identity.")
    selected = replace(old, path=registration["path"])
    data_module.DATASET_MIX_SOURCE_PRESETS[ALIAS] = selected
    try:
        yield selected
    finally:
        data_module.DATASET_MIX_SOURCE_PRESETS[ALIAS] = old


def validate_saved_registration(run_dir: str | Path, config: RegisteredDataConfig) -> None:
    """Check native provenance before the independent exporter can run."""
    run_dir = Path(run_dir)
    registration = config.data_registration
    snapshot = json.loads((run_dir / "normalized_e2e_runtime_args.json").read_text())
    if snapshot["canonical_config"]["data"]["data_registration"] != registration:
        raise ValueError("Native normalized configuration lost the isolated data identity.")
    for path in (run_dir / "trainer_state/checkpoint-400/checkpoint_meta.json",
                 run_dir / "final_model/checkpoint_meta.json"):
        meta = json.loads(path.read_text())
        if meta["immutable_resume_contract"]["data"]["data_registration"] != registration:
            raise ValueError(f"Native checkpoint data identity differs: {path}")
    sources = json.loads((run_dir / "run_meta.json").read_text())["data_sources"]
    if (len(sources) != 1 or sources[0]["alias"] != ALIAS
            or sources[0]["path"] != registration["path"]
            or sources[0]["raw_rows"] != registration["rows"]
            or sources[0]["is_iterable"] is not True):
        raise ValueError("The actual native dataset differs from the registered SFT source.")
