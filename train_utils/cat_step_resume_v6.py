"""CAT after-category optimizer-step exact-resume helpers for v6 checkpoints."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
import shutil
from typing import Any, Mapping, Optional, Sequence

from torch import nn

from train_utils.distill_data import FORMATTING_VERSION, tokenizer_identity


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return _jsonable(value.to_jsonable())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _local_path_manifest(path: str) -> Optional[dict]:
    abs_path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(abs_path):
        return None

    def _content_signature(entry_path: str, size: int) -> dict:
        full_hash_limit = 1 << 20
        sample_bytes = 1 << 16
        hasher = hashlib.sha256()
        with open(entry_path, "rb") as handle:
            if size <= full_hash_limit:
                hasher.update(handle.read())
                strategy = "full"
            else:
                offsets = (0, max(0, size // 2 - sample_bytes // 2), max(0, size - sample_bytes))
                for offset in offsets:
                    handle.seek(int(offset))
                    hasher.update(int(offset).to_bytes(8, "little", signed=False))
                    hasher.update(handle.read(sample_bytes))
                strategy = "first_middle_last_64k"
        return {"strategy": strategy, "sha256": hasher.hexdigest()}

    def _stat_entry(entry_path: str, *, relative_to: Optional[str] = None) -> dict:
        stat = os.stat(entry_path)
        size = int(stat.st_size)
        return {
            "path": os.path.relpath(entry_path, relative_to) if relative_to is not None else os.path.abspath(entry_path),
            "size": size,
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "inode": int(stat.st_ino),
            "content_signature": _content_signature(entry_path, size),
        }

    if os.path.isfile(abs_path):
        return {"kind": "file", "root": abs_path, "entries": [_stat_entry(abs_path)]}

    entries = []
    for root, dirnames, filenames in os.walk(abs_path):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            entries.append(_stat_entry(os.path.join(root, filename), relative_to=abs_path))
    return {"kind": "directory", "root": abs_path, "entries": entries}


def _dataset_object_identity(dataset: object, *, _seen: Optional[set[int]] = None) -> dict:
    if _seen is None:
        _seen = set()
    object_id = id(dataset)
    if object_id in _seen:
        return {"type": type(dataset).__name__, "cycle": True}
    _seen.add(object_id)

    payload: dict[str, Any] = {"type": type(dataset).__name__}
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is not None:
        payload["fingerprint"] = str(fingerprint)
    try:
        payload["length"] = int(len(dataset))
    except (TypeError, AttributeError):
        payload["length"] = None

    cache_files = getattr(dataset, "cache_files", None)
    if isinstance(cache_files, list):
        resolved_cache_files = []
        for item in cache_files:
            filename = item.get("filename") if isinstance(item, Mapping) else None
            if filename:
                local_manifest = _local_path_manifest(str(filename))
                if local_manifest is not None:
                    resolved_cache_files.append(local_manifest)
        if resolved_cache_files:
            payload["cache_files"] = resolved_cache_files

    raw_dataset = getattr(dataset, "raw_dataset", None)
    if raw_dataset is not None:
        payload["raw_dataset"] = _dataset_object_identity(raw_dataset, _seen=_seen)
    raw_datasets = getattr(dataset, "raw_datasets", None)
    if raw_datasets is not None:
        payload["raw_datasets"] = [
            _dataset_object_identity(item, _seen=_seen)
            for item in tuple(raw_datasets)
        ]
    return payload


def build_distill_dataset_identity(bundle) -> dict:
    source_stats = []
    for source in list(getattr(bundle, "source_stats", ()) or ()):
        if not isinstance(source, Mapping):
            source_stats.append(_jsonable(source))
            continue
        item = {str(key): _jsonable(value) for key, value in source.items()}
        source_path = source.get("path")
        if source_path:
            local_manifest = _local_path_manifest(str(source_path))
            if local_manifest is not None:
                item["local_manifest"] = local_manifest
        source_stats.append(item)
    return {
        "dataset_mix_spec": getattr(bundle, "dataset_mix_spec", None),
        "cache_key": _jsonable(getattr(bundle, "cache_key", None)),
        "source_stats": source_stats,
        "train_dataset": _dataset_object_identity(getattr(bundle, "train_dataset")),
    }


def model_identity(model: nn.Module, model_path: str) -> dict:
    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None) if config is not None else None
    payload = {
        "model_type": getattr(config, "model_type", None) if config is not None else None,
        "architectures": list(architectures or []),
        "hidden_size": getattr(config, "hidden_size", None) if config is not None else None,
        "num_hidden_layers": getattr(config, "num_hidden_layers", None) if config is not None else None,
        "vocab_size": getattr(config, "vocab_size", None) if config is not None else None,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "model_path": str(model_path),
        "revision_hint": None if config is None else getattr(config, "_commit_hash", None),
        "config_digest": digest,
        "config": payload,
        "local_manifest": _local_path_manifest(str(model_path)),
    }


def build_cat_step_immutable_resume_contract(
    *,
    stage,
    trainer_args,
    tokenizer,
    round_base_checkpoint_id: str,
    active_category: str,
    round_base_meta: Mapping[str, object],
    lora_target_names: Sequence[str],
    decoder_target_names: Sequence[str],
    teacher_identity: Optional[Mapping[str, object]],
    dataset_identity: Mapping[str, object],
    lora_config: Optional[Mapping[str, object]],
) -> dict:
    cfg = stage.config
    opt = _jsonable(cfg.opt)
    if isinstance(opt, dict):
        opt.pop("logging_steps", None)
    tokenizer_name, tokenizer_revision = tokenizer_identity(tokenizer)
    return {
        "version": 1,
        "round_base_checkpoint_id": str(round_base_checkpoint_id),
        "active_category": str(active_category),
        "after_category_mode": str(stage.mode),
        "compressed_targets": [str(v) for v in round_base_meta.get("compressed_targets") or ()],
        "pending_dense_targets": [str(v) for v in round_base_meta.get("pending_dense_targets") or ()],
        "skip_targets": [str(v) for v in round_base_meta.get("skip_targets") or ()],
        "completed_categories": [str(v) for v in round_base_meta.get("completed_categories") or ()],
        "compression_categories": [str(v) for v in round_base_meta.get("compression_categories") or ()],
        "target_layers": _jsonable(round_base_meta.get("target_layers")),
        "target_modules": [str(v) for v in round_base_meta.get("target_modules") or ()],
        "lora_target_names": [str(v) for v in lora_target_names],
        "decoder_target_names": [str(v) for v in decoder_target_names],
        "data": _jsonable(cfg.data),
        "dataset_identity": _jsonable(dict(dataset_identity)),
        "loss": _jsonable(cfg.loss),
        "optimization": opt,
        "lora": None if lora_config is None else _jsonable(dict(lora_config)),
        "aux": _jsonable(cfg.aux),
        "runtime": _jsonable(cfg.runtime),
        "tokenizer": {
            "identity": str(tokenizer_name),
            "revision": str(tokenizer_revision),
            "formatting_version": FORMATTING_VERSION,
        },
        "dataloader": {
            "num_workers": int(getattr(trainer_args, "dataloader_num_workers", 0) or 0),
            "drop_last": bool(getattr(trainer_args, "dataloader_drop_last", False)),
        },
        "distributed": {
            "world_size": int(getattr(trainer_args, "world_size", 1) or 1),
            "parallel_mode": str(cfg.runtime.parallel_mode),
            "layer_device_map": str(cfg.runtime.layer_device_map),
        },
        "precision": {
            "bf16": bool(getattr(trainer_args, "bf16", False)),
            "fp16": bool(getattr(trainer_args, "fp16", False)),
            "tf32": getattr(trainer_args, "tf32", None),
            "gradient_checkpointing": bool(getattr(trainer_args, "gradient_checkpointing", False)),
            "gradient_checkpointing_kwargs": _jsonable(
                getattr(trainer_args, "gradient_checkpointing_kwargs", None) or {}
            ),
        },
        "evaluation_execution": {
            "eval_strategy": str(getattr(trainer_args, "eval_strategy", "no") or "no").lower(),
            "eval_on_start": bool(getattr(trainer_args, "eval_on_start", False)),
        },
        "teacher_identity": None if teacher_identity is None else _jsonable(dict(teacher_identity)),
    }


def validate_cat_step_immutable_resume_contract(saved: Mapping[str, object], current: Mapping[str, object]) -> None:
    if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
        raise TypeError("CAT immutable resume contracts must be mappings.")
    if _jsonable(dict(saved)) != _jsonable(dict(current)):
        raise ValueError(
            "CAT exact-resume immutable contract mismatch. Dataset/loss/optimizer/runtime/target/topology "
            "settings must match the saved checkpoint."
        )


def prune_completed_cat_round_roots(
    run_output_dir: str,
    *,
    save_total_limit: Optional[int],
) -> tuple[str, ...]:
    """Prune oldest completed CAT round roots after boundary commit.

    ``None`` preserves HF's unlimited-retention semantics. The caller must
    invoke this only after the category-boundary checkpoint commit/barrier, so
    every removed round has a stable full-model successor.
    """
    if save_total_limit is None:
        return ()
    limit = int(save_total_limit)
    if limit < 1:
        raise ValueError("save_total_limit must be >= 1 when set.")
    rounds_root = os.path.join(os.path.abspath(str(run_output_dir)), "training_rounds")
    if not os.path.isdir(rounds_root):
        return ()
    candidates = []
    for entry in os.scandir(rounds_root):
        if not entry.is_dir():
            continue
        prefix, sep, _rest = entry.name.partition("_")
        if not sep or not prefix.isdigit():
            continue
        candidates.append((int(prefix), entry.name, entry.path))
    candidates.sort(key=lambda item: (item[0], item[1]))
    remove_count = max(0, len(candidates) - limit)
    removed = []
    for _idx, _name, path in candidates[:remove_count]:
        shutil.rmtree(path)
        removed.append(os.path.abspath(path))
    return tuple(removed)


def resolve_cat_round_root(run_output_dir: str, *, category: str, round_idx: int) -> str:
    return os.path.join(
        os.path.abspath(str(run_output_dir)),
        "training_rounds",
        f"{int(round_idx):04d}_{str(category)}",
    )


__all__ = [
    "build_distill_dataset_identity",
    "build_cat_step_immutable_resume_contract",
    "validate_cat_step_immutable_resume_contract",
    "model_identity",
    "prune_completed_cat_round_roots",
    "resolve_cat_round_root",
]
