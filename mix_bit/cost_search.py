from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from mix_bit.assembler import build_model_from_assignments, build_uniform_assignments
from mix_bit.calibration import CalibrationExample, build_causal_kl_mask
from mix_bit.checkpoint_pool import (
    CandidatePoolIndex,
    ModuleCandidate,
    load_compact_state_mmap,
)
from mix_bit.kl_metric import (
    KL_MODE_TEACHER_TOPK,
    MetricContract,
    per_sample_exact_forward_kl,
    per_sample_teacher_topk_forward_kl,
    validate_kl_mode_arguments,
)
from mix_bit.model_adapter import get_model_adapter
from mix_bit.model_inventory import ModelInventory
from mix_bit.module_swap import build_candidate_module, temporary_module_swap
from mix_bit.schema import ResolvedRunConfig, sha256_file
from mix_bit.teacher_cache import (
    load_teacher_cache_chunk,
    load_teacher_cache_index,
    validate_teacher_cache_against_inputs,
)

# Re-export for tests that patch these symbols on this module.
__all__ = [
    "CostWorkerContext",
    "audit_baseline_self_swap",
    "create_cost_worker",
    "evaluate_and_write_baseline_per_sample",
    "evaluate_student_per_sample_kl",
    "extract_prefixed_module_state",
    "load_compact_state_mmap",
    "load_teacher_cache_for_worker",
    "load_teacher_model",
    "module_safe_name",
    "run_category_mode_job",
    "summarize_paired_deltas",
    "write_baseline_mode_zero_rows",
    "build_candidate_module",
]


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def module_safe_name(module_name: str) -> str:
    digest = hashlib.sha256(str(module_name).encode("utf-8")).hexdigest()[:16]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(module_name)).strip("_")
    if not sanitized:
        sanitized = "module"
    if len(sanitized) > 48:
        sanitized = sanitized[:48]
    return f"{digest}__{sanitized}"


def summarize_paired_deltas(deltas: np.ndarray) -> dict[str, float]:
    arr = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if arr.size < 1:
        raise ValueError("deltas must contain at least one sample")
    mean = float(arr.mean())
    if arr.size == 1:
        std = 0.0
        se = 0.0
    else:
        std = float(np.std(arr, ddof=1))
        se = float(std / math.sqrt(arr.size))
    return {
        "mean_delta_kl": mean,
        "std_delta_kl": std,
        "standard_error_delta_kl": se,
    }


def write_json_atomic(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)


def write_npz_atomic(path: str | Path, **arrays: Any) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    # Use a file handle so numpy does not append an extra ``.npz`` suffix.
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, out)
    return sha256_file(out)


def extract_prefixed_module_state(
    compact_state: Mapping[str, torch.Tensor],
    module_name: str,
) -> dict[str, torch.Tensor]:
    prefix = f"{module_name}."
    prefixed = {key: value for key, value in compact_state.items() if key.startswith(prefix)}
    if not prefixed:
        raise ValueError(f"No compact state keys for module {module_name!r}")
    return prefixed


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_overlay(path: str | Path) -> dict[str, Any]:
    overlay_path = Path(path)
    if not overlay_path.is_file():
        raise FileNotFoundError(f"Missing baseline overlay: {overlay_path}")
    overlay = _read_json(overlay_path)
    if overlay.get("kind") != "uniform_baseline_overlay":
        raise ValueError(f"Unexpected baseline overlay kind: {overlay.get('kind')!r}")
    return overlay


def _load_calibration_examples(dataset_path: Path) -> list[CalibrationExample]:
    loaded = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, list):
        raise ValueError(f"Calibration dataset must be a list: {dataset_path}")
    examples: list[CalibrationExample] = []
    for item in loaded:
        examples.append(
            CalibrationExample(
                sample_id=int(item["sample_id"]),
                input_ids=item["input_ids"],
                attention_mask=item["attention_mask"],
                labels=item.get("labels"),
            )
        )
    return examples


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _pad_batch(examples: Sequence[CalibrationExample], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    lengths = [int(ex.input_ids.numel()) for ex in examples]
    max_len = max(lengths)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, ex in enumerate(examples):
        n = lengths[i]
        input_ids[i, :n] = ex.input_ids.to(dtype=torch.long)
        attention_mask[i, :n] = ex.attention_mask.to(dtype=torch.long)
        if ex.labels is not None:
            labels[i, :n] = ex.labels.to(dtype=torch.long)
        else:
            labels[i, :n] = ex.input_ids.to(dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@contextmanager
def _disable_use_cache(model: Any) -> Iterator[None]:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "use_cache"):
        yield
        return
    previous = bool(config.use_cache)
    config.use_cache = False
    try:
        yield
    finally:
        config.use_cache = previous


def load_teacher_model(
    resolved: ResolvedRunConfig,
    *,
    device: str,
    access_token: str | None = None,
) -> nn.Module:
    """Load one full-precision teacher for exact_full_vocab workers."""
    profile = resolved.config.model_profile
    adapter = get_model_adapter(profile.adapter)
    model = adapter.load_model(profile, access_token=access_token)
    model.eval()
    model.to(device)
    return model


@dataclass
class TeacherCacheView:
    index: dict[str, Any]
    cache_dir: Path
    index_sha256: str
    by_sample: dict[int, tuple[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.by_sample:
            return
        mapping: dict[int, tuple[str, int]] = {}
        for chunk in self.index.get("chunks", []):
            rel = str(chunk["path"])
            for local_idx, sample_id in enumerate(chunk.get("sample_ids", [])):
                sid = int(sample_id)
                if sid in mapping:
                    raise ValueError(f"Duplicate sample_id in teacher cache: {sid}")
                mapping[sid] = (rel, local_idx)
        self.by_sample = mapping


def load_teacher_cache_for_worker(cache_dir: str | Path) -> TeacherCacheView:
    root = Path(cache_dir)
    index_path = root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing teacher cache index: {index_path}")
    index = load_teacher_cache_index(index_path)
    return TeacherCacheView(
        index=index,
        cache_dir=root,
        index_sha256=sha256_file(index_path),
    )


def _chunk_local_slice(chunk: dict[str, Any], local_idx: int) -> dict[str, torch.Tensor]:
    offsets = chunk["token_offsets"]
    start = int(offsets[local_idx].item())
    end = int(offsets[local_idx + 1].item())
    return {
        "teacher_topk_indices": chunk["teacher_topk_indices"][start:end],
        "teacher_topk_probs": chunk["teacher_topk_probs"][start:end],
        "n_valid": end - start,
    }


def _gather_teacher_topk_for_batch(
    cache: TeacherCacheView,
    sample_ids: Sequence[int],
) -> dict[str, torch.Tensor]:
    indices_parts: list[torch.Tensor] = []
    probs_parts: list[torch.Tensor] = []
    counts: list[int] = []
    # Open each needed chunk at most once per batch.
    opened: dict[str, dict[str, Any]] = {}
    for sid in sample_ids:
        key = int(sid)
        if key not in cache.by_sample:
            raise ValueError(f"Teacher cache missing sample_id={key}")
        rel, local_idx = cache.by_sample[key]
        if rel not in opened:
            opened[rel] = load_teacher_cache_chunk(cache.cache_dir / rel)
        sliced = _chunk_local_slice(opened[rel], local_idx)
        indices_parts.append(sliced["teacher_topk_indices"])
        probs_parts.append(sliced["teacher_topk_probs"])
        counts.append(int(sliced["n_valid"]))
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return {
        "teacher_topk_indices": torch.cat(indices_parts, dim=0),
        "teacher_topk_probs": torch.cat(probs_parts, dim=0),
        "token_offsets": torch.tensor(offsets, dtype=torch.int64),
    }


def _validate_overlay_against_inputs(
    overlay: Mapping[str, Any],
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    overlay_path: Path,
) -> None:
    checks = {
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
    }
    for key, expected in checks.items():
        found = overlay.get(key)
        if found != expected:
            raise ValueError(
                f"Baseline overlay mismatch for {key}: overlay={found!r} current={expected!r}"
            )
    if overlay.get("mode") != resolved.config.candidate_space.baseline_mode:
        raise ValueError(
            "Baseline overlay mode mismatch: "
            f"overlay={overlay.get('mode')!r} "
            f"baseline={resolved.config.candidate_space.baseline_mode!r}"
        )
    declared = overlay.get("overlay_sha256")
    # overlay_sha256 is computed over payload without that field; recompute.
    payload = dict(overlay)
    payload.pop("overlay_sha256", None)
    recomputed = _sha256_bytes(_canonical_json_bytes(payload))
    if declared is not None and declared != recomputed:
        raise ValueError(
            f"Baseline overlay overlay_sha256 mismatch: declared={declared!r} recomputed={recomputed!r}"
        )


def _validate_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    dataset_path: Path,
) -> None:
    dataset_sha = sha256_file(dataset_path)
    if manifest.get("dataset_file_sha256") != dataset_sha:
        raise ValueError(
            "dataset_manifest dataset_file_sha256 mismatch against on-disk dataset: "
            f"manifest={manifest.get('dataset_file_sha256')!r} file={dataset_sha!r}"
        )
    for key, expected in {
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
    }.items():
        found = manifest.get(key)
        if found != expected:
            raise ValueError(
                f"Calibration manifest mismatch for {key}: "
                f"manifest={found!r} current={expected!r}"
            )


@dataclass
class CostWorkerContext:
    resolved: ResolvedRunConfig
    inventory: ModelInventory
    pool_index: CandidatePoolIndex
    overlay: dict[str, Any]
    overlay_path: str
    overlay_sha256: str
    assignments: dict[str, str]
    model: nn.Module
    examples: list[CalibrationExample]
    sample_ids: np.ndarray
    baseline_kl: np.ndarray
    cost_run_root: Path
    device: torch.device
    batch_size: int
    pad_token_id: int
    kl_mode: str
    metric_name: str
    teacher_topk: int | None
    teacher_cache_index: TeacherCacheView | None
    teacher_cache_index_sha256: str
    teacher_model: nn.Module | None
    teacher_model_id: str
    calibration_manifest_sha256: str
    candidate_manifest_sha256: str
    run_config_sha256: str
    model_inventory_sha256: str
    baseline_mode: str


def create_cost_worker(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    baseline_overlay_path: str | Path,
    dataset_path: str | Path,
    dataset_manifest_path: str | Path,
    cost_run_root: str | Path,
    kl_mode: str,
    teacher_topk: int | None = None,
    teacher_cache: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 1,
    pad_token_id: int | None = None,
    access_token: str | None = None,
    skip_baseline_build: bool = False,
    baseline_per_sample_path: str | Path | None = None,
) -> CostWorkerContext:
    contract = validate_kl_mode_arguments(
        kl_mode=kl_mode,
        teacher_topk=teacher_topk,
        teacher_cache=teacher_cache,
        vocab_size=None,
    )
    overlay_path = Path(baseline_overlay_path)
    overlay = _load_overlay(overlay_path)
    _validate_overlay_against_inputs(
        overlay,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        overlay_path=overlay_path,
    )

    dataset_file = Path(dataset_path)
    manifest_path = Path(dataset_manifest_path)
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Missing calibration dataset: {dataset_file}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing calibration manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    _validate_dataset_manifest(
        manifest,
        resolved=resolved,
        inventory=inventory,
        dataset_path=dataset_file,
    )
    examples = _load_calibration_examples(dataset_file)
    if len(examples) != int(manifest["sample_count"]):
        raise ValueError(
            f"Dataset sample_count mismatch: file={len(examples)} manifest={manifest['sample_count']}"
        )
    sample_ids = np.asarray([int(ex.sample_id) for ex in examples], dtype=np.int64)
    if len(set(int(x) for x in sample_ids.tolist())) != int(sample_ids.size):
        raise ValueError("Calibration dataset sample_ids must be unique")

    resolved_pad = pad_token_id
    if resolved_pad is None:
        resolved_pad = manifest.get("pad_token_id")
    if resolved_pad is None:
        raise ValueError("pad_token_id is required (pass explicitly or via dataset manifest)")

    teacher_view: TeacherCacheView | None = None
    teacher_model: nn.Module | None = None
    teacher_cache_sha = ""
    if contract.kl_mode == KL_MODE_TEACHER_TOPK:
        assert teacher_cache is not None
        teacher_view = load_teacher_cache_for_worker(teacher_cache)
        validate_teacher_cache_against_inputs(
            teacher_view.index,
            expected_sample_ids=[int(x) for x in sample_ids.tolist()],
            run_config_sha256=resolved.run_config_sha256,
            model_inventory_fingerprint=inventory.fingerprint_sha256,
            dataset_file_sha256=sha256_file(dataset_file),
            teacher_topk=int(contract.teacher_topk),
            vocab_size=int(teacher_view.index["vocab_size"]),
            cache_prob_dtype=str(teacher_view.index["cache_prob_dtype"]),
        )
        teacher_cache_sha = teacher_view.index_sha256
    else:
        teacher_model = load_teacher_model(resolved, device=device, access_token=access_token)

    cost_root = Path(cost_run_root)
    cost_root.mkdir(parents=True, exist_ok=True)
    assignments = build_uniform_assignments(
        pool_index, str(resolved.config.candidate_space.baseline_mode)
    )

    if skip_baseline_build:
        model = nn.Module()
        baseline_kl = np.zeros(sample_ids.shape[0], dtype=np.float64)
    else:
        model = build_model_from_assignments(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            assignments=assignments,
            device=device,
        )
        model.eval()
        if baseline_per_sample_path is None:
            baseline_info = evaluate_and_write_baseline_per_sample(
                resolved=resolved,
                inventory=inventory,
                pool_index=pool_index,
                baseline_overlay_path=overlay_path,
                dataset_path=dataset_file,
                dataset_manifest_path=manifest_path,
                cost_run_root=cost_root,
                kl_mode=contract.kl_mode,
                teacher_topk=contract.teacher_topk,
                teacher_cache=teacher_cache,
                device=device,
                batch_size=batch_size,
                pad_token_id=int(resolved_pad),
                access_token=access_token,
                existing_model=model,
                existing_teacher_model=teacher_model,
                existing_teacher_cache=teacher_view,
            )
            baseline_path = Path(baseline_info["baseline_per_sample_path"])
        else:
            baseline_path = Path(baseline_per_sample_path)
        baseline_data = np.load(baseline_path, allow_pickle=False)
        baseline_ids = np.asarray(baseline_data["sample_ids"], dtype=np.int64)
        if not np.array_equal(baseline_ids, sample_ids):
            raise ValueError("baseline_per_sample sample_ids must match calibration dataset order")
        _assert_baseline_npz_provenance(
            baseline_data,
            contract=contract,
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            overlay_sha256=sha256_file(overlay_path),
            calibration_manifest_sha256=sha256_file(manifest_path),
            teacher_cache_index_sha256=teacher_cache_sha,
        )
        baseline_kl = np.asarray(baseline_data["baseline_kl"], dtype=np.float64)

    return CostWorkerContext(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        overlay=overlay,
        overlay_path=str(overlay_path.resolve()),
        overlay_sha256=sha256_file(overlay_path),
        assignments=assignments,
        model=model,
        examples=examples,
        sample_ids=sample_ids,
        baseline_kl=baseline_kl,
        cost_run_root=cost_root,
        device=torch.device(device),
        batch_size=int(batch_size),
        pad_token_id=int(resolved_pad),
        kl_mode=contract.kl_mode,
        metric_name=contract.metric_name,
        teacher_topk=contract.teacher_topk,
        teacher_cache_index=teacher_view,
        teacher_cache_index_sha256=teacher_cache_sha,
        teacher_model=teacher_model,
        teacher_model_id=resolved.config.model_profile.model_id,
        calibration_manifest_sha256=sha256_file(manifest_path),
        candidate_manifest_sha256=sha256_file(pool_index.manifest_path),
        run_config_sha256=resolved.run_config_sha256,
        model_inventory_sha256=inventory.fingerprint_sha256,
        baseline_mode=str(resolved.config.candidate_space.baseline_mode),
    )


def _assert_baseline_npz_provenance(
    baseline_data: Any,
    *,
    contract: MetricContract,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    overlay_sha256: str,
    calibration_manifest_sha256: str,
    teacher_cache_index_sha256: str,
) -> None:
    checks = {
        "kl_mode": contract.kl_mode,
        "metric_name": contract.metric_name,
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_sha256": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "baseline_overlay_sha256": overlay_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "teacher_cache_index_sha256": teacher_cache_index_sha256,
        "teacher_model_id": resolved.config.model_profile.model_id,
    }
    for key, expected in checks.items():
        found = baseline_data[key]
        if isinstance(found, np.ndarray):
            found = found.item() if found.shape == () else found
        if str(found) != str(expected):
            raise ValueError(
                f"baseline_per_sample provenance mismatch for {key}: "
                f"file={found!r} current={expected!r}"
            )
    topk_file = int(np.asarray(baseline_data["teacher_topk"]).item())
    expected_topk = -1 if contract.teacher_topk is None else int(contract.teacher_topk)
    if topk_file != expected_topk:
        raise ValueError(
            f"baseline_per_sample teacher_topk mismatch: file={topk_file} expected={expected_topk}"
        )


def evaluate_student_per_sample_kl(ctx: CostWorkerContext) -> dict[str, np.ndarray]:
    model = ctx.model
    model.eval()
    sample_ids: list[int] = []
    per_sample: list[float] = []

    teacher_cm = (
        _disable_use_cache(ctx.teacher_model)
        if ctx.kl_mode != KL_MODE_TEACHER_TOPK and ctx.teacher_model is not None
        else contextlib.nullcontext()
    )
    with _disable_use_cache(model), teacher_cm, torch.inference_mode():
        for batch_examples in _iter_batches(ctx.examples, ctx.batch_size):
            padded = _pad_batch(batch_examples, pad_token_id=ctx.pad_token_id)
            input_ids = padded["input_ids"].to(ctx.device)
            attention_mask = padded["attention_mask"].to(ctx.device)
            labels = padded["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            shifted_student = logits[:, :-1, :]
            valid = build_causal_kl_mask(attention_mask.detach().cpu(), labels)
            batch_ids = [int(ex.sample_id) for ex in batch_examples]

            if ctx.kl_mode == KL_MODE_TEACHER_TOPK:
                if ctx.teacher_cache_index is None:
                    raise ValueError("teacher_topk evaluation requires a teacher cache")
                if ctx.teacher_model is not None:
                    raise ValueError("teacher_topk worker must not hold a teacher model")
                cache_tensors = _gather_teacher_topk_for_batch(ctx.teacher_cache_index, batch_ids)
                kl = per_sample_teacher_topk_forward_kl(
                    teacher_topk_indices=cache_tensors["teacher_topk_indices"],
                    teacher_topk_probs=cache_tensors["teacher_topk_probs"],
                    token_offsets=cache_tensors["token_offsets"],
                    shifted_student_logits=shifted_student,
                    valid_mask=valid,
                )
            else:
                if ctx.teacher_model is None:
                    raise ValueError("exact_full_vocab evaluation requires a resident teacher")
                teacher_out = ctx.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = (
                    teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out[0]
                )
                shifted_teacher = teacher_logits[:, :-1, :]
                kl = per_sample_exact_forward_kl(
                    shifted_teacher.detach(),
                    shifted_student.detach(),
                    valid,
                )

            sample_ids.extend(batch_ids)
            per_sample.extend(float(x) for x in kl.detach().cpu().tolist())

    return {
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
        "per_sample_kl": np.asarray(per_sample, dtype=np.float64),
    }


def evaluate_and_write_baseline_per_sample(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    baseline_overlay_path: str | Path,
    dataset_path: str | Path,
    dataset_manifest_path: str | Path,
    cost_run_root: str | Path,
    kl_mode: str,
    teacher_topk: int | None = None,
    teacher_cache: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 1,
    pad_token_id: int | None = None,
    access_token: str | None = None,
    existing_model: nn.Module | None = None,
    existing_teacher_model: nn.Module | None = None,
    existing_teacher_cache: TeacherCacheView | None = None,
) -> dict[str, Any]:
    contract = validate_kl_mode_arguments(
        kl_mode=kl_mode,
        teacher_topk=teacher_topk,
        teacher_cache=teacher_cache,
        vocab_size=None,
    )
    overlay_path = Path(baseline_overlay_path)
    overlay = _load_overlay(overlay_path)
    _validate_overlay_against_inputs(
        overlay,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        overlay_path=overlay_path,
    )
    dataset_file = Path(dataset_path)
    manifest_path = Path(dataset_manifest_path)
    manifest = _read_json(manifest_path)
    _validate_dataset_manifest(
        manifest,
        resolved=resolved,
        inventory=inventory,
        dataset_path=dataset_file,
    )
    examples = _load_calibration_examples(dataset_file)
    resolved_pad = pad_token_id if pad_token_id is not None else manifest.get("pad_token_id")
    if resolved_pad is None:
        raise ValueError("pad_token_id is required")

    assignments = build_uniform_assignments(
        pool_index, str(resolved.config.candidate_space.baseline_mode)
    )
    model = existing_model
    owns_model = False
    if model is None:
        model = build_model_from_assignments(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            assignments=assignments,
            device=device,
        )
        owns_model = True

    teacher_view = existing_teacher_cache
    teacher_model = existing_teacher_model
    teacher_cache_sha = ""
    owns_teacher = False
    try:
        if contract.kl_mode == KL_MODE_TEACHER_TOPK:
            if teacher_view is None:
                assert teacher_cache is not None
                teacher_view = load_teacher_cache_for_worker(teacher_cache)
            validate_teacher_cache_against_inputs(
                teacher_view.index,
                expected_sample_ids=[int(ex.sample_id) for ex in examples],
                run_config_sha256=resolved.run_config_sha256,
                model_inventory_fingerprint=inventory.fingerprint_sha256,
                dataset_file_sha256=sha256_file(dataset_file),
                teacher_topk=int(contract.teacher_topk),
                vocab_size=int(teacher_view.index["vocab_size"]),
                cache_prob_dtype=str(teacher_view.index["cache_prob_dtype"]),
            )
            teacher_cache_sha = teacher_view.index_sha256
            teacher_model = None
        else:
            if teacher_model is None:
                teacher_model = load_teacher_model(
                    resolved, device=device, access_token=access_token
                )
                owns_teacher = True

        tmp_ctx = CostWorkerContext(
            resolved=resolved,
            inventory=inventory,
            pool_index=pool_index,
            overlay=overlay,
            overlay_path=str(overlay_path.resolve()),
            overlay_sha256=sha256_file(overlay_path),
            assignments=assignments,
            model=model,
            examples=examples,
            sample_ids=np.asarray([int(ex.sample_id) for ex in examples], dtype=np.int64),
            baseline_kl=np.zeros(len(examples), dtype=np.float64),
            cost_run_root=Path(cost_run_root),
            device=torch.device(device),
            batch_size=int(batch_size),
            pad_token_id=int(resolved_pad),
            kl_mode=contract.kl_mode,
            metric_name=contract.metric_name,
            teacher_topk=contract.teacher_topk,
            teacher_cache_index=teacher_view,
            teacher_cache_index_sha256=teacher_cache_sha,
            teacher_model=teacher_model,
            teacher_model_id=resolved.config.model_profile.model_id,
            calibration_manifest_sha256=sha256_file(manifest_path),
            candidate_manifest_sha256=sha256_file(pool_index.manifest_path),
            run_config_sha256=resolved.run_config_sha256,
            model_inventory_sha256=inventory.fingerprint_sha256,
            baseline_mode=str(resolved.config.candidate_space.baseline_mode),
        )
        evaluated = evaluate_student_per_sample_kl(tmp_ctx)
        out_path = Path(cost_run_root) / "baseline_per_sample.npz"
        topk_scalar = -1 if contract.teacher_topk is None else int(contract.teacher_topk)
        digest = write_npz_atomic(
            out_path,
            sample_ids=evaluated["sample_ids"].astype(np.int64),
            baseline_kl=evaluated["per_sample_kl"].astype(np.float64),
            kl_mode=np.asarray(contract.kl_mode),
            metric_name=np.asarray(contract.metric_name),
            teacher_topk=np.int64(topk_scalar),
            run_config_sha256=np.asarray(resolved.run_config_sha256),
            model_inventory_sha256=np.asarray(inventory.fingerprint_sha256),
            candidate_manifest_sha256=np.asarray(sha256_file(pool_index.manifest_path)),
            baseline_overlay_sha256=np.asarray(sha256_file(overlay_path)),
            calibration_manifest_sha256=np.asarray(sha256_file(manifest_path)),
            teacher_cache_index_sha256=np.asarray(teacher_cache_sha),
            teacher_model_id=np.asarray(resolved.config.model_profile.model_id),
        )
        return {
            "baseline_per_sample_path": str(out_path.resolve()),
            "baseline_per_sample_sha256": digest,
            "sample_ids": evaluated["sample_ids"],
            "baseline_kl": evaluated["per_sample_kl"],
            "kl_mode": contract.kl_mode,
            "metric_name": contract.metric_name,
        }
    finally:
        if owns_teacher and teacher_model is not None:
            del teacher_model
        if owns_model and model is not None:
            del model


def _inventory_target_map(inventory: ModelInventory) -> dict[str, Any]:
    return {t.module_name: t for t in inventory.targets}


def _row_paths(cost_run_root: Path, module_name: str, mode: str) -> tuple[Path, Path]:
    safe = module_safe_name(module_name)
    stem = f"{safe}__{mode}"
    return (
        cost_run_root / "per_sample" / f"{stem}.npz",
        cost_run_root / "rows" / f"{stem}.json",
    )


def _modules_for_category_mode(
    pool_index: CandidatePoolIndex,
    inventory: ModelInventory,
    *,
    category: str,
    mode: str,
) -> list[ModuleCandidate]:
    order = {t.module_name: idx for idx, t in enumerate(inventory.targets)}
    selected: list[ModuleCandidate] = []
    for target in inventory.targets:
        if target.category != category:
            continue
        key = (target.module_name, mode)
        if key not in pool_index.candidates:
            raise ValueError(f"Missing candidate for module={target.module_name!r} mode={mode!r}")
        selected.append(pool_index.candidates[key])
    if not selected:
        raise ValueError(f"No inventory modules for category={category!r}")
    selected.sort(key=lambda c: order[c.module_name])
    paths = {c.source.compact_state_path for c in selected}
    if len(paths) != 1:
        raise ValueError(
            f"Category/mode job expects one compact artifact; found {sorted(paths)}"
        )
    return selected


def run_category_mode_job(
    ctx: CostWorkerContext,
    *,
    category: str,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == ctx.baseline_mode:
        raise ValueError(
            f"run_category_mode_job rejects baseline mode {mode!r}; "
            "use write_baseline_mode_zero_rows after self-swap audit"
        )
    candidates = _modules_for_category_mode(
        ctx.pool_index,
        ctx.inventory,
        category=category,
        mode=mode,
    )
    source = candidates[0].source
    digest = sha256_file(source.compact_state_path)
    if digest != source.compact_state_sha256:
        raise ValueError(
            f"Compact artifact hash mismatch for {source.compact_state_path}: "
            f"file={digest} declared={source.compact_state_sha256}"
        )
    compact_state = load_compact_state_mmap(source)
    target_map = _inventory_target_map(ctx.inventory)
    rows: list[dict[str, Any]] = []

    for cand in candidates:
        replacement = None
        try:
            prefixed = extract_prefixed_module_state(compact_state, cand.module_name)
            replacement = build_candidate_module(
                cand,
                prefixed,
                device=ctx.device,
            )
            with temporary_module_swap(ctx.model, cand.module_name, replacement):
                evaluated = evaluate_student_per_sample_kl(ctx)
            # Ensure restore already happened via context manager before write.
            cand_ids = evaluated["sample_ids"]
            if not np.array_equal(cand_ids, ctx.sample_ids):
                raise ValueError(
                    f"Candidate sample_ids mismatch for {cand.module_name}: "
                    f"candidate={cand_ids.tolist()} baseline={ctx.sample_ids.tolist()}"
                )
            candidate_kl = evaluated["per_sample_kl"].astype(np.float64)
            baseline_kl = ctx.baseline_kl.astype(np.float64)
            delta = candidate_kl - baseline_kl  # preserve negatives; never clip
            stats = summarize_paired_deltas(delta)
            inv_target = target_map[cand.module_name]
            param_count = int(inv_target.param_count)

            npz_path, row_path = _row_paths(ctx.cost_run_root, cand.module_name, mode)
            per_sample_sha = write_npz_atomic(
                npz_path,
                sample_ids=ctx.sample_ids.astype(np.int64),
                baseline_kl=baseline_kl,
                candidate_kl=candidate_kl,
                delta_kl=delta.astype(np.float64),
                kl_mode=np.asarray(ctx.kl_mode),
                metric_name=np.asarray(ctx.metric_name),
                teacher_topk=np.int64(-1 if ctx.teacher_topk is None else int(ctx.teacher_topk)),
                module_name=np.asarray(cand.module_name),
                mode=np.asarray(mode),
            )
            row: dict[str, Any] = {
                "module_name": cand.module_name,
                "category": cand.category,
                "module_suffix": cand.module_suffix,
                "block_index": int(cand.block_index),
                "mode": mode,
                "nominal_bit": float(cand.nominal_bit),
                "param_count": param_count,
                "kl_mode": ctx.kl_mode,
                "metric_name": ctx.metric_name,
                "teacher_topk": None if ctx.teacher_topk is None else int(ctx.teacher_topk),
                "sample_count": int(ctx.sample_ids.size),
                "baseline_kl_mean": float(baseline_kl.mean()),
                "candidate_kl_mean": float(candidate_kl.mean()),
                "mean_delta_kl": stats["mean_delta_kl"],
                "std_delta_kl": stats["std_delta_kl"],
                "standard_error_delta_kl": stats["standard_error_delta_kl"],
                "run_config_sha256": ctx.run_config_sha256,
                "model_inventory_sha256": ctx.model_inventory_sha256,
                "candidate_manifest_sha256": ctx.candidate_manifest_sha256,
                "calibration_manifest_sha256": ctx.calibration_manifest_sha256,
                "baseline_overlay_sha256": ctx.overlay_sha256,
                "teacher_cache_index_sha256": ctx.teacher_cache_index_sha256,
                "source_compact_state": source.compact_state_path,
                "source_compact_state_sha256": source.compact_state_sha256,
                "per_sample_file": str(npz_path.resolve()),
                "per_sample_sha256": per_sample_sha,
                "status": "complete",
            }
            write_json_atomic(row_path, row)
            rows.append(row)
        finally:
            if replacement is not None:
                del replacement

    return rows


def _short_batch_logits(model: nn.Module, *, seed: int = 0) -> torch.Tensor:
    config = getattr(model, "config", None)
    vocab = int(getattr(config, "vocab_size", 0) or 0)
    if vocab < 1:
        emb = model.get_input_embeddings()
        vocab = int(emb.num_embeddings)
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab, (2, 8), device=device)
    attention_mask = torch.ones_like(input_ids)
    with _disable_use_cache(model), torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return logits.detach().cpu()


def audit_baseline_self_swap(ctx: CostWorkerContext) -> dict[str, Any]:
    """Rebuild each baseline module via production swap path and compare short-batch logits.

    On equivalence failure, still write ``baseline_self_swap_audit.json`` with
    per-module max abs/rel errors (``passed=False``), then raise. Never write
    zero-cost baseline rows from a failed audit.
    """
    mode = ctx.baseline_mode
    candidates = [
        ctx.pool_index.candidates[(t.module_name, mode)]
        for t in ctx.inventory.targets
    ]
    # Group by compact artifact; open each once.
    by_path: dict[str, list[ModuleCandidate]] = {}
    for cand in candidates:
        by_path.setdefault(cand.source.compact_state_path, []).append(cand)

    module_reports: list[dict[str, Any]] = []
    reference_logits = _short_batch_logits(ctx.model, seed=0)
    rtol = 1e-4
    atol = 1e-4

    for path, group in by_path.items():
        source = group[0].source
        digest = sha256_file(source.compact_state_path)
        if digest != source.compact_state_sha256:
            raise ValueError(
                f"Compact artifact hash mismatch during self-swap audit: {path}"
            )
        compact_state = load_compact_state_mmap(source)
        for cand in group:
            prefixed = extract_prefixed_module_state(compact_state, cand.module_name)
            rebuilt = build_candidate_module(cand, prefixed, device=ctx.device)
            try:
                with temporary_module_swap(ctx.model, cand.module_name, rebuilt):
                    swapped_logits = _short_batch_logits(ctx.model, seed=0)
                if tuple(swapped_logits.shape) != tuple(reference_logits.shape):
                    module_reports.append(
                        {
                            "module_name": cand.module_name,
                            "mode": mode,
                            "max_abs_error": float("inf"),
                            "max_rel_error": float("inf"),
                            "passed": False,
                            "error": (
                                f"shape mismatch {tuple(swapped_logits.shape)} vs "
                                f"{tuple(reference_logits.shape)}"
                            ),
                        }
                    )
                    # Capture failure, write audit for modules so far, then abort.
                    break
                abs_err = (swapped_logits - reference_logits).abs()
                rel_err = abs_err / reference_logits.abs().clamp_min(1e-12)
                max_abs = float(abs_err.max().item())
                max_rel = float(rel_err.max().item())
                close = bool(
                    torch.allclose(
                        swapped_logits,
                        reference_logits,
                        rtol=rtol,
                        atol=atol,
                    )
                )
                module_reports.append(
                    {
                        "module_name": cand.module_name,
                        "mode": mode,
                        "max_abs_error": max_abs,
                        "max_rel_error": max_rel,
                        "passed": close,
                    }
                )
                if not close:
                    break
            finally:
                del rebuilt
        else:
            continue
        break

    audit = {
        "kind": "baseline_self_swap_audit",
        "mode": mode,
        "passed": bool(module_reports) and all(item["passed"] for item in module_reports),
        "rtol": rtol,
        "atol": atol,
        "module_count": len(module_reports),
        "modules": module_reports,
        "run_config_sha256": ctx.run_config_sha256,
        "model_inventory_sha256": ctx.model_inventory_sha256,
        "candidate_manifest_sha256": ctx.candidate_manifest_sha256,
        "baseline_overlay_sha256": ctx.overlay_sha256,
        "kl_mode": ctx.kl_mode,
        "metric_name": ctx.metric_name,
    }
    out_path = ctx.cost_run_root / "baseline_self_swap_audit.json"
    write_json_atomic(out_path, audit)
    audit["audit_path"] = str(out_path.resolve())
    audit["audit_sha256"] = sha256_file(out_path)
    if not audit["passed"]:
        raise ValueError(
            "Baseline self-swap audit failed; refusing zero-cost baseline rows"
        )
    return audit


def write_baseline_mode_zero_rows(
    ctx: CostWorkerContext,
    *,
    audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not audit.get("passed"):
        raise ValueError("Refusing baseline zero rows: self-swap audit did not pass")
    mode = ctx.baseline_mode
    rows: list[dict[str, Any]] = []
    baseline_kl = ctx.baseline_kl.astype(np.float64)
    delta = np.zeros_like(baseline_kl)
    for target in ctx.inventory.targets:
        cand = ctx.pool_index.candidates[(target.module_name, mode)]
        npz_path, row_path = _row_paths(ctx.cost_run_root, target.module_name, mode)
        per_sample_sha = write_npz_atomic(
            npz_path,
            sample_ids=ctx.sample_ids.astype(np.int64),
            baseline_kl=baseline_kl,
            candidate_kl=baseline_kl.copy(),
            delta_kl=delta,
            kl_mode=np.asarray(ctx.kl_mode),
            metric_name=np.asarray(ctx.metric_name),
            teacher_topk=np.int64(-1 if ctx.teacher_topk is None else int(ctx.teacher_topk)),
            module_name=np.asarray(target.module_name),
            mode=np.asarray(mode),
        )
        row: dict[str, Any] = {
            "module_name": target.module_name,
            "category": target.category,
            "module_suffix": target.module_suffix,
            "block_index": int(target.block_index),
            "mode": mode,
            "nominal_bit": float(cand.nominal_bit),
            "param_count": int(target.param_count),
            "kl_mode": ctx.kl_mode,
            "metric_name": ctx.metric_name,
            "teacher_topk": None if ctx.teacher_topk is None else int(ctx.teacher_topk),
            "sample_count": int(ctx.sample_ids.size),
            "baseline_kl_mean": float(baseline_kl.mean()),
            "candidate_kl_mean": float(baseline_kl.mean()),
            "mean_delta_kl": 0.0,
            "std_delta_kl": 0.0,
            "standard_error_delta_kl": 0.0,
            "run_config_sha256": ctx.run_config_sha256,
            "model_inventory_sha256": ctx.model_inventory_sha256,
            "candidate_manifest_sha256": ctx.candidate_manifest_sha256,
            "calibration_manifest_sha256": ctx.calibration_manifest_sha256,
            "baseline_overlay_sha256": ctx.overlay_sha256,
            "teacher_cache_index_sha256": ctx.teacher_cache_index_sha256,
            "source_compact_state": cand.source.compact_state_path,
            "source_compact_state_sha256": cand.source.compact_state_sha256,
            "per_sample_file": str(npz_path.resolve()),
            "per_sample_sha256": per_sample_sha,
            "status": "complete",
            "baseline_self_swap_audit_sha256": audit.get("audit_sha256"),
        }
        write_json_atomic(row_path, row)
        rows.append(row)
    return rows
