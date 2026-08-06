from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from mix_bit.calibration import CalibrationExample, build_causal_kl_mask
from mix_bit.kl_metric import (
    KL_MODE_TEACHER_TOPK,
    METRIC_NAME_TEACHER_TOPK,
    resolve_metric_contract,
)
from mix_bit.model_adapter import get_model_adapter
from mix_bit.model_inventory import load_model_inventory, validate_inventory_for_run
from mix_bit.schema import ResolvedRunConfig, sha256_file

CACHE_INDEX_KIND = "mix_bit_teacher_topk_cache_index"
ALLOWED_CACHE_PROB_DTYPES = frozenset({"bfloat16", "float32"})
FORBIDDEN_TAIL_FIELDS = frozenset(
    {
        "tail",
        "tail_prob",
        "tail_probs",
        "residual",
        "residual_mass",
        "tail_bucket",
        "omitted_mass",
    }
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _resolve_prob_dtype(name: str) -> torch.dtype:
    key = str(name)
    if key not in ALLOWED_CACHE_PROB_DTYPES:
        raise ValueError(
            f"cache_prob_dtype must be one of {sorted(ALLOWED_CACHE_PROB_DTYPES)}, got {key!r}"
        )
    return torch.bfloat16 if key == "bfloat16" else torch.float32


def build_teacher_topk_chunk(
    *,
    sample_ids: Sequence[int],
    shifted_teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    teacher_topk: int,
    cache_prob_dtype: str,
) -> dict[str, Any]:
    """Build one teacher top-k cache chunk for valid causal positions.

    ``teacher_topk_probs`` may be stored as bfloat16 for disk size; row sums can
    then drift off 1. Callers evaluating KL must cast to float32 and renormalize
    over K (see ``per_sample_teacher_topk_forward_kl``).
    """
    if shifted_teacher_logits.ndim != 3:
        raise ValueError(
            "shifted_teacher_logits must be [B, T-1, V], "
            f"got {tuple(shifted_teacher_logits.shape)}"
        )
    if not shifted_teacher_logits.is_floating_point():
        raise ValueError("shifted_teacher_logits must be floating-point")
    mask = valid_mask.bool()
    if mask.shape != shifted_teacher_logits.shape[:2]:
        raise ValueError(
            "valid_mask shape mismatch: "
            f"{tuple(mask.shape)} vs expected {tuple(shifted_teacher_logits.shape[:2])}"
        )
    batch = int(shifted_teacher_logits.shape[0])
    ids = [int(x) for x in sample_ids]
    if len(ids) != batch:
        raise ValueError(f"sample_ids length {len(ids)} != batch {batch}")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids within a chunk must be unique")

    vocab_size = int(shifted_teacher_logits.shape[-1])
    k = int(teacher_topk)
    if k < 1 or k > vocab_size:
        raise ValueError(
            f"teacher_topk must satisfy 1 <= K <= vocab_size; got K={k}, vocab_size={vocab_size}"
        )
    prob_dtype = _resolve_prob_dtype(cache_prob_dtype)

    logits_device = shifted_teacher_logits.device
    mask_device = mask.to(device=logits_device, dtype=torch.bool)
    counts_device = mask_device.sum(dim=-1, dtype=torch.int64)
    if bool((counts_device < 1).any()):
        bad = (counts_device < 1).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Chunk samples with zero valid tokens at local indices {bad}")

    # top-k over every [B,T] row; output is only [B,T,K]
    top_values, top_indices = torch.topk(
        shifted_teacher_logits,
        k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    valid_top_values = top_values[mask_device]
    valid_top_indices = top_indices[mask_device]
    probs = torch.softmax(valid_top_values.float(), dim=-1)

    # Only compact tensors move to CPU
    indices_cpu = valid_top_indices.to(dtype=torch.int32, device="cpu").contiguous()
    probs_cpu = probs.to(dtype=prob_dtype, device="cpu").contiguous()
    counts_cpu = counts_device.to(device="cpu")
    offsets = torch.zeros(batch + 1, dtype=torch.int64, device="cpu")
    offsets[1:] = counts_cpu.cumsum(dim=0)
    n_valid = int(offsets[-1].item())
    if int(indices_cpu.shape[0]) != n_valid:
        raise ValueError("Flattened valid teacher rows do not match token_offsets")

    chunk: dict[str, Any] = {
        "sample_ids": torch.tensor(ids, dtype=torch.int64),
        "token_offsets": offsets,
        "teacher_topk_indices": indices_cpu,
        "teacher_topk_probs": probs_cpu,
        "teacher_topk": k,
        "vocab_size": vocab_size,
        "metric_name": METRIC_NAME_TEACHER_TOPK,
        "kl_mode": KL_MODE_TEACHER_TOPK,
        "cache_prob_dtype": str(cache_prob_dtype),
    }
    if FORBIDDEN_TAIL_FIELDS.intersection(chunk):
        raise ValueError("Teacher cache chunk must not contain tail/residual fields")
    return chunk


def write_teacher_cache_chunk(path: str | Path, chunk: dict[str, Any]) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(chunk, tmp)
    os.replace(tmp, out)
    return sha256_file(out)


def load_teacher_cache_chunk(path: str | Path) -> dict[str, Any]:
    chunk = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(chunk, dict):
        raise ValueError(f"Teacher cache chunk must be a dict: {path}")
    if FORBIDDEN_TAIL_FIELDS.intersection(chunk):
        raise ValueError(f"Teacher cache chunk contains forbidden tail fields: {path}")
    return chunk


def validate_teacher_cache_against_inputs(
    index: dict[str, Any],
    *,
    expected_sample_ids: Sequence[int],
    run_config_sha256: str,
    model_inventory_fingerprint: str,
    dataset_file_sha256: str,
    teacher_topk: int,
    vocab_size: int,
    cache_prob_dtype: str,
) -> None:
    checks = {
        "run_config_sha256": run_config_sha256,
        "model_inventory_fingerprint": model_inventory_fingerprint,
        "dataset_file_sha256": dataset_file_sha256,
        "teacher_topk": int(teacher_topk),
        "vocab_size": int(vocab_size),
        "cache_prob_dtype": str(cache_prob_dtype),
        "kl_mode": KL_MODE_TEACHER_TOPK,
        "metric_name": METRIC_NAME_TEACHER_TOPK,
    }
    for key, expected in checks.items():
        found = index.get(key)
        if found != expected:
            raise ValueError(
                f"Teacher cache mismatch for {key}: existing={found!r} current={expected!r}"
            )

    index_ids = [int(x) for x in index.get("sample_ids", [])]
    expected_ids = [int(x) for x in expected_sample_ids]
    if index_ids != expected_ids:
        raise ValueError(
            "Teacher cache sample_ids order/content mismatch against dataset manifest order"
        )
    if len(set(index_ids)) != len(index_ids):
        raise ValueError("Teacher cache sample_ids must occur exactly once")


def load_teacher_cache_index(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    if data.get("kind") != CACHE_INDEX_KIND:
        raise ValueError(f"Unexpected teacher cache index kind: {data.get('kind')!r}")
    return data


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


def _load_dataset_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


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
def _disable_use_cache(model: Any):
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


def _chunk_is_complete(
    chunk_meta: dict[str, Any],
    *,
    cache_dir: Path,
    expected_sample_ids: Sequence[int],
) -> bool:
    rel = chunk_meta.get("path")
    if not isinstance(rel, str):
        return False
    path = cache_dir / rel
    if not path.is_file():
        return False
    start = int(chunk_meta["sample_start"])
    end = int(chunk_meta["sample_end"])
    declared = [int(x) for x in chunk_meta.get("sample_ids", [])]
    expected = [int(x) for x in expected_sample_ids[start:end]]
    if declared != expected:
        return False
    digest = sha256_file(path)
    return digest == chunk_meta.get("sha256")


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_teacher_topk_cache(
    resolved: ResolvedRunConfig,
    *,
    inventory_path: str | Path,
    dataset_path: str | Path,
    dataset_manifest_path: str | Path,
    teacher_topk: int,
    cache_prob_dtype: str = "bfloat16",
    chunk_samples: int = 16,
    batch_size: int = 1,
    device: str = "cuda",
    output_dir: str | Path | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    contract = resolve_metric_contract(kl_mode=KL_MODE_TEACHER_TOPK, teacher_topk=teacher_topk)
    k = int(contract.teacher_topk)
    prob_dtype_name = str(cache_prob_dtype)
    _resolve_prob_dtype(prob_dtype_name)
    if int(chunk_samples) < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")

    inventory = load_model_inventory(str(inventory_path))
    validate_inventory_for_run(inventory, resolved)

    dataset_file = Path(dataset_path)
    manifest_path = Path(dataset_manifest_path)
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Missing calibration dataset: {dataset_file}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing calibration manifest: {manifest_path}")

    manifest = _load_dataset_manifest(manifest_path)
    dataset_sha = sha256_file(dataset_file)
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

    examples = _load_calibration_examples(dataset_file)
    if len(examples) != int(manifest["sample_count"]):
        raise ValueError(
            f"Dataset sample_count mismatch: file={len(examples)} manifest={manifest['sample_count']}"
        )
    expected_sample_ids = [int(ex.sample_id) for ex in examples]
    if len(set(expected_sample_ids)) != len(expected_sample_ids):
        raise ValueError("Calibration dataset sample_ids must be unique")

    canonical_dir = (
        Path(resolved.canonical_run_root) / "calibration" / "teacher_topk" / f"k{k}"
    )
    cache_dir = Path(output_dir) if output_dir is not None else canonical_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_override = str(cache_dir.resolve()) != str(canonical_dir.resolve())
    index_path = cache_dir / "index.json"

    profile = resolved.config.model_profile
    adapter = get_model_adapter(profile.adapter)
    model = adapter.load_model(profile, access_token=access_token)
    model.eval()
    tokenizer = adapter.load_tokenizer(profile, access_token=access_token)
    vocab_size = int(getattr(model.config, "vocab_size", tokenizer.vocab_size))
    if k > vocab_size:
        raise ValueError(f"teacher_topk={k} exceeds vocabulary size {vocab_size}")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is required for right-padding teacher cache batches")

    existing_index: dict[str, Any] | None = None
    if index_path.is_file():
        existing_index = load_teacher_cache_index(index_path)
        try:
            validate_teacher_cache_against_inputs(
                existing_index,
                expected_sample_ids=expected_sample_ids,
                run_config_sha256=resolved.run_config_sha256,
                model_inventory_fingerprint=inventory.fingerprint_sha256,
                dataset_file_sha256=dataset_sha,
                teacher_topk=k,
                vocab_size=vocab_size,
                cache_prob_dtype=prob_dtype_name,
            )
            if existing_index.get("model_id") not in (None, profile.model_id):
                raise ValueError(
                    "Teacher cache model_id mismatch: "
                    f"existing={existing_index.get('model_id')!r} current={profile.model_id!r}"
                )
            if existing_index.get("model_path") not in (None, profile.model_path):
                raise ValueError(
                    "Teacher cache model_path mismatch: "
                    f"existing={existing_index.get('model_path')!r} "
                    f"current={profile.model_path!r}"
                )
        except ValueError as exc:
            raise ValueError(
                f"Resume output directory {cache_dir} contains a different metric/provenance "
                f"contract than the current inputs: {exc}"
            ) from exc

    chunk_metas: list[dict[str, Any]] = []
    if existing_index is not None:
        chunk_metas = list(existing_index.get("chunks", []))

    device_obj = torch.device(device)
    model.to(device_obj)

    with _disable_use_cache(model), torch.inference_mode():
        for chunk_idx, start in enumerate(range(0, len(examples), int(chunk_samples))):
            end = min(start + int(chunk_samples), len(examples))
            chunk_examples = examples[start:end]
            chunk_ids = [int(ex.sample_id) for ex in chunk_examples]
            rel_name = f"chunk_{chunk_idx:04d}.pt"
            chunk_path = cache_dir / rel_name

            existing_meta = None
            for meta in chunk_metas:
                if int(meta.get("sample_start", -1)) == start and int(meta.get("sample_end", -1)) == end:
                    existing_meta = meta
                    break
            if existing_meta is not None and _chunk_is_complete(
                existing_meta,
                cache_dir=cache_dir,
                expected_sample_ids=expected_sample_ids,
            ):
                continue

            sample_ids_acc: list[int] = []
            per_sample_counts: list[int] = []
            indices_parts: list[torch.Tensor] = []
            probs_parts: list[torch.Tensor] = []
            vocab_seen: int | None = None

            for batch_examples in _iter_batches(chunk_examples, int(batch_size)):
                padded = _pad_batch(batch_examples, pad_token_id=int(pad_token_id))
                input_ids = padded["input_ids"].to(device_obj)
                attention_mask = padded["attention_mask"].to(device_obj)
                labels = padded["labels"]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                shifted = outputs.logits[:, :-1, :]
                if int(shifted.shape[-1]) != vocab_size:
                    raise ValueError(
                        "Teacher logits vocab size disagrees with model/tokenizer vocab: "
                        f"logits_V={int(shifted.shape[-1])} vocab_size={vocab_size}"
                    )
                valid = build_causal_kl_mask(attention_mask.cpu(), labels)
                batch_chunk = build_teacher_topk_chunk(
                    sample_ids=[int(ex.sample_id) for ex in batch_examples],
                    shifted_teacher_logits=shifted.detach(),
                    valid_mask=valid,
                    teacher_topk=k,
                    cache_prob_dtype=prob_dtype_name,
                )
                if vocab_seen is None:
                    vocab_seen = int(batch_chunk["vocab_size"])
                elif int(batch_chunk["vocab_size"]) != vocab_seen:
                    raise ValueError("Inconsistent vocab_size across teacher cache batches")

                local_offsets = batch_chunk["token_offsets"]
                for i, ex in enumerate(batch_examples):
                    sample_ids_acc.append(int(ex.sample_id))
                    per_sample_counts.append(
                        int(local_offsets[i + 1].item()) - int(local_offsets[i].item())
                    )
                indices_parts.append(batch_chunk["teacher_topk_indices"])
                probs_parts.append(batch_chunk["teacher_topk_probs"])

            all_indices = torch.cat(indices_parts, dim=0)
            all_probs = torch.cat(probs_parts, dim=0)
            rebuilt_offsets = [0]
            for count in per_sample_counts:
                rebuilt_offsets.append(rebuilt_offsets[-1] + int(count))
            if rebuilt_offsets[-1] != int(all_indices.shape[0]):
                raise ValueError("Rebuilt token_offsets disagree with concatenated N_valid")
            if sample_ids_acc != chunk_ids:
                raise ValueError("Chunk sample_ids order drifted from dataset order")

            chunk = {
                "sample_ids": torch.tensor(sample_ids_acc, dtype=torch.int64),
                "token_offsets": torch.tensor(rebuilt_offsets, dtype=torch.int64),
                "teacher_topk_indices": all_indices,
                "teacher_topk_probs": all_probs,
                "teacher_topk": k,
                "vocab_size": int(vocab_seen if vocab_seen is not None else vocab_size),
                "metric_name": METRIC_NAME_TEACHER_TOPK,
                "kl_mode": KL_MODE_TEACHER_TOPK,
                "cache_prob_dtype": prob_dtype_name,
            }
            digest = write_teacher_cache_chunk(chunk_path, chunk)
            n_valid = int(all_indices.shape[0])
            meta = {
                "path": rel_name,
                "sample_start": start,
                "sample_end": end,
                "sample_ids": chunk_ids,
                "n_valid": n_valid,
                "sha256": digest,
            }
            chunk_metas = [
                m
                for m in chunk_metas
                if not (
                    int(m.get("sample_start", -1)) == start
                    and int(m.get("sample_end", -1)) == end
                )
            ]
            chunk_metas.append(meta)
            chunk_metas.sort(key=lambda m: int(m["sample_start"]))

    chunk_metas.sort(key=lambda m: int(m["sample_start"]))
    cursor = 0
    for meta in chunk_metas:
        start = int(meta["sample_start"])
        end = int(meta["sample_end"])
        if start != cursor:
            raise ValueError(
                f"Teacher cache chunk coverage gap/overlap near sample {cursor}: "
                f"found range [{start}, {end})"
            )
        if not _chunk_is_complete(
            meta,
            cache_dir=cache_dir,
            expected_sample_ids=expected_sample_ids,
        ):
            raise ValueError(
                f"Incomplete teacher cache chunk for samples [{start}, {end})"
            )
        cursor = end
    if cursor != len(examples):
        raise ValueError(
            f"Teacher cache covers {cursor} samples but dataset has {len(examples)}"
        )

    total_valid = sum(int(m.get("n_valid", 0)) for m in chunk_metas)

    chunk_hashes = [str(m["sha256"]) for m in chunk_metas]
    cache_content_sha256 = _sha256_bytes(_canonical_json_bytes(chunk_hashes))
    index = {
        "kind": CACHE_INDEX_KIND,
        "kl_mode": KL_MODE_TEACHER_TOPK,
        "metric_name": METRIC_NAME_TEACHER_TOPK,
        "teacher_topk": k,
        "vocab_size": vocab_size,
        "cache_prob_dtype": prob_dtype_name,
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "dataset_file": str(dataset_file.resolve()),
        "dataset_file_sha256": dataset_sha,
        "dataset_manifest": str(manifest_path.resolve()),
        "model_id": profile.model_id,
        "model_path": profile.model_path,
        "sample_count": len(examples),
        "sample_ids": expected_sample_ids,
        "total_valid_positions": int(total_valid),
        "chunks": chunk_metas,
        "cache_dir": str(cache_dir.resolve()),
        "canonical_cache_dir": str(canonical_dir.resolve()),
        "output_dir_override": output_override,
        "cache_content_sha256": cache_content_sha256,
    }
    validate_teacher_cache_against_inputs(
        index,
        expected_sample_ids=expected_sample_ids,
        run_config_sha256=resolved.run_config_sha256,
        model_inventory_fingerprint=inventory.fingerprint_sha256,
        dataset_file_sha256=dataset_sha,
        teacher_topk=k,
        vocab_size=vocab_size,
        cache_prob_dtype=prob_dtype_name,
    )
    _write_json_atomic(index_path, index)
    index["index_path"] = str(index_path.resolve())
    index["index_sha256"] = sha256_file(index_path)
    return index
