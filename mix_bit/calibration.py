from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from mix_bit.model_adapter import get_model_adapter, normalize_tokenizer_for_mix_bit
from mix_bit.model_inventory import ModelInventory, load_model_inventory, validate_inventory_for_run
from mix_bit.schema import ResolvedRunConfig, sha256_file

InputSchema = Literal["messages", "text", "prompt_response"]

TOKENIZER_FINGERPRINT_VERSION = 2

TOKENIZER_INIT_KWARGS_EXCLUDED: frozenset[str] = frozenset(
    {
        "name_or_path",
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "special_tokens_map_file",
        "tokenizer_config_file",
        "added_tokens_file",
        "cache_dir",
        "local_files_only",
        "revision",
        "token",
        "use_auth_token",
    }
)


@dataclass(frozen=True)
class CalibrationExample:
    sample_id: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None


@dataclass(frozen=True)
class CalibrationDatasetManifest:
    kind: str
    run_config_sha256: str
    model_profile_sha256: str
    candidate_space_sha256: str
    training_recipe_sha256: str
    model_inventory_fingerprint: str
    source_jsonl: str
    source_jsonl_sha256: str
    selected_source_line_ids: tuple[int, ...]
    tokenizer_class: str
    tokenizer_name_or_path: str
    tokenizer_vocab_size: int
    tokenizer_config_sha256: str
    tokenizer_fingerprint_version: int
    pad_token_id: int | None
    eos_token_id: int | None
    pad_token_normalized_from_eos: bool
    input_schema: InputSchema
    sample_count: int
    max_length: int
    seed: int
    label_mode: str
    dataset_file: str
    dataset_file_sha256: str


def build_causal_kl_mask(
    attention_mask: torch.Tensor,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    if labels is not None:
        valid &= labels[:, 1:].ne(-100)
    return valid


def resolve_record_schema(record: dict[str, Any]) -> InputSchema:
    if not isinstance(record, dict):
        raise ValueError(f"Unsupported calibration record type: {type(record)!r}")

    matches: list[InputSchema] = []

    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("Malformed messages record: messages must be a non-empty list")
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Malformed messages entry at index {idx}: expected object")
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"Malformed messages entry at index {idx}: missing role or content"
                )
            role = msg["role"]
            content = msg["content"]
            if not isinstance(role, str) or not role:
                raise ValueError(f"Malformed messages entry at index {idx}: invalid role")
            if not isinstance(content, str):
                raise ValueError(f"Malformed messages entry at index {idx}: content must be str")
        matches.append("messages")

    if "text" in record:
        text = record["text"]
        if not isinstance(text, str) or not text:
            raise ValueError("Unsupported text record: text must be a non-empty string")
        matches.append("text")

    has_prompt_response = ("prompt" in record) or ("response" in record)
    has_instruction_output = ("instruction" in record) or ("output" in record)
    if has_prompt_response or has_instruction_output:
        if has_prompt_response and has_instruction_output:
            raise ValueError(
                "Ambiguous prompt_response record: both prompt/response and "
                "instruction/output fields present"
            )
        if has_prompt_response:
            prompt = record.get("prompt")
            response = record.get("response")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Malformed prompt_response record: prompt must be non-empty str")
            if not isinstance(response, str) or not response:
                raise ValueError(
                    "Malformed prompt_response record: response must be non-empty str"
                )
        else:
            instruction = record.get("instruction")
            output = record.get("output")
            if not isinstance(instruction, str) or not instruction:
                raise ValueError(
                    "Malformed prompt_response record: instruction must be non-empty str"
                )
            if not isinstance(output, str) or not output:
                raise ValueError(
                    "Malformed prompt_response record: output must be non-empty str"
                )
        matches.append("prompt_response")

    if len(matches) == 0:
        raise ValueError(f"Unknown/unsupported calibration record keys: {sorted(record)}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous calibration record matches schemas {matches}")
    return matches[0]


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


def _write_torch_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _iter_jsonl_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at source line {line_index} in {path}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Calibration JSONL line {line_index} must be a JSON object, got {type(obj)!r}"
                )
            records.append((line_index, obj))
    return records


def _messages_from_prompt_response(record: dict[str, Any]) -> list[dict[str, str]]:
    if "prompt" in record or "response" in record:
        return [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    return [
        {"role": "user", "content": record["instruction"]},
        {"role": "assistant", "content": record["output"]},
    ]


def _tokenize_ids(
    tokenizer: Any,
    record: dict[str, Any],
    schema: InputSchema,
    *,
    max_length: int,
    source_line: int,
) -> list[int]:
    if schema == "messages":
        ids = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
    elif schema == "prompt_response":
        ids = tokenizer.apply_chat_template(
            _messages_from_prompt_response(record),
            tokenize=True,
            add_generation_prompt=False,
        )
    elif schema == "text":
        encoded = tokenizer(
            record["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        ids = encoded["input_ids"]
    else:
        raise ValueError(f"Unsupported schema {schema!r}")

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Empty tokenization at source line {source_line}")
    if any(not isinstance(x, int) for x in ids):
        # some tokenizers nest batch dim
        if len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]
        else:
            raise ValueError(
                f"Unexpected tokenization output at source line {source_line}: {type(ids)!r}"
            )
    if not ids:
        raise ValueError(f"Empty tokenization at source line {source_line}")
    if schema in {"messages", "prompt_response"} and len(ids) > max_length:
        ids = ids[:max_length]
    if not ids:
        raise ValueError(f"Empty tokenization after truncate at source line {source_line}")
    return ids


def _normalize_json_value(value: Any) -> Any:
    """Recursively normalize a value into deterministic JSON-serializable form.

    Only null/bool/int/finite float/str/list/tuple/dict-with-stringified-keys are
    allowed. ``Path`` is converted to str. Hugging Face ``AddedToken`` (or any
    object exposing a ``content`` attribute) is serialized as a fixed-shape dict.
    Other objects are recorded as ``{"unsupported_type": "ClassName"}`` so no
    memory address leaks into the digest.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float value in tokenizer payload: {value!r}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(val) for key, val in value.items()}
    # AddedToken-like objects expose a `content` attribute (the token string).
    if hasattr(value, "content") and not callable(value) and not isinstance(value, type):
        return {
            "content": _normalize_json_value(getattr(value, "content", "")),
            "single_word": bool(getattr(value, "single_word", False)),
            "lstrip": bool(getattr(value, "lstrip", False)),
            "rstrip": bool(getattr(value, "rstrip", False)),
            "normalized": bool(getattr(value, "normalized", True)),
            "special": bool(getattr(value, "special", False)),
        }
    return {"unsupported_type": type(value).__name__}


def _is_empty_extra_special_tokens(value: Any) -> bool:
    """HF save_pretrained writes extra_special_tokens={} even when hub omitted the key."""
    return value in (None, {}, [])


def _stable_init_kwargs(tokenizer: Any) -> dict[str, Any]:
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, dict):
        return {}
    stable: dict[str, Any] = {}
    for key, value in init_kwargs.items():
        if key in TOKENIZER_INIT_KWARGS_EXCLUDED:
            continue
        normalized = _normalize_json_value(value)
        if key == "extra_special_tokens" and _is_empty_extra_special_tokens(normalized):
            continue
        stable[key] = normalized
    return stable


def _core_tokenizer_bytes(tokenizer: Any) -> tuple[str, bytes]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        return "backend_tokenizer_json", backend.to_str().encode("utf-8")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError(
            "Tokenizer exposes neither backend_tokenizer.to_str nor get_vocab"
        )
    vocab_items = sorted(
        (str(token_text), int(token_id))
        for token_text, token_id in get_vocab().items()
    )
    return "sorted_vocab", _canonical_json_bytes(vocab_items)


def _special_tokens_map(tokenizer: Any) -> dict[str, Any]:
    raw = getattr(tokenizer, "special_tokens_map", None)
    if not isinstance(raw, dict):
        return {}
    return {str(key): _normalize_json_value(val) for key, val in raw.items()}


def _added_vocab(tokenizer: Any) -> list[list[Any]]:
    added = getattr(tokenizer, "get_added_vocab", None)
    if not callable(added):
        return []
    raw = added()
    if not isinstance(raw, dict):
        return []
    items = sorted(
        (str(text), _normalize_json_value(tid)) for text, tid in raw.items()
    )
    return [[text, tid] for text, tid in items]


def build_tokenizer_fingerprint_payload(tokenizer: Any) -> dict[str, Any]:
    """Return versioned provenance plus content fields used by the digest."""
    core_kind, core_bytes = _core_tokenizer_bytes(tokenizer)
    core_sha = _sha256_bytes(core_bytes)
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, (list, tuple)):
        chat_template_normalized: Any = [_normalize_json_value(item) for item in chat_template]
    else:
        chat_template_normalized = _normalize_json_value(chat_template)
    content = {
        "class_name": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size")),
        "model_max_length": _normalize_json_value(getattr(tokenizer, "model_max_length", None)),
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
        "bos_token_id": _normalize_json_value(getattr(tokenizer, "bos_token_id", None)),
        "eos_token_id": _normalize_json_value(getattr(tokenizer, "eos_token_id", None)),
        "pad_token_id": _normalize_json_value(getattr(tokenizer, "pad_token_id", None)),
        "unk_token_id": _normalize_json_value(getattr(tokenizer, "unk_token_id", None)),
        "chat_template": chat_template_normalized,
        "special_tokens_map": _special_tokens_map(tokenizer),
        "added_vocab": _added_vocab(tokenizer),
        "core_kind": core_kind,
        "core_sha256": core_sha,
        "stable_init_kwargs": _stable_init_kwargs(tokenizer),
    }
    return {
        "version": TOKENIZER_FINGERPRINT_VERSION,
        "reported_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "content": content,
    }


def compute_tokenizer_config_sha256(tokenizer: Any) -> str:
    """Hash only version and content; reported path is provenance-only."""
    payload = build_tokenizer_fingerprint_payload(tokenizer)
    digest_payload = {
        "version": payload["version"],
        "content": payload["content"],
    }
    return _sha256_bytes(_canonical_json_bytes(digest_payload))


def _example_to_dict(example: CalibrationExample) -> dict[str, Any]:
    return {
        "attention_mask": example.attention_mask.detach().cpu().contiguous(),
        "input_ids": example.input_ids.detach().cpu().contiguous(),
        "labels": None
        if example.labels is None
        else example.labels.detach().cpu().contiguous(),
        "sample_id": int(example.sample_id),
    }


def _manifest_to_dict(manifest: CalibrationDatasetManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["selected_source_line_ids"] = list(manifest.selected_source_line_ids)
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _resolve_model_seqlen(
    profile,
    *,
    access_token: str | None = None,
    seqlen: int | None = None,
    model: Any | None = None,
) -> int:
    """Return production ``model.seqlen`` (set by rotation loaders / adapter.load_model)."""
    if seqlen is not None:
        value_i = int(seqlen)
        if value_i <= 0:
            raise ValueError(f"model.seqlen must be positive, got {value_i}")
        return value_i

    owned_model = False
    if model is None:
        adapter = get_model_adapter(profile.adapter)
        model = adapter.load_model(profile, access_token=access_token)
        owned_model = True
    try:
        if not hasattr(model, "seqlen"):
            raise ValueError(
                f"Loaded model for {profile.model_path!r} has no seqlen attribute; "
                "production loaders must set model.seqlen"
            )
        value_i = int(model.seqlen)
        if value_i <= 0:
            raise ValueError(
                f"Loaded model for {profile.model_path!r} has non-positive seqlen={value_i}"
            )
        return value_i
    finally:
        if owned_model:
            del model
            gc.collect()


def _assert_resume_compatible(
    existing: dict[str, Any],
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    source_sha: str,
    tokenizer_sha: str,
    seed: int,
    max_samples: int,
    max_length: int,
    schema: InputSchema,
    label_mode: str,
) -> None:
    checks = {
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "source_jsonl_sha256": source_sha,
        "tokenizer_config_sha256": tokenizer_sha,
        "tokenizer_fingerprint_version": TOKENIZER_FINGERPRINT_VERSION,
        "seed": seed,
        "sample_count": max_samples,
        "max_length": max_length,
        "input_schema": schema,
        "label_mode": label_mode,
    }
    for key, expected in checks.items():
        found = existing.get(key)
        if found != expected:
            if key == "tokenizer_fingerprint_version" and found is None:
                raise ValueError(
                    "Calibration resume rejected: legacy manifest missing "
                    "tokenizer_fingerprint_version; regenerate calibration "
                    "with the current tokenizer fingerprint v2 implementation "
                    "(pass --overwrite to rebuild)."
                )
            raise ValueError(
                f"Calibration resume mismatch for {key}: existing={found!r} current={expected!r}. "
                "Pass --overwrite to rebuild."
            )


def prepare_calibration_dataset(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory | str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    access_token: str | None = None,
    tokenizer: Any | None = None,
    seqlen: int | None = None,
    model: Any | None = None,
) -> tuple[list[CalibrationExample], CalibrationDatasetManifest]:
    if isinstance(inventory, (str, Path)):
        inventory_obj = load_model_inventory(str(inventory))
    else:
        inventory_obj = inventory
    validate_inventory_for_run(inventory_obj, resolved)

    calib = resolved.config.calibration
    source_path = Path(calib.source_jsonl)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing calibration JSONL: {source_path}")

    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(resolved.canonical_run_root) / "calibration"
    )
    dataset_path = out_dir / "dataset.pt"
    manifest_path = out_dir / "dataset_manifest.json"

    source_sha = sha256_file(source_path)
    records = _iter_jsonl_records(source_path)
    if not records:
        raise ValueError(f"No valid JSONL records in {source_path}")

    configured_format = calib.input_format
    if configured_format == "auto":
        selected_schema = resolve_record_schema(records[0][1])
    else:
        selected_schema = configured_format
        first_schema = resolve_record_schema(records[0][1])
        if first_schema != selected_schema:
            raise ValueError(
                f"Configured input_format={selected_schema!r} but first record resolves to "
                f"{first_schema!r} at source line {records[0][0]}"
            )

    for line_index, record in records:
        schema = resolve_record_schema(record)
        if schema != selected_schema:
            raise ValueError(
                f"Mixed calibration schemas: selected={selected_schema!r} but source line "
                f"{line_index} resolves to {schema!r}"
            )

    if calib.max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {calib.max_samples}")
    if len(records) < calib.max_samples:
        raise ValueError(
            f"Requested max_samples={calib.max_samples} but only {len(records)} valid records "
            f"in {source_path}"
        )

    line_indices = [line_index for line_index, _ in records]
    rng = random.Random(calib.seed)
    rng.shuffle(line_indices)
    selected_line_ids = line_indices[: calib.max_samples]
    record_by_line = {line_index: record for line_index, record in records}

    profile = resolved.config.model_profile
    pad_normalized = False
    if tokenizer is None:
        adapter = get_model_adapter(profile.adapter)
        tokenizer = adapter.load_tokenizer(profile, access_token=access_token)
        pad_normalized = bool(getattr(tokenizer, "mix_bit_pad_token_normalized_from_eos", False))
    else:
        normalize_tokenizer_for_mix_bit(tokenizer, source_label=str(getattr(tokenizer, "name_or_path", "")))
        pad_normalized = bool(getattr(tokenizer, "mix_bit_pad_token_normalized_from_eos", False))

    tokenizer_sha = compute_tokenizer_config_sha256(tokenizer)

    effective_seqlen = _resolve_model_seqlen(
        profile,
        access_token=access_token,
        seqlen=seqlen,
        model=model,
    )
    if calib.max_length > effective_seqlen:
        raise ValueError(
            f"calibration.max_length={calib.max_length} exceeds model seqlen={effective_seqlen}"
        )

    if dataset_path.is_file() and manifest_path.is_file() and not overwrite:
        existing = _load_manifest(manifest_path)
        _assert_resume_compatible(
            existing,
            resolved=resolved,
            inventory=inventory_obj,
            source_sha=source_sha,
            tokenizer_sha=tokenizer_sha,
            seed=calib.seed,
            max_samples=calib.max_samples,
            max_length=calib.max_length,
            schema=selected_schema,
            label_mode=calib.label_mode,
        )
        current_dataset_sha = sha256_file(dataset_path)
        if existing.get("dataset_file_sha256") != current_dataset_sha:
            raise ValueError(
                "Calibration resume mismatch for dataset_file_sha256: "
                f"existing={existing.get('dataset_file_sha256')!r} "
                f"current={current_dataset_sha!r}. Pass --overwrite to rebuild."
            )
        loaded = torch.load(dataset_path, map_location="cpu", weights_only=False)
        examples = [
            CalibrationExample(
                sample_id=int(item["sample_id"]),
                input_ids=item["input_ids"],
                attention_mask=item["attention_mask"],
                labels=item.get("labels"),
            )
            for item in loaded
        ]
        return examples, CalibrationDatasetManifest(**{
            **existing,
            "selected_source_line_ids": tuple(existing["selected_source_line_ids"]),
        })

    if (dataset_path.is_file() or manifest_path.is_file()) and not overwrite:
        raise ValueError(
            f"Incomplete or stale calibration outputs under {out_dir}. Pass --overwrite to rebuild."
        )

    examples: list[CalibrationExample] = []
    for sample_id in selected_line_ids:
        record = record_by_line[sample_id]
        ids = _tokenize_ids(
            tokenizer,
            record,
            selected_schema,
            max_length=calib.max_length,
            source_line=sample_id,
        )
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels: torch.Tensor | None
        if calib.label_mode == "all_nonpad":
            labels = input_ids.clone()
            labels = labels.masked_fill(attention_mask.eq(0), -100)
        else:
            raise ValueError(f"Unsupported label_mode: {calib.label_mode!r}")

        batched_mask = build_causal_kl_mask(attention_mask.unsqueeze(0), labels.unsqueeze(0))
        if int(batched_mask.sum().item()) < 1:
            raise ValueError(
                f"Calibration sample at source line {sample_id} has no valid causal tokens"
            )

        examples.append(
            CalibrationExample(
                sample_id=int(sample_id),
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        )

    payload = [_example_to_dict(example) for example in examples]
    _write_torch_atomic(dataset_path, payload)
    dataset_sha = sha256_file(dataset_path)

    manifest = CalibrationDatasetManifest(
        kind="mix_bit_calibration_dataset_manifest",
        run_config_sha256=resolved.run_config_sha256,
        model_profile_sha256=resolved.model_profile_sha256,
        candidate_space_sha256=resolved.candidate_space_sha256,
        training_recipe_sha256=resolved.training_recipe_sha256,
        model_inventory_fingerprint=inventory_obj.fingerprint_sha256,
        source_jsonl=str(source_path.resolve()),
        source_jsonl_sha256=source_sha,
        selected_source_line_ids=tuple(int(x) for x in selected_line_ids),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_name_or_path=str(getattr(tokenizer, "name_or_path", "")),
        tokenizer_vocab_size=int(tokenizer.vocab_size),
        tokenizer_config_sha256=tokenizer_sha,
        tokenizer_fingerprint_version=TOKENIZER_FINGERPRINT_VERSION,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_normalized_from_eos=pad_normalized,
        input_schema=selected_schema,
        sample_count=len(examples),
        max_length=int(calib.max_length),
        seed=int(calib.seed),
        label_mode=str(calib.label_mode),
        dataset_file=str(dataset_path.resolve()),
        dataset_file_sha256=dataset_sha,
    )
    _write_json_atomic(manifest_path, _manifest_to_dict(manifest))
    return examples, manifest
