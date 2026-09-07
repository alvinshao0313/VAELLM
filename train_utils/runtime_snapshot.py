from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Dict


_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "auth_token",
        "hf_token",
        "api_token",
        "api_key",
        "password",
        "secret",
        "secret_key",
    }
)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower()
    if normalized in _SENSITIVE_KEYS or normalized == "token":
        return True
    return normalized.endswith(("_token", "_api_key", "_password", "_secret", "_secret_key"))


def to_runtime_jsonable(value: Any) -> Any:
    """Convert runtime/config objects to deterministic JSON-safe data.

    Private namespace fields and callable adapter helpers are intentionally
    omitted. Credential-like fields are omitted rather than redacted so a
    saved experiment snapshot can be shared without leaking secrets.
    """

    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return to_runtime_jsonable(value.to_jsonable())
    if isinstance(value, Enum):
        return to_runtime_jsonable(value.value)
    if is_dataclass(value):
        out: Dict[str, Any] = {}
        for field in fields(value):
            if _is_sensitive_key(field.name):
                continue
            item = getattr(value, field.name)
            if callable(item):
                continue
            out[field.name] = to_runtime_jsonable(item)
        return out
    if isinstance(value, argparse.Namespace):
        out = {}
        for key, item in vars(value).items():
            if str(key).startswith("_") or _is_sensitive_key(key) or callable(item):
                continue
            out[str(key)] = to_runtime_jsonable(item)
        return out
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if _is_sensitive_key(key) or callable(item):
                continue
            out[str(key)] = to_runtime_jsonable(item)
        return out
    if isinstance(value, (set, frozenset)):
        return [to_runtime_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [to_runtime_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    return str(value)


def format_runtime_snapshot(payload: object) -> str:
    return json.dumps(
        to_runtime_jsonable(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def write_runtime_snapshot(path: str, payload: object) -> str:
    normalized_path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
    with open(normalized_path, "w", encoding="utf-8") as handle:
        json.dump(
            to_runtime_jsonable(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return normalized_path


__all__ = ["format_runtime_snapshot", "to_runtime_jsonable", "write_runtime_snapshot"]
