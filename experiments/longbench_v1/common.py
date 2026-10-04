from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_stat(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino}


def check_stat(record):
    if file_stat(record["path"]) != record:
        raise ValueError(f"Input stat changed: {record['path']}")


def check_hashes(pins):
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError(f"Pinned file changed: {path}")


def source_pins():
    """Bind this runner and the existing loader/cache implementation it uses."""
    paths = {ROOT / "tools/cat_eval.py"}
    for directory in ("experiments/longbench_v1", "train_utils", "litebsq",
                      "e2e_common", "rotation", "sparse_bit_tuning"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update((ROOT / "experiments/longbench_v1/upstream").glob("*.json"))
    paths.add(ROOT / "experiments/longbench_v1/upstream/LICENSE")
    return {str(path): sha256(path) for path in sorted(paths)}
