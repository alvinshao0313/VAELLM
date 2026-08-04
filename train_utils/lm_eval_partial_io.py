"""File-based exchange of distributed lm-eval partial results.

Temporary files live under ``{run_output_dir}/lm_eval/.partial_{tag}/`` and are
deleted after rank0 finishes merging. Avoids NCCL ``gather_object`` waits when
task shards (e.g. mmlu) are highly unbalanced.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional


_DEFAULT_PARTIAL_WAIT_SEC = 10800
_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def resolve_lm_eval_partial_wait_sec(timeout_sec: Optional[int] = None) -> int:
    if timeout_sec is not None:
        resolved = int(timeout_sec)
    else:
        raw = str(os.environ.get("LM_EVAL_PARTIAL_WAIT_SEC", str(_DEFAULT_PARTIAL_WAIT_SEC))).strip()
        try:
            resolved = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"LM_EVAL_PARTIAL_WAIT_SEC must be an integer number of seconds, got {raw!r}."
            ) from exc
    if resolved <= 0:
        raise ValueError(f"partial wait timeout must be > 0, got {resolved}.")
    return int(resolved)


def _safe_tag(tag: str) -> str:
    text = str(tag).strip() or "eval"
    text = text.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    text = _SAFE_TAG_RE.sub("_", text)
    return text or "eval"


def lm_eval_partial_dir(run_output_dir: str, tag: str) -> str:
    root = str(run_output_dir).strip()
    if not root:
        raise ValueError("run_output_dir must be a non-empty path.")
    return os.path.join(root, "lm_eval", f".partial_{_safe_tag(tag)}")


def _rank_json_path(partial_dir: str, rank: int) -> str:
    return os.path.join(partial_dir, f"rank_{int(rank)}.json")


def _rank_done_path(partial_dir: str, rank: int) -> str:
    return os.path.join(partial_dir, f"rank_{int(rank)}.done")


def prepare_lm_eval_partial_dir(partial_dir: str) -> None:
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir)
    elif os.path.exists(partial_dir):
        os.remove(partial_dir)
    os.makedirs(partial_dir, exist_ok=True)


def cleanup_lm_eval_partial_dir(partial_dir: str) -> None:
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir, ignore_errors=True)
    elif os.path.exists(partial_dir):
        try:
            os.remove(partial_dir)
        except OSError:
            pass


def write_lm_eval_partial_result(
    partial_dir: str,
    *,
    rank: int,
    payload: Dict[str, Any],
) -> None:
    os.makedirs(partial_dir, exist_ok=True)
    json_path = _rank_json_path(partial_dir, rank)
    tmp_path = json_path + ".tmp"
    done_path = _rank_done_path(partial_dir, rank)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)
    with open(done_path, "w", encoding="utf-8") as handle:
        handle.write("ok\n")


def wait_and_load_lm_eval_partial_results(
    partial_dir: str,
    *,
    world_size: int,
    timeout_sec: Optional[int] = None,
    poll_interval_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    world = int(world_size)
    if world <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}.")
    wait_sec = resolve_lm_eval_partial_wait_sec(timeout_sec)
    poll = max(0.1, float(poll_interval_sec))
    deadline = time.monotonic() + float(wait_sec)

    while True:
        missing = [
            idx for idx in range(world) if not os.path.isfile(_rank_done_path(partial_dir, idx))
        ]
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {wait_sec}s waiting for lm-eval partial results under "
                f"{partial_dir}. Missing done markers for ranks: "
                + ",".join(str(idx) for idx in missing)
            )
        time.sleep(poll)

    gathered: List[Dict[str, Any]] = []
    for idx in range(world):
        json_path = _rank_json_path(partial_dir, idx)
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"lm-eval partial done marker exists but json missing for rank={idx}: {json_path}"
            )
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(
                f"lm-eval partial for rank={idx} must be a dict, got {type(payload).__name__}."
            )
        gathered.append(payload)
    return gathered


def exchange_lm_eval_partial_via_files(
    partial_result: Dict[str, Any],
    *,
    run_output_dir: str,
    tag: str,
    rank: int,
    world_size: int,
    is_main: bool,
    timeout_sec: Optional[int] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Write local partial; main rank waits/loads all ranks and returns gathered list.

    Cleanup is the caller's responsibility after merge (use ``cleanup_lm_eval_partial_dir``).
    """
    if not isinstance(partial_result, dict):
        raise TypeError(f"partial_result must be a dict, got {type(partial_result).__name__}.")
    partial_dir = lm_eval_partial_dir(run_output_dir, tag)
    write_lm_eval_partial_result(partial_dir, rank=int(rank), payload=partial_result)
    if not bool(is_main):
        return None
    return wait_and_load_lm_eval_partial_results(
        partial_dir,
        world_size=int(world_size),
        timeout_sec=timeout_sec,
    )
