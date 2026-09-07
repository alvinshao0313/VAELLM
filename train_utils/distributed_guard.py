"""Failure-safe execution for rank-zero operations followed by collectives."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

import torch


T = TypeVar("T")


class DistributedMainError(RuntimeError):
    pass


class DistributedRankError(RuntimeError):
    pass


def distributed_guarded_main(
    operation: Callable[[], T],
    *,
    main_rank: int = 0,
    barrier: bool = False,
) -> Optional[T]:
    """Run ``operation`` on one rank and propagate success/failure to every rank."""
    distributed = bool(torch.distributed.is_available() and torch.distributed.is_initialized())
    if not distributed:
        return operation()

    rank = int(torch.distributed.get_rank())
    status = None
    if rank == int(main_rank):
        try:
            status = {"ok": True, "result": operation()}
        except Exception as exc:
            status = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    payload = [status]
    torch.distributed.broadcast_object_list(payload, src=int(main_rank))
    resolved = payload[0]
    if not isinstance(resolved, dict) or not isinstance(resolved.get("ok"), bool):
        raise DistributedMainError("rank-zero operation broadcast an invalid status payload")
    if not resolved["ok"]:
        raise DistributedMainError(
            "rank-zero operation failed: "
            f"{resolved.get('error_type', 'Exception')}: {resolved.get('error_message', '')}"
        )
    if barrier:
        torch.distributed.barrier()
    return resolved.get("result")


def distributed_guarded_all(
    operation: Callable[[], T],
    *,
    barrier: bool = False,
) -> T:
    """Run a local operation on every rank and propagate any rank-local failure to all ranks."""
    distributed = bool(torch.distributed.is_available() and torch.distributed.is_initialized())
    if not distributed:
        return operation()

    rank = int(torch.distributed.get_rank())
    result = None
    status = {"rank": rank, "ok": True, "error_type": None, "error_message": None}
    try:
        result = operation()
    except Exception as exc:
        status = {
            "rank": rank,
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    statuses = [None for _ in range(int(torch.distributed.get_world_size()))]
    torch.distributed.all_gather_object(statuses, status)
    failures = [item for item in statuses if isinstance(item, dict) and item.get("ok") is False]
    invalid = [item for item in statuses if not isinstance(item, dict) or not isinstance(item.get("ok"), bool)]
    if invalid:
        raise DistributedRankError("rank-local operation gathered an invalid status payload")
    if failures:
        details = "; ".join(
            f"rank {item.get('rank')}: {item.get('error_type', 'Exception')}: {item.get('error_message', '')}"
            for item in failures
        )
        raise DistributedRankError(f"rank-local operation failed: {details}")
    if barrier:
        torch.distributed.barrier()
    return result


__all__ = [
    "DistributedMainError",
    "DistributedRankError",
    "distributed_guarded_all",
    "distributed_guarded_main",
]
