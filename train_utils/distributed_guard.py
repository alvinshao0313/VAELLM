"""Failure-safe execution for rank-zero operations followed by collectives."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

import torch


T = TypeVar("T")


class DistributedMainError(RuntimeError):
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


__all__ = ["DistributedMainError", "distributed_guarded_main"]
