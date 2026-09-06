from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp

from train_utils.distributed_guard import DistributedMainError, distributed_guarded_main


def _worker(rank: int, world_size: int, init_file: str, fail: bool, queue) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        def _operation():
            if fail:
                raise ValueError("intentional-main-failure")
            return {"value": 7}

        try:
            result = distributed_guarded_main(_operation, barrier=True)
            queue.put((rank, "ok", result))
        except Exception as exc:
            queue.put((rank, type(exc).__name__, str(exc)))
    finally:
        torch.distributed.destroy_process_group()


def _run_two_process(tmp_path, *, fail: bool):
    init_file = str(tmp_path / ("fail.init" if fail else "success.init"))
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(_worker, args=(2, init_file, fail, queue), nprocs=2, join=True)
    return sorted(queue.get() for _ in range(2))


def test_distributed_guard_single_process_preserves_result_and_exception():
    assert distributed_guarded_main(lambda: 3) == 3
    with pytest.raises(ValueError, match="single-failure"):
        distributed_guarded_main(lambda: (_ for _ in ()).throw(ValueError("single-failure")))


def test_distributed_guard_two_process_success(tmp_path):
    rows = _run_two_process(tmp_path, fail=False)
    assert rows == [(0, "ok", {"value": 7}), (1, "ok", {"value": 7})]


def test_distributed_guard_two_process_failure_reaches_all_ranks(tmp_path):
    rows = _run_two_process(tmp_path, fail=True)
    assert [row[1] for row in rows] == ["DistributedMainError", "DistributedMainError"]
    assert all("intentional-main-failure" in row[2] for row in rows)
