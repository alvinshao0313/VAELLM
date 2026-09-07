import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from train_utils.distributed_guard import DistributedRankError, distributed_guarded_all


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = int(dist.get_rank())

    value = distributed_guarded_all(lambda: rank + 100, barrier=True)
    if value != rank + 100:
        raise RuntimeError(f"rank{rank}: local guarded-all result mismatch: {value}")

    def _maybe_fail():
        if rank == 1:
            raise ValueError("intentional-nccl-rank1-failure")
        return rank

    try:
        distributed_guarded_all(_maybe_fail, barrier=False)
    except DistributedRankError as exc:
        message = str(exc)
        if "rank 1" not in message or "intentional-nccl-rank1-failure" not in message:
            raise RuntimeError(f"rank{rank}: unexpected propagated error: {message}") from exc
    else:
        raise RuntimeError(f"rank{rank}: expected DistributedRankError was not raised")

    if rank == 0:
        print("DISTRIBUTED_GUARDED_ALL_NCCL_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
