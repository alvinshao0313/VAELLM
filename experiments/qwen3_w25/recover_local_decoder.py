"""Run one local decoder recovery shard, or finalize explicitly supplied shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker", help="Run one independent GPU worker; cuda:0 is selected by the caller.")
    worker.add_argument("--checkpoint_dir", type=Path, required=True)
    worker.add_argument("--output_dir", type=Path, required=True)
    worker.add_argument("--shard_index", type=int, required=True)
    worker.add_argument("--num_shards", type=int, required=True)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument("--lr", type=float, required=True)
    finalize = commands.add_parser("finalize", help="Verify all workers and create one new complete v6 final on CPU.")
    finalize.add_argument("--checkpoint_dir", type=Path, required=True)
    finalize.add_argument("--shard_dirs", type=Path, nargs="+", required=True)
    finalize.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        from experiments.qwen3_w25.local_recovery import run_local_recovery_worker
        report = run_local_recovery_worker(
            args.checkpoint_dir, args.output_dir, shard_index=args.shard_index,
            num_shards=args.num_shards, steps=args.steps, lr=args.lr,
        )
        print(json.dumps({"status": report["status"], "completed_modules": len(report["modules"]),
                          "report": str(args.output_dir.resolve() / "worker.json")}), flush=True)
    else:
        from experiments.qwen3_w25.local_recovery_finalize import finalize_local_recovery
        report = finalize_local_recovery(args.checkpoint_dir, args.shard_dirs, args.output_dir)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
