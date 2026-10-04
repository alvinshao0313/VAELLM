"""Run one fixed teacher-input joint shard, or strictly assemble all four shards."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    finalize = commands.add_parser("finalize")
    for command in (worker, finalize):
        command.add_argument("--checkpoint_dir", type=Path, required=True)
        command.add_argument("--output_dir", type=Path, required=True)
    worker.add_argument("--pilot_json", type=Path, required=True)
    worker.add_argument("--shard_index", type=int, required=True, choices=range(4))
    finalize.add_argument("--shard_dirs", type=Path, nargs=4, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        from experiments.qwen3_w25.joint_block_recovery import run_joint_block_recovery
        run_joint_block_recovery(args.checkpoint_dir, args.pilot_json, args.output_dir, shard_index=args.shard_index)
    else:
        from experiments.qwen3_w25.joint_block_recovery_finalize import finalize_joint_block_recovery
        finalize_joint_block_recovery(args.checkpoint_dir, args.shard_dirs, args.output_dir)


if __name__ == "__main__":
    main()
