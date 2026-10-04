"""Write one shard of code-only Qwen3 recovery patches chosen by a completed pilot."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.local_bit_recovery import run_local_bit_recovery_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--pilot_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()
    run_local_bit_recovery_worker(args.checkpoint_dir, args.pilot_json, args.output_dir,
                                  shard_index=args.shard_index, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
