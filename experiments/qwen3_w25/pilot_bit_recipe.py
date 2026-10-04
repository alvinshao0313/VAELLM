"""Run one shard of the fixed protected-base packed-bit LR pilot."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.bit_recipe_pilot import run_bit_recipe_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    args = parser.parse_args()
    run_bit_recipe_pilot(args.checkpoint_dir, args.output_dir, shard_index=args.shard_index)


if __name__ == "__main__":
    main()
