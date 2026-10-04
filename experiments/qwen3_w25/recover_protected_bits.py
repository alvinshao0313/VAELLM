"""Write one protected-base code-only recovery shard from a completed LR pilot."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.protected_bit_recovery import run_protected_bit_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint_dir", "pilot_json", "output_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--focused_summary", type=Path,
                        help="Opt into the independently selected attention/MLP recipes; omit for the uniform pilot recipe.")
    args = parser.parse_args()
    run_protected_bit_recovery(args.checkpoint_dir, args.pilot_json, args.output_dir,
                              shard_index=args.shard_index, num_shards=args.num_shards,
                              focused_summary=args.focused_summary)


if __name__ == "__main__":
    main()
