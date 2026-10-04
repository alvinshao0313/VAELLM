"""Run one protected-base MLP packed-bit LR/budget diagnostic."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.protected_bit_focus import TARGETS, run_protected_bit_focus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--pilot_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    args = parser.parse_args()
    run_protected_bit_focus(args.checkpoint_dir, args.pilot_json, args.output_dir, target=TARGETS[args.target])


if __name__ == "__main__":
    main()
