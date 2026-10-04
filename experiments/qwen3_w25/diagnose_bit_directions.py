"""Run the bounded 552282 gate/down finite-bit direction diagnostic."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.bit_direction_diagnostic import run_bit_direction_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    run_bit_direction_diagnostic(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()

