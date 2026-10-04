"""Run the fixed 32+32-document, 128-bit-per-stage gradient stability diagnostic."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.bit_gradient_accumulation_diagnostic import run_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prepared_json", type=Path, required=True)
    args = parser.parse_args()
    run_diagnostic(args.checkpoint_dir, args.output_dir, args.prepared_json)


if __name__ == "__main__":
    main()

