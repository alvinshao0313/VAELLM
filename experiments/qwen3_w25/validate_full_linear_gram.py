"""Compare streaming and full-Gram objectives on the fixed protected Qwen3 source."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.gram_validation import run_gram_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint_dir", "pilot_json", "output_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run_gram_validation(args.checkpoint_dir, args.pilot_json, args.output_dir)


if __name__ == "__main__":
    main()
