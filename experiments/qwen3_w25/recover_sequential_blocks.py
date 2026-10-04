"""Run fixed-budget sequential local recovery of all 36 Qwen3 blocks."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.sequential_recovery import run_sequential_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_documents", type=int, choices=(64, 512), required=True)
    parser.add_argument("--steps", type=int, choices=(64, 192), required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    run_sequential_recovery(args.checkpoint_dir, args.output_dir,
                            train_documents=args.train_documents, steps=args.steps, lr=args.lr)


if __name__ == "__main__":
    main()
