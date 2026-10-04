"""Validate one joint-code/decoder Qwen3 block using fixed original-teacher inputs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen3_w25.joint_block_validation import run_joint_block_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint_dir", "pilot_json", "output_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--train_documents", type=int, choices=(64, 512), default=64,
                        help="Use this prefix of the fixed 512-document training pool; the public held-out 64 stay fixed.")
    args = parser.parse_args()
    run_joint_block_validation(args.checkpoint_dir, args.pilot_json, args.output_dir,
                               train_documents=args.train_documents)


if __name__ == "__main__":
    main()
