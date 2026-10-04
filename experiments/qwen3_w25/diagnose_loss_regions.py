"""Observe existing regional KL/CE on five bound BF16 deployment checkpoints."""
import argparse

from experiments.qwen3_w25.loss_region_diagnostic import run_loss_region_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True)
    args = parser.parse_args()
    run_loss_region_diagnostic(args.launch_record)


if __name__ == "__main__":
    main()
