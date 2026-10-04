"""Train one isolated matched ChatML/bare-space decoder or decoder/bit arm."""
import argparse

from experiments.qwen3_w25.format_recovery_diagnostic import run_format_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True)
    args = parser.parse_args()
    run_format_recovery(args.launch_record)


if __name__ == "__main__":
    main()
