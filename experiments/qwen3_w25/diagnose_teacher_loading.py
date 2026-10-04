"""Observe the fixed Qwen3 teacher loader A/B pair without training."""
import argparse

from experiments.qwen3_w25.teacher_loading_diagnostic import run_teacher_loading_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True)
    args = parser.parse_args()
    run_teacher_loading_diagnostic(args.launch_record)


if __name__ == "__main__":
    main()
