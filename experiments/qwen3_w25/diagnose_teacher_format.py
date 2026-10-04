"""Compare ChatML and explicitly space-separated bare answers with one real teacher."""
import argparse

from experiments.qwen3_w25.teacher_format_diagnostic import run_teacher_format_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_record", required=True)
    args = parser.parse_args()
    run_teacher_format_diagnostic(args.launch_record)


if __name__ == "__main__":
    main()
