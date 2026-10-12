"""CPU-only scoring of the fixed 12-task sensitive LongBench screen."""
import argparse
import json

from .core import score_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = score_directory(args.selection, args.archive, args.prediction_dir, args.output)
    print(json.dumps({key: report[key] for key in
                      ("status", "is_full_benchmark", "task_count", "sample_count", "high_macro_mean", "duration_seconds")}))


if __name__ == "__main__":
    main()
