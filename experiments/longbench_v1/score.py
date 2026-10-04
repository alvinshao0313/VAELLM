import argparse
from .scoring import score_directory


def main():
    parser = argparse.ArgumentParser(description="Score all 21 official LongBench v1 tasks, without partial averages.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = score_directory(args.manifest, args.prediction_dir, args.output)
    print(f"4750/4750; task macro={report['task_macro_mean']:.8f}; category macro={report['category_macro_mean']:.8f}")


if __name__ == "__main__":
    main()
