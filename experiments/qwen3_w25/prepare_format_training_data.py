"""Prepare the fixed, shared ChatML/bare-space task recovery archive on CPU."""
import argparse
import json

from experiments.qwen3_w25.format_training_data import prepare_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--hf_source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = prepare_archive(**vars(args))
    print(json.dumps({key: report[key] for key in ("status", "archive", "audit", "elapsed_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
