import argparse

from .data import download_and_extract, prepare


def main():
    parser = argparse.ArgumentParser(description="Prepare pinned full LongBench v1 shared token inputs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="/home/star/models/Qwen/Qwen3-8B")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--resume-download", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        print(download_and_extract(args.output, resume_download=args.resume_download))
    else:
        record = prepare(args.output, args.tokenizer, resume_download=args.resume_download)
        print(f"Prepared {record['sample_count']} samples; {record['inputs']['tokens']} shared tokens.")


if __name__ == "__main__":
    main()
