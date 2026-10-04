import argparse
import json
from pathlib import Path

from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Predict complete LongBench v1 dataset partitions, with no sample limits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float)
    args = parser.parse_args()
    if json.loads(Path(args.config).read_text())["mode"] != "full":
        raise ValueError("The formal entry requires a full configuration.")
    report = run(args.config, args.output, datasets=args.datasets, resume=args.resume,
                 max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps({k: report[k] for k in ("status", "new_predictions", "already_present", "duration_seconds")}))


if __name__ == "__main__":
    main()
