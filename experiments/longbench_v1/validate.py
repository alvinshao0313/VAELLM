import argparse
import json
from pathlib import Path

from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Three real input geometry cases; workflow/resource evidence only, never benchmark scores.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if json.loads(Path(args.config).read_text())["mode"] != "validation":
        raise ValueError("The short validation entry requires a validation configuration.")
    report = run(args.config, args.output)
    print(json.dumps({k: report[k] for k in ("status", "new_predictions", "duration_seconds")}))


if __name__ == "__main__":
    main()
