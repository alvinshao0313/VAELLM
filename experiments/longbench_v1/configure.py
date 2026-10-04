import argparse
from pathlib import Path

from .common import write_json
from .models import make_config


def main():
    parser = argparse.ArgumentParser(description="Prepare immutable dense/compressed LongBench configs; does not launch GPU work.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--compressed-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    for kind in ("dense", "compressed"):
        for mode in ("validation", "full"):
            path = directory / f"{kind}_{mode}.json"
            if path.exists():
                raise FileExistsError(f"Prepared config exists: {path}")
            config = make_config(args.manifest, kind=kind, hf_path=args.hf_model,
                                 checkpoint=args.compressed_model if kind == "compressed" else None, mode=mode)
            write_json(path, config)
            print(path)


if __name__ == "__main__":
    main()
