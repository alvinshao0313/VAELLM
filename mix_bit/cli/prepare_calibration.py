from __future__ import annotations

import argparse
from pathlib import Path

from mix_bit.calibration import prepare_calibration_dataset
from mix_bit.schema import resolve_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic KL calibration dataset")
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument(
        "--inventory",
        required=True,
        help="Path to model_inventory.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional calibration output directory (default: <run_root>/calibration)",
    )
    parser.add_argument(
        "--access_token",
        default=None,
        help="Optional Hugging Face access token",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild when existing calibration hashes do not match",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resolved = resolve_run_config(args.run_config, write=True)
    examples, manifest = prepare_calibration_dataset(
        resolved,
        args.inventory,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
        access_token=args.access_token,
    )

    manifest_path = Path(manifest.dataset_file).with_name("dataset_manifest.json")
    print(f"sample_count={manifest.sample_count}")
    print(f"input_schema={manifest.input_schema}")
    print(f"seed={manifest.seed}")
    print(f"source_jsonl_sha256={manifest.source_jsonl_sha256}")
    print(f"tokenizer_config_sha256={manifest.tokenizer_config_sha256}")
    print(f"pad_token_normalized_from_eos={manifest.pad_token_normalized_from_eos}")
    print(f"dataset_file={manifest.dataset_file}")
    print(f"dataset_file_sha256={manifest.dataset_file_sha256}")
    print(f"manifest={manifest_path}")
    print(f"retained_examples={len(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
