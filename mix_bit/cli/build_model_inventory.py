from __future__ import annotations

import argparse
import gc
from pathlib import Path

from mix_bit.model_inventory import build_model_inventory, maybe_write_inventory
from mix_bit.schema import resolve_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build mixed-bit model inventory")
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--output", required=True, help="Output model_inventory.json path")
    parser.add_argument(
        "--access_token",
        default=None,
        help="Optional Hugging Face access token",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing inventory when fingerprints differ",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resolved = resolve_run_config(args.run_config, write=True)
    inventory = build_model_inventory(resolved, access_token=args.access_token)
    maybe_write_inventory(inventory, args.output, overwrite=bool(args.overwrite))

    category_count = len(inventory.category_order)
    linear_count = len(inventory.targets)
    print(f"model_id={inventory.model_id}")
    print(f"C={category_count}")
    print(f"L={linear_count}")
    print(f"block_count={inventory.block_count}")
    print(f"total_target_parameters={inventory.total_target_parameters}")
    print(f"fingerprint_sha256={inventory.fingerprint_sha256}")
    print(f"output={Path(args.output).resolve()}")

    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
