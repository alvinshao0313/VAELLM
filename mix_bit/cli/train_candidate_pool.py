from __future__ import annotations

import argparse
import gc

from mix_bit.candidate_pool import run_candidate_pool
from mix_bit.model_inventory import load_model_inventory
from mix_bit.schema import resolve_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train mixed-bit candidate pool")
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--inventory", required=True, help="Path to model_inventory.json")
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU ids, e.g. 4,5,6,7",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Optional override for candidate_pool root (recorded when set)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print deterministic commands without launching subprocesses",
    )
    parser.add_argument("--access_token", default=None, help="Optional Hugging Face token")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    gpus = [part.strip() for part in str(args.gpus).split(",") if part.strip()]
    resolved = resolve_run_config(args.run_config, write=True)
    inventory = load_model_inventory(args.inventory)
    code = run_candidate_pool(
        resolved=resolved,
        inventory=inventory,
        inventory_path=args.inventory,
        gpus=gpus,
        dry_run=bool(args.dry_run),
        output_root=args.output_root,
        access_token=args.access_token,
    )
    gc.collect()
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
