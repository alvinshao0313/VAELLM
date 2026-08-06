from __future__ import annotations

import argparse
import gc

from mix_bit.assembler import prepare_uniform_baseline_overlay
from mix_bit.checkpoint_pool import build_candidate_pool_index_from_manifest
from mix_bit.model_inventory import load_model_inventory
from mix_bit.schema import resolve_run_config, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare tensor-free uniform baseline overlay and in-memory assembly audit"
    )
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--inventory", required=True, help="Path to model_inventory.json")
    parser.add_argument(
        "--pool_manifest",
        required=True,
        help="Path to candidate_manifest.json produced by candidate pool indexing",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for in-memory assembly audit (default: cuda)",
    )
    parser.add_argument(
        "--skip_audit",
        action="store_true",
        help="Write overlay only; skip dual in-memory build / logits audit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resolved = resolve_run_config(args.run_config, write=True)
    inventory = load_model_inventory(args.inventory)
    pool_index = build_candidate_pool_index_from_manifest(
        resolved, inventory, args.pool_manifest
    )

    result = prepare_uniform_baseline_overlay(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        device=str(args.device),
        skip_audit=bool(args.skip_audit),
    )

    print(f"mode={result['mode']}")
    print(f"assignment_count={result['assignment_count']}")
    print(f"baseline_dir={result['baseline_dir']}")
    print(f"baseline_overlay={result['baseline_overlay']}")
    print(f"baseline_overlay_sha256={sha256_file(result['baseline_overlay'])}")
    print(f"assembly_audit={result['assembly_audit']}")
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
