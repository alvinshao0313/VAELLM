from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from mix_bit.assembler import assemble_optimal_mixed_checkpoint, derive_mixed_model_dir
from mix_bit.checkpoint_pool import build_candidate_pool_index_from_manifest
from mix_bit.model_inventory import load_model_inventory, validate_inventory_for_run
from mix_bit.schema import resolve_run_config, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble the optimal mixed-bit full model checkpoint from allocation JSON"
    )
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--inventory", required=True, help="Path to model_inventory.json")
    parser.add_argument(
        "--pool_manifest",
        required=True,
        help="Path to candidate_manifest.json",
    )
    parser.add_argument(
        "--allocation",
        required=True,
        help="Path to optimal_2bit.json (or other allocation stem JSON)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for in-memory assembly (default: cuda)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory; default "
        "<run_root>/mixed_model/<kl-run>/<stem>/final_model",
    )
    parser.add_argument(
        "--allow_suboptimal",
        action="store_true",
        help="Accept an allocation that is not proven globally optimal",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mixed-model output with different provenance",
    )
    parser.add_argument(
        "--access_token",
        default=None,
        help="Optional Hugging Face access token passed to the source tokenizer/model loader",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resolved = resolve_run_config(args.run_config, write=False)
    inventory = load_model_inventory(args.inventory)
    validate_inventory_for_run(inventory, resolved)

    pool_index = build_candidate_pool_index_from_manifest(
        resolved, inventory, args.pool_manifest
    )

    allocation_path = Path(args.allocation)
    if not allocation_path.is_file():
        raise FileNotFoundError(f"Missing allocation file: {allocation_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(derive_mixed_model_dir(allocation_path))

    try:
        result = assemble_optimal_mixed_checkpoint(
            resolved=resolved,
            inventory=inventory,
            inventory_path=args.inventory,
            pool_index=pool_index,
            allocation_path=str(allocation_path),
            device=str(args.device),
            allow_suboptimal=bool(args.allow_suboptimal),
            overwrite=bool(args.overwrite),
            output_dir=output_dir,
            access_token=args.access_token,
        )
    except ValueError as exc:
        if "not globally optimal" in str(exc) and not args.allow_suboptimal:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        raise

    print(f"output_dir={result['output_dir']}")
    print(f"allocation_sha256={result.get('allocation_sha256', sha256_file(allocation_path))}")
    print(f"assignment_count={result.get('assignment_count', result.get('converted_module_count'))}")
    print(f"skipped_identical={result.get('skipped_identical', False)}")
    print(f"state_dict={result.get('state_dict')}")
    print(f"meta={result.get('meta')}")
    print(f"tokenizer_fingerprint_sha256={result.get('tokenizer_fingerprint_sha256', '')}")
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
