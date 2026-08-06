from __future__ import annotations

import argparse
import gc
from pathlib import Path

from mix_bit.candidate_pool import candidate_pool_root
from mix_bit.checkpoint_pool import build_candidate_pool_index
from mix_bit.model_inventory import load_model_inventory
from mix_bit.schema import resolve_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index mixed-bit candidate compact artifacts")
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--inventory", required=True, help="Path to model_inventory.json")
    parser.add_argument(
        "--output_root",
        default=None,
        help="Optional override for candidate_pool root (defaults to canonical run root)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resolved = resolve_run_config(args.run_config, write=True)
    inventory = load_model_inventory(args.inventory)
    pool_root = candidate_pool_root(resolved, output_root=args.output_root)

    index = build_candidate_pool_index(
        resolved,
        inventory,
        output_root=args.output_root,
    )

    expectations = resolved.config.model_profile.regression_expectations or {}
    if expectations:
        expected_c = expectations.get("category_count")
        expected_l = expectations.get("target_linear_count")
        if expected_c is not None and index.category_count != int(expected_c):
            raise ValueError(
                f"Profile regression C mismatch: expected={expected_c} actual={index.category_count}"
            )
        if expected_l is not None and index.target_linear_count != int(expected_l):
            raise ValueError(
                f"Profile regression L mismatch: expected={expected_l} actual={index.target_linear_count}"
            )
        expected_artifacts = index.category_count * index.mode_count
        expected_dense = index.target_linear_count * index.mode_count
        if len(index.sources) != expected_artifacts:
            raise ValueError(
                f"Expected {expected_artifacts} compact artifacts, got {len(index.sources)}"
            )
        if index.dense_module_mode_count != expected_dense:
            raise ValueError(
                f"Expected {expected_dense} dense module-mode entries, "
                f"got {index.dense_module_mode_count}"
            )

    print(f"candidate_pool_root={pool_root}")
    print(f"model_id={index.model_id}")
    print(f"run_id={index.run_id}")
    print(f"C={index.category_count}")
    print(f"L={index.target_linear_count}")
    print(f"R={index.mode_count}")
    print(f"expected_trial_count={index.expected_trial_count}")
    print(f"dense_module_mode_count={index.dense_module_mode_count}")
    print(f"artifact_count={len(index.sources)}")
    print(f"manifest={Path(index.manifest_path)}")
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
