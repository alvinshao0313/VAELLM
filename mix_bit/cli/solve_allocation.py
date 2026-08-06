from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mix_bit.checkpoint_pool import (
    build_candidate_pool_index,
    build_candidate_pool_index_from_manifest,
)
from mix_bit.model_inventory import load_model_inventory, validate_inventory_for_run
from mix_bit.schema import resolve_run_config, sha256_file
from mix_bit.solver import (
    derive_allocation_dir,
    load_cost_table_for_solve,
    solve_mixed_bit_allocation,
    write_allocation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve exact parameter-weighted mixed-bit allocation via SciPy MILP"
    )
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--inventory", required=True, help="Path to model_inventory.json")
    parser.add_argument(
        "--cost_table",
        required=True,
        help="Path to cost_table.jsonl",
    )
    parser.add_argument(
        "--cost_table_meta",
        required=True,
        help="Path to cost_table_meta.json",
    )
    parser.add_argument(
        "--pool_manifest",
        default=None,
        help="Optional candidate_manifest.json; provided path is authoritative",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory; default <run_root>/allocation/<kl-run-name>",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=None,
        help="Optional MILP time limit in seconds (absent => no limit)",
    )
    parser.add_argument(
        "--allow_suboptimal",
        action="store_true",
        help="Accept a time-limited feasible solution that is not proven globally optimal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resolved = resolve_run_config(args.run_config, write=False)
    inventory = load_model_inventory(args.inventory)
    validate_inventory_for_run(inventory, resolved)

    if args.pool_manifest is not None:
        pool_index = build_candidate_pool_index_from_manifest(
            resolved, inventory, args.pool_manifest
        )
    else:
        pool_index = build_candidate_pool_index(resolved, inventory)
    manifest_path = Path(pool_index.manifest_path).resolve()

    expected_hashes = {
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_sha256": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_space_sha256": resolved.candidate_space_sha256,
    }

    rows = load_cost_table_for_solve(
        args.cost_table,
        args.cost_table_meta,
        inventory=inventory,
        candidate_space=resolved.config.candidate_space,
        expected_hashes=expected_hashes,
    )

    # Prefer meta target if present via candidate space; CLI always uses space target.
    target = float(resolved.config.candidate_space.target_average_bit)
    result = solve_mixed_bit_allocation(
        rows,
        inventory=inventory,
        candidate_space=resolved.config.candidate_space,
        target_average_bit=target,
        time_limit_sec=args.time_limit,
        allow_suboptimal=bool(args.allow_suboptimal),
    )

    if not result.is_globally_optimal and not args.allow_suboptimal:
        print(
            "ERROR: allocation is not globally optimal and --allow_suboptimal was not set",
            file=sys.stderr,
        )
        return 1

    meta_path = Path(args.cost_table_meta)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError(f"Expected JSON object in {meta_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else derive_allocation_dir(args.cost_table)
    )
    provenance = {
        **expected_hashes,
        "cost_table_sha256": sha256_file(args.cost_table),
        "cost_table_meta_sha256": sha256_file(meta_path),
        "kl_mode": meta.get("kl_mode"),
        "metric_name": meta.get("metric_name"),
        "teacher_topk": meta.get("teacher_topk"),
        "baseline_kl_mean": meta.get("baseline_kl_mean"),
    }
    paths = write_allocation_outputs(
        result,
        output_dir=output_dir,
        model_id=inventory.model_id,
        run_id=resolved.config.run_id,
        provenance=provenance,
    )

    print(f"output_dir={output_dir}")
    print(f"allocation_json={paths['json']}")
    print(f"is_globally_optimal={result.is_globally_optimal}")
    print(f"allow_suboptimal={result.allow_suboptimal}")
    print(f"objective_delta_kl={result.objective_delta_kl}")
    print(f"achieved_average_bit={result.achieved_average_bit}")
    print(f"used_bit_units={result.used_bit_units}")
    print(f"budget_bit_units={result.budget_bit_units}")
    return 0 if result.is_globally_optimal or args.allow_suboptimal else 1


if __name__ == "__main__":
    raise SystemExit(main())
