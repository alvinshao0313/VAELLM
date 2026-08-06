from __future__ import annotations

import argparse
from pathlib import Path

from mix_bit.checkpoint_pool import build_candidate_pool_index_from_manifest
from mix_bit.cost_table import compute_cost_table, parse_gpu_list
from mix_bit.model_inventory import load_model_inventory
from mix_bit.schema import resolve_run_config, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-GPU resumable mixed-bit KL cost search and cost table finalization"
    )
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--inventory", required=True, help="Path to model_inventory.json")
    parser.add_argument(
        "--pool_manifest",
        required=True,
        help="Path to candidate_manifest.json produced by candidate pool indexing",
    )
    parser.add_argument(
        "--baseline_overlay",
        required=True,
        help="Path to baseline_overlay.json",
    )
    parser.add_argument("--dataset", required=True, help="Path to calibration dataset.pt")
    parser.add_argument(
        "--dataset_manifest",
        required=True,
        help="Path to calibration dataset_manifest.json",
    )
    parser.add_argument(
        "--kl_mode",
        required=True,
        choices=["teacher_topk", "exact_full_vocab"],
        help="KL metric mode (exactly one mode per invocation)",
    )
    parser.add_argument(
        "--teacher_topk",
        type=int,
        default=None,
        help="Teacher top-k K (required for teacher_topk; forbidden for exact_full_vocab)",
    )
    parser.add_argument(
        "--teacher_cache",
        default=None,
        help="Teacher top-k cache dir (required for teacher_topk; forbidden for exact)",
    )
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated physical GPU ids, e.g. 4,5,6,7",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Per-worker forward batch size")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Plan jobs and print counts without evaluating modules",
    )
    parser.add_argument(
        "--recompute",
        action="append",
        default=None,
        help="Force recompute module_name:mode (repeatable); other rows are kept",
    )
    parser.add_argument(
        "--access_token",
        default=None,
        help="Optional Hugging Face access token",
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

    overlay = Path(args.baseline_overlay)
    if not overlay.is_file():
        raise FileNotFoundError(f"Missing baseline overlay: {overlay}")

    gpus = parse_gpu_list(args.gpus)
    result = compute_cost_table(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        baseline_overlay_path=overlay,
        dataset_path=args.dataset,
        dataset_manifest_path=args.dataset_manifest,
        kl_mode=args.kl_mode,
        teacher_topk=args.teacher_topk,
        teacher_cache=args.teacher_cache,
        gpus=gpus,
        batch_size=int(args.batch_size),
        dry_run=bool(args.dry_run),
        recompute=args.recompute,
        access_token=args.access_token,
        inventory_path=args.inventory,
    )

    print(f"dry_run={result.dry_run}")
    print(f"finalized={result.finalized}")
    print(f"source_job_count={result.source_job_count}")
    print(f"non_baseline_module_evaluation_count={result.non_baseline_module_evaluation_count}")
    print(f"complete_row_count={result.complete_row_count}")
    print(f"pending_job_count={result.pending_job_count}")
    print(f"cost_run_root={result.cost_run_root}")
    if result.meta_path:
        print(f"cost_table_meta={result.meta_path}")
        print(f"cost_table_meta_sha256={sha256_file(result.meta_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
