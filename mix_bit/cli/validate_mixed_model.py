from __future__ import annotations

import argparse
import gc
import sys

from mix_bit.checkpoint_pool import build_candidate_pool_index_from_manifest
from mix_bit.model_inventory import load_model_inventory, validate_inventory_for_run
from mix_bit.schema import resolve_run_config, sha256_file
from mix_bit.validation import validate_mixed_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an assembled optimal mixed-bit checkpoint "
            "(structural/provenance, budget, save/reload logits, KL, optional downstream)"
        )
    )
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--inventory", required=True, help="Path to model_inventory.json")
    parser.add_argument(
        "--pool_manifest",
        required=True,
        help="Path to candidate_manifest.json",
    )
    parser.add_argument("--cost_table", required=True, help="Path to cost_table.jsonl")
    parser.add_argument(
        "--cost_table_meta",
        required=True,
        help="Path to cost_table_meta.json",
    )
    parser.add_argument(
        "--allocation",
        required=True,
        help="Path to optimal_2bit.json (or other allocation stem JSON)",
    )
    parser.add_argument(
        "--baseline_overlay",
        required=True,
        help="Path to baseline_overlay.json",
    )
    parser.add_argument(
        "--mixed_model_dir",
        required=True,
        help="Path to assembled final_model directory",
    )
    parser.add_argument("--dataset", required=True, help="Path to calibration dataset.pt")
    parser.add_argument(
        "--dataset_manifest",
        required=True,
        help="Path to calibration dataset_manifest.json",
    )
    parser.add_argument(
        "--teacher_cache",
        default=None,
        help="Teacher top-k cache dir (required for teacher_topk; forbidden for exact)",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--lm_batch_size",
        default="auto",
        help="lm-eval batch size (default: auto)",
    )
    parser.add_argument(
        "--skip_downstream_eval",
        action="store_true",
        help="Skip WikiText-2 PPL / lm-eval; structural+KL only",
    )
    parser.add_argument(
        "--allow_suboptimal",
        action="store_true",
        help="Accept an allocation that is not proven globally optimal",
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

    resolved = resolve_run_config(args.run_config, write=False)
    inventory = load_model_inventory(args.inventory)
    validate_inventory_for_run(inventory, resolved)

    pool_index = build_candidate_pool_index_from_manifest(
        resolved, inventory, args.pool_manifest
    )

    lm_batch_size: str | int
    if str(args.lm_batch_size).strip().lower() == "auto":
        lm_batch_size = "auto"
    else:
        lm_batch_size = int(args.lm_batch_size)

    try:
        report = validate_mixed_model(
            resolved=resolved,
            inventory=inventory,
            inventory_path=args.inventory,
            pool_index=pool_index,
            allocation_path=args.allocation,
            cost_table_path=args.cost_table,
            cost_table_meta_path=args.cost_table_meta,
            baseline_overlay_path=args.baseline_overlay,
            mixed_model_dir=args.mixed_model_dir,
            dataset_path=args.dataset,
            dataset_manifest_path=args.dataset_manifest,
            teacher_cache=args.teacher_cache,
            device=str(args.device),
            skip_downstream_eval=bool(args.skip_downstream_eval),
            lm_batch_size=lm_batch_size,
            access_token=args.access_token,
            allow_suboptimal=bool(args.allow_suboptimal),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"passed={report['passed']}")
    print(f"validation_json={report['validation_json']}")
    print(f"validation_md={report['validation_md']}")
    print(f"allocation_sha256={report['allocation_sha256']}")
    print(f"used_bit_units={report['budget']['used_bit_units']}")
    print(f"budget_bit_units={report['budget']['budget_bit_units']}")
    print(f"predicted_mixed_model_kl={report['kl']['predicted_mixed_model_kl']}")
    print(f"actual_mixed_model_kl={report['kl']['actual_mixed_model_kl']}")
    print(f"absolute_gap={report['kl']['absolute_gap']}")
    print(f"relative_gap={report['kl']['relative_gap']}")
    print(f"cost_table_sha256={sha256_file(args.cost_table)}")
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
