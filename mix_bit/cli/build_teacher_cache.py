from __future__ import annotations

import argparse

from mix_bit.schema import resolve_run_config
from mix_bit.teacher_cache import build_teacher_topk_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reusable teacher top-k cache for mix_bit KL search"
    )
    parser.add_argument("--run_config", required=True, help="Path to mix_bit run config JSON")
    parser.add_argument("--inventory", required=True, help="Path to model_inventory.json")
    parser.add_argument("--dataset", required=True, help="Path to calibration dataset.pt")
    parser.add_argument(
        "--dataset_manifest",
        required=True,
        help="Path to calibration dataset_manifest.json",
    )
    parser.add_argument(
        "--teacher_topk",
        required=True,
        type=int,
        help="Explicit teacher top-k K (required; no default)",
    )
    parser.add_argument(
        "--cache_prob_dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Dtype for cached teacher_topk_probs",
    )
    parser.add_argument(
        "--chunk_samples",
        type=int,
        default=16,
        help="Number of calibration samples per cache chunk file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Forward batch size while building the cache",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for teacher forward",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional cache output directory (default: <run_root>/calibration/teacher_topk/kK)",
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
    index = build_teacher_topk_cache(
        resolved,
        inventory_path=args.inventory,
        dataset_path=args.dataset,
        dataset_manifest_path=args.dataset_manifest,
        teacher_topk=int(args.teacher_topk),
        cache_prob_dtype=str(args.cache_prob_dtype),
        chunk_samples=int(args.chunk_samples),
        batch_size=int(args.batch_size),
        device=str(args.device),
        output_dir=args.output_dir,
        access_token=args.access_token,
    )

    print(f"kl_mode={index['kl_mode']}")
    print(f"metric_name={index['metric_name']}")
    print(f"teacher_topk={index['teacher_topk']}")
    print(f"vocab_size={index['vocab_size']}")
    print(f"cache_prob_dtype={index['cache_prob_dtype']}")
    print(f"sample_count={index['sample_count']}")
    print(f"total_valid_positions={index['total_valid_positions']}")
    print(f"cache_dir={index['cache_dir']}")
    print(f"output_dir_override={index['output_dir_override']}")
    print(f"cache_content_sha256={index['cache_content_sha256']}")
    print(f"index={index['index_path']}")
    print(f"index_sha256={index['index_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
