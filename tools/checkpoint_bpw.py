"""Report logical and physical Linear bit budgets from a stable v6 checkpoint."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_utils.checkpoint_bpw import audit_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max_bpw", type=float, default=2.5)
    parser.add_argument("--max_lora_rank", type=int, default=16, help="Maximum stored LoRA rank; 0 requires no LoRA.")
    args = parser.parse_args()
    result = audit_checkpoint(args.checkpoint_dir, max_bpw=args.max_bpw, max_lora_rank=args.max_lora_rank)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["budget_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
