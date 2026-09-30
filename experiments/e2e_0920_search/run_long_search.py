#!/usr/bin/env python3
"""Queue the finite seven-day DP search behind the existing three-trial pilot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.e2e_0920_search.search_workflow import DEFAULT_ROOT, preview, run_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("optimization", "loss_weights"), required=True)
    parser.add_argument("--budget-days", type=int, choices=(3, 7, 14), required=True)
    parser.add_argument("--predecessor-pid", type=int)
    parser.add_argument("--predecessor-start", help="Linux /proc/PID/stat starttime ticks")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(preview(args.scope, args.budget_days), ensure_ascii=False, indent=2))
        return
    if args.predecessor_pid is None or args.predecessor_start is None:
        parser.error("Queueing requires --predecessor-pid and --predecessor-start.")
    run_search(args.output_root, args.scope, args.budget_days, args.predecessor_pid, args.predecessor_start)


if __name__ == "__main__":
    main()
