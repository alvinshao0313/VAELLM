"""Create a new CAT experiment from a real, finalized category boundary on CPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / "result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation"
SNAPSHOT = TOPIC / "source_snapshot"
DEFAULT_SOURCE = (TOPIC / "formal_k38_gate_s4_seq500/native/"
                  "home_star_models_Qwen_Qwen3-8B_20261008_115322/after_o_proj")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profiles", nargs="+", choices=("C", "G", "D"), default=("C", "G"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/star/VAELLM/qwen_recompression_20261026"))
    parser.add_argument("--report", type=Path, default=ROOT / "result/qwen3_8b_mmlu65_20261004/qwen_recompression_20261026/prepare.json")
    parser.add_argument("--verify-profile", choices=("C", "G", "D"))
    parser.add_argument("--verify-saved-preparation", type=Path,
                        help="Write separate completion evidence for the saved D allocation-only stat failure.")
    parser.add_argument("--retry-failed-preparation", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CPU preparation requires CUDA_VISIBLE_DEVICES=''.")
    sys.path.insert(0, str(SNAPSHOT))
    from fork_core import prepare, verify_profile, verify_saved_preparation

    if args.verify_saved_preparation:
        if args.verify_profile:
            parser.error("Choose only one verification mode.")
        verify_saved_preparation(args.report, args.verify_saved_preparation)
    elif args.verify_profile:
        verify_profile(args.report.resolve(), args.verify_profile)
    else:
        prepare(args, snapshot=SNAPSHOT, topic=TOPIC)


if __name__ == "__main__":
    main()
