import os
import sys
from typing import Optional, Sequence


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train_utils.cat_checkpoint_distill import run_cat_checkpoint_distill
from train_utils.cat_runtime_adapter import parse_cat_runtime_args


def main(argv: Optional[Sequence[str]] = None) -> None:
    cat_args, hf_args, training_args, vae_args = parse_cat_runtime_args(argv)
    run_cat_checkpoint_distill(
        cat_args=cat_args,
        hf_args=hf_args,
        training_args=training_args,
        vae_args=vae_args,
    )


if __name__ == "__main__":
    main()
