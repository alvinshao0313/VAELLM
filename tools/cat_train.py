import os
import sys
from typing import Optional, Sequence


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train_utils.cat_train_args import process_cat_train_args
from train_utils.cat_train_pipeline import run_cat_train


def main(argv: Optional[Sequence[str]] = None) -> None:
    cat_args, hf_args, training_args, vae_args = process_cat_train_args(argv)
    run_cat_train(
        cat_args=cat_args,
        hf_args=hf_args,
        training_args=training_args,
        vae_args=vae_args,
    )


if __name__ == "__main__":
    main()
