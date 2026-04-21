from typing import Optional, Sequence

from dense_e2e_fintuning.args import parse_args
from dense_e2e_fintuning.runtime import run
from train_utils.utils import set_seed


def main(argv: Optional[Sequence[str]] = None) -> None:
    dense_args, hf_args, training_args = parse_args(argv)
    set_seed(int(training_args.seed))
    run(dense_args, hf_args, training_args)


if __name__ == "__main__":
    main()
