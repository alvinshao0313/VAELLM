from typing import Optional, Sequence

from e2e_fintuning.args import parse_args
from e2e_fintuning.runtime import run
from train_utils.utils import set_seed


def main(argv: Optional[Sequence[str]] = None) -> None:
    e2e_args, hf_args, training_args = parse_args(argv)
    set_seed(int(training_args.seed))
    run(e2e_args, hf_args, training_args)


if __name__ == "__main__":
    main()
