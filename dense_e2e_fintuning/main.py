from typing import Optional, Sequence

from dense_e2e_fintuning.args import parse_args
from dense_e2e_fintuning.runtime import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed


def main(argv: Optional[Sequence[str]] = None) -> None:
    dense_args, hf_args, training_args = parse_args(argv)
    configure_e2e_determinism(bool(training_args.full_determinism))
    set_e2e_seed(int(training_args.seed))
    run(dense_args, hf_args, training_args)


if __name__ == "__main__":
    main()
