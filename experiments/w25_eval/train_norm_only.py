"""Run native all-backbone-norm recovery for the first400 of the2000-step schedule."""
import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.train_subset_400 import Budget400Callback
from experiments.w25_eval.norm_state import validate_runtime


def run_training(argv=None):
    cfg, hf, args = parse_args(argv)
    validate_runtime(cfg, args)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    try:
        result = run(cfg, hf, args, extra_callbacks=(Budget400Callback(),))
        _require(result['global_step'] == 400, 'Native norm-only training did not stop at400.')
        return result
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    run_training()
