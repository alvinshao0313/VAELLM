"""Run canonical E2E training with isolated strict evaluation at each save."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.step_export_callback import StepExportCallback


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--export_cuda_visible_devices", required=True,
                        help="Single physical GPU reserved for independent saved-step export/evaluation.")
    export_args, remaining = parser.parse_known_args(argv)
    cfg, hf_args, training_args = parse_args(remaining)
    callback = StepExportCallback(source_dir=cfg.student_checkpoint_dir, run_root_dir=cfg.run_root_dir,
                                  train_mode=cfg.train_mode,
                                  export_cuda_visible_devices=export_args.export_cuda_visible_devices)
    configure_e2e_determinism(bool(training_args.full_determinism))
    set_e2e_seed(int(training_args.seed))
    try:
        run(cfg, hf_args, training_args, extra_callbacks=(callback,))
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
