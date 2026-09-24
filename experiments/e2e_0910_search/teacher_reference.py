"""Evaluate the frozen Qwen3 teacher through the same E2E eight-task path."""
import argparse
import logging

import torch

from compressed_e2e_fintuning.mid_eval import run_e2e_lm_eval
from compressed_e2e_fintuning.runtime_v6 import _sync_model_padding_config
from e2e_common.data import build_tokenizer
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from train_utils.base_reference import load_frozen_base_reference_model_distributed_from_hf_args
from train_utils.lora_utils import ensure_distill_process_group_initialized, get_distill_local_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("teacher_reference")
    configure_e2e_determinism(False)
    set_e2e_seed(0)
    ensure_distill_process_group_initialized()
    try:
        model_path = "Qwen/Qwen3-8B"
        device = get_distill_local_device(fallback="cuda")
        tokenizer = build_tokenizer(model_path)
        model = load_frozen_base_reference_model_distributed_from_hf_args(
            model_path, argparse.Namespace(access_token=None), device=device, logger=log,
        )
        model.to(dtype=torch.bfloat16)
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = False
        _sync_model_padding_config(model, tokenizer)
        args = argparse.Namespace(
            eval_tasks="boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu",
            eval_num_fewshot=0, eval_lm_batch_size="auto", eval_lm_limit=None,
            eval_hif4_act=False, eval_device="cuda",
        )
        run_e2e_lm_eval(
            model=model, tokenizer=tokenizer, args=args, base_model_path=model_path,
            output_dir=cli.output, log=log, eval_tag="teacher_reference",
            move_to_device=False, cache_decoded_weight=False,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
