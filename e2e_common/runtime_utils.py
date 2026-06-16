import argparse
import json
import os

import torch

from e2e_common.data import build_datasets
from train_utils.eval_utils import calculate_ppl
from train_utils.hif4_act import applied_hif4_act


def normalized_eval_strategy(training_args) -> str:
    eval_strategy = getattr(training_args, "eval_strategy", None)
    normalized = getattr(eval_strategy, "value", eval_strategy)
    if normalized is None:
        return "none"
    return str(normalized).strip().lower()


def build_datasets_with_main_process_first(args, training_args, tokenizer, log):
    eval_strategy = normalized_eval_strategy(training_args)
    skip_eval_preprocessing = eval_strategy == "no"
    log.info(
        "Dataset preprocess config: dataset_num_proc=%d eval_strategy=%s skip_eval_preprocessing=%s main_process_first=%s",
        int(getattr(args, "dataset_num_proc", 1)),
        eval_strategy,
        str(skip_eval_preprocessing).lower(),
        "true",
    )
    with training_args.main_process_first(local=False, desc="dataset preprocessing"):
        return build_datasets(args, training_args, tokenizer)


def eval_final_ppl(*, model, args, model_path: str, output_dir: str, log):
    if bool(getattr(args, "skip_ppl_eval", False)):
        log.info("Skipping final PPL evaluation because --skip_ppl_eval=true.")
        return None

    ppl_args = argparse.Namespace(
        model_path=str(model_path),
        seqlen=int(getattr(args, "ppl_seqlen", 2048)),
        limit=int(getattr(args, "ppl_limit", -1)),
    )
    log.info(
        "Start final PPL eval (seqlen=%d, limit=%d)...",
        int(ppl_args.seqlen),
        int(ppl_args.limit),
    )
    with applied_hif4_act(
        model,
        enabled=bool(getattr(args, "eval_hif4_act", False)),
        logger=log,
        log_prefix="[final_ppl] ",
    ):
        with torch.no_grad():
            ppl_result = calculate_ppl(model, ppl_args)

    result = {
        "wiki_ppl": float(ppl_result.get("wiki_ppl", float("nan"))),
        "nsamples": int(ppl_result.get("nsamples", 0)),
        "seqlen": int(ppl_result.get("seqlen", int(ppl_args.seqlen))),
    }
    ppl_path = os.path.join(output_dir, "final_ppl.json")
    with open(ppl_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    log.info(
        "Final PPL=%.4f (nsamples=%d, seqlen=%d) saved to %s",
        result["wiki_ppl"],
        result["nsamples"],
        result["seqlen"],
        ppl_path,
    )
    return {
        "result": result,
        "path": ppl_path,
    }
