import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

META_FILENAME = "checkpoint_meta.json"


def _build_logger(log_dir: str) -> Tuple[logging.Logger, str]:
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(log_dir, f"cat_eval_{ts}.log")

    logger = logging.getLogger("cat_eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger, log_path


def _resolve_checkpoint_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        if os.path.basename(abs_path) == META_FILENAME:
            return os.path.dirname(abs_path)
        raise FileNotFoundError(f"Expected {META_FILENAME} file, got: {abs_path}")

    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Path does not exist: {abs_path}")

    direct_meta = os.path.join(abs_path, META_FILENAME)
    if os.path.exists(direct_meta):
        return abs_path

    final_model_meta = os.path.join(abs_path, "final_model", META_FILENAME)
    if os.path.exists(final_model_meta):
        return os.path.join(abs_path, "final_model")

    candidates: List[str] = []
    for child in os.listdir(abs_path):
        child_dir = os.path.join(abs_path, child)
        if not os.path.isdir(child_dir):
            continue
        if os.path.exists(os.path.join(child_dir, META_FILENAME)):
            candidates.append(child_dir)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot find checkpoint metadata under: {abs_path}. "
        f"Please pass a directory containing {META_FILENAME}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoints saved by tools/cat_train.py"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=f"Path to checkpoint dir (contains {META_FILENAME}) or a cat_train run directory.",
    )
    parser.add_argument("--base_model_path", type=str, default=None, help="Override base model path in checkpoint meta.")
    parser.add_argument("--access_token", type=str, default=None, help="Hugging Face access token.")
    parser.add_argument("--map_location", type=str, default="cpu", help="Checkpoint load map location.")
    parser.add_argument("--strict", dest="strict", action="store_true")
    parser.add_argument("--no_strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)

    parser.add_argument("--eval_ppl", action="store_true", help="Run Wikitext-2 PPL evaluation.")
    parser.add_argument("--eval_lm_eval", action="store_true", help="Run lm_eval tasks.")
    parser.add_argument(
        "--eval_linear_mse",
        action="store_true",
        help="Compute per-VAELinear MSE/topk-MSE against base model linears.",
    )

    parser.add_argument("--eval_device", type=str, default="cuda", help="Device for PPL/lm_eval.")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="Sequence length for PPL evaluation.")
    parser.add_argument("--ppl_limit", type=int, default=-1, help="Max number of PPL samples, -1 for all.")

    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated lm_eval task names.")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot count for lm_eval.")
    parser.add_argument("--lm_batch_size", type=str, default="auto", help="Batch size for lm_eval.")
    parser.add_argument("--lm_limit", type=int, default=None, help="Optional lm_eval sample limit.")

    parser.add_argument("--topk", type=int, default=100, help="K used in topk-MSE for each linear.")
    parser.add_argument(
        "--ref_model_path",
        type=str,
        default=None,
        help="Override reference model path for linear MSE. Defaults to base model path.",
    )

    parser.add_argument("--eval_log_dir", type=str, default="./eval_log", help="Directory to store evaluation logs.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logger, log_path = _build_logger(args.eval_log_dir)
    logger.info("Eval log file: %s", log_path)
    logger.info("Input args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    if not (args.eval_ppl or args.eval_lm_eval or args.eval_linear_mse):
        raise ValueError("No evaluation selected. Please enable at least one of: --eval_ppl, --eval_lm_eval, --eval_linear_mse")
    if args.eval_lm_eval and (args.tasks is None or not str(args.tasks).strip()):
        raise ValueError("--tasks is required when --eval_lm_eval is enabled.")

    ckpt_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    logger.info("Resolved checkpoint directory: %s", ckpt_dir)

    logger.info("Loading evaluated model from checkpoint...")
    from train_utils.model_checkpoint_io import load_model_checkpoint

    model, meta, load_result = load_model_checkpoint(
        ckpt_dir,
        access_token=args.access_token,
        base_model_path=args.base_model_path,
        map_location=args.map_location,
        strict=args.strict,
    )
    logger.info(
        "Checkpoint loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s",
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
        str(meta.get("converted_module_count")),
    )

    base_model_path = args.base_model_path or meta.get("base_model_path")
    if not base_model_path:
        raise ValueError("Cannot determine base_model_path. Provide --base_model_path.")
    tokenizer_name = str(base_model_path)
    logger.info("Base model path: %s", base_model_path)

    summary: Dict[str, Any] = {
        "checkpoint_dir": ckpt_dir,
        "base_model_path": base_model_path,
        "evals": {},
    }

    if args.eval_linear_mse:
        logger.info("Loading reference model for linear MSE comparison...")
        from rotation.model_utils import get_model
        from train_utils.eval_utils import evaluate_vae_linear_mse

        ref_path = args.ref_model_path or base_model_path
        ref_model = None
        try:
            ref_model = get_model(ref_path, args.access_token)
            logger.info("Reference model loaded from %s", ref_path)
        except Exception as e:
            logger.warning(
                "Failed to load reference model from %s (%s). Will fallback to checkpoint original weights if available.",
                ref_path,
                e,
            )

        linear_result = evaluate_vae_linear_mse(
            model=model,
            ref_model=ref_model,
            topk=int(args.topk),
            log=logger,
        )
        summary["evals"]["linear_mse"] = linear_result
        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.eval_ppl:
        device = args.eval_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, fallback eval_device to cpu.")
            device = "cpu"
        from train_utils.eval_utils import calculate_ppl

        logger.info("[ppl] Run via train_utils.eval_utils.calculate_ppl")
        model.to(device)
        ppl_args = argparse.Namespace(
            model_path=tokenizer_name,
            seqlen=int(args.ppl_seqlen),
            limit=int(args.ppl_limit),
        )
        ppl_result = calculate_ppl(model, ppl_args)
        logger.info("[ppl] Result: %s", json.dumps(ppl_result, ensure_ascii=False))
        summary["evals"]["ppl"] = ppl_result

    if args.eval_lm_eval:
        device = args.eval_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, fallback eval_device to cpu.")
            device = "cpu"
        from transformers import AutoTokenizer
        from train_utils.eval_utils import run_lm_eval

        logger.info("[lm_eval] Run via train_utils.eval_utils.run_lm_eval")
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            trust_remote_code=True,
            token=args.access_token,
        )
        lm_args = argparse.Namespace(
            tasks=str(args.tasks),
            num_fewshot=int(args.num_fewshot),
            batch_size=str(args.lm_batch_size),
            lm_limit=args.lm_limit,
            model_path=tokenizer_name,
        )
        lm_result = run_lm_eval(model, tokenizer, lm_args)
        logger.info("[lm_eval] Tasks done: %s", ",".join(lm_result.get("tasks", [])))
        summary["evals"]["lm_eval"] = lm_result

    logger.info("Evaluation summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("All evaluations completed.")


if __name__ == "__main__":
    main()
