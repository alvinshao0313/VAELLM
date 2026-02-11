import argparse
import logging
import math
import os
import pprint
import sys
import time
import random
import numpy as np
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rotation.model_rotation import prepare_model4eval, prepare_model
from rotation.model_utils import get_model
from litebsq.vae_linear import VAELinear

# Try to import lm_eval
try:
    import lm_eval
    from lm_eval import evaluator
except ImportError:
    print("Error: lm_eval not installed. Please install it via `pip install lm_eval`.")
    # We continue to allow PPL evaluation if lm_eval is missing, but tasks will fail.

# Simple logger setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_wikitext2_test(seed, seqlen, model):
    import datasets
    testdata = datasets.load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return testenc


def calculate_avg_accuracy(task_names: list, results: dict) -> float:
    from lm_eval.tasks import get_task_dict

    n_tasks = len(task_names)
    acc_cumul = sum(
        result.get('acc_norm,none', result['acc,none']) for task, result in results.items() if 'mmlu' not in task
    )

    questions_per_mmlu_task = {
        task_name: get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get('acc_norm,none', result['acc,none']) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)


def calculate_mse_per_weight(model, args):
    if not args.ref_model_path:
        logger.error("Please provide --ref_model_path for MSE evaluation.")
        return

    logger.info(f"Loading reference model from {args.ref_model_path} ...")
    try:
        # Load reference model to CPU to save memory, or let get_model handle it
        # Since we compare parameter by parameter, we can keep ref on CPU and move individually
        ref_model = get_model(args.ref_model_path)
    except Exception as e:
        logger.error(f"Failed to load reference model: {e}")
        return

    logger.info("Computing MSE per weight...")

    model_dict = dict(model.named_parameters())
    ref_dict = dict(ref_model.named_parameters())

    all_mses = []

    # Iterate over ref_dict keys to ensure we compare against base
    for name, ref_param in tqdm(ref_dict.items(), desc="Calculating MSE"):
        if name in model_dict:
            param = model_dict[name]

            # To avoid OOM, calculate on CPU
            param_cpu = param.detach().cpu().float()
            ref_param_cpu = ref_param.detach().cpu().float()

            if param_cpu.shape != ref_param_cpu.shape:
                logger.warning(f"Shape mismatch for {name}: {param_cpu.shape} vs {ref_param_cpu.shape}")
                continue

            mse = (param_cpu - ref_param_cpu).abs().sum() / (param_cpu.abs().sum() + 1e-10)
            logger.info(f"{name} MSE: {mse:.4e}")
            all_mses.append(mse)
        else:
            logger.warning(f"Weight {name} missing in evaluated model")

    if all_mses:
        avg_mse = sum(all_mses) / len(all_mses)
        logger.info(f"Global Average Weight MSE: {avg_mse:.4e}")

    # cleanup
    del ref_model
    torch.cuda.empty_cache()


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    module: nn.Module = model
    for part in name.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def _compute_topk_mse(ref_weight: torch.Tensor, err_sq: torch.Tensor, topk: int):
    flat_ref = ref_weight.reshape(-1).abs()
    flat_err = err_sq.reshape(-1)
    k = min(int(topk), flat_ref.numel())
    if k <= 0:
        return None, 0
    topk_idx = torch.topk(flat_ref, k=k, dim=0, largest=True, sorted=False).indices
    topk_mse = float(flat_err[topk_idx].mean().item())
    return topk_mse, k


def evaluate_vae_linear_mse(
    model: nn.Module,
    ref_model: Optional[nn.Module] = None,
    topk: int = 100,
    topn: int = 10,
    log=logger,
) -> Dict[str, Any]:
    if topk <= 0:
        raise ValueError(f"`topk` must be > 0, got {topk}")

    log.info("[linear_mse] Start evaluation (topk=%d).", topk)
    metrics: List[Dict[str, Any]] = []
    vae_linear_count = 0
    skipped = 0

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, VAELinear):
                continue
            vae_linear_count += 1
            try:
                recon_weight = module._decode_weight(dtype=torch.float32).detach().cpu()
            except Exception as e:
                log.warning("[linear_mse] Skip %s: decode failed (%s)", name, e)
                skipped += 1
                continue

            ref_weight = None
            ref_source = "ref_model"
            if ref_model is not None:
                try:
                    ref_module = _get_module_by_name(ref_model, name)
                    if isinstance(ref_module, nn.Linear):
                        ref_weight = ref_module.weight.detach().cpu().float()
                except Exception:
                    ref_weight = None

            if ref_weight is None and module.original_weight is not None:
                ref_weight = module.original_weight.detach().cpu().float()
                ref_source = "checkpoint_original_weight"

            if ref_weight is None:
                log.warning("[linear_mse] Skip %s: no reference weight found.", name)
                skipped += 1
                continue

            if tuple(recon_weight.shape) != tuple(ref_weight.shape):
                log.warning(
                    "[linear_mse] Skip %s: shape mismatch recon=%s ref=%s",
                    name,
                    tuple(recon_weight.shape),
                    tuple(ref_weight.shape),
                )
                skipped += 1
                continue

            err_sq = (recon_weight - ref_weight).pow(2)
            mse = float(err_sq.mean().item())
            topk_mse, k_eff = _compute_topk_mse(ref_weight, err_sq, topk=topk)
            if topk_mse is None:
                topk_mse = float("nan")
            one = {
                "name": name,
                "mse": mse,
                "topk_mse": topk_mse,
                "topk": int(k_eff),
                "numel": int(ref_weight.numel()),
                "source": ref_source,
            }
            metrics.append(one)
            log.info(
                "[linear_mse] %s mse=%.6e topk_mse(k=%d)=%.6e ref=%s",
                name,
                mse,
                k_eff,
                topk_mse,
                ref_source,
            )

    if not metrics:
        return {
            "num_vae_linear": vae_linear_count,
            "num_compared": 0,
            "num_skipped": skipped,
            "avg_mse": None,
            "avg_topk_mse": None,
            "max_mse": None,
            "max_topk_mse": None,
            "worst_by_mse": [],
        }

    metrics_sorted = sorted(metrics, key=lambda x: x["mse"], reverse=True)
    avg_mse = float(sum(m["mse"] for m in metrics) / len(metrics))
    avg_topk_mse = float(sum(m["topk_mse"] for m in metrics) / len(metrics))
    max_mse = float(metrics_sorted[0]["mse"])
    max_topk_mse = float(max(metrics, key=lambda x: x["topk_mse"])["topk_mse"])
    worst = metrics_sorted[: max(1, int(topn))]

    log.info(
        "[linear_mse] Summary: compared=%d skipped=%d avg_mse=%.6e avg_topk_mse=%.6e max_mse=%.6e max_topk_mse=%.6e",
        len(metrics),
        skipped,
        avg_mse,
        avg_topk_mse,
        max_mse,
        max_topk_mse,
    )
    return {
        "num_vae_linear": vae_linear_count,
        "num_compared": len(metrics),
        "num_skipped": skipped,
        "avg_mse": avg_mse,
        "avg_topk_mse": avg_topk_mse,
        "max_mse": max_mse,
        "max_topk_mse": max_topk_mse,
        "worst_by_mse": worst,
    }


def calculate_ppl(model, args):
    logger.info("Evaluating Wikitext-2 PPL...")
    seqlen = int(getattr(args, "seqlen", 2048))
    limit = int(getattr(args, "limit", -1))

    testloader = get_wikitext2_test(
        seed=0, seqlen=seqlen, model=args.model_path)

    testenc = testloader.input_ids
    nsamples = testenc.numel() // seqlen
    if limit > 0:
        nsamples = min(nsamples, limit)

    # Save cache config
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    nlls = []
    with torch.no_grad():
        pbar = tqdm(range(nsamples), desc="PPL Eval")
        for i in pbar:
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(model.device)
            outputs = model(batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][
                :, 1:
            ].to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            neg_log_likelihood = loss.float()

            if not math.isnan(neg_log_likelihood):
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).mean())
            pbar.set_description(f"PPL: {ppl.item():.2f}")

        ppl = torch.exp(torch.stack(nlls).mean())

    logging.info(f'wikitext ppl : {ppl.item():.2f}')
    model.config.use_cache = use_cache  # Restore
    results = {'wiki_ppl': ppl.item(), 'seqlen': int(seqlen), 'nsamples': int(nsamples)}
    return results


def _parse_eval_batch_size(batch_size):
    text = str(batch_size).strip().lower()
    if text == "auto":
        return "auto"
    return int(text)


def _pick_task_metric(task_result):
    keys = ("acc_norm,none", "acc,none", "acc_norm", "acc", "exact_match,none", "exact_match")
    for key in keys:
        value = task_result.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return "n/a", None


def run_lm_eval(model, tokenizer, args):
    if "lm_eval" not in globals() or "evaluator" not in globals():
        raise ImportError("lm_eval not installed. Please install it via `pip install lm_eval`.")

    if not getattr(args, "tasks", None):
        raise ValueError("`args.tasks` is required for lm_eval.")

    logger.info(f"Evaluating on tasks: {args.tasks}")
    task_names = [t.strip() for t in args.tasks.split(',') if t.strip()]
    if not task_names:
        raise ValueError("No valid task names parsed from `args.tasks`.")

    default_batch_size = 8 if '70' in str(args.model_path) else "auto"
    batch_size = _parse_eval_batch_size(getattr(args, "batch_size", default_batch_size))
    lm_limit = getattr(args, "lm_limit", None)
    logger.info(
        "LM-Eval config: fewshot=%s batch_size=%s limit=%s",
        str(getattr(args, "num_fewshot", 0)),
        str(batch_size),
        str(lm_limit),
    )

    lm = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend='causal',
        trust_remote_code=True,
        batch_size=batch_size,
    )

    with torch.no_grad():
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_names,
            num_fewshot=getattr(args, "num_fewshot", 0),
            batch_size=batch_size,
            limit=lm_limit,
        )

    results_dict = results['results']
    logging.info(pprint.pformat(results_dict))

    metric_vals = {}
    task_metric_keys = {}
    for task, result in results_dict.items():
        key, metric = _pick_task_metric(result)
        task_metric_keys[task] = key
        metric_vals[task] = metric

    try:
        acc_avg = calculate_avg_accuracy(task_names, results_dict)
        metric_vals['average'] = float(acc_avg)
    except Exception as e:
        logger.warning(f"Could not calculate average accuracy: {e}")

    for task, result in metric_vals.items():
        if result is None:
            logging.info(f'Task {task} metric: N/A')
        else:
            logging.info(f'Task {task} acc: {result * 100 :.2f}')

    return {
        "tasks": task_names,
        "num_fewshot": int(getattr(args, "num_fewshot", 0)),
        "batch_size": batch_size,
        "limit": lm_limit,
        "task_metric_keys": task_metric_keys,
        "task_metrics": metric_vals,
        "raw_results": results_dict,
    }


def evaluate_model(model, tokenizer, args):
    # Ensure model is on GPU
    if model.device.type == 'cpu':
        logger.info("Moving model to CUDA...")
        model.to("cuda")

    # ============================ Evaluation
    if args.eval_mse:
        calculate_mse_per_weight(model, args)

    if args.eval_ppl:
        calculate_ppl(model, args)

    # LM Eval Harness
    if args.tasks:
        run_lm_eval(model, tokenizer, args)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Wikitext-2 PPL and LM-Eval Tasks")

    parser.add_argument('--model_path', type=str, required=True, help='Path to the model or HF hub ID')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--eval_ppl', action='store_true', help='Evaluate Wikitext-2 PPL')
    parser.add_argument('--limit', type=int, default=-1, help='Limit samples for PPL eval')
    parser.add_argument('--eval_mse', action='store_true', help='Evaluate MSE against reference model')
    parser.add_argument('--ref_model_path', type=str, default=None, help='Path to reference model for MSE evaluation')

    # Rotation and Hadamard Transform
    parser.add_argument('--rotate_vqmodel', action='store_true', default=False)
    parser.add_argument('--rotate', action='store_true', default=False)
    parser.add_argument('--rotate_mode', type=str, default='hadamard',
                        choices=['hadamard', 'group_hadamard', 'identity'])
    parser.add_argument('--online_partial_had', action='store_true', default=False)
    parser.add_argument('--online_down_had', action='store_true', default=True)
    parser.add_argument('--r1_path', type=str, default=None,
                        help='''Path to the R1 rotation matrix. Deafult is None.
                        If not specified, R1 will generated as "rotate_mode".''')

    # LM Eval args
    parser.add_argument('--tasks', type=str, default=None,
                        help='Comma separated list of tasks for lm-eval (e.g. piqa,arc_easy)')
    parser.add_argument('--num_fewshot', type=int, default=0, help='Number of few-shot examples')
    parser.add_argument('--batch_size', type=str, default='auto', help='Batch size for eval')

    args = parser.parse_args()

    set_seed(args.seed)

    logger.info(f"Loading model from {args.model_path} ...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    # Load Model
    # Determine device map based on availability
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    model = get_model(args.model_path)
    if args.rotate_vqmodel:
        model, _ = prepare_model4eval(model, args)
    if args.rotate:
        model, _ = prepare_model(model, args)

    evaluate_model(model, tokenizer, args)

    logger.info("Evaluation Complete.")


if __name__ == "__main__":
    main()
