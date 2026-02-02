import argparse
import logging
import math
import os
import pprint
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rotation.model_rotation import prepare_model4eval, prepare_model
from rotation.model_utils import get_model

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
    # User specified local parquet file
    parquet_path = "/home/shaoyuantian/.cache/huggingface/hub/datasets--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    if os.path.exists(parquet_path):
        logger.info(f"Loading local wikitext parquet from {parquet_path}")
        testdata = datasets.load_dataset("parquet", data_files={'test': parquet_path}, split='test')
    else:
        logger.warning(f"Local parquet not found at {parquet_path}, trying default load...")
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


def calculate_ppl(model, args):
    logger.info("Evaluating Wikitext-2 PPL...")
    seqlen = 2048  # hard-coding as per snippet

    testloader = get_wikitext2_test(
        seed=args.seed, seqlen=seqlen, model=args.model_path)

    testenc = testloader.input_ids
    nsamples = testenc.numel() // seqlen

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

            if args.limit > 0 and i >= args.limit:
                break

        ppl = torch.exp(torch.stack(nlls).mean())

    logging.info(f'wikitext ppl : {ppl.item():.2f}')
    model.config.use_cache = use_cache  # Restore
    results = {'wiki_ppl': ppl.item()}
    return results


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
        logger.info(f"Evaluating on tasks: {args.tasks}")
        task_names = args.tasks.split(',')

        # Prepare batch size
        if '70' in args.model_path:
            batch_size = 8
        else:
            batch_size = "auto"

        # Initialize HFLM wrapper
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
                num_fewshot=args.num_fewshot,
                batch_size=batch_size
            )

        results_dict = results['results']
        logging.info(pprint.pformat(results_dict))

        metric_vals = {}
        for task, result in results_dict.items():
            # Support new lm-eval structure (v0.4+)
            # result is a dict with 'acc', 'acc_norm', etc.
            # Handle possible keys
            acc = result.get('acc_norm,none', result.get('acc,none', result.get('acc', 0)))
            metric_vals[task] = round(acc, 4)

        try:
            acc_avg = calculate_avg_accuracy(task_names, results_dict)
            metric_vals['average'] = round(acc_avg, 4)
        except Exception as e:
            logger.warning(f"Could not calculate average accuracy: {e}")

        for task, result in metric_vals.items():
            logging.info(f'Task {task} acc: {result * 100 :.2f}')


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
