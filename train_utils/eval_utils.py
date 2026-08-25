import argparse
import json
import logging
import math
import os
import sys
import time
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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


def _build_logger(log_dir: str) -> Tuple[str, str]:
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(log_dir, f"eval_utils_{ts}.log")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if os.path.abspath(handler.baseFilename) == os.path.abspath(log_path):
                    return ts, log_path
            except Exception:
                continue

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)
    return ts, log_path


def _json_dump(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _format_metric_percent(metric: Optional[float]) -> str:
    if metric is None:
        return "N/A"
    return f"{metric * 100:.2f}"


def _format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(headers)
    table_rows = [headers, sep, *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _compute_weighted_group_metric(
    *,
    group_name: str,
    results: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[float]]:
    weighted_sum = 0.0
    total_weight = 0
    metric_key = "n/a"
    for task_name, task_result in results.items():
        if not task_name.startswith(f"{group_name}_"):
            continue
        one_metric_key, one_metric = _pick_task_metric(task_result)
        if one_metric is None:
            continue
        weight = int(task_result.get("samples", 0) or 0)
        if weight <= 0:
            continue
        metric_key = one_metric_key
        weighted_sum += one_metric * weight
        total_weight += weight
    if total_weight <= 0:
        return "n/a", None
    return metric_key, weighted_sum / total_weight


def _resolve_task_metric(
    *,
    task_name: str,
    results: Dict[str, Dict[str, Any]],
    groups: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[float]]:
    if task_name in groups:
        return _pick_task_metric(groups[task_name])
    if task_name in results:
        return _pick_task_metric(results[task_name])
    if task_name == "mmlu":
        return _compute_weighted_group_metric(group_name="mmlu", results=results)
    return "n/a", None


def _build_lm_eval_summary_rows(
    task_names: List[str],
    results: Dict[str, Dict[str, Any]],
    groups: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task_name in task_names:
        metric_key, metric_val = _resolve_task_metric(
            task_name=task_name,
            results=results,
            groups=groups,
        )
        rows.append(
            {
                "task": task_name,
                "metric_key": metric_key,
                "metric": metric_val,
                "score_percent": _format_metric_percent(metric_val),
            }
        )
    return rows


def merge_lm_eval_results(
    partial_results: List[Optional[Dict[str, Any]]],
    task_names: List[str],
) -> Dict[str, Any]:
    merged_metrics: Dict[str, Optional[float]] = {}
    merged_metric_keys: Dict[str, str] = {}
    merged_raw_results: Dict[str, Dict[str, Any]] = {}
    merged_group_results: Dict[str, Dict[str, Any]] = {}

    for partial in partial_results:
        if not partial:
            continue
        for task_name, metric in (partial.get("task_metrics") or {}).items():
            merged_metrics[str(task_name)] = metric
        for task_name, metric_key in (partial.get("task_metric_keys") or {}).items():
            merged_metric_keys[str(task_name)] = str(metric_key)
        raw_results = partial.get("raw_results")
        if isinstance(raw_results, dict):
            merged_raw_results.update(raw_results)
        group_results = partial.get("group_results")
        if isinstance(group_results, dict):
            merged_group_results.update(group_results)

    missing_tasks = [
        str(task_name)
        for task_name in task_names
        if merged_metrics.get(str(task_name)) is None
    ]
    if missing_tasks:
        raise ValueError(
            "Distributed lm_eval merge missing task metrics for: "
            + ",".join(missing_tasks)
        )

    summary_rows = _build_lm_eval_summary_rows(
        task_names=[str(task_name) for task_name in task_names],
        results=merged_raw_results,
        groups=merged_group_results,
    )
    for row in summary_rows:
        task_name = str(row["task"])
        metric_val = row.get("metric")
        if metric_val is not None:
            merged_metrics[task_name] = float(metric_val)
        merged_metric_keys[task_name] = str(row.get("metric_key", merged_metric_keys.get(task_name, "n/a")))

    table_headers = ["Task", "Metric", "Score(%)"]
    table_rows = [
        [str(row["task"]), str(row["metric_key"]), str(row["score_percent"])]
        for row in summary_rows
    ]
    summary_table = _format_markdown_table(table_headers, table_rows)

    return {
        "tasks": [str(task_name) for task_name in task_names],
        "task_metrics": {str(task_name): merged_metrics[str(task_name)] for task_name in task_names},
        "task_metric_keys": {
            str(task_name): merged_metric_keys.get(str(task_name), "n/a")
            for task_name in task_names
        },
        "summary_rows": summary_rows,
        "summary_table": summary_table,
        "raw_results": merged_raw_results,
        "group_results": merged_group_results,
    }
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
    if ref_model is None:
        raise ValueError("[linear_mse] ref_model is required; checkpoint original_weight fallback is no longer supported.")

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

            ref_module = _get_module_by_name(ref_model, name)
            if not isinstance(ref_module, nn.Linear):
                raise TypeError(f"[linear_mse] Reference module is not nn.Linear for {name}: {type(ref_module)}")
            ref_weight = ref_module.weight.detach().cpu().float()
            ref_source = "ref_model"

            if tuple(recon_weight.shape) != tuple(ref_weight.shape):
                raise ValueError(
                    f"[linear_mse] {name}: shape mismatch recon={tuple(recon_weight.shape)} "
                    f"ref={tuple(ref_weight.shape)}"
                )

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


_CHOICE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _extract_response_score(response: Any) -> float:
    if isinstance(response, (int, float)):
        return float(response)
    if isinstance(response, (list, tuple)) and response:
        first = response[0]
        if isinstance(first, (int, float)):
            return float(first)
        if len(response) == 1:
            return _extract_response_score(first)
    raise ValueError(f"Unsupported MMLU response score format: {response!r}")


def _coerce_choice_index(value: Any, *, choices: List[str], score_count: int) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value < score_count else None
    if isinstance(value, float) and value.is_integer():
        idx = int(value)
        return idx if 0 <= idx < score_count else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            idx = int(text)
            return idx if 0 <= idx < score_count else None
        upper = text.upper()
        labels = _CHOICE_LABELS[:score_count]
        if upper in labels:
            return labels.index(upper)
        if text in choices:
            return choices.index(text)
    if isinstance(value, (list, tuple)) and value:
        return _coerce_choice_index(value[0], choices=choices, score_count=score_count)
    return None


def _one_line_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _compact_mmlu_sample(task_name: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        raise TypeError(f"MMLU sample must be a dict, got {type(sample).__name__}")

    doc = sample.get("doc", {})
    if not isinstance(doc, dict):
        doc = {}
    choices_raw = doc.get("choices", [])
    choices = [str(item) for item in choices_raw] if isinstance(choices_raw, list) else []

    responses = sample.get("filtered_resps")
    if not isinstance(responses, list) or not responses:
        responses = sample.get("resps")
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"{task_name}: MMLU sample has no filtered_resps/resps.")

    scores = [_extract_response_score(response) for response in responses]
    if len(scores) > len(_CHOICE_LABELS):
        raise ValueError(f"{task_name}: too many MMLU choices to label: {len(scores)}")

    labels = _CHOICE_LABELS[: len(scores)]
    prediction_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    target_idx = _coerce_choice_index(
        sample.get("target", doc.get("answer")),
        choices=choices,
        score_count=len(scores),
    )
    prediction = labels[prediction_idx]
    target = labels[target_idx] if target_idx is not None else None
    correct = bool(prediction_idx == target_idx) if target_idx is not None else None

    return {
        "task": str(task_name),
        "doc_id": sample.get("doc_id"),
        "question": doc.get("question"),
        "choices": {labels[idx]: choices[idx] if idx < len(choices) else None for idx in range(len(scores))},
        "prediction": prediction,
        "prediction_index": int(prediction_idx),
        "target": target,
        "target_index": None if target_idx is None else int(target_idx),
        "correct": correct,
        "choice_scores": {labels[idx]: float(scores[idx]) for idx in range(len(scores))},
        "metrics": {
            key: value
            for key, value in sample.items()
            if key not in {"doc", "arguments", "resps", "filtered_resps"}
            and isinstance(value, (bool, int, float))
        },
    }


def _iter_mmlu_debug_samples(samples: Any):
    if not isinstance(samples, dict):
        return
    for task_name in sorted(samples.keys()):
        task_text = str(task_name)
        if task_text != "mmlu" and not task_text.startswith("mmlu_"):
            continue
        task_samples = samples.get(task_name)
        if not isinstance(task_samples, list):
            continue
        for sample in task_samples:
            yield task_text, sample


def write_mmlu_debug_samples(
    lm_result: Dict[str, Any],
    *,
    limit: int,
    output_dir: str,
    run_ts: Optional[str] = None,
    log=logger,
) -> Optional[Dict[str, Any]]:
    sample_limit = int(limit)
    if sample_limit <= 0:
        return None

    os.makedirs(output_dir, exist_ok=True)
    ts = str(run_ts or time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    json_path = os.path.join(output_dir, f"mmlu_debug_samples_{ts}.json")

    compact_samples: List[Dict[str, Any]] = []
    for task_name, sample in _iter_mmlu_debug_samples(lm_result.get("samples", {})):
        compact = _compact_mmlu_sample(task_name, sample)
        compact["raw_sample"] = sample
        compact_samples.append(compact)
        if len(compact_samples) >= sample_limit:
            break

    payload = {
        "run_ts": ts,
        "limit": sample_limit,
        "sample_count": len(compact_samples),
        "samples": compact_samples,
    }
    _json_dump(json_path, payload)

    for idx, sample in enumerate(compact_samples, start=1):
        scores = " ".join(
            f"{label}={score:.4f}" for label, score in sample.get("choice_scores", {}).items()
        )
        choices = " | ".join(
            f"{label}. {_one_line_text(text, max_chars=120)}"
            for label, text in sample.get("choices", {}).items()
        )
        log.info(
            "[mmlu_debug] sample %d/%d task=%s doc_id=%s pred=%s target=%s correct=%s scores=%s question=%s choices=%s",
            idx,
            sample_limit,
            sample.get("task"),
            str(sample.get("doc_id")),
            sample.get("prediction"),
            sample.get("target"),
            str(sample.get("correct")),
            scores,
            _one_line_text(sample.get("question"), max_chars=220),
            choices,
        )
    log.info("[mmlu_debug] Saved samples: %s", json_path)
    return {"path": json_path, "sample_count": len(compact_samples), "limit": sample_limit}


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

    results_dict = results.get("results", {})
    groups_dict = results.get("groups", {})
    logger.info("LM-Eval task keys: %s", ",".join(sorted(results_dict.keys())))
    if groups_dict:
        logger.info("LM-Eval group keys: %s", ",".join(sorted(groups_dict.keys())))

    metric_vals = {}
    task_metric_keys = {}
    summary_rows = _build_lm_eval_summary_rows(
        task_names=task_names,
        results=results_dict,
        groups=groups_dict,
    )
    for row in summary_rows:
        task = str(row["task"])
        key = str(row["metric_key"])
        metric = row["metric"]
        task_metric_keys[task] = key
        metric_vals[task] = metric

    for task, result in metric_vals.items():
        if result is None:
            logger.info(f'Task {task} metric: N/A')
        else:
            logger.info(f'Task {task} acc: {result * 100 :.2f}')

    table_headers = ["Task", "Metric", "Score(%)"]
    table_rows = [
        [str(row["task"]), str(row["metric_key"]), str(row["score_percent"])]
        for row in summary_rows
    ]
    summary_table = _format_markdown_table(table_headers, table_rows)
    logger.info("LM-Eval summary table:\n%s", summary_table)

    artifact_payload = {
        "tasks": task_names,
        "num_fewshot": int(getattr(args, "num_fewshot", 0)),
        "batch_size": batch_size,
        "limit": lm_limit,
        "summary_rows": summary_rows,
        "summary_table": summary_table,
        "task_metric_keys": task_metric_keys,
        "task_metrics": metric_vals,
        "raw_results": results_dict,
        "group_results": groups_dict,
        "configs": results.get("configs", {}),
        "versions": results.get("versions", {}),
        "higher_is_better": results.get("higher_is_better", {}),
        "n_shot": results.get("n-shot", {}),
        "n_samples": results.get("n-samples", {}),
    }

    mmlu_debug = write_mmlu_debug_samples(
        {**artifact_payload, "samples": results.get("samples", {})},
        limit=int(getattr(args, "mmlu_debug_samples", 0) or 0),
        output_dir=str(getattr(args, "mmlu_debug_log_dir", None) or getattr(args, "eval_log_dir", None) or "./eval_log"),
        run_ts=getattr(args, "mmlu_debug_run_ts", None) or getattr(args, "eval_run_ts", None),
        log=logger,
    )
    if mmlu_debug is not None:
        artifact_payload["mmlu_debug"] = mmlu_debug

    eval_log_dir = getattr(args, "eval_log_dir", None)
    eval_run_ts = getattr(args, "eval_run_ts", None)
    if isinstance(eval_log_dir, str) and eval_log_dir.strip():
        ts = str(eval_run_ts or time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        json_path = os.path.join(eval_log_dir, f"lm_eval_results_{ts}.json")
        table_path = os.path.join(eval_log_dir, f"lm_eval_summary_{ts}.md")
        _json_dump(json_path, artifact_payload)
        with open(table_path, "w", encoding="utf-8") as handle:
            handle.write(summary_table)
            handle.write("\n")
        logger.info("Saved LM-Eval artifacts: json=%s table=%s", json_path, table_path)

    return {
        **artifact_payload,
    }


def evaluate_model(model, tokenizer, args):
    summary: Dict[str, Any] = {}
    # Ensure model is on GPU
    if model.device.type == 'cpu':
        logger.info("Moving model to CUDA...")
        model.to("cuda")

    # ============================ Evaluation
    if args.eval_mse:
        calculate_mse_per_weight(model, args)
        summary["mse"] = True

    if args.eval_ppl:
        summary["ppl"] = calculate_ppl(model, args)

    # LM Eval Harness
    if args.tasks:
        summary["lm_eval"] = run_lm_eval(model, tokenizer, args)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Wikitext-2 PPL and LM-Eval Tasks")

    parser.add_argument('--model_path', type=str, required=True, help='Path to the model or HF hub ID')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--eval_ppl', action='store_true', help='Evaluate Wikitext-2 PPL')
    parser.add_argument('--limit', type=int, default=-1, help='Limit samples for PPL eval')
    parser.add_argument('--eval_mse', action='store_true', help='Evaluate MSE against reference model')
    parser.add_argument('--ref_model_path', type=str, default=None, help='Path to reference model for MSE evaluation')

    # LM Eval args
    parser.add_argument('--tasks', type=str, default=None,
                        help='Comma separated list of tasks for lm-eval (e.g. piqa,arc_easy)')
    parser.add_argument('--num_fewshot', type=int, default=0, help='Number of few-shot examples')
    parser.add_argument('--batch_size', type=str, default='auto', help='Batch size for eval')
    parser.add_argument('--eval_log_dir', type=str, default='./eval_log', help='Directory to store evaluation logs and summaries')

    args = parser.parse_args()
    eval_run_ts, log_path = _build_logger(args.eval_log_dir)
    args.eval_run_ts = eval_run_ts
    logger.info("Eval log file: %s", log_path)
    logger.info("Input args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2, default=str))

    set_seed(args.seed)

    logger.info(f"Loading model from {args.model_path} ...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    # Load Model
    # Determine device map based on availability
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    model = get_model(args.model_path)

    summary = evaluate_model(model, tokenizer, args)
    summary_path = os.path.join(args.eval_log_dir, f"eval_summary_{eval_run_ts}.json")
    _json_dump(summary_path, summary)
    logger.info("Saved evaluation summary: %s", summary_path)

    logger.info("Evaluation Complete.")


if __name__ == "__main__":
    main()
