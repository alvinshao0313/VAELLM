import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from litebsq.vae_linear import NamedVAELinearTarget, VAELinear, prime_named_vae_linear_cache
from train_utils.eval_utils import run_lm_eval
from train_utils.model_checkpoint_io import META_FILENAME, load_model_checkpoint, resolve_checkpoint_dir


DEFAULT_CATEGORY_SWEEP = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_TASKS = "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"
QWEN_CATEGORY_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _parse_bool_like(value, *, arg_name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid {arg_name} value {value!r}. Expected bool.")


def _build_logger(log_dir: str) -> Tuple[logging.Logger, str]:
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(log_dir, f"cat_category_prefix_eval_{ts}.log")

    logger = logging.getLogger("cat_category_prefix_eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_file_handler = logging.FileHandler(log_path)
    root_file_handler.setFormatter(formatter)
    root_logger.addHandler(root_file_handler)
    return logger, log_path


def _json_dump(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _jsonl_append(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str))
        handle.write("\n")


def _read_checkpoint_meta(checkpoint_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(str(checkpoint_dir), META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing checkpoint meta: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise TypeError(f"Checkpoint meta must be a dict, got {type(meta)}.")
    return meta


def _category_from_module_name(module_name: str) -> str:
    return str(module_name).rsplit(".", 1)[-1]


def _module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    module: torch.nn.Module = model
    for part in str(module_name).split("."):
        module = getattr(module, part)
    return module


def _task_names(tasks: str) -> List[str]:
    names = [task.strip() for task in str(tasks).split(",") if task.strip()]
    if not names:
        raise ValueError("--tasks must contain at least one lm-eval task.")
    return names


def parse_category_sweep(value: str, *, valid_categories: Sequence[str]) -> Tuple[str, ...]:
    categories = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not categories:
        raise ValueError("--category_sweep must contain at least one category.")

    valid = tuple(str(item) for item in valid_categories)
    valid_set = set(valid)
    seen = set()
    duplicates: List[str] = []
    unknown: List[str] = []
    for category in categories:
        if category in seen:
            duplicates.append(category)
        seen.add(category)
        if category not in valid_set:
            unknown.append(category)
    if duplicates:
        raise ValueError(f"--category_sweep contains duplicate categories: {duplicates}")
    if unknown:
        raise ValueError(f"--category_sweep contains unknown categories: {unknown}. Valid categories: {list(valid)}")
    return categories


def build_prefix_active_categories(*, prefix_index: int, category_sweep: Sequence[str]) -> Tuple[str, ...]:
    prefix = int(prefix_index)
    total = len(category_sweep)
    if prefix < 0 or prefix > total:
        raise ValueError(f"prefix_index must be in [0, {total}], got {prefix_index}.")
    return tuple(str(item) for item in category_sweep[:prefix])


def _resolve_targets_by_category_from_meta(meta: Dict[str, Any]) -> Dict[str, List[str]]:
    converted_modules = meta.get("converted_modules", [])
    if not isinstance(converted_modules, list) or not converted_modules:
        raise ValueError("Cat category prefix eval requires checkpoint_meta.converted_modules.")

    targets: Dict[str, List[str]] = {}
    for spec in converted_modules:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "")).strip()
        if not name:
            continue
        category = _category_from_module_name(name)
        if category not in QWEN_CATEGORY_ORDER:
            continue
        targets.setdefault(category, []).append(name)
    if not targets:
        raise ValueError("No supported Qwen projection categories found in checkpoint converted_modules.")
    return targets


def _ordered_categories_from_targets(targets_by_category: Dict[str, List[str]]) -> Tuple[str, ...]:
    return tuple(category for category in QWEN_CATEGORY_ORDER if targets_by_category.get(category))


def _validate_cat_checkpoint(meta: Dict[str, Any], checkpoint_dir: str) -> None:
    adapter_count = int(meta.get("adapter_module_count", 0) or 0)
    adapter_modules = meta.get("adapter_modules")
    extra = meta.get("extra_meta", {}) if isinstance(meta.get("extra_meta"), dict) else {}
    stage = str(extra.get("stage", "")).strip().lower()
    if adapter_count > 0 or (isinstance(adapter_modules, list) and adapter_modules):
        raise ValueError("Cat category prefix eval does not support adapter/e2e checkpoints.")
    if stage in {"block_vae_lora_final", "e2e_fintuning", "dense_e2e_fintuning"}:
        raise ValueError(
            f"Cat category prefix eval expects a cat final_model checkpoint, got extra_meta.stage={stage!r} "
            f"at {checkpoint_dir}."
        )


def _validate_original_weights_available(
    model: torch.nn.Module,
    *,
    targets_by_category: Dict[str, List[str]],
) -> None:
    missing: List[str] = []
    for module_names in targets_by_category.values():
        for module_name in module_names:
            module = _module_by_name(model, module_name)
            if not isinstance(module, VAELinear):
                raise TypeError(f"{module_name}: expected VAELinear, got {type(module).__name__}.")
            if getattr(module, "original_weight", None) is None:
                missing.append(module_name)
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            "Cat category prefix eval requires original_weight for every converted target. "
            f"Missing {len(missing)} targets: {preview}"
        )


def build_category_mode_summary(
    *,
    prefix_index: int,
    category_sweep: Sequence[str],
    all_categories: Sequence[str],
    targets_by_category: Dict[str, List[str]],
) -> Dict[str, Any]:
    compressed = list(build_prefix_active_categories(prefix_index=prefix_index, category_sweep=category_sweep))
    compressed_set = set(compressed)
    original = [str(category) for category in all_categories if str(category) not in compressed_set]
    compressed_targets = [
        module_name
        for category in compressed
        for module_name in targets_by_category.get(str(category), [])
    ]
    original_targets = [
        module_name
        for category in original
        for module_name in targets_by_category.get(str(category), [])
    ]
    return {
        "compressed_categories": compressed,
        "original_categories": original,
        "compressed_target_count": int(len(compressed_targets)),
        "original_target_count": int(len(original_targets)),
        "compressed_examples": compressed_targets[:8],
        "original_examples": original_targets[:8],
    }


@contextmanager
def category_eval_weight_scope(
    *,
    model: torch.nn.Module,
    targets_by_category: Dict[str, List[str]],
    active_categories: Sequence[str],
    eval_device: str,
    group_size: int,
    logger: Optional[logging.Logger],
) -> Iterator[Dict[str, Any]]:
    active_set = {str(category) for category in active_categories}
    previous_states: List[Tuple[VAELinear, bool]] = []
    touched: List[VAELinear] = []
    active_targets: List[NamedVAELinearTarget] = []

    try:
        for category, module_names in targets_by_category.items():
            is_active = str(category) in active_set
            for module_name in module_names:
                module = _module_by_name(model, module_name)
                if not isinstance(module, VAELinear):
                    raise TypeError(f"{module_name}: expected VAELinear, got {type(module).__name__}.")
                previous_states.append((module, bool(getattr(module, "temporary", True))))
                touched.append(module)
                module.set_temporary(is_active)
                module.clear_decoded_weight_cache()
                if is_active:
                    active_targets.append(NamedVAELinearTarget(name=module_name, base_layer=module))

        stats: Dict[str, Any]
        if active_targets:
            prewarm_stats = prime_named_vae_linear_cache(
                active_targets,
                group_size=int(group_size),
                compute_device=str(eval_device),
                logger=logger,
            )
            stats = dict(prewarm_stats)
        else:
            stats = {"total": 0, "warmed": 0, "skipped": 0, "failed": 0}
        stats["active_target_count"] = int(len(active_targets))
        yield stats
    finally:
        for module, temporary in previous_states:
            module.set_temporary(temporary)
        for module in touched:
            module.clear_decoded_weight_cache()


def summarize_task_metrics(*, task_names: Sequence[str], lm_result: Dict[str, Any]) -> Dict[str, Any]:
    task_metrics = lm_result.get("task_metrics", {})
    task_metric_keys = lm_result.get("task_metric_keys", {})
    if not isinstance(task_metrics, dict):
        raise TypeError("lm_result['task_metrics'] must be a dict.")
    if not isinstance(task_metric_keys, dict):
        raise TypeError("lm_result['task_metric_keys'] must be a dict.")

    rows: List[Dict[str, Any]] = []
    valid_metrics: List[float] = []
    for task_name in task_names:
        metric = task_metrics.get(str(task_name))
        metric_key = str(task_metric_keys.get(str(task_name), "n/a"))
        metric_value: Optional[float]
        if metric is None:
            metric_value = None
        else:
            metric_value = float(metric)
            valid_metrics.append(metric_value)
        rows.append(
            {
                "task": str(task_name),
                "metric_key": metric_key,
                "metric": metric_value,
                "score_percent": None if metric_value is None else f"{metric_value * 100.0:.2f}",
            }
        )

    average_accuracy = None if not valid_metrics else sum(valid_metrics) / float(len(valid_metrics))
    return {
        "rows": rows,
        "valid_task_count": int(len(valid_metrics)),
        "missing_task_count": int(len(rows) - len(valid_metrics)),
        "average_accuracy": average_accuracy,
    }


def build_category_result_row(
    *,
    prefix_index: int,
    category_sweep: Sequence[str],
    mode_summary: Dict[str, Any],
    prep_stats: Dict[str, Any],
    task_names: Sequence[str],
    lm_result: Dict[str, Any],
) -> Dict[str, Any]:
    task_summary = summarize_task_metrics(task_names=task_names, lm_result=lm_result)
    return {
        "prefix_index": int(prefix_index),
        "category_sweep": [str(category) for category in category_sweep],
        "compressed_categories": list(mode_summary["compressed_categories"]),
        "original_categories": list(mode_summary["original_categories"]),
        "active_target_count": int(prep_stats.get("active_target_count", 0)),
        "compressed_target_count": int(mode_summary["compressed_target_count"]),
        "original_target_count": int(mode_summary["original_target_count"]),
        "compressed_examples": list(mode_summary["compressed_examples"]),
        "original_examples": list(mode_summary["original_examples"]),
        "prep_stats": dict(prep_stats),
        "average_accuracy": task_summary["average_accuracy"],
        "valid_task_count": task_summary["valid_task_count"],
        "missing_task_count": task_summary["missing_task_count"],
        "task_rows": task_summary["rows"],
        "lm_eval": {
            "tasks": lm_result.get("tasks"),
            "num_fewshot": lm_result.get("num_fewshot"),
            "batch_size": lm_result.get("batch_size"),
            "limit": lm_result.get("limit"),
            "task_metrics": lm_result.get("task_metrics"),
            "task_metric_keys": lm_result.get("task_metric_keys"),
        },
    }


def _log_category_result(logger: logging.Logger, row: Dict[str, Any], *, total_prefixes: int) -> None:
    prefix_index = int(row["prefix_index"])
    compressed = ",".join(row["compressed_categories"])
    original = ",".join(row["original_categories"])
    average = row.get("average_accuracy")
    average_text = "N/A" if average is None else f"{float(average):.4f} ({float(average) * 100.0:.2f}%)"
    logger.info(
        "[prefix n=%d/%d] compressed_categories=%s original_categories=%s active_targets=%d",
        prefix_index,
        int(total_prefixes),
        compressed,
        original,
        int(row["active_target_count"]),
    )
    logger.info("[prefix n=%d/%d] average_accuracy=%s", prefix_index, int(total_prefixes), average_text)
    for task_row in row["task_rows"]:
        metric = task_row["metric"]
        metric_text = "N/A" if metric is None else f"{float(metric):.4f} ({float(metric) * 100.0:.2f}%)"
        logger.info(
            "[prefix n=%d/%d] task=%s %s=%s",
            prefix_index,
            int(total_prefixes),
            str(task_row["task"]),
            str(task_row["metric_key"]),
            metric_text,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate cat final_model by cumulative compressed category sweep.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--access_token", type=str, default=None)
    parser.add_argument("--map_location", type=str, default="cpu")
    parser.add_argument("--strict", dest="strict", action="store_true")
    parser.add_argument("--no_strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    parser.add_argument("--category_sweep", type=str, default=DEFAULT_CATEGORY_SWEEP)
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--lm_batch_size", type=str, default="auto")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--lm_limit", type=int, default=None)
    parser.add_argument("--prewarm_group_size", type=int, default=8)
    parser.add_argument(
        "--eval_hif4_act",
        type=lambda v: _parse_bool_like(v, arg_name="--eval_hif4_act"),
        default=False,
    )
    parser.add_argument("--eval_log_dir", type=str, default="./eval_log/cat_category_prefix_eval")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.prewarm_group_size) < 1:
        raise ValueError("--prewarm_group_size must be >= 1.")
    if str(args.eval_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested eval_device={args.eval_device}, but CUDA is not available.")

    logger, log_path = _build_logger(args.eval_log_dir)
    logger.info("Cat category prefix eval log file: %s", log_path)
    logger.info("Input args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    meta_preview = _read_checkpoint_meta(checkpoint_dir)
    _validate_cat_checkpoint(meta_preview, checkpoint_dir)
    targets_by_category = _resolve_targets_by_category_from_meta(meta_preview)
    all_categories = _ordered_categories_from_targets(targets_by_category)
    category_sweep = parse_category_sweep(args.category_sweep, valid_categories=all_categories)
    task_names = _task_names(args.tasks)
    base_model_path = args.base_model_path or meta_preview.get("base_model_path")
    if not base_model_path:
        raise ValueError("Cannot determine base_model_path. Provide --base_model_path.")

    results_path = os.path.join(args.eval_log_dir, "category_prefix_eval_results.jsonl")
    summary_path = os.path.join(args.eval_log_dir, "category_prefix_eval_summary.json")
    os.makedirs(args.eval_log_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8"):
        pass

    logger.info("Loading cat checkpoint with original weights preserved: %s", checkpoint_dir)
    model, meta, load_result = load_model_checkpoint(
        checkpoint_dir,
        access_token=args.access_token,
        base_model_path=args.base_model_path,
        map_location=args.map_location,
        strict=bool(args.strict),
        preserve_original_weights_from_base=True,
    )
    logger.info(
        "Checkpoint loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s",
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
        str(meta.get("converted_module_count")),
    )
    _validate_original_weights_available(model, targets_by_category=targets_by_category)
    model.to(str(args.eval_device))
    model.eval()

    from transformers import AutoTokenizer
    from train_utils.hif4_act import applied_hif4_act

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        use_fast=False,
        trust_remote_code=True,
        token=args.access_token,
    )

    logger.info(
        "Category prefix sweep ready: all_categories=%s category_sweep=%s tasks=%s results=%s",
        ",".join(all_categories),
        ",".join(category_sweep),
        ",".join(task_names),
        results_path,
    )

    all_rows: List[Dict[str, Any]] = []
    total_prefixes = len(category_sweep)
    for prefix_index in range(total_prefixes + 1):
        active_categories = build_prefix_active_categories(prefix_index=prefix_index, category_sweep=category_sweep)
        mode_summary = build_category_mode_summary(
            prefix_index=prefix_index,
            category_sweep=category_sweep,
            all_categories=all_categories,
            targets_by_category=targets_by_category,
        )
        logger.info(
            "[prefix n=%d/%d] starting lm_eval active_categories=%s",
            int(prefix_index),
            int(total_prefixes),
            ",".join(active_categories),
        )
        with category_eval_weight_scope(
            model=model,
            targets_by_category=targets_by_category,
            active_categories=active_categories,
            eval_device=str(args.eval_device),
            group_size=int(args.prewarm_group_size),
            logger=logger,
        ) as prep_stats:
            lm_args = argparse.Namespace(
                tasks=",".join(task_names),
                num_fewshot=int(args.num_fewshot),
                batch_size=str(args.lm_batch_size),
                lm_limit=args.lm_limit,
                model_path=str(base_model_path),
                eval_log_dir=None,
                eval_run_ts=None,
            )
            with applied_hif4_act(
                model,
                enabled=bool(args.eval_hif4_act),
                logger=logger,
                log_prefix=f"[prefix n={int(prefix_index)}] ",
            ):
                lm_result = run_lm_eval(model, tokenizer, lm_args)

        row = build_category_result_row(
            prefix_index=prefix_index,
            category_sweep=category_sweep,
            mode_summary=mode_summary,
            prep_stats=prep_stats,
            task_names=task_names,
            lm_result=lm_result,
        )
        if int(row["valid_task_count"]) <= 0:
            raise ValueError(f"prefix_index={prefix_index} produced no valid lm-eval metrics.")
        _log_category_result(logger, row, total_prefixes=total_prefixes)
        _jsonl_append(results_path, row)
        all_rows.append(row)

    best_row = max(all_rows, key=lambda item: float(item["average_accuracy"]))
    summary = {
        "checkpoint_dir": checkpoint_dir,
        "base_model_path": str(base_model_path),
        "all_categories": list(all_categories),
        "category_sweep": list(category_sweep),
        "tasks": task_names,
        "results_path": results_path,
        "log_path": log_path,
        "best_prefix_index": int(best_row["prefix_index"]),
        "best_average_accuracy": float(best_row["average_accuracy"]),
        "rows": all_rows,
    }
    _json_dump(summary_path, summary)
    logger.info("Cat category prefix eval summary written: %s", summary_path)
    logger.info("Cat category prefix eval completed.")


if __name__ == "__main__":
    main()
