import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from e2e_common.checkpoint_io import load_e2e_model_checkpoint
from e2e_common.peft_proxy import PeftVAELinearProxy
from e2e_common.temporary_mode import set_model_temporary
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import clear_model_vae_linear_cache
from train_utils.block_distill import (
    QWEN3_BLOCK_CATEGORIES,
    get_module_by_name,
    prepare_block_eval_decoded_weights,
    validate_block_categories,
)
from train_utils.eval_utils import run_lm_eval
from train_utils.model_checkpoint_io import META_FILENAME, resolve_checkpoint_dir


DEFAULT_TASKS = "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"
BLOCK_FINAL_STAGE = "block_vae_lora_final"


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
    log_path = os.path.join(log_dir, f"block_prefix_eval_{ts}.log")

    logger = logging.getLogger("block_prefix_eval")
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


def _extra_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    extra = meta.get("extra_meta", {})
    if not isinstance(extra, dict):
        raise TypeError("checkpoint_meta.extra_meta must be a dict.")
    return extra


def _validate_block_final_checkpoint(meta: Dict[str, Any], checkpoint_dir: str) -> None:
    stage = str(_extra_meta(meta).get("stage", "")).strip()
    if stage != BLOCK_FINAL_STAGE:
        raise ValueError(
            f"Expected a block final checkpoint at {checkpoint_dir}, "
            f"got extra_meta.stage={stage!r}."
        )


def _resolve_block_categories(meta: Dict[str, Any]) -> Tuple[str, ...]:
    extra = _extra_meta(meta)
    value = extra.get("block_vae_categories")
    if value is None:
        block_distill = extra.get("block_distill", {})
        if isinstance(block_distill, dict):
            value = block_distill.get("block_vae_categories")
    if value is None:
        return tuple(QWEN3_BLOCK_CATEGORIES)
    if not isinstance(value, list):
        raise TypeError("checkpoint extra_meta.block_vae_categories must be a list.")
    categories = tuple(str(item) for item in value)
    invalid = [category for category in categories if category not in QWEN3_BLOCK_CATEGORIES]
    if invalid:
        raise ValueError(f"Invalid block categories in checkpoint: {invalid}")
    return categories


def _resolve_block_train_mode(meta: Dict[str, Any]) -> str:
    extra = _extra_meta(meta)
    train_mode = str(extra.get("block_distill_train_mode", "")).strip().lower()
    if not train_mode:
        block_distill = extra.get("block_distill", {})
        if isinstance(block_distill, dict):
            train_mode = str(block_distill.get("block_distill_train_mode", "")).strip().lower()
    if not train_mode:
        raise ValueError("Cannot determine block_distill_train_mode from checkpoint metadata.")
    return train_mode


def _resolve_num_layers(model: torch.nn.Module) -> int:
    config_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if config_layers is not None:
        return int(config_layers)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Model does not expose config.num_hidden_layers or model.layers.")
    return int(len(layers))


def _task_names(tasks: str) -> List[str]:
    names = [task.strip() for task in str(tasks).split(",") if task.strip()]
    if not names:
        raise ValueError("--tasks must contain at least one lm-eval task.")
    return names


def build_prefix_active_targets(
    *,
    prefix_layers: int,
    num_layers: int,
    categories: Sequence[str],
) -> List[Tuple[int, str]]:
    prefix = int(prefix_layers)
    total = int(num_layers)
    if prefix < 0 or prefix > total:
        raise ValueError(f"prefix_layers must be in [0, {total}], got {prefix_layers}.")
    return [(layer_idx, str(category)) for layer_idx in range(prefix) for category in categories]


def summarize_task_metrics(
    *,
    task_names: Sequence[str],
    lm_result: Dict[str, Any],
) -> Dict[str, Any]:
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

    mean_metric = None if not valid_metrics else sum(valid_metrics) / float(len(valid_metrics))
    return {
        "rows": rows,
        "valid_task_count": int(len(valid_metrics)),
        "missing_task_count": int(len(rows) - len(valid_metrics)),
        "mean_metric": mean_metric,
    }


def build_prefix_result_row(
    *,
    prefix_layers: int,
    num_layers: int,
    active_target_count: int,
    prep_stats: Dict[str, Any],
    task_names: Sequence[str],
    lm_result: Dict[str, Any],
) -> Dict[str, Any]:
    task_summary = summarize_task_metrics(task_names=task_names, lm_result=lm_result)
    return {
        "prefix_layers": int(prefix_layers),
        "num_layers": int(num_layers),
        "active_target_count": int(active_target_count),
        "prep_stats": dict(prep_stats),
        "mean_metric": task_summary["mean_metric"],
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


def _iter_expected_block_modules(
    model: torch.nn.Module,
    *,
    num_layers: int,
    categories: Sequence[str],
) -> Iterable[Tuple[str, torch.nn.Module]]:
    for layer_idx in range(int(num_layers)):
        names_by_category = validate_block_categories(model, int(layer_idx))
        for category in categories:
            module_name = names_by_category.get(str(category))
            if module_name is None:
                raise ValueError(f"Layer {layer_idx} is missing block category {category!r}.")
            yield module_name, get_module_by_name(model, module_name)


def _validate_original_weights_available(
    model: torch.nn.Module,
    *,
    num_layers: int,
    categories: Sequence[str],
) -> None:
    missing: List[str] = []
    for module_name, module in _iter_expected_block_modules(
        model,
        num_layers=int(num_layers),
        categories=categories,
    ):
        base_layer = module.base_layer if isinstance(module, PeftVAELinearProxy) else module
        if not isinstance(base_layer, VAELinear):
            raise TypeError(f"{module_name}: expected VAELinear-backed block module, got {type(module)}.")
        if getattr(base_layer, "original_weight", None) is None:
            missing.append(module_name)
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            "Block prefix eval requires original_weight for every block target. "
            f"Missing {len(missing)} targets: {preview}"
        )


def _log_prefix_result(logger: logging.Logger, row: Dict[str, Any]) -> None:
    mean_metric = row.get("mean_metric")
    mean_text = "N/A" if mean_metric is None else f"{float(mean_metric):.4f} ({float(mean_metric) * 100.0:.2f}%)"
    logger.info(
        "[prefix n=%d/%d] active_targets=%d mean=%s valid_tasks=%d missing_tasks=%d",
        int(row["prefix_layers"]),
        int(row["num_layers"]),
        int(row["active_target_count"]),
        mean_text,
        int(row["valid_task_count"]),
        int(row["missing_task_count"]),
    )
    for task_row in row["task_rows"]:
        metric = task_row["metric"]
        metric_text = "N/A" if metric is None else f"{float(metric):.4f} ({float(metric) * 100.0:.2f}%)"
        logger.info(
            "[prefix n=%d] task=%s %s=%s",
            int(row["prefix_layers"]),
            str(task_row["task"]),
            str(task_row["metric_key"]),
            metric_text,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate block-distilled checkpoints by prefix layer count.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--access_token", type=str, default=None)
    parser.add_argument("--map_location", type=str, default="cpu")
    parser.add_argument("--strict", dest="strict", action="store_true")
    parser.add_argument("--no_strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
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
    parser.add_argument("--eval_log_dir", type=str, default="./eval_log/block_prefix_eval")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.prewarm_group_size) < 1:
        raise ValueError("--prewarm_group_size must be >= 1.")
    if str(args.eval_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested eval_device={args.eval_device}, but CUDA is not available.")

    logger, log_path = _build_logger(args.eval_log_dir)
    logger.info("Block prefix eval log file: %s", log_path)
    logger.info("Input args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    meta_preview = _read_checkpoint_meta(checkpoint_dir)
    _validate_block_final_checkpoint(meta_preview, checkpoint_dir)
    categories = _resolve_block_categories(meta_preview)
    train_mode = _resolve_block_train_mode(meta_preview)
    base_model_path = args.base_model_path or meta_preview.get("base_model_path")
    if not base_model_path:
        raise ValueError("Cannot determine base_model_path. Provide --base_model_path.")
    task_names = _task_names(args.tasks)

    results_path = os.path.join(args.eval_log_dir, "prefix_eval_results.jsonl")
    summary_path = os.path.join(args.eval_log_dir, "prefix_eval_summary.json")
    os.makedirs(args.eval_log_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8"):
        pass

    logger.info("Loading block final checkpoint: %s", checkpoint_dir)
    model, meta, load_result = load_e2e_model_checkpoint(
        checkpoint_dir,
        access_token=args.access_token,
        base_model_path=args.base_model_path,
        map_location=args.map_location,
        strict=bool(args.strict),
        materialize_proxy_decoded_linears=train_mode == "lora",
        proxy_group_size=int(args.prewarm_group_size),
        proxy_compute_device=args.eval_device if train_mode == "lora" else None,
        proxy_logger=logger,
        preserve_original_weights_from_base=True,
    )
    logger.info(
        "Checkpoint loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s adapter_module_count=%s train_mode=%s",
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
        str(meta.get("converted_module_count")),
        str(meta.get("adapter_module_count", 0)),
        train_mode,
    )

    set_model_temporary(model, True)
    clear_model_vae_linear_cache(model)
    num_layers = _resolve_num_layers(model)
    _validate_original_weights_available(model, num_layers=num_layers, categories=categories)
    logger.info(
        "Block prefix sweep ready: num_layers=%d categories=%s tasks=%s results=%s",
        int(num_layers),
        ",".join(categories),
        ",".join(task_names),
        results_path,
    )

    from transformers import AutoTokenizer
    from train_utils.hif4_act import applied_hif4_act

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        use_fast=False,
        trust_remote_code=True,
        token=args.access_token,
    )
    all_rows: List[Dict[str, Any]] = []
    for prefix_layers in range(int(num_layers) + 1):
        active_targets = build_prefix_active_targets(
            prefix_layers=prefix_layers,
            num_layers=num_layers,
            categories=categories,
        )
        logger.info(
            "[prefix n=%d/%d] starting lm_eval with active_targets=%d",
            int(prefix_layers),
            int(num_layers),
            len(active_targets),
        )
        with prepare_block_eval_decoded_weights(
            model=model,
            eval_device=str(args.eval_device),
            group_size=int(args.prewarm_group_size),
            train_mode=train_mode,
            active_block_targets=active_targets,
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
                log_prefix=f"[prefix n={int(prefix_layers)}] ",
            ):
                lm_result = run_lm_eval(model, tokenizer, lm_args)

        row = build_prefix_result_row(
            prefix_layers=prefix_layers,
            num_layers=num_layers,
            active_target_count=len(active_targets),
            prep_stats=prep_stats,
            task_names=task_names,
            lm_result=lm_result,
        )
        if int(row["valid_task_count"]) <= 0:
            raise ValueError(f"prefix_layers={prefix_layers} produced no valid lm-eval metrics.")
        _log_prefix_result(logger, row)
        _jsonl_append(results_path, row)
        all_rows.append(row)

    best_row = max(all_rows, key=lambda item: float(item["mean_metric"]))
    summary = {
        "checkpoint_dir": checkpoint_dir,
        "base_model_path": str(base_model_path),
        "num_layers": int(num_layers),
        "categories": list(categories),
        "tasks": task_names,
        "train_mode": train_mode,
        "results_path": results_path,
        "log_path": log_path,
        "best_prefix_layers": int(best_row["prefix_layers"]),
        "best_mean_metric": float(best_row["mean_metric"]),
        "rows": all_rows,
    }
    _json_dump(summary_path, summary)
    logger.info("Block prefix eval summary written: %s", summary_path)
    logger.info("Block prefix eval completed.")


if __name__ == "__main__":
    main()
