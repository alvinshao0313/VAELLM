from typing import Any, List, Optional

import torch
from torch import nn

from train_utils.eval_utils import calculate_ppl, merge_lm_eval_results, run_lm_eval
from train_utils.hif4_act import applied_hif4_act
from train_utils.lm_eval_partial_io import (
    cleanup_lm_eval_partial_dir,
    exchange_lm_eval_partial_via_files,
    lm_eval_partial_dir,
    prepare_lm_eval_partial_dir,
)
from train_utils.lora_utils import (
    distill_distributed_barrier,
    distill_rank,
    distill_world_size,
    get_distill_local_device,
    is_distill_distributed,
    is_distill_main_process,
    split_tasks_for_distill_rank,
    unwrap_distill_model,
)
from train_utils.utils import clone_namespace


def _log_task_metrics(
    *,
    category: str,
    task_names: List[str],
    lm_result: dict,
    logger: Any,
) -> None:
    task_metrics = lm_result.get("task_metrics", {})
    task_metric_keys = lm_result.get("task_metric_keys", {})
    valid_metrics: List[float] = []
    for task_name in task_names:
        metric = task_metrics.get(task_name)
        metric_key = str(task_metric_keys.get(task_name, "n/a"))
        if metric is None:
            logger.info("类别 %s 下游任务 %s: %s = N/A", category, task_name, metric_key)
            continue
        metric_val = float(metric)
        valid_metrics.append(metric_val)
        logger.info(
            "类别 %s 下游任务 %s: %s = %.4f (%.2f%%)",
            category,
            task_name,
            metric_key,
            metric_val,
            metric_val * 100.0,
        )
    if not valid_metrics:
        raise ValueError(f"类别 {category} 的下游任务评估全部为 N/A，无法计算均值。")
    mean_metric = sum(valid_metrics) / float(len(valid_metrics))
    logger.info("类别 %s 下游任务均值: %.4f (%.2f%%)", category, mean_metric, mean_metric * 100.0)


def _run_lm_eval_tasks(
    *,
    model: nn.Module,
    tokenizer: object,
    vae_args: Any,
    task_names: List[str],
    category: str,
    logger: Any,
    eval_hif4_act: bool,
) -> dict:
    logger.info("开始类别 %s 的下游任务评估: %s", category, ",".join(task_names))
    with applied_hif4_act(
        model,
        enabled=bool(eval_hif4_act),
        logger=logger,
        log_prefix=f"[lm_eval:{category}] ",
    ):
        with torch.no_grad():
            lm_args = clone_namespace(
                vae_args,
                tasks=",".join(task_names),
                num_fewshot=0,
                batch_size="auto",
                lm_limit=None,
                eval_log_dir=None,
                eval_run_ts=None,
            )
            return run_lm_eval(model, tokenizer, lm_args)


def _eval_after_category_distributed(
    *,
    model: nn.Module,
    vae_args: Any,
    ppl_limit: int,
    category: str,
    logger: Any,
    eval_device: str,
    eval_hif4_act: bool,
    eval_ppl: bool,
    task_names: List[str],
    tokenizer: object,
    run_output_dir: str,
) -> None:
    rank = distill_rank()
    world_size = distill_world_size()
    local_device = get_distill_local_device(fallback=eval_device)
    eval_model = unwrap_distill_model(model)
    tag = f"after_{category}"
    partial_dir = lm_eval_partial_dir(str(run_output_dir), tag)

    distill_distributed_barrier()
    if bool(eval_ppl):
        if is_distill_main_process():
            logger.info("开始类别 %s 的 PPL 评估...", category)
            eval_model.eval()
            eval_model.to(local_device)
            with applied_hif4_act(
                eval_model,
                enabled=bool(eval_hif4_act),
                logger=logger,
                log_prefix=f"[ppl:{category}] ",
            ):
                with torch.no_grad():
                    ppl_args = clone_namespace(vae_args, limit=int(ppl_limit))
                    ppl_result = calculate_ppl(eval_model, ppl_args)
            logger.info("类别 %s 训练后 PPL: %.2f", category, float(ppl_result.get("wiki_ppl", float("nan"))))
        distill_distributed_barrier()

    if not task_names:
        return

    if is_distill_main_process():
        prepare_lm_eval_partial_dir(partial_dir)
    distill_distributed_barrier()

    local_tasks = split_tasks_for_distill_rank(task_names, rank=rank, world_size=world_size)
    logger.info(
        "[rank=%d] 类别 %s 分布式 lm_eval 任务分配: %s",
        int(rank),
        category,
        ",".join(local_tasks) if local_tasks else "(none)",
    )

    eval_model.eval()
    eval_model.to(local_device)
    if local_tasks:
        partial_result = _run_lm_eval_tasks(
            model=eval_model,
            tokenizer=tokenizer,
            vae_args=vae_args,
            task_names=local_tasks,
            category=category,
            logger=logger,
            eval_hif4_act=eval_hif4_act,
        )
    else:
        partial_result = {"task_metrics": {}, "task_metric_keys": {}}

    try:
        gathered = exchange_lm_eval_partial_via_files(
            partial_result,
            run_output_dir=str(run_output_dir),
            tag=tag,
            rank=int(rank),
            world_size=int(world_size),
            is_main=is_distill_main_process(),
        )
        if is_distill_main_process():
            merged_result = merge_lm_eval_results(gathered or [], task_names)
            summary_table = merged_result.get("summary_table")
            if isinstance(summary_table, str) and summary_table.strip():
                logger.info("类别 %s LM-Eval summary table:\n%s", category, summary_table)
            _log_task_metrics(
                category=category,
                task_names=task_names,
                lm_result=merged_result,
                logger=logger,
            )
    finally:
        if is_distill_main_process():
            cleanup_lm_eval_partial_dir(partial_dir)

    distill_distributed_barrier()


def eval_after_category(
    *,
    model: nn.Module,
    vae_args: Any,
    ppl_limit: int,
    category: str,
    logger: Any,
    eval_device: str = "cuda",
    eval_hif4_act: bool = False,
    eval_ppl: bool = True,
    eval_tasks: str = "",
    tokenizer: Optional[object] = None,
    move_model_to_cpu_after_eval: bool = True,
    run_output_dir: Optional[str] = None,
) -> None:
    run_ppl = bool(eval_ppl)
    task_names = [task.strip() for task in str(eval_tasks).split(",") if task.strip()]
    run_tasks = len(task_names) > 0
    if not run_ppl and not run_tasks:
        if is_distill_main_process():
            logger.info("类别 %s 训练后评估已跳过：--eval_ppl=false 且 --eval_tasks 为空。", category)
        return
    if run_tasks and tokenizer is None:
        raise ValueError(f"类别 {category} 启用了 --eval_tasks，但 tokenizer 未提供。")

    resolved_run_output_dir = None if run_output_dir is None else str(run_output_dir).strip() or None

    if is_distill_distributed():
        if run_tasks and not resolved_run_output_dir:
            raise ValueError(
                f"类别 {category} 的分布式 lm_eval 需要非空 run_output_dir（本次 run 日志目录）。"
            )
        if is_distill_main_process():
            logger.info(
                "开始类别 %s 的训练后评估(分布式任务并行): eval_ppl=%s eval_tasks=%s world_size=%d",
                category,
                str(run_ppl).lower(),
                ",".join(task_names) if run_tasks else "",
                int(distill_world_size()),
            )
        _eval_after_category_distributed(
            model=model,
            vae_args=vae_args,
            ppl_limit=ppl_limit,
            category=category,
            logger=logger,
            eval_device=eval_device,
            eval_hif4_act=eval_hif4_act,
            eval_ppl=run_ppl,
            task_names=task_names,
            tokenizer=tokenizer,
            run_output_dir=str(resolved_run_output_dir or ""),
        )
        return

    logger.info(
        "开始类别 %s 的训练后评估: eval_ppl=%s eval_tasks=%s",
        category,
        str(run_ppl).lower(),
        ",".join(task_names) if run_tasks else "",
    )
    model.eval()
    model.to(eval_device)
    try:
        if run_ppl:
            logger.info("开始类别 %s 的 PPL 评估...", category)
            with applied_hif4_act(
                model,
                enabled=bool(eval_hif4_act),
                logger=logger,
                log_prefix=f"[ppl:{category}] ",
            ):
                with torch.no_grad():
                    ppl_args = clone_namespace(vae_args, limit=int(ppl_limit))
                    ppl_result = calculate_ppl(model, ppl_args)
            logger.info("类别 %s 训练后 PPL: %.2f", category, float(ppl_result.get("wiki_ppl", float("nan"))))

        if run_tasks:
            lm_result = _run_lm_eval_tasks(
                model=model,
                tokenizer=tokenizer,
                vae_args=vae_args,
                task_names=task_names,
                category=category,
                logger=logger,
                eval_hif4_act=eval_hif4_act,
            )
            _log_task_metrics(
                category=category,
                task_names=task_names,
                lm_result=lm_result,
                logger=logger,
            )
    finally:
        if bool(move_model_to_cpu_after_eval):
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
