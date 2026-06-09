from typing import Any, List, Optional

import torch
from torch import nn

from train_utils.eval_utils import calculate_ppl, run_lm_eval
from train_utils.hif4_act import applied_hif4_act
from train_utils.utils import clone_namespace


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
) -> None:
    run_ppl = bool(eval_ppl)
    task_names = [task.strip() for task in str(eval_tasks).split(",") if task.strip()]
    run_tasks = len(task_names) > 0
    if not run_ppl and not run_tasks:
        logger.info("类别 %s 训练后评估已跳过：--eval_ppl=false 且 --eval_tasks 为空。", category)
        return
    if run_tasks and tokenizer is None:
        raise ValueError(f"类别 {category} 启用了 --eval_tasks，但 tokenizer 未提供。")

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
                    lm_result = run_lm_eval(model, tokenizer, lm_args)

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
    finally:
        if bool(move_model_to_cpu_after_eval):
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
