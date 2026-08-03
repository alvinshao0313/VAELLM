"""Mid-training / distributed lm-eval helpers for compressed_e2e_fintuning."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers import TrainerCallback

from litebsq.vae_linear import VAELinear
from train_utils.eval_utils import merge_lm_eval_results, run_lm_eval
from train_utils.hif4_act import applied_hif4_act
from train_utils.lora_utils import (
    distill_distributed_barrier,
    distill_rank,
    distill_world_size,
    get_distill_local_device,
    is_distill_distributed,
    is_distill_main_process,
    split_tasks_for_distill_rank,
)


def parse_eval_task_names(eval_tasks: Optional[str]) -> List[str]:
    if eval_tasks is None:
        return []
    return [part.strip() for part in str(eval_tasks).split(",") if part.strip()]


def _iter_trainable_decode_modules(model: nn.Module) -> Iterator[VAELinear]:
    for module in model.modules():
        if isinstance(module, VAELinear) and bool(getattr(module, "trainable_decode", False)):
            yield module


@contextmanager
def temporary_inference_decode_mode(
    model: nn.Module,
    *,
    parallel_stage_decode: bool,
    cache_decoded_weight: bool = False,
):
    """Disable trainable_decode for eval, then restore for modules that had it enabled.

    Mid-training eval keeps cache_decoded_weight=False so full dense weights are not
    retained on GPU alongside optimizer/teacher memory.
    """
    enabled_modules = list(_iter_trainable_decode_modules(model))
    was_training = bool(model.training)
    model.eval()
    for module in enabled_modules:
        module.disable_trainable_decode()
        module.cache_decoded_weight = bool(cache_decoded_weight)
        if not bool(cache_decoded_weight):
            module.clear_decoded_weight_cache()
    try:
        yield
    finally:
        for module in enabled_modules:
            module.enable_trainable_decode(parallel_stage_decode=bool(parallel_stage_decode))
        if was_training:
            model.train()
        else:
            model.eval()


def _offload_optimizer_state_to_cpu(optimizer) -> int:
    if optimizer is None:
        return 0
    moved = 0
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict):
        return 0
    for param_state in state.values():
        if not isinstance(param_state, dict):
            continue
        for key, value in list(param_state.items()):
            if torch.is_tensor(value) and value.device.type == "cuda":
                param_state[key] = value.detach().to("cpu")
                moved += 1
    return int(moved)


def _restore_optimizer_state_to_device(optimizer, device: torch.device) -> int:
    if optimizer is None or device.type == "cpu":
        return 0
    moved = 0
    state = getattr(optimizer, "state", None)
    if not isinstance(state, dict):
        return 0
    for param_state in state.values():
        if not isinstance(param_state, dict):
            continue
        for key, value in list(param_state.items()):
            if torch.is_tensor(value) and value.device != device:
                param_state[key] = value.to(device)
                moved += 1
    return int(moved)


def _log_merged_task_metrics(*, tag: str, task_names: Sequence[str], lm_result: Dict[str, Any], log) -> None:
    task_metrics = lm_result.get("task_metrics", {}) or {}
    task_metric_keys = lm_result.get("task_metric_keys", {}) or {}
    valid_metrics: List[float] = []
    for task_name in task_names:
        metric = task_metrics.get(task_name)
        metric_key = str(task_metric_keys.get(task_name, "n/a"))
        if metric is None:
            log.info("[%s] task %s: %s = N/A", tag, task_name, metric_key)
            continue
        metric_val = float(metric)
        valid_metrics.append(metric_val)
        log.info(
            "[%s] task %s: %s = %.4f (%.2f%%)",
            tag,
            task_name,
            metric_key,
            metric_val,
            metric_val * 100.0,
        )
    if not valid_metrics:
        raise ValueError(f"[{tag}] all lm-eval task metrics are N/A.")
    mean_metric = sum(valid_metrics) / float(len(valid_metrics))
    log.info("[%s] task mean: %.4f (%.2f%%)", tag, mean_metric, mean_metric * 100.0)


def _write_lm_eval_artifacts(
    *,
    result: Dict[str, Any],
    output_dir: str,
    eval_tag: str,
    log,
) -> Dict[str, str]:
    eval_log_dir = os.path.join(output_dir, "lm_eval")
    os.makedirs(eval_log_dir, exist_ok=True)
    json_path = os.path.join(eval_log_dir, f"lm_eval_results_{eval_tag}.json")
    summary_path = os.path.join(eval_log_dir, f"lm_eval_summary_{eval_tag}.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    table = str(result.get("summary_table", "")).strip()
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(table + ("\n" if table else ""))
    log.info("[%s] wrote lm-eval artifacts: %s | %s", eval_tag, json_path, summary_path)
    return {"json_path": json_path, "summary_path": summary_path}


def _run_local_lm_eval(
    *,
    model: nn.Module,
    tokenizer,
    args,
    base_model_path: str,
    task_names: Sequence[str],
    eval_tag: str,
    log,
    eval_log_dir: Optional[str],
) -> Dict[str, Any]:
    lm_args = argparse.Namespace(
        tasks=",".join(task_names),
        num_fewshot=int(args.eval_num_fewshot),
        batch_size=str(args.eval_lm_batch_size),
        lm_limit=args.eval_lm_limit,
        model_path=str(base_model_path),
        eval_log_dir=eval_log_dir,
        eval_run_ts=str(eval_tag) if eval_log_dir else None,
    )
    log.info(
        "[%s] local lm_eval tasks=%s fewshot=%d batch_size=%s limit=%s",
        eval_tag,
        ",".join(task_names),
        int(args.eval_num_fewshot),
        str(args.eval_lm_batch_size),
        str(args.eval_lm_limit),
    )
    with applied_hif4_act(
        model,
        enabled=bool(args.eval_hif4_act),
        logger=log,
        log_prefix=f"[lm_eval:{eval_tag}] ",
    ):
        with torch.no_grad():
            return run_lm_eval(model, tokenizer, lm_args)


def run_e2e_lm_eval(
    *,
    model: nn.Module,
    tokenizer,
    args,
    base_model_path: str,
    output_dir: str,
    log,
    eval_tag: str,
    move_to_device: bool,
    parallel_stage_decode: bool,
    cache_decoded_weight: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run lm-eval; under WORLD_SIZE>1, shard tasks across ranks and merge on rank0."""
    task_names = parse_eval_task_names(getattr(args, "eval_tasks", None))
    if not task_names:
        return None

    tag = str(eval_tag).strip() or "eval"
    distributed = is_distill_distributed()
    with temporary_inference_decode_mode(
        model,
        parallel_stage_decode=bool(parallel_stage_decode),
        cache_decoded_weight=bool(cache_decoded_weight),
    ):
        if distributed:
            distill_distributed_barrier()
            rank = distill_rank()
            world_size = distill_world_size()
            local_device = get_distill_local_device(fallback=str(args.eval_device))
            local_tasks = split_tasks_for_distill_rank(task_names, rank=rank, world_size=world_size)
            log.info(
                "[rank=%d] [%s] distributed lm_eval task assignment: %s",
                int(rank),
                tag,
                ",".join(local_tasks) if local_tasks else "(none)",
            )
            if move_to_device:
                model.to(local_device)
            if local_tasks:
                partial_result = _run_local_lm_eval(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    base_model_path=base_model_path,
                    task_names=local_tasks,
                    eval_tag=f"{tag}_rank{rank}",
                    log=log,
                    eval_log_dir=None,
                )
            else:
                partial_result = {"task_metrics": {}, "task_metric_keys": {}}

            gathered: Optional[List[Optional[dict]]] = [None] * world_size if rank == 0 else None
            dist.gather_object(partial_result, gathered, dst=0)

            result_payload = None
            if is_distill_main_process():
                merged = merge_lm_eval_results(gathered or [], task_names)
                table = str(merged.get("summary_table", "")).strip()
                if table:
                    log.info("[%s] LM-Eval summary table:\n%s", tag, table)
                _log_merged_task_metrics(tag=tag, task_names=task_names, lm_result=merged, log=log)
                paths = _write_lm_eval_artifacts(
                    result=merged,
                    output_dir=output_dir,
                    eval_tag=tag,
                    log=log,
                )
                result_payload = {"result": merged, **paths}
            distill_distributed_barrier()
            return result_payload

        eval_device = str(args.eval_device).strip()
        if move_to_device:
            log.info("[%s] Moving model to %s for lm_eval ...", tag, eval_device)
            model.to(eval_device)
        eval_log_dir = os.path.join(output_dir, "lm_eval")
        os.makedirs(eval_log_dir, exist_ok=True)
        result = _run_local_lm_eval(
            model=model,
            tokenizer=tokenizer,
            args=args,
            base_model_path=base_model_path,
            task_names=task_names,
            eval_tag=tag,
            log=log,
            eval_log_dir=eval_log_dir,
        )
        table = str(result.get("summary_table", "")).strip()
        if table:
            log.info("[%s] LM-Eval summary table:\n%s", tag, table)
        _log_merged_task_metrics(tag=tag, task_names=task_names, lm_result=result, log=log)
        return {
            "result": result,
            "json_path": os.path.join(eval_log_dir, f"lm_eval_results_{tag}.json"),
            "summary_path": os.path.join(eval_log_dir, f"lm_eval_summary_{tag}.md"),
        }


class EvalBeforeSaveCallback(TrainerCallback):
    """Run lm-eval on save_steps before HF Trainer writes checkpoint-*."""

    def __init__(
        self,
        *,
        e2e_args,
        tokenizer,
        base_model_path: str,
        run_output_dir: str,
        log,
        parallel_stage_decode: bool,
        parallel_mode: str,
    ):
        self.e2e_args = e2e_args
        self.tokenizer = tokenizer
        self.base_model_path = str(base_model_path)
        self.run_output_dir = str(run_output_dir)
        self.log = log
        self.parallel_stage_decode = bool(parallel_stage_decode)
        self.parallel_mode = str(parallel_mode).strip().lower()
        self._last_eval_step: Optional[int] = None
        self._trainer = None

    def bind_trainer(self, trainer) -> None:
        self._trainer = trainer

    def _resolve_eval_model(self, model: nn.Module) -> nn.Module:
        trainer = self._trainer
        if trainer is not None and getattr(trainer, "accelerator", None) is not None:
            return trainer.accelerator.unwrap_model(trainer.model)
        return model

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not bool(getattr(self.e2e_args, "eval_before_save", False)):
            return control
        save_steps = int(getattr(args, "save_steps", 0) or 0)
        if save_steps < 1:
            return control
        global_step = int(getattr(state, "global_step", 0) or 0)
        if global_step < 1 or global_step % save_steps != 0:
            return control
        max_steps = int(getattr(args, "max_steps", 0) or 0)
        if max_steps > 0 and global_step >= max_steps:
            # Final packed-model eval runs after trainer.train(); skip duplicate mid eval.
            return control
        if self._last_eval_step == global_step:
            return control
        if model is None:
            raise RuntimeError("EvalBeforeSaveCallback requires model in on_step_end.")

        self._last_eval_step = global_step
        eval_tag = f"step_{global_step}"
        self.log.info(
            "Eval-before-save at global_step=%d (save_steps=%d, parallel_mode=%s)",
            global_step,
            save_steps,
            self.parallel_mode,
        )

        eval_model = self._resolve_eval_model(model)
        move_to_device = self.parallel_mode == "dp" or is_distill_distributed()
        trainer = self._trainer
        previous_teacher_device = None
        optimizer_device = None
        if trainer is not None:
            if hasattr(trainer, "offload_teacher_to_cpu"):
                previous_teacher_device = trainer.offload_teacher_to_cpu()
                if previous_teacher_device is not None and previous_teacher_device.type != "cpu":
                    self.log.info(
                        "Offloaded teacher to CPU for eval-before-save (was %s).",
                        previous_teacher_device,
                    )
            optimizer = getattr(trainer, "optimizer", None)
            if optimizer is not None:
                if previous_teacher_device is not None and previous_teacher_device.type != "cpu":
                    optimizer_device = previous_teacher_device
                else:
                    optimizer_device = torch.device(
                        get_distill_local_device(fallback=str(self.e2e_args.eval_device))
                    )
                moved = _offload_optimizer_state_to_cpu(optimizer)
                if moved > 0:
                    self.log.info("Offloaded %d optimizer state tensors to CPU for eval-before-save.", moved)
            if hasattr(eval_model, "zero_grad"):
                eval_model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            run_e2e_lm_eval(
                model=eval_model,
                tokenizer=self.tokenizer,
                args=self.e2e_args,
                base_model_path=self.base_model_path,
                output_dir=self.run_output_dir,
                log=self.log,
                eval_tag=eval_tag,
                move_to_device=bool(move_to_device),
                parallel_stage_decode=self.parallel_stage_decode,
                cache_decoded_weight=False,
            )
        finally:
            if trainer is not None:
                optimizer = getattr(trainer, "optimizer", None)
                if optimizer is not None and optimizer_device is not None:
                    moved = _restore_optimizer_state_to_device(optimizer, torch.device(optimizer_device))
                    if moved > 0:
                        self.log.info(
                            "Restored %d optimizer state tensors to %s after eval-before-save.",
                            moved,
                            optimizer_device,
                        )
                if hasattr(trainer, "restore_teacher_device"):
                    trainer.restore_teacher_device(previous_teacher_device)
                    if previous_teacher_device is not None and previous_teacher_device.type != "cpu":
                        self.log.info("Restored teacher to %s after eval-before-save.", previous_teacher_device)
        return control
