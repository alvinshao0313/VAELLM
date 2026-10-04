"""Pause linear-head training for isolated export and fixed full eight-task evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
import time

from transformers import TrainerCallback

from experiments.qwen3_w25.e2e_aux_export import inspect_step
from experiments.qwen3_w25.step_export_callback import SEVEN_TASKS, export_process_environment, run_logged_stage


LOG = logging.getLogger(__name__)


def stage_commands(*, checkpoint, source_dir, output_dir, diagnostic_eval_limit=None):
    """Use saved, fused BF16 weights for both shards; never evaluate a training wrapper."""
    output = Path(output_dir)
    final = output / "final_model"
    commands = [
        ("export", [sys.executable, "-m", "experiments.qwen3_w25.e2e_aux_export",
                    "--checkpoint_dir", str(checkpoint), "--output_dir", str(final)], (0,)),
        ("audit", [sys.executable, "-m", "experiments.qwen3_w25.e2e_aux_audit", "--source_dir", str(source_dir),
                   "--checkpoint_dir", str(final), "--output", str(output / "audit.json")], (0,)),
    ]
    for name, tasks, batch in (("mmlu", "mmlu", "64"), ("seven", SEVEN_TASKS, "32")):
        command = [sys.executable, "tools/cat_eval.py", "--checkpoint_dir", str(final),
                   "--strict", "--eval_lm_eval", "--eval_device", "cuda:0", "--tasks", tasks,
                   "--num_fewshot", "0", "--lm_batch_size", batch, "--tokenizer_use_fast", "true",
                   "--prewarm_group_size", "8", "--eval_log_dir", str(output / f"{name}_logs"),
                   "--output_json", str(output / f"{name}.json")]
        if diagnostic_eval_limit is not None:
            command.extend(["--lm_limit", str(diagnostic_eval_limit)])
        commands.append((name, command, (0,)))
    if diagnostic_eval_limit is None:
        commands.append(("summary", [sys.executable, "-m", "experiments.qwen3_w25.summarize_results",
            "--checkpoint_dir", str(final), "--results", str(output / "mmlu.json"), str(output / "seven.json"),
            "--output", str(output / "acceptance.json")], (0, 1)))
    return commands


class AuxStepExportCallback(TrainerCallback):
    def __init__(self, *, source_dir, run_root_dir, export_cuda_visible_devices, diagnostic_eval_limit=None):
        if diagnostic_eval_limit is not None and (type(diagnostic_eval_limit) is not int or diagnostic_eval_limit < 1):
            raise ValueError("diagnostic_eval_limit must be a positive integer or None.")
        self.source_dir = Path(source_dir).resolve()
        self.run_root_dir = Path(run_root_dir).resolve()
        self.env = export_process_environment(export_cuda_visible_devices)
        self.diagnostic_eval_limit = diagnostic_eval_limit
        self._last_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        strategy = getattr(args.save_strategy, "value", args.save_strategy)
        if strategy != "steps" or args.save_steps < 1 or int(args.save_steps) != args.save_steps:
            raise ValueError("Aux export requires positive integral save_steps and save_strategy=steps.")
        if args.max_steps < 1 or args.max_steps % int(args.save_steps) or args.save_total_limit != 1:
            raise ValueError("Use a final save boundary and save_total_limit=1.")
        return control

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == self._last_step:
            return control
        checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
        meta, saved_state, _snapshot, source, _base = inspect_step(checkpoint)
        if saved_state["global_step"] != step or source.resolve() != self.source_dir:
            raise ValueError("Saved step/source does not match the callback boundary.")
        output = self.run_root_dir / f"step_{step:04d}"
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "source_checkpoint": str(checkpoint), "source_checkpoint_id": meta["checkpoint_id"],
                  "global_step": step, "max_steps": int(args.max_steps),
                  "export_cuda_visible_devices": self.env["CUDA_VISIBLE_DEVICES"],
                  "diagnostic_eval_limit": self.diagnostic_eval_limit, "stages": {}}
        record_path = output / ("diagnostic.json" if self.diagnostic_eval_limit is not None else "step_export.json")
        started = time.monotonic()
        LOG.info("Pausing linear-head training at step %d for saved-step export/evaluation on GPU %s.",
                 step, self.env["CUDA_VISIBLE_DEVICES"])
        try:
            query = subprocess.run(["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
            if query.stdout.strip():
                raise RuntimeError(f"Export GPU is occupied: {query.stdout.strip()}.")
            record["export_gpu_idle_checked"] = True
            for name, command, allowed in stage_commands(checkpoint=checkpoint, source_dir=self.source_dir,
                    output_dir=output, diagnostic_eval_limit=self.diagnostic_eval_limit):
                record["stages"][name] = {"command": command}
                record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                record["stages"][name]["exit_code"] = run_logged_stage(
                    command, name=name, output_dir=output, env=self.env, allowed_exit_codes=allowed)
            record["status"] = "diagnostic_complete" if self.diagnostic_eval_limit is not None else "evaluated"
            self._last_step = step
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        LOG.info("Linear-head step %d export/evaluation finished in %.1fs; resuming training.",
                 step, record["elapsed_seconds"])
        return control
