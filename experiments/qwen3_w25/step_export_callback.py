"""Evaluate saved E2E steps in isolated processes while training is paused."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from transformers import TrainerCallback

from train_utils.checkpoint_v6 import load_v6_training_step_meta


REPO = Path(__file__).resolve().parents[2]
SEVEN_TASKS = "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa"
LOG = logging.getLogger(__name__)


def export_process_environment(cuda_visible_devices: str) -> dict[str, str]:
    """Give subprocesses their own CUDA namespace, outside the training GPUs."""
    requested = [part.strip() for part in cuda_visible_devices.split(",")]
    training = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if len(requested) != 1 or not requested[0] or requested[0] == "-1":
        raise ValueError("Export requires exactly one explicit CUDA device.")
    if not all(training) or set(requested) & set(training):
        raise ValueError("Set explicit, disjoint training and export CUDA_VISIBLE_DEVICES.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Step export supports single-process layer placement, not distributed ranks.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = requested[0]
    for key in tuple(env):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                   "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} or key.startswith("TORCHELASTIC_"):
            del env[key]
    return env


def run_logged_stage(command, *, name, output_dir, env, allowed_exit_codes=(0,)) -> int:
    """Execute a real child command, retaining its output and exit status."""
    output_dir = Path(output_dir)
    log_path = output_dir / f"{name}.log"
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n{shlex.join(command)}\n")
        log.flush()
        try:
            completed = subprocess.run(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        except OSError as exc:
            log.write(f"Subprocess launch failed: {type(exc).__name__}: {exc}\n")
            (output_dir / f"{name}.exit").write_text("launch_error\n", encoding="utf-8")
            raise
    (output_dir / f"{name}.exit").write_text(f"{completed.returncode}\n", encoding="utf-8")
    if completed.returncode not in allowed_exit_codes:
        raise RuntimeError(f"Step export stage {name} exited {completed.returncode}; see {log_path}.")
    return completed.returncode


class StepExportCallback(TrainerCallback):
    """Block each save boundary until export, audit and full evaluation finish.

    ``diagnostic_eval_limit`` is only for a directly constructed validation callback;
    the training CLI always performs unlimited evaluation and goal validation.
    """

    def __init__(self, *, source_dir, run_root_dir, train_mode, export_cuda_visible_devices,
                 diagnostic_eval_limit: int | None = None):
        if train_mode not in ("decoder", "decoder_sparse_bit"):
            raise ValueError("Step export requires decoder or decoder_sparse_bit training.")
        if diagnostic_eval_limit is not None and (type(diagnostic_eval_limit) is not int or diagnostic_eval_limit < 1):
            raise ValueError("diagnostic_eval_limit must be a positive integer or None.")
        self.source_dir = Path(source_dir).resolve()
        self.run_root_dir = Path(run_root_dir).resolve()
        self.train_mode = train_mode
        self.env = export_process_environment(export_cuda_visible_devices)
        self.diagnostic_eval_limit = diagnostic_eval_limit
        self._last_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        strategy = getattr(args.save_strategy, "value", args.save_strategy)
        save_steps = args.save_steps
        if strategy != "steps" or save_steps < 1 or int(save_steps) != save_steps:
            raise ValueError("Step export requires a positive integral save_steps and save_strategy=steps.")
        if args.max_steps < 1 or args.max_steps % int(save_steps):
            raise ValueError("The final training step must be a save boundary for strict evaluation.")
        if args.save_total_limit != 1:
            raise ValueError("This synchronous export recipe requires save_total_limit=1.")
        return control

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == self._last_step:
            return control
        checkpoint = Path(args.output_dir).resolve() / f"checkpoint-{step}"
        meta = load_v6_training_step_meta(str(checkpoint))
        saved_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        if saved_state["global_step"] != step or meta["train_mode"] != self.train_mode:
            raise ValueError("Saved checkpoint step/mode does not match the callback boundary.")
        if self.train_mode == "decoder_sparse_bit" and not (checkpoint / "sparse_bit_tuning/exact_state.pt").is_file():
            raise FileNotFoundError(f"Saved SparseBit sidecar is missing: {checkpoint}.")
        output = self.run_root_dir / f"step_{step:04d}"
        output.mkdir(parents=True, exist_ok=False)
        final_model = output / "final_model"
        started = time.monotonic()
        LOG.info("Pausing training at step %d for isolated export and evaluation on GPU %s; output=%s",
                 step, self.env["CUDA_VISIBLE_DEVICES"], output)
        commands = [
            ("export", [sys.executable, "-m", "experiments.qwen3_w25.export_e2e_selected",
                        "--checkpoint_dir", str(checkpoint), "--output_dir", str(final_model)], (0,)),
            ("audit", [sys.executable, "-m", "experiments.qwen3_w25.audit_e2e_decoder",
                       "--source_dir", str(self.source_dir), "--checkpoint_dir", str(final_model),
                       "--train_mode", self.train_mode, "--output", str(output / "audit.json")], (0,)),
        ]
        for name, tasks in (("mmlu", "mmlu"), ("seven", SEVEN_TASKS)):
            command = [sys.executable, "tools/cat_eval.py", "--checkpoint_dir", str(final_model),
                       "--strict", "--eval_lm_eval", "--eval_device", "cuda:0", "--tasks", tasks,
                       "--num_fewshot", "0", "--lm_batch_size", "auto", "--tokenizer_use_fast", "true",
                       "--prewarm_group_size", "8", "--eval_log_dir", str(output / f"{name}_logs"),
                       "--output_json", str(output / f"{name}.json")]
            if self.diagnostic_eval_limit is not None:
                command.extend(["--lm_limit", str(self.diagnostic_eval_limit)])
            commands.append((name, command, (0,)))
        if self.diagnostic_eval_limit is None:
            commands.append(("summary", [sys.executable, "-m", "experiments.qwen3_w25.summarize_results",
                                        "--checkpoint_dir", str(final_model), "--results",
                                        str(output / "mmlu.json"), str(output / "seven.json"),
                                        "--output", str(output / "acceptance.json")], (0, 1)))
        record = {"status": "running", "source_checkpoint": str(checkpoint), "global_step": step,
                  "max_steps": int(args.max_steps), "export_cuda_visible_devices": self.env["CUDA_VISIBLE_DEVICES"],
                  "diagnostic_eval_limit": self.diagnostic_eval_limit, "stages": {}}
        record_path = output / ("diagnostic.json" if self.diagnostic_eval_limit is not None else "step_export.json")
        try:
            gpu_query = subprocess.run(
                ["nvidia-smi", "--id=" + self.env["CUDA_VISIBLE_DEVICES"],
                 "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
            if gpu_query.stdout.strip():
                raise RuntimeError(f"Export GPU is occupied by compute processes: {gpu_query.stdout.strip()}.")
            record["export_gpu_idle_checked"] = True
            for name, command, allowed in commands:
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
        LOG.info("Step %d export/evaluation finished in %.1fs; resuming training.", step, record["elapsed_seconds"])
        return control
